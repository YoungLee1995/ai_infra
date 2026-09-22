"""Train a small binary classifier with PyTorch DDP.

Run with one process or through torchrun, for example:
    torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train --device cuda --amp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# 作用：判断当前进程是否已加入分布式进程组。
# 调用者：rank()、world_size()、train() 及训练流程中的分布式分支。
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


# 作用：返回当前进程的全局 rank；单进程运行时返回 0。
# 调用者：train() 以及 rank 0 的日志、指标和 checkpoint 分支。
def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


# 作用：返回参与训练的进程总数；单进程运行时返回 1。
# 调用者：train() 记录指标和判断有效的全局训练规模。
def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


# 作用：固定 Python、CPU 和可用 CUDA RNG，减少实验结果的随机差异。
# 调用者：train() 在进程组初始化后调用，并为不同 rank 设置进程本地 seed。
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 作用：收集当前进程的 Python、CPU、CUDA 和 DataLoader RNG 状态。
# 调用者：train() 在 checkpoint 保存前调用；恢复时由 restore_rng_state() 使用。
def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "loader": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


# 作用：将 checkpoint 中对应 rank 的 RNG 状态恢复到当前进程。
# 调用者：load_checkpoint() 在模型和优化器状态加载后调用。
def restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# 作用：让所有 rank 将各自 RNG 状态汇总到 rank 0，保证多卡恢复时每个进程可还原。
# 调用者：train() 每次准备保存 step checkpoint 时调用。
def gather_rng_states(loader_generator: torch.Generator) -> list[dict[str, Any]] | None:
    local_state = capture_rng_state(loader_generator)
    if not is_distributed():
        return [local_state]
    gathered: list[dict[str, Any] | None] | None = [None] * world_size() if rank() == 0 else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered  # type: ignore[return-value]


# 生成和处理数据
class SyntheticBinaryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic two-class data, generated once from a local RNG."""

    # 作用：生成固定的合成二分类特征和标签。
    # 调用者：train() 创建训练数据集；测试用例直接验证其可复现性。
    def __init__(self, samples: int = 4096, input_dim: int = 32, seed: int = 7) -> None:
        if samples < 1 or input_dim < 2:
            raise ValueError("samples must be positive and input_dim must be at least 2")
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(samples, input_dim, generator=generator)
        boundary = self.features[:, 0] + 0.7 * self.features[:, 1]
        self.labels = (boundary > 0).long()

    # 作用：返回数据集样本数量，供 DataLoader 和 DistributedSampler 使用。
    # 调用者：PyTorch DataLoader/DistributedSampler 的内部逻辑。
    def __len__(self) -> int:
        return self.features.shape[0]

    # 作用：按索引返回一个特征张量和对应标签。
    # 调用者：DataLoader 在迭代每个 batch 时调用。
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


# 网络结构
class MLP(nn.Module):
    # 作用：构造用于二分类的多层感知机网络。
    # 调用者：train() 创建基础模型；测试用例创建模型验证 checkpoint。
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    # 作用：执行一次前向传播，输出两个类别的 logits。
    # 调用者：train() 的训练循环通过 model(features) 隐式调用。
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


# 作用：选择 CPU/CUDA 设备，设置 CUDA 当前设备，并在多进程时初始化通信组。
# 调用者：train() 在创建数据和模型前调用。
def configure_process_group(device_name: str) -> torch.device:
    requested = device_name.lower()
    # [设备映射] 每个 torchrun 进程通过 LOCAL_RANK 绑定本机的一张 GPU。
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds visible CUDA devices")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    elif requested == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError("--device must be either cpu or cuda")

    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if requested_world_size > 1:
        if not dist.is_available():
            raise RuntimeError("This PyTorch build does not include torch.distributed")
        # [进程组初始化] env:// 读取 torchrun 注入的 rank/world size 等信息。
        dist.init_process_group(backend=backend, init_method="env://")
    return device


# 作用：把命令行参数整理成可序列化的实验配置快照。
# 调用者：train() 创建 checkpoint 和 config.json 时调用。
def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "epochs": args.epochs,
        "samples": args.samples,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "amp": args.amp,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "warmup_steps": args.warmup_steps,
        "world_size": world_size(),
    }


# 作用：创建带 warmup 和 cosine decay 的 step 级学习率调度器。
# 调用者：train() 在创建 optimizer 后调用；每个 optimizer step 后推进一次。
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 作用：阻止用不同训练语义的配置静默恢复，避免 step/LR/数据轨迹发生变化。
# 调用者：load_checkpoint() 在加载状态前调用。
def validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    required = (
        "device",
        "samples",
        "input_dim",
        "hidden_dim",
        "batch_size",
        "accumulation_steps",
        "learning_rate",
        "seed",
        "amp",
        "warmup_steps",
        "world_size",
    )
    mismatches = [key for key in required if saved.get(key) != current.get(key)]
    if mismatches:
        details = ", ".join(f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}" for key in mismatches)
        raise ValueError(f"resume config mismatch ({details}); use the original training configuration")


# 作用：原子保存模型、优化器、AMP scaler、step 位置和配置。
# 调用者：train() 在 rank 0 的指定 optimizer step 或 epoch 边界调用。
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    samples_seen: int,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    rng_states: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    # [Checkpoint 保存] 记录 step 级位置，恢复时可以跳过当前 epoch 已完成的 batch。
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "scheduler": scheduler.state_dict(),
        "rng_states": rng_states,
        "completed_epoch": epoch if batch_in_epoch == 0 else epoch,
        "config": config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


# 作用：从 checkpoint 恢复模型、优化器、AMP scaler 和 step 位置。
# 调用者：train() 启动时收到 --resume 参数后调用；测试用例验证恢复结果。
def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    loader_generator: torch.Generator,
    current_config: dict[str, Any],
) -> dict[str, int]:
    # [Checkpoint 恢复] 将训练状态加载到当前 rank 的设备，并返回 epoch/batch/step。
    state = torch.load(path, map_location=device, weights_only=False)
    validate_resume_config(state["config"], current_config)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state.get("scaler", {}))
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng_states"][rank()], loader_generator)
    if "epoch" not in state:
        return {"epoch": int(state["completed_epoch"]), "batch_in_epoch": 0, "global_step": 0}
    return {
        "epoch": int(state["epoch"]),
        "batch_in_epoch": int(state["batch_in_epoch"]),
        "global_step": int(state["global_step"]),
        "samples_seen": int(state["samples_seen"]),
    }


# 作用：创建 CUDA float16 autocast 上下文；禁用时保持普通精度执行。
# 调用者：train() 在每个 micro-batch 的前向和 loss 计算时调用。
def autocast_context(enabled: bool):
    # [AMP] CUDA 开启 float16 autocast；CPU 或未传 --amp 时保持关闭。
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


# 作用：执行完整训练流程，包括数据切分、DDP、梯度累积、指标和 checkpoint。
# 调用者：文件底部的 __main__ 入口；用户通过 python 或 torchrun 间接调用。
def train(args: argparse.Namespace) -> None:
    """Run one complete training job from process setup to checkpointed epochs.

    The control flow is intentionally explicit: every torchrun process executes
    this function, computes on its own data shard, synchronizes gradients through
    DDP, and lets rank 0 write shared output files.
    """
    # 1. 初始化设备和进程组。
    #    在 torchrun 下，每个进程先根据 LOCAL_RANK 绑定一张 GPU；随后
    #    configure_process_group() 用 RANK/WORLD_SIZE 建立 NCCL 通信组。
    device = configure_process_group(args.device)
    # 给不同 rank 使用不同的进程级随机种子；数据集本身仍使用 args.seed，
    # 这样各 rank 生成的训练数据内容一致，sampler 再负责切分索引。
    seed_everything(args.seed + rank())
    # config 会写入 checkpoint/config.json，用于记录本次实验的关键语义。
    config = build_config(args)
    output_dir = Path(args.checkpoint_dir)

    # 2. 构造数据管道。
    #    dataset 是所有 rank 都能看到的完整数据集；DistributedSampler 会
    #    根据 rank/world_size 产生当前进程专属的索引，DataLoader 再组成 batch。
    dataset = SyntheticBinaryDataset(args.samples, args.input_dim, args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    # [Sampler] 每个 rank 只消费自己的数据分片，drop_last 避免补齐导致尾部重复。
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if is_distributed() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    if len(loader) == 0:
        raise ValueError("dataset must contain at least one complete batch per rank")

    # 3. 构造模型和训练状态。
    #    先把模型移动到当前 rank 的设备，再用 DDP 包装。之后 model(features)
    #    会执行本地前向，backward() 时 DDP 自动同步各 rank 的梯度。
    model: nn.Module = MLP(args.input_dim, args.hidden_dim).to(device)
    if is_distributed():
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 每次 optimizer.step() 才算一个 global step；梯度累积的多个 micro-batch
    # 共享一个 optimizer step，因此 scheduler 也按这里计算出的步数推进。
    steps_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    scheduler = build_scheduler(optimizer, steps_per_epoch * args.epochs, args.warmup_steps)
    # scaler 在 FP16 CUDA 训练时防止梯度下溢；CPU 或未开启 AMP 时等价于关闭。
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # 这些变量表示“从哪里继续训练”。正常训练从 epoch/batch/step 0 开始；
    # resume 时由 checkpoint 覆盖。resume_batch 是已完成 batch 数，恢复后会跳过它们。
    start_epoch = 0
    resume_batch = 0
    global_step = 0
    samples_seen = 0
    if args.resume:
        # 所有 rank 都加载同一个 checkpoint，并恢复各自的 optimizer/scheduler/RNG。
        # 配置不一致会在这里报错，避免错误地把不同实验拼接起来。
        checkpoint_position = load_checkpoint(
            Path(args.resume), model, optimizer, scaler, scheduler, device, loader_generator, config
        )
        start_epoch = checkpoint_position["epoch"]
        resume_batch = checkpoint_position["batch_in_epoch"]
        global_step = checkpoint_position["global_step"]
        samples_seen = checkpoint_position["samples_seen"]
        if rank() == 0:
            print(
                f"resumed from epoch {start_epoch}, batch {resume_batch}, step {global_step}: {args.resume}",
                flush=True,
            )

    metrics_path = output_dir / "metrics.jsonl"
    # 共享输出只由 rank 0 创建和写入，避免多个进程竞争同一个文件。
    # 共享输出：config/metrics/日志只由 rank 0 写，避免竞争。
    # 大模型：日志仍 rank 0 写；checkpoint 分片并行写，rank 0 只提交元数据。
    # 多节点：需共享存储；barrier 保证 rank 0 写完后继续。
    if rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    criterion = nn.CrossEntropyLoss()
    stop_requested = False
    # 4. 外层 epoch 循环。
    #    resume 的第一个 epoch 可能只执行剩余 batch；后续 epoch 执行完整数据集。
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            # 必须让 sampler 感知 epoch，否则每个 epoch 的 shuffle 顺序会重复。
            sampler.set_epoch(epoch)
        # train() 开启 dropout/batchnorm 等训练行为；本 MLP 当前没有这两类层。
        model.train()
        # 梯度只在 optimizer step 后清空；因此一个 accumulation window 内会累积。
        # [梯度累积几何] 多个 micro-batch 的梯度先向量相加再平均，沿合向量方向更新一次。
        # 样本梯度可同向/反向/任意夹角；方向是平均方向，lr 不变，但实际步长 = lr*||g_avg||，会随夹角和模长变化。
        # 除以 accumulation_steps 是为了平均，避免梯度放大 N 倍。
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss_sum = torch.zeros(1, device=device)
        sample_count = torch.zeros(1, device=device)
        batch_count = len(loader)
        window_samples = 0

        # ============================================================
        # DDP 训练核心概念速览
        # ============================================================
        # 1. DDP：一进程一设备，每个 rank 持完整模型副本，读不同数据分片。
        # 2. 前向/反向各 rank 独立算；反向时 all-reduce 同步梯度，得到平均梯度。
        # 3. 各 rank 用相同平均梯度各自 optimizer.step()，参数保持一致。
        # 4. 数据分片由 DistributedSampler 完成；每个 micro-batch 是不同数据。
        # 5. 梯度累积：单 rank 内多个 micro-batch 串行前向/反向，梯度累加后平均。
        # 6. all-reduce 与梯度累积数学上都是对梯度做加权平均，只是维度不同：
        #    - 梯度累积：同一 rank 内，时间/批次维度
        #    - all-reduce：跨 rank，设备维度
        #    全局梯度 = (1/(A*W)) * Σ_r Σ_a g_{r,a}
        # 7. 单 rank 多 micro-batch 必须串行：单卡只有一份计算/显存资源；
        #    并行会成倍占显存，且梯度累加存在写竞争。
        # 8. 多 rank 能并行：多张卡独立计算/显存，空间换时间。
        # ============================================================

        # 5. 内层 micro-batch 循环。
        #    一个 DataLoader batch 是一个 micro-batch；accumulation_steps 个
        #    micro-batch 才组成一次 optimizer update。
        for batch_index, (features, labels) in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch:
                # checkpoint 已经记录这些 batch 被消费并完成了 optimizer step，
                # 这里只重建 DataLoader 顺序，不再次执行前向/反向。
                continue
            # 将当前 rank 的 batch 搬到自己的设备；CUDA pin_memory 配合
            # non_blocking=True 可以减少主机到设备的拷贝等待。
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # 最后一个窗口可能不足 accumulation_steps，因此按实际窗口大小
            # 缩放 loss，保证累计梯度量级与等效大 batch 一致。
            window_size = min(
                args.accumulation_steps,
                batch_count - (batch_index // args.accumulation_steps) * args.accumulation_steps,
            )
            # 当前 batch 是否是本次梯度累积窗口的最后一个 batch。
            is_update = (batch_index + 1) % args.accumulation_steps == 0 or batch_index == batch_count - 1
            # [no_sync] 累积窗口的中间 batch 不做梯度 all-reduce，最后一个 batch 再同步。
            sync_context = nullcontext() if is_update or not is_distributed() else model.no_sync()
            # 6. 前向、loss 和反向。
            #    autocast 只影响 CUDA AMP；loss.backward() 由 autograd 计算梯度，
            #    DDP 在需要同步的反向阶段自动 all-reduce 各 rank 的梯度。
            with sync_context, autocast_context(args.amp and device.type == "cuda"):
                loss = criterion(model(features), labels)
                scaled_loss = loss / window_size
            scaler.scale(scaled_loss).backward()
            # 这些计数用于 epoch 指标；loss 按样本数加权，而不是简单平均 batch loss。
            loss_sum += loss.detach() * labels.size(0)
            sample_count += labels.size(0)
            window_samples += labels.size(0)
            # 梯度更新只和 backward() 算梯度、optimizer.step() 用梯度更新参数有关；epoch 尾部的指标汇总、保存 checkpoint、barrier 都不参与梯度计算，对梯度本身没有影响。可能影响后续训练的是 scheduler、RNG、数据顺序这类状态。
            if is_update:
                # 7. optimizer step 边界。
                #    只有这里真正更新参数、推进 scheduler/global_step，并允许
                #    保存 step checkpoint；这保证 checkpoint 不落在累积窗口中间。
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                # 每个 rank 处理相同数量样本，因此用 world_size 换算全局样本数。
                samples_seen += window_samples * world_size()
                window_samples = 0
                stop_after_this_step = (
                    args.stop_after_steps is not None and global_step == args.stop_after_steps
                )
                should_checkpoint = global_step % args.checkpoint_every_steps == 0 or stop_after_this_step
                if should_checkpoint:
                    # 先汇总所有 rank 的 RNG；rank 0 随后写入统一 checkpoint。
                    rng_states = gather_rng_states(loader_generator)
                if should_checkpoint and rank() == 0:
                    # epoch 最后一批已完成时，恢复位置应指向下一 epoch，不能再跳过
                    # 一个已经没有剩余 batch 的 epoch。
                    checkpoint_epoch = epoch + 1 if batch_index == batch_count - 1 else epoch
                    checkpoint_batch = 0 if batch_index == batch_count - 1 else batch_index + 1
                    # 编号快照用于回退，checkpoint.pt 作为最新状态入口。
                    save_checkpoint(
                        output_dir / f"checkpoint-step-{global_step:08d}.pt",
                        model,
                        optimizer,
                        scaler,
                        checkpoint_epoch,
                        checkpoint_batch,
                        global_step,
                        samples_seen,
                        scheduler,
                        rng_states,
                        config,
                    )
                    save_checkpoint(
                        output_dir / "checkpoint.pt",
                        model,
                        optimizer,
                        scaler,
                        checkpoint_epoch,
                        checkpoint_batch,
                        global_step,
                        samples_seen,
                        scheduler,
                        rng_states,
                        config,
                    )
                if should_checkpoint and is_distributed():
                    # rank 0 写盘期间其他 rank 等待，确保所有进程从同一个 step 继续。
                    dist.barrier()
                if stop_after_this_step:
                    if batch_index != batch_count - 1:
                        raise ValueError("--stop-after-steps must fall on an epoch boundary")
                    stop_requested = True
                    if rank() == 0:
                        print(f"stopped after checkpoint at step {global_step}", flush=True)
                    break

        # 8. epoch 汇总。
        #    loss_sum/sample_count 先 all-reduce，rank 0 才能得到全局平均 loss。
        #    全局平均 loss 是训练过程的“仪表盘”，用来判断训练是否正常、收敛如何、不同实验怎么比；它不参与梯度计算，也不直接影响参数更新。
        elapsed = time.perf_counter() - started
        if is_distributed():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        # 所有 rank 都参与 RNG 汇总，只有 rank 0 负责指标和 epoch 边界 checkpoint。
        rng_states = gather_rng_states(loader_generator)
        if rank() == 0:
            samples = int(sample_count.item())
            record = {
                "epoch": epoch + 1,
                "loss": loss_sum.item() / samples,
                "samples_per_second": samples / elapsed,
                "world_size": world_size(),
            }
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(record) + "\n")
            save_checkpoint(
                output_dir / "checkpoint.pt",
                model,
                optimizer,
                scaler,
                epoch + 1,
                0,
                global_step,
                samples_seen,
                scheduler,
                rng_states,
                config,
            )
            print(json.dumps(record), flush=True)
        if is_distributed():
            # 确保 rank 0 保存完成后，所有 rank 再进入下一个 epoch。
            dist.barrier()
        if stop_requested:
            break
    # 9. 训练结束，释放进程组；单进程模式下不执行 destroy。
    if is_distributed():
        dist.destroy_process_group()


# 作用：解析并校验训练命令行参数。
# 调用者：文件底部的 __main__ 入口，将结果传给 train()。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="enable CUDA float16 autocast")
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=100,
        help="save a numbered checkpoint and update checkpoint.pt every optimizer step interval",
    )
    parser.add_argument("--checkpoint-dir", default="runs/default")
    parser.add_argument("--resume", help="path to a checkpoint created by this trainer")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="test-only controlled stop at an epoch-boundary checkpoint",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.accumulation_steps < 1
        or args.checkpoint_every_steps < 1
        or args.warmup_steps < 0
        or (args.stop_after_steps is not None and args.stop_after_steps < 1)
    ):
        parser.error(
            "epochs, batch-size, accumulation-steps, checkpoint-every-steps, and stop-after-steps must be positive; warmup-steps cannot be negative"
        )
    return args


if __name__ == "__main__":
    train(parse_args())
