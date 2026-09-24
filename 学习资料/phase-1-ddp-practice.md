你的判断是对的。之前那些文档确实是“架构设计书”而不是“操作手册”——它们告诉你“要做一个平台”，但没告诉你“DDP 的梯度到底在哪一步同步”“AMP 的 scaler 漏更新会导致什么”“checkpoint 少了什么状态会导致恢复后 loss 跳变”。这些恰恰是你面试时会被追问、也是从 NPU 建模转向训练 Infra 必须补上的“训练系统底座”。

下面这份手册按**第 1 阶段（第 1–2 个月）**设计，目标就是你的总体方案里那句话：**“手写一个最小但可解释的 PyTorch DDP trainer，包含数据并行、AMP、梯度累积、断点续训、日志和单元测试”**。

---

## 第 1 阶段操作手册：训练系统底座

> 周期：第 1–2 个月，每周 15–20 小时
> 环境：101（CPU 开发/测试）+ 200（GPU 真实训练）
> 产出：一个可解释的 DDP trainer，代码 + 实验数据 + 设计文档

### 0. 本阶段要回答的问题

做完这个阶段，你应该能**不看代码**回答：

1. `rank` 和 `world_size` 各自表示什么？`LOCAL_RANK` 和 `RANK` 在多机场景下有什么区别？
2. DDP 的梯度同步发生在 backward 的哪一步？为什么说它和 backward 计算是**重叠**的？
3. AMP 的 `GradScaler` 如果不调用 `update()` 会发生什么？为什么梯度裁剪前必须 `unscale_`？
4. checkpoint 里必须存哪些东西才能保证恢复后 **step 连续、loss 不跳变**？
5. 一个训练 step 里，计算、通信、数据加载各占多少时间？怎么用 profiler 证明？

### 1. 环境与基线（第 1 周）

#### 1.1 101 侧：准备纯 CPU 开发环境

```bash
cd /workspace/GIT/ai_infra
. .venv/bin/activate
# 确认 PyTorch 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**目标**：在 101 上能跑通单进程的 toy training loop，不涉及分布式。

#### 1.2 200 侧：确认 GPU 与通信后端

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 确认 NCCL 可用（PyTorch 自带）
python -c "import torch.distributed as dist; print(dist.is_nccl_available())"
```

**记录到 `docs/evidence/environment.md`**：GPU 型号、驱动版本、PyTorch 版本、NCCL 版本、CUDA 版本。

#### 1.3 知识补充：DDP 到底做了什么

DDP 的核心行为在 PyTorch 官方文档里写得很清楚：

- **构造时**：rank 0 的 `state_dict()` 被广播到所有进程，保证所有副本初始参数完全一致。
- **forward 时**：每个进程独立计算，没有通信。
- **backward 时**：`Reducer` 把参数梯度分成多个 bucket，**按 backward 中梯度就绪的顺序**逐 bucket 做 all-reduce。这就是“通信与计算重叠”的来源——当某个 bucket 的梯度算完，它的 all-reduce 可以立刻开始，不需要等整个 backward 结束。
- **backward 返回时**：`param.grad` 已经包含**同步后的梯度**，可以直接 `optimizer.step()`。

**面试会追问**：bucket 的顺序为什么是 `Model.parameters()` 的**反序**？因为 DDP 期望梯度在 backward 中大致按这个顺序就绪，这样 bucket 可以尽早开始通信。

#### 1.4 动手：一个 20 行的 DDP toy example

在 `training-platform-mvp/tests/` 下建一个 `toy_ddp.py`，用官方文档里的最小示例改写：

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def example(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(example, args=(2,), nprocs=2, join=True)
```

在 101 上跑（CPU 用 gloo 后端）：

```bash
python toy_ddp.py
```

**验收**：两个进程都正常退出，无 hang。

**记录到 `docs/evidence/day01-toy-ddp.md`**：命令、输出、你的解释（rank/world_size 的含义、gloo 和 nccl 的区别）。

---

### 2. DDP trainer 核心实现（第 2–3 周）

#### 2.1 项目结构

在 `training-platform-mvp/trainer/` 下建立：

```text
trainer/
  __init__.py
  ddp_trainer.py       # 主训练脚本
  dataset.py           # 数据加载 + DistributedSampler
  model.py             # 模型定义（先用小模型，后期换真实模型）
  checkpoint.py        # checkpoint 保存/加载
  utils.py             # 日志、指标、seed 固定
tests/
  test_ddp_setup.py    # 单机多进程集成测试
  test_checkpoint.py   # checkpoint 往返测试
```

#### 2.2 进程组初始化（`ddp_trainer.py`）

```python
import os
import torch
import torch.distributed as dist

def setup_distributed():
    """从 torchrun 提供的环境变量初始化进程组。"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 关键：先设置设备，再初始化进程组，避免 GPU 0 被所有进程占用
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",  # GPU 用 nccl，CPU 用 gloo
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()
```

**为什么先 `set_device` 再 `init_process_group`**：如果不先设设备，所有进程默认用 GPU 0 做通信初始化，在 GPU 0 上产生不必要的显存占用和初始化竞争。

**`LOCAL_RANK` vs `RANK`**：
- `RANK`：全局进程编号，多机场景下跨节点唯一。
- `LOCAL_RANK`：**本机内的 GPU 编号**，用来 `set_device`。
- 单机场景下两者相同；多机场景下必须区分。

#### 2.3 数据加载：`DistributedSampler`

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def create_dataloader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,   # 注意：用 sampler 时不要再设 shuffle=True
        num_workers=4,
        pin_memory=True,
    )
    return loader, sampler
```

**关键细节**：每个 epoch 开始前必须调用 `sampler.set_epoch(epoch)`，否则每个 epoch 的数据顺序**完全相同**，shuffle 失效。

```python
for epoch in range(epochs):
    sampler.set_epoch(epoch)   # ← 必须调用
    for batch in loader:
        ...
```

**面试追问**：为什么 `DistributedSampler` 要按 rank 切分数据？如果不切分会怎样？
→ 每个 rank 训练全量数据，等于重复计算 `world_size` 倍，且梯度同步时“平均”了相同的更新，收敛极慢。

#### 2.4 模型包装与训练循环

```python
from torch.nn.parallel import DistributedDataParallel as DDP

def build_ddp_model(model, local_rank):
    model = model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    return ddp_model

def train_one_epoch(ddp_model, loader, optimizer, scheduler, criterion, epoch,
                    rank, scaler=None, grad_accum_steps=1):
    ddp_model.train()
    loader.sampler.set_epoch(epoch)   # 每个 epoch 重置 sampler

    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # AMP forward
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps   # 梯度累积时缩放 loss

        # AMP backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积：每 accum_steps 步才更新一次
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                # 梯度裁剪前必须 unscale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
```

**这里每一行都有面试考点**：

| 代码 | 为什么必须这样写 |
|---|---|
| `loss / grad_accum_steps` | 累积 `N` 个 micro-batch 后 loss 之和是原 loss 的 `N` 倍，梯度也是 `N` 倍；不除的话等效 learning rate 变了 |
| `scaler.scale(loss).backward()` | 把 loss 乘一个缩放因子，防止 fp16 梯度下溢 |
| `scaler.unscale_(optimizer)` 在 `clip_grad_norm_` 前 | `clip_grad_norm_` 的 `max_norm` 是针对**未缩放**梯度的阈值；不 unscale 的话阈值失效 |
| `scaler.step(optimizer)` 替代 `optimizer.step()` | scaler 检查梯度是否有 inf/NaN，有则**跳过这一步更新**，防止 NaN 污染参数 |
| `scaler.update()` | 根据是否发生 inf/NaN，动态调整缩放因子 |

#### 2.5 验证 DDP 正确性

**测试 1：参数一致性**

```python
def test_ddp_consistency():
    """两个 rank 的模型参数在 optimizer.step() 后必须完全一致。"""
    # 在 toy 模型上跑一步，对比 rank 0 和 rank 1 的 state_dict
```

**测试 2：梯度同步验证**

在 `ddp_model.backward()` 之后、`optimizer.step()` 之前，检查 `param.grad` 是否已被 all-reduce。方法：让两个 rank 产生**不同的 loss**（通过不同的输入），确认梯度仍然是同步后的值。

**记录到 `docs/evidence/day02-ddp-consistency.md`**。

---

### 3. AMP 与梯度累积（第 4 周）

#### 3.1 AMP 的正确用法

完整训练循环中的 AMP 使用：

```python
from torch.amp import GradScaler, autocast

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast(device_type="cuda", dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)   # 如果要用梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

#### 3.2 梯度累积与 DDP 的交互

梯度累积在 DDP 下的一个**容易错**的点：`DDP` 的梯度 all-reduce **发生在每次 `backward()` 调用时**，即使你还没调用 `optimizer.step()`。

也就是说，如果你做 `accum_steps=4`：

- 第 1 次 backward：DDP 做 all-reduce，梯度被同步。
- 第 2 次 backward：DDP **再做一次** all-reduce，把新梯度加上去再同步。
- …

结果：梯度被同步了 4 次，虽然最终值和“先累积再同步”**在数学上等价**（all-reduce 是线性的），但**通信开销没有节省**。

**面试考点**：DDP + 梯度累积时，通信次数是否随 `accum_steps` 减少？
→ **不减少**。DDP 在每个 backward 都会触发 all-reduce。如果要真正节省通信，需要用 `no_sync()` 上下文：

```python
for i, (data, target) in enumerate(loader):
    with ddp_model.no_sync() if (i + 1) % accum_steps != 0 else contextlib.nullcontext():
        outputs = ddp_model(data)
        loss = criterion(outputs, target) / accum_steps
        scaler.scale(loss).backward()
    # 只有第 accum_steps 步才真正同步梯度
```

`no_sync()` 会**跳过本次 backward 的 all-reduce**，梯度只在本地累积；最后一次 backward 时才做同步。

#### 3.3 练习

1. 跑一个 toy 模型，对比 `accum_steps=1` 和 `accum_steps=4` 的：
   - 每步 wall time
   - GPU 显存占用
   - 最终 loss 是否一致
2. 用 `no_sync()` 改写，对比通信时间。

**记录到 `docs/evidence/day04-amp-grad-accum.md`**。

---

### 4. Checkpoint 与恢复（第 5–6 周）

#### 4.1 checkpoint 里必须存什么

根据生产级 checkpoint 实践：

```python
def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler=None,
                    global_step=0, best_metric=None):
    checkpoint = {
        # 核心状态
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,

        # 随机性状态 —— 不存这些，恢复后 loss 会跳变
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),  # 所有 GPU
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),

        # AMP 状态
        "scaler_state_dict": scaler.state_dict() if scaler else None,

        # 元信息
        "best_metric": best_metric,
        "pytorch_version": torch.__version__,
        "config": {"lr": ..., "batch_size": ...},
    }
    torch.save(checkpoint, path)
```

**关键**：
- `cuda_rng_state` 必须用 `get_rng_state_all()`，不能用 `get_rng_state()`——后者只存当前设备。
- 不存 `rng_state` 和 `cuda_rng_state`，恢复后 dropout、数据 shuffle 的随机序列不连续，loss 曲线会“跳”。
- checkpoint 里存 `pytorch_version`，防止跨版本加载时静默出错。

#### 4.2 原子写入

```python
import os
import tempfile

def atomic_save(checkpoint, path):
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)   # 原子操作，要么成功，要么保持旧文件
```

**为什么**：如果 `torch.save` 写到一半进程被 kill，直接写目标路径会留下**损坏的 checkpoint**，恢复时才发现。先写 `.tmp` 再 `os.replace`，保证任何时刻磁盘上的 `checkpoint.pt` 要么是旧的完整版，要么是新的完整版。

#### 4.3 恢复逻辑

```python
def load_checkpoint(path, model, optimizer, scheduler, scaler=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    torch.set_rng_state(checkpoint["rng_state"].cpu())
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # 关键：checkpoint 在 epoch N 保存，表示 epoch N 已完成
    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]

    return start_epoch, global_step, checkpoint.get("best_metric")
```

**面试考点**：checkpoint 在 epoch 5 结束时保存，恢复后应该从哪个 epoch 开始？
→ **epoch 6**。如果从 5 开始，epoch 5 会被跑两次。

#### 4.4 练习：故障恢复验证

1. 在 200 上跑一个训练，`checkpoint-every-steps=10`。
2. 跑到 step 50 时，`kill -TERM -<PGID>` 杀掉进程组。
3. 用 `--resume` 从最近 checkpoint 恢复。
4. 对比两次 `metrics.jsonl` 的 `global_step`：**第二次的起始 step 必须 ≥ 第一次已保存的最大 step**，不能倒退。

**记录到 `docs/evidence/day05-checkpoint-resume.md`**：注入命令、checkpoint 内容、恢复前后的 step、loss 曲线。

---

### 5. Profiler 与性能分析（第 7–8 周）

#### 5.1 用 PyTorch Profiler 找到瓶颈

官方 profiler 可以同时记录 CPU 和 CUDA 活动：

```python
from torch.profiler import profile, ProfilerActivity

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/ddp-profile"),
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
```

**关键参数**：
- `wait=1`：跳过第 1 步（编译/warmup 开销）。
- `warmup=1`：第 2 步开始采样但**丢弃结果**。
- `active=3`：第 3–5 步**记录结果**。
- `repeat=1`：整个周期重复 1 次。

**为什么要 warmup**：第一次运行有 cuDNN benchmark、内存分配等一次性开销，直接记录会污染数据。

#### 5.2 分析 DDP 的通信占比

在 profiler 输出中找 `nccl:all_reduce` 或 `nccl:allgather` 的耗时：

```python
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**你要回答的问题**：
- all-reduce 占 backward 总时间的百分比是多少？
- 如果通信占比 >30%，瓶颈可能在：梯度 bucket 太大（通信无法与计算充分重叠）、GPU 间带宽不足、或 backward 计算本身太慢。
- 对比 1 卡和 2 卡的 step time：如果 2 卡 step time > 1 卡 step time × 1.5，通信效率可能有问题。

#### 5.3 NCCL 基础概念

NCCL 的通信原语包括 AllReduce、Broadcast、AllGather、ReduceScatter。DDP 的梯度同步用的是 **AllReduce**。

**面试可能问**：AllReduce 和 ReduceScatter + AllGather 的关系？
→ Ring AllReduce 在实现上就是 ReduceScatter + AllGather 的两阶段。NCCL 会根据拓扑（NVLink、PCIe、InfiniBand）选择最优算法。

**记录到 `docs/evidence/day06-profiler-ddp.md`**：trace 文件、关键指标表、你的分析结论。

---

### 6. 本阶段交付清单

| 交付物 | 路径 | 内容 |
|---|---|---|
| DDP trainer | `trainer/ddp_trainer.py` | 完整的单机多卡训练脚本 |
| 数据集/采样器 | `trainer/dataset.py` | DistributedSampler 正确使用 |
| Checkpoint 模块 | `trainer/checkpoint.py` | 原子写入 + 完整状态保存 + 恢复 |
| 单元测试 | `tests/test_ddp_setup.py` | 进程组初始化、参数一致性 |
| 单元测试 | `tests/test_checkpoint.py` | checkpoint 往返、RNG 恢复 |
| 环境证据 | `docs/evidence/environment.md` | GPU/PyTorch/NCCL 版本 |
| DDP 证据 | `docs/evidence/day02-ddp-consistency.md` | 参数一致性验证 |
| AMP 证据 | `docs/evidence/day04-amp-grad-accum.md` | 累积步数对比 |
| 恢复证据 | `docs/evidence/day05-checkpoint-resume.md` | 杀进程 → 恢复 → step 连续 |
| Profiler 证据 | `docs/evidence/day06-profiler-ddp.md` | 通信占比分析 |

---

### 7. 面试自测题（来自本阶段知识）

1. DDP 的 `Reducer` 为什么按 backward 顺序分 bucket？
2. `LOCAL_RANK` 和 `RANK` 在多机场景下分别用来做什么？
3. AMP 的 `GradScaler` 跳过 `optimizer.step()` 的机制是什么？
4. 梯度累积时，DDP 的通信次数会减少吗？怎么用 `no_sync()` 减少？
5. checkpoint 里不存 `rng_state` 会导致什么现象？
6. `torch.cuda.get_rng_state()` 和 `get_rng_state_all()` 的区别？
7. profiler 的 `wait/warmup/active` 各参数的含义？
8. NCCL 的 AllReduce 在 Ring 算法下等价于哪两个操作的组合？

---

### 8. 与你总体目标的衔接

这份手册直接对应你总体方案里**第 1 阶段的实践交付**，也是后续第 2 阶段（性能优化）和第 3 阶段（平台 MVP）的基础：

- **第 2 阶段做性能优化时**，你已经有 profiler 能力和 DDP/AMP 的正确理解，才能定位“通信占比高”还是“计算瓶颈”。
- **第 3 阶段做平台时**，checkpoint 模块可以直接复用，`LocalProcessExecutor` 启动的 `torchrun` 命令的 `--nproc_per_node`、`--amp`、`--checkpoint-every-steps` 参数都有了代码出处，不再是黑盒。

你可以把这份手册当成**阶段 1 的“操作卡”**：每周照着做，产生对应的证据文件。做完之后，你手上有的是**可解释的代码 + 可验证的数据**，而不是“我学过 DDP”这种简历话术。