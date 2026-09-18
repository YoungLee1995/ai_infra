# 第 1 阶段实践：200 CUDA 实验任务

## 服务器定位

200 是真实 GPU 实验机，但不能运行 Agent。这里不依赖在线 Agent 编码，使用 101 已验证并同步过来的代码，专门测量 CUDA、NCCL、DDP、AMP 和 checkpoint 行为。

## 开始前检查

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

只有 `torch.cuda.is_available()` 为 `True` 且设备数量大于 0，才继续 CUDA 实验。`--nproc_per_node` 不得超过可见 GPU 数量。

## 实验顺序

1. 从 101 同步代码和依赖，确认文件版本与 CPU 验证版本一致。
2. 先跑 1 卡 CUDA 基线，记录 GPU 型号、PyTorch/CUDA/NCCL 版本、batch、累积步数、AMP 状态、samples/s 和 checkpoint。
3. 再用完全相同配置跑 2 卡 DDP；只改变 `--nproc_per_node`，不要同时改变 batch、数据或 epoch。
4. 对同一配置做 `--amp` 与关闭 AMP 的对照，单独记录吞吐和显存峰值。
5. 设置 `--checkpoint-every-steps` 和 `--warmup-steps` 做 step 级保存，测试 `--resume`，确认已保存的 epoch/batch/global step、samples_seen、scheduler 和 RNG 能恢复，并把结果和日志回传 101。

## CUDA 命令模板

```bash
# 单卡
torchrun --standalone --nproc_per_node=1 -m ddp_baseline.train \
  --device cuda --amp --epochs 5 --batch-size 64 \
  --checkpoint-dir runs/cuda-1gpu

# 双卡
torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 5 --batch-size 64 --checkpoint-every-steps 100 \
  --checkpoint-dir runs/cuda-2gpu

# 恢复
torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 8 --checkpoint-dir runs/cuda-2gpu \
  --resume runs/cuda-2gpu/checkpoint.pt
```

如果 200 不能访问 Agent 服务，不影响训练脚本执行；脚本、配置和命令应由 101 发布。不要在 200 上直接修改实验代码后忘记回传，否则性能结果无法复现。

## 结果回传清单

- `config.json`、`metrics.jsonl`、`checkpoint.pt` 是否生成。
- `checkpoint-step-XXXXXXXX.pt` 是否按指定 optimizer step 生成。
- `nvidia-smi` 输出中的 GPU 型号、数量、显存占用。
- PyTorch、CUDA、NCCL 和驱动版本。
- 1 卡与 2 卡的 samples/s、loss、运行时间和失败信息。
- AMP 开关、梯度累积步数、启动命令和完整 stderr/stdout。

当前 MLP 只用于验证训练系统。不要从它推导 Transformer 或大模型的 MFU；后续 Transformer 实验仍需先在 101 完成结构和测试，再在 200 测性能。
