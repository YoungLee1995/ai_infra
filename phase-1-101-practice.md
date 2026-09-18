# 第 1 阶段实践：101 开发与验证任务

## 服务器定位

101 是 Agent 和代码开发中心，但没有显卡。这里不做 GPU 性能结论，主要完成理解、实现、回归验证和实验准备，然后把稳定版本同步到 200。

## 学习顺序

1. 阅读 `ddp_baseline/train.py`，按下表标出进程组初始化、设备映射、sampler、`no_sync`、AMP 和 checkpoint 的位置，并对照代码中的 `[机制标签]` 注释。
2. 修改训练代码或增加测试，先用单元测试验证数据可复现和 checkpoint 恢复。
3. 运行 CPU 双进程 smoke test，确认各 rank collective 次数一致、loss 能下降、rank 0 独占写文件。
4. 设计 200 上的 1 卡/2 卡对照实验，固定代码版本、数据量、batch size、累积步数和 epoch。
5. 将代码、启动命令和实验配置同步给 200；实验结果回传后在 101 分析吞吐、loss、恢复和异常日志。

## `train.py` 代码标注地图

| 机制 | 位置 | 代码段作用 |
| --- | --- | --- |
| 设备映射 | `configure_process_group()`，约第 80-88 行 | 读取 `LOCAL_RANK`，将当前进程绑定到对应 CUDA GPU；CPU 路径使用 `gloo`。 |
| 进程组初始化 | `configure_process_group()`，约第 96-100 行 | 多进程时通过 `env://` 读取 `torchrun` 环境变量并建立 NCCL/Gloo 通信组；单进程不初始化通信组。 |
| DistributedSampler | `train()`，约第 164-171 行 | 将数据索引分给不同 rank，避免多个进程重复读取相同 batch；每轮在约第 193-194 行调用 `set_epoch()` 更新 shuffle。 |
| DDP 包装 | `train()`，约第 175-177 行 | 为多进程模型注册梯度同步；CUDA 通过 `device_ids` 指定当前进程的 GPU。 |
| AMP | `autocast_context()`，约第 153-155 行；`train()` 约第 179、206-209 行 | autocast 在 CUDA 上使用 float16，GradScaler 缩放 loss 和梯度，降低 FP16 梯度下溢风险。 |
| `no_sync()` | `train()`，约第 203-209 行 | 梯度累积的中间 micro-batch 跳过 all-reduce，更新前的最后一个 micro-batch 才同步梯度。 |
| Checkpoint 保存 | `save_checkpoint()`，约第 209-241 行；调用点在训练循环内和 epoch 边界 | rank 0 按 optimizer step 保存模型、优化器、scheduler、scaler、epoch、batch、global step、samples_seen 和 RNG，并通过临时文件替换降低写入中断风险。 |
| Checkpoint 恢复 | `load_checkpoint()`，约第 244-273 行；调用点约第 319-328 行 | 校验关键配置，恢复模型、优化器、scheduler、scaler、RNG 和位置状态，使训练从中断位置继续。 |

### 先读哪几段

建议按“设备/进程组 → 数据 sampler → DDP 包装 → 训练循环 → `no_sync`/AMP → checkpoint”的顺序阅读。每段先回答两个问题：当前代码运行在什么进程/设备上？这个操作是否需要所有 rank 同时参与？

## 101 上可执行命令

```bash
python3 -m unittest discover -s tests -v
torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cpu --epochs 2 --samples 512 --batch-size 16 \
  --checkpoint-dir runs/cpu-101
```

这一步只验证 DDP 控制流，不模拟 GPU 性能。101 上不要把 CPU 的 samples/s 与 200 的 GPU 结果放在同一张性能结论表里。

## 发布给 200 前的检查

- `python3 -m py_compile ddp_baseline/train.py tests/test_train.py` 通过。
- 单元测试通过。
- CPU 双进程训练生成 `config.json`、`metrics.jsonl` 和 `checkpoint.pt`。
- 启动命令明确包含 `--device cuda --amp`，并设置与 200 可见 GPU 数量匹配的 `--nproc_per_node`。
- 记录当前代码版本、配置、预期输出文件和恢复命令。

## 101 的核心学习目标

101 重点掌握 DDP 的控制语义：rank/world size/local rank、sampler.set_epoch、梯度同步、累积窗口缩放、rank 0 checkpoint 和 resume。Transformer 学习也应先在 101 完成模型结构、mask、loss 和测试，再把可运行版本交给 200 做 GPU 实验。
