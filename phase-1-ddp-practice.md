# 第 1 阶段实践记录：可解释的 PyTorch DDP Trainer

日期：2026-09-15  
状态：CUDA/DDP 代码基线已建立；学习与实验拆分到 101 和 200 两台服务器，尚未产生可信的 CUDA 实测性能数据。

## 服务器分工

本阶段不建议把 101 和 200 直接组成多机 DDP。101 可以访问 Agent 但没有 GPU，200 有 GPU 但不能访问 Agent；两台机器承担不同职责：

| 服务器 | 主要职责 | 不承担的工作 |
| --- | --- | --- |
| 101 | 阅读原理、修改代码、运行单元测试和 CPU 控制流 smoke test、整理实验命令和分析结果 | CUDA 性能结论 |
| 200 | 接收经过验证的代码，使用真实 GPU 执行 CUDA/NCCL 单卡与多卡训练、AMP 和吞吐对比，保存日志与 checkpoint | Agent 辅助编码和在线技术检索 |

具体操作和检查清单见 [`phase-1-101-practice.md`](phase-1-101-practice.md) 与 [`phase-1-200-practice.md`](phase-1-200-practice.md)。代码通过 Git、内网文件同步或其他既有安全方式从 101 发布到 200；不要在 200 上依赖 Agent 服务。

## 目标与边界

本次实践落实转型方案中“第 0 阶段建立基线”和“第 1 阶段手写最小 DDP trainer”的第一项交付。它不是模型效果项目，而是训练系统实验底座。固定的合成二分类数据和小型 MLP 使模型收敛、数据读取和网络环境的变量尽量少，后续可以把瓶颈定位集中在训练循环、内存、集合通信和恢复语义上。

当前交付包括：DDP、AMP（CUDA）、梯度累积、分布式采样、rank 0 checkpoint、resume、JSONL 指标和基础单元测试。NPU 运行需要把设备选择、自动混合精度和通信后端替换为所在厂商的 PyTorch 适配包和 HCCL 后端；不应把 CUDA 命令直接当作 NPU 验证。

## 工程地图

| 文件 | 作用 |
| --- | --- |
| `ddp_baseline/train.py` | 训练入口、进程组、训练循环、指标和 checkpoint |
| `tests/test_train.py` | 合成数据可复现性与 checkpoint 恢复的回归测试 |
| `runs/<experiment>/config.json` | 实验配置快照 |
| `runs/<experiment>/metrics.jsonl` | 每 epoch 的 loss 与吞吐记录 |
| `runs/<experiment>/checkpoint.pt` | 最新完整训练状态，按 optimizer step 更新 |
| `runs/<experiment>/checkpoint-step-XXXXXXXX.pt` | 带 global step 编号的 checkpoint 快照 |

## 已实现的关键机制

### 1. rank、world size 与设备映射

`torchrun` 启动多个独立 Python 进程，并注入 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`。`rank` 是全局进程编号，`world size` 是参与训练的进程总数，`local rank` 决定该节点进程绑定的 GPU。CUDA 使用 NCCL，CPU 使用 Gloo。每个 rank 只处理自己数据分片上的前向和反向；DDP 在需要时 all-reduce 梯度，使下一次 optimizer step 的参数保持一致。

### 2. 数据不重复与可复现

`DistributedSampler` 按 rank 切分数据。这里显式设为 `drop_last=True`，避免为凑齐 rank 数而补齐并重复尾部样本，代价是每个 epoch 会舍弃无法整除 world size 的少量样本。若业务要求一个 epoch 覆盖全量数据，可改为补齐策略，但必须在报告中说明潜在重复。每一个 epoch 必须调用 `sampler.set_epoch(epoch)`，否则每轮 shuffle 顺序相同，既影响训练随机性，也是在面试中很常见的 DDP 漏项。合成数据和 sampler 都固定了 seed；实验报告仍需记录 PyTorch 版本、硬件、驱动和启动命令。

### 3. 梯度累积和通信

micro-batch 的 loss 先除以当前累积窗口的 micro-batch 数，因此累积后的梯度量级与一个等效大 batch 一致；最后一个不足 `accumulation_steps` 的窗口也会正确缩放。非更新步通过 `model.no_sync()` 抑制 DDP 梯度同步，最后一个 micro-batch 再通信并更新参数。这是计算/通信时间权衡的最小可观察点：累积能降低同步频率，却会提高单次参数更新延迟并改变有效 batch size。

### 4. AMP 与显存

CUDA 路径用 `torch.autocast` 和 `GradScaler`。前者选择适合的低精度算子执行，后者在 FP16 情况下减轻梯度下溢。AMP 不保证一定更快或更省显存，必须以同一模型、batch、数据和代码版本做 A/B 测量。BF16 的策略与硬件支持应单独记录。

### 5. checkpoint 与一致性

仅 rank 0 写 checkpoint，避免多进程竞争同一路径。现在按 optimizer step 保存，包含 model、optimizer、scheduler、scaler、epoch、当前 epoch 内已完成 batch、global step、samples_seen、各 rank RNG 和配置；恢复会跳过已完成 batch，从中断位置继续。epoch 结束时额外保存边界状态。这个版本仍未实现分片 checkpoint、动态 batch、streaming token 游标和多数据源采样状态；生产版本还要考虑这些状态、保存前 barrier 和对象存储的最终一致性。

## 验证步骤

1. 安装 Python 3.10+ 和匹配机器的 PyTorch；GPU 场景从 PyTorch 官方安装页选择对应 CUDA wheel，NPU 场景使用厂商适配的 PyTorch 发行包。
2. 创建虚拟环境并安装依赖：`python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`。
3. 执行单元测试：`python -m unittest discover -s tests -v`。
4. 在 101 完成 CPU 双进程控制流测试：`torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train --device cpu --epochs 2 --samples 512 --batch-size 16 --checkpoint-dir runs/cpu-101`。
5. 将代码同步到 200，在 200 确认 `nvidia-smi` 和 `torch.cuda.is_available()` 都能看到 GPU，再运行 CUDA 基线：`torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train --device cuda --amp --epochs 5 --checkpoint-dir runs/cuda-200`。
6. 在 200 使用 `--resume runs/cuda-200/checkpoint.pt` 再跑一次，确认 step 和 batch 位置连续；把 `metrics.jsonl`、step checkpoint、配置和硬件信息回传 101 分析。

不要把 CPU smoke test 的吞吐与 GPU 结果比较；它只验证进程组、采样和恢复的控制流。

## 基线记录模板

每一次实验创建独立的 `runs/<name>` 目录，并在报告中填完整下表。没有实测值时填写“未测”，不要估算。

| 项目 | 值 |
| --- | --- |
| 日期、代码提交/文件版本 | 未测 |
| 主机、GPU/NPU 型号和数量、驱动 | 未测 |
| PyTorch、CUDA/HCCL/NCCL 版本 | 未测 |
| world size、batch size、累积步数、精度 | 未测 |
| 模型、参数量、序列长度/输入形状、数据来源 | 本项目：MLP，32 维合成输入 |
| 每 epoch samples/s、step/s、峰值显存 | 未测 |
| 通信耗时占比、MFU、失败率 | 未测；需 profiler 和长稳实验补充 |
| 启动命令与异常日志位置 | 见本节验证步骤 |

本基线目前只输出 samples/s。MFU、显存峰值和通信占比需要进入第 2 阶段后，使用 PyTorch Profiler 与硬件厂商 profiler 在真实模型上采集；不要从这个 MLP 推导大模型的 MFU。

## 首轮排障清单

| 现象 | 首先检查 | 原因方向 |
| --- | --- | --- |
| DDP hang | 所有 rank 是否执行相同 collective 次数；`TORCH_DISTRIBUTED_DEBUG=DETAIL` | 控制流分叉、某 rank OOM/退出、网络或通信库问题 |
| loss 不一致/不收敛 | `DistributedSampler`、loss 缩放、每 rank batch、学习率 | 数据重复、累积缩放错误、有效 batch 改变 |
| OOM | batch、激活、optimizer state、AMP、碎片 | 模型状态或激活超出显存；先测峰值再选 FSDP/ZeRO |
| resume 后指标异常 | model/optimizer 是否同时恢复，epoch 是否加一 | 状态不完整或重复训练一个 epoch |

## 云账号与资源决策

**当前不需要创建云账号。** 101 负责代码和单元测试，200 已提供真实 GPU；只有当 200 的 GPU 资源不足或需要多机实验时，才评估额外资源。

当需要验证 GPU/NPU 多进程性能而本地没有至少两张可用加速卡时，才需要你提供以下任一种资源：公司/实验室集群权限，或一个云账号和已启用的计费方式。创建账号前请先确认预算、地区、GPU/NPU 型号、镜像是否支持 PyTorch 与 SSH 访问；只租按量付费实例，并设置预算告警和自动关机。不要在没有明确预算上限的情况下创建长期运行实例。

多机阶段还需要至少两台节点、节点间网络信息和共享/对象存储方案；这属于后续第 2--4 阶段，不是本次基线的阻塞条件。

## 下一步

先在 101 固化代码和 CPU 控制流，再在 200 按同一配置运行 1 卡与 2 卡 CUDA 基线并保存结果，再开始 profiler。只有先有基线，后续 AMP、梯度累积、FSDP/ZeRO 或通信计算重叠的收益才可归因。
