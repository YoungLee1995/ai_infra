## 第 2 阶段操作手册：性能优化与瓶颈定位

> 周期：第 3–4 个月，每周 15–20 小时
> 环境：200（GPU 真实训练）
> 前置：完成第 1 阶段，已有可运行的 DDP trainer 和 profiler 基础
> 产出：一份包含 profiling → 假设 → 改动 → 对照实验的完整性能优化报告

### 0. 本阶段要回答的问题

做完这个阶段，你应该能在面试中**用数据回答**：

1. 你的训练 MFU 是多少？瓶颈在计算、通信还是数据加载？
2. DDP 的 AllReduce 占 backward 时间的百分比？如何减少？
3. ZeRO-1/2/3 各自省了哪部分显存？代价是什么？
4. 梯度累积时通信次数会不会减少？`no_sync()` 省了多少时间？
5. Checkpoint 的同步写入占了多少训练时间？异步写入能省多少？
6. 你做了什么改动，端到端吞吐提升了多少百分比？

### 1. 性能体检框架（第 3 周第 1–2 天）

在改任何代码之前，先建立**可量化的基线**。没有基线，任何“优化”都无法证明有效。

#### 1.1 统一度量标准

固定以下变量，后续所有实验只在**一个维度**上变化：

| 维度 | 固定值 |
|---|---|
| 模型 | 选定一个 1B–7B 的开源模型（如 GPT-2 1.5B 或同等规模） |
| 数据 | 固定数据集和预处理逻辑 |
| 硬件 | 200 上的 GPU 型号和数量（单卡或双卡） |
| 精度 | 明确 fp32 / fp16 / bf16 |
| batch size | 全局 batch = per_device_batch × world_size × grad_accum |
| 序列长度 | 固定（如 2048） |

#### 1.2 必测的四个指标

| 指标 | 含义 | 获取方式 |
|---|---|---|
| **吞吐** | samples/s 或 tokens/s | 训练日志计时 |
| **step time** | 每个 optimizer step 的 wall time（毫秒） | 日志计时 |
| **MFU** | 有效 FLOPs / 硬件峰值 FLOPs | 代码计算或 `mfu_tracker`  |
| **峰值显存** | 训练过程中的最大 GPU 显存占用 | `torch.cuda.max_memory_allocated()` |

**MFU 是核心指标**。GPU 利用率 90% 不代表高效——可能 90% 时间在跑低效内核。MFU 才反映“实际算出的有效运算占硬件理论峰值的比例”。稠密模型预训练的健康线是 **35%–50%**；低于 35% 提示有瓶颈。

#### 1.3 建立基线：跑一次 profiling

用第 1 阶段学到的 PyTorch Profiler，采集 20 个稳定 step（warmup 10 step 后）：

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=5, warmup=5, active=20),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_baseline"),
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
        if step >= 30:
            break
```

**产出**：`docs/evidence/baseline-profile.md`，包含：
- 命令、模型、batch、精度、GPU 型号
- samples/s、step time、MFU、峰值显存
- profiler trace 文件路径
- **时间线截图**：GPU 内核是否紧密连续？是否有“空洞”？

#### 1.4 读懂时间线：三种“病”

用 profiler 的时间线视图（Perfetto 或 Chrome）识别瓶颈类型：

| 形态 | 表现 | 病因 |
|---|---|---|
| **GPU 饿肚子** | 内核之间有长空洞 | CPU 喂不动：数据加载慢、Python 开销大 |
| **小内核风暴** | 大量几微秒的内核 | 启动开销拖垮吞吐：需要 CUDA Graph 或算子融合 |
| **同步等待** | 某个 rank 在等别的 rank | 通信未重叠或掉队者 |

**先诊断，再开药。** 看到哪种形态，就知道该改哪一层。

### 2. 显存优化：ZeRO 与激活重计算（第 3 周第 3–5 天）

如果你的瓶颈是**显存不够**（OOM 或无法增大 batch），优先做这一块。

#### 2.1 理解显存构成

训练时 GPU 显存被四样东西占用：

| 组成 | 典型大小（1.5B 模型，Adam） |
|---|---|
| 模型参数（fp16） | ~3 GB |
| 梯度（fp16） | ~3 GB |
| **优化器状态（Adam：fp32 参数副本 + 一阶矩 + 二阶矩）** | **~18 GB** |
| 激活值 | 取决于 batch 和序列长度 |

**Adam 优化器状态是显存大头**。DeepSpeed 的实测显示，1.5B GPT-2 的 Adam 状态消耗 18 GB，在 32 GB V100 上直接 OOM。

#### 2.2 ZeRO 的三个阶段

ZeRO（Zero Redundancy Optimizer）通过**分片**消除数据并行中的冗余存储：

| 阶段 | 分片内容 | 显存节省（8 卡理想值） | 代价 |
|---|---|---|---|
| **ZeRO-1** | 优化器状态 | 从 18 GB → 2.25 GB（约 87.5%） | 每步 update 后需要 AllGather 参数 |
| **ZeRO-2** | + 梯度 | 进一步减少梯度冗余 | backward 时 ReduceScatter 梯度 |
| **ZeRO-3** | + 参数 | 参数也分片，显存线性扩展 | forward/backward 每层都要 AllGather 参数，**通信量显著增加** |

**ZeRO-1 是最划算的起点**：仅分片优化器状态，通信开销小，显存节省巨大。ZeRO-3 适合“模型太大单卡放不下”的场景，但通信代价高。

**面试考点**：ZeRO-3 的参数分片和 TP（张量并行）有什么区别？
→ TP 是**层内**切分（权重矩阵按行/列拆），每层计算后需要 AllReduce；ZeRO-3 是**层间**分片（每卡只存部分层参数），forward 时按需 AllGather。

#### 2.3 激活重计算（Gradient Checkpointing）

激活值是 forward 时保存、backward 时使用的中间张量。激活重计算的做法是：**forward 时不保存全部激活，backward 时重新计算**。

- **显存节省**：可大幅降低激活内存（具体取决于 checkpoint 频率）。
- **计算代价**：多一次 forward，增加约 33% 计算量。
- **MFU 的影响**：重计算不影响 MFU（MFU 按算法 FLOPs 算），但**HFU（Hardware FLOPs Utilization）会上升**——GPU 实际做了更多计算。

**练习**：在 toy 模型上对比开启/关闭激活重计算的：
- 峰值显存
- step time
- 最大可用的 batch size

#### 2.4 实践任务

1. 在 DDP trainer 中集成 **DeepSpeed ZeRO-1**（只改 JSON 配置，不改模型代码）。
2. 对比 baseline 与 ZeRO-1 的：
   - 峰值显存（记录 `max_memory_allocated`）
   - step time
   - 最大 batch size
3. 如果显存仍不够，再尝试 ZeRO-2 和激活重计算。
4. **记录到 `docs/evidence/day10-zero-comparison.md`**。

### 3. 通信优化：重叠与分桶（第 4 周第 1–3 天）

如果你的时间线上有 **“同步等待”形态**（某个 rank 空转等通信），说明通信没有和计算充分重叠。

#### 3.1 DDP 的通信机制

DDP 在 backward 时做 **AllReduce** 同步梯度。关键优化是**梯度分桶（bucketing）**：

- DDP 不是每个参数梯度算完就通信，而是把相近参数的梯度攒成一个“桶”。
- 桶的顺序大致是 `Model.parameters()` 的**反序**，期望 backward 时梯度按这个顺序就绪。
- 当一个桶的梯度全部就绪，立刻在**通信流**上启动 AllReduce，同时 backward 在**计算流**上继续。

这就是 DDP 的“通信-计算重叠”：AllReduce 和剩余 backward 计算并行执行。

#### 3.2 检查重叠效果

在 profiler 中观察：
- `nccl:all_reduce` 的时间段是否与 `backward` 的 CUDA 内核时间段**重叠**？
- 如果 AllReduce 结束后还有很长的“空白”，说明重叠不充分。

**可能原因**：
- 桶太大：一个桶的 AllReduce 时间超过了剩余 backward 时间，无法完全隐藏。
- 桶太小：通信启动次数过多，开销大。
- **调整** `DDP(bucket_cap_mb=25)` 参数（默认 25 MB），观察对 step time 的影响。

#### 3.3 梯度累积与 `no_sync()`

第 1 阶段提到过：DDP 的 AllReduce **在每个 backward 时都会触发**，即使你还没调用 `optimizer.step()`。

```python
# 没有 no_sync：每次 backward 都 AllReduce（通信次数 = accum_steps）
loss.backward()  # ← 这里 AllReduce

# 有 no_sync：只累积本地梯度，最后一次才 AllReduce
with model.no_sync() if not is_last_accum_step else contextlib.nullcontext():
    loss.backward()  # ← 不 AllReduce
```

**`no_sync()` 的收益**：如果 `accum_steps=4`，通信次数从 4 次降到 1 次。**但注意**：AllReduce 是线性的，累积后的梯度数值不变，所以**数学上等价**，只是省了通信次数。

**练习**：
1. 对比 `accum_steps=4` 时，有/无 `no_sync()` 的 step time。
2. 计算通信时间占比：`(无 no_sync 的 step time - 有 no_sync 的 step time) / 无 no_sync 的 step time`。

#### 3.4 Ring AllReduce 原理（面试必备）

AllReduce 的两种经典实现：

| 算法 | 通信量 | 适用场景 |
|---|---|---|
| **Ring AllReduce** | 约 `2(N-1)/N × 数据量`，与卡数无关 | 带宽最优，大规模 |
| **Tree AllReduce** | 延迟更低（log 级跳数） | 中小规模或延迟敏感 |

Ring AllReduce 分两阶段：
1. **Scatter-Reduce**：N-1 轮，沿环传递并就地规约。
2. **All-Gather**：N-1 轮，沿环传递已规约的分片。

NCCL 会根据拓扑（NVLink、PCIe、InfiniBand）自动选择 Ring 或 Tree，并拆成多个 channel 并行执行。

**记录到 `docs/evidence/day12-comm-overlap.md`**。

### 4. 数据输入优化（第 4 周第 4–5 天）

如果时间线上有 **“GPU 饿肚子”形态**（内核之间大空洞），瓶颈在数据管线。

#### 4.1 诊断数据管线瓶颈

用**数据缓存策略**定位瓶颈在哪一层：

```python
# Step 1: 缓存一个 batch 到 GPU，循环训练，测吞吐上限
batch = next(iter(loader)).to(device)
for i in range(100):
    train_step(batch)  # 这是 GPU 计算的上限吞吐

# Step 2: 缓存一个 batch 到 CPU，每次 .to(device)
host_batch = next(iter(loader))
for i in range(100):
    batch = host_batch.to(device, non_blocking=True)
    train_step(batch)  # 如果明显变慢，瓶颈在 H2D 拷贝

# Step 3: 逐步前移缓存点（collate 后、augment 前、augment 后、raw load 后）
# 找到吞吐下降的位置，就是瓶颈所在
```

#### 4.2 常见优化手段

| 优化 | 做法 | 效果 |
|---|---|---|
| **多进程预取** | `num_workers = 4 × num_gpu`（经验值） | 掩盖 I/O 和预处理延迟 |
| **pin_memory + non_blocking** | `pin_memory=True`，`.to(device, non_blocking=True)` | H2D 拷贝异步化 |
| **persistent_workers** | `persistent_workers=True` | 避免每个 epoch 重建 worker |
| **存储优化** | 小文件合并（LMDB/TFRecord）、SSD/NVMe | 降低 I/O 延迟 |
| **预处理下沉** | 解码/增强放到 DataLoader worker 中 | 训练循环只做计算 |

**注意**：`num_workers` 不是越大越好。数据集小或预处理极轻时，过大的 worker 数会带来调度和内存开销。

#### 4.3 实践任务

1. 用缓存策略定位你的数据管线瓶颈。
2. 只改**一个参数**（如 `num_workers` 从 4 到 8），对比 step time。
3. 开启 `pin_memory + non_blocking`，对比 step time。
4. **记录到 `docs/evidence/day13-dataloader.md`**。

### 5. Checkpoint 优化：异步写入（第 4 周第 6–7 天）

同步 checkpoint 会**阻塞训练**：`torch.save` 期间 GPU 空转。

#### 5.1 测量 checkpoint 开销

```python
import time
start = time.time()
torch.save(checkpoint, path)
save_time = time.time() - start
print(f"Checkpoint save time: {save_time:.2f}s")
```

如果 `save_time` 占 step time 的显著比例，就需要异步化。

#### 5.2 使用 `dcp.async_save`

PyTorch 的 Distributed Checkpoint（DCP）提供 `async_save`，把 checkpoint 写入放到**后台线程/进程**，不阻塞训练循环：

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful

checkpoint_future = None
for step in range(total_steps):
    train_step()
    
    if step % checkpoint_every == 0:
        if checkpoint_future is not None:
            checkpoint_future.result()  # 等待上一次完成
        
        state_dict = {"app": AppState(model, optimizer)}
        checkpoint_future = dcp.async_save(
            state_dict,
            checkpoint_id=f"checkpoint_step{step}"
        )
```

**关键**：`checkpoint_future.result()` 确保不堆积多个 checkpoint 请求，避免内存暴涨。

#### 5.3 实践任务

1. 对比同步 `torch.save` 和 `dcp.async_save` 的：
   - checkpoint 保存期间训练 step time 的变化
   - 总训练时间差异（训练 100 step，每 20 step checkpoint 一次）
2. **记录到 `docs/evidence/day14-async-checkpoint.md`**。

### 6. 综合对照实验（第 4 周第 8–10 天）

现在你已经有了多个优化手段。**一次只改一个变量**，做三次对照，保留或回退。

#### 6.1 实验矩阵

| 实验 | 变量 | 测什么 |
|---|---|---|
| A | baseline | step time, MFU, 峰值显存 |
| B | A + ZeRO-1 | 显存变化、step time 变化 |
| C | B + no_sync (accum=4) | 通信时间节省 |
| D | C + 异步 checkpoint | checkpoint 开销消除 |
| E | D + DataLoader 优化 | 数据管线瓶颈消除 |

每次只对比**相邻两行**，归因明确。

#### 6.2 最终报告

`docs/evidence/phase2-report.md` 必须包含：

| 内容 | 要求 |
|---|---|
| **环境** | GPU 型号、数量、PyTorch 版本、NCCL 版本、commit |
| **基线** | step time, samples/s, MFU, 峰值显存 |
| **每项优化的结果** | 改了什么、预期什么、实际什么、归因 |
| **profiler 证据** | 优化前后的 trace 文件路径、时间线截图 |
| **最终结果** | 端到端吞吐提升 X%，MFU 从 X% → Y%，显存下降 Z% |
| **已知限制** | 什么没有测、什么结论不能声称 |

### 7. 面试自测题

1. MFU 和 GPU 利用率的区别？为什么后者可能误导？
2. ZeRO-1/2/3 分别分片什么？各自的通信代价是什么？
3. DDP 的 AllReduce 在 backward 的哪一步启动？桶的大小如何影响重叠效果？
4. 梯度累积时通信次数会不会减少？`no_sync()` 省了多少？
5. Checkpoint 异步写入的 `future.result()` 为什么要调用？
6. Ring AllReduce 的通信量为什么与卡数无关？
7. 你的优化中，哪一项对端到端吞吐贡献最大？为什么？

### 8. 与第 3 阶段的衔接

第 2 阶段产出的 **profiler 能力、ZeRO 配置、异步 checkpoint 模块**，可以直接带入第 3 阶段的平台 MVP：

- `LocalProcessExecutor` 启动 `torchrun` 时，`--amp`、`--checkpoint-every-steps` 参数已经有实验依据。
- 平台需要展示的“性能报告”，就是你在第 2 阶段产出的 `phase2-report.md`。
- 面试时讲“我做的训练平台”时，可以引用“在优化阶段，我通过 X 手段把 MFU 从 A% 提升到 B%”，而不是空谈架构。