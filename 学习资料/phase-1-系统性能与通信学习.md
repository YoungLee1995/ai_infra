# 第 2 阶段补充学习：训练性能诊断与通信

> 阅读目标：能够在一次分布式训练变慢、卡住或失败时，按“训练进程 -> CPU -> GPU/NPU -> 通信 -> 网络”的顺序定位证据，而不是靠猜测调参。
>
> 本文以当前 DDP baseline 为背景。CUDA/NCCL 命令不能直接用于 NPU；NPU 部分应替换为厂商 CANN/Profiler/HCCL 的等价能力，并以当前软件版本文档为准。

## 1. 先建立正确的性能模型

一次 optimizer step 的墙钟时间可近似写为：

```text
T_step = T_input + T_forward + T_backward + T_optim + T_comm - T_overlap + T_wait
```

- `T_input`：读取、解码、tokenize、H2D 拷贝。
- `T_forward/T_backward/T_optim`：GPU/NPU kernel 的真实执行时间。
- `T_comm`：梯度 all-reduce、参数/状态分片通信等。
- `T_overlap`：通信与反向计算重叠掉的部分，不能超过两者较小值。
- `T_wait`：CPU 调度、同步、负载不均、straggler 和资源争用造成的空等。

所以 GPU 利用率低不必然是 GPU 慢，通信时间长也不一定是网络慢。先固定模型、数据、全局 batch、精度、代码提交和硬件，再比较每一步时间、吞吐和显存。优化只能改一个变量，并至少重复三次取中位数。

常用指标：

| 指标 | 含义 | 使用边界 |
| --- | --- | --- |
| samples/s 或 tokens/s | 端到端吞吐 | 最高优先级；必须同时写明全局 batch 和序列长度。 |
| step time | 一次参数更新的耗时 | 要区分 warmup 后稳态与 checkpoint 等偶发事件。 |
| scaling efficiency | `throughput_N / (N * throughput_1)` | 强扩展时通常下降；比较时固定每卡 batch。 |
| 峰值显存 | 参数、梯度、优化器状态、激活和碎片总效果 | 不能只看 `nvidia-smi` 的瞬时数值。 |
| communication share | profiler 中通信相关时间/step time | 注意重叠后应看关键路径，不应把所有 stream 时间简单相加。 |
| MFU | 实际模型 FLOPs/s 与硬件峰值 FLOPs/s 的比值 | 需要模型 FLOPs 估算和匹配精度的峰值规格；小 MLP 不适合推导大模型 MFU。 |

## 2. Linux：进程、CPU 与 IO 的证据链

训练任务首先是 Linux 进程。`torchrun` 启动 launcher，再启动每个 rank 的 Python 进程；一个 rank 通常独占一张卡。排障前记下 PID、rank、GPU/NPU、CPU 核、NUMA 节点和日志文件。

```bash
ps -eo pid,ppid,psr,pcpu,pmem,stat,etime,args --forest
top -H -p <pid>
pidstat -dru -p <pid> 1
numactl --hardware
```

- `STAT=D` 表示不可中断 IO 等待；`R` 很多且 CPU 饱和才是 CPU 算力不足的强证据。
- `top -H` 看线程，而不是只看 Python 进程总数。DataLoader worker、通信线程和 Python 主线程的状态不同。
- `pidstat -d` 中持续高读写或较高 `iodelay`，配合 GPU 空闲，优先怀疑数据输入或 checkpoint。
- 多 socket 主机上，GPU、NIC 和 CPU 内存跨 NUMA 访问会增加延迟。用 `nvidia-smi topo -m`（或 NPU 拓扑工具）确认设备亲和性。

### perf：回答“CPU 时间花在哪里”

`perf` 采样 CPU，而不是 GPU kernel。它适合发现 tokenize、Python 调用、内核调度、锁竞争和频繁系统调用；若 GPU 已满，perf 不是第一个工具。

```bash
# 总体 CPU/调度/上下文切换概览，运行训练的同时执行
perf stat -p <pid> -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions -- sleep 30

# 采样调用栈；需要符号与权限，先做短时间采集
sudo perf record -F 99 -g -p <pid> -- sleep 30
sudo perf report
```

重点解读：

- `context-switches` 或 `cpu-migrations` 异常高：线程过多、CPU 争用或绑核差；检查 DataLoader `num_workers`、OMP/MKL 线程数。
- IPC（`instructions/cycles`）很低：可能是缓存未命中、分支、等待或解释器开销，不能单凭它断言原因。
- 大量 `futex`、`pthread_mutex`、`sched_yield`：线程锁竞争或等待；要再看调用栈确认是谁在等谁。
- 大量缺页：首次加载、内存压力或 mmap 数据集行为；区分 minor/major page fault。

权限受限时，先用 `perf stat` 或联系管理员。不要为了采样永久降低 `kernel.perf_event_paranoid`；把改动、范围和恢复方式记录下来。

### 火焰图：把采样结果变成可读图

火焰图横向宽度代表样本数/时间，纵向代表调用栈深度；它不是“时间线”。顶部的函数宽，说明该调用路径出现得多；颜色通常没有性能语义。搜索 `DataLoader`、`tokenizer`、`futex`、`all_reduce` 或业务函数名，先找到最宽的叶子和其调用者。

如果安装了 FlameGraph 工具：

```bash
sudo perf script > out.perf
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded
./FlameGraph/flamegraph.pl out.folded > cpu-flamegraph.svg
```

火焰图适合 CPU 热点，不适合证明 GPU kernel 或 NCCL 的时间占比。`perf record` 只采集 20--60 秒的稳态窗口，采集前避开下载、编译、首次 CUDA context 初始化和 checkpoint。

## 3. gdb：把 hang 和 native 崩溃变成栈证据

Python 报错优先保留 traceback；gdb 用于 Python 无输出卡住、C/CUDA 扩展段错误或 native 库死锁。生产任务先保留日志、PID 和 core dump 策略，避免直接杀掉唯一现场。

```bash
# 附加到疑似卡住的某个 rank，观察所有线程
gdb -p <pid>
(gdb) set pagination off
(gdb) info threads
(gdb) thread apply all bt
(gdb) detach
(gdb) quit

# 新启动调试。torchrun 场景先缩到单 rank 或直接调试子进程。
gdb --args python -m ddp_baseline.train --device cpu --epochs 2
```

判断原则：

- 所有 rank 都停在通信库等待：检查 collective 顺序、网络和某个 rank 是否更早出错。
- 一个 rank 在 DataLoader/IO，其他 rank 在 all-reduce：根因通常是该 rank 的慢输入或异常，而非 all-reduce 本身。
- 一个 rank OOM 或 Python 异常后退出，其他 rank 仍等通信：首先找最早退出 rank 的 stderr。
- 栈内缺少符号并不等于没有信息。记录共享库版本、地址和所有线程 backtrace，使用匹配 debuginfo 的环境再分析。

DDP hang 的最低成本证据组合是：各 rank 的完整日志、`TORCH_DISTRIBUTED_DEBUG=DETAIL`、超时配置、每个 rank 的 PID/栈、GPU/NPU 状态和网络错误。不要只截取“卡住的那一行”。

## 4. CUDA/NPU profiler：看时间线而非猜 kernel

先用 PyTorch Profiler 形成可重复的第一层证据，再在必要时用 Nsight Systems/Compute 或 NPU 厂商 profiler 下钻。每次采集只覆盖少量稳态 step，profiler 本身会引入开销。

### PyTorch Profiler

核心是 CPU 线程、CUDA/NPU activity、operator 统计、memory 和 trace。推荐 schedule：warmup 若干 step，采集 5--20 个 step，导出 Chrome trace 或 TensorBoard。

应从 trace 回答四个问题：

1. GPU/NPU 时间线上是否存在明显空洞？空洞前的 CPU、DataLoader、同步事件是什么？
2. 反向阶段的通信 collectives 出现在何处，是否同计算 stream 重叠？
3. 最长 kernel/算子是什么，调用次数是否异常？
4. H2D、allocator、checkpoint 或同步是否落在关键路径？

不要把 profiler 的 CPU self time 与 GPU device time相加成“总耗时”；二者可并发。以 step 的 wall-clock 区间和关键路径为准。

### CUDA：Nsight 的分工

- **Nsight Systems (`nsys`)**：先用它看全局时间线，包含 CPU、CUDA API、kernel、NCCL、memcpy 和 stream；用于发现空洞与缺少重叠。
- **Nsight Compute (`ncu`)**：只对已确认的热点 kernel 做微观分析，如 Tensor Core 利用、内存吞吐、occupancy 和访存瓶颈；不要直接 profile 整个训练。

示例仅用于短小复现实验：

```bash
nsys profile --trace=cuda,nvtx,osrt -o reports/ddp-step \
  torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 1 --samples 2048 --batch-size 64
```

多 rank 会产生多个进程/报告。必须把 rank 与 GPU 对上，并确认采集窗口包含稳态反向。不要在共享机器上默认开启高侵入指标采集。

### NPU：同样的问题，不同工具链

NPU 的 CANN/Ascend Profiler 等工具名称、环境变量和 trace 格式随版本变化。执行前查对应版本官方文档，确认：运行模式、算子 trace、HCCL 事件、CPU 事件、内存与采样开销。分析框架仍相同：先找到 step 关键路径，再看算子、数据搬运和 HCCL 是否重叠；不要把 CUDA/Nsight 的结论照搬到 NPU。

## 5. TCP/IP 与 RDMA：通信发生在什么链路上

NCCL/HCCL 是集合通信库，不等于 TCP。单机优先走 PCIe、NVLink/NVSwitch 或 NPU 内部互连；跨机可能走 TCP sockets、RoCE/RDMA 或 InfiniBand。实际 transport 由硬件、驱动、拓扑、环境变量和通信库探测共同决定。

最小网络心智模型：

```text
rank -> NIC -> 交换机/网络 -> 对端 NIC -> rank
        TCP: 内核协议栈、可靠字节流、拥塞控制
        RDMA: NIC 直接读写已注册内存，减少 CPU/内核参与，但仍依赖网络无损与配置
```

- **TCP**：面向连接、可靠有序的字节流；没有消息边界。连接建立使用三次握手，关闭常见 FIN/ACK；丢包由重传恢复，拥塞控制会降低发送窗口。
- **端口**：四元组 `源 IP/源端口/目的 IP/目的端口` 标识连接。`MASTER_ADDR`/`MASTER_PORT` 是 rendezvous，并不代表后续所有 NCCL/HCCL 数据都只走这一个端口。
- **带宽与延迟**：小消息常由延迟主导，大消息受有效带宽主导。通信时间可粗略理解为 `latency + bytes / bandwidth`，集合通信还要乘算法和拓扑带来的阶段数。
- **RDMA/RoCE**：降低 CPU copy 和内核切换，但不是“自动更快”。MTU、PFC/ECN、网卡/驱动、GID、路由、NUMA 和拥塞都会影响稳定性。

只在获授权的测试网络执行连通/带宽测试。排查顺序是接口与路由、DNS、端口/防火墙、丢包和重传、NIC 计数器，再看通信库日志：

```bash
ip -br addr
ip route
ss -tanp
ethtool -S <nic>
ping -c 20 <peer-ip>
```

`ping` 成功只能证明 ICMP 可达，不能证明训练需要的 TCP/RDMA、端口、MTU 和带宽都正常。不要将 `ping` 延迟直接当作 all-reduce 延迟。

## 6. NCCL/HCCL：从 DDP 梯度到 collective

DDP 在反向传播期间把梯度按 bucket 聚合，并对每个 bucket 发起 all-reduce。每个 rank 最终得到相同的平均梯度，然后各自执行相同的 optimizer step。因此 collective 有三个硬条件：参与者集合一致、调用顺序一致、每次张量形状/类型等语义一致。

常见操作：

| collective | 结果 | 训练中的典型用途 |
| --- | --- | --- |
| all-reduce | 每个 rank 都拿到规约结果 | DDP 同步梯度。 |
| all-gather | 每个 rank 都拿到全部分片 | FSDP 参数/激活收集。 |
| reduce-scatter | 规约后每个 rank 拿一片 | ZeRO/FSDP 梯度分片。 |
| broadcast | 一个 rank 的数据发给全部 rank | 初始参数、控制信息。 |
| all-to-all | 每个 rank 与全部 rank 交换不同分片 | MoE token dispatch。 |

ring all-reduce 通常带宽利用率高，但每个 rank 经历多个邻居阶段；tree 算法可能对小消息或特定拓扑更有利。NCCL/HCCL 会根据拓扑和消息规模选择算法/协议，手工强设环境变量应是有 profile 证据后的对照实验，不是默认调优手段。

**计算通信重叠**依赖反向传播的梯度就绪顺序、bucket 大小、独立 stream、足够的后续计算以及无阻塞同步。更早发起通信不必然更快：bucket 太小会增加 launch/协议开销，太大则等到更多反向计算结束才通信。测量时看关键路径是否缩短，而不是只看通信事件变多。

### NCCL/HCCL 排障最小流程

1. 缩小问题：单卡 -> 单机多卡 -> 两机；固定模型与 batch。
2. 确认每 rank 绑定正确设备且进程数不超过可见设备数。
3. 收集各 rank 的最早错误、退出码和时间戳；最先报错者通常最接近根因。
4. 开启有限范围的通信日志，例如 CUDA 环境中的 `NCCL_DEBUG=INFO`；敏感网络信息不要上传公开仓库。
5. 核对 NIC 选择、路由、容器网络、主机名解析和跨机端口策略；NPU 则使用厂商 HCCL 日志与健康检查。
6. 以独立通信 benchmark 验证链路，再回到真实模型验证端到端收益。

不要用 `NCCL_IB_DISABLE=1`、禁用 P2P 或强设接口来“解决”问题后就结束。它们可作为隔离变量的诊断实验，但会改变 transport，必须记录并恢复。

## 7. 一次可复现的诊断实验

在 GPU 服务器上选择一个固定 Transformer 或当前 baseline 的放大任务，创建独立实验目录。先不改任何性能参数：

```text
实验名：baseline-<date>
固定项：代码提交、模型、数据、每卡 batch、world size、精度、seed、设备/驱动
采集：30 个 warmup step 后的 100 个 step wall time、samples/s、峰值显存、trace、CPU 采样
```

之后只做三组实验：

1. `world_size=1` 与 `world_size=2`，固定每卡 batch，计算吞吐与扩展效率。
2. AMP 开关或 BF16/FP16 策略对照，检查数值稳定、吞吐和显存。
3. 梯度累积/`no_sync` 对照，固定有效 batch，观察通信次数、step time 和收敛语义。

每组输出一页结论：现象、trace 中的证据、假设、唯一改动、数值结果、回归风险和是否保留。没有 profiler 或日志证据时，结论应写“待验证”，而不是“网络瓶颈”。

## 8. 面试时应能说清的边界

- `perf`/火焰图解释 CPU 调用栈；Nsight/厂商 profiler 解释加速器时间线，二者互补。
- DDP hang 常是 collective 语义不一致或某 rank 先失败，通信库只是首先暴露等待的位置。
- GPU utilization 是采样指标，不是端到端性能结论；以稳定吞吐和 step time 为主。
- TCP 可达不等于 NCCL/HCCL 可用；跨机还受接口、端口、RDMA、MTU、拓扑和容器网络影响。
- NCCL/HCCL 是通信实现，数据并行/张量并行/FSDP 是训练并行策略；不要混为一谈。

完成本文后，下一步是把一个真实 profiling 报告作为第 2 阶段证据包：命令、环境、trace 截图、指标表和一次被数据推翻或验证的优化假设。


## 9. 当前项目的 CPU 实操：从单进程到 Gloo DDP、checkpoint 与诊断边界

本章只使用仓库中真实存在的 `ddp_baseline/train.py` 和 `tests/test_train.py`。它训练的是确定性的合成二分类数据集和一个小型 MLP；它的目标是验证训练控制流，而不是测量 GPU 性能。不要运行本章以外的 `ddp_baseline.bench`、`ddp_baseline.data` 或假想的 Transformer 命令，它们不在当前项目中。

### 9.1 这次实操要验证什么，不能验证什么

本机 CPU 上能够验证：单进程训练、`torchrun` 启动两个 rank、Gloo collective、`DistributedSampler` 数据切分、DDP 梯度同步路径、梯度累积中的 `no_sync`、rank 0 独占写入、以及 checkpoint 的状态保存和恢复。

本机 CPU 上不能验证：CUDA kernel、CUDA AMP、显存、H2D、NCCL/RDMA/网络拓扑和计算通信重叠。因此本章的 `samples_per_second` 只用于确认任务完成，不能作为 GPU 基线，更不能用于推断 NCCL 或网络性能。

本项目的对应关系如下：

| 训练概念 | 当前实现 |
| --- | --- |
| 单进程设备 | `--device cpu`，不创建进程组 |
| CPU DDP 后端 | `torchrun` 设置 `WORLD_SIZE` 后，`configure_process_group()` 初始化 Gloo |
| 数据分片 | `DistributedSampler(..., drop_last=True)` |
| 梯度同步 | `DistributedDataParallel` 在同步的 `backward()` 中执行 |
| 梯度累积 | 中间 micro-batch 使用 `model.no_sync()`，窗口末尾才同步和 `optimizer.step()` |
| checkpoint | 模型、优化器、scheduler、scaler、每个 rank 的 RNG、epoch/batch/step 位置和配置 |

### 9.2 Step 0：进入项目并确认 CPU 环境

从仓库根目录开始。必须用 `.venv/bin/python`，因为系统 `python` 在本环境中不存在。

```bash
cd /workspace/GIT/ai_infra
.venv/bin/python --version
.venv/bin/python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('distributed=', torch.distributed.is_available()); print('gloo=', torch.distributed.is_gloo_available())"
```

验收条件：`cuda=False`、`distributed=True`、`gloo=True`。如果 Gloo 为 `False`，停止后续两进程实验；重新安装带 distributed 支持的 PyTorch，而不是尝试 CUDA/NCCL 命令。

### 9.3 Step 1：先运行组件测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

验收条件：两个测试都显示 `ok`，并以 `OK` 结束。它们覆盖：

1. `SyntheticBinaryDataset` 在相同 seed 下生成相同数据；
2. checkpoint 往返后模型参数、优化器和训练位置可恢复。

若失败，先修复测试失败，再进行 DDP 实验；否则后面的训练完成并不能证明 checkpoint 语义正确。

### 9.4 Step 2：运行单进程 CPU 基线

下面的 batch size 是单进程的 micro-batch。`512 / 32 = 16` 个 batch，每 2 个 batch 做一次优化器更新，因此每个 epoch 有 8 个 `global_step`；两 epoch 后应为 16。输出目录使用本章专用路径，避免覆盖已有实验。

```bash
.venv/bin/python -m ddp_baseline.train \
  --device cpu --epochs 2 --samples 512 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 4 \
  --checkpoint-dir runs/ch9-cpu-20260928/single
```

验收条件：打印两行 JSON，`world_size` 为 1；目录中至少有 `config.json`、`metrics.jsonl`、`checkpoint.pt` 和按 step 编号的 checkpoint。查看结果：

```bash
cat runs/ch9-cpu-20260928/single/metrics.jsonl
```

这里的损失应随训练下降，但小型 CPU 实验的吞吐会受宿主机负载影响，不把它作为性能比较结论。

### 9.5 Step 3：运行两个 CPU rank 的 Gloo DDP

`torchrun` 负责注入 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 和 rendezvous 地址。CPU 不传 `--amp`，因为本项目只在 CUDA 下启用 float16 autocast。每个 rank 的 DataLoader 收到数据的一半：每 rank 有 `512 / 2 / 32 = 8` 个 batch；累积 2 次后每 epoch 有 4 次更新，两 epoch 后 `global_step` 为 8。

```bash
torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cpu --epochs 2 --samples 512 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 4 \
  --checkpoint-dir runs/ch9-cpu-20260928/ddp
```

验收条件：命令以退出码 0 结束；rank 0 打印两行 JSON，且 `world_size` 为 2。`torchrun` 可能提示默认设置 `OMP_NUM_THREADS=1`，这是一条防止两个进程过度占用 CPU 的提示，不是失败。

核对 checkpoint 中的真实状态。这里不应期待单进程和两进程的 `global_step` 相同：固定全局样本数时，DDP 每个 optimizer step 消费更多样本，因此 DDP 的 step 数更少。

```bash
.venv/bin/python -c "import torch; from pathlib import Path
for name in ('single', 'ddp'):
 p = Path('runs/ch9-cpu-20260928') / name / 'checkpoint.pt'
 s = torch.load(p, map_location='cpu', weights_only=False)
 print(name, {'epoch': s['epoch'], 'batch_in_epoch': s['batch_in_epoch'], 'global_step': s['global_step'], 'samples_seen': s['samples_seen'], 'rng_states': len(s['rng_states']), 'world_size': s['config']['world_size']})"
```

预期关系：`single` 为 `global_step=16`、`rng_states=1`；`ddp` 为 `global_step=8`、`rng_states=2`；二者的 `samples_seen` 均为 1024。这说明 DDP checkpoint 收集并保存了两个 rank 各自的 RNG 状态，而非只保存 rank 0。

### 9.6 Step 4：在 epoch 边界验证断点续训

先完成一个 epoch，得到可恢复 checkpoint；再以完全相同的训练语义恢复到第二个 epoch。注意 `--checkpoint-dir` 不是恢复来源，`--resume` 才是；恢复后输出写到新的目录，便于检查。

```bash
.venv/bin/python -m ddp_baseline.train \
  --device cpu --epochs 1 --samples 256 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 4 \
  --checkpoint-dir runs/ch9-cpu-20260928/resume-source

.venv/bin/python -m ddp_baseline.train \
  --device cpu --epochs 2 --samples 256 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 4 \
  --checkpoint-dir runs/ch9-cpu-20260928/resumed \
  --resume runs/ch9-cpu-20260928/resume-source/checkpoint.pt
```

验收条件：第二条命令先打印 `resumed from epoch 1, batch 0, step 4`，再打印 epoch 2 的指标。恢复命令的设备、样本数、模型维度、batch、累积步数、学习率、seed、AMP、warmup 和 world size 必须与保存时一致；实现会拒绝不一致的配置，避免把两个不同实验错误拼接。

### 9.7 Step 5：CPU 下如何做进程与卡住诊断

先在一个终端启动较长任务，再在另一个终端查进程。示例的输出目录可自行改为新的、未使用的目录。

```bash
torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cpu --epochs 100 --samples 4096 --batch-size 32 \
  --checkpoint-dir runs/ch9-cpu-20260928/observe

ps -eo pid,ppid,psr,pcpu,pmem,stat,etime,args --forest | rg 'torchrun|ddp_baseline'
top -H -p <rank-pid>
```

如果任务看似卡住，不要先杀进程。先保存两个 rank 的 stderr，再对每个 rank 收集栈：

```bash
gdb -p <rank-pid>
(gdb) set pagination off
(gdb) thread apply all bt
(gdb) detach
(gdb) quit
```

判断时看“哪个 rank 最先异常”：一个 rank 在 Python/DataLoader、另一个在 Gloo collective 等待，通常应先调查慢 rank；所有 rank 都在 collective 等待时，检查 collective 顺序和最早 stderr。CPU 的 Gloo 栈不能用来判断 NCCL、RDMA 或 GPU kernel。

本机没有安装 `pidstat` 与 `perf`，因此本次没有生成伪造的 CPU 使用率、火焰图或 perf 数字。若目标机器安装了 `sysstat` 和 perf，可在训练运行期间追加：

```bash
pidstat -dru -p <rank-pid> 1 10
perf stat -p <rank-pid> -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions -- sleep 30
```

将 PID、时间窗口、完整输出和当时的训练命令一同保存；不要把一次 CPU 采样解释为 GPU 或网络瓶颈。

### 9.8 本次 CPU 执行记录（2026-09-28）

以下不是示例数据，而是在本仓库、当前 CPU 环境实际执行的记录。数值中的吞吐受共享 CPU 负载影响，只作为任务完成证据。

| 项目 | 实际结果 |
| --- | --- |
| Python | `.venv/bin/python`，Python 3.10.12 |
| PyTorch | 2.14.0+cu130 |
| CUDA | `False` |
| distributed / Gloo | `True` / `True` |
| 组件测试 | 2 passed：dataset 可复现、checkpoint 往返恢复 |
| 单进程训练 | epoch 1：loss 0.686563，11895.22 samples/s；epoch 2：loss 0.665828，21634.23 samples/s |
| 两进程 Gloo DDP | epoch 1：loss 0.690397，13966.15 samples/s；epoch 2：loss 0.676618，46621.69 samples/s |
| 恢复来源 | epoch 1 完成，step 4，samples_seen 256 |
| 恢复后训练 | 从 `epoch 1, batch 0, step 4` 恢复，完成 epoch 2：loss 0.680191，8291.43 samples/s |
| 单进程 checkpoint | epoch 2，step 16，samples_seen 1024，1 个 RNG state |
| 两进程 checkpoint | epoch 2，step 8，samples_seen 1024，2 个 RNG states |
| 诊断工具限制 | `perf`、`pidstat` 未安装；未执行 GPU profiler、NCCL 或网络测试 |

实际生成物位于 `runs/ch9-cpu-20260928/`：`single/`、`ddp/`、`resume-source/` 与 `resumed/` 各包含 config、metrics 和 checkpoint，可用于复查本章记录。

### 9.9 GPU 完整实操：CUDA、AMP、NCCL DDP、恢复与诊断

本节必须在一台至少有两张可见 NVIDIA GPU 的机器上执行。本次 CPU 环境没有 GPU，以下命令尚未在本仓库执行；执行后应将真实输出补到本节末尾的证据表，不能复用第 9.8 节的 CPU/Gloo 结果。

本项目的 GPU 路径是固定的：`--device cuda` 使每个 `torchrun` rank 按 `LOCAL_RANK` 绑定一张 GPU，使用 NCCL 初始化进程组；`--amp` 才启用 CUDA float16 autocast 和 `GradScaler`。若 CUDA 不可用，程序会报错，不会回退到 CPU。

#### 9.9.1 Step 0：GPU 环境预检

进入仓库和 GPU 的 Python 环境。以下检查全部通过才能继续：`nvidia-smi` 能显示设备，PyTorch 可见至少两张卡，且 NCCL 可用。

```bash
cd /workspace/GIT/ai_infra
.venv/bin/python -m unittest discover -s tests -v
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
.venv/bin/python -c "import torch; import torch.distributed as dist; print({'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count(), 'nccl_available': dist.is_nccl_available(), 'nccl_version': torch.cuda.nccl.version() if dist.is_nccl_available() else None})"
```

验收条件：组件测试通过；`cuda_available=True`、`device_count >= 2`、`nccl_available=True`。记录 GPU 型号、显存、驱动、`torch`、CUDA build、NCCL 版本和 `nvidia-smi topo -m` 输出。若只有一张卡，完成 9.9.2 的单卡流程，不执行 9.9.3 的双卡 DDP。

#### 9.9.2 Step 1：单卡 CUDA + AMP 基线

先跑单卡，隔离模型、数据、CUDA AMP 与 checkpoint；这一步成功不代表 NCCL 已验证。`--nproc_per_node=1` 仍使用 `torchrun`，但不会创建多 rank 的进程组。

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 -m ddp_baseline.train \
  --device cuda --amp --epochs 2 --samples 512 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 4 \
  --checkpoint-dir runs/ch9-gpu-<date>/single

cat runs/ch9-gpu-<date>/single/config.json
cat runs/ch9-gpu-<date>/single/metrics.jsonl
```

验收条件：退出码为 0，输出两行 `world_size: 1` 的指标；`config.json` 中 `device` 为 `cuda`、`amp` 为 `true`、`world_size` 为 1。目录中存在 `checkpoint.pt` 和带 step 编号的 checkpoint。单卡每个 epoch 有 `512 / 32 / 2 = 8` 次 optimizer step；两 epoch 后 checkpoint 的 `global_step` 应为 16，`samples_seen` 应为 1024。

在另一个终端、训练仍在运行时观察一次显存和利用率。短小 MLP 可能只使用很少显存且利用率波动很大，这不是故障。

```bash
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 1
```

#### 9.9.3 Step 2：两卡 NCCL DDP + AMP

确认 0、1 两张卡未被其他任务占用后执行。`--batch-size 32` 是**每 rank** micro-batch；两个 rank 的每次同步 micro-batch 是 `2 * 32 = 64` 个样本，累积两次后的有效全局 batch 是 `128`。它与单卡流程的有效 batch 64 不同，不能直接比较 loss 曲线或吞吐来判断“DDP 加速”。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 2 --samples 512 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 4 \
  --checkpoint-dir runs/ch9-gpu-<date>/ddp
```

验收条件：退出码为 0；rank 0 输出两个 `world_size: 2` 的指标；只有 rank 0 写入 `config.json`、`metrics.jsonl` 和 checkpoint。每个 rank 每 epoch处理 8 个 batch，累积两次后有 4 个 optimizer step；两 epoch后 checkpoint 的 `global_step` 应为 8、`samples_seen` 应为 1024、`rng_states` 长度应为 2。

用下面命令读取 checkpoint，而不是凭终端输出猜测：

```bash
.venv/bin/python -c "import torch; from pathlib import Path
for name in ('single', 'ddp'):
 p = Path('runs/ch9-gpu-<date>') / name / 'checkpoint.pt'
 s = torch.load(p, map_location='cpu', weights_only=False)
 print(name, {'device': s['config']['device'], 'amp': s['config']['amp'], 'world_size': s['config']['world_size'], 'epoch': s['epoch'], 'global_step': s['global_step'], 'samples_seen': s['samples_seen'], 'rng_states': len(s['rng_states'])})"
```

不要比较单卡和双卡 checkpoint 的参数是否完全相等：两种运行的每次更新样本组成和更新次数不同。这里要验证的是每个运行内部的配置、step 和 rank RNG 状态一致。

#### 9.9.4 Step 3：在 CUDA + AMP 下验证断点续训

恢复必须使用与来源相同的 device、AMP、world size、模型、数据、batch、累积步数、学习率、seed 和 warmup；当前实现会显式拒绝不一致的配置。先完成一 epoch，再恢复到第二 epoch：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 1 --samples 256 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 2 \
  --checkpoint-dir runs/ch9-gpu-<date>/resume-source

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 2 --samples 256 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 2 \
  --checkpoint-dir runs/ch9-gpu-<date>/resumed \
  --resume runs/ch9-gpu-<date>/resume-source/checkpoint.pt
```

验收条件：恢复命令打印 `resumed from epoch 1, batch 0, step 2`，并完成 epoch 2；`resumed/checkpoint.pt` 的 `epoch=2`、`global_step=4`、`samples_seen=512`、`rng_states=2`。这同时验证模型、优化器、scheduler、CUDA AMP scaler 和两个 rank 的 RNG 恢复路径。

#### 9.9.5 Step 4：做可比较的单卡与双卡吞吐实验

性能实验必须固定**有效全局 batch**，并分别重复至少三次。为使两种运行都是有效全局 batch 128，可用：单卡 `batch-size=32, accumulation-steps=4`；双卡 `batch-size=32, accumulation-steps=2`。合成 MLP 太小，得到的数字只练习测量流程，不代表真实模型扩展效率。

```bash
# 单卡：有效全局 batch = 1 * 32 * 4 = 128。将末尾目录依次改为 perf-single-1、perf-single-2、perf-single-3，完整执行三次。
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f 'elapsed=%e s' torchrun --standalone --nproc_per_node=1 -m ddp_baseline.train \
  --device cuda --amp --epochs 10 --samples 8192 --batch-size 32 --accumulation-steps 4 \
  --checkpoint-every-steps 1000 --checkpoint-dir runs/ch9-gpu-<date>/perf-single-1

# 双卡：有效全局 batch = 2 * 32 * 2 = 128。将末尾目录依次改为 perf-ddp-1、perf-ddp-2、perf-ddp-3，完整执行三次。
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -f 'elapsed=%e s' torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 10 --samples 8192 --batch-size 32 --accumulation-steps 2 \
  --checkpoint-every-steps 1000 --checkpoint-dir runs/ch9-gpu-<date>/perf-ddp-1
```

把每次终端的 `elapsed` 和每个目录最后一行 `metrics.jsonl` 记录为表格，取中位数。强扩展效率公式为 `throughput_2 / (2 * throughput_1)`。不要跨实验目录拼接指标；每次命令都使用新目录，避免旧 metrics 文件污染结果。

#### 9.9.6 Step 5：采集 GPU、CPU 与 NCCL 的诊断证据

先运行一次短小的双卡任务，并把标准输出和标准错误完整保留。`NCCL_DEBUG` 日志可能包含接口、主机和拓扑信息，不要提交到公开仓库。

```bash
mkdir -p reports/ch9-gpu-<date>
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET \
  torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 2 --samples 512 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 4 \
  --checkpoint-dir runs/ch9-gpu-<date>/nccl-log \
  2>&1 | tee reports/ch9-gpu-<date>/nccl-ddp.log

nvidia-smi topo -m | tee reports/ch9-gpu-<date>/topology.txt
```

验收条件不是“日志中一定出现 Ring”或某个固定网卡名，而是：所有 rank 正常退出；日志没有 `unhandled system error`、`connection refused`、`Failed to initialize` 或 `Watchdog caught collective operation timeout`；记录 NCCL 最终选用的 transport/拓扑信息。NCCL 会随硬件和版本选择不同算法。

若任务卡住，保留日志，再取得 launcher 和每个 rank 的 PID：

```bash
ps -eo pid,ppid,psr,pcpu,pmem,stat,etime,args --forest | rg 'torchrun|ddp_baseline'
top -H -p <rank-pid>
gdb -p <rank-pid>
(gdb) set pagination off
(gdb) thread apply all bt
(gdb) detach
(gdb) quit
```

排查顺序：先找最早退出或报错的 rank；再看其他 rank 是否在 `ProcessGroupNCCL` 等待；最后核对同一训练循环中是否有条件分支导致 collective 次数或张量形状不一致。不要先设置 `NCCL_IB_DISABLE`、`NCCL_P2P_DISABLE` 或强设网卡作为“修复”；它们只能作为单独记录的隔离实验。

#### 9.9.7 Step 6：用 profiler 看 GPU 时间线

当前 `ddp_baseline.train` **没有** `--profile` 参数，也没有调用 `torch.profiler`，所以不能假装运行一个不存在的 profiler CLI。可先用 Nsight Systems 对一个短任务采集 CUDA API、kernel、memcpy 和 NCCL 时间线；确认本机安装 `nsys` 后执行：

```bash
nsys --version
mkdir -p reports/ch9-gpu-<date>
CUDA_VISIBLE_DEVICES=0,1 nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --force-overwrite true -o reports/ch9-gpu-<date>/ddp-timeline \
  torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 2 --samples 2048 --batch-size 32 \
  --accumulation-steps 2 --checkpoint-every-steps 1000 \
  --checkpoint-dir runs/ch9-gpu-<date>/nsys
```

如果 Nsight 版本只跟踪 launcher 而没有跟踪 torchrun 子进程，使用该版本文档规定的 child-process tracing 选项，或直接 profile 某个 rank；不要把一个没有 CUDA/NCCL event 的 report 当作训练时间线。打开报告后，逐项回答：GPU 是否有空洞；空洞前 CPU/加载/同步事件是什么；backward 的 NCCL collective 是否与后续 kernel 重叠；最长 kernel 和 memcpy 是否在关键路径。小 MLP 的时间线极短，主要用于熟悉工具，不足以推导真实大模型的 MFU 或扩展效率。

若需要可重复的 PyTorch trace，下一步是在 `train.py` 增加显式 profiler schedule、`prof.step()` 和 `tensorboard_trace_handler`；这是代码改动，不能用本章现有命令替代。

#### 9.9.8 GPU 证据记录模板

每次 GPU 实验建立一个新日期目录，并将下表填为真实值：

| 项目 | 必填记录 |
| --- | --- |
| 环境 | 主机、Git commit、GPU/显存/驱动、PyTorch、CUDA、NCCL、`CUDA_VISIBLE_DEVICES` |
| 命令 | 完整 `torchrun` 命令、环境变量、开始/结束时间、退出码 |
| 训练语义 | world size、每卡 batch、累积步数、有效全局 batch、AMP、seed、samples、epoch |
| 正确性 | `metrics.jsonl`、checkpoint 的 epoch/step/samples/RNG state 数、恢复结果 |
| 性能 | 每次 samples/s、墙钟时间、中位数、单卡与双卡强扩展效率 |
| 拓扑与通信 | `nvidia-smi topo -m`、NCCL 日志路径、是否存在错误/超时 |
| profiler | trace/report 路径、GPU 空洞、collective 重叠、关键路径结论 |
| 结论 | 现象、证据、唯一改动、结果、回归风险；无证据时写“待验证” |
