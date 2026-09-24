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
