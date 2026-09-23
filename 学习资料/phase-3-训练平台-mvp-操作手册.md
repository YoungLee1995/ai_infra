# 训练作业平台 MVP：裸机 GPU 操作手册

> 配套计划：[phase-3-训练平台-mvp-practice.md](phase-3-训练平台-mvp-practice.md)。本版本适合当前环境：101 开发控制面与 CPU 测试，200 直接运行真实 GPU 训练。重心是 DDP、checkpoint、故障恢复、性能分析和算子/图优化，不包含集群搭建或操作。

## 0. 完成定义与边界

最终应能演示：提交两个竞争 GPU 的训练任务，显示排队原因；启动多卡 DDP；查看日志、指标和 profiler 证据；终止 worker 后从 checkpoint 重试；取消另一个任务。

| 层 | 第一版实现 |
| --- | --- |
| 控制面 | python3 + FastAPI + Typer CLI |
| 状态库 | SQLite，单 controller |
| 执行器 | 受控启动、观察、终止 200 上的 `torchrun` 进程组 |
| 训练负载 | 当前 `ddp_baseline`，后续替换为真实模型 |
| checkpoint | 200 的每作业独立目录 |
| 观测 | JSON 日志、JSONL、Prometheus metrics、PyTorch Profiler |

范围外：集群安装、容器编排、跨节点调度、复杂抢占和 Web 控制台。没有第二台可用 GPU 主机时，不对多节点训练做模拟结论。

## 1. 101 与 200 分工（第 1 天）

| 机器 | 要做的事 | 不要声称的事 |
| --- | --- | --- |
| 101 CPU | API、状态机、队列、重试、FakeExecutor、单元测试 | GPU 性能或通信结论 |
| 200 GPU | CUDA、DDP/NCCL、checkpoint、进程恢复、profile | 多节点扩展效率 |

在 101 先通过当前训练基线：

```bash
cd /workspace/GIT/ai_infra
. .venv/bin/activate
python3 -m pytest tests -v
python3 -m ddp_baseline.train --device cpu --epochs 1 --samples 512 \
  --batch-size 16 --checkpoint-every-steps 5 --checkpoint-dir runs/platform-smoke
test -f runs/platform-smoke/checkpoint.pt && tail -n 1 runs/platform-smoke/metrics.jsonl
```

在 200 记录环境并确认 GPU 可用：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

将日期、GPU 型号、驱动、PyTorch 版本和 Git commit 记录到 `docs/evidence/environment.md`。完成判据：101 测试通过，200 能看到目标 GPU。

## 2. 建立真实 GPU 基线（第 1--2 天）

从 101 同步代码到 200 的个人目录，保持 Git commit 一致。每次实验使用唯一输出目录，避免覆盖其他实验。

```bash
cd <200上的项目目录>
. .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 -m ddp_baseline.train \
  --device cuda --amp --epochs 3 --samples 4096 --batch-size 64 \
  --checkpoint-every-steps 20 --checkpoint-dir runs/gpu-single

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m ddp_baseline.train --device cuda --amp --epochs 3 --samples 4096 \
  --batch-size 64 --checkpoint-every-steps 20 --checkpoint-dir runs/gpu-ddp2
```

`--nproc_per_node` 不得超过可见 GPU 数量。另开终端观察：

```bash
nvidia-smi -l 1
tail -f runs/gpu-ddp2/metrics.jsonl
```

完成判据：单卡与多卡均完成，rank 0 写出 checkpoint，日志无 NCCL 错误。记录命令、每卡 batch、world size、AMP、wall time、samples/s、显存和 commit；不要只记录 GPU 利用率。

## 3. 创建控制面项目与执行器接口（第 3 天）

平台控制面独立于 `ddp_baseline`：

```text
training-platform-mvp/
  api/                 # HTTP 路由、CLI、spec 校验
  controller/          # queue、状态机、reconciler
  executor/            # FakeExecutor、LocalProcessExecutor、远程适配器
  models/              # SQLite 表和领域对象
  examples/            # 任务 YAML
  tests/               # unit、integration、failure scenarios
  docs/                # 架构、runbook、故障复盘、证据
  pyproject.toml
  Makefile
```

固定依赖版本，至少加入 `fastapi`、`uvicorn`、`typer`、`pydantic`、`sqlalchemy`、`alembic`、`prometheus-client`、`pytest`、`httpx`、`ruff`。controller 只依赖下列接口：

```python3
class Executor(Protocol):
    def start(self, job: Job, attempt: int) -> ExecutionHandle: ...
    def inspect(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
```

`FakeExecutor` 在 101 模拟成功、失败、卡住和取消，用于确定性测试。`LocalProcessExecutor` 在 200 启动 `torchrun`，持久化 PID、PGID、日志路径、命令和开始时间。

完成判据：`make test`、`make lint` 可运行，FakeExecutor 的成功/失败/取消测试通过。提交 `chore: scaffold bare-metal training platform`。

## 4. 任务规格、数据库和状态机（第 1 周）

第一版任务 YAML：

```yaml
apiVersion: training.example.io/v1alpha1
kind: TrainingJob
metadata:
  name: ddp-two-gpu
spec:
  host: gpu200
  command: ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "ddp_baseline.train"]
  args: ["--device", "cuda", "--amp", "--epochs", "20", "--checkpoint-every-steps", "10"]
  gpus: [0, 1]
  priority: 50
  checkpointDir: "/path/to/runs/platform/ddp-two-gpu"
  retryPolicy:
    maxRetries: 2
    backoffSeconds: 30
```

Pydantic 必须拒绝未知字段，并验证命令非空、GPU ID 无重复且位于登记 host、checkpoint 处于允许根目录、重试参数合法。用户不得传 `CUDA_VISIBLE_DEVICES`、`RANK`、`WORLD_SIZE` 等执行器控制的环境变量。

| 表 | 必要字段 |
| --- | --- |
| `jobs` | `id`、`generation`、`spec_json`、`desired_state`、`status`、`attempt`、`reason`、时间戳、`next_retry_at` |
| `events` | `job_id`、`attempt`、前后状态、`reason`、时间戳、`payload_json` |
| `allocations` | `job_id`、`generation`、`host`、`gpu_ids_json`、`released_at` |

状态机：

```text
PENDING -> ADMITTED -> STARTING -> RUNNING -> SUCCEEDED
                    |             |  |
                    |             |  +-> CANCELLING -> CANCELLED
                    |             +----> RETRYING -> STARTING
                    +------------------> FAILED
```

终态不能被普通 reconcile 改写；状态变化和 event 插入在同一事务；重复观察同一进程状态不得重复计数。为非法 spec、非法跳转、终态不变和重复 event 写单元测试。

## 5. API 与 CLI（第 2 周前半）

API 只接收意图和查询状态，成功提交返回 `202 Accepted`，绝不等待训练结束：

| 操作 | HTTP | CLI |
| --- | --- | --- |
| 提交 | `POST /v1/jobs` | `trainctl submit job.yaml` |
| 查询 | `GET /v1/jobs/{id}` | `trainctl get ID` |
| 列表 | `GET /v1/jobs` | `trainctl list` |
| 取消 | `POST /v1/jobs/{id}/cancel` | `trainctl cancel ID` |
| 事件 | `GET /v1/jobs/{id}/events` | `trainctl events ID` |

用 `Idempotency-Key` 实现提交幂等：同 key 同 spec 返回原任务，同 key 不同 spec 返回 409。API 重启后 SQLite 记录仍必须存在。取消只写 `desired_state=CANCELLED`，由 controller 实际结束进程。

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
trainctl submit examples/cpu-smoke.yaml
trainctl list
trainctl get <job-id>
trainctl cancel <job-id>
```

完成判据：单卡、双卡、非法 GPU 请求三份样例都展示预期结果。

## 6. Controller 与 LocalProcessExecutor（第 2 周后半）

controller 每 2--5 秒扫描非终态任务：

```text
if cancelled: terminate process group; release GPU; transition to CANCELLED after exit
elif PENDING and all requested GPUs are free: atomically allocate all GPUs; transition to ADMITTED
elif ADMITTED: start one torchrun process group; persist handle; transition to STARTING
elif alive and training metrics appears: transition to RUNNING
elif exit code == 0: release GPU; transition to SUCCEEDED
elif exit code != 0: classify; release GPU; RETRYING or FAILED
elif RETRYING and retry time arrived: increment attempt; transition to STARTING
```

实现规则：

1. `subprocess.Popen(..., start_new_session=True)` 启动独立进程组；保存 PGID。
2. 显式设置 `CUDA_VISIBLE_DEVICES`；stdout/stderr 追加至 `runs/platform/jobs/<job-id>/attempt-<n>/trainer.log`。
3. 只允许命令白名单、固定工作目录和受控环境；禁止 `shell=True` 拼接用户 YAML。
4. 取消先对 PGID 发送 `SIGTERM`，超时才 `SIGKILL`。controller 重启后通过 PID/PGID、日志、checkpoint 重建状态。

在 200 用耗时足够长的任务测试：停止 controller 后重启，确认不出现第二个 `torchrun`。完成判据：重复 reconcile 不重复拉起进程，取消后所有 rank 退出。

## 7. GPU 队列、优先级和完整 admission（第 3 周）

第一版只管理本平台启动的进程，不抢占其他用户任务。admission 前检查实际使用情况：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi pmon -c 1
```

`select_next_admission(jobs, free_gpus)` 是纯函数：只看 `PENDING`，按 `priority DESC, created_at ASC, id ASC` 排序。任务请求 `[0,1]` 而只空闲 `[0]` 时必须保持 `PENDING`：

```text
waiting for full GPU allocation: requested=[0,1], available=[0], queue_position=2
```

以 SQLite `BEGIN IMMEDIATE` 包住“读取空闲 GPU、选择任务、写 allocation、更新状态”，避免重复分配。验证高优先级优先、同优先级 FIFO、资源不足不启动、终态和取消释放 allocation。任务永远不能只分到请求的一部分 GPU。

## 8. Checkpoint、重试和恢复（第 4 周）

每个 attempt 使用独立输出目录：

```text
runs/platform/jobs/<job-id>/
  attempt-1/trainer.log
  attempt-1/checkpoints/checkpoint.pt
  attempt-2/trainer.log
  attempt-2/checkpoints/checkpoint.pt
```

规则：rank 0 以临时文件加原子 rename 写 checkpoint；记录 global step、文件大小、sha256 和版本；controller 仅使用校验通过的 `last_valid_checkpoint`；下一 attempt 用新目录输出并通过 `--resume` 读取旧 checkpoint；训练配置不兼容时明确失败。

| 类别 | 例子 | 默认动作 |
| --- | --- | --- |
| `INFRASTRUCTURE` | 受控 worker 被终止、进程意外消失 | 指数退避重试 |
| `TRANSIENT_IO` | 可证明的暂时读写失败 | 指数退避重试 |
| `USER_ERROR` | 参数错误、traceback、约定退出码 | 直接失败 |
| `CANCELLED` | 用户取消 | 不重试 |

退避：`min(base * 2^(attempt-1), max_backoff)`。event 记录退出码、分类、下次重试时间和 checkpoint 版本。验收：终止运行中的进程组后出现新 attempt，且恢复的 `global_step` 不倒退。

## 9. 性能与算子/图优化主线（第 4--5 周）

平台不应挤占性能项目。每轮优化固定执行：**基线 -> profiler 证据 -> 单一假设 -> 单一改动 -> 三次对照 -> 保留或回退**。

| 实验 | 采集 | 合格结论 |
| --- | --- | --- |
| AMP/BF16 | 吞吐、峰值显存、数值稳定性 | 明确精度与性能取舍 |
| DDP 1/2 卡 | 吞吐、step time、扩展效率 | 区分计算与通信瓶颈 |
| 算子/图优化 | 热点、kernel 次数、端到端 step time | 报端到端收益，非只报微基准 |

短暂采集稳态窗口：

```bash
nsys profile --trace=cuda,nvtx,osrt -o reports/ddp-step \
  torchrun --standalone --nproc_per_node=2 -m ddp_baseline.train \
  --device cuda --amp --epochs 1 --samples 4096 --batch-size 64
```

没有 Nsight 时，先用 PyTorch Profiler 和 wall time、吞吐、显存数据。算子融合前必须证明热点可融合；报告需含数值一致性、端到端吞吐、回归风险。诊断细节见 [phase-2-系统性能与通信学习.md](phase-2-系统性能与通信学习.md)。

## 10. 可观测性与故障演练（第 5 周）

JSON 日志最少包含 `timestamp`、`level`、`component`、`job_id`、`attempt`、`host`、`gpu_ids`、`event`、`reason`、`exit_code`，且不得包含凭据或完整环境变量。

控制器暴露 `/metrics`：

```text
training_jobs{state="PENDING"}                    gauge
training_queue_wait_seconds                         histogram
training_admission_latency_seconds                  histogram
training_attempts_total{outcome="retry|success"}   counter
training_recovery_seconds                            histogram
training_failures_total{classification="..."}      counter
```

```bash
curl -fsS http://127.0.0.1:8001/metrics | rg '^training_'
trainctl events <job-id>
tail -n 100 runs/platform/jobs/<job-id>/attempt-1/trainer.log
```

所有故障只作用于自己启动的 job，并将时间、命令、预期/实际状态、日志/指标证据、恢复时间和清理结果记到 `docs/incidents/<date>-<case>.md`。

| 故障 | 安全注入 | 验收 |
| --- | --- | --- |
| worker 消失 | 对保存 PGID 发送 `SIGTERM` | `RETRYING`、新 attempt、step 连续 |
| 通信超时 | 仅在受控多 rank 测试设置短 timeout 或让测试 rank 延迟退出 | 有界失败，不无限 hang |
| 磁盘写入失败 | 使用测试目录与小文件限制，绝不写满共享磁盘 | 旧 checkpoint 可读 |
| 主进程异常 | 测试专用 `FAIL_AT_STEP` 和固定退出码 | `USER_ERROR` 且不重试 |

## 11. 远程执行与多节点（第 6 周，可选）

若 controller 位于 101、训练位于 200，用受限 SSH 账号调用固定 runner 脚本。runner 接收 job JSON，不接受任意 shell 字符串；持久化远程 PID/PGID、日志路径、commit 和启动时间。先让本地执行器稳定，再抽取 `RemoteProcessExecutor`，并复用同一接口和状态机测试。

只有未来存在第二台可登录 GPU 主机时才做多节点：确认网络、端口、驱动、PyTorch/NCCL、checkpoint 路径均可用，再设置固定 `MASTER_ADDR`、`MASTER_PORT`、`--nnodes`、`--node_rank`。没有真实条件时标注“未验证多节点”。

## 12. 最终验收与证据包

```bash
make lint
make test
pytest tests/integration -v
nvidia-smi
trainctl list
```

演示：展示状态机和 101/200 边界；提交竞争 GPU 任务；展示 `torchrun`、GPU 映射和日志；展示 metrics/事件/profiler；终止 worker 后恢复；取消任务；最后讲一份算子/图优化或 DDP 性能报告。

交付包含：运行/清理说明、架构图、状态机与幂等设计、三个任务 YAML、测试报告、四份故障记录、一份性能优化报告和结果表。所有数字附环境、commit、命令与原始证据路径；无真实多机结果时明确标注。

| 周 | 必须可运行的东西 | 必须验证 |
| --- | --- | --- |
| 1 | 200 单/多卡基线；101 领域模型 | checkpoint、spec/状态机测试 |
| 2 | API/CLI、Fake/Local executor | 重复 reconcile/重启不重复启动 |
| 3 | GPU 队列与完整 admission | 不足不启动，释放后完整分配 |
| 4 | 恢复与性能证据 | 删进程后恢复；完成 profiler 分析 |
| 5 | 指标和故障演练 | 每类故障有日志、事件、退出码和终态 |
| 6 | 远程/多节点（可选） | 只报告真实执行过的结论 |
