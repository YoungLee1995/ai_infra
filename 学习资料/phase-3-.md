# 第 3 阶段操作手册：训练平台 MVP（裸机路线）

> 周期：第 5-6 个月，每周 15-20 小时
> 环境：200（单机完成控制面开发、CPU 测试和 GPU 真实训练）
> 前置：完成第 1 阶段 DDP trainer 和第 2 阶段性能优化
> 产出：可提交、排队、成组启动、观测、取消、重试和恢复的训练作业平台

本文是按指令执行的手册。全程在 200 完成：同一台机器运行 API、Controller、SQLite、LocalProcessExecutor 和真实 `torchrun`。200 可以联网，因此 Python 包和后续所需资源直接在 200 下载。每一步完成通过标准并保存证据，未通过不要进入下一步；Kubernetes 留到第 4 阶段。

## 0. 固定环境、路径和边界

### 本章目的

先把控制面、执行面、代码目录和产物目录固定下来，避免后续把“平台状态”和“训练状态”混在一起。这里的核心不是创建目录，而是建立系统边界：API 接收意图，Controller 做决策，Executor 负责副作用。

### 本章知识点

- 控制面与数据面/执行面分离：控制面保存期望状态，执行面产生真实进程和指标。
- `run/` 与 `runs/` 分离可以降低清理风险，也方便把平台日志和训练日志分别归档。
- 裸机 MVP 的边界是单主机、单数据库、GPU ID 分配；不要把它误认为多机调度器。

在 200 执行：

```bash
export PLATFORM_ROOT=/workspace/GIT/ai_infra/training-platform-mvp
export RUN_ROOT=/workspace/GIT/ai_infra/training-platform-runs
mkdir -p "$PLATFORM_ROOT" "$RUN_ROOT"
```

200 同时承担 API、Typer、SQLite、Controller、FakeExecutor 测试和 LocalProcessExecutor 的真实训练。逻辑上仍保留控制面和执行面分离：API 接收意图，Controller 决策，Executor 负责启动/观测/取消同机的 `torchrun`。`run/` 保存平台运行时（DB、锁、handle），`runs/` 保存训练输出（日志、metrics、checkpoint）。模型和数据复用第 2 阶段本地目录，不提交 Git。

```text
training-platform-mvp/
  api/{main.py,routes.py,cli.py}
  controller/{worker.py,state_machine.py,queue.py,reconciler.py}
  executor/{base.py,fake.py,local.py}
  models/{spec.py,db.py,domain.py}
  examples/{cpu-smoke.yaml,gpu-two.yaml,invalid.yaml}
  tests/{test_spec.py,test_state_machine.py,test_queue.py,test_fake_executor.py,test_reconciler.py,test_local_executor.py}
  tests/integration/{test_api.py,test_recovery.py}
  docs/{architecture.md,runbook.md,evidence/,incidents/}
  run/ runs/ pyproject.toml Makefile
```

### 本章知识点总结

完成后，你应能画出“用户/API → Controller → Executor → torchrun”的链路，并解释同机部署时为什么仍要把 API、Controller 和 Executor 分层，而不是让 API 直接启动训练进程。

## Step 1：准备主机和 Python 环境（第 5 周第 1 天）

### 本步目的

验证开发依赖、CUDA、GPU 数量和磁盘路径，建立可复查的环境基线。训练平台的问题经常来自环境漂移；先记录版本，后面的失败才有证据可比对。

### 关键知识点

- Python venv 隔离平台依赖；在 200 上直接复用第 2 阶段已验证的 CUDA/PyTorch venv，并通过网络安装平台依赖。
- `cuda=True` 只说明 CUDA 可用，不代表显存、驱动、NCCL 或模型路径都正确。
- 环境证据应包含 Python、框架版本、GPU 型号、显存、驱动和设备数量。

在 200 创建平台目录，并直接安装控制面依赖：

```bash
cd "$PLATFORM_ROOT"
mkdir -p api controller executor models examples tests/integration docs/evidence docs/incidents run runs
export PHASE2_ROOT="$HOME/qwen3_phase2"  # 改为第 2 阶段实际项目目录
source "$PHASE2_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
pip install fastapi 'uvicorn[standard]' typer pydantic sqlalchemy aiosqlite httpx pytest pytest-asyncio pyyaml prometheus-client structlog
python - <<'PY' | tee docs/evidence/environment-200.txt
import sys, fastapi, pydantic, sqlalchemy, torch
print('python=', sys.version)
print('fastapi=', fastapi.__version__)
print('pydantic=', pydantic.__version__)
print('sqlalchemy=', sqlalchemy.__version__)
print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'devices=', torch.cuda.device_count())
PY
```

继续在 200 记录 GPU 和驱动信息：

```bash
cd "$PLATFORM_ROOT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee docs/evidence/nvidia-smi-200.txt
```

**通过标准**：200 能导入平台依赖，`cuda=True`，GPU 数量正确。所有证据、代码、SQLite 和训练产物都位于 200 的固定路径。

### 本步知识点总结

你应能区分“控制面依赖安装成功”和“执行面 GPU 运行条件满足”，并能用证据文件回答“这次实验在 200 的哪个 Python/CUDA 环境运行”。

## Step 2：实现任务规格和输入校验（第 5 周第 2 天）

### 本步目的

把用户的 YAML 意图转换成安全、类型明确、可持久化的 `JobSpec`。校验必须发生在启动进程前，防止非法路径、重复 GPU 或用户伪造 rank 破坏 Controller 的假设。

### 关键知识点

- Pydantic 的 `extra="forbid"` 防止拼写错误字段静默失效。
- `Field(ge/le/min_length)` 是声明式边界校验；自定义 validator 负责跨字段和安全策略。
- `CUDA_VISIBLE_DEVICES`、`RANK`、`WORLD_SIZE` 等属于 Executor 控制面，不应由用户 spec 覆盖。
- checkpoint 根目录白名单同时解决路径越权和产物可回收问题。

创建 `models/spec.py`：

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = "/workspace/GIT/ai_infra/training-platform-runs/"
FORBIDDEN = {"CUDA_VISIBLE_DEVICES", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"}

class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maxRetries: int = Field(ge=0, le=10, default=1)
    backoffSeconds: int = Field(ge=1, le=3600, default=30)

class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "gpu200"
    command: list[str] = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    gpus: list[int] = Field(min_length=1)
    priority: int = Field(ge=0, le=100, default=50)
    checkpointDir: str
    retryPolicy: RetryPolicy = Field(default_factory=RetryPolicy)

    @field_validator("gpus")
    @classmethod
    def unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("GPU IDs must not contain duplicates")
        return value

    @field_validator("checkpointDir")
    @classmethod
    def under_root(cls, value):
        if not value.startswith(ROOT):
            raise ValueError("checkpointDir is outside allowed root")
        return value

    @field_validator("args")
    @classmethod
    def no_executor_env(cls, value):
        if any(any(key in arg for key in FORBIDDEN) for arg in value):
            raise ValueError("executor-controlled env override")
        return value
```

创建三个 YAML。`examples/gpu-two.yaml`：

```yaml
host: gpu200
command: [torchrun]
args: ["--standalone", "src/train_qwen3_lora.py", "--steps", "8"]
gpus: [0, 1]
priority: 50
checkpointDir: /workspace/GIT/ai_infra/training-platform-runs/demo/checkpoints
retryPolicy: {maxRetries: 1, backoffSeconds: 5}
```

`cpu-smoke.yaml` 使用单卡；`invalid.yaml` 故意放重复 GPU、`RANK=1` 或越权 checkpoint。为以下输入添加 `tests/test_spec.py`：空命令、重复 GPU、负重试次数、越权 checkpoint、RANK 参数、未知字段。

```bash
pytest -q tests/test_spec.py | tee docs/evidence/day15-spec-validation.txt
```

**通过标准**：六类非法输入均被拒绝，合法 YAML 可被 `JobSpec.model_validate` 解析。

### 本步知识点总结

你应能说明“校验不是用户体验功能，而是调度器的安全边界”，并能为新增字段补充类型约束、拒绝策略和对应测试。

## Step 3：数据库、event 和状态机（第 5 周第 3-4 天）

### 本步目的

建立平台的持久化事实和可重放状态迁移。Controller 可能崩溃、重复观察或重复 reconcile，因此状态、事件和资源分配必须可恢复、可审计、可去重。

### 关键知识点

- `desired_state` 是用户意图，`status` 是 Controller 观察到的事实，两者不能混用。
- 状态机用纯函数表达合法迁移，便于单元测试和故障恢复。
- 状态更新、event 写入、allocation 释放必须在一个事务内完成，否则会出现“状态成功但资源未释放”等不一致。
- `event_hash` 唯一约束提供幂等事件写入；终态是单向的，普通 reconcile 不得改写。

在 `models/db.py` 建立表：

```text
jobs: id,generation,spec_json,desired_state,status,attempt,reason,created_at,updated_at,next_retry_at,idempotency_key,spec_hash
events: job_id,attempt,from_state,to_state,reason,timestamp,payload_json,event_hash（唯一约束）
allocations: job_id,generation,host,gpu_ids_json,released_at
```

在 `controller/state_machine.py` 实现：

```python
VALID = {
    "PENDING": {"ADMITTED", "CANCELLING"},
    "ADMITTED": {"STARTING", "FAILED", "CANCELLING"},
    "STARTING": {"RUNNING", "RETRYING", "FAILED", "CANCELLING"},
    "RUNNING": {"SUCCEEDED", "RETRYING", "FAILED", "CANCELLING"},
    "RETRYING": {"STARTING", "FAILED", "CANCELLING"},
    "CANCELLING": {"CANCELLED"},
    "SUCCEEDED": set(), "FAILED": set(), "CANCELLED": set(),
}
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}

def transition(current, desired):
    if current in TERMINAL:
        return current
    if desired not in VALID[current]:
        raise ValueError(f"invalid transition: {current}->{desired}")
    return desired
```

状态变化、event 插入、allocation 释放必须在一个事务中完成；用 `event_hash` 去重。测试合法/非法迁移、终态不变、重复 event：

```bash
pytest -q tests/test_state_machine.py | tee docs/evidence/day16-state-machine.txt
```

**通过标准**：重放同一 reconcile 不增加相同 event。

### 本步知识点总结

你应能从数据库记录恢复某个 job 的完整生命周期，并解释为什么“数据库状态”和“外部进程状态”仍需在 reconcile 时再次核对。

## Step 4：API、幂等提交和 CLI（第 5 周第 5-6 天）

### 本步目的

提供稳定的异步用户接口，让提交请求快速返回，而不是把 HTTP 请求绑定到训练时长。幂等提交保证客户端重试不会意外创建多个相同任务。

### 关键知识点

- `202 Accepted` 表示请求已接受、任务尚未完成；`200` 更适合已完成的同步查询。
- `Idempotency-Key` 必须和规范内容的 hash 一起校验：同 key 同 spec 重放，异 spec 返回 409。
- API、CLI 和数据库之间应共享领域服务，避免 CLI 绕过 API 产生另一套语义。
- API 重启后可查询 SQLite，说明状态不依赖进程内存。

在 `api/routes.py`、`api/main.py` 实现：

| 方法 | 路径 | 语义 |
|---|---|---|
| POST | `/v1/jobs` | 创建 PENDING job，返回 202 和 id |
| GET | `/v1/jobs/{id}` | 查询状态 |
| GET | `/v1/jobs` | 列表 |
| POST | `/v1/jobs/{id}/cancel` | 请求取消，返回 202 |
| GET | `/v1/jobs/{id}/events` | 读取 event |

提交必须要求 `Idempotency-Key`：同 key 同 spec 返回原任务；同 key 不同 spec 返回 409。API 只接收意图，绝不等待训练结束。

`api/cli.py` 提供：

```bash
trainctl submit examples/cpu-smoke.yaml --idempotency-key smoke-001
trainctl list
trainctl get <JOB_ID>
trainctl cancel <JOB_ID>
trainctl events <JOB_ID>
```

验证：

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
# 另开 200 终端：重复提交同 key；修改 YAML 后复用 key；提交 invalid.yaml
```

**通过标准**：首次提交 202，重复提交返回同一 job，改 YAML 后 409，非法输入 422，重启 API 后仍可查询 SQLite。输出保存 `docs/evidence/day17-api-idempotency.txt`。

### 本步知识点总结

你应能解释异步 API 的状态码语义、幂等键的冲突处理，以及为什么 API 只写入 `PENDING` 而不直接启动训练。

## Step 5：Executor 抽象和 FakeExecutor（第 5 周第 7-8 天）

### 本步目的

先用不接触真实进程的 FakeExecutor 验证 Controller 逻辑，再把同一协议替换成真实执行器。这样可以把状态机测试与 GPU、NCCL、操作系统进程测试解耦。

### 关键知识点

- Protocol/接口描述的是 Controller 需要的能力，而不是某种实现。
- `ExecutionHandle` 是持久化的外部事实入口；只保存 PID 不足以恢复日志、PGID 和 attempt。
- FakeExecutor 是确定性测试替身，可主动制造 hang、失败和取消。

在 `executor/base.py` 定义 `ExecutionHandle(job_id, attempt, pid, pgid, log_path, started_at)`、`ExecutionStatus(phase, exit_code, message)` 和 `start/inspect/cancel` Protocol。在 `executor/fake.py` 实现不启动真实进程的 FakeExecutor，可编程返回 `RUNNING`、`SUCCEEDED`、`FAILED`、`CANCELLED`。

```bash
pytest -q tests/test_fake_executor.py tests/test_reconciler.py | tee docs/evidence/day18-fake-executor.txt
```

**通过标准**：即使不实际启动 GPU 训练，200 上的 FakeExecutor 仍能覆盖 Controller 的成功、失败、hang 和取消分支。

### 本步知识点总结

你应能用同一组 Controller 测试替换 FakeExecutor 和 LocalProcessExecutor，并说明这种依赖倒置如何降低测试成本。

## Step 6：LocalProcessExecutor、torchrun 和 PGID（第 6 周第 1-2 天）

### 本步目的

把抽象的 job 变成 200 上可观测、可取消的真实进程组，并验证 DDP 子进程不会在取消后遗留。

### 关键知识点

- `start_new_session=True` 创建新的 session/进程组，PGID 通常等于 torchrun 的 PID。
- 杀 PID 只终止父进程；`os.killpg` 才能覆盖 torchrun 和所有 rank。
- Executor 必须显式构造环境变量和 argv，避免 shell 注入、用户覆盖 rank 或错误继承环境。
- inspect 是轮询外部事实的入口，不能用“曾经启动过”代替“现在仍在运行”。

在 `executor/local.py` 中只允许白名单命令 `torchrun`，按 GPU 数补 `--nproc_per_node`。executor 设置 `CUDA_VISIBLE_DEVICES`，用户不得传 `RANK/WORLD_SIZE`；日志写入 `runs/platform/jobs/<id>/attempt-<n>/trainer.log`。

```python
proc = subprocess.Popen(
    argv, start_new_session=True, cwd=PLATFORM_ROOT, env=env,
    stdout=log_file, stderr=subprocess.STDOUT,
)
pgid = os.getpgid(proc.pid)
```

`inspect()` 查询进程和持久化退出码；`cancel()` 先 `os.killpg(pgid, SIGTERM)`，等待 10 秒后才 `SIGKILL`。不要只杀 PID，否则 DDP rank 会成为孤儿进程。

200 上从平台实际启动一次 2 step 作业，并验证日志、handle、退出码和 PGID 取消：

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
# 另开 200 终端
python -m controller.worker --interval 2
# 再开终端
trainctl submit examples/cpu-smoke.yaml --idempotency-key local-smoke-001
trainctl get <JOB_ID>
cat run/jobs/<JOB_ID>/attempt-1/handle.json
kill -TERM -<PGID>
```

**通过标准**：日志、handle 和退出码均可查询；发送 SIGTERM 后整个进程组消失；重复 reconcile 不得启动第二个 torchrun。保存 `docs/evidence/day19-reconcile-idempotent.txt`。

### 本步知识点总结

你应能画出 torchrun、rank 子进程、PGID 和日志文件的关系，并能解释优雅终止、超时强杀和退出码采集的先后顺序。

## Step 7：Controller reconcile（第 6 周第 2-3 天）

### 本步目的

实现平台的控制循环：反复读取数据库和执行器观察结果，逐步把实际状态收敛到用户期望状态。reconcile 必须允许重复运行，因此天然要求幂等。

### 关键知识点

- Controller 是 level-triggered 控制循环，不是只执行一次的脚本。
- 每个状态分支只做一个可验证的推进，避免一次循环跨越过多状态导致恢复困难。
- 重试要区分可重试基础设施故障和用户错误；取消优先级高于普通启动。
- Controller 重启恢复靠持久化 handle + inspect，而不是依靠进程内存。

`controller/worker.py` 每 2 秒扫描非终态任务；`reconciler.py` 的基本流转：

```text
PENDING + GPU 足够 -> ADMITTED -> STARTING -> RUNNING -> SUCCEEDED
RUNNING 失败 -> RETRYING（指数退避）或 FAILED
任意非终态 + desired=CANCELLED -> CANCELLING -> CANCELLED
```

每次启动前检查已有 handle；Controller 重启后通过 `executor.inspect(handle)` 校验外部事实，不能只信数据库中的 RUNNING。测试成功、失败、取消、重试、重启不重复启动：

```bash
pytest -q tests/test_reconciler.py
```

### 本步知识点总结

你应能解释一次 reconcile 为什么可以安全执行多次，并能从任意中间状态推导下一步动作和幂等条件。

## Step 8：队列和 gang admission（第 6 周第 4-5 天）

### 本步目的

在有限 GPU 上实现公平、可解释且不超卖的准入。DDP 任务必须一次拿到完整 GPU 集合，不能先启动一部分 rank 再等待其余资源。

### 关键知识点

- gang admission 的全有或全无语义避免部分启动和死锁。
- 优先级、创建时间和 job id 组成确定性排序，便于复现和面试解释。
- `BEGIN IMMEDIATE` 让“检查空闲 GPU”和“写入 allocation”不可被并发 reconcile 插队。
- 排队 reason 是可观测性的一部分，不能只显示 PENDING。

在 `controller/queue.py` 按 `priority DESC, created_at ASC, id ASC` 选择任务；GPU 必须全有或全无。分配时用 SQLite `BEGIN IMMEDIATE`，在锁内重新读取 free set、插入 allocation、更新状态。

```bash
python -m controller.hosts register --host gpu200 --gpus 0,1
trainctl submit examples/gpu-two.yaml --idempotency-key gpu-a
trainctl submit examples/gpu-two.yaml --idempotency-key gpu-b
watch -n 1 'trainctl list; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

**通过标准**：第一个占 `[0,1]` 时第二个保持 PENDING，reason 写 requested/available；释放后第二个一次性 ADMITTED；并发测试不超卖。保存 `docs/evidence/day20-gang-admission.txt`。

### 本步知识点总结

你应能解释资源分配中的检查-使用竞态（TOCTOU），并能证明为什么事务锁和全量 GPU 集合能避免超卖与部分启动。

## Step 9：checkpoint、重试和恢复（第 6 周第 6-7 天）

### 本步目的

让训练作业在 worker 被杀、进程异常或临时 I/O 失败后可恢复，同时避免把用户代码错误无限重试。每个 attempt 独立留痕，恢复点必须可验证。

### 关键知识点

- attempt 是一次执行尝试，不等于 job；job 可以有多个 attempt。
- 错误分类决定重试策略；指数退避避免故障时的重试风暴。
- 临时文件 + fsync + `os.replace` 提供同文件系统内的近似原子发布，避免半写 checkpoint 被误读。
- 恢复验证要比较 global step、checkpoint 可读性和 event 时间线，而不只看最终 exit code。

```text
runs/platform/jobs/<id>/attempt-1/{trainer.log,checkpoints/checkpoint.pt}
runs/platform/jobs/<id>/attempt-2/{trainer.log,checkpoints/checkpoint.pt}
```

退出码 2 是 `USER_ERROR`（不重试），137/143 是 `INFRASTRUCTURE`（重试）；临时 I/O 可重试；取消不重试。退避：`min(base * 2 ** (attempt - 1), 600)`。checkpoint 先写临时文件和 fsync，再 `os.replace(tmp, final)`。

```bash
trainctl submit examples/gpu-two.yaml --idempotency-key recovery-001
cat run/jobs/<JOB_ID>/attempt-1/handle.json
kill -TERM -<PGID>
watch -n 1 'trainctl get <JOB_ID>; trainctl events <JOB_ID>'
```

**通过标准**：出现 RETRYING、attempt-2，从最近 checkpoint 恢复，后一次 `metrics.jsonl` 的最大 global_step 不倒退。保存 `docs/evidence/day21-checkpoint-recovery.txt`。

### 本步知识点总结

你应能区分 job 重试、checkpoint 恢复和从头重跑，并能说明哪些退出码不应自动重试。

## Step 10：指标、日志和故障演练（第 6 周第 8-10 天）

### 本步目的

把“平台是否健康”变成可查询的指标、日志和故障证据，并用受控故障验证设计，而不是只在成功路径上演示。

### 关键知识点

- Gauge 表示当前状态，Counter 表示累计次数，Histogram 适合等待时间和恢复耗时分布。
- Prometheus label 必须是有限集合；job id 会造成高基数和内存增长。
- 结构化日志要包含关联字段，使一次 job 的状态、进程和恢复事件可串联。
- 故障演练一次只改变一个变量，否则无法归因。

### 本章知识点总结

你应能从指标回答“队列是否变长、失败是否增加、恢复是否变慢”，并能从 incident 记录复盘注入、现象、根因和清理结果。

暴露 `training_jobs{state}`、`training_queue_wait_seconds`、`training_admission_latency_seconds`、`training_attempts_total{outcome}`、`training_recovery_seconds`、`training_failures_total{classification}`。label 只能使用有限集合，禁止 job id。JSON 日志至少含 timestamp、component、job_id、attempt、host、gpu_ids、event、reason、exit_code。

每次只注入一种故障：worker SIGTERM、受控 NCCL timeout、测试目录磁盘写失败、固定 `FAIL_AT_STEP` 主进程异常。验收分别为重试恢复、有界失败、旧 checkpoint 可读、USER_ERROR 不重试。每次记录到 `docs/incidents/<date>-<case>.md`，并汇总 `docs/evidence/day22-failure-drills.md`。

## Step 11：最终演示和验收（第 6 周第 11-14 天）

### 本步目的

用一条固定演示链路证明平台从提交到恢复的闭环，并把代码、测试、运行手册和第 2 阶段性能证据打包成可复现交付物。

### 关键知识点

- 端到端验收验证的是组件之间的契约，不是重复单元测试。
- 演示顺序应覆盖排队、运行、观测、失败恢复和取消五种生命周期。
- 已知限制必须显式写出：本阶段无真实多机和 K8s，不把单机结论外推。

固定演示顺序：提交两个竞争 GPU 的任务；启动双卡 DDP；查看 metrics/events/profiler；终止 worker 展示 `RETRYING -> attempt-2`；取消另一个任务展示 `CANCELLING -> CANCELLED`；讲解第 2 阶段性能报告。

```bash
make lint
make test
pytest tests/integration -v
nvidia-smi
trainctl list
find docs/evidence docs/incidents -type f | sort
git status --short
```

交付架构图、状态机/幂等/队列/checkpoint 设计、runbook、单卡/双卡/非法 YAML、测试报告、四份故障记录、第 2 阶段性能报告及“无多机/K8s”限制。

### 本步知识点总结

你应能让另一位工程师仅凭交付包复现一次提交、排队、训练、取消和恢复，并能明确哪些结论来自实测、哪些只是后续扩展设计。

## 完成检查

- [ ] 200 的环境、路径和证据已固定。
- [ ] 规格校验、状态机、event 去重、事务测试通过。
- [ ] API 返回 202，幂等 key 和 CLI 正常。
- [ ] Fake/Local Executor、PGID 取消和 reconcile 重启测试通过。
- [ ] gang admission 不超卖，checkpoint 可恢复，重试分类正确。
- [ ] 四类故障独立注入并有 incident 记录。
- [ ] `make test`、集成测试和最终演示通过。

## 面试自测

1. 为什么 API 返回 202 而不是 200？
2. Controller 重启后如何验证 RUNNING 的外部事实？
3. 为什么杀 PID 不等于杀 DDP 作业？
4. gang admission 为什么需要 `BEGIN IMMEDIATE`？
5. 同 key 不同 spec 为何返回 409？
6. checkpoint 为什么用临时文件加 `os.replace`？
7. Prometheus label 为什么不能放 job id？
8. Controller 崩溃后如何避免重复启动？

## 与第 4 阶段衔接

稳定后可将 `LocalProcessExecutor` 替换为 `KubernetesExecutor`，或在具备多机资源时增加远程执行器。API、状态机、幂等、队列、事件和恢复语义保持不变。
