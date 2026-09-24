## 第 3 阶段操作手册：训练平台 MVP

> 周期：第 5–6 个月，每周 15–20 小时
> 环境：101（CPU 控制面开发/测试）+ 200（GPU 真实训练）
> 前置：完成第 1 阶段（DDP trainer）和第 2 阶段（性能优化）
> 产出：一个能提交、排队、成组启动、观测、取消、失败恢复的训练作业平台

### 0. 路线选择（先做决定）

你手上有两份路线冲突的材料，必须先选一条：

| 维度 | 裸机路线 | K8s 路线 |
|---|---|---|
| 执行器 | `subprocess.Popen` + PGID + `torchrun` | Kubernetes Job/Pod |
| 学习价值 | 深入进程组、信号、故障恢复 | 深入容器编排、调度、声明式 API |
| 环境要求 | 一台 GPU 机器（200）即可 | 需要 K8s 集群（kind/minikube 或内网） |
| 面试匹配 | **更贴“训练 Infra 系统工程师”** | 更贴“平台/云原生”岗位 |
| 你的背景 | NPU 建模偏底层，裸机更顺 | 需补 K8s 生态 |

**建议：先裸机，后 K8s。** 你的目标岗位是“分布式训练/训练平台/训练性能”，裸机路线让你真正理解“DDP 是进程组”这件事——这是训练 Infra 的核心认知，也是 K8s 路线容易掩盖的。K8s 可以作为第 4 阶段的扩展。

**下面按裸机路线展开。** 如果你选 K8s，告诉我，我给对应版本。

### 1. 系统架构与边界（第 5 周第 1 天）

#### 1.1 控制面 / 执行面分离

```text
101（控制面）                          200（执行面）
┌─────────────────────┐                ┌──────────────────────┐
│ FastAPI (API)       │                │ torchrun 进程组       │
│ Typer CLI (trainctl)│                │  ├─ rank 0           │
│ Controller/Reconciler│  ── 启动 ──→  │  ├─ rank 1           │
│ SQLite (状态库)      │                │  └─ ...              │
│ FakeExecutor (测试)  │  ←─ 观测 ──   │ checkpoint / metrics │
└─────────────────────┘                └──────────────────────┘
```

**核心原则**：API 只接收意图，返回 `202 Accepted`，**绝不等待训练结束**。Controller 才负责最终执行。

#### 1.2 项目结构

```text
training-platform-mvp/
  api/
    main.py            # FastAPI app
    routes.py          # HTTP 路由
    cli.py             # Typer CLI (trainctl)
  controller/
    worker.py          # 主 reconcile 循环
    state_machine.py   # 状态迁移纯函数
    queue.py           # 队列/优先级/admission
    reconciler.py      # 单个 job 的 reconcile 逻辑
  executor/
    base.py            # Executor Protocol, ExecutionHandle, ExecutionStatus
    fake.py            # FakeExecutor（101 测试）
    local.py           # LocalProcessExecutor（200 真实）
  models/
    spec.py            # Pydantic spec 模型
    db.py              # SQLAlchemy 表定义
    domain.py          # 领域对象
  examples/
    cpu-smoke.yaml
    gpu-two.yaml
    invalid.yaml
  tests/
    test_spec.py
    test_state_machine.py
    test_queue.py
    test_reconciler.py
    test_local_executor.py
    integration/
      test_api.py
      test_recovery.py
  docs/
    architecture.md
    runbook.md
    incidents/
    evidence/
  run/                 # 运行时产物（DB、handle）
  runs/                # 训练输出（checkpoint、日志）
  pyproject.toml
  Makefile
```

**注意 `run/` 和 `runs/` 的分工**：
- `run/`：平台自身运行时（SQLite DB、handle.json、controller 锁）
- `runs/`：训练输出（checkpoint、metrics.jsonl、trainer.log）

#### 1.3 三层事实

| 层 | 含义 | 存储 |
|---|---|---|
| **spec** | 用户意图 | YAML → `jobs.spec_json` |
| **数据库状态** | 控制面事实 | SQLite `jobs`/`events`/`allocations` |
| **执行器状态** | 外部事实 | 进程是否存活、退出码、日志 |

**面试考点**：为什么不能只信数据库？→ controller 重启后，数据库说 `RUNNING`，但进程可能已经死了。必须通过 `inspect()` 重新观测外部事实。

### 2. 任务规格与校验（第 5 周第 2–3 天）

#### 2.1 Pydantic spec 模型

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict

class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maxRetries: int = Field(ge=0, le=10)
    backoffSeconds: int = Field(ge=1, le=3600)

class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 拒绝未知字段
    host: str
    command: list[str] = Field(min_length=1)     # 命令非空
    args: list[str] = []
    gpus: list[int] = Field(min_length=1)        # 至少一张卡
    priority: int = Field(ge=0, le=100)
    checkpointDir: str
    retryPolicy: RetryPolicy

    @field_validator("gpus")
    @classmethod
    def no_duplicate_gpus(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("GPU IDs must not contain duplicates")
        return v

    @field_validator("checkpointDir")
    @classmethod
    def checkpoint_in_allowed_root(cls, v):
        if not v.startswith("/path/to/runs/platform/"):
            raise ValueError("checkpointDir must be under allowed root")
        return v
```

#### 2.2 拒绝执行器控制的环境变量

用户不得在 spec 里传 `CUDA_VISIBLE_DEVICES`、`RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`。这些由 executor 显式设置。

```python
FORBIDDEN_ENV_KEYS = {"CUDA_VISIBLE_DEVICES", "RANK", "WORLD_SIZE",
                       "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"}

@field_validator("args")
@classmethod
def no_forbidden_env(cls, v):
    for arg in v:
        if any(key in arg for key in FORBIDDEN_ENV_KEYS):
            raise ValueError(f"env var {arg} is controlled by executor")
    return v
```

#### 2.3 拒绝测试

为以下输入各写一个测试：

```python
def test_empty_command_rejected(): ...
def test_duplicate_gpu_rejected(): ...
def test_negative_retries_rejected(): ...
def test_checkpoint_outside_root_rejected(): ...
def test_rank_in_args_rejected(): ...
def test_unknown_field_rejected(): ...
```

**记录到 `docs/evidence/day15-spec-validation.md`**。

### 3. 数据库与状态机（第 5 周第 4–5 天）

#### 3.1 三张表

```python
class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True)
    generation = Column(Integer, default=1)
    spec_json = Column(Text)
    desired_state = Column(String)      # 用户期望
    status = Column(String)             # 实际状态
    attempt = Column(Integer, default=0)
    reason = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    next_retry_at = Column(DateTime, nullable=True)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    attempt = Column(Integer)
    from_state = Column(String)
    to_state = Column(String)
    reason = Column(String)
    timestamp = Column(DateTime)
    payload_json = Column(Text)
    event_hash = Column(String)
    __table_args__ = (
        UniqueConstraint("job_id", "attempt", "to_state", "event_hash",
                         name="uq_event_dedup"),
    )

class Allocation(Base):
    __tablename__ = "allocations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    generation = Column(Integer)
    host = Column(String)
    gpu_ids_json = Column(Text)
    released_at = Column(DateTime, nullable=True)
```

#### 3.2 状态机

```text
PENDING ──→ ADMITTED ──→ STARTING ──→ RUNNING ──→ SUCCEEDED
                │              │          │
                │              │          ├──→ CANCELLING ──→ CANCELLED
                │              │          └──→ RETRYING ──→ STARTING
                └──────────────┴──────────────→ FAILED
```

**规则**：
1. 终态（SUCCEEDED / FAILED / CANCELLED）不能被普通 reconcile 改写。
2. 状态变化和 event 插入在**同一事务**。
3. 重复观察同一进程状态**不重复计数**（靠 `event_hash` 唯一索引）。

#### 3.3 状态迁移纯函数

```python
VALID_TRANSITIONS = {
    "PENDING": {"ADMITTED", "CANCELLING"},
    "ADMITTED": {"STARTING", "FAILED", "CANCELLING"},
    "STARTING": {"RUNNING", "RETRYING", "FAILED", "CANCELLING"},
    "RUNNING": {"SUCCEEDED", "RETRYING", "FAILED", "CANCELLING"},
    "RETRYING": {"STARTING", "FAILED", "CANCELLING"},
    "CANCELLING": {"CANCELLED"},
    "SUCCEEDED": set(),   # 终态
    "FAILED": set(),
    "CANCELLED": set(),
}

def transition(current: str, desired: str) -> str:
    if current in TERMINAL_STATES:
        return current   # 终态不变
    if desired not in VALID_TRANSITIONS[current]:
        raise InvalidTransition(f"{current} -> {desired}")
    return desired
```

#### 3.4 测试

```python
def test_legal_transition(): ...
def test_illegal_transition_raises(): ...
def test_terminal_state_unchanged(): ...
def test_duplicate_event_not_counted(): ...
```

**记录到 `docs/evidence/day16-state-machine.md`**。

### 4. API 与 CLI（第 5 周第 6–7 天）

#### 4.1 HTTP 接口

| 操作 | HTTP | 返回 |
|---|---|---|
| 提交 | `POST /v1/jobs` | `202 Accepted` + job id |
| 查询 | `GET /v1/jobs/{id}` | job 状态 |
| 列表 | `GET /v1/jobs` | job 列表 |
| 取消 | `POST /v1/jobs/{id}/cancel` | `202` |
| 事件 | `GET /v1/jobs/{id}/events` | event 列表 |

#### 4.2 幂等提交

用 `Idempotency-Key` 实现：

```python
@app.post("/v1/jobs", status_code=202)
def submit_job(spec: JobSpec, idempotency_key: str = Header(...)):
    existing = db.get_by_idempotency_key(idempotency_key)
    if existing:
        if existing.spec_hash == hash_spec(spec):
            return existing   # 同 key 同 spec → 返回原任务
        raise HTTPException(409, "Idempotency key reused with different spec")
    # 新任务
    job = create_job(spec, idempotency_key)
    return job
```

#### 4.3 CLI

```bash
trainctl submit examples/cpu-smoke.yaml --idempotency-key smoke-001
trainctl list
trainctl get <JOB_ID>
trainctl cancel <JOB_ID>
trainctl events <JOB_ID>
```

#### 4.4 验证

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
# 另开终端
trainctl submit examples/cpu-smoke.yaml --idempotency-key smoke-001
# 重复同 key → 返回同一 job
# 改 YAML 后同 key → 409
# 重启 API → trainctl get 仍能查到
```

**记录到 `docs/evidence/day17-api-idempotency.md`**。

### 5. Executor 接口与 FakeExecutor（第 5 周第 8 天）

#### 5.1 接口定义

```python
from typing import Protocol

class ExecutionHandle(BaseModel):
    job_id: str
    attempt: int
    pid: int | None = None
    pgid: int | None = None
    log_path: str
    started_at: datetime

class ExecutionStatus(BaseModel):
    phase: str        # RUNNING / SUCCEEDED / FAILED / UNKNOWN
    exit_code: int | None = None
    message: str | None = None

class Executor(Protocol):
    def start(self, job: Job, attempt: int) -> ExecutionHandle: ...
    def inspect(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
```

#### 5.2 FakeExecutor

```python
class FakeExecutor:
    def __init__(self, outcome: str = "succeed"):
        self.outcome = outcome
        self.handles: dict[str, ExecutionStatus] = {}

    def start(self, job, attempt):
        handle = ExecutionHandle(
            job_id=job.id, attempt=attempt,
            log_path=f"/fake/{job.id}/attempt-{attempt}.log",
            started_at=datetime.utcnow(),
        )
        self.handles[handle.job_id] = ExecutionStatus(phase="RUNNING")
        return handle

    def inspect(self, handle):
        return self.handles[handle.job_id]

    def complete(self, handle, outcome: str):
        self.handles[handle.job_id] = ExecutionStatus(
            phase=outcome.upper(),
            exit_code=0 if outcome == "succeed" else 1,
        )

    def cancel(self, handle):
        self.handles[handle.job_id] = ExecutionStatus(phase="CANCELLED")
```

#### 5.3 测试

```python
def test_fake_success(): ...
def test_fake_failure(): ...
def test_fake_hang(): ...
def test_fake_cancel(): ...
```

**记录到 `docs/evidence/day18-fake-executor.md`**。

### 6. Controller 与 LocalProcessExecutor（第 6 周第 1–3 天）

#### 6.1 reconcile 循环

```python
def reconcile_loop(db, executor, interval=2):
    while True:
        for job in db.list_non_terminal_jobs():
            reconcile_one(job, db, executor)
        time.sleep(interval)

def reconcile_one(job, db, executor):
    with db.transaction():   # BEGIN IMMEDIATE
        if job.desired_state == "CANCELLED" and job.status != "CANCELLED":
            handle = db.get_handle(job.id, job.attempt)
            executor.cancel(handle)
            release_allocation(db, job)
            transition_and_event(db, job, "CANCELLED", reason="user cancel")
            return

        if job.status == "PENDING" and all_gpus_free(db, job.spec.gpus):
            allocate_gpus(db, job)
            transition_and_event(db, job, "ADMITTED", reason="gpus available")
            return

        if job.status == "ADMITTED":
            handle = executor.start(job, job.attempt + 1)
            save_handle(db, handle)
            transition_and_event(db, job, "STARTING", reason="process started")
            return

        if job.status == "STARTING":
            status = executor.inspect(db.get_handle(job.id, job.attempt))
            if status.phase == "RUNNING":
                transition_and_event(db, job, "RUNNING", reason="metrics appeared")
            return

        if job.status == "RUNNING":
            status = executor.inspect(db.get_handle(job.id, job.attempt))
            if status.phase == "SUCCEEDED":
                release_allocation(db, job)
                transition_and_event(db, job, "SUCCEEDED", reason="exit 0")
            elif status.phase == "FAILED":
                release_allocation(db, job)
                classification = classify_failure(status.exit_code)
                if classification == "RETRYABLE" and job.attempt < job.spec.retryPolicy.maxRetries:
                    next_retry = now() + backoff(job.attempt)
                    transition_and_event(db, job, "RETRYING",
                                          reason=f"exit {status.exit_code}",
                                          next_retry_at=next_retry)
                else:
                    transition_and_event(db, job, "FAILED",
                                          reason=f"exit {status.exit_code}")
            return

        if job.status == "RETRYING":
            if now() >= job.next_retry_at:
                job.attempt += 1
                transition_and_event(db, job, "STARTING", reason="retry")
            return
```

#### 6.2 LocalProcessExecutor

```python
class LocalProcessExecutor:
    def start(self, job, attempt):
        argv = self._build_argv(job)
        log_path = f"runs/platform/jobs/{job.id}/attempt-{attempt}/trainer.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in job.spec.gpus)
        # 不设置 RANK/WORLD_SIZE —— 由 torchrun 自己管

        log_file = open(log_path, "a")
        proc = subprocess.Popen(
            argv,
            start_new_session=True,     # ← 独立进程组
            cwd="/workspace/GIT/ai_infra/training-platform-mvp",
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        pgid = os.getpgid(proc.pid)

        handle = ExecutionHandle(
            job_id=job.id, attempt=attempt,
            pid=proc.pid, pgid=pgid,
            log_path=log_path,
            started_at=datetime.utcnow(),
        )
        self._save_handle(handle)
        return handle

    def _build_argv(self, job):
        # 白名单校验
        if job.spec.command[0] != "torchrun":
            raise ValueError("only torchrun allowed")
        argv = list(job.spec.command)
        # 根据 gpus 数量生成 --nproc_per_node
        if "--nproc_per_node" not in " ".join(argv):
            argv.insert(1, f"--nproc_per_node={len(job.spec.gpus)}")
        argv.extend(job.spec.args)
        return argv

    def inspect(self, handle):
        try:
            os.kill(handle.pid, 0)   # 检查进程是否存在
            return ExecutionStatus(phase="RUNNING")
        except ProcessLookupError:
            exit_code = self._read_exit_code(handle)
            return ExecutionStatus(
                phase="SUCCEEDED" if exit_code == 0 else "FAILED",
                exit_code=exit_code,
            )

    def cancel(self, handle):
        try:
            os.killpg(handle.pgid, signal.SIGTERM)
            # 等待 10 秒
            for _ in range(10):
                time.sleep(1)
                try:
                    os.killpg(handle.pgid, 0)
                except ProcessLookupError:
                    return
            os.killpg(handle.pgid, signal.SIGKILL)   # 超时才强杀
        except ProcessLookupError:
            pass
```

**关键点**：
- `start_new_session=True`：让子进程成为**新进程组组长**，PGID = PID。
- `os.killpg(pgid, SIGTERM)`：杀**整个进程组**，不是单个 PID。DDP 有多个 rank，杀一个 PID 会留下孤儿进程。
- `CUDA_VISIBLE_DEVICES` 由 executor 显式设置，用户 YAML 不能覆盖。

#### 6.3 验证：重复 reconcile 不重复启动

```python
def test_reconcile_idempotent():
    # 同一个 job 连续 reconcile 两次
    reconcile_one(job, db, executor)
    reconcile_one(job, db, executor)
    # 只应启动一个进程
    assert executor.start_count == 1
```

```python
def test_controller_restart_no_duplicate():
    # 启动 controller，等 job 到 RUNNING
    # 停止 controller
    # 重启 controller
    # 确认没有第二个 torchrun 进程
```

**记录到 `docs/evidence/day19-reconcile-idempotent.md`**。

### 7. 队列与 gang admission（第 6 周第 4–5 天）

#### 7.1 选择纯函数

```python
def select_next_admission(pending_jobs, free_gpus):
    """只看 PENDING，按 priority DESC, created_at ASC, id ASC 排序。"""
    candidates = sorted(
        [j for j in pending_jobs if j.status == "PENDING"],
        key=lambda j: (-j.spec.priority, j.created_at, j.id),
    )
    for job in candidates:
        requested = set(job.spec.gpus)
        if requested.issubset(free_gpus):
            return job
    return None
```

#### 7.2 gang admission

**核心规则：全有或全无。** 任务请求 `[0,1]` 而只空闲 `[0]` 时，必须保持 `PENDING`：

```text
waiting for full GPU allocation: requested=[0,1], available=[0], queue_position=2
```

#### 7.3 原子分配

```python
def allocate_gpus(db, job):
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")   # 排他锁
        free = read_free_gpus(conn)
        if not set(job.spec.gpus).issubset(free):
            conn.rollback()
            return False
        conn.execute(
            "INSERT INTO allocations (job_id, generation, host, gpu_ids_json) "
            "VALUES (?, ?, ?, ?)",
            (job.id, job.generation, job.spec.host, json.dumps(job.spec.gpus)),
        )
        conn.execute("UPDATE jobs SET status = 'ADMITTED' WHERE id = ?", (job.id,))
        conn.commit()
        return True
```

#### 7.4 验证

```bash
python -m controller.hosts register --host gpu200 --gpus 0,1
trainctl submit examples/gpu-two.yaml
trainctl submit examples/gpu-two.yaml --name gpu-two-b
watch -n 1 'trainctl list; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

**验收**：
- 第二个任务保持 `PENDING`，event reason 说明请求 `[0,1]` 而可用 `[0]`。
- 第一个释放后，第二个**一次性**进入 `ADMITTED`。
- 并发 reconcile 测试的 allocation 总数不超过登记 GPU 数。

**记录到 `docs/evidence/day20-gang-admission.md`**。

### 8. Checkpoint、重试与恢复（第 6 周第 6–7 天）

#### 8.1 每 attempt 独立目录

```text
runs/platform/jobs/<job-id>/
  attempt-1/trainer.log
  attempt-1/checkpoints/checkpoint.pt
  attempt-2/trainer.log
  attempt-2/checkpoints/checkpoint.pt
```

#### 8.2 错误分类

| 类别 | 例子 | 动作 |
|---|---|---|
| `INFRASTRUCTURE` | worker 被终止、进程意外消失 | 指数退避重试 |
| `TRANSIENT_IO` | 暂时读写失败 | 指数退避重试 |
| `USER_ERROR` | 参数错误、traceback、退出码 2 | 直接失败 |
| `CANCELLED` | 用户取消 | 不重试 |

```python
def classify_failure(exit_code):
    if exit_code == 2:
        return "USER_ERROR"
    if exit_code in (137, 143):   # SIGKILL / SIGTERM
        return "INFRASTRUCTURE"
    return "UNKNOWN"

def backoff(attempt, base=30, max_backoff=600):
    return min(base * 2 ** (attempt - 1), max_backoff)
```

#### 8.3 恢复验证

```bash
# 确认 attempt-1 已产生 checkpoint
cat run/jobs/<JOB_ID>/attempt-1/handle.json
kill -TERM -<PGID>
watch -n 1 'trainctl get <JOB_ID>; trainctl events <JOB_ID>'
```

**验收**：
- 出现 `RETRYING` 和 `attempt-2`。
- 从最近有效 checkpoint 恢复。
- 比较两次 `metrics.jsonl` 的最大 `global_step`，后一次不得倒退。

**记录到 `docs/evidence/day21-checkpoint-recovery.md`**。

### 9. 可观测性与故障演练（第 6 周第 8–10 天）

#### 9.1 Prometheus 指标

```text
training_jobs{state="PENDING"}                    gauge
training_queue_wait_seconds                         histogram
training_admission_latency_seconds                  histogram
training_attempts_total{outcome="retry|success"}   counter
training_recovery_seconds                            histogram
training_failures_total{classification="..."}      counter
```

**标签只使用有限集合，禁止把 job id 放进 Prometheus label**（基数爆炸）。

#### 9.2 结构化日志

JSON 日志最少包含：

```json
{
  "timestamp": "...",
  "level": "INFO",
  "component": "controller",
  "job_id": "...",
  "attempt": 1,
  "host": "gpu200",
  "gpu_ids": [0, 1],
  "event": "state_transition",
  "reason": "exit 0",
  "exit_code": 0
}
```

#### 9.3 四类故障演练

| 故障 | 安全注入 | 验收 |
|---|---|---|
| worker 消失 | 对保存的 PGID 发 `SIGTERM` | `RETRYING`、新 attempt、step 连续 |
| 通信超时 | 仅在受控多 rank 测试设置短 timeout | 有界失败，不无限 hang |
| 磁盘写入失败 | 测试目录 + 小文件限制 | 旧 checkpoint 可读 |
| 主进程异常 | 固定退出码 `FAIL_AT_STEP` | `USER_ERROR` 且不重试 |

每次只注入一个故障，记录到 `docs/incidents/<date>-<case>.md`：

- 时间、commit、主机/GPU
- 注入命令
- 预期/实际状态
- event/log/metrics 路径
- 恢复耗时、checkpoint step、清理结果

**记录到 `docs/evidence/day22-failure-drills.md`**。

### 10. 最终演示与证据包（第 6 周第 11–14 天）

#### 10.1 演示顺序（固定）

```text
1. 提交两个竞争 GPU 的任务 → 展示排队原因
2. 启动双卡 DDP → 展示 torchrun、GPU 映射、日志
3. 查看 metrics / events / profiler
4. 终止 worker → 展示 RETRYING → attempt-2 → step 连续
5. 取消另一个任务 → 展示 CANCELLING → CANCELLED
6. 讲一份算子/图优化或 DDP 性能报告
```

#### 10.2 最终验收

```bash
make lint
make test
pytest tests/integration -v
nvidia-smi
trainctl list
find docs/evidence docs/incidents -type f | sort
git status --short
```

#### 10.3 交付清单

| 交付物 | 内容 |
|---|---|
| **架构图** | 控制面/执行面边界 |
| **设计文档** | 状态机、幂等、队列/gang、checkpoint 语义 |
| **运行手册** | 部署、清理、常见故障排查 |
| **任务 YAML** | 单卡、双卡、非法三份 |
| **测试报告** | 单元 + 集成 + 故障注入 |
| **性能报告** | 第 2 阶段的 profiler 和优化结果 |
| **故障记录** | 四份 `docs/incidents/*.md` |
| **已知限制** | 无真实多机时明确标注 |

### 11. 面试自测题

1. 为什么 API 返回 `202` 而不是 `200`？两者语义区别是什么？
2. Controller 重启后，如何判断一个 `RUNNING` 的 job 是否真的还在跑？
3. 为什么杀 PID 不等于杀 DDP 作业？PGID 的作用是什么？
4. gang admission 的“全有或全无”为什么必须用 `BEGIN IMMEDIATE` 保护？
5. `Idempotency-Key` 同 key 不同 spec 返回 409，这个设计防住了什么？
6. checkpoint 原子写入为什么用 `os.replace` 而不是直接覆盖？
7. 为什么 Prometheus label 里不能放 job id？
8. 你的平台在 controller 崩溃后如何恢复状态？

### 12. 与第 4 阶段的衔接

第 3 阶段的裸机执行器稳定后，第 4 阶段可以：

- **抽象出 `RemoteProcessExecutor`**：controller 在 101，训练在 200，通过受限 SSH 调用固定 runner 脚本。
- **或迁移到 K8s**：把 `LocalProcessExecutor` 替换为 `KubernetesExecutor`，其余状态机、队列、API 全部复用。

无论哪条，**第 3 阶段的 Executor Protocol、状态机、队列逻辑都不变**——这正是接口抽象的价值。