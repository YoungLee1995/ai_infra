# 第 3 阶段操作手册：训练平台 MVP（裸机路线）

> 周期：第 5-6 个月，每周 15-20 小时
> 环境：101（CPU 控制面开发/测试）+ 200（GPU 真实训练）
> 前置：完成第 1 阶段 DDP trainer 和第 2 阶段性能优化
> 产出：可提交、排队、成组启动、观测、取消、重试和恢复的训练作业平台

本文是按指令执行的手册。先完成裸机版：101 运行控制面，200 运行真实 `torchrun`。每一步完成通过标准并保存证据，未通过不要进入下一步；Kubernetes 留到第 4 阶段。

## 0. 固定环境、路径和边界

101 与 200 都执行：

```bash
export PLATFORM_ROOT=/workspace/GIT/ai_infra/training-platform-mvp
export RUN_ROOT=/workspace/GIT/ai_infra/training-platform-runs
mkdir -p "$PLATFORM_ROOT" "$RUN_ROOT"
```

101 用于 FastAPI、Typer、SQLite、Controller 的开发和 CPU/FakeExecutor 测试。真实 GPU 集成时，将 API、Controller 和 `LocalProcessExecutor` 一起部署到 200，后者再启动同机的 `torchrun`；不能让 101 上的 LocalProcessExecutor 直接管理 200 的进程。跨主机控制要等第 4 阶段实现受限 SSH Executor。`run/` 保存平台运行时（DB、锁、handle），`runs/` 保存训练输出（日志、metrics、checkpoint）。模型和数据复用第 2 阶段本地目录，不提交 Git。

```text
training-platform-mvp/
  api/{main.py,routes.py,cli.py}
  controller/{worker.py,state_machine.py,queue.py,reconciler.py}
  executor/{base.py,fake.py,local.py}
  models/{spec.py,db.py,domain.py}
  examples/{cpu-smoke.yaml,gpu-two.yaml,invalid.yaml}
  tests/{test_spec.py,test_state_machine.py,test_queue.py,test_reconciler.py,test_local_executor.py}
  tests/integration/{test_api.py,test_recovery.py}
  docs/{architecture.md,runbook.md,evidence/,incidents/}
  run/ runs/ pyproject.toml Makefile
```

## Step 1：准备主机和 Python 环境（第 5 周第 1 天）

101：

```bash
cd "$PLATFORM_ROOT"
mkdir -p api controller executor models examples tests/integration docs/evidence docs/incidents run runs
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install fastapi 'uvicorn[standard]' typer pydantic sqlalchemy aiosqlite httpx pytest pytest-asyncio pyyaml prometheus-client structlog
python - <<'PY' | tee docs/evidence/environment-101.txt
import sys, fastapi, pydantic, sqlalchemy
print('python=', sys.version)
print('fastapi=', fastapi.__version__)
print('pydantic=', pydantic.__version__)
print('sqlalchemy=', sqlalchemy.__version__)
PY
```

200（真实集成前先使用已验证的第 2 阶段 CUDA Python 环境）：

```bash
cd "$PLATFORM_ROOT"
source /path/to/phase2/.venv/bin/activate  # 替换为第 2 阶段实际 venv 路径
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee docs/evidence/nvidia-smi-200.txt
python - <<'PY' | tee docs/evidence/environment-200.txt
import torch
print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'devices=', torch.cuda.device_count())
PY
```

**通过标准**：101 能导入依赖；200 `cuda=True` 且 GPU 数量正确。两台机器必须能读取同一代码；若不是共享盘，用 Git 和受限 SSH 同步。

## Step 2：实现任务规格和输入校验（第 5 周第 2 天）

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

## Step 3：数据库、event 和状态机（第 5 周第 3-4 天）

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

## Step 4：API、幂等提交和 CLI（第 5 周第 5-6 天）

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
# 另开 101 终端：重复提交同 key；修改 YAML 后复用 key；提交 invalid.yaml
```

**通过标准**：首次提交 202，重复提交返回同一 job，改 YAML 后 409，非法输入 422，重启 API 后仍可查询 SQLite。输出保存 `docs/evidence/day17-api-idempotency.txt`。

## Step 5：Executor 抽象和 FakeExecutor（第 5 周第 7-8 天）

在 `executor/base.py` 定义 `ExecutionHandle(job_id, attempt, pid, pgid, log_path, started_at)`、`ExecutionStatus(phase, exit_code, message)` 和 `start/inspect/cancel` Protocol。在 `executor/fake.py` 实现不启动真实进程的 FakeExecutor，可编程返回 `RUNNING`、`SUCCEEDED`、`FAILED`、`CANCELLED`。

```bash
pytest -q tests/test_fake_executor.py tests/test_reconciler.py | tee docs/evidence/day18-fake-executor.txt
```

**通过标准**：101 无 GPU 时仍能覆盖 Controller 的成功、失败、hang 和取消分支。

## Step 6：LocalProcessExecutor、torchrun 和 PGID（第 6 周第 1-2 天）

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

## Step 7：Controller reconcile（第 6 周第 2-3 天）

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

## Step 8：队列和 gang admission（第 6 周第 4-5 天）

在 `controller/queue.py` 按 `priority DESC, created_at ASC, id ASC` 选择任务；GPU 必须全有或全无。分配时用 SQLite `BEGIN IMMEDIATE`，在锁内重新读取 free set、插入 allocation、更新状态。

```bash
python -m controller.hosts register --host gpu200 --gpus 0,1
trainctl submit examples/gpu-two.yaml --idempotency-key gpu-a
trainctl submit examples/gpu-two.yaml --idempotency-key gpu-b
watch -n 1 'trainctl list; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

**通过标准**：第一个占 `[0,1]` 时第二个保持 PENDING，reason 写 requested/available；释放后第二个一次性 ADMITTED；并发测试不超卖。保存 `docs/evidence/day20-gang-admission.txt`。

## Step 9：checkpoint、重试和恢复（第 6 周第 6-7 天）

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

## Step 10：指标、日志和故障演练（第 6 周第 8-10 天）

暴露 `training_jobs{state}`、`training_queue_wait_seconds`、`training_admission_latency_seconds`、`training_attempts_total{outcome}`、`training_recovery_seconds`、`training_failures_total{classification}`。label 只能使用有限集合，禁止 job id。JSON 日志至少含 timestamp、component、job_id、attempt、host、gpu_ids、event、reason、exit_code。

每次只注入一种故障：worker SIGTERM、受控 NCCL timeout、测试目录磁盘写失败、固定 `FAIL_AT_STEP` 主进程异常。验收分别为重试恢复、有界失败、旧 checkpoint 可读、USER_ERROR 不重试。每次记录到 `docs/incidents/<date>-<case>.md`，并汇总 `docs/evidence/day22-failure-drills.md`。

## Step 11：最终演示和验收（第 6 周第 11-14 天）

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

## 完成检查

- [ ] 101/200 环境和路径已固定，证据齐全。
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

稳定后只替换 Executor：可以实现 101 到 200 的受限 SSH `RemoteProcessExecutor`，也可以替换为 `KubernetesExecutor`。API、状态机、幂等、队列、事件和恢复语义保持不变。
