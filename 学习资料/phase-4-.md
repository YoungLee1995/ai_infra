## 第 4 阶段操作手册：远程执行、生产打磨与作品集

> 周期：第 7–8 个月，每周 15–20 小时
> 环境：101（控制面）+ 200（GPU 执行面），**无 K8s**
> 前置：完成第 1–3 阶段，裸机平台 MVP 已能在 101 调度 200 上的训练
> 产出：可远程执行的生产级平台 + 多机验证（若有资源）+ 完整作品集

### 0. 本阶段定位

第 3 阶段的 `LocalProcessExecutor` 假设 controller 和训练在**同一台机器**。第 4 阶段要解决：

1. **控制面在 101、训练在 200** 时，如何远程启动、观测、取消？
2. 如何按**生产标准**打磨：压测、混沌、回归、容量估算？
3. 如何把前 3 阶段的代码、数据、复盘整理成**面试作品集**？
4. 如果有第二台 GPU 机器，怎么做**真实多节点**？没有时怎么标注边界？

### 1. 远程执行架构（第 7 周第 1–2 天）

#### 1.1 为什么不用裸 SSH 命令拼接

第 3 阶段的 `LocalProcessExecutor` 用 `subprocess.Popen` 直接起进程。远程版本**不能**简单改成：

```python
# ❌ 危险：把用户 YAML 拼进 SSH 命令
subprocess.run(f"ssh gpu200 'torchrun {user_args}'", shell=True)
```

原因：
- **命令注入**：用户 YAML 里的 `args` 可包含 `; rm -rf /`。
- **环境不可控**：远程 shell 的 PATH、venv、CUDA 环境不确定。
- **无法持久化**：SSH 断开后进程可能被 SIGHUP 杀掉。

#### 1.2 正确做法：受限 SSH + 固定 runner 脚本

```text
101 controller ──SSH──→ 200 runner.py ──subprocess──→ torchrun 进程组
     │                        │
     └─── 传 job JSON ────────┘
     └─── 读 handle JSON ─────┘
```

**规则**：
1. runner 只接收 **JSON**，不接受任意 shell 字符串。
2. runner 内部做**白名单校验**（命令必须是 `torchrun`）。
3. runner 用 `start_new_session=True` 起进程组，写 `handle.json`。
4. controller 通过 SSH 调用 runner 的 `start` / `inspect` / `cancel` 三个子命令。

#### 1.3 远程 runner 脚本

在 200 上建 `executor/remote_runner.py`：

```python
#!/usr/bin/env python3
"""受限远程执行器：只接受 JSON 参数，不做 shell 拼接。"""
import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNNER_ROOT = Path("/workspace/GIT/ai_infra/training-platform-mvp")
RUN_DIR = RUNNER_ROOT / "run" / "remote"
RUNS_DIR = RUNNER_ROOT / "runs" / "platform" / "jobs"

ALLOWED_COMMANDS = {"torchrun"}
FORBIDDEN_ENV = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"}


def cmd_start(args):
    job = json.loads(args.job_json)
    job_id = job["id"]
    attempt = job["attempt"]
    spec = job["spec"]

    # 白名单校验
    if spec["command"][0] not in ALLOWED_COMMANDS:
        raise SystemExit(f"command not allowed: {spec['command'][0]}")

    # 构建 argv
    argv = list(spec["command"])
    if "--nproc_per_node" not in " ".join(argv):
        argv.insert(1, f"--nproc_per_node={len(spec['gpus'])}")
    argv.extend(spec.get("args", []))

    # 受控环境
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in spec["gpus"])
    for key in FORBIDDEN_ENV:
        env.pop(key, None)

    # 输出目录
    attempt_dir = RUNS_DIR / job_id / f"attempt-{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / "trainer.log"

    # 启动进程组
    log_file = open(log_path, "a")
    proc = subprocess.Popen(
        argv,
        start_new_session=True,
        cwd=str(RUNNER_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    pgid = os.getpgid(proc.pid)

    # 持久化 handle
    handle = {
        "job_id": job_id,
        "attempt": attempt,
        "pid": proc.pid,
        "pgid": pgid,
        "log_path": str(log_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "argv": argv,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    handle_path = RUN_DIR / f"{job_id}-attempt-{attempt}.json"
    handle_path.write_text(json.dumps(handle, indent=2))

    print(json.dumps(handle))


def cmd_inspect(args):
    handle = json.loads(Path(args.handle_path).read_text())
    pid = handle["pid"]
    try:
        os.kill(pid, 0)
        print(json.dumps({"phase": "RUNNING"}))
    except ProcessLookupError:
        exit_code = _read_exit_code(handle)
        phase = "SUCCEEDED" if exit_code == 0 else "FAILED"
        print(json.dumps({"phase": phase, "exit_code": exit_code}))


def cmd_cancel(args):
    handle = json.loads(Path(args.handle_path).read_text())
    pgid = handle["pgid"]
    try:
        os.killpg(pgid, signal.SIGTERM)
        for _ in range(10):
            import time; time.sleep(1)
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                print(json.dumps({"cancelled": True}))
                return
        os.killpg(pgid, signal.SIGKILL)
        print(json.dumps({"cancelled": True, "forced": True}))
    except ProcessLookupError:
        print(json.dumps({"cancelled": True, "already_gone": True}))


def _read_exit_code(handle):
    # 简化：从 log 末尾解析，或写一个退出码文件
    exit_file = Path(handle["log_path"]).parent / "exit_code"
    if exit_file.exists():
        return int(exit_file.read_text().strip())
    return 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start")
    p_start.add_argument("--job-json", required=True)
    p_start.set_defaults(func=cmd_start)

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--handle-path", required=True)
    p_inspect.set_defaults(func=cmd_inspect)

    p_cancel = sub.add_parser("cancel")
    p_cancel.add_argument("--handle-path", required=True)
    p_cancel.set_defaults(func=cmd_cancel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

#### 1.4 受限 SSH 配置

在 101 上生成专用密钥：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/platform_runner -N ""
```

在 200 的 `~/.ssh/authorized_keys` 里加**强制命令**：

```text
command="/workspace/GIT/ai_infra/training-platform-mvp/.venv/bin/python /workspace/GIT/ai_infra/training-platform-mvp/executor/remote_runner.py",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAA... platform-runner
```

这样**任何 SSH 连接都会被强制走 runner**，不能执行任意命令。

#### 1.5 RemoteProcessExecutor

在 101 上实现：

```python
class RemoteProcessExecutor:
    def __init__(self, host: str, ssh_key: str, runner_path: str):
        self.host = host
        self.ssh_key = ssh_key
        self.runner_path = runner_path

    def _ssh(self, *remote_args: str) -> str:
        cmd = [
            "ssh", "-i", self.ssh_key,
            "-o", "StrictHostKeyChecking=yes",
            "-o", "BatchMode=yes",
            self.host,
            *remote_args,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ssh failed: {result.stderr}")
        return result.stdout

    def start(self, job, attempt):
        job_json = json.dumps({
            "id": job.id, "attempt": attempt,
            "spec": job.spec.model_dump(),
        })
        # 通过 stdin 传 JSON，避免 shell 拼接
        out = self._ssh(f"--job-json={job_json}", "start")
        handle = json.loads(out)
        return ExecutionHandle(**handle)

    def inspect(self, handle):
        out = self._ssh(f"--handle-path={handle.handle_path}", "inspect")
        return ExecutionStatus(**json.loads(out))

    def cancel(self, handle):
        self._ssh(f"--handle-path={handle.handle_path}", "cancel")
```

**关键**：JSON 通过 SSH 参数或 stdin 传递，**绝不拼进 shell 字符串**。

#### 1.6 测试

```python
def test_remote_executor_start_inspect_cancel():
    # 用 101 上的 localhost SSH 模拟
    executor = RemoteProcessExecutor("localhost", "~/.ssh/platform_runner", ...)
    handle = executor.start(job, attempt=1)
    assert executor.inspect(handle).phase == "RUNNING"
    executor.cancel(handle)
    assert executor.inspect(handle).phase == "CANCELLED"

def test_remote_executor_rejects_shell_injection():
    job.spec.args = ["; rm -rf /"]
    # runner 内部白名单校验应拒绝
    with pytest.raises(RuntimeError):
        executor.start(job, attempt=1)
```

**记录到 `docs/evidence/day23-remote-executor.md`**。

### 2. 状态重建与 controller 重启（第 7 周第 3 天）

#### 2.1 重启后如何重建

controller 重启时，**不能假设内存里的状态**。从 SQLite + 远程 inspect 重建：

```python
def rebuild_state(db, executor):
    for job in db.list_non_terminal_jobs():
        handle = db.get_handle(job.id, job.attempt)
        if handle is None:
            # 没有 handle：可能是 ADMITTED 但还没 start
            continue
        status = executor.inspect(handle)
        if status.phase == "RUNNING" and job.status in ("STARTING", "RUNNING"):
            # 进程还在跑，保持状态
            continue
        if status.phase == "SUCCEEDED":
            transition_and_event(db, job, "SUCCEEDED", reason="rebuilt: exit 0")
        elif status.phase == "FAILED":
            # 走重试或失败逻辑
            ...
```

#### 2.2 验证

```bash
# 启动一个长时间任务
trainctl submit examples/gpu-two.yaml
# 等它到 RUNNING
trainctl get <JOB_ID>

# 停 controller（Ctrl+C）
# 重启 controller
python -m controller.worker --db run/platform.db --interval 2

# 确认：
# 1. 不会产生第二个 torchrun
# 2. job 状态仍然是 RUNNING
# 3. events 里没有重复的 STARTING
```

**记录到 `docs/evidence/day24-controller-restart.md`**。

### 3. 压测与容量估算（第 7 周第 4–5 天）

#### 3.1 并发提交压测

```python
import concurrent.futures

def submit_job(i):
    return httpx.post("http://127.0.0.1:8000/v1/jobs",
                      json=make_spec(f"job-{i}"),
                      headers={"Idempotency-Key": f"load-{i}"})

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
    results = list(ex.map(submit_job, range(1000)))
```

**测什么**：
- API 在 1000 次并发提交下的 P50/P99 延迟
- SQLite 是否出现 `database is locked`
- 是否有重复 job（幂等性是否守住）

#### 3.2 容量估算

| 指标 | 计算 |
|---|---|
| 单 GPU 可并发 job 数 | 1（gang admission 保证全有或全无） |
| N GPU 集群的队列容量 | 受 checkpoint 存储和 DB 连接数限制 |
| 每 job 的 DB 写入 | 状态迁移 ~10 次 + event ~10 次 |
| SQLite WAL 的写入吞吐 | 约 1 万 TPS（单机足够 MVP） |

**记录到 `docs/evidence/day25-load-test.md`**：并发数、延迟分布、错误率、DB 锁情况。

### 4. 混沌故障演练（第 7 周第 6–7 天）

在第 3 阶段四类故障基础上，**远程版本**要加：

| 故障 | 注入 | 验收 |
|---|---|---|
| SSH 连接断开 | 训练中途 kill SSH 客户端 | 训练进程不受影响（`start_new_session`） |
| 200 网络不可达 | 101 上防火墙规则临时阻断 | controller 有界超时，不无限 hang |
| runner 脚本被篡改 | 手动改 runner 权限 | 下次调用失败，有清晰错误 |
| 200 磁盘满 | 写满测试目录 | 旧 checkpoint 可读，新写入失败有分类 |
| 200 重启 | 模拟机器重启 | controller 重建状态，不重复启动 |

每次只注入一个，记录到 `docs/incidents/<date>-<case>.md`。

**记录到 `docs/evidence/day26-chaos-remote.md`**。

### 5. 多机验证（第 8 周第 1–3 天，可选）

#### 5.1 如果没有第二台 GPU 机器

**明确标注**：

```text
已在本机验证：
- 控制面/执行面分离（101 控制，200 执行）
- 远程启动、观测、取消
- controller 重启后状态重建

未验证：
- 真实多节点 DDP 性能
- 跨节点 NCCL 通信效率
- 多节点 gang admission

原因：当前环境只有一台 GPU 主机（200）。
```

**不要把“200 上两个本地 rank”称为多节点**。这是第 3 阶段就强调的边界。

#### 5.2 如果有第二台 GPU 机器

假设有 `gpu201`，与 `gpu200` 同网段：

```bash
# 在 200 上启动 rank 0
MASTER_ADDR=gpu200 MASTER_PORT=29500 \
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \
  --master_addr=gpu200 --master_port=29500 \
  -m ddp_baseline.train --device cuda --amp ...

# 在 201 上启动 rank 1
MASTER_ADDR=gpu200 MASTER_PORT=29500 \
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=2 \
  --master_addr=gpu200 --master_port=29500 \
  -m ddp_baseline.train --device cuda --amp ...
```

**平台侧**：`LocalProcessExecutor` 需要扩展成**同时启动两个节点上的 runner**。这是第 3 阶段接口的第一次真正扩展：

```python
class MultiNodeExecutor:
    def start(self, job, attempt):
        # 对每个 node 调远程 runner
        handles = []
        for node_rank, host in enumerate(job.spec.hosts):
            h = self.remote_executors[host].start(
                job_with_node_rank(job, node_rank, len(job.spec.hosts)),
                attempt,
            )
            handles.append(h)
        return MultiNodeHandle(handles=handles)
```

**注意**：
- 多节点的 `MASTER_ADDR` 必须是 rank 0 所在的节点。
- `--node_rank` 从 0 开始。
- 任一节点失败，整个 job 失败（gang 语义）。

**记录到 `docs/evidence/day27-multinode.md`**（有资源时）。

### 6. 生产标准打磨（第 8 周第 4–6 天）

#### 6.1 结构化日志 + trace

把 JSON 日志升级为带 `trace_id` 的格式，方便跨 101/200 追踪：

```json
{
  "timestamp": "...",
  "trace_id": "job-abc-attempt-1",
  "span": "controller.reconcile",
  "job_id": "abc",
  "attempt": 1,
  "host": "gpu200",
  "event": "state_transition",
  "from": "STARTING",
  "to": "RUNNING"
}
```

#### 6.2 回归测试

```bash
pytest tests/ -v
pytest tests/integration/ -v
pytest tests/failure_scenarios/ -v
```

**必须覆盖**：
- spec 校验（6 类拒绝）
- 状态机（合法边、非法跳转、终态不变、重复 event）
- 幂等（重复 reconcile、重复提交）
- 队列（优先级、FIFO、gang）
- 恢复（controller 重启、worker 被杀、SSH 断开）

#### 6.3 运行手册

`docs/runbook.md` 必须包含：

| 场景 | 命令 |
|---|---|
| 启动 controller | `python -m controller.worker --db run/platform.db --interval 2` |
| 启动 API | `uvicorn api.main:app --host 127.0.0.1 --port 8000` |
| 提交任务 | `trainctl submit examples/gpu-two.yaml` |
| 查看状态 | `trainctl get <ID>` / `trainctl events <ID>` |
| 取消任务 | `trainctl cancel <ID>` |
| 清理 | `rm -rf run/ runs/` |
| 排查“状态不动” | 查 controller 日志、`trainctl events`、`nvidia-smi` |
| 排查“取消不彻底” | 查 PGID、`ps -ef | grep torchrun` |
| 排查“远程失败” | SSH 连通性、runner 权限、handle.json 是否存在 |

### 7. 作品集整理（第 8 周第 7–10 天）

#### 7.1 一页架构图

```text
┌─────────────────────────────────────────────────────────┐
│ 101 (控制面)                                             │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ FastAPI  │→ │ Controller   │→ │ SQLite (状态)    │   │
│  │ + CLI    │  │ + Reconciler │  │ jobs/events/    │   │
│  └──────────┘  └──────┬───────┘  │ allocations     │   │
│                       │           └─────────────────┘   │
│                       │ Executor Protocol                │
│         ┌─────────────┼─────────────┐                   │
│         ↓             ↓             ↓                   │
│  FakeExecutor   LocalProcess   RemoteProcess            │
│  (单元测试)      Executor       Executor                 │
│                  (200 本地)     (101→200 SSH)            │
└─────────────────────────────────────────────────────────┘
                            │
                            ↓ SSH (受限 runner)
┌─────────────────────────────────────────────────────────┐
│ 200 (执行面)                                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │ torchrun 进程组                                    │   │
│  │  ├─ rank 0 (GPU 0)                               │   │
│  │  └─ rank 1 (GPU 1)                               │   │
│  └──────────────────────────────────────────────────┘   │
│  runs/platform/jobs/<job-id>/attempt-N/                  │
│    ├─ trainer.log                                       │
│    └─ checkpoints/checkpoint.pt                         │
└─────────────────────────────────────────────────────────┘
```

#### 7.2 一页性能对照表

| 配置 | step time | samples/s | MFU | 峰值显存 |
|---|---|---|---|---|
| baseline (fp32) | ... | ... | ... | ... |
| + AMP (bf16) | ... | ... | ... | ... |
| + ZeRO-1 | ... | ... | ... | ... |
| + no_sync (accum=4) | ... | ... | ... | ... |
| + 异步 checkpoint | ... | ... | ... | ... |
| + DataLoader 优化 | ... | ... | ... | ... |

**每项数字附**：GPU 型号、PyTorch 版本、commit、profiler 文件路径。

#### 7.3 一页故障复盘

| 故障 | 注入 | 预期 | 实际 | 恢复时间 | 证据 |
|---|---|---|---|---|---|
| worker 消失 | `kill -TERM -PGID` | RETRYING → attempt-2 | ... | ... | `docs/incidents/...` |
| SSH 断开 | kill SSH 客户端 | 训练不受影响 | ... | ... | ... |
| 200 重启 | 模拟重启 | 状态重建 | ... | ... | ... |
| 磁盘满 | 写满测试目录 | FAILED，旧 checkpoint 可读 | ... | ... | ... |

#### 7.4 10 分钟演示脚本

```text
0:00–0:30  介绍架构：101 控制面 / 200 执行面
0:30–2:00  提交两个竞争 GPU 任务 → 展示排队原因
2:00–4:00  启动双卡 DDP → 展示 torchrun、GPU 映射、日志
4:00–5:00  查看 metrics / events / profiler
5:00–7:00  终止 worker → 展示 RETRYING → attempt-2 → step 连续
7:00–8:00  取消另一个任务 → 展示 CANCELLING → CANCELLED
8:00–9:30  讲一份性能优化报告：MFU 从 X% → Y%
9:30–10:00 讲边界：无多机资源时明确标注
```

### 8. 面试证据包（最终版）

按你的总体方案，简历不要写“学习了 Kubernetes / DeepSpeed”，改写成结果：

> 设计并实现裸机训练作业平台 MVP，支持任务提交、队列、gang admission、进程组级取消和从 checkpoint 恢复；通过 AMP + ZeRO-1 + no_sync 将 MFU 从 X% 提升到 Y%，单 job 恢复时间降至 Z 分钟；在 101/200 分离环境下验证远程启动、观测和故障恢复，明确标注未验证多节点性能。

**三份材料**：
1. 一页架构图（上面第 7.1 节）
2. 一页性能对照表（第 7.2 节）
3. 一页故障复盘（第 7.3 节）

**NPU 项目的翻译**：

| NPU 建模语言 | 训练 Infra 语言 |
|---|---|
| 模型适配 | 端到端训练链路 |
| 算子优化 | kernel/graph 性能 |
| 硬件调优 | 集群利用率 |
| 问题定位 | 可观测性和故障闭环 |

### 9. 面试自测题

1. 远程执行器为什么不能直接拼 SSH 命令？受限 runner 防住了什么？
2. `authorized_keys` 里的 `command=` 强制命令，为什么能防止任意命令执行？
3. controller 重启后，如何判断一个 `RUNNING` 的远程 job 是否真的还在跑？
4. SSH 断开后，`start_new_session=True` 为什么能保护训练进程？
5. 没有第二台 GPU 时，哪些结论不能声称？
6. 你的平台在 200 重启后如何恢复？哪些状态从 DB 重建，哪些从远程 inspect？
7. 你的性能优化中，哪一项对端到端吞吐贡献最大？怎么证明？

### 10. 最终交付清单

| 交付物 | 路径 |
|---|---|
| 远程 runner | `executor/remote_runner.py` |
| RemoteProcessExecutor | `executor/remote.py` |
| 架构图 | `docs/architecture.md` |
| 运行手册 | `docs/runbook.md` |
| 性能报告 | `docs/evidence/phase2-report.md` |
| 故障复盘 ×4 | `docs/incidents/*.md` |
| 远程执行证据 | `docs/evidence/day23-remote-executor.md` |
| 压测证据 | `docs/evidence/day25-load-test.md` |
| 混沌证据 | `docs/evidence/day26-chaos-remote.md` |
| 多机证据（若有） | `docs/evidence/day27-multinode.md` |
| 演示脚本 | `docs/demo-script.md` |

### 11. 与总体目标的衔接

第 4 阶段完成后，你手上有：

- **一个可解释的 DDP trainer**（第 1 阶段）
- **一份性能优化报告**（第 2 阶段）
- **一个能远程执行的训练平台**（第 3–4 阶段）
- **可验证的故障恢复记录**（贯穿始终）

这正好对应你总体方案里“面试证据包”的三份材料，也对应“100 万年包现实策略”里**主线档**（70–85 万）的要求：**进入高质量训练 Infra 团队并获得可量化业务 ownership**。

如果第 4 阶段做完仍只有单机结果，优先争取**内部转岗**到真实训练集群团队，把这里的方法论用到生产环境——那才是从 60 万到 100 万最可靠的路径。