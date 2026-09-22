# 第 3 阶段实践方案：训练作业平台 MVP

> 周期：6--8 周，每周 15--20 小时。目标不是复刻 Kubernetes 或商业训练平台，而是交付一个能提交、排队、成组启动、观测、取消、失败恢复训练任务的最小闭环。

## 1. 项目定义

项目名暂定 `training-platform-mvp`。使用 Kubernetes 作为执行平面，Python 作为控制面语言，先复用已有 DDP trainer 作为工作负载。这样可以把重心放在作业语义、调度约束和恢复，而不是再写一遍训练框架。

**范围内：**

- 用户以 YAML/CLI 提交训练任务，声明镜像、命令、节点数、每节点 GPU/NPU 数、优先级、checkpoint 路径和重试策略。
- API/控制器维护持久化任务状态，并创建 Kubernetes Job 或等价工作负载。
- 支持 FIFO 队列 + 优先级；同一任务所需 worker 必须全部获得资源才启动（gang admission）。
- 支持日志、事件、核心指标、取消和从有效 checkpoint 恢复。
- 注入 worker 消失、网络超时、磁盘满、主进程异常四类故障，形成可验证恢复记录。

**明确不做：**多租户计费、复杂抢占/回填、自动扩缩容、完整 Web 控制台、通用工作流编排，以及手写 Kubernetes scheduler。需要多节点 gang 语义时优先接入现成队列/调度能力；若本地环境无法提供，只实现 admission 模拟并清楚标注边界。

## 2. 目标架构

```text
CLI/YAML -> API 服务 -> PostgreSQL/SQLite（任务状态）
                 |-> Controller/Reconciler -> Kubernetes API -> Job/Pod -> DDP trainer
                 |                                      |                  |
                 |                                      +-> Events/Logs    +-> checkpoint PVC/object store
                 |
                 +-> Prometheus metrics -> Grafana
```

控制面负责接收意图、验证、状态机和幂等 reconcile；执行面负责真正跑 Pod。两者不能混在同一个“提交后阻塞等待”的 HTTP 请求中，否则 API 重启或超时会丢失控制逻辑。

## 3. 必须实现的状态机

```text
PENDING -> ADMITTED -> STARTING -> RUNNING -> SUCCEEDED
                    |             |  |         
                    |             |  +-> CANCELLING -> CANCELLED
                    |             +----> RETRYING -> STARTING
                    +------------------> FAILED
```

- `PENDING`：请求已持久化，但未满足资源/队列条件。
- `ADMITTED`：已原子性保留所需资源，尚未确认所有 worker Ready。
- `STARTING`：Kubernetes 工作负载已创建，等待 rendezvous/worker 就绪。
- `RUNNING`：至少一次训练心跳与 worker 健康检查成立。
- `RETRYING`：失败可重试，记录失败原因、尝试次数和 checkpoint 版本。
- 终态 `SUCCEEDED`、`FAILED`、`CANCELLED` 不得被普通 reconcile 改写。

每次 reconcile 都要以 `(job_id, generation)` 或等价幂等键创建资源；控制器重启后从数据库与 Kubernetes 实际状态重建，不依赖内存队列。取消优先写入期望状态，再由 controller 删除/终止执行资源。

## 4. 分周交付

### 第 1 周：环境与最小作业

- 用 kind/minikube 或可用内网集群建立开发环境；确认 GPU/NPU device plugin、StorageClass 和指标端点是否可用。
- 将现有 trainer 封装为容器镜像，运行单 worker Job，挂载 PVC，产出 checkpoint 与 JSONL 指标。
- 交付：`README` 的一条命令运行、镜像版本记录、单 worker 成功截图/日志。

### 第 2 周：任务规格与提交 API

- 定义 `TrainingJob` YAML schema 与 Python 数据模型，拒绝未知字段和非法资源请求。
- 实现 `submit/get/list/cancel` CLI；数据库保存原始 spec、状态、时间戳、失败原因和 retry 次数。
- 交付：三个样例 YAML（单机、双 worker、非法请求）与 API 单元测试。

### 第 3 周：controller 与幂等执行

- 实现轮询或 watch 驱动的 reconciler，创建命名可预测且带 `job-id` label 的 Kubernetes 资源。
- 实现状态迁移、控制器重启恢复和取消语义；给每次操作记录结构化 event。
- 交付：重复提交/reconcile 不产生重复 Pod；重启 controller 后状态正确的集成测试。

### 第 4 周：队列、优先级与 gang admission

- 先实现严格 FIFO，同优先级内按提交时间排序；高优先级可在未 admission 的任务中优先。
- 计算任务所需总卡数。资源不足时保持 `PENDING` 并说明等待原因；资源完整时一次性 admission 并创建 worker group。
- 交付：两份竞争资源的 job，展示队列顺序、等待原因与完整资源到位后同时启动。

### 第 5 周：checkpoint、重试与训练集成

- `RUNNING` job 将 checkpoint 位置写入状态或由约定路径发现；重试前验证 checkpoint 完整性与版本。
- 区分可重试的基础设施错误与不可重试的用户配置/代码错误，设置最大 retry 和指数退避。
- 交付：杀死一个 worker 后生成新的 attempt，并从最近有效 checkpoint 恢复；训练输出可证明 global step 连续。

### 第 6 周：可观测性与故障演练

- Prometheus 指标：队列长度、等待时长、job 状态数量、admission 延迟、attempt 数、恢复耗时、失败原因计数。
- 每个 job event 带 `job_id`、attempt、rank/worker、状态迁移和错误码；日志可按 job 查询。
- 执行下文四类故障，写故障复盘和恢复时间。

### 第 7--8 周：多节点验证与作品集

- 在可用环境运行 2 节点 DDP；没有真实多节点资源时，用本地多 worker 验证控制语义，并将网络性能结论标注为未测。
- 补齐架构图、API 文档、运行手册、测试报告、5--10 分钟演示脚本和一页结果表。
- 增加关键状态机、spec 校验和 controller 幂等性的回归测试。

## 5. 推荐仓库结构

```text
training-platform-mvp/
  api/                 # HTTP/CLI 与 spec 校验
  controller/          # reconcile、队列、状态机
  executor/            # Kubernetes 资源渲染与状态读取
  models/              # 数据库模型与领域对象
  manifests/           # RBAC、CRD（若使用）、监控与样例任务
  tests/               # unit + integration + failure scenarios
  docs/                # 架构、运行手册、故障复盘、性能报告
```

初版优先使用 Kubernetes Job 加 ConfigMap/环境变量传递 rank 信息。真正的多节点 DDP 需要可靠 rendezvous、固定的 worker 数和失败后的 worker group 重建；不要把“分别启动 N 个普通 Job”误称为分布式训练支持。

## 6. 故障注入验收表

| 故障 | 注入方式 | 预期用户状态 | 验收证据 |
| --- | --- | --- | --- |
| worker 消失 | 删除一个运行中的 worker Pod | `RETRYING`，后续新 attempt `RUNNING` 或终态失败 | event、旧/新 Pod UID、重试次数、恢复后的 global step。 |
| 网络超时 | 在测试 namespace 对指定 worker 注入 NetworkPolicy/受控网络故障 | 有界超时后失败或重试，不无限 hang | 通信超时日志、controller 事件、最终状态与耗时。 |
| 磁盘满 | 使用独立临时 volume 写满配额 | 明确 `FAILED` 或可重试的存储错误 | kube event、stderr、磁盘指标；不得损坏旧 checkpoint。 |
| 主进程异常 | trainer 主进程以固定错误码退出 | 区分用户错误与基础设施错误 | 退出码、失败分类、是否重试符合策略。 |

只在隔离的 namespace 和可删除的测试数据上注入故障。每次实验记录开始/结束时间、命令、版本、期望状态、实际状态和清理结果。

## 7. 核心测试清单

- spec：缺少镜像、负资源、GPU/NPU 类型不匹配、checkpoint URI 非法均被拒绝。
- 状态机：非法状态跳转被拒绝，终态保持不变，重复 event 不重复计数。
- 幂等：同一 `job_id/generation` reconcile 两次只产生一个执行资源。
- 队列：资源不足保持 `PENDING`；资源释放后按优先级/FIFO admission。
- 恢复：controller 重启、API 重试和 worker 重建后状态一致。
- 安全：最小 RBAC；用户任务与 controller service account 分离；不在 spec、日志或镜像中放凭据。

## 8. 最终证据包与完成门槛

完成阶段三时，应能在 10 分钟内演示：提交两个资源竞争任务、查看排队原因、启动一个分布式训练任务、查看指标和日志、杀死 worker、自动从 checkpoint 恢复、取消任务并解释最终状态。

提交材料至少包括：

- 一页架构图与控制面/执行面边界。
- 状态机、幂等策略、队列/gang 取舍和 checkpoint 语义的设计文档。
- 可重复部署命令、三份任务 spec、版本锁定方式和清理命令。
- 故障验收表、恢复时间、成功率和已知限制。
- 真实多机结果（若有）或清楚标注“本地控制语义验证，未声称集群性能”。

这套项目最重要的评价标准是语义完整和可复现：一个功能少但能解释失败、恢复和状态一致性的系统，比只有 Web 页面和 YAML 包装的“平台”更能证明训练 Infra 能力。
