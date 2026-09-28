# 第 2 阶段操作手册：Qwen3-8B 性能优化与瓶颈定位

> 周期：第 3-4 个月，每周 15-20 小时
> 环境：200（CUDA GPU）；101 仅用于编辑、CPU 校验和查看结果
> 前置：完成第 1 阶段，理解 `torchrun`、DDP、AMP、梯度累积和 checkpoint
> 基模：`Qwen/Qwen3-8B`
> 产出：可复现训练脚本、原始指标/trace、逐项对照实验和 `phase2-report.md`

本文是按指令执行的手册。命令中的 `PROJECT_ROOT`、`MODEL_DIR`、`DATA_DIR` 可按服务器调整，但同一实验期间不得改变。

## 0. 路线、资源与材料

Qwen3-8B 的 BF16 权重约 16 GB；全参数 AdamW 还需要梯度、master weight、优化器状态和激活，**不适合你的 24 GB 单卡**。你的默认路线应是 Qwen3-8B **4-bit QLoRA + 两卡 DDP**，用它测吞吐、数据加载、梯度累积和 checkpoint。BF16 LoRA只作为一次显存探测，不作为主基线。

| 机器条件 | 默认配置 | 结论边界 |
|---|---|---|
| 1 x 24 GB | 4-bit QLoRA，`seq_len=512`，`batch=1` | 不做 DDP 通信结论 |
| **2 x 24 GB 4090（你的配置）** | **4-bit QLoRA + DDP，`seq_len=512`，`batch=1`，`grad_accum=8`** | **通信只覆盖 LoRA 梯度，不能外推到全参训练** |
| 2 x 48 GB 或以上 | BF16 LoRA + DDP | LoRA 优化器状态很小，ZeRO 通常无收益 |
| 4 x 80 GB 或以上 | 可选 BF16 全参 + ZeRO-2/3 | 需单独写全参结论 |

### 材料下载路径

| 材料 | 下载地址 | 本地路径/用途 |
|---|---|---|
| Qwen3-8B | [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-8B) | `$MODEL_DIR/Qwen3-8B` |
| ultrachat_200k | [数据集卡](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `$DATA_DIR/hf-cache`，固定切片写入 `artifacts/data/ultrachat-2000.jsonl` |
| FineTome-100k（备用） | [数据集卡](https://huggingface.co/datasets/mlabonne/FineTome-100k) | 同上 |
| Transformers | [GitHub](https://github.com/huggingface/transformers) | venv |
| PEFT | [GitHub](https://github.com/huggingface/peft) | venv |
| DeepSpeed（可选） | [GitHub](https://github.com/microsoft/DeepSpeed) | 仅全参 ZeRO 分支 |
| PyTorch Profiler | [官方 recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | trace 存于 `artifacts/traces` |

数据集须先阅读 dataset card 和许可证；模型/数据不要提交 Git。

## Step 1：创建工作区并盘点硬件

```bash
cd /workspace/GIT/ai_infra
export PROJECT_ROOT="$PWD/qwen3_phase2"
export MODEL_DIR=/data/models
export DATA_DIR=/data/datasets
mkdir -p "$PROJECT_ROOT"/{src,configs,artifacts/{data,logs,metrics,traces,checkpoints,reports}}
mkdir -p "$MODEL_DIR" "$DATA_DIR"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
df -h "$MODEL_DIR" "$DATA_DIR" "$PROJECT_ROOT"
```

将 GPU、拓扑和空间记录到：

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv \
  > "$PROJECT_ROOT/artifacts/reports/environment.txt"
python - <<'PY' >> "$PROJECT_ROOT/artifacts/reports/environment.txt"
import platform, torch
print('python=', platform.python_version())
print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
print('nccl=', torch.cuda.nccl.version() if torch.cuda.is_available() else None)
PY
```

**通过标准**：确认模型至少有 25 GB、产物至少有 50 GB 空间。若 `/data` 不存在，先把两个变量改成明确的绝对路径。

## Step 2：创建环境并验证依赖

```bash
cd "$PROJECT_ROOT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install 'torch>=2.4' 'transformers>=4.51.0' 'accelerate>=1.5.0' \
  'datasets>=3.3.0' 'peft>=0.14.0' 'tensorboard>=2.18.0' \
  'safetensors>=0.5.0' 'huggingface_hub[cli]>=0.28.0' 'psutil>=6.1.0'
python - <<'PY'
import torch, transformers, datasets, peft
print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())
print('transformers=', transformers.__version__)
print('qwen3_config=', hasattr(transformers, 'Qwen3Config'))
print('devices=', torch.cuda.device_count())
PY
```

**通过标准**：`cuda=True`、`qwen3_config=True` 且卡数为 2。你的 4090 必须使用 CUDA 版 PyTorch；安装 `bitsandbytes>=0.45.0`。仅在全参 ZeRO 分支安装 `deepspeed>=0.16.0`。

## Step 3：下载模型并固定数据切片

```bash
. "$PROJECT_ROOT/.venv/bin/activate"
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir "$MODEL_DIR/Qwen3-8B" --local-dir-use-symlinks False
du -sh "$MODEL_DIR/Qwen3-8B"
find "$MODEL_DIR/Qwen3-8B" -maxdepth 1 -type f | sort
```

无法访问 Hugging Face 时：

```bash
pip install 'modelscope>=1.24.0'
python - <<'PY'
from modelscope import snapshot_download
import os
snapshot_download('Qwen/Qwen3-8B', local_dir=os.environ['MODEL_DIR'] + '/Qwen3-8B')
PY
```

下载并固定 2,000 条训练样本：

```bash
export HF_HOME="$DATA_DIR/hf-cache"
export HF_DATASETS_CACHE="$DATA_DIR/hf-cache/datasets"
python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import os
out = Path(os.environ['PROJECT_ROOT']) / 'artifacts/data/ultrachat-2000.jsonl'
out.parent.mkdir(parents=True, exist_ok=True)
ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft[:2000]')
ds.to_json(str(out))
print(len(ds), out)
PY
wc -l "$PROJECT_ROOT/artifacts/data/ultrachat-2000.jsonl"
```

**通过标准**：模型有 `config.json`、tokenizer 和 `*.safetensors`；数据正好 2,000 行。数据集字段/revision 变化时使用 FineTome，并在报告记录实际 URL/revision。

## Step 4：实现 Qwen3 LoRA 训练脚本

创建 `$PROJECT_ROOT/src/train_qwen3_lora.py`，并先完成：

```bash
cd "$PROJECT_ROOT"
. .venv/bin/activate
python - <<'PY'
from transformers import AutoTokenizer, AutoConfig
import os
p = os.environ['MODEL_DIR'] + '/Qwen3-8B'
print(AutoConfig.from_pretrained(p, local_files_only=True).model_type)
print(AutoTokenizer.from_pretrained(p, local_files_only=True).eos_token)
PY
```

脚本必须满足以下接口和语义：

1. `--model-path` 只从本地加载；使用 `AutoTokenizer`、`AutoModelForCausalLM`、`use_cache=False`。
2. 用 `apply_chat_template(..., add_generation_prompt=False)` 格式化数据；非 assistant token 的 label 设为 `-100`。
3. 默认 4-bit QLoRA：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`，再挂载 LoRA；`r=16`、`lora_alpha=32`、`lora_dropout=0.05`，target modules 为 `q/k/v/o/gate/up/down_proj`。保留 `--precision bf16` 作为显存探测分支。
4. 支持 `--steps`、`--warmup-steps`、`--batch-size`、`--seq-len`、`--grad-accum`、`--num-workers`、`--pin-memory`、`--persistent-workers`、`--profile`、`--checkpoint-every`、`--async-checkpoint`。
5. 每个 optimizer step 记录 `step_time_ms`、`tokens_per_second`、loss、峰值显存到 JSONL；只由 rank 0 写文件。
6. DDP 使用 `DistributedSampler.set_epoch()`；除最后一个 micro-batch 外使用 `model.no_sync()`，loss 必须除以 `grad_accum`。
7. checkpoint 至少保存 LoRA 权重、optimizer、scheduler、AMP scaler、global step、配置和随机数状态。

```bash
python -m py_compile "$PROJECT_ROOT/src/train_qwen3_lora.py"
```

不要把第 1 阶段的 MLP 结果当作 Qwen3 证据；可复用其 DDP/checkpoint 控制流，但 tokenizer、causal loss、序列长度和显存必须由本脚本实测。

## Step 5：单卡 QLoRA 冒烟

```bash
export GPU_ID=0
cd "$PROJECT_ROOT"
. .venv/bin/activate
CUDA_VISIBLE_DEVICES="$GPU_ID" python src/train_qwen3_lora.py \
  --model-path "$MODEL_DIR/Qwen3-8B" \
  --data-path artifacts/data/ultrachat-2000.jsonl --output-dir artifacts/smoke \
  --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bf16 \
  --batch-size 1 --seq-len 512 --grad-accum 1 --gradient-checkpointing \
  --steps 8 --warmup-steps 2 --num-workers 2 --pin-memory
```

**通过标准**：8 step 无 OOM、loss 有限、`metrics.jsonl` 有 8 行。OOM 时只按顺序改一个变量：`seq_len 512 -> 256`，再降低 `grad_accum`（不改变单卡显存），不要同时改多个变量。

## Step 6：建立 baseline 和 profiler 证据

固定模型目录、数据文件、seed、序列长度、global batch、GPU 数和软件版本。你的主 baseline 使用两张 4090、QLoRA、`batch=1`、`grad_accum=8`：

```bash
CUDA_VISIBLE_DEVICES="0,1" torchrun --standalone --nproc_per_node=2 src/train_qwen3_lora.py \
  --model-path "$MODEL_DIR/Qwen3-8B" \
  --data-path artifacts/data/ultrachat-2000.jsonl --output-dir artifacts/baseline \
  --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bf16 \
  --batch-size 1 --seq-len 512 --grad-accum 8 --gradient-checkpointing \
  --steps 40 --warmup-steps 10 --num-workers 2 --pin-memory \
  --profile --profile-wait 5 --profile-warmup 5 --profile-active 20
```

profiler 应使用 `CPU + CUDA`、`wait=5,warmup=5,active=20`、`record_shapes=True`、`profile_memory=True`，trace 写到 `artifacts/traces`。查看：

```bash
tensorboard --logdir "$PROJECT_ROOT/artifacts/baseline" --port 6006
```

记录 `artifacts/reports/baseline.md`：完整命令、GPU、配置、跳过前 10 step 后的 median `step_time_ms`/`tokens_per_second`/峰值显存、trace 和时间线截图。观察 GPU 空洞、H2D copy、backward 和（多卡时）NCCL 重叠。

为后续对照实验固定公共参数（在 `PROJECT_ROOT` 目录执行）：

```bash
export TRAIN_COMMON=(--model-path "$MODEL_DIR/Qwen3-8B"
  --data-path artifacts/data/ultrachat-2000.jsonl
  --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bf16
  --batch-size 1 --seq-len 512 --grad-accum 8 --gradient-checkpointing --warmup-steps 10
  --steps 40)
```

LoRA 的“可训练参数 MFU”没有代表性；报告以 tokens/s 和 step time 为主。若估算 dense-equivalent MFU，必须写出参数量、FLOPs 公式、峰值来源和“估算”字样。

## Step 7：数据加载对照（一次只改一个变量）

复制 Step 6 的全部参数，分别执行：

```bash
# A
CUDA_VISIBLE_DEVICES="$GPU_ID" python src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/dataloader-a --num-workers 0 --no-pin-memory
# B
CUDA_VISIBLE_DEVICES="$GPU_ID" python src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/dataloader-b --num-workers 0 --pin-memory
# C
CUDA_VISIBLE_DEVICES="$GPU_ID" python src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/dataloader-c --num-workers 2 --pin-memory --persistent-workers
```

报告 `artifacts/reports/day13-dataloader.md`：warmup 后 median step time、tokens/s、profiler 判断。GPU 已经连续满载时，DataLoader 无收益是有效结论。

## Step 8：两卡 DDP 与 `no_sync()`

只有两张卡可用时执行；单卡在最终报告标注“未测通信”。

```bash
export GPU_IDS=0,1
CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun --standalone --nproc_per_node=2 \
  src/train_qwen3_lora.py --model-path "$MODEL_DIR/Qwen3-8B" \
  --data-path artifacts/data/ultrachat-2000.jsonl --output-dir artifacts/ddp-smoke \
  --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bf16 \
  --batch-size 1 --seq-len 512 --grad-accum 8 --gradient-checkpointing \
  --steps 8 --warmup-steps 2 --num-workers 2 --pin-memory
```

通过标准：两个 rank 启动、rank 0 写 metrics、无 NCCL timeout。然后固定 global batch，比较 `grad_accum=4` 的同步和 `no_sync`：

```bash
# 同步每个 micro-batch
CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun --standalone --nproc_per_node=2 \
  src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" --output-dir artifacts/ddp-sync --grad-accum 4 --disable-no-sync
# 最后一个 micro-batch 才同步
CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun --standalone --nproc_per_node=2 \
  src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" --output-dir artifacts/ddp-no-sync --grad-accum 4
```

报告 `artifacts/reports/day12-comm-overlap.md`：两组稳定段指标、`nccl:all_reduce` 与 backward 的重叠、trace 路径。LoRA 通信量很小导致收益不明显时，要如实记录。

## Step 9：显存分支与 BF16 探测

你的主路线已经是 QLoRA。若想知道 BF16 LoRA 是否能在 24GB 卡上运行，可单独做一次探测；失败是预期结果：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python src/train_qwen3_lora.py \
  --model-path "$MODEL_DIR/Qwen3-8B" --data-path artifacts/data/ultrachat-2000.jsonl \
  --output-dir artifacts/bf16-probe --precision bf16 --batch-size 1 --seq-len 256 \
  --grad-accum 8 --gradient-checkpointing --steps 2
```

若 BF16 探测 OOM，继续使用 Step 5/6 的 QLoRA 配置。QLoRA 命令如下：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/qlora --load-in-4bit --bnb-4bit-quant-type nf4 \
  --bnb-4bit-compute-dtype bf16 --gradient-checkpointing
```

记录显存和 step time。QLoRA 改变了数值精度，不能和 BF16 baseline 直接声称性能归因。

## Step 10：同步/异步 checkpoint

使用相同配置跑 100 optimizer steps、每 20 step 保存：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/checkpoint-sync --steps 100 --checkpoint-every 20
CUDA_VISIBLE_DEVICES="$GPU_ID" python src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/checkpoint-async --steps 100 --checkpoint-every 20 --async-checkpoint
```

异步实现必须在启动下一次保存前和训练结束前调用 `pending_save.result()`；否则可能内存堆积或留下不完整 checkpoint。记录每次保存时长、总 wall time 和恢复验证，写入 `artifacts/reports/day14-async-checkpoint.md`。

## Step 11：最终报告与材料校验

实验矩阵只填写实际运行的结果：A baseline；B 数据加载；C checkpointing/QLoRA（如需要）；D 两卡 `no_sync`；E async checkpoint。每组记录 median step time、tokens/s、峰值显存、trace；D 额外记录 NCCL overlap。

`artifacts/reports/phase2-report.md` 必须包含：

1. GPU/互连、驱动、CUDA、PyTorch、Transformers、PEFT、DeepSpeed（若用）、Git commit。
2. 模型 URL/revision、本地路径、数据 URL/revision、切片脚本和 SHA256。
3. 每条完整命令和 `per_device_batch x world_size x grad_accum` 的 global batch。
4. warmup 舍弃规则、统计口径、trace/截图路径。
5. 相邻实验的变化百分比：`(new_tokens_s / old_tokens_s - 1) * 100%`。
6. 限制：LoRA 不代表全参、单卡无通信、量化/序列长度改变不能直接归因。

```bash
sha256sum "$PROJECT_ROOT/artifacts/data/ultrachat-2000.jsonl" \
  > "$PROJECT_ROOT/artifacts/reports/data.sha256"
git -C /workspace/GIT/ai_infra rev-parse HEAD \
  >> "$PROJECT_ROOT/artifacts/reports/environment.txt"
```

## Step 12（可选）：全参数 Qwen3-8B + ZeRO

只在约 `4 x 80 GB` 级别资源上做。先 BF16 全参 baseline，再分别 ZeRO-2、ZeRO-3，保持数据、序列长度、global batch 和步数不变。ZeRO-1 分片优化器状态，ZeRO-2 再分片梯度，ZeRO-3 再分片参数；LoRA 的优化器状态很小，不能拿 LoRA 的 ZeRO 结果当性能演示。将结果写到 `day10-zero-comparison.md`，包含峰值显存、step time、通信 trace 和可增大的 batch。

## 完成检查

- [ ] Qwen3-8B 本地加载且 8 step 冒烟成功。
- [ ] 有 40 step baseline 的 JSONL、配置、trace 和截图。
- [ ] 数据加载至少完成两组单变量对照。
- [ ] 有两卡时完成 DDP + `no_sync`；无两卡时明确标注未测。
- [ ] 同步/异步 checkpoint 均验证可恢复。
- [ ] `phase2-report.md` 可由另一人按路径、版本、命令和校验值复跑。

## 面试自测

1. 为什么 Qwen3-8B LoRA 不能代表全参数 ZeRO？
2. DDP AllReduce 在 backward 的哪一刻启动？bucket 大小如何影响重叠？
3. `grad_accum=4` 时，为什么 `no_sync()` 数学等价却可能更快？
4. GPU 利用率、tokens/s、MFU 分别回答什么问题？
5. activation checkpointing 用什么换取什么？
6. 异步 checkpoint 为什么仍需 `future.result()`？
