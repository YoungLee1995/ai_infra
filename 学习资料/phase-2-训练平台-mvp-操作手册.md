# 第 2 阶段操作手册：Qwen3-8B 性能优化与瓶颈定位（ModelScope 版）

> 周期：第 3-4 个月，每周 15-20 小时
> 环境：200（CUDA GPU）；101 仅用于编辑、CPU 校验和查看结果
> 前置：完成第 1 阶段，理解 `torchrun`、DDP、AMP、梯度累积和 checkpoint
> 基模：`Qwen/Qwen3-8B`
> 产出：可复现训练脚本、原始指标/trace、逐项对照实验和 `phase2-report.md`
> 本版本：**全程不依赖 HuggingFace 在线服务**，模型与数据集均通过 ModelScope 获取

本文是按指令执行的手册。命令中的 `PROJECT_ROOT`、`MODEL_DIR`、`DATA_DIR` 可按服务器调整，但同一实验期间不得改变。

## 0. 路线、资源与材料

Qwen3-8B 的 BF16 权重约 16 GB；全参数 AdamW 还需要梯度、master weight、优化器状态和激活，**不适合你的 24 GB 单卡**。你的默认路线应是 Qwen3-8B **4-bit QLoRA + 两卡 DDP**，用它测吞吐、数据加载、梯度累积和 checkpoint。BF16 LoRA 只作为一次显存探测，不作为主基线。

| 机器条件 | 默认配置 | 结论边界 |
|---|---|---|
| 1 x 24 GB | 4-bit QLoRA，`seq_len=512`，`batch=1` | 不做 DDP 通信结论 |
| **2 x 24 GB 4090（你的配置）** | **4-bit QLoRA + DDP，`seq_len=512`，`batch=1`，`grad_accum=8`** | **通信只覆盖 LoRA 梯度，不能外推到全参训练** |
| 2 x 48 GB 或以上 | BF16 LoRA + DDP | LoRA 优化器状态很小，ZeRO 通常无收益 |
| 4 x 80 GB 或以上 | 可选 BF16 全参 + ZeRO-2/3 | 需单独写全参结论 |

### 材料下载路径

| 材料 | 来源 | 本地路径/用途 |
|---|---|---|
| Qwen3-8B | ModelScope: `Qwen/Qwen3-8B` | `$MODEL_DIR/Qwen3-8B` |
| ultrachat_200k | ModelScope: `AI-ModelScope/ultrachat_200k`（若 ID 不存在，去 modelscope.cn 搜 `ultrachat` 取实际 ID） | `$DATA_DIR/ms-cache/ultrachat_200k`，固定切片写入 `artifacts/data/ultrachat-2000.jsonl` |
| FineTome-100k（备用） | ModelScope 搜 `FineTome-100k` 镜像 | 同上 |
| Transformers | PyPI | venv |
| PEFT | PyPI | venv |
| bitsandbytes | PyPI（QLoRA 必需） | venv |
| DeepSpeed（可选） | PyPI | 仅全参 ZeRO 分支 |
| PyTorch Profiler | 官方 recipe | trace 存于 `artifacts/traces` |

数据集须先阅读 dataset card 和许可证；模型/数据不要提交 Git。**全程不访问 huggingface.co。**

## Step 1：创建工作区并盘点硬件

```bash
cd /workspace/GIT/ai_infra
# ---- qwen3_phase2 project ----
export PROJECT_ROOT="$HOME/qwen3_phase2"
export MODEL_DIR="$PROJECT_ROOT/data/models"
export DATA_DIR="$PROJECT_ROOT/data/datasets"
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

python -m pip install --upgrade pip
pip install 'torch>=2.4' 'transformers>=4.51.0' 'accelerate>=1.5.0' \
  'datasets>=3.3.0' 'peft>=0.14.0' 'tensorboard>=2.18.0' \
  'safetensors>=0.5.0' 'modelscope>=1.24.0' 'psutil>=6.1.0'
pip install 'bitsandbytes>=0.45.0'
python - <<'PY'
import torch, transformers, datasets, peft, modelscope
print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())
print('transformers=', transformers.__version__)
print('qwen3_config=', hasattr(transformers, 'Qwen3Config'))
print('devices=', torch.cuda.device_count())
print('modelscope=', modelscope.__version__)
PY
```

**通过标准**：`cuda=True`、`qwen3_config=True` 且卡数为 2。你的 4090 必须使用 CUDA 版 PyTorch；QLoRA 需要 `bitsandbytes>=0.45.0`。仅在全参 ZeRO 分支安装 `deepspeed>=0.16.0`。

> 说明：`transformers`、`datasets`、`peft` 仍从 PyPI 安装，但**不再调用 HuggingFace 在线服务**。`modelscope` 替代 `huggingface_hub[cli]` 负责模型/数据集下载。

## Step 3：下载模型并固定数据切片（ModelScope）

```bash
. "$PROJECT_ROOT/.venv/bin/activate"
```

### 3.1 下载模型

```bash
modelscope download --model Qwen/Qwen3-8B \
  --local_dir "$MODEL_DIR/Qwen3-8B"

du -sh "$MODEL_DIR/Qwen3-8B"
find "$MODEL_DIR/Qwen3-8B" -maxdepth 1 -type f | sort
```

核对文件完整性：

```bash
python - <<'PY'
import os
p = os.path.join(os.environ['MODEL_DIR'], 'Qwen3-8B')
need = ['config.json', 'tokenizer_config.json']
files = os.listdir(p)
for n in need:
    print(n, n in files)
print('safetensors:', [f for f in files if f.endswith('.safetensors')])
PY
```

### 3.2 下载数据集

```bash
modelscope download --dataset AI-ModelScope/ultrachat_200k \
  --local_dir "$DATA_DIR/ms-cache/ultrachat_200k"
```

如果提示找不到该数据集 ID，去 https://modelscope.cn/datasets 搜索 `ultrachat`，用实际 ID 替换。备用数据集：

```bash
# FineTome-100k 备用（若 ultrachat 镜像不可用）
modelscope download --dataset AI-ModelScope/FineTome-100k \
  --local_dir "$DATA_DIR/ms-cache/FineTome-100k"
```

### 3.3 从本地文件里抽出 2,000 条，存成训练用的 jsonl

python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import os, glob, re, ast, json

data_root = os.path.join(os.environ['DATA_DIR'], 'ms-cache/ultrachat_200k')
cands = sorted(glob.glob(os.path.join(data_root, 'data', 'train_sft-*.csv')))

out = Path(os.environ['PROJECT_ROOT']) / 'artifacts/data/ultrachat-2000.jsonl'
out.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset('csv', data_files=cands, split='train')

def parse_one(m):
    if not isinstance(m, str):
        return m
    # 1) 先直接试
    for f in (json.loads, ast.literal_eval):
        try:
            return f(m)
        except Exception:
            pass
    # 2) 修复 pprint 风格：在 } 和 { 之间缺逗号的地方补逗号
    fixed = re.sub(r"\}\s*\n\s*\{", "},\n{", m)   # }换行{ -> },\n{
    fixed = re.sub(r"\}\s+\{", "}, {", fixed)      # } 空格 { -> }, {
    # 3) 把单引号 JSON 尝试转双引号
    for candidate in (fixed, re.sub(r"(?<!\\)'", '"', fixed)):
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None

kept, bad = [], 0
bad_samples = []
for ex in ds:
    m = parse_one(ex['messages'])
    if m is None:
        bad += 1
        if len(bad_samples) < 2:
            bad_samples.append(ex['messages'][:300])
        continue
    kept.append({'messages': m})
    if len(kept) >= 2000:
        break

print('kept=', len(kept), 'bad skipped=', bad)
for s in bad_samples:
    print('--- still bad:', repr(s))

with out.open('w', encoding='utf-8') as f:
    for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
print(out)
PY

wc -l "$PROJECT_ROOT/artifacts/data/ultrachat-2000.jsonl"

> 关键点：不再使用 `HF_ENDPOINT` / `HF_HOME` / `HF_DATASETS_CACHE`，也不再调用 `load_dataset('HuggingFaceH4/ultrachat_200k', ...)`。改为从本地 parquet 文件加载，切片用 `train[:2000]`。

**通过标准**：模型有 `config.json`、tokenizer 和 `*.safetensors`；数据正好 2,000 行。数据集字段/revision 变化时改用 FineTome，并在报告记录实际 ModelScope 数据集 ID 和本地路径。

## Step 4：实现 Qwen3 LoRA 训练脚本

创建 `$PROJECT_ROOT/src/train_qwen3_lora.py`，并先完成本地校验：

```bash
cd "$PROJECT_ROOT"
python - <<'PY'
from transformers import AutoTokenizer, AutoConfig
import os
p = os.environ['MODEL_DIR'] + '/Qwen3-8B'
print(AutoConfig.from_pretrained(p, local_files_only=True).model_type)
print(AutoTokenizer.from_pretrained(p, local_files_only=True).eos_token)
PY
```

脚本必须满足以下接口和语义：

1. `--model-path` 只从本地加载；所有 `from_pretrained` 必须带 `local_files_only=True`。使用 `AutoTokenizer`、`AutoModelForCausalLM`、`use_cache=False`。
2. 用 `apply_chat_template(..., add_generation_prompt=False)` 格式化数据；非 assistant token 的 label 设为 `-100`。
3. 默认 4-bit QLoRA：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`，再挂载 LoRA；`r=16`、`lora_alpha=32`、`lora_dropout=0.05`，target modules 为 `q/k/v/o/gate/up/down_proj`。保留 `--precision bf16` 作为显存探测分支。
4. 支持 `--steps`、`--warmup-steps`、`--batch-size`、`--seq-len`、`--grad-accum`、`--num-workers`、`--pin-memory`、`--persistent-workers`、`--profile`、`--checkpoint-every`、`--async-checkpoint`。
5. 每个 optimizer step 记录 `step_time_ms`、`tokens_per_second`、loss、峰值显存到 JSONL；只由 rank 0 写文件。
6. DDP 使用 `DistributedSampler.set_epoch()`；除最后一个 micro-batch 外使用 `model.no_sync()`，loss 必须除以 `grad_accum`。
7. checkpoint 至少保存 LoRA 权重、optimizer、scheduler、AMP scaler、global step、配置和随机数状态。

数据侧直接读取 `artifacts/data/ultrachat-2000.jsonl`，**不使用 datasets 在线流式**。

```bash
python -m py_compile "$PROJECT_ROOT/src/train_qwen3_lora.py"
```

不要把第 1 阶段的 MLP 结果当作 Qwen3 证据；可复用其 DDP/checkpoint 控制流，但 tokenizer、causal loss、序列长度和显存必须由本脚本实测。

## Step 5：单卡 QLoRA 冒烟

```bash
export GPU_ID=0
cd "$PROJECT_ROOT"

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
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.run \
  --standalone --nproc_per_node=2 src/train_qwen3_lora.py \
  --model-path "$MODEL_DIR/Qwen3-8B" \
  --data-path artifacts/data/ultrachat-2000.jsonl --output-dir artifacts/baseline \
  --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bf16 \
  --batch-size 1 --seq-len 512 --grad-accum 8 --gradient-checkpointing \
  --steps 40 --warmup-steps 10 --num-workers 2 --pin-memory \
  --tensorboard
```

profiler 应使用 `CPU + CUDA`、`wait=5,warmup=5,active=20`、`record_shapes=True`、`profile_memory=True`，trace 写到 `artifacts/traces`。查看：

```bash
tensorboard --logdir "$PROJECT_ROOT/artifacts/baseline" --port 6006
```

或者使用

cd /workspace/GIT/ai_infra/qwen3_phase2
python - <<'PY'
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open("artifacts/baseline/metrics.jsonl")]
steps = [r["step"] for r in rows]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes[0,0].plot(steps, [r["loss"] for r in rows], marker="o"); axes[0,0].set_title("loss")
axes[0,1].plot(steps, [r["tokens_per_second"] for r in rows], marker="o"); axes[0,1].set_title("tokens/s")
axes[1,0].plot(steps, [r["step_time_ms"] for r in rows], marker="o"); axes[1,0].set_title("step_time_ms")
axes[1,1].plot(steps, [r["peak_mem_gb"] for r in rows], marker="o"); axes[1,1].set_title("peak_mem_gb")
for ax in axes.ravel():
    ax.grid(True); ax.set_xlabel("step")
plt.tight_layout()
plt.savefig("baseline_metrics.png", dpi=120)
print("saved baseline_metrics.png")
PY
直接生成图片

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

#复制 Step 6 的全部参数，分别执行：
cd /workspace/GIT/ai_infra/qwen3_phase2
export GPU_ID=0
export PROJECT_ROOT=/workspace/GIT/ai_infra/qwen3_phase2
export MODEL_DIR="$PROJECT_ROOT/data/models"
export DATA_DIR="$PROJECT_ROOT/data/datasets" 

export TRAIN_COMMON=(
  --model-path "/workspace/GIT/ai_infra/qwen3_phase2/data/models/Qwen3-8B"
  --data-path "/workspace/GIT/ai_infra/qwen3_phase2/artifacts/data/ultrachat-2000.jsonl"
  --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bf16
  --batch-size 1 --seq-len 512 --grad-accum 8 --gradient-checkpointing --warmup-steps 10
  --steps 40
)

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
生成对比图
cd /workspace/GIT/ai_infra/qwen3_phase2
python - <<'PY'
import json, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = {
    "A": ("artifacts/dataloader-a/metrics.jsonl", "nw=0, pin=off"),
    "B": ("artifacts/dataloader-b/metrics.jsonl", "nw=0, pin=on"),
    "C": ("artifacts/dataloader-c/metrics.jsonl", "nw=2, pin=on, persistent"),
}
WARMUP = 10

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
metrics = ["step_time_ms", "tokens_per_second", "peak_mem_gb"]
titles  = ["step_time_ms", "tokens/s", "peak_mem_gb"]

summary = {}
for name, (path, desc) in GROUPS.items():
    rows = sorted([json.loads(l) for l in open(path)], key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    summary[name] = {
        "desc": desc,
        "median": {
            m: statistics.median(r[m] for r in rows if r["step"] > WARMUP)
            for m in metrics
        },
    }
    for ax, m in zip(axes, metrics):
        ax.plot(steps, [r[m] for r in rows], marker="o",
                markersize=3, linewidth=1, label=f"{name} ({desc})")

for ax, m, t in zip(axes, metrics, titles):
    ax.axvline(WARMUP, color="gray", linestyle="--", linewidth=1)
    ax.set_title(t)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("dataloader_abc_compare.png", dpi=120)
print("saved dataloader_abc_compare.png")

print("\n跳过前 10 step 的 median：")
print(f"{'组':<4}{'配置':<32}{'step_time_ms':>14}{'tokens/s':>12}{'peak_mem_gb':>14}")
for name, s in summary.items():
    m = s["median"]
    print(f"{name:<4}{s['desc']:<32}{m['step_time_ms']:>14.2f}"
          f"{m['tokens_per_second']:>12.2f}{m['peak_mem_gb']:>14.3f}")
PY
报告 `artifacts/reports/day13-dataloader.md`：warmup 后 median step time、tokens/s、profiler 判断。GPU 已经连续满载时，DataLoader 无收益是有效结论。

## Step 8：两卡 DDP 与 `no_sync()`

只有两张卡可用时执行；单卡在最终报告标注“未测通信”。

### 8.1 准备

```bash
cd /workspace/GIT/ai_infra/qwen3_phase2
source /workspace/GIT/ai_infra/.venv/bin/activate
export GPU_IDS=0,1
```

### 8.2 DDP smoke（通过标准：两个 rank 启动、rank 0 写 metrics、无 NCCL timeout）

```bash
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  src/train_qwen3_lora.py --model-path "$MODEL_DIR/Qwen3-8B" \
  --data-path artifacts/data/ultrachat-2000.jsonl --output-dir artifacts/ddp-smoke \
  --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bf16 \
  --batch-size 1 --seq-len 512 --grad-accum 8 --gradient-checkpointing \
  --steps 8 --warmup-steps 2 --num-workers 2 --pin-memory
```

验证：

```bash
cat artifacts/ddp-smoke/metrics.jsonl   # 应有 8 行
```

### 8.3 固定 global batch，比较 sync 与 no_sync

`TRAIN_COMMON` 不含 `--grad-accum` / `--output-dir` / `--num-workers` / `--pin-memory` / `--profile`：

```bash
export TRAIN_COMMON=(
  --model-path "/workspace/GIT/ai_infra/qwen3_phase2/data/models/Qwen3-8B"
  --data-path "/workspace/GIT/ai_infra/qwen3_phase2/artifacts/data/ultrachat-2000.jsonl"
  --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bf16
  --batch-size 1 --seq-len 512 --gradient-checkpointing --warmup-steps 10
  --steps 40 --num-workers 2 --pin-memory
)
```

global batch 固定为 `1 × 4 × 2 = 8`。

```bash
# 同步每个 micro-batch
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/ddp-sync --grad-accum 4 --disable-no-sync

# 最后一个 micro-batch 才同步
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/ddp-no-sync --grad-accum 4
```

### 8.4 单独短跑拿 trace

```bash
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/ddp-sync-prof --grad-accum 4 --disable-no-sync \
  --steps 15 --profile --profile-wait 1 --profile-warmup 2 --profile-active 8

CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  src/train_qwen3_lora.py "${TRAIN_COMMON[@]}" \
  --output-dir artifacts/ddp-no-sync-prof --grad-accum 4 \
  --steps 15 --profile --profile-wait 1 --profile-warmup 2 --profile-active 8
```

trace 路径：

- `artifacts/ddp-sync-prof/traces/trace.json`
- `artifacts/ddp-no-sync-prof/traces/trace.json`

### 8.5 汇总 median

```bash
cd /workspace/GIT/ai_infra/qwen3_phase2
for d in ddp-sync ddp-no-sync; do
  echo "=== $d ==="
  python - <<PY
import json, statistics
rows = [json.loads(l) for l in open("artifacts/$d/metrics.jsonl")]
rows = [r for r in rows if r["step"] > 10]
print("median step_time_ms     :", round(statistics.median(r["step_time_ms"] for r in rows), 2))
print("median tokens_per_second:", round(statistics.median(r["tokens_per_second"] for r in rows), 2))
print("median peak_mem_gb      :", round(statistics.median(r["peak_mem_gb"] for r in rows), 3))
PY
done
```

### 8.6 报告 `artifacts/reports/day12-comm-overlap.md`

- 两组稳定段指标（跳过前 10 step，median `step_time_ms` / `tokens_per_second` / `peak_mem_gb`）。
- `nccl:all_reduce` 与 backward 的重叠：sync 组每个 micro-batch 后是否都有 all_reduce；no_sync 组是否只在每 4 个 micro-batch 后一次；是否与 backward 重叠；是否形成 GPU 空洞。
- trace 路径：`artifacts/ddp-sync-prof/traces/trace.json`、`artifacts/ddp-no-sync-prof/traces/trace.json`。
- LoRA 通信量很小（43.6M 参数 ≈ 87MB bf16）导致收益不明显时，如实记录，结论为“当前配置下通信不是瓶颈”。

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

## 显存分支：BF16 LoRA vs QLoRA

| 配置 | seq_len | peak_mem_gb | 结果 |
|---|---|---|---|
| BF16 LoRA | 256 | 16.52 | 能跑，未 OOM |
| BF16 LoRA | 512 | （待测） | 文档预期 OOM |
| QLoRA 4-bit | 512 | 10.41 | 能跑，未 OOM |

说明：
- 文档预期 BF16 在 24GB 卡上 OOM，这是针对 seq_len=512 的主配置。
- 实测 seq_len=256 时 BF16 能跑通，peak 16.52GB，接近 24GB 上限。
- 若要验证文档预期，需用 seq_len=512 重跑 BF16 探测。
- QLoRA 在 seq_len=512 下 peak 仅 10.41GB，显存优势明显（约 6GB）。
- 注意：QLoRA 改变了数值精度（4-bit 量化 + bf16 compute），
  其 step_time / loss 不能和 BF16 baseline 直接做性能归因对比。

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
2. 模型来源（ModelScope ID 与本地路径）、数据来源（ModelScope ID、本地 parquet 路径、切片脚本）和 SHA256。
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

安装：

```bash
pip install 'deepspeed>=0.16.0'
```

## 完成检查

- [ ] Qwen3-8B 本地加载且 8 step 冒烟成功。
- [ ] 有 40 step baseline 的 JSONL、配置、trace 和截图。
- [ ] 数据加载至少完成两组单变量对照。
- [ ] 有两卡时完成 DDP + `no_sync`；无两卡时明确标注未测。
- [ ] 同步/异步 checkpoint 均验证可恢复。
- [ ] `phase2-report.md` 可由另一人按路径、版本、命令和校验值复跑。
- [ ] 全程未访问 huggingface.co，模型与数据均来自 ModelScope。

## 面试自测

1. 为什么 Qwen3-8B LoRA 不能代表全参数 ZeRO？
2. DDP AllReduce 在 backward 的哪一刻启动？bucket 大小如何影响重叠？
3. `grad_accum=4` 时，为什么 `no_sync()` 数学等价却可能更快？
4. GPU 利用率、tokens/s、MFU 分别回答什么问题？
5. activation checkpointing 用什么换取什么？
6. 异步 checkpoint 为什么仍需 `future.result()`？

1. **为什么 Qwen3-8B LoRA 不能代表全参数 ZeRO**
   LoRA 只训 ~43.6M 参数（0.53%），优化器状态、梯度、通信量都按 LoRA 部分算；ZeRO 的分片对象是全模型 8.23B 参数、梯度和优化器状态。两者显存和通信量的量级完全不同，LoRA 的省显存主要来自冻结 base + 4-bit 量化，不是 ZeRO 的分片机制，不能互相代表。

2. **DDP AllReduce 在 backward 的哪一刻启动？bucket 大小如何影响重叠？**
   DDP 在构造时把参数梯度按 bucket（默认 25MB）分组，并在每个参数上注册 autograd hook。反向计算中，某个 bucket 内所有梯度算完时，该 bucket 立即触发 all-reduce，不等整个 backward 结束。bucket 越大，触发越晚、单次通信越长、与 backward 重叠的窗口越短；bucket 越小，触发越早、通信碎片化、启动开销变大但重叠更好。调 bucket 大小是在"重叠程度"和"通信启动开销"之间权衡。这里的 all-reduce 是把同一个 bucket 里、所有 rank 上算出的梯度做一次全局求和（再平均），让每张卡拿到相同的、全局一致的梯度。

3. **`grad_accum=4` 时，为什么 `no_sync()` 数学等价却可能更快？**
   数学上，DDP 的 all-reduce 是线性的，累积 4 个 micro-batch 的梯度再同步一次，等于每个 micro-batch 同步后再相加。所以最终梯度一致，训练结果等价。但前 3 个 micro-batch 的梯度会被后续覆盖，同步是浪费。`no_sync()` 跳过前 3 次 all-reduce，只留最后一次，通信次数从 4 降到 1。更快是因为省掉了无效通信，尤其在全参训练或网络慢时明显；LoRA 通信量小时差异可能落在噪声里。

4. **GPU 利用率、tokens/s、MFU 分别回答什么问题？**
   - GPU 利用率：GPU 在采样窗口内有没有在干活，回答"卡闲不闲"，不反映效率高低。
   - tokens/s：单位时间处理多少 token，回答"吞吐多快"，是最直接的实测指标。
   - MFU：实际 FLOPs 占硬件理论峰值的比例，回答"算力用得多满"，依赖 FLOPs 估算和峰值来源，是派生指标。

5. **activation checkpointing 用什么换取什么？**
   用**计算换显存**：前向时不保存中间 activation，只存少数 checkpoint 边界；反向时重新前向计算这些 activation。显存下降，step time 上升（多一次前向）。适合显存受限、算力有余的场景。
   checkpointing：
   前向时只存少数几个"边界"的 activation（比如每层的输入）；
   中间那些不存，直接扔掉；
   反向需要时，从最近的边界重新前向算一遍，把需要的 activation 临时算出来，用完再扔。

6. **异步 checkpoint 为什么仍需 `future.result()`？**
   异步保存提交到后台线程，主线程继续训练，但：
   - 必须在训练结束或下一个 checkpoint 前调用 `future.result()`，确保保存完成、异常能被抛出；
   - 否则可能在写入未完成时进程退出，checkpoint 不完整；
   - 同时在下一个异步保存前 `result()`，避免多个保存任务堆积、并发写盘。
   所以 `result()` 是"同步点"，保证异步任务被正确收敛，而不是完全 fire-and-forget。

---

