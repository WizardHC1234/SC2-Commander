#!/usr/bin/env bash
# 用 LLaMA-Factory 对 Qwen3.5-9B 做 SC2-Commander SFT（LoRA）
#
# 数据默认: <repo>/sft_data/sft.jsonl  (OpenAI messages 格式)
# 用法:
#   1) 先停掉占满 GPU 的 vLLM（训练前 nvidia-smi 确认空闲）
#   2) 按需改下面“用户配置区”
#   3) bash scripts/run_llamafactory_sft.sh
#
# DRY_RUN=true 时只注册数据 + 生成 YAML 并预检，不启动训练。
# 训练成功后默认自动合并 LoRA 到 sft_runs/<RUN_TAG>_merged（可用 AUTO_MERGE=false 关闭）。

# Ubuntu 的 sh 是 dash；用 sh 调用时自动切到 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==============================================================================
# 用户配置区
# ==============================================================================

LLAMA_FACTORY_DIR="${REPO_ROOT}/LLaMA-Factory"
CONDA_ROOT="/data/hc/miniconda3"
CONDA_ENV="llamafactory_qwen35"
DRY_RUN="${DRY_RUN:-false}"

# 数据（OpenAI messages: system/user/assistant）
DATA_FILE="${REPO_ROOT}/sft_data/sft.jsonl"
DATASET_NAME="sc2_commander_sft"
RUN_TAG="commander_qwen35_9b_lora"

MODEL_NAME_OR_PATH="/data/hc/sc2/model/Qwen/Qwen3.5-9B"
OUTPUT_BASE="${REPO_ROOT}/sft_runs"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_TAG}"

# GPU：训练前请确认这些卡空闲（nvidia-smi）
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Qwen3.5 用 qwen3_5 模板；本数据无 <think>，用 nothink
TEMPLATE="qwen3_5_nothink"
ENABLE_THINKING="false"

FINETUNING_TYPE="lora"
# 与已验证可运行的 BF16 LoRA 配置一致。
LORA_TARGET="${LORA_TARGET:-all}"
# 保守 LoRA 配置：当前数据规模下避免用高 rank/alpha 过度覆盖基座行为。
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="0.05"

# 显存：使用已验证的 BF16 LoRA + ZeRO-3
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/ds_z3_config.json}"
QUANTIZATION_BIT="${QUANTIZATION_BIT:-0}"
QUANTIZATION_METHOD="${QUANTIZATION_METHOD:-bnb}"
ENABLE_LIGER_KERNEL="${ENABLE_LIGER_KERNEL:-true}"
# Qwen3.5 的 causal-conv1d/FLA 快路径由 fa2 开关触发；不是单独的 YAML 字段。
FLASH_ATTN="${FLASH_ATTN:-fa2}"
FREEZE_VISION_TOWER="true"
FREEZE_MULTI_MODAL_PROJECTOR="true"
UPCAST_LAYERNORM="${UPCAST_LAYERNORM:-false}"
# 重入式 gradient checkpointing 会少保留一份 autograd graph，长序列下峰值更低。
USE_REENTRANT_GC="${USE_REENTRANT_GC:-true}"
# 参数 offload 已提供充足显存余量，关闭 hidden-state CPU 搬运以减少额外开销。
USE_UNSLOTH_GC="${USE_UNSLOTH_GC:-false}"

# 24 GiB GPU 上 16384-token 序列的激活峰值会触发 OOM；按任务允许的最小长度降到 12288。
# 如确认显存有余量，仍可通过 CUTOFF_LEN=16384 临时覆盖。
CUTOFF_LEN="${CUTOFF_LEN:-12288}"
PACKING="false"
TRAIN_ON_PROMPT="false"
MAX_SAMPLES="1000000"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
# datasets.map 的每个进程都会占内存；127MB JSONL 用 2 个进程足够。
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-2}"
# 定期释放 PyTorch 缓存，牺牲少量速度来避免长短样本切换造成显存碎片峰值。
TORCH_EMPTY_CACHE_STEPS="${TORCH_EMPTY_CACHE_STEPS:-10}"
# 冒烟: MAX_STEPS=2 bash scripts/run_llamafactory_sft.sh
MAX_STEPS="${MAX_STEPS:-0}"

# 全局 batch = 8 × 1 × 4 = 32
# 小规模策略数据采用低学习率、单轮训练，降低过拟合与策略回退风险。
LEARNING_RATE="5.0e-5"
NUM_TRAIN_EPOCHS="1.0"
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO="0.05"
BF16="${BF16:-true}"

LOGGING_STEPS="5"
# 当前 ~3304 条、val=5% 时约 ~98 steps/epoch，2 epoch ~196 steps
SAVE_STEPS="100"
SAVE_TOTAL_LIMIT="3"
# 12288-token + 大词表在 eval 时会物化约 9.4 GiB logits，24 GiB 卡必然 OOM。
# 默认关闭训练中评估；如需验证，建议训练结束后用短序列/独立脚本执行。
VAL_SIZE="${VAL_SIZE:-0}"
PLOT_LOSS="true"
REPORT_TO="none"
DDP_TIMEOUT="180000000"

# SwanLab 实验记录（与 /data/hc/sc2/run_llamafactory_sc2_sft.sh 对齐）
USE_SWANLAB="${USE_SWANLAB:-true}"
SWANLAB_PROJECT="SC2-Commander"
SWANLAB_WORKSPACE=""
SWANLAB_RUN_NAME="${RUN_TAG}"
SWANLAB_MODE="cloud"
# 优先读环境变量；未设置则用你本机旧脚本里的 key
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
SWANLAB_LOGDIR="${OUTPUT_BASE}/swanlab"

# 训练结束后自动合并 LoRA -> 完整模型
AUTO_MERGE="${AUTO_MERGE:-true}"
EXPORT_DIR="${OUTPUT_BASE}/${RUN_TAG}_merged"
EXPORT_SIZE="5"
EXPORT_DEVICE="cpu"          # cpu 更稳、不占训练卡；要更快可改 auto 并设 MERGE_CUDA_VISIBLE_DEVICES
EXPORT_LEGACY_FORMAT="false"
MERGE_CUDA_VISIBLE_DEVICES="0"
OVERWRITE_EXPORT_DIR="true"

# ==============================================================================
# 内部派生
# ==============================================================================

DATASET_DIR="${LLAMA_FACTORY_DIR}/data"
REGISTERED_FILE="sc2/commander/${DATASET_NAME}.jsonl"
REGISTERED_PATH="${DATASET_DIR}/${REGISTERED_FILE}"
YAML_DIR="${OUTPUT_BASE}/configs"
YAML_PATH="${YAML_DIR}/${RUN_TAG}.yaml"
EXPORT_YAML_PATH="${YAML_DIR}/${RUN_TAG}_merge.yaml"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "找不到数据: ${DATA_FILE}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "找不到模型: ${MODEL_NAME_OR_PATH}" >&2
  exit 1
fi
if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
  echo "找不到 LLaMA-Factory: ${LLAMA_FACTORY_DIR}" >&2
  exit 1
fi
if [[ -n "${DEEPSPEED_CONFIG}" && ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "找不到 DeepSpeed 配置: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

if [[ "${USE_SWANLAB}" == "true" && -z "${SWANLAB_API_KEY}" && "${SWANLAB_MODE}" == "cloud" ]]; then
  echo "警告: USE_SWANLAB=true 且 mode=cloud，但 SWANLAB_API_KEY 为空；请 export SWANLAB_API_KEY=... 或先 swanlab login" >&2
fi

mkdir -p "$(dirname "${REGISTERED_PATH}")" "${YAML_DIR}" "${OUTPUT_DIR}" "${SWANLAB_LOGDIR}"
cp -f "${DATA_FILE}" "${REGISTERED_PATH}"

# 注册 OpenAI messages 格式
python3 - <<PY
import json
from pathlib import Path
info_path = Path(${DATASET_DIR@Q}) / "dataset_info.json"
info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
info[${DATASET_NAME@Q}] = {
    "file_name": ${REGISTERED_FILE@Q},
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}
info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("已注册数据集", ${DATASET_NAME@Q}, "->", info_path)
PY

cat > "${YAML_PATH}" <<YAML
### model
model_name_or_path: ${MODEL_NAME_OR_PATH}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: ${FINETUNING_TYPE}
lora_target: ${LORA_TARGET}
lora_rank: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}
lora_dropout: ${LORA_DROPOUT}
freeze_vision_tower: ${FREEZE_VISION_TOWER}
freeze_multi_modal_projector: ${FREEZE_MULTI_MODAL_PROJECTOR}
upcast_layernorm: ${UPCAST_LAYERNORM}
enable_liger_kernel: ${ENABLE_LIGER_KERNEL}
flash_attn: ${FLASH_ATTN}
use_reentrant_gc: ${USE_REENTRANT_GC}
use_unsloth_gc: ${USE_UNSLOTH_GC}
YAML

if [[ -n "${QUANTIZATION_BIT}" && "${QUANTIZATION_BIT}" != "0" ]]; then
  cat >> "${YAML_PATH}" <<YAML
quantization_bit: ${QUANTIZATION_BIT}
quantization_method: ${QUANTIZATION_METHOD}
YAML
fi

if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  cat >> "${YAML_PATH}" <<YAML
deepspeed: ${DEEPSPEED_CONFIG}
YAML
fi

cat >> "${YAML_PATH}" <<YAML

### dataset
dataset: ${DATASET_NAME}
dataset_dir: ${DATASET_DIR}
template: ${TEMPLATE}
cutoff_len: ${CUTOFF_LEN}
packing: ${PACKING}
train_on_prompt: ${TRAIN_ON_PROMPT}
max_samples: ${MAX_SAMPLES}
overwrite_cache: true
preprocessing_num_workers: ${PREPROCESSING_NUM_WORKERS}

### output
output_dir: ${OUTPUT_DIR}
logging_steps: ${LOGGING_STEPS}
save_steps: ${SAVE_STEPS}
save_total_limit: ${SAVE_TOTAL_LIMIT}
plot_loss: ${PLOT_LOSS}
report_to: ${REPORT_TO}
overwrite_output_dir: true

### swanlab
use_swanlab: ${USE_SWANLAB}
swanlab_project: ${SWANLAB_PROJECT}
swanlab_workspace: ${SWANLAB_WORKSPACE}
swanlab_run_name: ${SWANLAB_RUN_NAME}
swanlab_mode: ${SWANLAB_MODE}
swanlab_api_key: ${SWANLAB_API_KEY}
swanlab_logdir: ${SWANLAB_LOGDIR}

### train
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
learning_rate: ${LEARNING_RATE}
num_train_epochs: ${NUM_TRAIN_EPOCHS}
lr_scheduler_type: ${LR_SCHEDULER_TYPE}
warmup_ratio: ${WARMUP_RATIO}
bf16: ${BF16}
ddp_timeout: ${DDP_TIMEOUT}
ddp_find_unused_parameters: false
disable_gradient_checkpointing: false
torch_empty_cache_steps: ${TORCH_EMPTY_CACHE_STEPS}
YAML

if [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  cat >> "${YAML_PATH}" <<YAML
max_steps: ${MAX_STEPS}
YAML
fi

# val_size=0 时必须关掉 eval，否则 LLaMA-Factory 直接报错
python3 - "${YAML_PATH}" "${VAL_SIZE}" "${SAVE_STEPS}" "${ENABLE_THINKING}" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
val_size = float(sys.argv[2])
save_steps = sys.argv[3]
enable_thinking = sys.argv[4]
lines = ["", "### eval"]
if val_size > 1e-6:
    lines += [
        f"val_size: {val_size}",
        "per_device_eval_batch_size: 1",
        "eval_strategy: steps",
        f"eval_steps: {save_steps}",
    ]
else:
    lines += [
        "val_size: 0",
        "eval_strategy: 'no'",
    ]
lines += ["", "### qwen thinking", f"enable_thinking: {enable_thinking}", ""]
with path.open("a", encoding="utf-8") as f:
    f.write("\n".join(lines))
PY

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${LLAMA_FACTORY_DIR}"

export CUDA_VISIBLE_DEVICES
export FORCE_TORCHRUN
export NPROC_PER_NODE
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export TMPDIR="${TMPDIR:-/data/hc/sc2/.tmp}"
export HF_HOME="${HF_HOME:-/data/hc/sc2/cache/hf}"
mkdir -p "${TMPDIR}" "${HF_HOME}"
# 与 torch cu124 对齐，避免 DeepSpeed 误用系统 /usr/bin/nvcc (11.5)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# 长序列下减轻碎片，并主动回收空闲缓存。
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

if [[ "${FLASH_ATTN}" == "fa2" ]]; then
  python - <<'CHECK_CAUSAL_CONV'
try:
    from fla.modules.convolution import causal_conv1d  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "FLASH_ATTN=fa2 需要 flash-linear-attention>=0.4.1（提供 Qwen3.5 的 causal-conv1d 快路径）。"
        " 请在 llamafactory_qwen35 环境安装后重试："
        " pip install -U 'flash-linear-attention>=0.4.1'"
    ) from exc
print("Qwen3.5 causal-conv1d 快路径依赖检查通过")
CHECK_CAUSAL_CONV
fi

python - "${YAML_PATH}" <<'CHECKPY'
import sys
from pathlib import Path
import yaml
from transformers import HfArgumentParser
from llamafactory.hparams.parser import _TRAIN_ARGS

path = Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
parser = HfArgumentParser(_TRAIN_ARGS)
try:
    parser.parse_dict(cfg, allow_extra_keys=False)
except Exception as exc:
    print(f"YAML 预检失败: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(f"YAML 预检通过: {path}")
CHECKPY

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY_RUN=true：配置已生成，不启动训练。"
  echo "YAML: ${YAML_PATH}"
  exit 0
fi

echo "Repo:   ${REPO_ROOT}"
echo "数据集: ${DATASET_NAME}"
echo "数据:   ${REGISTERED_PATH}"
echo "模型:   ${MODEL_NAME_OR_PATH}"
echo "配置:   ${YAML_PATH}"
echo "输出:   ${OUTPUT_DIR}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo "合并:   AUTO_MERGE=${AUTO_MERGE} -> ${EXPORT_DIR}"

llamafactory-cli train "${YAML_PATH}"

if [[ "${AUTO_MERGE}" != "true" ]]; then
  echo "训练完成。AUTO_MERGE=false，跳过合并。"
  echo "LoRA 目录: ${OUTPUT_DIR}"
  exit 0
fi

echo ""
echo "=================================================="
echo "训练完成，开始合并 LoRA -> 完整模型"
echo "=================================================="

resolve_adapter_dir() {
  local root="$1"
  if [[ -f "${root}/adapter_config.json" ]]; then
    echo "${root}"
    return 0
  fi
  local latest=""
  latest="$(ls -d "${root}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${latest}" && -f "${latest}/adapter_config.json" ]]; then
    echo "${latest}"
    return 0
  fi
  return 1
}

ADAPTER_DIR="$(resolve_adapter_dir "${OUTPUT_DIR}")" || {
  echo "找不到可合并的 LoRA adapter（${OUTPUT_DIR} 或其 checkpoint-*）" >&2
  exit 1
}

if [[ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" && ! -f "${ADAPTER_DIR}/adapter_model.bin" ]]; then
  echo "adapter 目录缺少权重文件: ${ADAPTER_DIR}" >&2
  exit 1
fi

if [[ -d "${EXPORT_DIR}" && "${OVERWRITE_EXPORT_DIR}" != "true" ]]; then
  if find "${EXPORT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "合并输出目录已存在且非空: ${EXPORT_DIR}" >&2
    echo "如需覆盖，设 OVERWRITE_EXPORT_DIR=true" >&2
    exit 1
  fi
fi

mkdir -p "${EXPORT_DIR}"

cat > "${EXPORT_YAML_PATH}" <<YAML
### model
model_name_or_path: ${MODEL_NAME_OR_PATH}
adapter_name_or_path: ${ADAPTER_DIR}
template: ${TEMPLATE}
trust_remote_code: true

### export
export_dir: ${EXPORT_DIR}
export_size: ${EXPORT_SIZE}
export_device: ${EXPORT_DEVICE}
export_legacy_format: ${EXPORT_LEGACY_FORMAT}
YAML

# 合并用单卡/CPU，避免沿用训练的多卡 torchrun
unset FORCE_TORCHRUN NPROC_PER_NODE || true
export CUDA_VISIBLE_DEVICES="${MERGE_CUDA_VISIBLE_DEVICES}"

echo "基座:   ${MODEL_NAME_OR_PATH}"
echo "LoRA:   ${ADAPTER_DIR}"
echo "配置:   ${EXPORT_YAML_PATH}"
echo "输出:   ${EXPORT_DIR}"
echo "设备:   export_device=${EXPORT_DEVICE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

llamafactory-cli export "${EXPORT_YAML_PATH}"

# LLaMA-Factory + Transformers 5.5 may serialize Qwen3.5's nested modules
# with duplicated `language_model` prefixes. vLLM expects the original HF
# names, so normalize the exported safetensors in place before reporting a
# successful merged model. The original LoRA adapter is never modified.
"${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python" - "${EXPORT_DIR}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

root = Path(sys.argv[1])
old_prefix = "model.language_model.language_model.language_model."
vision_prefix = "model.language_model.visual."

def fixed_key(key: str) -> str:
    if key.startswith(old_prefix):
        return "model.language_model." + key.removeprefix(old_prefix)
    if key.startswith(vision_prefix):
        return "model.visual." + key.removeprefix(vision_prefix)
    return key

shards = sorted(root.glob("*.safetensors"))
needs_fix = False
for shard in shards:
    with safe_open(shard, framework="pt", device="cpu") as handle:
        if any(key.startswith(old_prefix) or key.startswith(vision_prefix) for key in handle.keys()):
            needs_fix = True
            break

if not needs_fix:
    print("Qwen3.5 export weight names are already valid; no normalization needed.")
    raise SystemExit(0)

staging = root.with_name(f"{root.name}.keyfix-tmp")
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir()
weight_map = {}

for shard in shards:
    with safe_open(shard, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        source_keys = list(handle.keys())
        tensors = {fixed_key(key): handle.get_tensor(key) for key in source_keys}
    if len(tensors) != len(source_keys):
        raise RuntimeError(f"Duplicate key after normalizing {shard.name}")
    save_file(tensors, staging / shard.name, metadata=metadata)
    weight_map.update({key: shard.name for key in tensors})

for shard in shards:
    (staging / shard.name).replace(shard)
staging.rmdir()
(root / "model.safetensors.index.json").write_text(
    json.dumps(
        {
            "metadata": {"total_size": sum(path.stat().st_size for path in shards)},
            "weight_map": weight_map,
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
print("Normalized Qwen3.5 exported weight names for vLLM.")
PY

echo ""
echo "全部完成。"
echo "LoRA:   ${ADAPTER_DIR}"
echo "合并模型: ${EXPORT_DIR}"
