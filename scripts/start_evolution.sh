#!/usr/bin/env bash
# Ubuntu 的 sh 是 dash；用 sh 调用时自动切到 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# SC2-Commander 自动策略进化（Linux）
# 用法:
#   1) 改脚本顶部「配置区」，直接: ./scripts/start_evolution.sh
#   2) 或命令行覆盖: ./scripts/start_evolution.sh --matches 20 --concurrency 8
set -euo pipefail

export PYTHONIOENCODING=utf-8
export LANG="${LANG:-en_US.UTF-8}"
# 本地 SC2 / 内网 vLLM 不走代理，避免 httpx SOCKS 缺依赖或误走代理
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Keep the Linux runtime identical to interactive development and child batch
# processes. The explicit interpreter selection below remains as a fallback.
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

# =============================================================================
# 配置区（按需修改；命令行参数会覆盖对应项）
# =============================================================================
STRATEGY="tank"
COMMANDER_MODEL="kimi-k2.5"
EVOLUTION_MODEL="kimi-k2.5"     # 空 = 与 commander 相同
DIFFICULTIES="harder,veryhard,cheatvision,cheatmoney,cheatinsane"
MATCHES=10
CANDIDATE_INITIAL_MATCHES=6
CANDIDATE_MAX_MATCHES=10
CANDIDATE_STEP_MATCHES=2
CONCURRENCY=5
MAX_GENERATIONS=10
RUN_DIR=""                      # 续跑时填 evolution_runs/... 路径
BASELINE_BATCH_DIR=""           # 新 run 可复用已完成的基线批次
# =============================================================================

usage() {
  cat <<'EOF'
Usage: start_evolution.sh [options]

Defaults live in the config block at the top of this script.
CLI flags override those defaults when provided.

Options:
  --strategy STRATEGY
  --commander-model KEY
  --evolution-model KEY   默认与 commander-model 相同（由 evolution 模块处理）
  --difficulties LIST     逗号分隔
  --matches N             新难度冠军基线局数
  --candidate-initial-matches N  候选首轮局数
  --candidate-max-matches N      候选最多局数
  --candidate-step-matches N     结果不确定时追加局数
  --concurrency N         并发
  --max-generations N     最大代数
  --run-dir PATH          已有 run 目录，用于 resume
  --baseline-batch-dir PATH  新 run 复用一个已完成基线批次
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strategy) STRATEGY="$2"; shift 2 ;;
    --commander-model) COMMANDER_MODEL="$2"; shift 2 ;;
    --evolution-model) EVOLUTION_MODEL="$2"; shift 2 ;;
    --difficulties) DIFFICULTIES="$2"; shift 2 ;;
    --matches) MATCHES="$2"; shift 2 ;;
    --candidate-initial-matches) CANDIDATE_INITIAL_MATCHES="$2"; shift 2 ;;
    --candidate-max-matches) CANDIDATE_MAX_MATCHES="$2"; shift 2 ;;
    --candidate-step-matches) CANDIDATE_STEP_MATCHES="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --max-generations) MAX_GENERATIONS="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --baseline-batch-dir) BASELINE_BATCH_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${STRATEGY}" || -z "${COMMANDER_MODEL}" ]]; then
  echo "--strategy and --commander-model are required." >&2
  usage >&2
  exit 1
fi

if [[ -z "${SC2PATH:-}" || ! -d "${SC2PATH}" ]]; then
  export SC2PATH="/data/hc/sc2/StarCraftII"
fi

if [[ ! -f "${REPO_ROOT}/llm/config.json" ]]; then
  echo "Missing llm/config.json. Copy llm/config.example.json and fill in keys." >&2
  exit 1
fi

usable_python() {
  local exe="$1"
  [[ -n "${exe}" && -x "${exe}" ]] || return 1
  "${exe}" -c "import openai" >/dev/null 2>&1
}

PYTHON_EXE=""
CANDIDATES=()
[[ -n "${VIRTUAL_ENV:-}" ]] && CANDIDATES+=("${VIRTUAL_ENV}/bin/python")
[[ -x "${REPO_ROOT}/venv/bin/python" ]] && CANDIDATES+=("${REPO_ROOT}/venv/bin/python")
[[ -x "/data/hc/miniconda3/envs/SC2/bin/python" ]] && CANDIDATES+=("/data/hc/miniconda3/envs/SC2/bin/python")
command -v python >/dev/null 2>&1 && CANDIDATES+=("$(command -v python)")
command -v python3 >/dev/null 2>&1 && CANDIDATES+=("$(command -v python3)")

for cand in "${CANDIDATES[@]}"; do
  if usable_python "${cand}"; then
    PYTHON_EXE="${cand}"
    break
  fi
done

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "No usable Python found (need import openai). Hint: conda activate SC2" >&2
  exit 1
fi

echo "Using Python : ${PYTHON_EXE}"
echo "Repo root    : ${REPO_ROOT}"
echo "SC2PATH      : ${SC2PATH}"
echo "Strategy     : ${STRATEGY}"
echo "Commander    : ${COMMANDER_MODEL}"
[[ -n "${EVOLUTION_MODEL}" ]] && echo "Evolution    : ${EVOLUTION_MODEL}"
echo "Difficulties : ${DIFFICULTIES}"
echo "Matches/gen  : ${MATCHES}, concurrency=${CONCURRENCY}, max_gen=${MAX_GENERATIONS}"
echo "Candidate    : ${CANDIDATE_INITIAL_MATCHES} initial, +${CANDIDATE_STEP_MATCHES}, max ${CANDIDATE_MAX_MATCHES}"

ARGS=(
  -m evolution
  --strategy "${STRATEGY}"
  --commander-model "${COMMANDER_MODEL}"
  --difficulties "${DIFFICULTIES}"
  --matches "${MATCHES}"
  --candidate-initial-matches "${CANDIDATE_INITIAL_MATCHES}"
  --candidate-max-matches "${CANDIDATE_MAX_MATCHES}"
  --candidate-step-matches "${CANDIDATE_STEP_MATCHES}"
  --concurrency "${CONCURRENCY}"
  --max-generations "${MAX_GENERATIONS}"
)

if [[ -n "${EVOLUTION_MODEL}" ]]; then
  ARGS+=(--evolution-model "${EVOLUTION_MODEL}")
fi
if [[ -n "${RUN_DIR}" ]]; then
  ARGS+=(--run-dir "${RUN_DIR}")
fi
if [[ -n "${BASELINE_BATCH_DIR}" ]]; then
  ARGS+=(--baseline-batch-dir "${BASELINE_BATCH_DIR}")
fi

exec "${PYTHON_EXE}" "${ARGS[@]}"
