#!/usr/bin/env bash
# Ubuntu 的 sh 是 dash；用 sh 调用时自动切到 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# SC2-Commander 实验矩阵（Linux）
# 按 EXPERIMENTS 列表逐组调用 start_batch.sh
set -euo pipefail

export PYTHONIOENCODING=utf-8
export LANG="${LANG:-en_US.UTF-8}"
# 本地 SC2 / 内网 vLLM 不走代理，避免 httpx SOCKS 缺依赖或误走代理
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${SC2PATH:-}" || ! -d "${SC2PATH}" ]]; then
  export SC2PATH="/data/hc/sc2/StarCraftII"
fi

# =============================================================================
# 1. Shared game config
# =============================================================================
MY_BOT_NAME="commander"
MAP_NAME="KairosJunctionLE"
REAL_TIME="0"
BOT_RACE="terran"
BOT_INSTRUCT=""

# Default strategy when an experiment row omits strategy=
FORCE_STRATEGY="tank"

# =============================================================================
# 2. Commander model (must exist in llm/config.json -> llm_agents_pool)
# =============================================================================
# COMMANDER_MODEL="qwen3.5-9b"
# COMMANDER_MODEL="qwen3.5-27b"
# COMMANDER_MODEL="kimi-k2.5"
# COMMANDER_MODEL="deepseek-v4-flash"
COMMANDER_MODEL="qwen3.5-9b-lora"
# COMMANDER_MODEL="qwen3-32b-reasoning"

# =============================================================================
# 3. Run control
# =============================================================================
DEFAULT_MATCHES_PER_EXPERIMENT=10
CONCURRENCY=5

# =============================================================================
# 4. Experiment list
# 字段: enemy_race|enemy_difficulty|enemy_build|strategy|matches|model
#   strategy / matches / model 可留空：分别回退 FORCE_STRATEGY /
#   DEFAULT_MATCHES_PER_EXPERIMENT / COMMANDER_MODEL
# =============================================================================
EXPERIMENTS=(
  # "terran|veryeasy|macro|marine|20|"
  # "terran|veryeasy|macro|tank|20|"
  # "terran|veryeasy|macro|battlecruiser|20|"

  # "terran|mediumhard|macro|marine|20|"
  # "terran|mediumhard|macro|tank|20|"
  # "terran|mediumhard|macro|battlecruiser|20|"
  # "terran|hard|macro|marine|20|"
  # "terran|hard|macro|tank|20|"
  # "terran|hard|macro|battlecruiser|20|"
  # "terran|harder|macro|marine|20|"
  # "terran|harder|macro|tank|20|"
  # "terran|harder|macro|battlecruiser|20|"
  # "terran|veryhard|macro|marine|20|"
  # "terran|veryhard|macro|tank|20|"
  # "terran|veryhard|macro|battlecruiser|20|"

    "terran|mediumhard|macro|tank|20|"
    # "terran|hard|macro|tank|20|"
    # "terran|veryhard|macro|tank|20|"
    # "terran|harder|macro|tank|20|"
)

safe_name() {
  echo "$1" | sed 's/[^a-zA-Z0-9_-]/_/g'
}

BATCH_SCRIPT="${SCRIPT_DIR}/start_batch.sh"
if [[ ! -x "${BATCH_SCRIPT}" && ! -f "${BATCH_SCRIPT}" ]]; then
  echo "Batch script not found: ${BATCH_SCRIPT}" >&2
  exit 1
fi
chmod +x "${BATCH_SCRIPT}" 2>/dev/null || true

if (( CONCURRENCY <= 0 )); then
  echo "CONCURRENCY must be greater than 0." >&2
  exit 1
fi
if (( DEFAULT_MATCHES_PER_EXPERIMENT < 0 )); then
  echo "DEFAULT_MATCHES_PER_EXPERIMENT cannot be negative." >&2
  exit 1
fi
if [[ -z "${COMMANDER_MODEL}" ]]; then
  echo "COMMANDER_MODEL cannot be empty." >&2
  exit 1
fi
if (( ${#EXPERIMENTS[@]} == 0 )); then
  echo "EXPERIMENTS is empty. Add at least one experiment." >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/llm/config.json" ]]; then
  echo "Missing llm/config.json." >&2
  exit 1
fi

echo ""
echo "Preparing to run ${#EXPERIMENTS[@]} experiment(s)."
echo "Work dir: ${REPO_ROOT}"
echo "Batch launcher: ${BATCH_SCRIPT}"
echo "Commander model (default): ${COMMANDER_MODEL}"
echo "SC2PATH: ${SC2PATH}"

TOTAL_FAILURES=0
for ((experiment_index = 0; experiment_index < ${#EXPERIMENTS[@]}; experiment_index++)); do
  display_index=$((experiment_index + 1))
  row="${EXPERIMENTS[$experiment_index]}"

  IFS='|' read -r enemy_race enemy_difficulty enemy_build strategy matches model <<<"${row}"

  strategy="${strategy:-${FORCE_STRATEGY}}"
  matches="${matches:-${DEFAULT_MATCHES_PER_EXPERIMENT}}"
  model="${model:-${COMMANDER_MODEL}}"

  if [[ -z "${enemy_race}" || -z "${enemy_difficulty}" || -z "${enemy_build}" ]]; then
    echo "Experiment #${display_index} is missing enemy_race/difficulty/build." >&2
    exit 1
  fi
  if [[ -z "${strategy}" ]]; then
    echo "Experiment #${display_index} has an empty strategy." >&2
    exit 1
  fi
  if [[ -z "${model}" ]]; then
    echo "Experiment #${display_index} has an empty model." >&2
    exit 1
  fi

  if [[ "${strategy}" != "none" ]]; then
    strategy_path="${REPO_ROOT}/skills/${BOT_RACE}/${strategy}"
    if [[ ! -d "${strategy_path}" ]]; then
      echo "Strategy folder not found for experiment #${display_index}: ${strategy_path}" >&2
      exit 1
    fi
  fi

  if (( matches <= 0 )); then
    echo "Skipping experiment #${display_index} because matches=${matches}."
    continue
  fi

  ts="$(date +%Y%m%d_%H%M%S)"
  batch_name="batch_${ts}_e${display_index}_$(safe_name "${MAP_NAME}")_$(safe_name "${BOT_RACE}")v$(safe_name "${enemy_race}")_$(safe_name "${enemy_difficulty}")_$(safe_name "${enemy_build}")_$(safe_name "${strategy}")_$(safe_name "${model}")"

  echo ""
  echo "=================================================="
  echo "Experiment ${display_index} / ${#EXPERIMENTS[@]}"
  echo "=================================================="
  echo "Enemy AI : ${enemy_race} | difficulty=${enemy_difficulty} | build=${enemy_build}"
  echo "Strategy : ${strategy}"
  echo "Model    : ${model}"
  echo "Run      : ${matches} matches, concurrency=${CONCURRENCY}"
  echo "Batch    : ${batch_name}"
  echo "=================================================="

  set +e
  batch_args=(
    --my-bot-name "${MY_BOT_NAME}"
    --map-name "${MAP_NAME}"
    --real-time "${REAL_TIME}"
    --enemy-race "${enemy_race}"
    --enemy-difficulty "${enemy_difficulty}"
    --enemy-build "${enemy_build}"
    --bot-race "${BOT_RACE}"
    --force-strategy "${strategy}"
    --commander-model "${model}"
    --total-matches "${matches}"
    --concurrency "${CONCURRENCY}"
    --batch-name "${batch_name}"
  )
  if [[ -n "${BOT_INSTRUCT}" ]]; then
    batch_args+=(--bot-instruct "${BOT_INSTRUCT}")
  fi

  bash "${BATCH_SCRIPT}" "${batch_args[@]}"
  batch_exit_code=$?
  set -e

  if (( batch_exit_code != 0 )); then
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
    echo "Experiment ${display_index} failed with exit code ${batch_exit_code}."
  else
    echo "Experiment ${display_index} finished successfully."
  fi
done

echo ""
if (( TOTAL_FAILURES > 0 )); then
  echo "All experiments finished, but ${TOTAL_FAILURES} experiment(s) failed." >&2
  exit 1
fi

echo "All experiments finished successfully."
