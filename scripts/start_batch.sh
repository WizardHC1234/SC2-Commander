#!/usr/bin/env bash
# Ubuntu 的 sh 是 dash；用 sh 调用时自动切到 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# SC2-Commander 批量对局（Linux）
# 用法示例:
#   ./scripts/start_batch.sh
#   ./scripts/start_batch.sh --force-strategy tank --enemy-difficulty hard \
#       --total-matches 10 --concurrency 2 --commander-model qwen3.5-27b
set -euo pipefail

export PYTHONIOENCODING=utf-8
export LANG="${LANG:-en_US.UTF-8}"
# 本地 SC2 / 内网 vLLM 不走代理，避免 httpx SOCKS 缺依赖或误走代理
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------- 默认参数 ----------------
MY_BOT_NAME="commander"
MAP_NAME="KairosJunctionLE"
REAL_TIME="0"

ENEMY_RACE="terran"
ENEMY_DIFFICULTY="hard"
ENEMY_BUILD="macro"

BOT_RACE="terran"
FORCE_STRATEGY="tank"
BOT_INSTRUCT=""

COMMANDER_MODEL="qwen3.5-27b"

TOTAL_MATCHES=20
CONCURRENCY=2
START_INDEX=0
BATCH_NAME=""

usage() {
  cat <<'EOF'
Usage: start_batch.sh [options]

Options:
  --my-bot-name NAME
  --map-name NAME
  --real-time 0|1
  --enemy-race RACE
  --enemy-difficulty DIFF
  --enemy-build BUILD
  --bot-race RACE
  --force-strategy STRATEGY
  --bot-instruct TEXT
  --commander-model KEY
  --total-matches N
  --concurrency N
  --start-index N
  --batch-name NAME
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --my-bot-name) MY_BOT_NAME="$2"; shift 2 ;;
    --map-name) MAP_NAME="$2"; shift 2 ;;
    --real-time) REAL_TIME="$2"; shift 2 ;;
    --enemy-race) ENEMY_RACE="$2"; shift 2 ;;
    --enemy-difficulty) ENEMY_DIFFICULTY="$2"; shift 2 ;;
    --enemy-build) ENEMY_BUILD="$2"; shift 2 ;;
    --bot-race) BOT_RACE="$2"; shift 2 ;;
    --force-strategy) FORCE_STRATEGY="$2"; shift 2 ;;
    --bot-instruct) BOT_INSTRUCT="$2"; shift 2 ;;
    --commander-model) COMMANDER_MODEL="$2"; shift 2 ;;
    --total-matches) TOTAL_MATCHES="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --batch-name) BATCH_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${SC2PATH:-}" || ! -d "${SC2PATH}" ]]; then
  export SC2PATH="/data/hc/sc2/StarCraftII"
fi

safe_name() {
  echo "$1" | sed 's/[^a-zA-Z0-9_-]/_/g'
}

usable_python() {
  local exe="$1"
  [[ -n "${exe}" && -x "${exe}" ]] || return 1
  "${exe}" -c "import openai" >/dev/null 2>&1
}

find_python() {
  local candidates=()
  [[ -n "${VIRTUAL_ENV:-}" ]] && candidates+=("${VIRTUAL_ENV}/bin/python")
  [[ -x "${REPO_ROOT}/venv/bin/python" ]] && candidates+=("${REPO_ROOT}/venv/bin/python")
  [[ -x "/data/hc/miniconda3/envs/SC2/bin/python" ]] && candidates+=("/data/hc/miniconda3/envs/SC2/bin/python")
  command -v python >/dev/null 2>&1 && candidates+=("$(command -v python)")
  command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")

  local cand
  for cand in "${candidates[@]}"; do
    if usable_python "${cand}"; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

validate_config() {
  if (( CONCURRENCY <= 0 )); then
    echo "CONCURRENCY must be greater than 0." >&2
    exit 1
  fi
  if (( START_INDEX < 0 )); then
    echo "START_INDEX cannot be negative." >&2
    exit 1
  fi
  if (( TOTAL_MATCHES <= 0 )); then
    echo "TOTAL_MATCHES=${TOTAL_MATCHES}, nothing to run."
    exit 0
  fi
  if [[ -z "${FORCE_STRATEGY}" ]]; then
    echo "FORCE_STRATEGY cannot be empty." >&2
    exit 1
  fi
  if [[ "${FORCE_STRATEGY}" != "none" ]]; then
    # Evolution sets SC2_STRATEGY_ROOT to evolution_runs/.../strategies.
    local candidates=()
    if [[ -n "${SC2_STRATEGY_ROOT:-}" ]]; then
      candidates+=("${SC2_STRATEGY_ROOT%/}/${FORCE_STRATEGY}")
    fi
    candidates+=("${REPO_ROOT}/skills/${BOT_RACE}/${FORCE_STRATEGY}")
    local resolved=""
    local candidate
    for candidate in "${candidates[@]}"; do
      if [[ -f "${candidate}/strategy.md" ]]; then
        resolved="${candidate}"
        break
      fi
    done
    if [[ -z "${resolved}" ]]; then
      echo "Strategy folder not found for '${FORCE_STRATEGY}'. Searched: ${candidates[*]}" >&2
      exit 1
    fi
  fi
  if [[ ! -f "${REPO_ROOT}/llm/config.json" ]]; then
    echo "Missing llm/config.json. Copy llm/config.example.json and fill in keys." >&2
    exit 1
  fi
}

default_batch_name() {
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "batch_${ts}_$(safe_name "${MAP_NAME}")_$(safe_name "${BOT_RACE}")v$(safe_name "${ENEMY_RACE}")_$(safe_name "${ENEMY_DIFFICULTY}")_$(safe_name "${ENEMY_BUILD}")_$(safe_name "${FORCE_STRATEGY}")_$(safe_name "${COMMANDER_MODEL}")"
}

move_match_console_log() {
  local out_file="$1"
  local record_dir_file="$2"

  if [[ ! -f "${record_dir_file}" ]]; then
    echo "${out_file}"
    return 0
  fi

  local match_dir
  match_dir="$(tr -d '\r\n' < "${record_dir_file}")"
  rm -f "${record_dir_file}"

  if [[ -z "${match_dir}" || ! -d "${match_dir}" ]]; then
    echo "${out_file}"
    return 0
  fi

  local match_id canonical_log
  match_id="$(basename "${match_dir}")"
  canonical_log="${match_dir}/${match_id}.log"

  if [[ -f "${canonical_log}" ]]; then
    if [[ -f "${out_file}" && "${out_file}" != "${canonical_log}" ]]; then
      rm -f "${out_file}"
    fi
    echo "${canonical_log}"
    return 0
  fi

  if [[ -f "${out_file}" ]]; then
    mv -f "${out_file}" "${canonical_log}"
    echo "${canonical_log}"
    return 0
  fi

  echo "${out_file}"
}

run_one_match() {
  local idx="$1"
  local out_file="$2"
  local record_dir_file="$3"

  local args=(
    "${REPO_ROOT}/run_vs_ai.py"
    --my-bot-name "${MY_BOT_NAME}"
    --map-name "${MAP_NAME}"
    --enemy-race "${ENEMY_RACE}"
    --enemy-difficulty "${ENEMY_DIFFICULTY}"
    --enemy-build "${ENEMY_BUILD}"
    --bot-race "${BOT_RACE}"
    --commander-model "${COMMANDER_MODEL}"
    --force-strategy "${FORCE_STRATEGY}"
    --batch-name "${BATCH_NAME}"
    --run-index "${idx}"
    --output-base-dir "${RECORD_ROOT}"
    --record-dir-file "${record_dir_file}"
    --skip-version-update
  )

  if [[ -n "${BOT_INSTRUCT}" ]]; then
    args+=(--bot-instruct "${BOT_INSTRUCT}")
  fi
  if [[ "${REAL_TIME}" == "1" || "${REAL_TIME}" == "true" || "${REAL_TIME}" == "True" ]]; then
    args+=(--real-time)
  fi

  set +e
  "${PYTHON_EXE}" "${args[@]}" >>"${out_file}" 2>&1
  local exit_code=$?
  set -e

  out_file="$(move_match_console_log "${out_file}" "${record_dir_file}")"
  echo "${exit_code}|${out_file}"
}

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -rp | wc -l | tr -d ' ')"
    if (( running < CONCURRENCY )); then
      break
    fi
    # bash 4.3+：等任意子进程；否则 sleep 轮询
    if wait -n 2>/dev/null; then
      :
    else
      sleep 1
    fi
  done
}

validate_config
PYTHON_EXE="$(find_python)" || {
  echo "No usable Python found (need import openai). Hint: conda activate SC2" >&2
  exit 1
}
RECORD_ROOT="${REPO_ROOT}/game_records"

echo ""
echo "=================================================="
echo "SC2-Commander experiment batch"
echo "=================================================="
echo "Bot      : ${MY_BOT_NAME} (${BOT_RACE})"
echo "Enemy    : ${ENEMY_RACE} | difficulty=${ENEMY_DIFFICULTY} | build=${ENEMY_BUILD}"
echo "Map      : ${MAP_NAME}"
echo "Strategy : ${FORCE_STRATEGY}"
echo "Model    : ${COMMANDER_MODEL}"
echo "Run      : ${TOTAL_MATCHES} matches, concurrency=${CONCURRENCY}"
echo "Python   : ${PYTHON_EXE}"
echo "SC2PATH  : ${SC2PATH}"
echo "=================================================="
echo ""

if [[ -z "${BATCH_NAME}" ]]; then
  BATCH_NAME="$(default_batch_name)"
fi

SAFE_BATCH="$(safe_name "${BATCH_NAME}")"
LOG_DIR="${TMPDIR:-/tmp}/sc2-commander/${SAFE_BATCH}-$$"
mkdir -p "${LOG_DIR}"

# 单局
if (( TOTAL_MATCHES <= 1 )); then
  echo "Starting one match."
  echo "Batch: ${BATCH_NAME}"
  out_file="${LOG_DIR}/fg_run_0.log"
  record_dir_file="${LOG_DIR}/.record_dir_0.txt"
  result="$(run_one_match "${START_INDEX}" "${out_file}" "${record_dir_file}")"
  exit_code="${result%%|*}"
  out_file="${result#*|}"
  echo "Log: ${out_file}"
  rmdir "${LOG_DIR}" 2>/dev/null || true
  exit "${exit_code}"
fi

echo "Starting batch: ${BATCH_NAME}"
echo "Temporary logs (failed startup only): ${LOG_DIR}"
echo ""

FAILED_MATCHES=0
RESULT_DIR="${LOG_DIR}/results"
mkdir -p "${RESULT_DIR}"

cleanup_jobs() {
  local pids
  pids="$(jobs -rp)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    wait 2>/dev/null || true
  fi
}
trap cleanup_jobs INT TERM

for ((i = 0; i < TOTAL_MATCHES; i++)); do
  wait_for_slot
  match_index=$((START_INDEX + i))
  out_file="${LOG_DIR}/fg_run_${match_index}.log"
  record_dir_file="${LOG_DIR}/.record_dir_${match_index}.txt"
  result_file="${RESULT_DIR}/${match_index}.txt"

  (
    result="$(run_one_match "${match_index}" "${out_file}" "${record_dir_file}")"
    echo "${match_index}|${result}" >"${result_file}"
  ) &
  echo "  Submitted match ${i} (pid $!)"
done

echo "Waiting for all batch jobs..."
wait || true

shopt -s nullglob
for result_file in "${RESULT_DIR}"/*.txt; do
  line="$(tr -d '\r\n' < "${result_file}")"
  idx="${line%%|*}"
  rest="${line#*|}"
  exit_code="${rest%%|*}"
  log_file="${rest#*|}"
  echo "[Job ${idx}] completed, exit=${exit_code}, log=${log_file}"
  if [[ "${exit_code}" != "0" ]]; then
    FAILED_MATCHES=$((FAILED_MATCHES + 1))
  fi
done
shopt -u nullglob

trap - INT TERM

if (( FAILED_MATCHES > 0 )); then
  echo "Batch finished, but ${FAILED_MATCHES} match(es) failed. Check the log files above." >&2
  exit 1
fi

# 结果已归档到 game_records 时，清理空临时目录
rm -rf "${LOG_DIR}" 2>/dev/null || true
echo "Batch finished successfully."

