#!/usr/bin/env bash
# =============================================================================
# export_sft_jsonl.sh
# Export Victory decision turns from game_records_json to SFT jsonl.
# Edit the config below, then run: ./tools/export_sft_jsonl.sh
# Use empty arrays for "no filter" on that axis, e.g. MODELS=()
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Paths (empty string = repo default)
# -----------------------------------------------------------------------------
RECORDS_DIR=""   # default: <repo>/game_records_json
OUT_PATH=""      # default: <repo>/sft_data/sft.jsonl

# -----------------------------------------------------------------------------
# 2. Filters (add/remove strings; empty array = no filter)
# Options:
#   models:       kimi-k2.5 / qwen3-32b / qwen3.5-27b / deepseek-v4-flash
#   strategies:   marine / tank / battlecruiser
#   races:        terran / zerg / protoss
#   difficulties: mediumhard / hard / harder / veryhard
#   styles:       macro / rush / timing
# -----------------------------------------------------------------------------
MODELS=(
  "deepseek-v4-flash"
)

STRATEGIES=(
  "tank"
)

RACES=(
  "terran"
)

DIFFICULTIES=(
  "mediumhard",
  "hard",
  "harder",
  "veryhard"
)

STYLES=(
  "macro"
)

# -----------------------------------------------------------------------------
# 3. Export rules
# -----------------------------------------------------------------------------
MIN_WINRATE=""            # e.g. 0.7 ; empty = no limit
TOOL_MODE=""              # e.g. json ; empty = any
LIMIT=0                   # 0 = no limit
DRY_RUN=0                 # 1 = count only
INCLUDE_LOSSES=0          # 1 = keep non-victory matches
ALLOW_UNACCEPTED=0        # 1 = keep accepted!=True turns

# =============================================================================
# Runtime (usually no need to edit)
# =============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT}/tools/export_sft_jsonl.py"

PYTHON=""
if [[ -x "${ROOT}/venv/bin/python" ]]; then
  PYTHON="${ROOT}/venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "Python not found. Create ./venv or install python3 first." >&2
  exit 1
fi

ARGS=()
if [[ -n "${RECORDS_DIR}" ]]; then
  ARGS+=(--records-dir "${RECORDS_DIR}")
fi
if [[ -n "${OUT_PATH}" ]]; then
  ARGS+=(--out "${OUT_PATH}")
fi

append_filter() {
  local flag="$1"
  shift
  local value
  for value in "$@"; do
    if [[ -n "${value}" ]]; then
      ARGS+=("${flag}" "${value}")
    fi
  done
}

append_filter --model "${MODELS[@]+"${MODELS[@]}"}"
append_filter --strategy "${STRATEGIES[@]+"${STRATEGIES[@]}"}"
append_filter --race "${RACES[@]+"${RACES[@]}"}"
append_filter --difficulty "${DIFFICULTIES[@]+"${DIFFICULTIES[@]}"}"
append_filter --style "${STYLES[@]+"${STYLES[@]}"}"

if [[ -n "${MIN_WINRATE}" ]]; then
  ARGS+=(--min-winrate "${MIN_WINRATE}")
fi
if [[ -n "${TOOL_MODE}" ]]; then
  ARGS+=(--tool-mode "${TOOL_MODE}")
fi
if [[ "${LIMIT}" -gt 0 ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  ARGS+=(--dry-run)
fi
if [[ "${INCLUDE_LOSSES}" -eq 1 ]]; then
  ARGS+=(--include-losses)
fi
if [[ "${ALLOW_UNACCEPTED}" -eq 1 ]]; then
  ARGS+=(--allow-unaccepted)
fi

# Extra CLI args still work, e.g. ./tools/export_sft_jsonl.sh --dry-run
ARGS+=("$@")

join_by() {
  local IFS="$1"
  shift
  echo "$*"
}

echo "Python : ${PYTHON}"
echo "Config : model=[$(join_by ',' "${MODELS[@]+"${MODELS[@]}"}")] strategy=[$(join_by ',' "${STRATEGIES[@]+"${STRATEGIES[@]}"}")] race=[$(join_by ',' "${RACES[@]+"${RACES[@]}"}")] difficulty=[$(join_by ',' "${DIFFICULTIES[@]+"${DIFFICULTIES[@]}"}")] style=[$(join_by ',' "${STYLES[@]+"${STYLES[@]}"}")]"
echo "Args   : ${ARGS[*]}"

cd "${ROOT}"
exec "${PYTHON}" "${PY_SCRIPT}" "${ARGS[@]}"
