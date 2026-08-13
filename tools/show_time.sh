#!/usr/bin/env bash
# Ubuntu 的 sh 是 dash；用 sh 调用时自动切到 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# Linux wrapper for tools/batch_time_stats.py (wall-clock / in-game time)
# Examples:
#   ./tools/show_time.sh
#   ./tools/show_time.sh --group-by strategy
#   ./tools/show_time.sh --list-matches
set -euo pipefail

export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${SCRIPT_DIR}/batch_time_stats.py"
cd "${REPO_ROOT}"

usable_python() {
  local exe="$1"
  [[ -n "${exe}" && -x "${exe}" ]] || return 1
  "${exe}" -c "import json" >/dev/null 2>&1
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
  echo "Python not found. Use the repo venv or: conda activate SC2" >&2
  exit 1
fi

exec "${PYTHON_EXE}" "${TARGET}" "$@"
