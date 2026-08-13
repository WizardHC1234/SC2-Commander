#!/usr/bin/env bash
# Ubuntu 的 sh 是 dash；用 sh 调用时自动切到 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# SC2-Commander 单局 demo（Linux）
set -euo pipefail

export PYTHONIOENCODING=utf-8
export LANG="${LANG:-en_US.UTF-8}"
# 本地 SC2 / 内网 vLLM 不走代理，避免 httpx SOCKS 缺依赖或误走代理
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------- 对局参数（按需改） ----------------
MAP_NAME="KairosJunctionLE"
ENEMY_RACE="terran"
ENEMY_DIFFICULTY="hard"
ENEMY_BUILD="macro"
BOT_RACE="terran"
FORCE_STRATEGY="tank"
COMMANDER_MODEL="qwen3.5-9b"
# COMMANDER_MODEL="qwen3-32b"
# COMMANDER_MODEL="deepseek-v4-flash"
# COMMANDER_MODEL="kimi-k2.5"

# ---------------- 环境 ----------------
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
  echo "Trying Python: ${cand}"
  if usable_python "${cand}"; then
    PYTHON_EXE="${cand}"
    break
  fi
done

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "No usable Python found (need import openai)." >&2
  echo "Hint: conda activate SC2" >&2
  exit 1
fi

echo "Using Python: ${PYTHON_EXE}"
echo "Repo root  : ${REPO_ROOT}"
echo "SC2PATH    : ${SC2PATH}"
echo "Strategy   : ${FORCE_STRATEGY}"
echo "Model      : ${COMMANDER_MODEL}"

exec "${PYTHON_EXE}" "${REPO_ROOT}/run_vs_ai.py" \
  --my-bot-name commander \
  --map-name "${MAP_NAME}" \
  --bot-race "${BOT_RACE}" \
  --enemy-race "${ENEMY_RACE}" \
  --enemy-difficulty "${ENEMY_DIFFICULTY}" \
  --enemy-build "${ENEMY_BUILD}" \
  --force-strategy "${FORCE_STRATEGY}" \
  --commander-model "${COMMANDER_MODEL}"
