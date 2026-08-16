#!/bin/sh

export PYTHONIOENCODING="utf-8"

# 当前脚本所在目录
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# 项目根目录：脚本目录的上一级
ROOT=$(dirname -- "$SCRIPT_DIR")

SCRIPT="$SCRIPT_DIR/organize_batch_json.py"

PYTHON=""

# 1. 优先使用项目自己的 venv
VENV_PYTHON="$ROOT/venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
fi

# 2. 如果当前已经激活 virtualenv，则使用它
if [ -z "$PYTHON" ] && [ -n "${VIRTUAL_ENV:-}" ]; then
    ACTIVE_PYTHON="$VIRTUAL_ENV/bin/python"
    if [ -x "$ACTIVE_PYTHON" ]; then
        PYTHON="$ACTIVE_PYTHON"
    fi
fi

# 3. 尝试系统 Python
if [ -z "$PYTHON" ]; then
    for name in python python3; do
        if command -v "$name" >/dev/null 2>&1; then
            PYTHON=$(command -v "$name")
            break
        fi
    done
fi

# 4. 找不到 Python
if [ -z "$PYTHON" ]; then
    echo "Python not found. Use the repo venv or install Python first." >&2
    exit 1
fi

cd "$ROOT" || exit 1

exec "$PYTHON" "$SCRIPT" "$@"