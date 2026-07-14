#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Override when necessary, for example: PYTHON=python3.12 bash setup_venv.sh
PYTHON_COMMAND="${PYTHON:-python3}"

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
    echo "Python 3.10+ was not found. Install Python and check PATH." >&2
    exit 1
fi

if ! "$PYTHON_COMMAND" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10 or newer is required ($("$PYTHON_COMMAND" --version 2>&1))." >&2
    exit 1
fi

echo "Using $($PYTHON_COMMAND --version)"

VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    "$PYTHON_COMMAND" -m venv "$VENV_DIR"
fi

# The eager strategy also fixes incompatible packages left in an existing venv.
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install --upgrade --upgrade-strategy eager -r requirements.txt
"$VENV_PYTHON" -m pip check

printf '\nSetup complete. Open notebooks/resource_comparison.ipynb in VS Code.\n'
printf 'If VS Code does not select the kernel automatically, choose:\n  %s\n' "$VENV_PYTHON"
printf 'This script does not launch browser-based Jupyter.\n'

