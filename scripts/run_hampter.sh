#!/usr/bin/env bash
# Ejecuta Hampter usando .venv_hampter si existe, o python3 del sistema.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PATH="$PROJECT_ROOT/src/app.py"

if [[ -x "$PROJECT_ROOT/.venv_hampter/bin/python" ]]; then
  exec "$PROJECT_ROOT/.venv_hampter/bin/python" "$APP_PATH" "$@"
fi

exec python3 "$APP_PATH" "$@"
