#!/usr/bin/env bash
# Primera ejecucion: crea venv e instala dependencias automaticamente.
# Siguientes ejecuciones: abre la app directo.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PATH="$PROJECT_ROOT/src/app.py"
VENV_PATH="$PROJECT_ROOT/.venv_hampter"
PYTHON_BIN="$VENV_PATH/bin/python"
PIP_BIN="$VENV_PATH/bin/pip"
REQ_FILE="$PROJECT_ROOT/requirements.txt"
STAMP_FILE="$VENV_PATH/.requirements_installed"

if ! command -v python3 >/dev/null 2>&1; then
  echo "No se encontro python3. Instala Python 3.10+ y vuelve a intentar."
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creando entorno virtual..."
  python3 -m venv "$VENV_PATH"
fi

if [[ ! -f "$STAMP_FILE" || "$REQ_FILE" -nt "$STAMP_FILE" ]]; then
  echo "Instalando dependencias (solo la primera vez)..."
  "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
  "$PIP_BIN" install -r "$REQ_FILE"
  touch "$STAMP_FILE"
fi

exec "$PYTHON_BIN" "$APP_PATH" "$@"
