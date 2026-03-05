#!/usr/bin/env bash
# Creates an isolated virtual environment using uv and installs all dependencies.
# Usage: bash setup_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== Setting up virtual environment with uv ==="

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    uv venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Install project in editable mode (pulls in all dependencies)
uv pip install -e "$SCRIPT_DIR" --python "$VENV_DIR/bin/python"

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo "  python run_backtest.py"
