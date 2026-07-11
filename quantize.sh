#!/bin/bash

set -e

# ==========================================
# Configuration 
# ==========================================
VENV_DIR=".venv"
CONFIG_PATH="configs/quantize.yaml"
# ==========================================

cd "$(dirname "$0")" || exit 1

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment '$VENV_DIR' not found in the current directory."
    echo "Please create it using: python3 -m venv $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

python -m quantize.main \
    --config "$CONFIG_PATH" \

deactivate