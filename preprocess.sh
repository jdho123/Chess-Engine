#!/bin/bash

set -e

# ==========================================
# Configuration 
# ==========================================
VENV_DIR=".venv"
INPUT_FILE="data/raw/lichess_db_eval.jsonl.zst"  
OUTPUT_DIR="data/processed" 
TARGET_MB=100
BATCH_SIZE=256
DOWNLOAD_LINK="https://database.lichess.org/lichess_db_eval.jsonl.zst"
# ==========================================

cd "$(dirname "$0")" || exit 1

INPUT_DIR=$(dirname "$INPUT_FILE")

mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file not found. Downloading from $DOWNLOAD_URL..."
    curl -L -o "$INPUT_FILE" "$DOWNLOAD_URL"
else
    echo "Input file '$INPUT_FILE' already exists. Skipping download."
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment '$VENV_DIR' not found in the current directory."
    echo "Please create it using: python3 -m venv $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

python -m preprocess.main \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --target_mb "$TARGET_MB" \
    --batch_size "$BATCH_SIZE" \

deactivate