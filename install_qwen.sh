#!/usr/bin/env bash
set -euo pipefail

# Each entry: "REPO|FILENAME_PREFIX|LOCAL_DIR"
MODELS=(
    "bartowski/Qwen2.5-7B-Instruct-GGUF|Qwen2.5-7B-Instruct|./models/qwen2.5-7b"
    "bartowski/Qwen2.5-14B-Instruct-GGUF|Qwen2.5-14B-Instruct|./models/qwen2.5-14b"
)
QUANTS=("Q4_K_M" "Q8_0" "f16")

for ENTRY in "${MODELS[@]}"; do
    IFS="|" read -r REPO PREFIX LOCAL_DIR <<< "$ENTRY"
    mkdir -p "$LOCAL_DIR"
    echo "=================================================="
    echo "Model: $PREFIX (repo: $REPO)"
    echo "=================================================="
    for Q in "${QUANTS[@]}"; do
        FILENAME="${PREFIX}-${Q}.gguf"
        echo "Downloading: $FILENAME"
        uvx hf download "$REPO" "$FILENAME" --local-dir "$LOCAL_DIR"
    done
    echo "Files in $LOCAL_DIR:"
    ls -lh "$LOCAL_DIR"
done

echo "Done."
