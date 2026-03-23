#!/usr/bin/env bash
set -euo pipefail

MODELS=(
    "mlx-community/Qwen2.5-14B-Instruct-4bit"
    "mlx-community/Qwen2.5-7B-Instruct-4bit"
)
MODELS_DIR="$HOME/.omlx/models"

# Install omlx via brew if not available
if ! command -v omlx &>/dev/null; then
    echo "Installing omlx via brew..."
    brew install omlx
fi

mkdir -p "$MODELS_DIR"

# Download models from HuggingFace
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME="${MODEL##*/}"
    DEST="$MODELS_DIR/$MODEL_NAME"
    echo "Downloading $MODEL -> $DEST"
    uvx hf download "$MODEL" --local-dir "$DEST"
done

echo "Done."
