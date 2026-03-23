#!/bin/bash
set -euo pipefail

MODELS=(
    "qwen2.5:14b-instruct-q4_K_M"
    "qwen2.5:7b-instruct-q4_K_M"
)
OLLAMA_BIN="$HOME/.local/bin/ollama"
STARTED_SERVER=false

cleanup() {
    if $STARTED_SERVER && [ -n "${OLLAMA_PID:-}" ]; then
        kill "$OLLAMA_PID" 2>/dev/null && wait "$OLLAMA_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# Download ollama if not installed
if ! command -v ollama &>/dev/null && [ ! -x "$OLLAMA_BIN" ]; then
    echo "Installing ollama..."
    curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar x --zstd -C ~/.local/
fi

OLLAMA="$(command -v ollama 2>/dev/null || echo "$OLLAMA_BIN")"
[ -x "$OLLAMA" ] || { echo "Error: ollama not found" >&2; exit 1; }

# Start server if not already running
if ! "$OLLAMA" list &>/dev/null; then
    "$OLLAMA" serve &>/dev/null &
    OLLAMA_PID=$!
    STARTED_SERVER=true
    for i in $(seq 1 15); do
        "$OLLAMA" list &>/dev/null && break
        kill -0 "$OLLAMA_PID" 2>/dev/null || { echo "Error: server crashed" >&2; exit 1; }
        sleep 1
    done
    "$OLLAMA" list &>/dev/null || { echo "Error: server timeout" >&2; exit 1; }
fi

# Pull model
for MODEL in "${MODELS[@]}"; do
    echo "Pulling $MODEL..."
    "$OLLAMA" pull "$MODEL"
done
