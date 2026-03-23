#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <backend>"
    echo "  backend: cuda | mps"
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
fi

BACKEND="$1"

case "$BACKEND" in
    cuda)
        CMAKE_FLAGS="-DGGML_CUDA=ON -DGGML_METAL=OFF"
        ;;
    mps)
        CMAKE_FLAGS="-DGGML_METAL=ON -DGGML_CUDA=OFF"
        ;;
    *)
        echo "Error: unknown backend '$BACKEND'"
        usage
        ;;
esac

cd "$(dirname "$0")/llama.cpp"

echo "Building llama.cpp with backend: $BACKEND"
cmake -B build $CMAKE_FLAGS
cmake --build build --config Release -j

echo "Downloading models..."
cd ..
"$(dirname "$0")/install_qwen.sh"
echo "Done."
