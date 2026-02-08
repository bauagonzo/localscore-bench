#!/usr/bin/env bash
# setup-backends.sh — Download llama-bench binaries for Vulkan and CUDA backends
#
# Downloads the latest release from:
#   - ggml-org/llama.cpp       → Vulkan build (Linux x64)
#   - ai-dock/llama.cpp-cuda   → CUDA build (Linux x64)
#
# Extracts llama-bench + required shared libraries into:
#   ./backends/vulkan/
#   ./backends/cuda/
#
# Usage:
#   ./setup-backends.sh              # download latest for both
#   ./setup-backends.sh --vulkan     # vulkan only
#   ./setup-backends.sh --cuda       # cuda only
#   ./setup-backends.sh --tag b7966  # specific version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKENDS_DIR="${SCRIPT_DIR}/backends"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Defaults
DO_VULKAN=true
DO_CUDA=true
TAG=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vulkan)  DO_VULKAN=true; DO_CUDA=false; shift ;;
        --cuda)    DO_VULKAN=false; DO_CUDA=true; shift ;;
        --tag)     TAG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--vulkan] [--cuda] [--tag <version>]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

get_latest_tag() {
    local repo="$1"
    curl -sf "https://api.github.com/repos/${repo}/releases/latest" | jq -r '.tag_name'
}

download_and_extract() {
    local url="$1"
    local dest="$2"
    local strip="$3"  # strip leading path components

    echo "  Downloading: $(basename "$url")"
    mkdir -p "$dest"
    curl -sL "$url" | tar xz -C "$TMP_DIR"

    # Find the extracted directory
    local src_dir
    src_dir=$(find "$TMP_DIR" -maxdepth 1 -mindepth 1 -type d | head -1)

    # Copy llama-bench + all .so files
    cp "$src_dir/llama-bench" "$dest/"
    chmod +x "$dest/llama-bench"
    find "$src_dir" -name '*.so*' -exec cp -a {} "$dest/" \;

    # Clean tmp for next use
    rm -rf "$TMP_DIR"/*

    echo "  Installed to: $dest"
}

# Resolve tag
if [[ -z "$TAG" ]]; then
    echo "Resolving latest release tags..."
    if $DO_VULKAN; then
        VULKAN_TAG=$(get_latest_tag "ggml-org/llama.cpp")
        echo "  ggml-org/llama.cpp: $VULKAN_TAG"
    fi
    if $DO_CUDA; then
        CUDA_TAG=$(get_latest_tag "ai-dock/llama.cpp-cuda")
        echo "  ai-dock/llama.cpp-cuda: $CUDA_TAG"
    fi
else
    VULKAN_TAG="$TAG"
    CUDA_TAG="$TAG"
    echo "Using tag: $TAG"
fi

# Download Vulkan
if $DO_VULKAN; then
    echo ""
    echo "=== Vulkan backend (ggml-org/llama.cpp ${VULKAN_TAG}) ==="
    VULKAN_URL="https://github.com/ggml-org/llama.cpp/releases/download/${VULKAN_TAG}/llama-${VULKAN_TAG}-bin-ubuntu-vulkan-x64.tar.gz"
    VULKAN_DIR="${BACKENDS_DIR}/vulkan"
    rm -rf "$VULKAN_DIR"
    download_and_extract "$VULKAN_URL" "$VULKAN_DIR" 1

    # Write version marker
    echo "$VULKAN_TAG" > "$VULKAN_DIR/VERSION"
    echo "  Version: $VULKAN_TAG"
fi

# Download CUDA
if $DO_CUDA; then
    echo ""
    echo "=== CUDA backend (ai-dock/llama.cpp-cuda ${CUDA_TAG}) ==="

    # Find the right CUDA asset — pick the highest CUDA version available
    echo "  Finding best CUDA asset..."
    CUDA_ASSET=$(curl -sf "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/tags/${CUDA_TAG}" \
        | jq -r '.assets[].name' \
        | grep '\.tar\.gz$' \
        | sort -t'-' -k4 -V \
        | tail -1)

    if [[ -z "$CUDA_ASSET" ]]; then
        echo "  ERROR: No CUDA tarball found for tag ${CUDA_TAG}"
        exit 1
    fi

    CUDA_URL="https://github.com/ai-dock/llama.cpp-cuda/releases/download/${CUDA_TAG}/${CUDA_ASSET}"
    CUDA_DIR="${BACKENDS_DIR}/cuda"
    rm -rf "$CUDA_DIR"
    download_and_extract "$CUDA_URL" "$CUDA_DIR" 1

    # Write version marker
    echo "$CUDA_TAG" > "$CUDA_DIR/VERSION"
    echo "  CUDA asset: $CUDA_ASSET"
    echo "  Version: $CUDA_TAG"
fi

echo ""
echo "Done! Backends ready in: $BACKENDS_DIR"
echo ""
ls -la "$BACKENDS_DIR"/*/llama-bench 2>/dev/null || true
