#!/usr/bin/env bash
# Install the backend dependencies plus a PyTorch build for local inference.
#
#   ./scripts/install-torch.sh cuda          NVIDIA GPU (CUDA 12.8 wheels)
#   ./scripts/install-torch.sh rocm-gfx1151  AMD Ryzen AI Max (Radeon 8060S) via AMD's ROCm wheels
#   ./scripts/install-torch.sh cpu           CPU only; works everywhere but is slow
#   ./scripts/install-torch.sh mac           Apple silicon (Metal/MPS build from PyPI)
#
# Set TORCH_INDEX_URL to use any other PyTorch wheel index.
set -euo pipefail

flavor="${1:-cuda}"
case "$flavor" in
  cuda) index="https://download.pytorch.org/whl/cu128" ;;
  rocm-gfx1151) index="https://repo.amd.com/rocm/whl/gfx1151/" ;;
  cpu) index="https://download.pytorch.org/whl/cpu" ;;
  mac) index="" ;;
  *) echo "unknown flavor '$flavor' (use cuda, rocm-gfx1151, cpu or mac)" >&2; exit 1 ;;
esac
index="${TORCH_INDEX_URL:-$index}"

cd "$(dirname "$0")/../backend"

# --inexact keeps an already installed torch; plain `uv sync` would remove it.
uv sync --extra local --inexact

if [ -n "$index" ]; then
  # The extra index takes priority over PyPI, so torch comes from it.
  uv pip install --extra-index-url "$index" torch torchvision
else
  uv pip install torch torchvision
fi

.venv/bin/python -c "import torch; gpu = torch.cuda.is_available(); mps = torch.backends.mps.is_available(); print('torch', torch.__version__, '| accelerator:', torch.cuda.get_device_name(0) if gpu else ('Apple MPS' if mps else 'none (CPU)'))"
