#!/usr/bin/env bash
# Native CUDA deployment for the arena's implemented vision models.
# Run on the Linux NVIDIA host; never run this against a Windows/ROCm environment.
set -euo pipefail
if [ "$(uname -s)" != Linux ]; then
  echo "This installer requires Linux with an NVIDIA GPU." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || { echo "Install the NVIDIA driver first." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
command -v gcc >/dev/null || { echo "Install build-essential (GCC) for Triton." >&2; exit 1; }
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/" >&2; exit 1; }
cd "$(dirname "$0")/../backend"

# The lock deliberately excludes torch: use the GPU-specific wheel set explicitly.
uv sync --locked --extra local --inexact --python 3.12
.venv/bin/python -m playwright install --with-deps chromium
uv pip install --python .venv/bin/python --reinstall-package torch --reinstall-package torchvision \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0

# CUDA wheels supply Triton. Check its actual compiler/kernel path, not just importability.
.venv/bin/python - <<'PY'
import sys
import torch
import triton
from torch.nn.attention.flex_attention import flex_attention
assert sys.platform == 'linux' and torch.cuda.is_available() and not torch.version.hip
q = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.bfloat16)
result = torch.compile(flex_attention)(q, q, q)
torch.cuda.synchronize()
assert torch.isfinite(result).all(), 'FlexAttention produced invalid values'
print('CUDA + Triton + FlexAttention ready:', torch.cuda.get_device_name(0), triton.__version__)
PY

RUNTIME_PROFILE=linux-nvidia .venv/bin/python scripts/verify-runtime.py
echo 'Set RUNTIME_PROFILE=linux-nvidia in the repo .env, then run:'
echo '  cd backend && .venv/bin/python scripts/verify-runtime.py --smoke'
echo '  .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000'
