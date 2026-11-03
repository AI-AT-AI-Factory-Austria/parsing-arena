<#
.SYNOPSIS
  Install the backend dependencies plus a PyTorch build for local inference.

.PARAMETER Flavor
  cuda          NVIDIA GPU (CUDA 12.8 wheels from download.pytorch.org)
  rocm-gfx1151  AMD Ryzen AI Max (Radeon 8060S, gfx1151) via AMD's ROCm wheels
  cpu           CPU only; works everywhere but is slow (minutes per page)

.EXAMPLE
  ./scripts/install-torch.ps1 -Flavor cuda
#>
param(
  [ValidateSet("cuda", "rocm-gfx1151", "cpu")]
  [string]$Flavor = "cuda"
)

$ErrorActionPreference = "Stop"
$indexes = @{
  "cuda"         = "https://download.pytorch.org/whl/cu128"
  "rocm-gfx1151" = "https://repo.amd.com/rocm/whl/gfx1151/"
  "cpu"          = "https://download.pytorch.org/whl/cpu"
}
$index = if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { $indexes[$Flavor] }

Push-Location (Join-Path $PSScriptRoot "..\backend")
try {
  # --inexact keeps an already installed torch; plain `uv sync` would remove it.
  uv sync --extra local --inexact
  if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

  # The extra index takes priority over PyPI, so torch comes from it and the
  # remaining dependencies from PyPI.
  uv pip install --extra-index-url $index torch torchvision
  if ($LASTEXITCODE -ne 0) { throw "installing torch failed" }

  .venv\Scripts\python.exe -c "import torch; gpu = torch.cuda.is_available(); print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if gpu else 'none (CPU)')"
}
finally {
  Pop-Location
}
