#!/usr/bin/env bash
set -euo pipefail
# Fail clearly if a CUDA image is started without GPU passthrough. CPU images
# cannot accidentally start remote/GPU extraction even if an env file asks for it.
if [ "${ARENA_IMAGE_ACCELERATOR:-cpu}" = cpu ]; then
  export LOCAL_DEVICE=cpu RUNTIME_PROFILE=auto CUDA_VISIBLE_DEVICES=-1
else
  export LOCAL_DEVICE=cuda RUNTIME_PROFILE=linux-nvidia
  python -c "import torch; assert torch.cuda.is_available() and not torch.version.hip, 'NVIDIA GPU unavailable: install the host driver + NVIDIA Container Toolkit and use compose.nvidia.yaml.'"
fi
umask 077
exec "$@"
