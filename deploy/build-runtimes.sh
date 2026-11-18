#!/usr/bin/env bash
# Build once into the image, outside the persistent documents/model-cache volumes.
set -euo pipefail
accelerator="${1:?cpu or cuda}"
case "$accelerator" in cpu|cuda) ;; *) echo "Unknown accelerator" >&2; exit 1 ;; esac
export RUNTIME_DIR=/opt/arena-runtimes
export CMAKE_BUILD_PARALLEL_LEVEL="${NATIVE_BUILD_JOBS:-4}"
# Explicit targets work on a build host without a GPU; no -march=native assumptions.
export CMAKE_CUDA_ARCHITECTURES='75;80;86;89;90;120'
mkdir -p "$RUNTIME_DIR/llama"
if [ "$accelerator" = cuda ]; then
  # Official CUDA release, pinned by version AND checksum. Avoid recompiling its
  # unmodified kernels on every CI builder; the patched Unlimited runtime is separate.
  curl --fail --location --retry 3 --output /tmp/llama-cuda.tar.gz \
    https://github.com/ggml-org/llama.cpp/releases/download/b12110/llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz
  echo '4af1e1da23661db4a5b84b29aae16ebc053df689c5b8b1427c4670f2c6406d8c  /tmp/llama-cuda.tar.gz' | sha256sum --check
  tar -xzf /tmp/llama-cuda.tar.gz -C "$RUNTIME_DIR/llama" --strip-components=1
  "$RUNTIME_DIR/llama/llama-server" --version
  rm /tmp/llama-cuda.tar.gz
else
git clone --quiet --depth 1 --branch b12110 https://github.com/ggml-org/llama.cpp.git /tmp/llama-src
test "$(git -C /tmp/llama-src rev-parse HEAD)" = cd959bae99ff4bed22bef2604f45241c09bbb8c8
cmake -S /tmp/llama-src -B /tmp/llama-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
  -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
cmake --build /tmp/llama-build --target llama-server --parallel "$CMAKE_BUILD_PARALLEL_LEVEL"
cp -a /tmp/llama-build/bin/. "$RUNTIME_DIR/llama/"
cp /tmp/llama-src/LICENSE "$RUNTIME_DIR/llama/LICENSE"
fi
uv run --no-project --script scripts/install-skyvision.py
extra=()
if [ "$accelerator" = cpu ]; then extra+=(--cpu); fi
uv run --no-project --script scripts/install-unlimited.py "${extra[@]}" --skip-device-check
# Retain only runtime binaries, provenance and licenses in the final image.
.venv/bin/python - <<'PY'
import json
import os
import shutil
from pathlib import Path
root = Path(os.environ['RUNTIME_DIR'])
unlimited = root / 'unlimited-ocr2'
info_path = unlimited / 'build-info.json'
info = json.loads(info_path.read_text())
source_bin = (unlimited / info['binary']).parent
shutil.copytree(source_bin, unlimited / 'bin', dirs_exist_ok=True)
info['binary'] = 'bin/llama-server'
info_path.write_text(json.dumps(info, indent=2))
for folder in (unlimited, root / 'skyvision'):
    for child in folder.iterdir():
        if child.is_dir() and child.name.startswith('source'):
            for license_file in child.glob('LICENSE*'):
                shutil.copy2(license_file, folder / license_file.name)
            shutil.rmtree(child)
        elif child.suffix == '.zip':
            child.unlink()
PY
