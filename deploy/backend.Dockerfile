# syntax=docker/dockerfile:1
ARG BUILD_IMAGE=ubuntu:24.04
ARG RUN_IMAGE=ubuntu:24.04
FROM ghcr.io/astral-sh/uv:0.12.14 AS uv

FROM ${BUILD_IMAGE} AS build
ARG ACCELERATOR=cpu
ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    RUNTIME_DIR=/opt/arena-runtimes PLAYWRIGHT_BROWSERS_PATH=/opt/browsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv build-essential git curl ca-certificates \
    cmake ninja-build pkg-config libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/arena/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --extra local --no-dev --python /usr/bin/python3 && \
    uv --no-config pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/$(if [ "$ACCELERATOR" = cuda ]; then echo cu128; else echo cpu; fi) \
    torch==2.11.0 torchvision==0.26.0
COPY backend/runtimes/marker/pyproject.toml backend/runtimes/marker/uv.lock runtimes/marker/
RUN uv sync --locked --no-dev --project runtimes/marker --python /usr/bin/python3
COPY backend/scripts/ scripts/
COPY deploy/build-runtimes.sh /tmp/build-runtimes.sh
ARG NATIVE_BUILD_JOBS=4
RUN bash /tmp/build-runtimes.sh "$ACCELERATOR"
RUN .venv/bin/python -m playwright install chromium

FROM ${RUN_IMAGE} AS runtime
ARG ACCELERATOR=cpu
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/arena/backend/.venv/bin:$PATH \
    DATA_DIR=/data RUNTIME_DIR=/opt/arena-runtimes \
    HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch XDG_CACHE_HOME=/cache \
    PLAYWRIGHT_BROWSERS_PATH=/opt/browsers \
    MARKER_PYTHON=/opt/arena/backend/runtimes/marker/.venv/bin/python \
    LLAMA_CPP_BINARY=/opt/arena-runtimes/llama/llama-server \
    ARENA_IMAGE_ACCELERATOR=$ACCELERATOR LOCAL_DEVICE=cpu MAX_LOADED_MODELS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev build-essential ca-certificates libgomp1 libgl1 \
    libglib2.0-0t64 libcurl4t64 openjdk-21-jre-headless \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/arena/backend
COPY --from=build /opt/arena/backend/.venv .venv
COPY --from=build /opt/arena/backend/runtimes/marker/.venv runtimes/marker/.venv
COPY --from=build /opt/arena-runtimes /opt/arena-runtimes
COPY --from=build /opt/browsers /opt/browsers
RUN python -m playwright install-deps chromium && rm -rf /var/lib/apt/lists/*
COPY backend/app/ app/
COPY backend/scripts/ scripts/
COPY backend/samples/ samples/
COPY deploy/check-install.py /opt/arena/check-install.py
COPY deploy/backend-entrypoint.sh /opt/arena/backend-entrypoint.sh
# SkyVision statically links ggml; preserve that component's notice too.
COPY deploy/licenses/GGML-LICENSE /opt/arena-runtimes/skyvision/GGML-LICENSE
RUN useradd --uid 10001 --create-home arena && mkdir -p /data /cache \
    && chown arena:arena /data /cache && chmod 700 /data /cache \
    && ln -s /cache /home/arena/.cache
ENV HOME=/home/arena DOCTR_CACHE_DIR=/cache/doctr \
    EASYOCR_MODULE_PATH=/cache/easyocr DOCLING_CACHE_DIR=/cache/docling \
    FONT_DIR=/cache/fonts FONT_PATH=/cache/fonts/GoNotoCurrent-Regular.ttf
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=4 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=12)"
ENTRYPOINT ["bash", "/opt/arena/backend-entrypoint.sh"]
# Outlast the frontend proxy's idle sockets, avoiding resets while polling results.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "65"]
