# Deployment & configuration

[← README](../README.md) · [Model presets](model-presets.md) · [Architecture & development](architecture.md)

Start with the [README quick start](../README.md#run-it-locally). This guide covers
the choices needed for a shared installation or a custom environment.

## Containers

The CPU and NVIDIA variants share a production Next.js frontend and a FastAPI
backend. The backend image includes both Python environments, Java 21, Tesseract
(English/German), Chromium, llama.cpp, SkyVision and Unlimited-OCR's patched runtime.
Dependencies are locked; native runtimes use pinned sources or verified releases.
Model weights download on first use and are cached separately.

Only the frontend is published, on `127.0.0.1:3000` by default. The backend is
reachable within Compose as `backend:8000`. Users need only a browser.

### Check the installation

```bash
docker compose exec backend python /opt/arena/check-install.py --smoke
```

This checks dependencies, method availability, browser rendering and a bundled-page
pypdf extraction without document-processing API calls. For NVIDIA, include both
`-f compose.yaml -f compose.nvidia.yaml`; the check also runs CUDA and Triton kernels.
Packaging checks on a CPU host do not validate GPU inference.

The CPU image enforces CPU execution. The NVIDIA image requires a compatible GPU
and driver, using CUDA 12.8 / PyTorch 2.11.0; a 24 GB GPU is a useful starting point,
with actual memory needs depending on model and page complexity. NVIDIA inference
must still be verified on the deployment host. CPU mode remains selectable there.

CUDA builds compile Unlimited's patched runtime. More build parallelism is optional:

```bash
docker compose -f compose.yaml -f compose.nvidia.yaml build --build-arg NATIVE_BUILD_JOBS=8 backend
```

Allow extra RAM and disk for compilation and build caches. Building and distributing
images once avoids repeating this work on every user's machine.

### Storage and restarts

| Volume | Contents |
| --- | --- |
| `arena-data` → `/data` | Uploads, rendered pages, outputs, SQLite and UI-saved API keys |
| `model-cache` → `/cache` | Hugging Face weights and parser/browser-support caches |

Runtime executables live in the image under `/opt/arena-runtimes`. `docker compose stop`
and `docker compose down` preserve the volumes; **`down -v` deletes them**. Stop active
runs before backing up or migrating data. Container storage is separate from the
checkout's native `data/` directory. Private documents, `.env` files and `exports/`
are excluded from image build contexts.

New containers have an empty sample gallery. Copy intended files into
`/data/sample-docs/`, then [prepare saved results](architecture.md#saved-sample-gallery).

### Ports and shared access

For a different port, create a gitignored `.env.deployment`:

```dotenv
ARENA_PORT=3030
CORS_ORIGINS=["http://localhost:3030","http://127.0.0.1:3030"]
```

```bash
docker compose --env-file .env.deployment up -d
```

For a shared URL, set `ARENA_BIND` and `CORS_ORIGINS` to match your reverse proxy.
Matching origins are required for saving API keys. Put authentication and TLS in
front of shared deployments: storage, history and credentials are currently shared
across the installation. There are no separate user workspaces.

### API keys

The judging confirmation dialog accepts missing LlamaParse/Mistral keys and the
LlamaParse region. **Save API keys** persists them; typing **OK** separately authorizes
the run. Saving alone starts no processing. Existing keys are not shown in the browser.

Keys are stored unencrypted in `/data/judging-keys.json`, with restricted filesystem
permissions; protect this volume and its backups. Native installs use `DATA_DIR`.
Environment/`.env` keys take precedence. The save form fills missing keys; operators
rotate configured keys in their environment or the server-side credentials file.

Supported environment names are `LLAMA_CLOUD_API_KEY`, `MISTRAL` (or
`MISTRAL_API_KEY`) and `LLAMA_CLOUD_BASE_URL`. The base URL defaults to
`https://api.cloud.llamaindex.ai`; EU keys use `https://api.cloud.eu.llamaindex.ai`.
Docker does **not** automatically pass the repo's `.env` keys into the backend:
supply them explicitly through a Compose environment override or Kubernetes Secrets,
or use the UI. [Data transfers and judging rules](architecture.md#quality-judging).

### Build once, distribute images

The repo includes a manual **Actions → Build container images** workflow. Choose
CPU, NVIDIA or both; the frontend is always included. Publishing to the repository's
GitHub Container Registry is opt-in. CPU parsing checks and NVIDIA packaging checks
run before publication; the latter do not replace a GPU-host check.

After publishing, set these values in `.env.deployment` using your registry namespace:

```dotenv
ARENA_FRONTEND_IMAGE=ghcr.io/OWNER/REPO/frontend:latest
ARENA_BACKEND_IMAGE=ghcr.io/OWNER/REPO/backend:cpu
```

```bash
docker compose --env-file .env.deployment pull
docker compose --env-file .env.deployment up -d --no-build
```

For NVIDIA, use `backend:nvidia` and both Compose files. Configure registry access
before distributing images; use immutable digests for controlled releases.

## Kubernetes

Use the same two images with a frontend Service/Ingress and a backend Service.

- Keep **one backend replica and one Uvicorn worker**: SQLite and the in-process
  extraction/judging queues are designed for a single backend process.
- Mount persistent volumes at `/data` and `/cache`, writable by UID/GID **10001**.
  Request one `nvidia.com/gpu` for the NVIDIA variant and configure the NVIDIA runtime.
- Name the backend Service **`backend`**, port **8000**. For another address, build
  the frontend with `--build-arg BACKEND_URL=http://your-service:8000` and set its
  runtime `BACKEND_URL` to the same address. The API proxy is fixed at build time.
- Use `/api/health` on the backend for health probes; add authenticated TLS ingress
  in front of the frontend and keep the backend internal.

Stopping containers does not stop GPU-VM billing. On-demand GPU provisioning and
shutdown require separate orchestration; these images do not implement it.

## Windows / AMD

The [native helper](../scripts/setup-windows.ps1) targets Radeon 8060S / Ryzen AI Max
(`gfx1151`). Prerequisites: signed Python 3.13 with `py`, Node.js 22 LTS, `uv`, and
an AMD driver compatible with the ROCm wheel set. Other AMD families need their
own compatible environment. Docker Desktop's CPU image does not use the AMD GPU.

Run the two commands in the [README](../README.md#run-it-locally). Setup creates the
main and Marker environments, installs portable runtimes and Chromium, builds the
frontend, and records the profile in `data/installation.json`. It preserves an
existing `.env`; Tesseract's installer may request elevation. `-CheckOnly` checks
prerequisites without modifying environments. `-Profile cpu` selects native CPU setup.

`start-windows.ps1` launches hidden processes, prints their PIDs and writes logs to
`data/logs/`. It refuses occupied ports. After runs finish, stop each process tree
with `taskkill /PID <PID> /T /F` using those PIDs, then rerun the launcher to restart.
The command-scoped execution policy does not change Windows' permanent policy.

If Windows blocks Python, ROCm or PDFium, inspect the backend logs. Use signed Python;
the app does not change OS security settings. Saved results can remain readable even
when a native library is unavailable, while new extraction or rendering may fail.

## Configuration

Native settings come from environment variables and repo/backend `.env` files.
The minimal [`.env.example`](../.env.example) covers optional judging keys only.
Restart the backend after changes. Full defaults are in
[`Settings`](../backend/app/config.py); containers and launchers override some defaults.

| Setting | Default / purpose |
| --- | --- |
| `RUNTIME_PROFILE` | `auto`; explicit `windows-amd` / `linux-nvidia` must match the host |
| `LOCAL_DEVICE`, `LOCAL_DTYPE` | `auto`; force CPU/CUDA or a supported tensor precision |
| `CPU_THREADS`, `CPU_DTYPE` | `8`, `auto`; CPU VLMs use supported BF16 or float32 |
| `MAX_LOADED_MODELS`, `KEEP_MODELS_LOADED` | `2`, `true` natively; containers use `1`, `false` |
| `LOCAL_MAX_BATCH` | `8`; KDL-region / MinerU-block batching |
| `LOCAL_INFERENCE` | `true`; permits local model loading |
| `<MODEL>_ENDPOINT_URL`, `<MODEL>_SERVED_MODEL`, `VLLM_API_KEY` | Optional remote serving; see [model presets](model-presets.md#remote-model-servers) |
| `DATA_DIR`, `RUNTIME_DIR` | Native `data/` and `data/runtimes/`; container `/data` and `/opt/arena-runtimes` |
| `MARKER_PYTHON`, `JAVA_CMD`, `LLAMA_CPP_BINARY`, `TESSERACT_CMD` | Optional executable overrides; otherwise auto-discovered |
| `TESSERACT_LANG` | `eng`; use `deu+eng` with German language data installed |
| `SKYVISION_BINARY`, `SKYVISION_THREADS`, `SKYVISION_TIMEOUT_S` | Auto-discovered binary, physical-core count, `1200` seconds |
| `OVISOCR3_CROSSPAGE_TABLES`, `OVISOCR3_BROWSER_EXECUTABLE` | Table merging on by default; optional Chromium path |
| `UNLIMITED_OCR_2_MAX_PAGES`, `UNLIMITED_OCR_2_CONTEXT_LENGTH`, `UNLIMITED_OCR_2_MAX_TOKENS` | `16`, `32768`, `24000`; the UI batch limit is still 12 unique pages |
| `MAX_UPLOAD_MB`, `MAX_PAGES_PER_EXPERIMENT` | `100`, `50`; the batch API additionally limits each batch to 12 unique pages |
| `ALLOW_UNSTABLE_ROCM_MODELS` | `false`; unsafe native paths remain blocked; Vulkan alternatives do not require this override |
| `CORS_ORIGINS` | JSON list of allowed frontend origins |
| `BACKEND_URL` | Frontend proxy/server API address; native default `http://127.0.0.1:8000` |

CPU-only runs ignore remote model endpoints and disable GPU offloading, release
models after use, and preserve their mode on rerun. The five slow CPU exclusions
are listed in the README. Optional API judging remains external in CPU mode.
