# OCR Arena

Evaluate the open-source OCR models that best suit your documents.

Drop in a PDF or image. OCR Arena parses it with open OCR models of different sizes, times
every run, and shows each model's output next to the original page and next to
**LlamaParse Agentic Plus** as a reference. Scores, parameter counts and licenses come from the
[ParseBench](https://github.com/run-llama/ParseBench) leaderboard.

| Model | Publisher | Params | ParseBench | License | In the arena |
| --- | --- | --- | --- | --- | --- |
| LlamaParse Agentic Plus (reference) | LlamaIndex | Hosted API | 86.7 | Commercial API | yes (needs an API key) |
| rakedoc-nano | CloudRaker | 1.2B | 77.2 | AGPL-3.0 | yes |
| Aeddix Alpine OCR | Aeddix Labs | 1.2B | 76.8 | Apache-2.0 | yes |
| florin-parser-nano | Florin | 1.2B | 76.7 | AGPL-3.0 | coming soon |
| Nemotron-3.5-Parse | NVIDIA | 0.631B | 75.1 | OpenMDW-1.1 | coming soon |
| LightOnOCR-3-0.8B | LightOn | 0.8B class | 74.7 | Apache-2.0 | coming soon |
| MinerU2.5-Pro-2605 | OpenDataLab | 1.2B | 69.3 | Apache-2.0 (weights) | coming soon |
| Infinity-Parser2-Flash | inftech.ai / infly | ≈2B | 68.6 | Apache-2.0 | coming soon |

rakedoc-nano, Aeddix Alpine OCR and florin-parser-nano are marked *unverified* on the leaderboard.
Both live models run through the KDL two-stage pipeline (layout detection, then region
recognition) that their ParseBench scores require. The pipeline code is vendored from ParseBench.

## How it works

- **Backend** (`backend/`): FastAPI, SQLite and a single background worker. Open models run
  **one after another** on the same hardware so their timings are comparable. Model loading is
  measured separately and is not counted as parse time. LlamaParse runs in parallel through the
  official `llama-cloud` SDK, with ParseBench's settings (`tier=agentic_plus`, `version=latest`,
  `disable_cache=true`). Its time is the API turnaround.
- **Frontend** (`frontend/`): Next.js 16 app with a landing page and model table, an arena for
  uploading documents and starting runs, and a results page with live progress, seconds per page,
  side-by-side outputs (rendered or raw Markdown), and the detected layout drawn on the page.
- Uploads, page renders, outputs and the database live in `data/` (gitignored).

## Setup

You need [uv](https://docs.astral.sh/uv/) and Node.js 20 or later. A GPU is optional but strongly
recommended. On an AMD Radeon 8060S a page takes roughly 10–30 s per model; on a CPU it takes
minutes.

### 1. Configure

```bash
cp .env.example .env        # Windows: Copy-Item .env.example .env
```

Put your LlamaParse key into `.env` as `LLAMA_CLOUD_API_KEY=llx-...`. Keys are region-specific:
for an EU account, also set `LLAMA_CLOUD_BASE_URL=https://api.cloud.eu.llamaindex.ai`. Without a
key the arena still works, but only with the open models.

### 2. Backend

Install the dependencies and a PyTorch build that matches your hardware:

```powershell
./scripts/install-torch.ps1 -Flavor cuda           # NVIDIA
./scripts/install-torch.ps1 -Flavor rocm-gfx1151   # AMD Ryzen AI Max / Radeon 8060S
./scripts/install-torch.ps1 -Flavor cpu            # no GPU
```

```bash
./scripts/install-torch.sh cuda   # or rocm-gfx1151, cpu, mac
```

Then start the API:

```bash
cd backend
uv run uvicorn app.main:app --port 8000
```

The first run of each model downloads its weights from Hugging Face (about 2.2 GB per model).

> Use `uv sync --extra local --inexact` if you re-sync. A plain `uv sync` removes the separately
> installed PyTorch.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The frontend proxies `/api/*` to the backend. Set `BACKEND_URL` if
the backend isn't at `http://127.0.0.1:8000`.

### Faster: serve the models with vLLM

On a CUDA machine you can serve the models with vLLM, using the same settings ParseBench used,
and point the arena at them. The arena then sends requests there instead of loading the
weights itself:

```bash
vllm serve cloudraker/rakedoc-nano --served-model-name rakedoc-nano \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 24 \
  --trust-remote-code --limit-mm-per-prompt '{"image":1}' --port 8001

vllm serve aeddix-labs/aeddix-alpine-ocr --revision v0.1 --served-model-name aeddix-alpine-ocr \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 24 \
  --trust-remote-code --limit-mm-per-prompt '{"image":1}' --port 8002
```

```dotenv
RAKEDOC_NANO_ENDPOINT_URL=http://gpu-host:8001/v1
AEDDIX_ALPINE_OCR_ENDPOINT_URL=http://gpu-host:8002/v1
```

## Configuration

All settings are environment variables (or entries in `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLAMA_CLOUD_API_KEY` | empty | Enables LlamaParse Agentic Plus |
| `LLAMA_CLOUD_BASE_URL` | `https://api.cloud.llamaindex.ai` | EU: `https://api.cloud.eu.llamaindex.ai` |
| `RAKEDOC_NANO_ENDPOINT_URL`, `AEDDIX_ALPINE_OCR_ENDPOINT_URL` | empty | OpenAI-compatible endpoint (e.g. vLLM); empty = run locally |
| `VLLM_API_KEY` | empty | Bearer token for those endpoints |
| `LOCAL_INFERENCE` | `true` | Allow running models in the API process |
| `LOCAL_DEVICE`, `LOCAL_DTYPE` | `auto` | Force `cuda`/`cpu` and `bfloat16`/`float16`/`float32` |
| `LOCAL_MAX_BATCH` | `8` | Regions decoded together |
| `KEEP_MODELS_LOADED` | `true` | Keep weights in memory between runs |
| `MAX_UPLOAD_MB`, `MAX_PAGES_PER_EXPERIMENT` | `100`, `50` | Limits |
| `DATA_DIR` | `data/` in the repo | Uploads, renders, results, SQLite |
| `BACKEND_URL` (frontend) | `http://127.0.0.1:8000` | Where the Next.js server proxies `/api` |

## API

The frontend uses only this JSON API, so other clients can use it too:

| Method & path | Purpose |
| --- | --- |
| `GET /api/health` | Status, LlamaParse configured, local accelerator, limits |
| `GET /api/models` | Catalog with ParseBench scores and per-model availability |
| `POST /api/documents` | Upload (`multipart/form-data`, field `file`) |
| `POST /api/documents/sample` | Add the bundled sample report |
| `GET /api/documents/{id}/file`, `/thumbnail`, `/pages/{n}` | Original, thumbnail, page render |
| `POST /api/experiments` | Start: `{"document_id", "models": [...], "pages": "1-3,5"}` |
| `GET /api/experiments`, `GET /api/experiments/{id}` | History; detail with progress and outputs |
| `POST /api/experiments/{id}/rerun`, `DELETE /api/experiments/{id}` | Run again, delete |

Interactive docs: <http://127.0.0.1:8000/docs>.

## Development

```bash
cd backend && uv run pytest && uv run ruff check app scripts tests
cd frontend && npm run lint && npm run build
```

`backend/scripts/smoke_kdl.py` runs a model on a PDF from the command line, and
`backend/scripts/bench_local.py` measures prefill and decode speed.

## Privacy

Open models keep documents on the machine that runs the backend. When LlamaParse is selected,
the selected pages are uploaded to LlamaIndex Cloud.

## Roadmap

- LLM-as-a-judge comparing each open model's output with the LlamaParse reference
- The remaining five models
- Workspaces and authentication for multiple clients
