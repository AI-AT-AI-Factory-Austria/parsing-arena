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
| florin-parser-nano | Florin | 1.2B | 76.7 | AGPL-3.0 | yes |
| Nemotron-3.5-Parse | NVIDIA | 0.631B | 75.1 | OpenMDW-1.1 | yes¹ |
| LightOnOCR-3-0.8B | LightOn | 0.8B class | 74.7 | Apache-2.0 | yes¹ |
| OvisOCR3 | ATH-MaaS | 0.85B | 71.8 | Apache-2.0 | yes¹ |
| MinerU2.5-Pro-2605 | OpenDataLab | 1.2B | 69.3 | Apache-2.0 (weights) | yes |
| Infinity-Parser2-Flash | inftech.ai / infly | ≈2B | 68.6 | Apache-2.0 | yes¹ |
| Surya OCR 2 | Datalab | 0.69B | 60.6 | OpenRAIL-M (weights) | coming soon |
| TeleOCR | StarDoc-AI | 1.4B | 57.5 | Apache-2.0 | coming soon |
| Unlimited-OCR 2 | Baidu | 3.3B | 57.2 | MIT | coming soon |
| Falcon-OCR | TII | 0.27B | 52.1 | Apache-2.0 | yes² |
| Docling | IBM / LF AI & Data | ≈0.3B | 50.7 | MIT, Apache-2.0 / CDLA | yes, CPU |
| pypdf | py-pdf | no model | 14.9 | BSD-3-Clause | yes, CPU |
| Tesseract | Tesseract OCR | classic OCR | not benchmarked | Apache-2.0 | yes, CPU³ |

¹ Needs an NVIDIA GPU or a vLLM endpoint; refused on AMD ROCm GPUs, where these models (or ones of
the same architecture) crashed a Radeon 8060S laptop. ² Needs Triton, i.e. Linux with an NVIDIA GPU.
³ Needs the Tesseract program (Windows: `winget install UB-Mannheim.TesseractOCR`). ParseBench
has a Tesseract pipeline but no leaderboard score for it.

rakedoc-nano, Aeddix Alpine OCR, florin-parser-nano, OvisOCR3, Unlimited-OCR 2, Falcon-OCR,
Docling and pypdf are marked *unverified* on the leaderboard.

Every model runs the pipeline its ParseBench score was measured with, at the same page resolution
and with the same prompts, sampling settings and post-processing:

| Model | Pipeline | Pages rendered at |
| --- | --- | --- |
| rakedoc-nano, Aeddix Alpine OCR, florin-parser-nano | KDL two-stage pipeline (layout, then region recognition); Florin markdown emission for rakedoc and florin | 144 DPI |
| Nemotron-3.5-Parse | NVIDIA's `nemotron_parse` SDK from the model repository, triple pass (page, detection, table and chart crops) | 150 DPI, JPEG |
| LightOnOCR-3-0.8B | One pass with the `grounding` prompt | 400 DPI, at most 5 MP |
| MinerU2.5-Pro-2605 | `mineru-vl-utils` two-step extraction with image analysis | 150 DPI |
| Infinity-Parser2-Flash | `doc2json`, then each figure re-read as a markdown table | 300 DPI |
| OvisOCR3 | Model card's structured JSON prompt per page, then the chart prompt per chart | 200 DPI |
| Falcon-OCR | The model's `generate_with_layout` (PP-DocLayoutV3 regions, OCR per region) | 200 DPI |
| Docling | Docling's default PDF pipeline on the CPU (ParseBench used a private Docling server, so its settings may differ) | original file |
| pypdf | `page.extract_text()`, no OCR | original file |
| Tesseract | `pytesseract.image_to_string`, English (`TESSERACT_LANG`) | 300 DPI |

Post-processing code from ParseBench (Apache-2.0) is vendored in `backend/app/engines/vendor` and
`backend/app/engines/kdl`.

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
recommended. On an AMD Radeon 8060S a page takes roughly 10–60 s per model; on a CPU it takes
minutes. A full arena run therefore takes several minutes per page, and the vLLM setup below is
much faster.

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

The first run of each model downloads its weights from Hugging Face (1.7–4.5 GB per model,
about 18 GB for all seven). At most `MAX_LOADED_MODELS` (default 2) stay in memory; the least
recently used one is unloaded when another is needed.

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

vllm serve florin-inc/florin-parser-nano --served-model-name florin-parser-nano \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 24 \
  --trust-remote-code --limit-mm-per-prompt '{"image":1}' --port 8003

# vLLM 0.30 or later
vllm serve lightonai/LightOnOCR-3-0.8B --served-model-name loocr-grounding \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 24576 --port 8004

vllm serve opendatalab/MinerU2.5-Pro-2605-1.2B \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor --port 8005
```

```dotenv
RAKEDOC_NANO_ENDPOINT_URL=http://gpu-host:8001/v1
AEDDIX_ALPINE_OCR_ENDPOINT_URL=http://gpu-host:8002/v1
FLORIN_PARSER_NANO_ENDPOINT_URL=http://gpu-host:8003/v1
LIGHTONOCR_3_0_8B_ENDPOINT_URL=http://gpu-host:8004/v1
MINERU2_5_PRO_2605_ENDPOINT_URL=http://gpu-host:8005/v1
```

Nemotron-3.5-Parse needs NVIDIA's vLLM plugin and Infinity-Parser2-Flash its own serving setup;
follow their model cards, then set `NEMOTRON_3_5_PARSE_ENDPOINT_URL` (served model name
`nemotron-parse`) and `INFINITY_PARSER2_FLASH_ENDPOINT_URL` (served model name
`infly/Infinity-Parser2-Flash`).

## Configuration

All settings are environment variables (or entries in `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLAMA_CLOUD_API_KEY` | empty | Enables LlamaParse Agentic Plus |
| `LLAMA_CLOUD_BASE_URL` | `https://api.cloud.llamaindex.ai` | EU: `https://api.cloud.eu.llamaindex.ai` |
| `<MODEL>_ENDPOINT_URL` | empty | vLLM endpoint for one model, e.g. `RAKEDOC_NANO_ENDPOINT_URL`; empty = run locally. `<MODEL>` is the model id in capitals with `_` for other characters |
| `<MODEL>_SERVED_MODEL` | see `.env.example` | Served model name on that endpoint |
| `VLLM_API_KEY` | empty | Bearer token for those endpoints |
| `LOCAL_INFERENCE` | `true` | Allow running models in the API process |
| `LOCAL_DEVICE`, `LOCAL_DTYPE` | `auto` | Force `cuda`/`cpu` and `bfloat16`/`float16`/`float32` |
| `LOCAL_MAX_BATCH` | `8` | Requests decoded together (KDL regions, MinerU blocks) |
| `KEEP_MODELS_LOADED` | `true` | Keep weights in memory between runs |
| `MAX_LOADED_MODELS` | `2` | Checkpoints kept in memory at once |
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

`backend/scripts/smoke_model.py` runs one model on a few pages from the command line (stop the API
server first so only one model uses the GPU):

```bash
cd backend
uv run python scripts/smoke_model.py samples/sample-report.pdf --model nemotron-3.5-parse --pages 1
```

`backend/scripts/bench_local.py` measures prefill and decode speed.

## Privacy

Open models keep documents on the machine that runs the backend. When LlamaParse is selected,
the selected pages are uploaded to LlamaIndex Cloud.

## Roadmap

- LLM-as-a-judge comparing each open model's output with the LlamaParse reference
- Workspaces and authentication for multiple clients
