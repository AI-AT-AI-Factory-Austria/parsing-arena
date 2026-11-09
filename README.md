# OCR Arena

Evaluate the open-source OCR models that best suit your documents.

Drop in a PDF or image. OCR Arena parses it with open OCR models of different sizes, times
every run, and shows each model's output next to the original page and next to
**LlamaParse Agentic Plus** as an optional reference. The selection lists 24 implemented
methods, grouped roughly from larger models to compact pipelines and lightweight tools.
Each method highlights its strengths and use cases based on its model card and the preset implemented here.
Parameter counts and licenses come from the official model cards and project documentation.

| Model | Approximate size | Strengths & use cases | License |
| --- | --- | --- | --- |
| LlamaParse Agentic Plus (optional reference API) | Hosted API | Complex tables & charts; judging reference | Commercial API |
| Unlimited-OCR 2 | 3.3B | Reads multiple pages together with shared context | MIT |
| Infinity-Parser2-Flash | ≈2B | Chart data as tables & scientific page layouts | Apache-2.0 |
| rakedoc-nano | 1.2B | Merged cells & complex table headers | AGPL-3.0 |
| Aeddix Alpine OCR | 1.2B | Tables & inline text formatting | Apache-2.0 |
| florin-parser-nano | 1.2B | Preserves bold, strikethrough & superscripts | AGPL-3.0 |
| MinerU2.5-Pro-2605 | 1.2B | Formulas, charts & technical diagrams | Apache-2.0 (weights) |
| GLM-OCR 0.9B | 900M | Compact full-page text recognition | MIT |
| OvisOCR3 | 850M | Joins tables across pages & extracts chart data | Apache-2.0 |
| LightOnOCR-3-0.8B | 800M | Charts & page layout in one pass | Apache-2.0 |
| Marker Balanced | ≈689M + auxiliaries | Scanned pages, tables & inline math | GPL-3.0 (code), OpenRAIL (Surya weights) |
| Nemotron SkyVision CPU | 631M | Structured document extraction on CPU | OpenMDW-1.1 |
| Docling Standard | ≈300M | Reading order, page structure & tables | MIT (code), Apache-2.0 / CDLA (models) |
| OpenDataLoader Docling | Backend-dependent | Mixed digital/scanned PDFs with German/English OCR | Apache-2.0 (code), model-specific weights |
| Granite Docling 258M | 258M | Tables, equations & page structure in a compact model | Apache-2.0 |
| docTR Large transformer | ≈110.6M | Accuracy-focused text recognition | Apache-2.0 |
| PaddleOCR Medium | 34.5M | Multilingual, screen & scene text | Apache-2.0 |
| docTR default | ≈26.4M | Balanced OCR for printed documents | Apache-2.0 |
| Marker Fast | ≈20M + auxiliaries | Digital PDFs; embedded text, no OCR | GPL-3.0 (code), model-specific weights |
| PaddleOCR Small | 7.7M | Lightweight multilingual text OCR | Apache-2.0 |
| docTR compact | ≈6.3M | Text detection & recognition with low compute needs | Apache-2.0 |
| PaddleOCR Tiny | 1.5M | Tiny multilingual text OCR | Apache-2.0 |
| pypdf | No model | Embedded PDF text; no OCR | BSD-3-Clause |
| Tesseract¹ | Classic OCR (LSTM) | Clean scans of printed text | Apache-2.0 |
| OpenDataLoader Core | No neural model | Digital PDF structure; no OCR | Apache-2.0 |

Pipeline sizes are approximate and may exclude auxiliary models. OpenDataLoader Docling
is listed beside Docling Standard because its total depends on the backend configuration.
These specialties describe useful applications, not a quality ranking.

¹ Needs the Tesseract program (Windows: `winget install UB-Mannheim.TesseractOCR`).

Nemotron-3.5-Parse and Falcon-OCR have been removed from the active catalog. Saved
results retain their model names. Nemotron SkyVision CPU remains available.

The table below describes each local pipeline. Some implementations derive from
[ParseBench](https://github.com/run-llama/ParseBench); local settings are documented below.

| Model | Pipeline | Pages rendered at |
| --- | --- | --- |
| rakedoc-nano, Aeddix Alpine OCR, florin-parser-nano | KDL two-stage pipeline (layout, then region recognition); Florin markdown emission for rakedoc and florin | 144 DPI |
| Nemotron SkyVision CPU | SkyVision.cpp single full-page pass, F16 GGUF; NVIDIA text normalization to Markdown, HTML tables and layout boxes | 150 DPI, PNG |
| LightOnOCR-3-0.8B | One pass with the `grounding` prompt | 400 DPI, at most 5 MP |
| MinerU2.5-Pro-2605 | `mineru-vl-utils` two-step extraction with image analysis | 150 DPI |
| Infinity-Parser2-Flash | `doc2json`, then each figure re-read as a markdown table | 300 DPI |
| OvisOCR3 | Structured pages, chart reading, then recursive cross-page table reconstruction | 200 DPI |
| Docling Standard | Docling's default PDF pipeline on the CPU (ParseBench used a private Docling server, so its settings may differ) | original file |
| Granite Docling 258M | Docling VLM pipeline with the pinned IBM checkpoint; greedy DocTags generation, then Markdown and layout export | original file, Docling scale 2 (144 DPI for PDF) |
| pypdf | `page.extract_text()`, no OCR | original file |
| Tesseract | `pytesseract.image_to_string`, English (`TESSERACT_LANG`) | 300 DPI |
| PaddleOCR Tiny / Small / Medium | PP-OCRv6 matching detector + recognizer, official Transformers exports; text lines and boxes | 200 DPI |
| GLM-OCR 0.9B | Full-page `Text Recognition:` prompt, greedy Transformers inference, up to 8192 output tokens | 200 DPI |
| Unlimited-OCR 2 | All selected pages in one request, 1024px global views, DASH routing + page-relative RoPE; BF16 decoder / F16 vision | 144 DPI |

Post-processing code from ParseBench (Apache-2.0) is vendored in `backend/app/engines/vendor` and
`backend/app/engines/kdl`.

**OvisOCR3 cross-page tables** are enabled by default. After page parsing, the same model
checks adjacent selected pages containing tables, working backwards through multi-page
chains. Modified pages are re-rendered before the next merge, following the
[pinned upstream workflow](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/17f255e2c8ae6d18d5d5edb0a474bd4ed39aef90/document_crosspage.py)
(Apache-2.0; vendored with explicit UTF-8 text I/O for Windows).
Whole-document view and Markdown downloads contain joined tables. Individual page
outputs, layout boxes and page-level judging retain the original extraction, with
collapsible joined-table previews and source-page numbers. The total parsing time includes
reconstruction; individual page timings cover page extraction only. Saved runs are not
reprocessed automatically; start a new Ovis run to use this feature.

The conservative upstream binding joins the last table on the earlier page with the first
table on the next page. Skipped page numbers, incomplete page JSON and chart-derived data
tables are excluded. Ambiguous multiple merges at one boundary and malformed responses
are retained as diagnostics instead of overwriting page OCR. Failures in this optional
stage leave completed page extractions available and appear in the results panel.

For chains of three or more pages, install the browser after installing backend dependencies:
`backend/.venv/Scripts/python.exe -m playwright install chromium` on Windows, or
`backend/.venv/bin/python -m playwright install --with-deps chromium` on Linux.
An installed Windows Chrome is also detected; `OVISOCR3_BROWSER_EXECUTABLE` can specify
another Chromium executable. The pinned local MathJax renderer is downloaded from the
model repository on first use; document rendering blocks network requests. Remote vLLM
servers need to accept two images per request (`--limit-mm-per-prompt '{"image":2}'`).
Set `OVISOCR3_CROSSPAGE_TABLES=false` to reproduce the original page-only pipeline.

**Docling Standard** retains the existing `docling` model ID, so saved experiments still work.
**Granite Docling 258M** (`granite-docling-258m`) uses the separate VLM pipeline and reads page
images without substituting the embedded PDF text. It follows `LOCAL_DEVICE` and `LOCAL_DTYPE`,
uses Transformers without quantization or FlashAttention 2, and generates up to 8192 tokens per
page at temperature 0. Selected pages are processed individually, with their own timing and
progress. Incomplete or truncated generations are reported as page failures. The first run
downloads the [IBM model](https://huggingface.co/ibm-granite/granite-docling-258M); the existing
model cache limit also applies to this pipeline.

**PaddleOCR Tiny / Small / Medium** use the
[PP-OCRv6 family](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6), with stable IDs
`paddleocr_tiny`, `paddleocr_small` and `paddleocr_medium`. The published sizes include
the detector and recognizer together; Medium is 34.5M (approximately 35M). All six
official safetensors checkpoints are pinned in `backend/app/engines/paddle.py`.
They run in float32 through Transformers on the device selected by `LOCAL_DEVICE`,
including AMD ROCm and NVIDIA CUDA. On Windows/ROCm the same selective BatchNorm
GPU dispatch workaround as docTR avoids unsupported MIOpen kernels. Medium's eight
9x9 detector convolutions use PyTorch's native GPU implementation to avoid MIOpen
kernel-selection stalls; their weights and float32 precision remain unchanged.

Detection uses the checkpoint's preprocessing (minimum side 736, maximum side 4000),
threshold 0.3, box threshold 0.6 and unclip ratio 1.5. Text quadrilaterals are rectified
and recognized in batches of eight; accepted lines have confidence at least 0.5.
Raw line polygons and confidence scores are saved with each page. Orientation
classification, document unwarping and table/layout recognition models are disabled,
so the displayed sizes do not hide additional networks. Outputs contain text and
line boxes; tables are not reconstructed. Tiny supports 49 languages; Small and
Medium support 50, including Japanese. PDFs are rasterized, including those with
embedded text, and image uploads use the original image.

**GLM-OCR 0.9B** (`glm-ocr`) uses the pinned
[Z.ai checkpoint](https://huggingface.co/zai-org/GLM-OCR) and its official Transformers
processor. This option runs the model directly on the full page, without the SDK's
additional PP-DocLayoutV3 detector. It therefore returns text/Markdown without layout
boxes, and SDK benchmark scores are not assigned to this option. It follows
`LOCAL_DEVICE` and `LOCAL_DTYPE`, uses SDPA attention and greedy generation, and marks
an output stopped by the 8192-token limit as incomplete. These four options require
the local dependencies (`uv sync --extra local --inexact`) and the hardware-specific
PyTorch installation. Their checkpoints are downloaded once and cached locally;
documents are not sent to an external OCR service.
GLM's vision patch embedding and 2x2 merger use equivalent matrix multiplications
with the original weights and precision to avoid slow AMD convolution kernel selection.
Its pipe tables are converted to HTML so irregular header widths cannot cause the
Markdown viewer to discard extracted cells. Raw model text is kept in `glm-page-N.md`.

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

## Saved sample documents

Place PDF, PNG, JPEG, WebP or TIFF files in `data/sample-docs/` (under `DATA_DIR` if
overridden). The arena discovers them and shows thumbnails below the upload area.
Sample cards open their saved comparisons directly; **Use a sample report** chooses a
random saved sample. Neither action runs OCR, sends documents to an API, or uses credits.
The model and page selections on the upload form only apply to new uploads.

With the API running, prepare the gallery once from `backend/`:

```bash
uv run python scripts/prepare_samples.py --include-reference
```

This runs all available open models plus LlamaParse and, when configured, Mistral judging.
The reference option uploads the samples to LlamaParse and sends extracted text to Mistral;
omit `--include-reference` for open-model results only. Use `--models pypdf docling` to
choose specific open models. Preparation returns immediately; progress appears on the
results pages. Repeating the command reuses the saved or already queued comparisons.
Only `--refresh` explicitly creates replacement runs.

The sample index, model outputs and judgments persist in SQLite alongside the originals
and previews in `data/`. Keep this directory when restarting or moving the app. Files are
identified by their SHA-256 content: changing a file requires preparing its new version;
renaming it retains its results. Removing a source file hides its card without deleting
past experiments. Newly added files appear as unprepared until the command is run again.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Node.js 20 or later. A GPU is optional but strongly
recommended. On an AMD Radeon 8060S a page takes roughly 10–60 s per model; on a CPU it takes
minutes. A full arena run therefore takes several minutes per page, and the vLLM setup below is
much faster.

### Unlimited-OCR 2: local multi-page parsing

`unlimited-ocr-2` now processes **all selected pages together**, in one request.
It uses the [community v2 GGUF runtime](https://huggingface.co/RX5950XT/Unlimited-OCR-2-GGUF),
pinned to revision `1e3e5f3e82435578cd1e3c191673e1444b617f52`, with a **BF16 decoder and
F16 vision encoder** (about 6.7 GB of weights). This retains the 3.3B parameter model;
it is not a 4/8-bit quantization. The original checkpoint is pinned to
`0480966938291445591f69f08a13b041c073a96a`.

Install the dedicated runtime and optionally cache its weights:

```powershell
uv run --no-project --script backend/scripts/install-unlimited.py --download-model
```

On Windows the installer verifies checksums for the official llama.cpp **b11913**
Vulkan package and the author's patched DLLs. On Linux it builds the same patch for
CUDA; that path requires a C++ compiler and CUDA Toolkit and has not been tested on
this Windows machine. Files live in `data/runtimes/unlimited-ocr2`, separately from
the other models' llama.cpp installation. Restart the backend after installation.

This implementation preserves **DASH routing and page-relative RoPE**, but computes
masked dense attention. **It does not provide the paper's optimized sparse-memory
scaling**: longer documents still consume more memory. It is intended to evaluate
the model's multi-page extraction on the available GPU, not to benchmark unlimited
document length. The catalog's ParseBench score is the published model score, not a
new measurement of this runtime.

Defaults: 16 selected pages, a 32,768-token context, up to 24,000 output tokens,
greedy decoding and a 1024px global view per page (273 visual tokens, no tiles).
`UNLIMITED_OCR_2_MAX_PAGES`, `UNLIMITED_OCR_2_CONTEXT_LENGTH`, and
`UNLIMITED_OCR_2_MAX_TOKENS` can be configured in `.env`. The output budget is reduced
when necessary to fit the input. Context shifting and prompt caching are disabled
to preserve page positions between requests. Table special tokens are retained.

The app validates page-marker counts, saves the raw stream as `unlimited-raw.txt`,
and maps blocks, headings, tables and normalized layout boxes onto the original
page numbers. Token exhaustion or invalid page boundaries fail the run instead of
silently accepting a partial document. Results show **total document time**;
seconds/page in the overview is an average, not an independently measured page time.

Tested on the **AMD Radeon 8060S using Vulkan**: the bundled two-page report
completed in 19.1 seconds, a distinct four-page synthetic document in 17.6 seconds,
and a subsequent pages-2-and-4 selection in 7.7 seconds using the loaded model.
These are functional smoke tests, not long-document scaling or accuracy benchmarks.
No external document-processing API is used.

### Runtime profiles

Set **`RUNTIME_PROFILE=auto`** in the repo `.env` (the default), or choose explicitly:

- **`windows-amd`**: LightOnOCR-3, Infinity-Parser2-Flash and OvisOCR3 use local
  **llama.cpp / Vulkan with F16 GGUF weights and F16 vision projectors**. The other GPU
  models retain their existing ROCm implementations. Both inference paths use the AMD GPU.
- **`linux-nvidia`**: GPU models use their Transformers/CUDA implementations;
  Unlimited-OCR 2 uses its separately installed CUDA runtime.

An explicit profile is checked against the actual OS and GPU runtime; it cannot enable
CUDA on an AMD machine. Restart the backend after changing `.env`. Configured model
endpoints take precedence and remain usable from either platform. `LOCAL_DEVICE=cpu`
explicitly selects CPU inference instead of the Vulkan alternatives.

The Windows profile uses the pinned Vulkan llama.cpp build installed by
`backend/.venv/Scripts/python.exe backend/scripts/install-preset-tools.py`.
`LLAMA_CPP_BINARY` can point to another compatible Vulkan `llama-server` executable.
First use downloads community format conversions from
[LightOnOCR-3](https://huggingface.co/mradermacher/LightOnOCR-3-0.8B-GGUF),
[OvisOCR3](https://huggingface.co/mradermacher/OvisOCR3-GGUF), and
[Infinity-Parser2-Flash](https://huggingface.co/prithivMLmods/Infinity-Parser2-Flash-GGUF).
Exact revisions and filenames are pinned in `backend/app/engines/backends/llama_cpp.py`.
These use full 16-bit weights, not 4/8-bit quantization. Their conversion revision and
runtime are recorded with every run. Rendering, prompts, crop passes and normalization
stay in the existing engines; outputs can still differ from native inference, and the
leaderboard scores are not new measurements of the Vulkan versions.

Model selection shows **AMD Vulkan · F16 GGUF** for these variants. Local model servers
bind only to `127.0.0.1`, share the existing model cache limit, and stop on unload/shutdown.
Documents stay local. Logs are in `data/runtime-logs/*-vulkan.log`.

The three Vulkan pipelines completed an end-to-end test on a Radeon 8060S / Windows
using page 1 of the bundled synthetic report. **Nemotron SkyVision CPU** runs locally
on the CPU and is described below.

### Nemotron SkyVision CPU

The separate `nemotron-skyvision-cpu` option runs
[SkyContact's Nemotron F16 conversion](https://huggingface.co/SkyContact/Nemotron-3.5-Parse-GGUF)
with [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp). It has about **631 million
parameters and a 1.31 GB weight file**. It uses CPU inference on both hardware profiles.
No ParseBench score is assigned to this different pipeline.

From the repo root, build the runtime once:

```bash
uv run --no-project --script backend/scripts/install-skyvision.py
```

The installer pins SkyVision and ggml commits and builds a static AVX2/FMA/F16C CPU
executable under `data/runtimes/skyvision`. On Windows x64 it downloads a checksum-verified
portable LLVM/MinGW compiler; Linux requires an installed C/C++ compiler. CMake and Ninja
run in an isolated uv environment. System tools and the app's PyTorch are unchanged.
The build includes small portability fixes and rejects truncated generations as errors.

Restart the API after installation. The first run downloads the pinned F16 checkpoint,
tokenizer and NVIDIA's text-only post-processing helpers. Set `SKYVISION_BINARY` to use
an installation in another location (keep its
`build-info.json` next to the executable), `SKYVISION_THREADS` to override the default
physical-core count, or `SKYVISION_TIMEOUT_S` to adjust the 20-minute per-request timeout.

One local worker keeps the model loaded under the existing cache limit and communicates
over stdin/stdout; no network server or external document upload is involved. The parser
uses its native image-resolution policy, greedy decoding, 8,744 output tokens, and a
24,576-token context. Each page exports Markdown and layout boxes, with raw structured
output saved beside the run artifacts. NVIDIA's pinned `postprocess_document` helper
formats the raw token stream: it preserves merged headers that the current SkyVision
table converter drops. This step loads no neural model and uses no GPU; both the raw
output and the normalizer revision are recorded. Runtime logs are in
`data/runtime-logs/nemotron-skyvision-cpu.log`.

### Deploying on Linux / NVIDIA

On the target host, install the NVIDIA driver, Python 3.12, `uv`, and a C compiler
(`sudo apt-get install build-essential` on Ubuntu). Then, from the repo root:

```bash
bash scripts/install-linux-nvidia.sh
```

This installs the locked local dependencies plus PyTorch 2.11.0 / torchvision 0.26.0
from the official CUDA 12.8 wheel index, and compiles/runs a real FlexAttention kernel.
It does not change `.env`, API keys, or saved data. Set `RUNTIME_PROFILE=linux-nvidia`
in `.env`, then verify the five affected models and start the API:

```bash
cd backend
.venv/bin/python scripts/verify-runtime.py --smoke
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The smoke test runs one bundled synthetic page per model and saves outputs and runtime
metadata under `data/runtime-checks/`. It downloads missing weights but does not process
gallery documents or use LlamaParse/Mistral/remote endpoints. Without `--smoke`, it only
checks configuration. On Windows, select the compatible models explicitly:

```powershell
backend/.venv/Scripts/python.exe backend/scripts/verify-runtime.py --smoke --models lightonocr-3-0.8b infinity-parser2-flash ovisocr3
```

The Linux/NVIDIA dispatch is covered by automated tests, but full model inference on
NVIDIA hardware still needs to be verified on the deployment host; the development
machine has an AMD GPU. Separate optional tools (Tesseract, Java and the isolated Marker
environment) retain their own setup instructions below.

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

Infinity-Parser2-Flash needs its own serving setup; follow its model card, then set
`INFINITY_PARSER2_FLASH_ENDPOINT_URL` (served model name
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
| `GET /api/models` | Catalog with sizes, specialties and per-model availability (legacy benchmark metadata retained) |
| `POST /api/documents` | Upload (`multipart/form-data`, field `file`) |
| `POST /api/documents/sample` | Add the bundled sample report |
| `GET /api/documents/{id}/file`, `/thumbnail`, `/pages/{n}` | Original, thumbnail, page render |
| `POST /api/experiments` | Start: `{"document_id", "models": [...], "pages": "1-3,5"}` |
| `GET /api/experiments`, `GET /api/experiments/{id}` | History; detail with progress and outputs |
| `POST /api/experiments/{id}/rerun`, `DELETE /api/experiments/{id}` | Run again, delete |

Interactive docs: <http://127.0.0.1:8000/docs>.

## Development

### Additional local library presets

Seven independent options are available, with stable IDs:

- **Marker Fast** (`marker_fast`): `marker-pdf==2.1.0`, `PdfConverter`,
  `mode="fast"`, `disable_ocr=True`, `use_llm=False`. Reads embedded PDF text with
  a roughly 20M layout detector. No OCR on scans, and no Surya OCR VLM is loaded.
  The lightweight detector itself is provided by Surya's helper library.
- **Marker Balanced** (`marker_balanced`): the same package, `mode="balanced"`,
  `disable_ocr=False`, `force_ocr=False`, `use_llm=False`, Surya checkpoint
  `datalab-to/surya-ocr-2.5` (roughly 689M plus auxiliaries). Uses the official
  unquantized `surya-2.5.gguf` and multimodal projector through local llama.cpp.
  Neither Marker preset uses cloud LLM postprocessing.
- **OpenDataLoader Core** (`opendataloader_core`): `opendataloader-pdf==2.5.18`,
  `hybrid="off"`. Java text/structure extraction, no neural model and no OCR.
- **OpenDataLoader Docling** (`opendataloader_docling`): the package's `[hybrid]`
  extra, `hybrid="docling-fast"`, `hybrid_mode="auto"`, `hybrid_fallback=False`.
  The backend starts on loopback with CPU Docling, EasyOCR languages `de,en`,
  automatic OCR and accurate TableFormer enabled. Formula enrichment and picture
  descriptions are disabled. Automatic triage sends only pages that need it to
  Docling; backend failures are surfaced instead of silently using Java output.
- **docTR compact** (`doctr_compact`): `python-doctr==1.1.0`, detector
  `db_mobilenet_v3_large`, recognizer `crnn_mobilenet_v3_small` (approximately 6.3M).
- **docTR default** (`doctr_default`): detector `fast_base`, recognizer
  `crnn_vgg16_bn` (approximately 26.4M at inference).
- **docTR Large transformer** (`doctr_large_transformer`): detector `db_resnet50`,
  recognizer `vitstr_base` (approximately 110.6M).

All three docTR presets use `ocr_predictor(pretrained=True, detect_layout=False,
detect_tables=False)` with automatic GPU selection (including AMD ROCm), four CPU
threads for preprocessing, 200-DPI PDF images, float32 weights, and the library's
remaining defaults. `LOCAL_DEVICE=cpu` explicitly selects CPU; `auto` uses CPU
only when no supported GPU is available. They output plain text and word bounding boxes;
table cells are not reconstructed. Marker exports Markdown/layout boxes;
OpenDataLoader exports Markdown/JSON with normalized layout boxes. Selected page
numbers, timings, exact preset configuration and package versions are saved with
each result. Sizes exclude auxiliary components where noted.

On Windows ROCm, docTR's batch-normalization and LSTM layers use PyTorch's native
GPU kernels to work around MIOpen's missing `type_traits` HIPRTC compile error.
Convolutions continue to use MIOpen. The original weights, float32 precision and
architectures are preserved, and model tensors stay on the GPU. Saved run metadata
records the actual device, ROCm/CUDA runtime and affected layer counts.

Install the main local dependencies with `uv sync --extra local --inexact` in
`backend`. Marker has a separate environment because its Pillow `<11` requirement
conflicts with MinerU's Pillow `>=11` requirement:

```powershell
cd backend
uv sync --extra local --inexact
uv sync --project runtimes/marker
.\.venv\Scripts\python.exe scripts/install-preset-tools.py
```

The last command installs checksum-verified portable Java 21 and llama.cpp Vulkan
for Windows x64 under `data/runtimes`, without changing system PATH. On other
platforms, install Java 11+ and `llama-server` yourself. `JAVA_CMD`,
`LLAMA_CPP_BINARY` and `MARKER_PYTHON` can override executable locations. Model
weights download on first use; documents stay local. Marker's isolated CPU
PyTorch handles auxiliary models and llama.cpp handles the Balanced VLM. The
existing main-environment CUDA/ROCm PyTorch is preserved by `--inexact`.

Workers are cached separately by preset and stopped on eviction/server shutdown.
An isolated Windows compatibility fix uses process handles instead of Surya
0.23's POSIX signal-based liveness check when stopping its own servers.

To test an individual preset without running samples or external API judging:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_model.py samples/sample-report.pdf --model marker_fast --pages 1
```

### Checks

```bash
cd backend && uv run pytest && uv run ruff check app scripts tests
cd frontend && npm run lint && npm run build
```

`backend/scripts/smoke_model.py` runs one model on a few pages from the command line (stop the API
server first so only one model uses the GPU):

```bash
cd backend
uv run python scripts/smoke_model.py samples/sample-report.pdf --model lightonocr-3-0.8b --pages 1
```

`backend/scripts/bench_local.py` measures prefill and decode speed.

## Privacy

Open models keep documents on the machine that runs the backend. When LlamaParse is selected,
the selected pages are uploaded to LlamaIndex Cloud.

When `MISTRAL` (or `MISTRAL_API_KEY`) is configured and a completed LlamaParse reference is
available, the candidate and reference **extracted text** are sent to Mistral for quality
judging. Original files and page images are not sent to the judge.

## Quality judging

The default judge is [Mistral Large 4](https://docs.mistral.ai/models/mistral-large), using
high reasoning effort and structured JSON responses. `JUDGE_MODEL` can override the model;
`JUDGE_TIMEOUT_S` defaults to 180 seconds per request. The API key stays on the backend.

Each selected page is compared to the matching LlamaParse page for content, exact values,
tables, structure, reading order and formulas. Equivalent Markdown/HTML and cosmetic
differences are ignored. Findings include short source quotes that the backend validates.
The document verdict is the worst page's verdict:

- **Excellent:** no supported discrepancies against the reference.
- **Pass:** only minor transcription or structural issues, without changed meaning.
- **Fail:** a material omission, incorrect critical value, broken association or other
  major extraction error.

The score is relative to LlamaParse, not a guarantee of correctness against the original.
An empty, failed or unusable reference produces an unavailable check, never a Fail score.
API errors likewise remain distinct from extraction quality. Oversized pages are rejected
without silently truncating the comparison.

New runs are checked automatically after both outputs complete. Existing experiments have
a **Judge outputs** action, backed by `POST /api/experiments/{id}/judge`. The action retries
failed checks and skips already judged, unchanged outputs. Judging has its own background
worker and does not affect OCR timings. Results persist in a separate SQLite table with
the model, rubric version, input fingerprint and page findings; interrupted checks resume
from completed pages on restart. Only explicitly queued checks resume, not all old experiments.

## Roadmap

- Workspaces and authentication for multiple clients
