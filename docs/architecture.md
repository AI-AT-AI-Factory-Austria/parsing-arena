# Architecture & development

[← README](../README.md) · [Deployment](deployment.md) · [Model presets](model-presets.md)

## Application structure

| Component | Responsibility |
| --- | --- |
| [`frontend/`](../frontend) | Next.js 16 UI: uploads, page selection, live results, model comparison and saved history |
| [`backend/app/api/`](../backend/app/api) | FastAPI endpoints for documents, batches, experiments, samples and judging settings |
| [`runner.py`](../backend/app/runner.py) | In-process extraction queue; sequential local methods and progress persistence |
| [`engines/`](../backend/app/engines) | Model adapters, local model pool, isolated library workers and optional remote endpoints |
| [`judging.py`](../backend/app/judging.py) | Separate reference-based quality-check queue |
| [`db.py`](../backend/app/db.py), `data/` | SQLite records, originals, page renders, outputs and saved credentials |

The frontend proxies `/api/*` to the backend. Keep one backend replica/Uvicorn worker
because queues live in-process and share SQLite. The model pool limits resident
checkpoints and evicts least-recently-used models; CPU runs release models after use.
Marker has an isolated environment to avoid dependency conflicts.

## Documents, runs and timing

- Supported uploads: **PDF, PNG, JPEG, WebP and TIFF**, up to 100 MB by default.
  The batch UI/API accepts at most **12 unique pages**, identified by document content
  hash plus page number. Repeated uploads are deduplicated.
- A batch contains one experiment per document. Documents remain separate for
  multi-page extraction, table reconstruction and judging. Reruns preserve the
  selection and CPU mode; saved samples open their existing results.
- Local methods run sequentially on the host. **Model loading is excluded from parse
  time**. LlamaParse runs independently through the official `llama-cloud` SDK with
  `tier=agentic_plus`, `version=latest`, `disable_cache=true`; its timing is API turnaround.
- The batch average is total measured parsing time divided by its timed pages.
  Coverage is shown explicitly. Whole-document methods report average seconds/page;
  they do not provide independently measured page timings.
- Page failures remain distinct from successful pages. Partial results can be inspected;
  unavailable quality checks do not turn a successful extraction into a parsing failure.

Model outputs can contain Markdown, plain text, HTML tables and normalized layout
boxes, depending on the preset. The UI supports rendered/Markdown views, original-page
layout overlays, per-page navigation and downloads. Runtime metadata accompanies outputs.

## Quality judging

Judging is optional and off by default. Selecting LlamaParse and confirming with **OK**
uploads the **full original files** to LlamaParse to process the selected pages.
Candidate and reference **extracted text** go to Mistral; the judge does not receive
the original files or images. CPU mode does not change this. Saving API keys alone
does not start processing, and each new run/rerun requires its own UI confirmation.

The judge compares matching pages for content, critical values, tables, structure,
reading order and formulas. Cosmetic differences and equivalent Markdown/HTML are ignored.

- **Excellent:** no supported discrepancies against the reference.
- **Pass:** minor transcription/structural issues without a change in meaning.
- **Fail:** a material omission, incorrect critical value, broken association or other major error.

The document verdict is its worst page. Batch counts cover only current judgments;
pass rate is `(Excellent + Pass) / judged pages`. Missing, outdated or failed checks
are excluded and their coverage is shown. A reference error can affect a rating:
**these are relative assessments, not ground truth**.

LlamaParse headers/footers are included in display and judging, avoiding duplicate
body edges. Candidate-only headers, footers or navigation are not treated as invented
solely because the reference omits them. Empty/unusable references and API errors
produce unavailable checks. Oversized comparisons are rejected rather than silently
truncated. The backend validates response structure; it does not require evidence
excerpts to occur as contiguous input-text substrings.

| Setting | Default |
| --- | --- |
| `JUDGE_MODEL` | `mistral-large-4` |
| `JUDGE_REASONING_EFFORT` | `medium` (`low` / `high` also supported) |
| `JUDGE_CONCURRENCY` | `4` concurrent page checks across documents/models |
| `JUDGE_TIMEOUT_S` | `180` seconds/request |
| `JUDGE_REQUEST_ATTEMPTS` | `4` attempts/request |

Transient connection errors, HTTP 408/429/5xx and unreadable responses use backoff
with jitter and `Retry-After`. Invalid assessments can receive up to two repair
attempts; truncated responses receive a larger output allowance. Authentication and
input errors are not retried. One failed check does not stop the others.

New reference-backed runs are judged when both outputs complete. **Judge outputs**
retries missing/failed or outdated checks while skipping unchanged completed ones.
Judging has its own queue and does not affect OCR timings. Persisted input fingerprints,
rubric version, reasoning effort and page findings support resuming explicitly queued
work after restart. Old experiments are not automatically rejudged. Older rubrics show
**Recheck needed**; text absent from an old reference requires refreshing that reference.

## Saved sample gallery

Place intended sample files in `DATA_DIR/sample-docs/` (native default:
`data/sample-docs/`; container: `/data/sample-docs/`). With the API running, prepare
them once:

```bash
# Containers, from the repo root
docker compose exec backend python scripts/prepare_samples.py --models pypdf docling

# Native environment, from backend/
uv run --no-sync python scripts/prepare_samples.py --models pypdf docling
```

Omit `--models` to use all available local methods. Add `--include-reference` only
when you intend to upload the samples to LlamaParse and send text to Mistral using
your API credits. These are CLI instructions; the CLI flag authorizes those transfers.
Use both Compose files for the NVIDIA deployment.

Preparation queues work and returns result links. Repeating it reuses saved/queued
runs; `--refresh` explicitly replaces them. Files are identified by SHA-256:
renaming keeps results, changed content needs preparation, and removing a source file
hides its card without deleting past experiments. New files remain unprepared until run.

Sample cards open saved comparisons; **Use a sample report** picks one randomly.
Neither action reruns OCR or spends API credits. Model/page choices in the upload form
apply to new uploads. Originals, previews, assessments and results persist with the database.

## API

Interactive native API docs: **http://127.0.0.1:8000/docs**. Compose keeps this port
internal by default; use a private port-forward when you need the backend docs.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health`, `GET /api/models` | Health, runtime availability and model catalog; `?cpu_only=true` selects CPU availability |
| `POST /api/documents` | Multipart upload, field `file` |
| `GET /api/documents/{id}/file`, `/thumbnail`, `/pages/{n}` | Original, thumbnail and rendered page |
| `POST /api/batches` | Start a batch with documents, selected pages, model IDs and `cpu_only` |
| `GET /api/batches`, `GET /api/batches/{id}` | Saved batches and page-weighted aggregates |
| `POST /api/batches/{id}/rerun`, `DELETE /api/batches/{id}` | Rerun selection/mode or delete a finished batch |
| `POST /api/experiments`, `GET /api/experiments/{id}` | Single-document runs, progress and outputs |
| `GET /api/experiments`, `POST /api/experiments/{id}/rerun`, `DELETE /api/experiments/{id}` | History, rerun and deletion |
| `POST /api/experiments/{id}/judge` | Queue checks for saved outputs |
| `GET /api/samples`, `POST /api/samples/{id}/prepare` | Sample discovery and preparation |
| `GET /api/settings/judging`, `POST /api/settings/judging` | Key availability and saving missing keys; secrets are never returned |

Example batch body:

```json
{"documents": [{"document_id": "...", "pages": [1, 3]}], "models": ["pypdf", "docling"], "cpu_only": true}
```

## Development

Use the [Windows helper](../scripts/setup-windows.ps1) for the tested AMD setup.
For manual Linux development, `bash scripts/install-torch.sh cpu` installs main
dependencies and CPU PyTorch. For native NVIDIA, use
`bash scripts/install-linux-nvidia.sh`; it installs pinned CUDA wheels and checks
Triton/FlexAttention. Complete parser/runtime packaging is provided by the containers.

Native auxiliary setup: `uv sync --project backend/runtimes/marker` for Marker;
install Java, Tesseract and Chromium, plus the native runtimes linked in
[model presets](model-presets.md). On Windows, `install-preset-tools.py` installs
portable Java and llama.cpp. Keep the hardware-specific main PyTorch environment intact:
use `uv sync --extra local --inexact` when resyncing and `--no-sync` when running tools.

Start two terminals:

```bash
# Backend
cd backend
uv run --no-sync python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
# Frontend (Node.js 22 LTS)
cd frontend
npm ci
npm run dev
```

Checks, in their respective directories:

```bash
# backend/
uv run --no-sync pytest
uv run --no-sync ruff check app scripts tests
uv run --no-sync python scripts/smoke_model.py samples/sample-report.pdf --model docling --pages 1 --cpu-only

# frontend/
npm run lint
npm run build
```

Stop active GPU runs before CLI smoke tests. `scripts/verify-runtime.py --smoke`
checks selected GPU paths against a bundled synthetic page, downloading weights if
needed, without external document-processing APIs. `scripts/bench_local.py` measures
prefill/decode speed. Full NVIDIA-model inference still needs a deployment-host check.
