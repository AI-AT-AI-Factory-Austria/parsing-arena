# Model presets

[← Model list](../README.md#models) · [Deployment](deployment.md) · [Architecture](architecture.md)

Parsing Arena compares defined presets. The app records model/runtime information,
selected pages, timings and available parser metadata with each run. Library versions
are locked and model revisions are pinned where specified. The
[catalog](../backend/app/catalog.py) contains model-card links, sizes and licenses;
the [engine implementations](../backend/app/engines) are the source of truth for settings.

## Pipelines at a glance

| Method / stable ID | Configuration in this app | Output and input rendering |
| --- | --- | --- |
| rakedoc-nano / `rakedoc-nano` | KDL layout detection, then region recognition; Florin Markdown emission | Markdown + layout; 144 DPI |
| Aeddix Alpine OCR / `aeddix-alpine-ocr` | KDL two-stage pipeline | Markdown + layout; 144 DPI |
| florin-parser-nano / `florin-parser-nano` | KDL two-stage pipeline with Florin formatting | Markdown + layout; 144 DPI |
| MinerU / `mineru2.5-pro-2605` | `mineru-vl-utils` two-step extraction with image analysis | Markdown + layout; 150 DPI |
| LightOnOCR / `lightonocr-3-0.8b` | Single pass with the `grounding` prompt | Markdown + layout; 400 DPI, at most 5 MP |
| Infinity / `infinity-parser2-flash` | `doc2json`, then figures re-read as Markdown tables | Structured page + chart tables; 300 DPI |
| OvisOCR3 / `ovisocr3` | Structured pages and chart reading, followed by cross-page table reconstruction | Page outputs + joined document tables; 200 DPI |
| GLM-OCR / `glm-ocr` | Full-page `Text Recognition:` prompt; greedy SDPA inference, 8192-token limit; no SDK layout detector | Text/Markdown, no layout boxes; 200 DPI |
| Unlimited-OCR 2 / `unlimited-ocr-2` | All selected pages of a document in one patched llama.cpp request; BF16 decoder + F16 vision | Markdown + layout; 144 DPI, 1024px global views |
| Nemotron SkyVision CPU / `nemotron-skyvision-cpu` | F16 GGUF full-page parse, followed by NVIDIA's text-only normalization | Markdown, HTML tables + layout; 150 DPI |
| Docling Standard / `docling` | Default PDF pipeline, CPU, OCR when needed, layout and TableFormer | Markdown + layout; original file |
| Granite Docling / `granite-docling-258m` | Docling VLM pipeline; page images → DocTags → Markdown; greedy, up to 8192 tokens/page | Markdown + layout; scale 2 / 144 DPI for PDFs |
| Marker Fast / `marker_fast` | `marker-pdf==2.1.0`, `PdfConverter`, `mode="fast"`, `disable_ocr=True`, `use_llm=False` | Embedded text + lightweight layout; **no OCR on scans** |
| Marker Balanced / `marker_balanced` | Same package; `mode="balanced"`, `disable_ocr=False`, `force_ocr=False`, `use_llm=False`; `datalab-to/surya-ocr-2.5` | Markdown + layout; Surya through local llama.cpp |
| OpenDataLoader Core / `opendataloader_core` | `opendataloader-pdf==2.5.18`, `hybrid="off"` | Java text/structure extraction; **no OCR** |
| OpenDataLoader Docling / `opendataloader_docling` | `[hybrid]`, `hybrid="docling-fast"`, `hybrid_mode="auto"`, `hybrid_fallback=False` | Markdown/JSON + layout; CPU Docling/EasyOCR, languages `de,en` |
| docTR compact / `doctr_compact` | `db_mobilenet_v3_large` + `crnn_mobilenet_v3_small` | Text + word boxes; 200 DPI |
| docTR default / `doctr_default` | `fast_base` + `crnn_vgg16_bn` | Text + word boxes; 200 DPI |
| docTR Large / `doctr_large_transformer` | `db_resnet50` + `vitstr_base` | Text + word boxes; 200 DPI |
| PaddleOCR / `paddleocr_tiny`, `paddleocr_small`, `paddleocr_medium` | Matching PP-OCRv6 detectors + recognizers through Transformers; float32 | Text + line polygons/confidence; 200 DPI |
| Tesseract / `tesseract` | `pytesseract.image_to_string`, English by default | Plain text; 300 DPI |
| pypdf / `pypdf` | `page.extract_text()` | Existing PDF text; **no OCR** |

Some adapters and post-processing derive from
[ParseBench](https://github.com/run-llama/ParseBench), with vendored Apache-2.0 code
under `engines/vendor` and `engines/kdl`. Preset results are not new measurements of
the upstream leaderboard, and full-precision GGUF conversions can differ from native inference.

## Hardware paths

- **Windows / AMD:** PyTorch uses ROCm. LightOnOCR, Infinity and Ovis use F16 GGUF
  weights/projectors through llama.cpp Vulkan. Both paths use the AMD GPU.
- **Linux / NVIDIA:** native vision models use Transformers/CUDA. Marker Balanced
  uses CUDA llama.cpp; Unlimited uses its own patched CUDA runtime.
- **CPU mode:** GPU offloading and remote model endpoints are disabled. Native VLMs
  use supported BF16 or float32; Docling, docTR and Paddle keep their preset precision.
  GGUF paths run on CPU, with Transformers fallback where supported. Five slow methods
  are intentionally disabled; see the README.
- **Always on CPU:** Docling Standard, OpenDataLoader, pypdf, Tesseract, Marker Fast
  and Nemotron SkyVision CPU. Marker Balanced's auxiliary PyTorch models also use CPU.

Local llama.cpp servers bind to loopback, follow the model-cache limit and stop on
unload/shutdown. Their logs are in `data/runtime-logs/`. Exact conversion revisions
and filenames are in [the llama.cpp backend](../backend/app/engines/backends/llama_cpp.py).
F16/BF16 here means 16-bit weights, not 4/8-bit quantization.

## Details that affect comparison

### OvisOCR3: tables across pages

After per-page extraction, the model checks adjacent selected pages backwards through
multi-page table chains, following the
[pinned upstream workflow](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/17f255e2c8ae6d18d5d5edb0a474bd4ed39aef90/document_crosspage.py).
Whole-document view and downloads include joined tables; individual page output,
layout and judging retain the original extraction. Joined previews show source pages.
Total parsing time includes reconstruction; per-page timings cover extraction only.

The workflow joins the earlier page's last table to the later page's first table.
Skipped pages, incomplete JSON and chart-derived tables are excluded. Ambiguous merges
and malformed replies become diagnostics; completed page extractions remain available.
Chains of three or more pages require Chromium for re-rendering. Containers and the
Windows installer include it; manual setups use `python -m playwright install chromium`
(Linux: `--with-deps chromium`). MathJax is cached on first use. Set
`OVISOCR3_CROSSPAGE_TABLES=false` for page-only extraction or
`OVISOCR3_BROWSER_EXECUTABLE` for another browser. Existing runs are not rewritten.

### Unlimited-OCR 2: multi-page context

The [community runtime](https://huggingface.co/RX5950XT/Unlimited-OCR-2-GGUF) uses about
6.7 GB of BF16/F16 weights and preserves DASH routing and page-relative RoPE.
**It computes masked dense attention and does not provide the paper's optimized
sparse-memory scaling.** Memory still grows with document length.

Engine defaults are 16 selected pages, a 32,768-token context, up to 24,000 output
tokens, greedy decoding and a 1024px global view (273 visual tokens/page, no tiles).
The UI batch limit remains 12 unique pages. Output allowance shrinks to fit the input;
context shifting and prompt caching are disabled. Table tokens and supplied heading
levels are retained. Invalid page boundaries or token exhaustion are reported as failures.
Raw output is saved as `unlimited-raw.txt`. Seconds/page is an average of document time.

The separate runtime is installed by
[`install-unlimited.py`](../backend/scripts/install-unlimited.py), which pins the
b11913 base, patch revision and Windows binary checksums. Manual installation from
the repo root: `uv run --no-project --script backend/scripts/install-unlimited.py --download-model`.
Linux requires a C++ toolchain and CUDA Toolkit, or `--cpu` for a CPU build.
Containers already package it. Keep it separate from the other models' llama.cpp.

### Docling and Granite

Docling Standard combines embedded text with its default CPU pipeline. Granite
reads page images through the VLM pipeline, without substituting embedded PDF text,
quantization or FlashAttention 2. Selected pages have separate timings; truncated
generations are failures. Both exports retain recognized headers, footers, captions
and text inside picture regions, while excluding image-classification labels such as “Logo”.

### Marker and OpenDataLoader

Marker has a separate CPU PyTorch environment because its Pillow requirement conflicts
with MinerU's. Fast loads the lightweight layout helper, not the Surya OCR VLM.
Balanced uses the unquantized `surya-2.5.gguf` and projector through llama.cpp.
Neither preset uses cloud LLM post-processing. The Windows worker uses process handles
for shutdown where Surya's POSIX liveness checks are incompatible.

OpenDataLoader Docling uses automatic OCR and accurate TableFormer, with formula
enrichment and picture descriptions disabled. Automatic triage sends only pages that
need it to its loopback Docling backend. Backend errors are surfaced; there is no
silent Java-only fallback. Core reads only existing PDF text/structure.

### docTR, PaddleOCR and GLM-OCR

docTR uses `python-doctr==1.1.0`, pretrained predictors, `detect_layout=False`,
`detect_tables=False`, float32, and four preprocessing CPU threads. It recognizes
words and boxes but does not reconstruct tables.

Paddle's displayed sizes cover detector + recognizer, without orientation, unwarping
or table/layout networks. Detection uses min/max sides 736/4000, threshold 0.3,
box threshold 0.6 and unclip ratio 1.5; recognition batches eight rectified crops and
accepts confidence ≥0.5. PDFs are rasterized even when they contain text. The pinned
[checkpoints and settings](../backend/app/engines/paddle.py) define the three presets.

On Windows ROCm, docTR/Paddle BatchNorm and relevant LSTM layers use native PyTorch
GPU kernels to avoid unsupported MIOpen paths. Paddle Medium's eight 9×9 detector
convolutions also use the native GPU path. Weights and float32 precision are unchanged.

GLM runs the full-page model directly, so SDK scores that include its extra layout
detector do not describe this preset. Vision patch/merger operations use equivalent
matrix multiplications to avoid slow AMD kernel selection. Pipe tables are normalized
to HTML to preserve irregular cells; raw text remains in `glm-page-N.md`.

### Nemotron SkyVision CPU

This is [SkyContact's 631M F16 conversion](https://huggingface.co/SkyContact/Nemotron-3.5-Parse-GGUF),
about 1.31 GB of weights, running on CPU under SkyVision.cpp. NVIDIA's pinned text-only
postprocessor preserves merged table headers; it loads no neural model.
The worker uses stdin/stdout, greedy decoding, 8,744 output tokens and a 24,576-token
context. Markdown, boxes, raw structured output and normalizer provenance are retained.

[`install-skyvision.py`](../backend/scripts/install-skyvision.py) pins SkyVision/ggml
and builds a static AVX2/FMA/F16C executable. It downloads a portable compiler on
Windows; Linux requires a C/C++ compiler. Run it with
`uv run --no-project --script backend/scripts/install-skyvision.py` for manual setup.
Containers and the Windows helper already include it. Keep `build-info.json` beside
the executable when overriding `SKYVISION_BINARY`.

## Remote model servers

Compatible models can use an OpenAI-compatible serving endpoint, such as vLLM.
Set `<MODEL>_ENDPOINT_URL` using the model ID uppercased with punctuation replaced
by underscores, and optionally `<MODEL>_SERVED_MODEL` / `VLLM_API_KEY`.
Endpoints take precedence over local execution except in CPU-only mode.
Remote parsing sends page content to the configured server.

Example on a compatible CUDA serving host:

```bash
vllm serve cloudraker/rakedoc-nano --served-model-name rakedoc-nano \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 24 \
  --trust-remote-code --limit-mm-per-prompt '{"image":1}' --port 8001
```

```dotenv
RAKEDOC_NANO_ENDPOINT_URL=http://gpu-host:8001/v1
```

Other adapter defaults and serving requirements:

- Aeddix: `aeddix-labs/aeddix-alpine-ocr`, revision `v0.1`, served name
  `aeddix-alpine-ocr`; the same KDL flags as above.
- Florin: `florin-inc/florin-parser-nano`, served name `florin-parser-nano`;
  the same KDL flags as above.
- LightOnOCR: `lightonai/LightOnOCR-3-0.8B`, served name `loocr-grounding`,
  context 24576, one image per prompt; the existing setup uses vLLM 0.30 or later.
- MinerU: `opendatalab/MinerU2.5-Pro-2605-1.2B`, with
  `--logits-processors mineru_vl_utils:MinerULogitsProcessor`.
- Infinity/Ovis: follow their model-card serving requirements and set
  `INFINITY_PARSER2_FLASH_ENDPOINT_URL` / `OVISOCR3_ENDPOINT_URL`.

Model-server setup and capacity are managed separately from the Arena containers.
