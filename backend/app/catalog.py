"""Models in the arena.

Scores are the official ParseBench leaderboard values
(github.com/run-llama/ParseBench, leaderboard.csv at commit 0ccc167).
"""

from dataclasses import asdict, dataclass, field
from typing import Literal

LLAMAPARSE_ID = "llamaparse-agentic-plus"


@dataclass(frozen=True)
class Scores:
    overall: float
    tables: float
    charts: float
    content: float
    formatting: float
    grounding: float


@dataclass(frozen=True)
class ArenaModel:
    id: str
    name: str
    publisher: str
    params: str
    params_b: float | None
    license: str
    # None for models that are not on the ParseBench leaderboard.
    scores: Scores | None
    kind: Literal["open", "reference"] = "open"
    implemented: bool = False
    hf_repo: str | None = None
    revision: str | None = None
    # ParseBench marks community submissions that the maintainers have not re-run.
    verified: bool = True
    lineage: str = ""
    pipeline: str = ""
    specialty: str = ""
    links: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


MODELS: list[ArenaModel] = [
    ArenaModel(
        id="rakedoc-nano",
        name="rakedoc-nano",
        specialty="Merged cells & complex table headers",
        publisher="CloudRaker",
        params="1.2B",
        params_b=1.2,
        license="AGPL-3.0",
        scores=Scores(77.23, 86.44, 64.89, 88.84, 71.68, 74.28),
        implemented=True,
        hf_repo="cloudraker/rakedoc-nano",
        verified=False,
        lineage="Fine-tune of florin-parser-nano (KDL-Frontier-Parser-nano family)",
        pipeline="KDL two-stage pipeline with Florin markdown emission",
    ),
    ArenaModel(
        id="aeddix-alpine-ocr",
        name="Aeddix Alpine OCR",
        specialty="Tables & inline text formatting",
        publisher="Aeddix Labs",
        params="1.2B",
        params_b=1.2,
        license="Apache-2.0",
        scores=Scores(76.77, 85.89, 68.78, 87.90, 66.05, 75.21),
        implemented=True,
        hf_repo="aeddix-labs/aeddix-alpine-ocr",
        revision="v0.1",
        verified=False,
        lineage="Fine-tune of MinerU2.5-Pro-2605-1.2B",
        pipeline="KDL two-stage pipeline",
    ),
    ArenaModel(
        id="florin-parser-nano",
        name="florin-parser-nano",
        specialty="Preserves bold, strikethrough & superscripts",
        publisher="Florin",
        params="1.2B",
        params_b=1.2,
        license="AGPL-3.0",
        scores=Scores(76.69, 86.10, 65.19, 87.37, 70.64, 74.14),
        implemented=True,
        hf_repo="florin-inc/florin-parser-nano",
        revision="61cf54a26b6bf24ce6993e0c81e7bdcab4c423c4",
        verified=False,
        lineage="Fine-tune of KDL-Frontier-Parser-nano",
        pipeline="KDL two-stage pipeline with Florin markdown emission",
    ),
    ArenaModel(
        id="nemotron-3.5-parse",
        name="Nemotron-3.5-Parse",
        publisher="NVIDIA",
        params="0.631B",
        params_b=0.631,
        license="OpenMDW-1.1",
        scores=Scores(75.07, 84.62, 63.85, 88.04, 69.06, 69.78),
        implemented=True,
        hf_repo="nvidia/Nemotron-3.5-Parse",
        revision="4035ffb1d8f8bafcdb867dfaf6f7514f101464f9",
        lineage="C-RADIOv4 vision encoder with an mBART decoder",
        pipeline="Nemotron Parse SDK, triple pass (page, detection, table and chart crops)",
    ),
    ArenaModel(
        id="nemotron-skyvision-cpu",
        name="Nemotron SkyVision CPU",
        specialty="Structured document extraction on CPU",
        publisher="NVIDIA / SkyContact",
        params="0.631B",
        params_b=0.631,
        license="OpenMDW-1.1",
        scores=None,
        implemented=True,
        hf_repo="SkyContact/Nemotron-3.5-Parse-GGUF",
        revision="eaa805bb6d020798998722837a57107a5d4f94b8",
        lineage="Nemotron-3.5-Parse, F16 GGUF conversion by SkyContact",
        pipeline=(
            "SkyVision CPU, F16 full-page parse; NVIDIA text normalization to Markdown and layout"
        ),
    ),
    ArenaModel(
        id="lightonocr-3-0.8b",
        name="LightOnOCR-3-0.8B",
        specialty="Charts & page layout in one pass",
        publisher="LightOn",
        params="0.8B class",
        params_b=0.8,
        license="Apache-2.0",
        scores=Scores(74.74, 84.42, 64.55, 88.82, 66.44, 69.47),
        implemented=True,
        hf_repo="lightonai/LightOnOCR-3-0.8B",
        revision="36d8636925327c671bd8e479c517a004f7566bd9",
        lineage="Qwen3.5 vision-language architecture",
        pipeline="Single pass with the grounding prompt (markdown and layout together)",
    ),
    ArenaModel(
        id="mineru2.5-pro-2605",
        name="MinerU2.5-Pro-2605",
        specialty="Formulas, charts & technical diagrams",
        publisher="OpenDataLab",
        params="1.2B",
        params_b=1.2,
        license="Apache-2.0 (weights)",
        scores=Scores(69.27, 77.61, 59.12, 88.23, 47.15, 74.24),
        implemented=True,
        hf_repo="opendatalab/MinerU2.5-Pro-2605-1.2B",
        revision="08aaea840498d49ce16247b6263196cf02814885",
        lineage="Qwen2-VL architecture",
        pipeline="mineru-vl-utils two-step extraction with image analysis",
    ),
    ArenaModel(
        id="infinity-parser2-flash",
        name="Infinity-Parser2-Flash",
        specialty="Chart data as tables & scientific page layouts",
        publisher="inftech.ai / infly",
        params="≈2B",
        params_b=2.0,
        license="Apache-2.0",
        scores=Scores(68.63, 82.88, 52.67, 88.61, 44.04, 74.95),
        implemented=True,
        hf_repo="infly/Infinity-Parser2-Flash",
        revision="cfddc4106b0abc4706f05575a164f7dd35d09c46",
        lineage="Qwen3.5 vision-language architecture",
        pipeline="doc2json single pass plus figure-to-table deep parsing",
    ),
    # Experiments run models in this order; these follow from lightest to heaviest.
    ArenaModel(
        id="pypdf",
        name="pypdf",
        specialty="Embedded PDF text · no OCR",
        publisher="py-pdf (open source)",
        params="No model",
        params_b=0.0,
        license="BSD-3-Clause",
        scores=Scores(14.87, 0.00, 0.00, 62.50, 0.91, 10.92),
        implemented=True,
        verified=False,
        lineage="Reads the text layer of PDFs; no OCR, so scans come back empty",
        pipeline="Embedded PDF text extraction (pypdf_baseline)",
        links={"code": "https://github.com/py-pdf/pypdf"},
    ),
    ArenaModel(
        id="tesseract",
        name="Tesseract",
        specialty="Clean scans of printed text",
        publisher="Tesseract OCR (open source)",
        params="Classic OCR (LSTM)",
        params_b=0.0,
        license="Apache-2.0",
        scores=None,
        implemented=True,
        lineage="Plain text per page; no tables, headings or layout",
        pipeline="pytesseract image_to_string at 300 DPI (ParseBench tesseract_eng)",
        links={"code": "https://github.com/tesseract-ocr/tesseract"},
    ),
    ArenaModel(
        id="docling",
        name="Docling Standard",
        specialty="Reading order, page structure & tables",
        publisher="IBM / LF AI & Data",
        params="≈0.3B (layout + TableFormer)",
        params_b=0.3,
        license="MIT (code), Apache-2.0 / CDLA (models)",
        scores=Scores(50.65, 66.41, 52.76, 66.93, 1.03, 66.11),
        implemented=True,
        hf_repo="docling-project/docling-models",
        verified=False,
        lineage="Layout detection and TableFormer models, runs on the CPU",
        pipeline="Docling default PDF pipeline on the CPU",
    ),
    ArenaModel(
        id="granite-docling-258m",
        name="Granite Docling 258M",
        specialty="Tables, equations & page structure in a compact model",
        publisher="IBM Research",
        params="258M",
        params_b=0.258,
        license="Apache-2.0",
        scores=None,
        implemented=True,
        hf_repo="ibm-granite/granite-docling-258M",
        revision="982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe",
        lineage="Compact VLM with a SigLIP2 vision encoder and Granite language model",
        pipeline="Docling VLM pipeline, Granite DocTags to Markdown and layout boxes",
    ),
    ArenaModel(
        id="falcon-ocr",
        name="Falcon-OCR",
        publisher="TII",
        params="0.27B",
        params_b=0.27,
        license="Apache-2.0",
        scores=Scores(52.07, 74.70, 0.82, 78.59, 47.91, 58.32),
        implemented=True,
        hf_repo="tiiuae/Falcon-OCR",
        revision="0bc28f89c90fbe8bdb063097fab144d4ce4c19b5",
        verified=False,
        lineage="Early-fusion OCR model with a PP-DocLayoutV3 layout detector",
        pipeline="generate_with_layout: layout detection, then OCR per region",
    ),
    ArenaModel(
        id="ovisocr3",
        name="OvisOCR3",
        specialty="Joins tables across pages & extracts chart data",
        publisher="ATH-MaaS",
        params="0.85B",
        params_b=0.85,
        license="Apache-2.0",
        scores=Scores(71.81, 80.18, 62.50, 89.12, 53.29, 73.94),
        implemented=True,
        hf_repo="ATH-MaaS/OvisOCR3",
        revision="17f255e2c8ae6d18d5d5edb0a474bd4ed39aef90",
        verified=False,
        lineage="Built on Qwen3.5-0.8B",
        pipeline="Structured pages and chart reading, then cross-page table reconstruction",
    ),
    ArenaModel(
        id="unlimited-ocr-2",
        name="Unlimited-OCR 2",
        specialty="Reads multiple pages together with shared context",
        publisher="Baidu",
        params="3.3B",
        params_b=3.3,
        license="MIT",
        scores=Scores(57.15, 72.74, 1.06, 89.16, 49.57, 73.20),
        hf_repo="baidu/Unlimited-OCR-2",
        implemented=True,
        revision="0480966938291445591f69f08a13b041c073a96a",
        lineage="Baidu Unlimited-OCR 2; BF16 GGUF decoder with F16 vision encoder",
        pipeline=(
            "All selected pages in one request (up to 16 by default); 1024px global views, "
            "DASH routing and page-relative RoPE. Patched llama.cpp on local CPU/Vulkan/CUDA; "
            "masked dense computation, without the optimized sparse-memory scaling."
        ),
        links={"runtime": "https://huggingface.co/RX5950XT/Unlimited-OCR-2-GGUF"},
        verified=False,
    ),
]

# These exact presets have no published ParseBench scores. Sizes describe their
# principal networks, not the sum of every auxiliary component.
MODELS.extend(
    [
        ArenaModel(
            id="marker_fast",
            name="Marker Fast",
            specialty="Digital PDFs · embedded text, no OCR",
            publisher="Datalab",
            params="≈20M layout + auxiliaries",
            params_b=0.020,
            license="GPL-3.0 (code), model-specific weights",
            scores=None,
            implemented=True,
            lineage="Lightweight layout detector; embedded PDF text only, no OCR or Surya VLM",
            pipeline="PdfConverter: fast, disable_ocr=True, use_llm=False; no OCR on scans",
            links={"docs": "https://github.com/datalab-to/marker"},
        ),
        ArenaModel(
            id="marker_balanced",
            name="Marker Balanced",
            specialty="Scanned pages, tables & inline math",
            publisher="Datalab",
            params="≈689M Surya + auxiliaries",
            params_b=0.689,
            license="GPL-3.0 (code), OpenRAIL (Surya weights)",
            scores=None,
            implemented=True,
            hf_repo="datalab-to/surya-ocr-2.5",
            lineage="Surya OCR 2.5 with layout and auxiliary components",
            pipeline="PdfConverter: balanced, disable_ocr=False, force_ocr=False, use_llm=False",
            links={"docs": "https://github.com/datalab-to/marker"},
        ),
        ArenaModel(
            id="opendataloader_core",
            name="OpenDataLoader Core",
            specialty="Digital PDF structure · no OCR",
            publisher="OpenDataLoader",
            params="0 neural parameters",
            params_b=0,
            license="Apache-2.0",
            scores=None,
            implemented=True,
            lineage="Java extraction of embedded PDF text and structure; no OCR",
            pipeline="OpenDataLoader: hybrid=off; embedded PDF text only, no OCR on scans",
            links={"docs": "https://www.opendataloader.org/docs"},
        ),
        ArenaModel(
            id="opendataloader_docling",
            name="OpenDataLoader Docling",
            specialty="Mixed digital/scanned PDFs with German/English OCR",
            publisher="OpenDataLoader",
            params="Backend-dependent",
            params_b=None,
            license="Apache-2.0 (code), model-specific weights",
            scores=None,
            implemented=True,
            lineage="Java extraction with local Docling, EasyOCR (de/en), layout and tables",
            pipeline=(
                "docling-fast, auto triage, no fallback; EasyOCR de,en; "
                "OCR/tables on; enrichments off"
            ),
            links={"docs": "https://www.opendataloader.org/docs/hybrid-mode"},
        ),
        ArenaModel(
            id="doctr_compact",
            name="docTR compact",
            specialty="Text detection & recognition with low compute needs",
            publisher="Mindee",
            params="≈6.3M",
            params_b=0.0063,
            license="Apache-2.0",
            scores=None,
            implemented=True,
            lineage="MobileNet detector and CRNN MobileNet recognizer",
            pipeline=(
                "db_mobilenet_v3_large + crnn_mobilenet_v3_small; pretrained; layout/tables off"
            ),
            links={"docs": "https://mindee.github.io/doctr/using_doctr/using_models.html"},
        ),
        ArenaModel(
            id="doctr_default",
            name="docTR default",
            specialty="Balanced OCR for printed documents",
            publisher="Mindee",
            params="≈26.4M at inference",
            params_b=0.0264,
            license="Apache-2.0",
            scores=None,
            implemented=True,
            lineage="FAST base detector and CRNN VGG16 recognizer",
            pipeline="fast_base + crnn_vgg16_bn; pretrained; layout/tables off",
            links={"docs": "https://mindee.github.io/doctr/using_doctr/using_models.html"},
        ),
        ArenaModel(
            id="doctr_large_transformer",
            name="docTR Large transformer",
            specialty="Accuracy-focused text recognition",
            publisher="Mindee",
            params="≈110.6M",
            params_b=0.1106,
            license="Apache-2.0",
            scores=None,
            implemented=True,
            lineage="DB ResNet50 detector and ViTSTR base transformer recognizer",
            pipeline="db_resnet50 + vitstr_base; pretrained; layout/tables off",
            links={"docs": "https://mindee.github.io/doctr/using_doctr/using_models.html"},
        ),
    ]
)

# PP-OCRv6 sizes include the matching detector + recognizer, without optional models.
MODELS.extend(
    ArenaModel(
        id=f"paddleocr_{tier}",
        name=f"PaddleOCR {tier.title()}",
        specialty={
            "tiny": "Tiny multilingual text OCR",
            "small": "Lightweight multilingual text OCR",
            "medium": "Multilingual, screen & scene text",
        }[tier],
        publisher="PaddlePaddle",
        params=size,
        params_b=billions,
        license="Apache-2.0",
        scores=None,
        implemented=True,
        hf_repo=f"PaddlePaddle/PP-OCRv6_{tier}_det_safetensors",
        revision=revision,
        lineage=f"PP-OCRv6 {tier}: detector + matching multilingual text recognizer",
        pipeline=(
            "PP-OCRv6 detection + recognition via Transformers; "
            "text lines and boxes; orientation, unwarping and table models off"
        ),
        links={
            "recognizer": f"https://huggingface.co/PaddlePaddle/PP-OCRv6_{tier}_rec_safetensors",
            "docs": "https://huggingface.co/blog/PaddlePaddle/pp-ocrv6",
        },
    )
    for tier, size, billions, revision in [
        ("tiny", "1.5M", 0.0015, "07595f982703daf0d4e120a12a01da8073542f3a"),
        ("small", "7.7M", 0.0077, "eae2ee920a39fb3087637d3dbb58df1896ec1f24"),
        ("medium", "34.5M", 0.0345, "4236c2b61741a259c091fd879dcc4edc339e916c"),
    ]
)
MODELS.append(
    ArenaModel(
        id="glm-ocr",
        name="GLM-OCR 0.9B",
        specialty="Compact full-page text recognition",
        publisher="Z.ai",
        params="0.9B",
        params_b=0.9,
        license="MIT",
        scores=None,
        implemented=True,
        hf_repo="zai-org/GLM-OCR",
        revision="2e85a62840ccac27daa451df36c736c4636b8628",
        lineage="CogViT vision encoder with a GLM language decoder",
        pipeline="Full-page Text Recognition via Transformers; no additional layout detector",
        links={"docs": "https://huggingface.co/docs/transformers/en/model_doc/glm_ocr"},
    )
)

REFERENCE = ArenaModel(
    id=LLAMAPARSE_ID,
    name="LlamaParse Agentic Plus",
    specialty="Complex tables & charts · judging reference",
    publisher="LlamaIndex",
    params="Hosted API",
    params_b=None,
    license="Commercial API",
    scores=Scores(86.69, 93.27, 92.19, 89.74, 82.84, 75.43),
    kind="reference",
    implemented=True,
    pipeline="LlamaParse v2 API, tier agentic_plus",
    links={"docs": "https://developers.llamaindex.ai/python/cloud/llamaparse/"},
)

LLAMAPARSE_CREDITS_PER_PAGE = 45
LLAMAPARSE_USD_PER_CREDIT = 0.00125

# Keep metadata for saved runs, but exclude retired methods from the active catalog.
RETIRED_MODELS = {m.id: m for m in MODELS if m.id in {"nemotron-3.5-parse", "falcon-ocr"}}
MODELS = [m for m in MODELS if m.id not in RETIRED_MODELS]
ALL_MODELS: dict[str, ArenaModel] = {m.id: m for m in [*MODELS, REFERENCE]}


def get_model(model_id: str) -> ArenaModel | None:
    return ALL_MODELS.get(model_id) or RETIRED_MODELS.get(model_id)
