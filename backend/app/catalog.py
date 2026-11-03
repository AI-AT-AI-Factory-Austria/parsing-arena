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
    links: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


MODELS: list[ArenaModel] = [
    ArenaModel(
        id="rakedoc-nano",
        name="rakedoc-nano",
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
        id="lightonocr-3-0.8b",
        name="LightOnOCR-3-0.8B",
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
        name="Docling",
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
        id="surya-ocr-2",
        name="Surya OCR 2",
        publisher="Datalab",
        params="0.69B",
        params_b=0.69,
        license="OpenRAIL-M (weights)",
        scores=Scores(60.63, 82.74, 19.04, 87.84, 63.59, 49.93),
        hf_repo="datalab-to/surya-ocr-2",
    ),
    ArenaModel(
        id="ovisocr3",
        name="OvisOCR3",
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
        pipeline="Structured JSON parse per page, then a chart-reading pass per chart",
    ),
    ArenaModel(
        id="teleocr",
        name="TeleOCR",
        publisher="StarDoc-AI",
        params="1.4B",
        params_b=1.4,
        license="Apache-2.0",
        scores=Scores(57.52, 72.77, 9.97, 84.59, 45.74, 74.55),
        hf_repo="StarDoc-AI/TeleOCR",
    ),
    ArenaModel(
        id="unlimited-ocr-2",
        name="Unlimited-OCR 2",
        publisher="Baidu",
        params="3.3B",
        params_b=3.3,
        license="MIT",
        scores=Scores(57.15, 72.74, 1.06, 89.16, 49.57, 73.20),
        hf_repo="baidu/Unlimited-OCR-2",
        verified=False,
    ),
]

REFERENCE = ArenaModel(
    id=LLAMAPARSE_ID,
    name="LlamaParse Agentic Plus",
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

ALL_MODELS: dict[str, ArenaModel] = {m.id: m for m in [*MODELS, REFERENCE]}


def get_model(model_id: str) -> ArenaModel | None:
    return ALL_MODELS.get(model_id)
