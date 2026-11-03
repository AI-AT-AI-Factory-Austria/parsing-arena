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
    scores: Scores
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
        hf_repo="florin-inc/florin-parser-nano",
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
        hf_repo="nvidia/Nemotron-3.5-Parse",
    ),
    ArenaModel(
        id="lightonocr-3-0.8b",
        name="LightOnOCR-3-0.8B",
        publisher="LightOn",
        params="0.8B class",
        params_b=0.8,
        license="Apache-2.0",
        scores=Scores(74.74, 84.42, 64.55, 88.82, 66.44, 69.47),
        hf_repo="lightonai/LightOnOCR-3-0.8B",
    ),
    ArenaModel(
        id="mineru2.5-pro-2605",
        name="MinerU2.5-Pro-2605",
        publisher="OpenDataLab",
        params="1.2B",
        params_b=1.2,
        license="Apache-2.0 (weights)",
        scores=Scores(69.27, 77.61, 59.12, 88.23, 47.15, 74.24),
        hf_repo="opendatalab/MinerU2.5-Pro-2605-1.2B",
    ),
    ArenaModel(
        id="infinity-parser2-flash",
        name="Infinity-Parser2-Flash",
        publisher="inftech.ai / infly",
        params="≈2B",
        params_b=2.0,
        license="Apache-2.0",
        scores=Scores(68.63, 82.88, 52.67, 88.61, 44.04, 74.95),
        hf_repo="infly/Infinity-Parser2-Flash",
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
