from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

# Receives partial progress updates such as {"phase": "layout", "page": 2}.
Report = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class RenderSpec:
    """How a pipeline rasterises PDF pages. Image uploads are used as uploaded."""

    dpi: int
    max_pixels: int | None = None


@dataclass(frozen=True)
class PageSource:
    """Pre-rendered pages on disk, loaded one at a time to bound memory, plus the
    original file for pipelines that read the document itself."""

    numbers: list[int]
    paths: dict[int, Path]
    source: Path | None = None
    extension: str | None = None

    def load(self, page_number: int) -> Image.Image:
        with Image.open(self.paths[page_number]) as image:
            return image.convert("RGB")

    def load_all(self) -> list[Image.Image]:
        return [self.load(n) for n in self.numbers]


@dataclass
class PageOutput:
    page_number: int
    markdown: str
    seconds: float | None = None
    elements: list[dict[str, Any]] | None = None
    error: str | None = None


@dataclass
class ParseOutput:
    markdown: str
    pages: list[PageOutput]
    meta: dict[str, Any] = field(default_factory=dict)


class EngineUnavailable(RuntimeError):
    """The engine is not configured on this server."""


def element(category: str, bbox: list[float], content: str, order: int) -> dict[str, Any]:
    """One layout element in the arena's format: bbox is [x1, y1, x2, y2] in 0-1."""
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in bbox)
    return {
        "category": category,
        "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
        "content": content,
        "layout_order": order,
    }
