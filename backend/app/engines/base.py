from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# Receives partial progress updates such as {"phase": "layout", "page": 2}.
Report = Callable[[dict[str, Any]], Awaitable[None]]


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
