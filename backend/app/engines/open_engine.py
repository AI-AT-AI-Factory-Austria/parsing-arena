"""Shared plumbing for open-model engines: remote endpoint or local transformers."""

import asyncio
import base64
import io
import logging
import re
import time
from collections.abc import Awaitable, Callable, Hashable
from pathlib import Path
from typing import Any

from PIL import Image

from ..catalog import ArenaModel
from ..config import Settings
from .backends.openai_compat import OpenAICompatBackend
from .backends.transformers_local import (
    LocalModel,
    LocalModelPool,
    describe_local_device,
    local_inference_missing,
)
from .base import EngineUnavailable, PageOutput, PageSource, ParseOutput, RenderSpec, Report

logger = logging.getLogger(__name__)

# (page number, page image) -> (markdown, elements)
PageParser = Callable[[int, Image.Image], Awaitable[tuple[str, list[dict[str, Any]]]]]


def settings_prefix(model_id: str) -> str:
    """'nemotron-3.5-parse' -> 'nemotron_3_5_parse' (the Settings field prefix)."""
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class PageFailed(RuntimeError):
    """One page could not be parsed; the run continues with the next page."""


async def parse_page_by_page(
    pages: PageSource, report: Report, parse_one: PageParser
) -> ParseOutput:
    """Run a single-page pipeline over every page in order, reporting progress."""
    outputs: list[PageOutput] = []
    for index, number in enumerate(pages.numbers):
        await report({"phase": "parsing", "page": number, "page_index": index})
        started = time.perf_counter()
        image = await asyncio.to_thread(pages.load, number)
        try:
            markdown, elements = await parse_one(number, image)
            outputs.append(PageOutput(number, markdown, time.perf_counter() - started, elements))
        except PageFailed as exc:
            logger.warning("page %d failed: %s", number, exc)
            outputs.append(
                PageOutput(number, "", time.perf_counter() - started, [], error=str(exc))
            )
        seconds = outputs[-1].seconds or 0.0
        await report(
            {"pages_done": index + 1, "page": number, "last_page_seconds": round(seconds, 3)}
        )
    if all(p.error for p in outputs):
        raise RuntimeError(outputs[0].error or "No page could be parsed.")
    markdown = "\n\n".join(p.markdown for p in outputs if p.markdown)
    return ParseOutput(markdown=markdown, pages=outputs)


class OpenModelEngine:
    """Base class. Subclasses implement ``build_local`` and ``parse``."""

    # None: the pipeline reads the original file and needs no page images.
    render: RenderSpec | None = RenderSpec(dpi=144)
    # Why local inference is refused on AMD ROCm GPUs, for models that crashed it.
    rocm_unsupported: str | None = None
    # False for pipelines that only run in this process (no vLLM endpoint mode).
    supports_endpoint = True

    def __init__(self, model: ArenaModel, settings: Settings, pool: LocalModelPool):
        self.model = model
        self.settings = settings
        self.pool = pool
        prefix = settings_prefix(model.id)
        endpoint = getattr(settings, f"{prefix}_endpoint_url", "") if self.supports_endpoint else ""
        self.endpoint_url: str = str(endpoint).strip()
        self.served_model: str = str(getattr(settings, f"{prefix}_served_model", "") or "")

    @property
    def backend_kind(self) -> str:
        return "endpoint" if self.endpoint_url else "local"

    @property
    def pool_key(self) -> Hashable:
        return (type(self).__name__, self.model.hf_repo, self.model.revision)

    def unavailable_reason(self) -> str | None:
        if self.endpoint_url:
            return None
        if not self.settings.local_inference:
            return "Local inference is disabled and no endpoint URL is configured."
        missing = local_inference_missing()
        if missing:
            return missing
        if self.rocm_unsupported and not self.settings.allow_unstable_rocm_models:
            if describe_local_device(self.settings.local_device).get("runtime") == "ROCm":
                return self.rocm_unsupported
        return None

    def is_loaded(self) -> bool:
        return self.backend_kind == "local" and self.pool.is_loaded(self.pool_key)

    async def prepare(self) -> tuple[Any, float]:
        """Return (backend, seconds spent loading)."""
        reason = self.unavailable_reason()
        if reason:
            raise EngineUnavailable(reason)
        if self.endpoint_url:
            return self.endpoint_backend(), 0.0
        return await self.pool.acquire(self.pool_key, self.build_local)

    def endpoint_backend(self) -> Any:
        return OpenAICompatBackend(
            self.endpoint_url,
            api_key=self.settings.vllm_api_key,
            timeout_s=self.settings.endpoint_timeout_s,
        )

    def build_local(self) -> LocalModel:
        raise NotImplementedError

    def describe(self, backend: Any) -> dict[str, Any]:
        info: dict[str, Any] = {"pipeline": self.model.pipeline, "revision": self.model.revision}
        if self.endpoint_url:
            info.update(
                {
                    "backend": "endpoint",
                    "endpoint": self.endpoint_url,
                    "served_model": self.served_model,
                }
            )
        else:
            info.update(backend.describe())
        return info

    async def close_backend(self, backend: Any) -> None:
        if isinstance(backend, OpenAICompatBackend):
            await backend.aclose()

    async def release(self) -> None:
        await self.pool.release(self.pool_key)

    async def parse(
        self, backend: Any, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        raise NotImplementedError
