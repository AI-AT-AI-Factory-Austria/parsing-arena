"""Arena engine for checkpoints that run on the KDL two-stage pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ...catalog import ArenaModel
from ...config import Settings
from ..backends.openai_compat import OpenAICompatBackend
from ..backends.transformers_local import LocalModelPool, local_inference_missing
from ..base import EngineUnavailable, PageOutput, ParseOutput, Report
from . import florin_emission
from . import kdl_frontier_nano as K


@dataclass(frozen=True)
class KdlVariant:
    florin_emission: bool
    endpoint_setting: str
    served_model_setting: str


VARIANTS: dict[str, KdlVariant] = {
    "rakedoc-nano": KdlVariant(True, "rakedoc_nano_endpoint_url", "rakedoc_nano_served_model"),
    "aeddix-alpine-ocr": KdlVariant(
        False, "aeddix_alpine_ocr_endpoint_url", "aeddix_alpine_ocr_served_model"
    ),
}


class KdlEngine:
    def __init__(self, model: ArenaModel, settings: Settings, pool: LocalModelPool):
        self.model = model
        self.variant = VARIANTS[model.id]
        self.settings = settings
        self.pool = pool
        self.endpoint_url: str = getattr(settings, self.variant.endpoint_setting).strip()
        self.served_model: str = getattr(settings, self.variant.served_model_setting)

    @property
    def backend_kind(self) -> str:
        return "endpoint" if self.endpoint_url else "local"

    def unavailable_reason(self) -> str | None:
        if self.endpoint_url:
            return None
        if not self.settings.local_inference:
            return "Local inference is disabled and no endpoint URL is configured."
        return local_inference_missing()

    async def prepare(self) -> tuple[Any, float]:
        """Return (chat backend, seconds spent loading)."""
        reason = self.unavailable_reason()
        if reason:
            raise EngineUnavailable(reason)
        if self.endpoint_url:
            backend = OpenAICompatBackend(
                self.endpoint_url,
                api_key=self.settings.vllm_api_key,
                timeout_s=self.settings.endpoint_timeout_s,
            )
            return backend, 0.0
        assert self.model.hf_repo
        return await self.pool.acquire(self.model.hf_repo, self.model.revision)

    def describe(self, backend: Any) -> dict[str, Any]:
        info: dict[str, Any] = {"pipeline": self.model.pipeline, "revision": self.model.revision}
        if isinstance(backend, OpenAICompatBackend):
            info.update(
                {"backend": "endpoint", "endpoint": backend.url, "served_model": self.served_model}
            )
        else:
            info.update(backend.describe())
        return info

    async def parse(
        self,
        backend: Any,
        images: list[Image.Image],
        page_numbers: list[int],
        *,
        report: Report,
        artifact_dir: Path,
    ) -> ParseOutput:
        async def on_event(event: dict[str, Any]) -> None:
            kind = event["event"]
            if kind == "page_start":
                await report(
                    {
                        "phase": "layout",
                        "page": event["page"],
                        "page_index": event["index"],
                        "regions_done": 0,
                        "regions_total": None,
                    }
                )
            elif kind == "layout_done":
                await report(
                    {
                        "phase": "recognition",
                        "page": event["page"],
                        "regions_done": 0,
                        "regions_total": event["regions"],
                    }
                )
            elif kind == "region_done":
                await report({"regions_done": event["done"], "regions_total": event["regions"]})
            elif kind == "page_done":
                await report(
                    {
                        "pages_done": event["index"] + 1,
                        "page": event["page"],
                        "last_page_seconds": round(event["seconds"], 3),
                    }
                )

        engine = K._NanoEngine(
            backend,
            self.served_model,
            self.settings.kdl_max_concurrent,
            category_map=(
                florin_emission.FLORIN_NATIVE_LAYOUT_CATEGORY_MAP
                if self.variant.florin_emission
                else None
            ),
            build_markdown=florin_emission.build_markdown if self.variant.florin_emission else None,
            progress=on_event,
            artifact_dir=artifact_dir,
        )
        try:
            raw = await engine.parse_pages(images, page_numbers)
        finally:
            if isinstance(backend, OpenAICompatBackend):
                await backend.aclose()

        markdown_by_page = {p["page_number"]: p["content"] for p in raw["markdown_pages"]}
        elements_by_page = {p["page_number"]: p["elements"] for p in raw["pages"]}
        pages = [
            PageOutput(
                page_number=n,
                markdown=markdown_by_page.get(n, ""),
                seconds=raw["page_seconds"].get(n),
                elements=elements_by_page.get(n, []),
            )
            for n in page_numbers
        ]
        return ParseOutput(markdown=raw["markdown"], pages=pages)
