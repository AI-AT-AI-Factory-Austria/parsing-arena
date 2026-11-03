"""Arena engine for checkpoints that run on the KDL two-stage pipeline."""

from pathlib import Path
from typing import Any

from ..backends.transformers_local import LocalVisionModel
from ..base import PageOutput, PageSource, ParseOutput, Report
from ..open_engine import OpenModelEngine
from . import florin_emission
from . import kdl_frontier_nano as K

# Checkpoints whose ParseBench provider swaps in the Florin markdown emission.
FLORIN_EMISSION = {"rakedoc-nano", "florin-parser-nano"}


class KdlEngine(OpenModelEngine):
    def build_local(self) -> LocalVisionModel:
        assert self.model.hf_repo
        return LocalVisionModel(
            self.model.hf_repo,
            self.model.revision,
            device=self.settings.local_device,
            dtype=self.settings.local_dtype,
            max_batch=self.settings.local_max_batch,
            max_model_len=self.settings.local_max_model_len,
        )

    async def parse(
        self, backend: Any, pages: PageSource, *, report: Report, artifact_dir: Path
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

        florin = self.model.id in FLORIN_EMISSION
        engine = K._NanoEngine(
            backend,
            self.served_model,
            self.settings.kdl_max_concurrent,
            category_map=florin_emission.FLORIN_NATIVE_LAYOUT_CATEGORY_MAP if florin else None,
            build_markdown=florin_emission.build_markdown if florin else None,
            progress=on_event,
            artifact_dir=artifact_dir,
        )
        try:
            raw = await engine.parse_pages(pages.load_all(), pages.numbers)
        finally:
            await self.close_backend(backend)

        markdown_by_page = {p["page_number"]: p["content"] for p in raw["markdown_pages"]}
        elements_by_page = {p["page_number"]: p["elements"] for p in raw["pages"]}
        outputs = [
            PageOutput(
                page_number=n,
                markdown=markdown_by_page.get(n, ""),
                seconds=raw["page_seconds"].get(n),
                elements=elements_by_page.get(n, []),
            )
            for n in pages.numbers
        ]
        return ParseOutput(markdown=raw["markdown"], pages=outputs)
