"""OvisOCR3: structured JSON parse per page, then each chart re-read as chart JSON.

Mirrors ParseBench's ``ovisocr3_vllm`` pipeline: 200-DPI PNG pages, the model card's
``document_structured`` and ``chart_parsing`` prompts, greedy decoding with 16384 max
tokens and thinking disabled.
"""

from pathlib import Path
from typing import Any

from PIL import Image

from .backends.transformers_local import LocalVisionModel
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import PageFailed, parse_page_by_page, png_data_url
from .qwen35 import Qwen35Engine
from .vendor import parsebench_ovisocr3 as O

MAX_MODEL_LEN = 32768


def crop_region(page: Image.Image, bbox: list[float]) -> Image.Image | None:
    """The model card's crop conversion (``x * width / 1000``), unpadded."""
    width, height = page.size
    x1 = max(0, min(width, round(bbox[0] * width / O.COORD_SCALE)))
    y1 = max(0, min(height, round(bbox[1] * height / O.COORD_SCALE)))
    x2 = max(0, min(width, round(bbox[2] * width / O.COORD_SCALE)))
    y2 = max(0, min(height, round(bbox[3] * height / O.COORD_SCALE)))
    # Too small to hold a readable chart — not worth a request.
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return page.crop((x1, y1, x2, y2)).convert("RGB")


class OvisOcr3Engine(Qwen35Engine):
    render = RenderSpec(dpi=200)
    rocm_unsupported = (
        "Disabled on AMD ROCm GPUs: it shares LightOnOCR-3's Qwen3.5 architecture, whose local "
        "runs crashed Windows on a Radeon 8060S. Run it on an NVIDIA GPU or via a vLLM endpoint "
        "(OVISOCR3_ENDPOINT_URL)."
    )

    def build_local(self):
        if self.uses_vulkan:
            return self.vulkan_backend(MAX_MODEL_LEN)
        assert self.model.hf_repo
        # Qwen3.5 linear-attention layers: one request at a time, no left padding.
        return LocalVisionModel(
            self.model.hf_repo,
            self.model.revision,
            device=self.settings.local_device,
            dtype=self.settings.local_dtype,
            max_batch=1,
            max_model_len=MAX_MODEL_LEN,
        )

    async def _ask(self, backend: Any, image: Image.Image, prompt: str) -> str:
        payload = {
            "model": self.served_model or O.SERVED_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": png_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": O.DEFAULT_MAX_TOKENS,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return await backend.complete(payload) or ""

    async def parse(
        self, backend: Any, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        async def parse_one(number: int, image: Image.Image):
            content = await self._ask(backend, image, O.STRUCTURED_PROMPT)
            if not content:
                raise PageFailed("The model returned no output for this page.")
            regions = O.parse_regions(content)
            for region in regions:
                if region["category"] != "chart":
                    continue
                crop = crop_region(image, region["bbox"])
                if crop is None:
                    continue
                region["chart_html"] = O.chart_description_to_html(
                    await self._ask(backend, crop, O.CHART_PROMPT)
                )
            markdown, items = O.page_markdown(regions)
            layout = [
                element(item["label"], [v / O.COORD_SCALE for v in item["bbox"]], item["text"], i)
                for i, item in enumerate(items)
            ]
            return markdown, layout

        try:
            return await parse_page_by_page(pages, report, parse_one)
        finally:
            await self.close_backend(backend)
