"""Infinity-Parser2: doc2json in one pass, then figures re-read as markdown tables.

Mirrors ParseBench's ``infinity_parser2_flash`` pipeline: pages rendered at 300 DPI,
the infinity_parser2 SDK's vLLM-server request (smart-resized PNG, greedy decoding,
32768 max tokens), its JSON post-processing, the provider's deep-parsing pass over
``figure`` crops, and the provider's normalization.
"""

import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .backends.transformers_local import LocalVisionModel
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import PageFailed, parse_page_by_page, png_data_url
from .qwen35 import Qwen35Engine
from .vendor import parsebench_infinity2 as P

# The doc2json prompt as published on the Apache-2.0 model card (infly/Infinity-Parser2-Flash).
PROMPT_DOC2JSON = """
- Extract layout information from the provided PDF image.
- For each layout element, output its bbox, category, and the text content within the bbox.
- Bbox format: [x1, y1, x2, y2].
- Allowed layout categories: ['header', 'title', 'text', 'figure', 'table', 'formula', 'figure_caption', 'table_caption', 'formula_caption', 'figure_footnote', 'table_footnote', 'page_footnote', 'footer'].
- Text extraction and formatting:
  1) For 'figure', the text field must be an empty string.
  2) For 'formula', format text as LaTeX.
  3) For 'table', format text as HTML.
  4) For all other categories (e.g., text, title), format text as Markdown.
- The output text must be exactly the original text from the image, with no translation or rewriting.
- Sort all layout elements in human reading order.
- Final output must be a single JSON object.
"""  # noqa: E501
DEEP_PARSE_PROMPT = "please convert the image to a markdown table"

MIN_PIXELS = 2048
MAX_PIXELS = 16777216
RESIZE_FACTOR = 32
MAX_TOKENS = 32768
DEEP_PARSE_MAX_TOKENS = 2048
MAX_MODEL_LEN = 65536


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Qwen-VL resize policy: sides on a 32 px grid, total pixels within bounds."""
    h_bar = max(RESIZE_FACTOR, round(height / RESIZE_FACTOR) * RESIZE_FACTOR)
    w_bar = max(RESIZE_FACTOR, round(width / RESIZE_FACTOR) * RESIZE_FACTOR)
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(RESIZE_FACTOR, math.floor(height / beta / RESIZE_FACTOR) * RESIZE_FACTOR)
        w_bar = max(RESIZE_FACTOR, math.floor(width / beta / RESIZE_FACTOR) * RESIZE_FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / RESIZE_FACTOR) * RESIZE_FACTOR
        w_bar = math.ceil(width * beta / RESIZE_FACTOR) * RESIZE_FACTOR
    return h_bar, w_bar


def request_image(image: Image.Image) -> str:
    height, width = smart_resize(image.height, image.width)
    return png_data_url(image.convert("RGB").resize((width, height)))


def extract_json(raw: str) -> str:
    """The JSON inside a ```json fence (closed or cut off), else the raw text."""
    match = re.search(r"```json\n(.*?)\n```", raw, re.DOTALL) or re.search(
        r"```json\n(.*)", raw, re.DOTALL
    )
    return match.group(1).strip() if match else raw


def drop_incomplete_tail(text: str) -> str:
    """Cut a truncated element list back to its last complete element."""
    if len(text) <= 65536 and text.rstrip().endswith("]"):
        return text
    if text.count('{"bbox":') <= 1:
        return text
    cut = text[: text.rfind('{"bbox":')].rstrip()
    return cut[:-1] + "]" if cut.endswith(",") else cut


def load_elements(raw: str, width: int, height: int) -> list[dict[str, Any]]:
    """Parse the element list and map its 0-1000 boxes to page pixels."""
    try:
        data = json.loads(drop_incomplete_tail(extract_json(raw)))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    elements = [e for e in data if isinstance(e, dict)]
    for item in elements:
        bbox = item.get("bbox")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(c, int | float) for c in bbox)
        ):
            x1, y1, x2, y2 = bbox
            item["bbox"] = [
                int(x1 / 1000.0 * width),
                int(y1 / 1000.0 * height),
                int(x2 / 1000.0 * width),
                int(y2 / 1000.0 * height),
            ]
    return elements


class InfinityParser2Engine(Qwen35Engine):
    render = RenderSpec(dpi=300)
    rocm_unsupported = (
        "Disabled on AMD ROCm GPUs: it shares LightOnOCR-3's architecture, whose local runs "
        "crashed Windows on a Radeon 8060S, with even larger pages. Run it on an NVIDIA GPU or "
        "via a vLLM endpoint (INFINITY_PARSER2_FLASH_ENDPOINT_URL)."
    )

    def build_local(self):
        if self.uses_vulkan:
            return self.vulkan_backend(MAX_MODEL_LEN)
        assert self.model.hf_repo
        return LocalVisionModel(
            self.model.hf_repo,
            self.model.revision,
            device=self.settings.local_device,
            dtype=self.settings.local_dtype,
            max_batch=1,
            max_model_len=MAX_MODEL_LEN,
        )

    async def _ask(self, backend: Any, image: Image.Image, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.served_model or self.model.hf_repo,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": request_image(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        return await backend.complete(payload) or ""

    async def parse(
        self, backend: Any, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        async def parse_one(number: int, image: Image.Image):
            raw = await self._ask(backend, image, PROMPT_DOC2JSON, MAX_TOKENS)
            if not raw:
                raise PageFailed("The model returned no output for this page.")
            elements = load_elements(raw, image.width, image.height)
            if not elements:
                raise PageFailed("The model's answer was not a readable element list.")

            for item in elements:
                if str(item.get("category", "")).strip().lower() != "figure":
                    continue
                x1, y1, x2, y2 = (int(v) for v in item["bbox"])
                crop = image.crop(
                    (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
                )
                if crop.width > 0 and crop.height > 0:
                    item["text"] = await self._ask(
                        backend, crop, DEEP_PARSE_PROMPT, DEEP_PARSE_MAX_TOKENS
                    )

            markdown, items = P.normalize_page(elements)
            width, height = image.width, image.height
            layout = []
            for i, item in enumerate(items):
                bbox = item["bbox"]
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                x1, y1, x2, y2 = bbox
                normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
                layout.append(element(item["category"], normalized, item["md"], i))
            return markdown, layout

        try:
            return await parse_page_by_page(pages, report, parse_one)
        finally:
            await self.close_backend(backend)
