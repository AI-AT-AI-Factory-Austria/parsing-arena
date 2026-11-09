"""Falcon-OCR: PP-DocLayoutV3 layout detection, then OCR per region.

ParseBench's ``falconocr_pipeline`` sends 200-DPI pages to a server that runs the
model's own ``generate_with_layout`` (``max_new_tokens=4096``, greedy) and returns the
regions plus their text joined in reading order. The provider converts pipe tables
to HTML and quotes HTML attributes.
"""

import asyncio
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .backends.transformers_local import (
    describe_local_device,
    free_accelerator_memory,
    resolve_device,
    resolve_dtype,
)
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import OpenModelEngine, PageFailed, parse_page_by_page
from .vendor import parsebench_lightonocr3 as tables

MAX_NEW_TOKENS = 4096

# PP-DocLayoutV3 labels -> arena categories (ParseBench's canonical mapping).
LABELS = {
    "doc_title": "Title",
    "paragraph_title": "Section-header",
    "text": "Text",
    "vertical_text": "Text",
    "number": "Text",
    "abstract": "Text",
    "content": "Text",
    "reference": "Text",
    "aside_text": "Text",
    "reference_content": "Text",
    "formula_number": "Text",
    "header": "Page-header",
    "header_image": "Page-header",
    "footer": "Page-footer",
    "footer_image": "Page-footer",
    "footnote": "Footnote",
    "vision_footnote": "Footnote",
    "image": "Picture",
    "chart": "Chart",
    "seal": "Picture",
    "figure_title": "Caption",
    "table": "Table",
    "formula": "Formula",
    "display_formula": "Formula",
    "inline_formula": "Formula",
    "algorithm": "Code",
}
PICTURES = {"Picture", "Chart"}


def triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


class FalconRuntime:
    def __init__(self, repo: str, revision: str | None, *, device: str, dtype: str):
        self.repo = repo
        self.revision = revision
        self._device_pref = device
        self._dtype_pref = dtype
        self.device: str | None = None
        self.dtype_name: str | None = None
        self.model: Any = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM

        path = snapshot_download(self.repo, revision=self.revision)
        device = resolve_device(self._device_pref)
        dtype = resolve_dtype(self._dtype_pref, device)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=dtype)
        model.to(device)
        model.eval()
        self.model = model
        self.device = device
        self.dtype_name = str(dtype).removeprefix("torch.")

    def unload(self) -> None:
        self.model = None
        free_accelerator_memory()

    def describe(self) -> dict[str, Any]:
        info = dict(describe_local_device(self._device_pref))
        info.update({"backend": "transformers", "device": self.device, "dtype": self.dtype_name})
        return info

    def regions(self, image: Image.Image) -> list[dict[str, Any]]:
        return self.model.generate_with_layout(
            image, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0
        )[0]


class FalconOcrEngine(OpenModelEngine):
    render = RenderSpec(dpi=200)
    supports_endpoint = False

    def unavailable_reason(self) -> str | None:
        reason = super().unavailable_reason()
        if reason:
            return reason
        device = describe_local_device(self.settings.local_device)
        if sys.platform != "linux" or device.get("runtime") != "CUDA":
            return (
                "Requires Linux with an NVIDIA GPU for Triton/FlexAttention. "
                "Deploy with RUNTIME_PROFILE=linux-nvidia and scripts/install-linux-nvidia.sh."
            )
        if not triton_available():
            return "Triton is missing. Run scripts/install-linux-nvidia.sh on the NVIDIA host."
        if not (shutil.which("gcc") or shutil.which("clang")):
            return "Falcon-OCR needs a C compiler for Triton. Install build-essential on Linux."
        return None

    def build_local(self) -> FalconRuntime:
        assert self.model.hf_repo
        return FalconRuntime(
            self.model.hf_repo,
            self.model.revision,
            device=self.settings.local_device,
            dtype=self.settings.local_dtype,
        )

    async def parse(
        self, backend: FalconRuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        async def parse_one(number: int, image: Image.Image):
            try:
                regions = await asyncio.to_thread(backend.regions, image)
            except Exception as exc:
                raise PageFailed(f"{type(exc).__name__}: {exc}") from exc
            width, height = image.size
            texts, layout = [], []
            for region in regions:
                category = LABELS.get(str(region.get("category", "")).strip().lower())
                bbox = region.get("bbox")
                if category is None or not isinstance(bbox, list | tuple) or len(bbox) != 4:
                    continue
                text = str(region.get("text") or "")
                if text and category not in PICTURES:
                    texts.append(text)
                x1, y1, x2, y2 = (float(v) for v in bbox)
                normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
                layout.append(element(category, normalized, text, len(layout)))
            markdown = "\n\n".join(texts)
            if markdown:
                markdown = tables._convert_md_tables_to_html(markdown)
                markdown = tables._sanitize_html_attributes(markdown)
            return markdown, layout

        return await parse_page_by_page(pages, report, parse_one)
