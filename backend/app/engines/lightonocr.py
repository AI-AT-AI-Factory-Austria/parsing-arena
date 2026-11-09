"""LightOnOCR-3: one pass with the grounding prompt gives markdown and layout."""

from pathlib import Path
from typing import Any

from PIL import Image

from .backends.transformers_local import LocalVisionModel
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import PageFailed, parse_page_by_page, png_data_url
from .qwen35 import Qwen35Engine
from .vendor import parsebench_lightonocr3 as L

# ParseBench's lightonocr_3_vllm_parse pipeline and the vLLM flags it was served with.
TEMPERATURE = 0.1
MAX_TOKENS = 12288
MAX_MODEL_LEN = 24576


class LightOnOcr3Engine(Qwen35Engine):
    render = RenderSpec(dpi=400, max_pixels=5_000_000)
    rocm_unsupported = (
        "Disabled on AMD ROCm GPUs: local runs crashed Windows (stop error 0xD1) on a Radeon "
        "8060S. Run it on an NVIDIA GPU or via a vLLM endpoint (LIGHTONOCR_3_0_8B_ENDPOINT_URL)."
    )

    def build_local(self):
        if self.uses_vulkan:
            return self.vulkan_backend(MAX_MODEL_LEN)
        assert self.model.hf_repo
        # Qwen3.5's linear-attention layers are run one request at a time:
        # left-padded batches are not guaranteed to match unpadded results.
        return LocalVisionModel(
            self.model.hf_repo,
            self.model.revision,
            device=self.settings.local_device,
            dtype=self.settings.local_dtype,
            max_batch=1,
            max_model_len=MAX_MODEL_LEN,
        )

    async def parse(
        self, backend: Any, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        async def parse_one(number: int, image: Image.Image):
            payload = {
                "model": self.served_model or L.SERVED_MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": png_data_url(image)}},
                            {"type": "text", "text": L.DEFAULT_PROMPT},
                        ],
                    }
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            content = await backend.complete(payload)
            if not content:
                raise PageFailed("The model returned no output for this page.")
            markdown, items = L.normalize_page(content)
            elements = [
                element(item["category"], [v / 1000 for v in item["bbox"]], item["text"], i)
                for i, item in enumerate(items)
            ]
            return markdown, elements

        try:
            return await parse_page_by_page(pages, report, parse_one)
        finally:
            await self.close_backend(backend)
