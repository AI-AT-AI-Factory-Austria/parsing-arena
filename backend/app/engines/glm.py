"""GLM-OCR 0.9B, full-page recognition through the official Transformers model."""

import asyncio
from importlib.metadata import version
from pathlib import Path

from .backends.transformers_local import (
    cpu_model_options,
    describe_local_device,
    free_accelerator_memory,
    patch_embed_convs_as_matmul,
    resolve_device,
    resolve_dtype,
)
from .base import PageSource, ParseOutput, RenderSpec, Report
from .open_engine import OpenModelEngine, PageFailed, parse_page_by_page
from .vendor import parsebench_lightonocr3 as tables

PROMPT = "Text Recognition:"
MAX_NEW_TOKENS = 8192


def patch_merger_as_matmul(model):
    """The merger convolves one 2x2 patch per item: this is exactly a linear map.

    MIOpen stalls while choosing a convolution kernel for thousands of tiny patches.
    Keep the original parameters and precision, performing that operation as GEMM.
    """
    from torch import nn

    class PatchMerger(nn.Module):
        def __init__(self, conv):
            super().__init__()
            self.weight, self.bias = conv.weight, conv.bias
            self.kernel_size, self.stride = conv.kernel_size, conv.stride

        def forward(self, x):
            if tuple(x.shape[2:]) != self.kernel_size:
                return nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride)
            y = nn.functional.linear(x.flatten(1), self.weight.flatten(1), self.bias)
            return y[:, :, None, None]

    conv = model.model.visual.downsample
    if (
        not isinstance(conv, nn.Conv2d)
        or conv.kernel_size != conv.stride
        or conv.padding != (0, 0)
        or conv.dilation != (1, 1)
        or conv.groups != 1
    ):
        raise RuntimeError("GLM-OCR vision merger has an unsupported configuration.")
    model.model.visual.downsample = PatchMerger(conv)


def decode_generation(processor, tokens, eos_ids):
    """Do not present an output stopped by the token budget as a complete page."""
    stop = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
    if stop is None:
        raise PageFailed("GLM-OCR reached the output token limit; the page is incomplete.")
    return processor.decode(
        tokens[:stop], skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()


class GlmOcrRuntime:
    def __init__(self, repo, revision, device="auto", dtype="auto"):
        self.repo, self.revision = repo, revision
        self.device_preference, self.dtype_preference = device, dtype
        self.device = self.dtype = None
        self.attention = "sdpa"
        self.model = self.processor = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        import torch
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration

        torch.set_num_threads(4)
        self.device = resolve_device(self.device_preference)
        self.dtype = resolve_dtype(self.dtype_preference, self.device)
        self.attention = cpu_model_options(self.device, self.dtype).get(
            "attn_implementation", "sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(self.repo, revision=self.revision)
        self.model = GlmOcrForConditionalGeneration.from_pretrained(
            self.repo, revision=self.revision, dtype=self.dtype, attn_implementation=self.attention
        )
        patch_embed_convs_as_matmul(self.model)
        patch_merger_as_matmul(self.model)
        self.model.to(self.device).eval()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def unload(self):
        self.model = self.processor = None
        self.device = self.dtype = None
        free_accelerator_memory()

    def describe(self):
        return {
            **describe_local_device(self.device_preference),
            "backend": "transformers",
            "transformers_version": version("transformers"),
            "device": self.device,
            "dtype": str(self.dtype).removeprefix("torch."),
            "render_dpi": 200,
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "attention": self.attention,
            "vision_patch_and_merger": "equivalent linear projections",
            "layout_model": None,
            "output_format": "full-page text / Markdown",
            "table_normalization": "pipe tables to HTML, retaining ragged-row cells",
        }

    def extract(self, image):
        import torch

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}],
            }
        ]
        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            prompt_length = inputs["input_ids"].shape[-1]
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            tokens = output[0, prompt_length:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        eos_ids.add(self.processor.tokenizer.eos_token_id)
        return decode_generation(self.processor, tokens, eos_ids)


class GlmOcrEngine(OpenModelEngine):
    render = RenderSpec(dpi=200)
    supports_endpoint = False

    @property
    def pool_key(self):
        return (*super().pool_key, self.settings.local_device, self.settings.local_dtype)

    def build_local(self):
        return GlmOcrRuntime(
            self.model.hf_repo,
            self.model.revision,
            self.settings.local_device,
            self.settings.local_dtype,
        )

    async def parse(
        self, backend: GlmOcrRuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        async def parse_one(number, image):
            content = await asyncio.to_thread(backend.extract, image)
            (artifact_dir / f"glm-page-{number}.md").write_text(content, encoding="utf8")
            # GFM discards cells beyond the header width. Preserve every extracted value
            # in HTML tables, using the same converter as the existing Falcon pipeline.
            # Raw generated Markdown remains available in the page artifact above.
            return tables._convert_md_tables_to_html(content), []

        return await parse_page_by_page(pages, report, parse_one)
