"""MinerU 2.5 Pro: the official mineru-vl-utils two-step extraction.

ParseBench's ``mineru2605pro_vllm`` server runs ``MinerUClient.two_step_extract`` with
``image_analysis=True`` and ``enable_table_formula_eq_wrap=True`` on pages rendered at
150 DPI and returns the blocks plus their markdown (``json2md``). The provider then
closes unbalanced tables, promotes first rows to headers and quotes HTML attributes.
"""

import asyncio
from pathlib import Path
from typing import Any

from PIL import Image

from .backends.transformers_local import (
    cpu_model_options,
    describe_local_device,
    free_accelerator_memory,
    patch_embed_convs_as_matmul,
    resolve_device,
    resolve_dtype,
)
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import OpenModelEngine, PageFailed, parse_page_by_page
from .vendor import parsebench_mineru as P

CLIENT_FLAGS = {"image_analysis": True, "enable_table_formula_eq_wrap": True}


def _quiet_mineru_logging() -> None:
    """mineru-vl-utils logs every raw layout answer at DEBUG through loguru."""
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


class MinerURuntime:
    """A MinerUClient on a local transformers model or a remote vLLM server."""

    def __init__(
        self,
        repo: str,
        revision: str | None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 8,
        server_url: str = "",
        model_name: str = "",
        api_key: str = "",
        timeout_s: float = 900.0,
    ):
        self.repo = repo
        self.revision = revision
        self._device_pref = device
        self._dtype_pref = dtype
        self.batch_size = batch_size
        self.server_url = server_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.device: str | None = None
        self.dtype_name: str | None = None
        self.attention: str | None = None
        self.client: Any = None

    @property
    def loaded(self) -> bool:
        return self.client is not None

    def connect(self) -> None:
        from mineru_vl_utils import MinerUClient

        _quiet_mineru_logging()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self.client = MinerUClient(
            backend="http-client",
            server_url=self.server_url,
            model_name=self.model_name or None,
            server_headers=headers,
            http_timeout=int(self.timeout_s),
            use_tqdm=False,
            **CLIENT_FLAGS,
        )

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from mineru_vl_utils import MinerUClient
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoProcessor,
            PretrainedConfig,
            Qwen2VLForConditionalGeneration,
        )

        _quiet_mineru_logging()
        path = snapshot_download(self.repo, revision=self.revision)
        device = resolve_device(self._device_pref)
        dtype = resolve_dtype(self._dtype_pref, device)

        # mineru_vl_utils.transformers_loading, minus accelerate's device_map: keep the
        # checkpoint's own word-embedding tying under transformers 5.
        config = AutoConfig.from_pretrained(path)
        raw_config, _ = PretrainedConfig.get_config_dict(path)
        text_config = config.get_text_config(decoder=True)
        tied = raw_config.get("tie_word_embeddings")
        if tied is None:
            tied = text_config.tie_word_embeddings
        text_config.tie_word_embeddings = tied
        config.tie_word_embeddings = tied
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            path, config=config, dtype=dtype, **cpu_model_options(device, dtype)
        )
        self.attention = model.config._attn_implementation
        patch_embed_convs_as_matmul(model)
        model.to(device)
        model.eval()

        processor = AutoProcessor.from_pretrained(path)
        processor.image_processor = AutoImageProcessor.from_pretrained(path, backend="torchvision")
        # Batched generation needs left padding.
        processor.tokenizer.padding_side = "left"

        self.client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
            batch_size=self.batch_size,
            use_tqdm=False,
            **CLIENT_FLAGS,
        )
        self.device = device
        self.dtype_name = str(dtype).removeprefix("torch.")

    def unload(self) -> None:
        self.client = None
        free_accelerator_memory()

    def describe(self) -> dict[str, Any]:
        info = dict(describe_local_device(self._device_pref))
        info.update(
            {
                "backend": "transformers",
                "device": self.device or info.get("device"),
                "dtype": self.dtype_name,
                "attention": self.attention,
                "max_batch": self.batch_size,
            }
        )
        return info

    def extract(self, image: Image.Image) -> list[dict[str, Any]]:
        return [dict(block) for block in self.client.two_step_extract(image)]


class MinerUEngine(OpenModelEngine):
    render = RenderSpec(dpi=150)

    def build_local(self) -> MinerURuntime:
        assert self.model.hf_repo
        return MinerURuntime(
            self.model.hf_repo,
            self.model.revision,
            device=self.settings.local_device,
            dtype=self.settings.local_dtype,
            batch_size=self.settings.local_max_batch,
        )

    def endpoint_backend(self) -> MinerURuntime:
        assert self.model.hf_repo
        runtime = MinerURuntime(
            self.model.hf_repo,
            self.model.revision,
            server_url=self.endpoint_url,
            model_name=self.served_model,
            api_key=self.settings.vllm_api_key,
            timeout_s=self.settings.endpoint_timeout_s,
        )
        runtime.connect()
        return runtime

    async def parse(
        self, backend: MinerURuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        from mineru_vl_utils.post_process.json2markdown import json2md

        async def parse_one(number: int, image: Image.Image):
            try:
                blocks = await asyncio.to_thread(backend.extract, image)
            except Exception as exc:
                raise PageFailed(f"{type(exc).__name__}: {exc}") from exc
            markdown = P.normalize_markdown(json2md(blocks))
            layout = [
                element(
                    P.LABEL_MAP.get(str(block.get("type") or "text").lower(), "Text"),
                    block["bbox"],
                    str(block.get("content") or ""),
                    i,
                )
                for i, block in enumerate(blocks)
                if isinstance(block.get("bbox"), list) and len(block["bbox"]) == 4
            ]
            return markdown, layout

        return await parse_page_by_page(pages, report, parse_one)
