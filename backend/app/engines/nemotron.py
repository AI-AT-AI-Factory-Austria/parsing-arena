"""Nemotron 3.5 Parse through the SDK that ships in its model repository.

ParseBench's ``nemotron_parse_3_5_sdk`` pipeline renders pages at 150 DPI as JPEG
(quality 92) and runs the SDK's ``triple`` pass (full page, detection, then Table and
Chart crops, regions merged by union) against a vLLM server. Locally the same SDK
runs on its own ``TransformersBackend``.
"""

import asyncio
import importlib
import importlib.util
import io
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image

from .backends.transformers_local import (
    describe_local_device,
    free_accelerator_memory,
    patch_embed_convs_as_matmul,
    resolve_device,
    resolve_dtype,
)
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import OpenModelEngine, PageFailed, parse_page_by_page

MODE = "triple"
JPEG_QUALITY = 92
SDK_FILES = ["nemotron_parse/**", "*.py", "*.json"]

_sdk_lock = threading.Lock()


def load_sdk(root: Path) -> ModuleType:
    """Import ``nemotron_parse.inference`` from a downloaded model snapshot."""
    with _sdk_lock:
        if "nemotron_parse" not in sys.modules:
            package_dir = root / "nemotron_parse"
            spec = importlib.util.spec_from_file_location(
                "nemotron_parse",
                package_dir / "__init__.py",
                submodule_search_locations=[str(package_dir)],
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules["nemotron_parse"] = module
            spec.loader.exec_module(module)
        return importlib.import_module("nemotron_parse.inference")


class NemotronRuntime:
    """The SDK's DocumentParser on a local transformers model or a vLLM endpoint."""

    def __init__(
        self,
        repo: str,
        revision: str | None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        base_url: str = "",
        model_name: str = "",
        api_key: str = "",
        timeout_s: float = 900.0,
    ):
        self.repo = repo
        self.revision = revision
        self._device_pref = device
        self._dtype_pref = dtype
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.device: str | None = None
        self.dtype_name: str | None = None
        self.parser: Any = None

    @property
    def loaded(self) -> bool:
        return self.parser is not None

    def connect(self) -> None:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(self.repo, revision=self.revision, allow_patterns=SDK_FILES))
        sdk = load_sdk(root)
        self.parser = sdk.DocumentParser(
            self.base_url,
            self.model_name or "nemotron-parse",
            api_key=self.api_key or None,
            timeout=self.timeout_s,
        )

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoProcessor

        root = Path(snapshot_download(self.repo, revision=self.revision))
        sdk = load_sdk(root)
        device = resolve_device(self._device_pref)
        dtype = resolve_dtype(self._dtype_pref, device)
        model = AutoModel.from_pretrained(root, trust_remote_code=True, dtype=dtype)
        patch_embed_convs_as_matmul(model)
        model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(root, trust_remote_code=True)
        self.parser = sdk.DocumentParser(backend=sdk.TransformersBackend(model, processor))
        self.device = device
        self.dtype_name = str(dtype).removeprefix("torch.")

    def unload(self) -> None:
        if self.parser is not None:
            self.parser.close()
        self.parser = None
        free_accelerator_memory()

    def describe(self) -> dict[str, Any]:
        info = dict(describe_local_device(self._device_pref))
        info.update(
            {
                "backend": "transformers",
                "device": self.device or info.get("device"),
                "dtype": self.dtype_name,
            }
        )
        return info

    def parse(self, image: Image.Image) -> dict[str, Any]:
        return self.parser.parse(image, mode=MODE)


def as_submitted(image: Image.Image) -> Image.Image:
    """The page as ParseBench sent it: JPEG at quality 92, decoded again."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
        return decoded.convert("RGB")


class NemotronEngine(OpenModelEngine):
    render = RenderSpec(dpi=150)
    rocm_unsupported = (
        "Requires an NVIDIA host for GPU inference; its native ROCm path crashes. "
        "Use RUNTIME_PROFILE=linux-nvidia there, or NEMOTRON_3_5_PARSE_ENDPOINT_URL."
    )

    def build_local(self) -> NemotronRuntime:
        assert self.model.hf_repo
        return NemotronRuntime(
            self.model.hf_repo,
            self.model.revision,
            device=self.settings.local_device,
            dtype=self.settings.local_dtype,
        )

    def endpoint_backend(self) -> NemotronRuntime:
        assert self.model.hf_repo
        runtime = NemotronRuntime(
            self.model.hf_repo,
            self.model.revision,
            base_url=self.endpoint_url,
            model_name=self.served_model,
            api_key=self.settings.vllm_api_key,
            timeout_s=self.settings.endpoint_timeout_s,
        )
        runtime.connect()
        return runtime

    async def close_backend(self, backend: Any) -> None:
        if self.endpoint_url and backend.parser is not None:
            backend.parser.close()

    async def parse(
        self, backend: NemotronRuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        async def parse_one(number: int, image: Image.Image):
            try:
                document = await asyncio.to_thread(backend.parse, as_submitted(image))
            except Exception as exc:
                raise PageFailed(f"{type(exc).__name__}: {exc}") from exc
            layout = [
                element(
                    str(block.get("class") or "Text"),
                    block["bbox"],
                    str(block.get("content") or block.get("text") or "").strip(),
                    i,
                )
                for i, block in enumerate(document.get("blocks") or [])
                if isinstance(block.get("bbox"), list) and len(block["bbox"]) == 4
            ]
            return document.get("markdown", ""), layout

        try:
            return await parse_page_by_page(pages, report, parse_one)
        finally:
            await self.close_backend(backend)
