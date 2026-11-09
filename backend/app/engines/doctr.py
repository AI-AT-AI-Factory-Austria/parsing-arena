"""docTR detector/recognizer presets; plain text and normalized word boxes."""

import asyncio
import importlib.util
import sys
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .backends.transformers_local import free_accelerator_memory, resolve_device
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import OpenModelEngine, parse_page_by_page
from .presets import doctr_options


def use_native_rocm_layers(predictor) -> dict[str, int]:
    """Keep BN/LSTM on GPU while avoiding broken Windows MIOpen JIT kernels.

    The parameters and operations are unchanged; only the PyTorch dispatch backend
    for these layers changes. Convolutions still use MIOpen. The arena executes
    local models serially, so the scoped backend flag cannot affect another run.
    See https://github.com/ROCm/TheRock/issues/7839 (missing type_traits).
    """
    import torch
    from torch import nn

    counts = {"batchnorm": 0, "lstm": 0}

    def native_forward(original):
        @wraps(original)
        def forward(*args, **kwargs):
            with torch.backends.cudnn.flags(enabled=False):
                return original(*args, **kwargs)

        return forward

    for layer in predictor.modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LSTM)):
            counts["lstm" if isinstance(layer, nn.LSTM) else "batchnorm"] += 1
            layer.forward = native_forward(layer.forward)
    return counts


class DoctrRuntime:
    def __init__(self, preset: str, device: str = "auto"):
        self.preset = preset
        self.device_preference = device
        self.device: str | None = None
        self.device_name: str | None = None
        self.runtime: str | None = None
        self.native_rocm_layers: dict[str, int] = {}
        self.predictor: Any = None

    @property
    def loaded(self):
        return self.predictor is not None

    def load(self):
        import torch
        from doctr.models import ocr_predictor

        torch.set_num_threads(4)
        device = resolve_device(self.device_preference)
        # PyTorch exposes AMD ROCm devices through its CUDA device API too.
        # Preserve docTR's pretrained float32 precision and preset options.
        self.predictor = ocr_predictor(**doctr_options(self.preset)).to(device).eval()
        if sys.platform == "win32" and device.startswith("cuda") and torch.version.hip:
            self.native_rocm_layers = use_native_rocm_layers(self.predictor)
        self.device = device
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            self.device_name = torch.cuda.get_device_name(device)
            self.runtime = "ROCm" if torch.version.hip else "CUDA"
        elif device == "mps":
            self.device_name, self.runtime = "Apple GPU", "MPS"
        else:
            self.device_name, self.runtime = "CPU", "CPU"

    def unload(self):
        self.predictor = None
        self.device = self.device_name = self.runtime = None
        self.native_rocm_layers = {}
        free_accelerator_memory()

    def describe(self):
        return {
            "backend": "python-doctr",
            "version": version("python-doctr"),
            "device": self.device,
            "device_name": self.device_name,
            "runtime": self.runtime,
            "dtype": "float32",
            "native_rocm_layers": self.native_rocm_layers,
            "threads": 4,
            "preset": self.preset,
            "render_dpi": 200,
            "output_format": "plain text + word boxes",
            **doctr_options(self.preset),
        }

    def extract(self, image):
        import numpy as np
        import torch

        with torch.inference_mode():
            page = self.predictor([np.asarray(image.convert("RGB"))]).pages[0]
        boxes = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geometry = np.asarray(word.geometry)
                    lo, hi = geometry.min(axis=0), geometry.max(axis=0)
                    boxes.append(element("Text", [*lo, *hi], word.value, len(boxes)))
        return page.render(), boxes


class DoctrEngine(OpenModelEngine):
    render = RenderSpec(dpi=200)
    supports_endpoint = False

    @property
    def pool_key(self):
        return type(self).__name__, self.model.id, self.settings.local_device

    def unavailable_reason(self):
        if importlib.util.find_spec("doctr") is None:
            return "docTR is not installed (uv sync --extra local --inexact)."
        return super().unavailable_reason()

    def build_local(self):
        return DoctrRuntime(self.model.id, device=self.settings.local_device)

    async def parse(
        self, backend: DoctrRuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        async def parse_one(number, image):
            return await asyncio.to_thread(backend.extract, image)

        return await parse_page_by_page(pages, report, parse_one)
