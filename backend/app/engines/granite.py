"""Granite Docling's local VLM pipeline, independently selectable from Docling Standard."""

import asyncio
import importlib.util
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ..catalog import ArenaModel
from ..config import Settings
from .backends.transformers_local import resolve_device, resolve_dtype
from .base import PageOutput, PageSource, ParseOutput, Report
from .open_engine import OpenModelEngine, PageFailed
from .tools import DOCLING_THREADS, DoclingRuntime


def pipeline_options(model: ArenaModel, device: str, dtype: str):
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions

    vlm = VlmConvertOptions.from_preset(
        "granite_docling",
        engine_options=TransformersVlmEngineOptions(
            device=device,
            torch_dtype=dtype,
            quantized=False,
        ),
    ).model_copy(deep=True)
    vlm.model_spec.default_repo_id = model.hf_repo
    vlm.model_spec.revision = model.revision
    vlm.model_spec.temperature = 0.0
    vlm.model_spec.max_new_tokens = 8192
    vlm.scale = 2.0
    vlm.batch_size = 1
    vlm.force_backend_text = False
    return VlmPipelineOptions(
        vlm_options=vlm,
        force_backend_text=False,
        accelerator_options=AcceleratorOptions(
            device=device,
            num_threads=DOCLING_THREADS,
            cuda_use_flash_attention2=False,
        ),
    )


class GraniteDoclingRuntime(DoclingRuntime):
    def __init__(self, model: ArenaModel, settings: Settings) -> None:
        super().__init__()
        self.model = model
        self.settings = settings
        self.device = "cpu"
        self.dtype = "float32"

    def load(self) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        self.device = resolve_device(self.settings.local_device)
        # Docling's Transformers VLM engine supports CPU/CUDA, not MPS.
        if self.device == "mps":
            self.device = "cpu"
        self.dtype = str(resolve_dtype(self.settings.local_dtype, self.device)).removeprefix(
            "torch."
        )
        options = pipeline_options(self.model, self.device, self.dtype)
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=options,
                ),
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=options,
                ),
            },
        )
        converter.initialize_pipeline(InputFormat.PDF)
        self.converter = converter

    def describe(self) -> dict[str, Any]:
        import torch

        gpu = self.device.startswith("cuda")
        return {
            "backend": "transformers",
            "engine": "docling-vlm",
            "device": self.device,
            "device_name": torch.cuda.get_device_name(self.device) if gpu else "CPU",
            "runtime": ("ROCm" if torch.version.hip else "CUDA") if gpu else "CPU",
            "dtype": self.dtype,
            "docling_version": version("docling"),
            "render_dpi": 144,
            "max_new_tokens": 8192,
            "temperature": 0.0,
            "output_format": "DocTags converted to Markdown and layout boxes",
        }

    def convert_page(self, pages: PageSource, number: int) -> tuple[str, list[dict[str, Any]]]:
        from docling.datamodel.base_models import ConversionStatus

        assert self.converter is not None and pages.source is not None
        kwargs = {"page_range": (number, number)} if pages.extension == ".pdf" else {}
        result = self.converter.convert(str(pages.source), raises_on_error=False, **kwargs)
        if result.status != ConversionStatus.SUCCESS:
            errors = "; ".join(error.error_message for error in result.errors)
            raise PageFailed(errors or f"Docling conversion ended with status {result.status}.")
        page_no = number if pages.extension == ".pdf" else 1
        document = result.document
        return document.export_to_markdown(page_no=page_no), self._elements(document, page_no)


class GraniteDoclingEngine(OpenModelEngine):
    render = None  # Docling renders the original PDF/image at the preset's scale.
    supports_endpoint = False

    def unavailable_reason(self) -> str | None:
        reason = super().unavailable_reason()
        if reason:
            return reason
        if importlib.util.find_spec("docling") is None:
            return "Docling is not installed (uv sync --extra local --inexact)"
        return None

    def build_local(self) -> GraniteDoclingRuntime:
        return GraniteDoclingRuntime(self.model, self.settings)

    async def parse(
        self,
        backend: GraniteDoclingRuntime,
        pages: PageSource,
        *,
        report: Report,
        artifact_dir: Path,
    ) -> ParseOutput:
        outputs: list[PageOutput] = []
        for index, number in enumerate(pages.numbers):
            await report({"phase": "parsing", "page": number, "page_index": index})
            started = time.perf_counter()
            try:
                markdown, elements = await asyncio.to_thread(backend.convert_page, pages, number)
                page = PageOutput(number, markdown, time.perf_counter() - started, elements)
            except PageFailed as exc:
                page = PageOutput(number, "", time.perf_counter() - started, [], error=str(exc))
            outputs.append(page)
            await report(
                {
                    "pages_done": index + 1,
                    "page": number,
                    "last_page_seconds": round(page.seconds or 0.0, 3),
                }
            )
        if all(page.error for page in outputs):
            raise RuntimeError(outputs[0].error or "No page could be parsed.")
        return ParseOutput(
            markdown="\n\n".join(p.markdown for p in outputs if p.markdown), pages=outputs
        )
