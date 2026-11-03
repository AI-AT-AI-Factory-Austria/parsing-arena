"""Parsers that are not vision-language models and run on the CPU: pypdf, Tesseract
and Docling."""

import asyncio
import importlib.util
import os
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .backends.transformers_local import free_accelerator_memory
from .base import PageOutput, PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import OpenModelEngine, parse_page_by_page


class CpuEngine(OpenModelEngine):
    """Engines without an endpoint mode that never touch the GPU."""

    render = None
    supports_endpoint = False

    @property
    def backend_kind(self) -> str:
        return "cpu"

    def unavailable_reason(self) -> str | None:
        return None

    def is_loaded(self) -> bool:
        return self.pool.is_loaded(self.pool_key)

    async def close_backend(self, backend: Any) -> None:
        return None


# --- pypdf --------------------------------------------------------------------------


class PypdfEngine(CpuEngine):
    """ParseBench's ``pypdf_baseline``: ``page.extract_text()`` of every page, no OCR."""

    async def prepare(self) -> tuple[Any, float]:
        return None, 0.0

    def describe(self, backend: Any) -> dict[str, Any]:
        import pypdf

        return {"pipeline": self.model.pipeline, "backend": "pypdf", "version": pypdf.__version__}

    async def release(self) -> None:
        return None

    async def parse(
        self, backend: Any, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        if pages.extension != ".pdf" or pages.source is None:
            raise RuntimeError("pypdf reads the embedded text of PDFs; this document is an image.")
        return await asyncio.to_thread(self._extract, pages)

    @staticmethod
    def _extract(pages: PageSource) -> ParseOutput:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError

        reader = PdfReader(str(pages.source))
        outputs = []
        for number in pages.numbers:
            started = time.perf_counter()
            try:
                text = reader.pages[number - 1].extract_text()
                outputs.append(PageOutput(number, text, time.perf_counter() - started, []))
            except (PyPdfError, ValueError, KeyError, TypeError) as exc:
                outputs.append(
                    PageOutput(number, "", time.perf_counter() - started, [], error=str(exc))
                )
        return ParseOutput(markdown="\n\n".join(p.markdown for p in outputs), pages=outputs)


# --- Tesseract ----------------------------------------------------------------------

WINDOWS_TESSERACT = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
]


def find_tesseract(configured: str) -> str | None:
    """The tesseract executable: the configured one, else PATH, else the Windows default."""
    if configured:
        return configured if Path(configured).is_file() or shutil.which(configured) else None
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    return next((str(p) for p in WINDOWS_TESSERACT if p.is_file()), None)


class TesseractEngine(CpuEngine):
    """ParseBench's ``tesseract_eng``: ``image_to_string`` on 300-DPI pages, plain text."""

    render = RenderSpec(dpi=300)

    def unavailable_reason(self) -> str | None:
        if importlib.util.find_spec("pytesseract") is None:
            return "pytesseract is not installed (uv sync --extra local --inexact)"
        if find_tesseract(self.settings.tesseract_cmd) is None:
            return (
                "The Tesseract program is not installed. On Windows: "
                "winget install UB-Mannheim.TesseractOCR (or set TESSERACT_CMD)."
            )
        return None

    async def prepare(self) -> tuple[Any, float]:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = find_tesseract(self.settings.tesseract_cmd)
        return pytesseract, 0.0

    def describe(self, backend: Any) -> dict[str, Any]:
        return {
            "pipeline": self.model.pipeline,
            "backend": "tesseract",
            "version": str(backend.get_tesseract_version()),
            "lang": self.settings.tesseract_lang,
        }

    async def release(self) -> None:
        return None

    async def parse(
        self, backend: Any, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        lang = self.settings.tesseract_lang

        async def parse_one(number: int, image: Image.Image):
            text = await asyncio.to_thread(backend.image_to_string, image, lang=lang, config="")
            return text, []

        return await parse_page_by_page(pages, report, parse_one)


# --- Docling ------------------------------------------------------------------------

# Docling's DocItemLabel values -> the arena's layout categories.
DOCLING_LABELS = {
    "title": "Title",
    "section_header": "Section-header",
    "text": "Text",
    "paragraph": "Text",
    "list_item": "List-item",
    "table": "Table",
    "picture": "Picture",
    "chart": "Chart",
    "caption": "Caption",
    "formula": "Formula",
    "footnote": "Footnote",
    "page_header": "Page-header",
    "page_footer": "Page-footer",
    "code": "Code",
}
DOCLING_THREADS = 4


class DoclingRuntime:
    """A Docling DocumentConverter with its layout and table models on the CPU."""

    def __init__(self) -> None:
        self.converter: Any = None

    @property
    def loaded(self) -> bool:
        return self.converter is not None

    def load(self) -> None:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )

        options = PdfPipelineOptions()
        options.accelerator_options = AcceleratorOptions(
            num_threads=min(DOCLING_THREADS, os.cpu_count() or DOCLING_THREADS),
            device=AcceleratorDevice.CPU,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
            }
        )
        converter.initialize_pipeline(InputFormat.PDF)
        self.converter = converter

    def unload(self) -> None:
        self.converter = None
        free_accelerator_memory()

    def describe(self) -> dict[str, Any]:
        import docling

        return {
            "backend": "docling",
            "device": "cpu",
            "device_name": "CPU",
            "threads": DOCLING_THREADS,
            "version": getattr(docling, "__version__", None),
        }

    def convert(self, pages: PageSource) -> ParseOutput:
        assert pages.source is not None
        first, last = min(pages.numbers), max(pages.numbers)
        started = time.perf_counter()
        kwargs = {"page_range": (first, last)} if pages.extension == ".pdf" else {}
        document = self.converter.convert(str(pages.source), **kwargs).document
        seconds = (time.perf_counter() - started) / len(pages.numbers)

        outputs = []
        for number in pages.numbers:
            # Images are a single page 1 whatever the arena calls it.
            page_no = number if pages.extension == ".pdf" else 1
            markdown = document.export_to_markdown(page_no=page_no)
            outputs.append(PageOutput(number, markdown, seconds, self._elements(document, page_no)))
        return ParseOutput(markdown="\n\n".join(p.markdown for p in outputs), pages=outputs)

    @staticmethod
    def _elements(document: Any, page_no: int) -> list[dict[str, Any]]:
        page = document.pages.get(page_no)
        if page is None or not page.size.width or not page.size.height:
            return []
        width, height = page.size.width, page.size.height
        layout = []
        for item, _level in document.iterate_items(page_no=page_no):
            prov = next((p for p in getattr(item, "prov", []) if p.page_no == page_no), None)
            if prov is None:
                continue
            box = prov.bbox.to_top_left_origin(page_height=height)
            label = str(getattr(item.label, "value", item.label))
            layout.append(
                element(
                    DOCLING_LABELS.get(label, "Text"),
                    [box.l / width, box.t / height, box.r / width, box.b / height],
                    str(getattr(item, "text", "") or ""),
                    len(layout),
                )
            )
        return layout


class DoclingEngine(CpuEngine):
    """Docling's default PDF pipeline (layout model, TableFormer, OCR where needed)."""

    def unavailable_reason(self) -> str | None:
        import importlib.util

        if importlib.util.find_spec("docling") is None:
            return "Docling is not installed (uv sync --extra local --inexact)"
        return None

    def build_local(self) -> DoclingRuntime:
        return DoclingRuntime()

    def describe(self, backend: DoclingRuntime) -> dict[str, Any]:
        return {"pipeline": self.model.pipeline, **backend.describe()}

    async def parse(
        self, backend: DoclingRuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        await report({"phase": "parsing", "page": pages.numbers[0], "page_index": 0})
        output = await asyncio.to_thread(backend.convert, pages)
        await report({"pages_done": len(pages.numbers), "page": pages.numbers[-1]})
        return output
