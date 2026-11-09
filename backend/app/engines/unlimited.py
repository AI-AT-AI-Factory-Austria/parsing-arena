"""Unlimited-OCR 2: all selected pages in one local, routed-attention request.

The community runtime preserves DASH routing and page RoPE but computes masked
dense attention. It does NOT implement the paper's optimized sparse-memory path.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from ..config import Settings
from ..runtime import profile_error, runtime_profile
from .backends.llama_cpp import GgufSpec, LlamaCppRuntime
from .base import PageOutput, PageSource, ParseOutput, RenderSpec, Report, element
from .open_engine import OpenModelEngine, png_data_url

SPEC = GgufSpec(
    "RX5950XT/Unlimited-OCR-2-GGUF",
    "1e3e5f3e82435578cd1e3c191673e1444b617f52",
    "Unlimited-OCR-2-BF16.gguf",
    "mmproj-Unlimited-OCR-2-F16.gguf",
)
LLAMA_VERSION = "b11913"
BLOCK = re.compile(
    r"<BLOCK>(?P<label>[^\[<>]+)(?:<LEVEL(?P<level>[1-6])>)?"
    r"\[\s*(?P<x1>\d+)\s*,\s*(?P<y1>\d+)\s*,\s*(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\s*\]<FORK>"
)
EOS = ("<\uff5cend\u2581of\u2581sentence\uff5c>", "</s>")


def installed_runtime(settings: Settings) -> tuple[Path, dict] | None:
    root = settings.data_dir / "runtimes/unlimited-ocr2"
    try:
        info = json.loads((root / "build-info.json").read_text(encoding="utf-8"))
        binary = (root / info["binary"]).resolve()
        if (
            info.get("patch_revision") != SPEC.revision
            or info.get("llama_version") != LLAMA_VERSION
            or not binary.is_relative_to(root.resolve())
            or not binary.is_file()
        ):
            return None
        return binary, info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_document(text: str, numbers: list[int]) -> list[PageOutput]:
    """Validate page boundaries before mapping output onto original page numbers."""
    text = text.rstrip()
    for marker in EOS:
        text = text.removesuffix(marker).rstrip()
    prefix, *parts = text.split("<PAGE>")
    if prefix.strip() or len(parts) != len(numbers):
        raise RuntimeError(
            f"Unlimited-OCR returned {len(parts)} page markers for {len(numbers)} selected pages. "
            "The output cannot be mapped reliably; try fewer pages."
        )
    pages = []
    for number, part in zip(numbers, parts, strict=True):
        blocks = list(BLOCK.finditer(part))
        unparsed_prefix = part[: blocks[0].start()] if blocks else part
        if unparsed_prefix.strip():
            raise RuntimeError(f"Invalid Unlimited-OCR block format on page {number}.")
        markdown, layout = [], []
        for index, block in enumerate(blocks):
            end = blocks[index + 1].start() if index + 1 < len(blocks) else len(part)
            content = part[block.end() : end].strip()
            if "<BLOCK>" in content or "<FORK>" in content:
                raise RuntimeError(f"Incomplete Unlimited-OCR block on page {number}.")
            category = block["label"].strip()
            bbox = [int(block[key]) / 1000 for key in ("x1", "y1", "x2", "y2")]
            if bbox[0] > bbox[2] or bbox[1] > bbox[3] or any(v > 1 for v in bbox):
                raise RuntimeError(f"Invalid Unlimited-OCR coordinates on page {number}.")
            layout.append(element(category, bbox, content, index))
            if content:
                if block["level"] and not content.startswith("#"):
                    content = "#" * int(block["level"]) + " " + content
                markdown.append(content)
        pages.append(PageOutput(number, "\n\n".join(markdown), elements=layout))
    return pages


class UnlimitedRuntime(LlamaCppRuntime):
    def __init__(self, settings: Settings):
        installed = installed_runtime(settings)
        if installed is None:
            raise RuntimeError("Install the Unlimited-OCR 2 runtime first.")
        binary, self.build = installed
        super().__init__(
            "unlimited-ocr-2",
            settings,
            settings.unlimited_ocr_2_context_length,
            spec=SPEC,
            binary=str(binary),
            device=f"{self.build['backend']}0",
            flash_attention=False,
            image_max_tokens=273,
            # Page-relative RoPE depends on the complete, single document's token stream.
            # <td>/<tr> and their closing tags are special tokens in this checkpoint.
            extra_args=("--no-context-shift", "--no-cache-prompt", "--special"),
        )

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "dtype": "bfloat16",
            "vision_dtype": "float16",
            "format": "GGUF BF16 + F16 vision",
            "llama_version": LLAMA_VERSION,
            "patch_revision": SPEC.revision,
            "processing_mode": "whole_document",
            "attention": "DASH routing, masked dense compute",
            "sparse_memory_optimized": False,
            "image_max_tokens": 273,
            "max_output_tokens": self.settings.unlimited_ocr_2_max_tokens,
            "max_document_pages": self.settings.unlimited_ocr_2_max_pages,
        }


class UnlimitedOcrEngine(OpenModelEngine):
    supports_endpoint = False
    render = RenderSpec(dpi=144)

    @property
    def runtime_label(self) -> str:
        gpu = "AMD Vulkan" if runtime_profile(self.settings) == "windows-amd" else "CUDA"
        return f"{gpu} · multi-page · BF16 GGUF"

    @property
    def max_pages(self) -> int:
        return self.settings.unlimited_ocr_2_max_pages

    @property
    def pool_key(self):
        return (
            *super().pool_key,
            SPEC,
            runtime_profile(self.settings),
            str(self.settings.data_dir.resolve()),
            self.settings.unlimited_ocr_2_context_length,
        )

    def unavailable_reason(self) -> str | None:
        if not self.settings.local_inference:
            return "Local inference is disabled."
        if error := profile_error(self.settings):
            return error
        if self.settings.local_device == "cpu":
            return "Unlimited-OCR 2 currently requires the local Vulkan or CUDA GPU runtime."
        profile = runtime_profile(self.settings)
        if profile not in {"windows-amd", "linux-nvidia"}:
            return "Unlimited-OCR 2 supports Windows/AMD Vulkan or Linux/NVIDIA CUDA."
        installed = installed_runtime(self.settings)
        if installed is None:
            return (
                "Install its dedicated runtime: "
                "uv run --no-project --script backend/scripts/install-unlimited.py"
            )
        expected = "Vulkan" if profile == "windows-amd" else "CUDA"
        if installed[1].get("backend") != expected:
            return f"The Unlimited-OCR runtime must be built for {expected} on this machine."
        return None

    def build_local(self) -> UnlimitedRuntime:
        return UnlimitedRuntime(self.settings)

    async def parse(
        self, backend: Any, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        count = len(pages.numbers)
        if not 1 <= count <= self.settings.unlimited_ocr_2_max_pages:
            raise RuntimeError(
                f"This Unlimited-OCR preset accepts 1–{self.settings.unlimited_ocr_2_max_pages} "
                "pages per document. Select fewer pages; they are processed together."
            )
        await report({"phase": "document", "pages_done": 0})

        def build_content():
            content = []
            for number in pages.numbers:
                with pages.load(number) as image:
                    content.extend(
                        [
                            {"type": "text", "text": "<PAGE>"},
                            {"type": "image_url", "image_url": {"url": png_data_url(image)}},
                        ]
                    )
            content.append({"type": "text", "text": "Multi page parsing."})
            return content

        content = await asyncio.to_thread(build_content)
        # 273 visual tokens plus the page marker and prompt; retain a margin for formatting.
        budget = min(
            self.settings.unlimited_ocr_2_max_tokens,
            self.settings.unlimited_ocr_2_context_length - 274 * count - 128,
        )
        if budget < 1024:
            raise RuntimeError("Too many pages for the configured Unlimited-OCR context length.")
        raw = await backend.complete(
            {
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": budget,
                "cache_prompt": False,
            }
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "unlimited-raw.txt").write_text(raw, encoding="utf-8")
        outputs = parse_document(raw, pages.numbers)
        await report({"phase": "document", "pages_done": count})
        return ParseOutput(
            markdown="\n\n".join(p.markdown for p in outputs),
            pages=outputs,
            meta={"processing_mode": "whole_document", "selected_page_numbers": pages.numbers},
        )
