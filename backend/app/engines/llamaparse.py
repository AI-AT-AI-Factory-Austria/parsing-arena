"""LlamaParse Agentic Plus through the official llama-cloud SDK.

Request settings match ParseBench's `llamaparse_agentic_plus` pipeline:
tier=agentic_plus, version=latest, disable_cache=True.
The v2 response exposes running headers and footers separately; retain them in
the page text used for display and reference judging.
"""

import asyncio
import io
import logging
import time
from pathlib import Path

from PIL import Image, ImageSequence

from ..config import Settings
from ..documents import to_page_ranges
from .base import EngineUnavailable, PageOutput, ParseOutput, Report

logger = logging.getLogger(__name__)

TERMINAL_FAILURES = {"FAILED", "CANCELLED"}


class LlamaParseError(RuntimeError):
    pass


def page_markdown(markdown: str, header: str | None, footer: str | None) -> str:
    """Assemble a full page without repeating a region already at its body edge.

    Only whitespace is normalized for matching; the returned Markdown is kept
    intact. A header mentioned inside the body is still a separate page region.
    """
    body = markdown.strip()
    body_text = " ".join(body.split())
    parts = []
    for region, at_start in ((header, True), (footer, False)):
        text = (region or "").strip()
        if not text:
            continue
        region_text = " ".join(text.split())
        already_present = body_text == region_text or (
            body_text.startswith(region_text + " ")
            if at_start
            else body_text.endswith(" " + region_text)
        )
        if not already_present:
            parts.append((0 if at_start else 2, text))
    if body:
        parts.append((1, body))
    return "\n\n".join(text for _, text in sorted(parts))


def _image_as_pdf(path: Path) -> bytes:
    """ParseBench sends images to LlamaParse as a PDF; do the same."""
    with Image.open(path) as image:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]
    buffer = io.BytesIO()
    frames[0].save(buffer, "PDF", resolution=100.0, save_all=True, append_images=frames[1:])
    return buffer.getvalue()


def _friendly_error(exc: Exception, base_url: str) -> str:
    import llama_cloud

    if isinstance(exc, getattr(llama_cloud, "AuthenticationError", ())):
        return (
            "LlamaParse rejected the API key. Keys are region-specific: EU keys need "
            "LLAMA_CLOUD_BASE_URL=https://api.cloud.eu.llamaindex.ai "
            f"(currently {base_url or 'the default US endpoint'})."
        )
    if isinstance(exc, getattr(llama_cloud, "PermissionDeniedError", ())):
        return "LlamaParse denied the request (check the key's project and plan)."
    if isinstance(exc, getattr(llama_cloud, "RateLimitError", ())):
        return "LlamaParse rate limit reached. Try again in a minute."
    if isinstance(exc, getattr(llama_cloud, "APIConnectionError", ())):
        return f"Could not reach LlamaParse at {base_url or 'the default endpoint'}."
    return f"LlamaParse request failed: {exc}"


class LlamaParseEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def unavailable_reason(self) -> str | None:
        if not self.settings.llamaparse_configured:
            return "Add your LLAMA_CLOUD_API_KEY to the .env file to enable LlamaParse."
        return None

    def describe(self) -> dict:
        return {
            "backend": "api",
            "tier": self.settings.llamaparse_tier,
            "base_url": self.settings.llama_cloud_base_url,
            "page_assembly": "header_body_footer_v1",
        }

    async def parse(
        self,
        source: Path,
        *,
        filename: str,
        media_type: str,
        page_numbers: list[int],
        page_count: int,
        report: Report,
    ) -> ParseOutput:
        reason = self.unavailable_reason()
        if reason:
            raise EngineUnavailable(reason)
        from llama_cloud import AsyncLlamaCloud

        settings = self.settings
        base_url = settings.llama_cloud_base_url.strip()
        client = AsyncLlamaCloud(
            api_key=settings.llama_cloud_api_key.strip(), base_url=base_url or None
        )
        if media_type == "application/pdf":
            upload = (filename, source.read_bytes(), media_type)
        else:
            upload = (Path(filename).stem + ".pdf", _image_as_pdf(source), "application/pdf")

        create_kwargs: dict = {
            "tier": settings.llamaparse_tier,
            "version": "latest",
            "disable_cache": True,
            "upload_file": upload,
        }
        if page_numbers != list(range(1, page_count + 1)):
            create_kwargs["page_ranges"] = {"target_pages": to_page_ranges(page_numbers)}

        job_id: str | None = None
        try:
            await report({"phase": "uploading"})
            job = await client.parsing.create(**create_kwargs, timeout=180)
            job_id = str(job.id)
            await report({"phase": "queued", "job_id": job_id})

            deadline = time.monotonic() + settings.llamaparse_timeout_s
            last_status = None
            while True:
                status_response = await client.parsing.get(job_id)
                status = status_response.job.status
                if status != last_status:
                    last_status = status
                    if status == "RUNNING":
                        await report({"phase": "parsing"})
                if status == "COMPLETED":
                    break
                if status in TERMINAL_FAILURES:
                    message = status_response.job.error_message or status.lower()
                    raise LlamaParseError(f"LlamaParse job {status.lower()}: {message}")
                if time.monotonic() > deadline:
                    raise LlamaParseError("Timed out waiting for LlamaParse.")
                await asyncio.sleep(settings.llamaparse_poll_interval_s)

            result = await client.parsing.get(job_id, expand=["markdown", "usage"])
        except (LlamaParseError, EngineUnavailable):
            raise
        except Exception as exc:
            raise LlamaParseError(_friendly_error(exc, base_url)) from exc
        finally:
            await client.close()

        returned = list(result.markdown.pages) if result.markdown else []
        numbers = [p.page_number for p in returned]
        # Map results back to document page numbers if the API renumbered a subset.
        if numbers and not set(numbers) <= set(page_numbers) and len(numbers) == len(page_numbers):
            remap = dict(zip(sorted(numbers), page_numbers, strict=True))
        else:
            remap = {n: n for n in numbers}
        by_number = {remap[p.page_number]: p for p in returned}

        pages: list[PageOutput] = []
        page_regions: list[dict] = []
        for number in page_numbers:
            page = by_number.get(number)
            if page is None:
                pages.append(PageOutput(number, "", error="No result for this page."))
            elif page.success:
                pages.append(
                    PageOutput(number, page_markdown(page.markdown, page.header, page.footer))
                )
                page_regions.append(
                    {"page_number": number, "header": page.header, "footer": page.footer}
                )
            else:
                pages.append(PageOutput(number, "", error=getattr(page, "error", "failed")))

        credits = None
        usage = getattr(result.job, "usage", None)
        if usage is not None and usage.credits is not None:
            credits = float(usage.credits)
        markdown = "\n\n---\n\n".join(p.markdown for p in pages if p.markdown)
        return ParseOutput(
            markdown=markdown,
            pages=pages,
            meta={
                "job_id": job_id,
                "credits": credits,
                "page_regions": page_regions,
                **self.describe(),
            },
        )
