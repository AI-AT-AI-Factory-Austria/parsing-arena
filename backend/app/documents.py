"""Document ingestion and page rendering."""

import hashlib
import shutil
import threading
from pathlib import Path
from typing import BinaryIO

import pypdfium2 as pdfium
from PIL import Image, ImageSequence

from . import storage
from .models import Document, new_id

SUPPORTED_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".webp": "image/webp",
    ".tif": "image/tiff",
}
THUMBNAIL_WIDTH = 480

# pdfium is not thread-safe; every call into it goes through this lock.
_pdfium_lock = threading.Lock()


class DocumentError(ValueError):
    """The upload cannot be used (unsupported, too large, unreadable)."""


def sniff_extension(head: bytes) -> str | None:
    if head.startswith(b"%PDF"):
        return ".pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    return None


def ingest(stream: BinaryIO, filename: str, *, max_bytes: int) -> Document:
    """Store an upload, validate it and count its pages. Returns an unsaved Document."""
    document_id = new_id()
    target_dir = storage.document_dir(document_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = target_dir / "upload.part"
    digest = hashlib.sha256()
    size = 0
    try:
        with tmp_path.open("wb") as out:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise DocumentError(
                        f"File is larger than the {max_bytes // (1024 * 1024)} MB limit."
                    )
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise DocumentError("The file is empty.")
        with tmp_path.open("rb") as fh:
            extension = sniff_extension(fh.read(16))
        if extension is None:
            raise DocumentError("Unsupported file type. Upload a PDF, PNG, JPEG, WebP or TIFF.")
        final_path = storage.original_path(document_id, extension)
        tmp_path.replace(final_path)
        page_count = count_pages(final_path, extension)
        if page_count < 1:
            raise DocumentError("The document has no pages.")
        write_thumbnail(document_id, final_path, extension)
    except DocumentError:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise DocumentError(f"The file could not be read ({exc}).") from exc

    return Document(
        id=document_id,
        filename=Path(filename or f"document{extension}").name[:255],
        media_type=SUPPORTED_TYPES[extension],
        extension=extension,
        size_bytes=size,
        sha256=digest.hexdigest(),
        page_count=page_count,
    )


def count_pages(path: Path, extension: str) -> int:
    if extension == ".pdf":
        with _pdfium_lock:
            pdf = pdfium.PdfDocument(path)
            try:
                return len(pdf)
            finally:
                pdf.close()
    with Image.open(path) as image:
        return getattr(image, "n_frames", 1) if extension == ".tif" else 1


def render_page(path: Path, extension: str, page_number: int, *, dpi: int) -> Image.Image:
    """Render one 1-based page as RGB. Images are used at their native resolution."""
    if extension == ".pdf":
        with _pdfium_lock:
            pdf = pdfium.PdfDocument(path)
            try:
                page = pdf[page_number - 1]
                try:
                    return page.render(scale=dpi / 72).to_pil().convert("RGB")
                finally:
                    page.close()
            finally:
                pdf.close()
    with Image.open(path) as image:
        frames = ImageSequence.Iterator(image)
        frame = frames[page_number - 1] if extension == ".tif" else image
        return frame.convert("RGB")


def page_image(document: Document, page_number: int, *, dpi: int) -> Path:
    """Path of the rendered page PNG, rendering it on first use."""
    target = storage.page_image_path(document.id, page_number, dpi)
    if not target.exists():
        source = storage.original_path(document.id, document.extension)
        image = render_page(source, document.extension, page_number, dpi=dpi)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        image.save(tmp, format="PNG")
        tmp.replace(target)
    return target


def write_thumbnail(document_id: str, source: Path, extension: str) -> None:
    image = render_page(source, extension, 1, dpi=72 if extension == ".pdf" else 0)
    if image.width > THUMBNAIL_WIDTH:
        height = round(image.height * THUMBNAIL_WIDTH / image.width)
        image = image.resize((THUMBNAIL_WIDTH, height), Image.Resampling.LANCZOS)
    image.save(storage.thumbnail_path(document_id), format="WEBP", quality=82)


def parse_page_selection(spec: str | None, page_count: int) -> list[int]:
    """'1-3, 5' -> [1, 2, 3, 5]. Empty means every page."""
    if not spec or not spec.strip():
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s) if start_s else 1
            end = int(end_s) if end_s else page_count
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    selected = sorted(p for p in pages if 1 <= p <= page_count)
    if not selected:
        raise DocumentError(f"No valid pages in '{spec}' (document has {page_count}).")
    return selected


def to_page_ranges(pages: list[int]) -> str:
    """[1, 2, 3, 5] -> '1-3,5'."""
    ranges: list[str] = []
    start = prev = None
    for page in sorted(pages):
        if start is None:
            start = prev = page
        elif page == prev + 1:
            prev = page
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = page
    if start is not None:
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)
