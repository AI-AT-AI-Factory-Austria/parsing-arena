"""Run one arena model on a few pages of a document and print the result.

Loads exactly one model in this process and exits afterwards, so nothing stays in
memory. Stop the API server first when testing on a machine with a shared GPU.

uv run python scripts/smoke_model.py samples/sample-report.pdf --model lightonocr-3-0.8b --pages 2
"""

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.documents import SUPPORTED_TYPES, parse_page_selection, render_page, sniff_extension
from app.engines.base import PageSource
from app.engines.registry import ENGINES, EngineRegistry


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document")
    parser.add_argument("--model", required=True, choices=sorted(ENGINES))
    parser.add_argument("--pages", default="1", help='page selection, e.g. "1" or "1-3"')
    parser.add_argument("--out", help="write the markdown to this file")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    source = Path(args.document)
    extension = sniff_extension(source.read_bytes()[:16])
    if extension not in SUPPORTED_TYPES:
        raise SystemExit("unsupported document type")
    if extension == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(source)
        page_count = len(pdf)
        pdf.close()
    else:
        page_count = 1
    numbers = parse_page_selection(args.pages, page_count)

    engine = EngineRegistry(get_settings()).open_engine(args.model)
    assert engine is not None
    workdir = Path(tempfile.mkdtemp(prefix="ocr-arena-smoke-"))
    paths = {}
    spec = engine.render
    for n in numbers if spec else []:
        image = render_page(source, extension, n, dpi=spec.dpi, max_pixels=spec.max_pixels)
        paths[n] = workdir / f"page-{n:04d}.png"
        image.save(paths[n])
        print(f"page {n}: {image.width}x{image.height}")

    async def report(update: dict) -> None:
        if "phase" in update or "pages_done" in update:
            print("  ", update, flush=True)

    backend, load_seconds = await engine.prepare()
    print(f"loaded in {load_seconds:.1f}s: {engine.describe(backend)}", flush=True)
    started = time.perf_counter()
    output = await engine.parse(
        backend,
        PageSource(numbers, paths, source=source, extension=extension),
        report=report,
        artifact_dir=workdir / "artifacts",
    )
    print(f"parsed {len(numbers)} page(s) in {time.perf_counter() - started:.1f}s")
    for page in output.pages:
        status = page.error or f"{len(page.elements or [])} layout elements"
        print(f"  page {page.page_number}: {page.seconds or 0:.1f}s, {status}")
    if args.out:
        Path(args.out).write_text(output.markdown, encoding="utf-8")
        print(f"markdown written to {args.out}")
    else:
        print("=" * 80)
        print(output.markdown)


if __name__ == "__main__":
    asyncio.run(main())
