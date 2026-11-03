"""Run one KDL-pipeline model locally on a PDF and print the markdown.

uv run python scripts/smoke_kdl.py samples/sample-report.pdf --model rakedoc-nano
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdfium2 as pdfium

from app.engines.backends.transformers_local import LocalVisionModel
from app.engines.kdl import florin_emission
from app.engines.kdl import kdl_frontier_nano as K

MODELS = {
    "rakedoc-nano": ("cloudraker/rakedoc-nano", None, True),
    "aeddix-alpine-ocr": ("aeddix-labs/aeddix-alpine-ocr", "v0.1", False),
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--model", default="rakedoc-nano", choices=sorted(MODELS))
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", help="write the markdown to this file")
    parser.add_argument("--debug", action="store_true", help="log every generation batch")
    args = parser.parse_args()
    if args.debug:
        import logging

        logging.basicConfig(format="%(asctime)s %(message)s")
        logging.getLogger("app.engines.backends.transformers_local").setLevel(logging.DEBUG)

    pdf = pdfium.PdfDocument(args.pdf)
    images = [pdf[i].render(scale=144 / 72).to_pil() for i in range(min(args.pages, len(pdf)))]
    pdf.close()

    repo, revision, florin = MODELS[args.model]
    model = LocalVisionModel(repo, revision, max_batch=args.batch)
    started = time.perf_counter()
    model.load()
    print(f"load+warmup {time.perf_counter() - started:.1f}s on {model.describe()}")

    def progress(event: dict) -> None:
        if event["event"] != "region_done":
            print("  ", event, flush=True)

    engine = K._NanoEngine(
        model,
        args.model,
        8,
        category_map=florin_emission.FLORIN_NATIVE_LAYOUT_CATEGORY_MAP if florin else None,
        build_markdown=florin_emission.build_markdown if florin else None,
        progress=progress,
    )
    started = time.perf_counter()
    result = await engine.parse_pages(images)
    print(f"parse {time.perf_counter() - started:.1f}s  page_seconds={result['page_seconds']}")
    if args.out:
        Path(args.out).write_text(result["markdown"], encoding="utf-8")
        print(f"markdown written to {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print("=" * 80)
        print(result["markdown"])


if __name__ == "__main__":
    asyncio.run(main())
