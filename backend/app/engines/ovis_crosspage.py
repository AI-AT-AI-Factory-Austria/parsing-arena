"""Ovis's second stage, preserving page OCR while reconstructing document tables.

Uses the pinned upstream fragment binding, validation and renderer. Stage 2 visits
adjacent table-bearing pages backwards, re-rendering modified pages for chains.
Chart-derived data tables are deliberately excluded from continuation detection.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .vendor import ovis_document_crosspage as C
from .vendor import parsebench_ovisocr3 as O

logger = logging.getLogger(__name__)
UPSTREAM_REVISION = "17f255e2c8ae6d18d5d5edb0a474bd4ed39aef90"
# Official prompts/crosspage_table.txt at UPSTREAM_REVISION (Apache-2.0).
PROMPT = (
    "Find and reconstruct every table whose content continues across two or more consecutive "
    "input pages.\n\n"
    "Return only one JSON array. Each cross-page table is one object with exactly these fields: "
    "source_pages and html. source_pages contains the input-local 1-based pages of that table. "
    "html contains one complete HTML <table>...</table>.\n\n"
    'If no cross-page table exists, return [{"source_pages":[],"html":"No cross-page table exists '
    'between these pages."}]. Preserve figures inside cross-page tables as '
    '<img src="page_N/images/bbox_left_top_right_bottom.jpg" />; page_N is 1-based. '
    "Return no explanation or code fence."
)


def region_parts(regions: list[dict]) -> tuple[list[str], list[str]]:
    """Separate model-visible content from chart-enriched display content."""
    original, display = [], []
    for region in regions:
        display.append(O.page_markdown([region])[0])
        if region["category"] in {"chart", "image", "header_image", "footer_image"}:
            original.append(region["content"])
        else:
            original.append(display[-1])
    return original, display


class PageRenderer:
    """Lazy, offline browser renderer; all operations run on one worker thread."""

    def __init__(self, executable: str = ""):
        self.executable = executable
        self.playwright = self.browser = self.renderer = None

    def render(self, *args):
        if self.renderer is None:
            from huggingface_hub import hf_hub_download
            from playwright.sync_api import sync_playwright

            mathjax = hf_hub_download(
                "ATH-MaaS/OvisOCR3", "assets/mathjax.js", revision=UPSTREAM_REVISION
            )
            self.playwright = sync_playwright().start()
            executable = self.executable
            # A bundled browser is preferred; an installed Chrome also works on Windows.
            if not executable and not Path(self.playwright.chromium.executable_path).is_file():
                for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
                    if base := os.environ.get(variable):
                        candidate = Path(base) / "Google/Chrome/Application/chrome.exe"
                        if candidate.is_file():
                            executable = str(candidate)
                            break
            self.browser = self.playwright.chromium.launch(
                headless=True, executable_path=executable or None
            )
            self.renderer = C.Renderer(self.browser, mathjax, max_input_pixels=2880 * 2880)
        return self.renderer.render(*args)

    def close(self):
        try:
            if self.browser:
                self.browser.close()
        finally:
            if self.playwright:
                self.playwright.stop()


def export_document(output, regions_by_page, groups):
    """Replace only bound table fragments; preserve all other normalized output."""
    replacements = {}
    for group in groups:
        fragments = sorted(map(tuple, group["fragment_ids"]))
        for index, fragment in enumerate(fragments):
            replacements[fragment] = group["html"] if index == 0 else ""
    documents = []
    for local, page in enumerate(output.pages, 1):
        regions = regions_by_page.get(page.page_number, [])
        if page.error or not regions:
            if page.markdown:
                documents.append(page.markdown)
            continue
        original, display = region_parts(regions)
        fragment = 0
        parts = []
        for before, shown in zip(original, display, strict=True):
            # A chart's HTML data table was not present in Stage 1's image reference.
            if C.TABLE.search(before):
                shown = C.page_markdown(shown, local)

                def replace(match, owner=local):
                    nonlocal fragment
                    value = replacements.get((owner, fragment), match[0])
                    fragment += 1
                    return value

                shown = C.TABLE.sub(replace, shown)
            if shown:
                parts.append(shown)
        documents.append("\n\n".join(parts))
    return "\n\n".join(documents)


async def reconstruct(
    engine, backend, source, output, regions_by_page, complete_pages, report, root
):
    started = time.perf_counter()
    root = root / "artifacts" / "crosspage"
    root.mkdir(parents=True, exist_ok=True)
    images = [str(source.paths[p.page_number]) for p in output.pages]
    state = C.DocumentState()
    for local, page in enumerate(output.pages, 1):
        original, _ = region_parts(regions_by_page.get(page.page_number, []))
        markdown = "\n\n".join(original).strip()
        if markdown:
            markdown = C.page_markdown(markdown, local)
        state.pages.append(C.PageState(local, images[local - 1], markdown, markdown, "stop"))

    pairs = [
        (left, left + 1)
        for left in range(len(state.pages) - 1, 0, -1)
        if output.pages[left].page_number == output.pages[left - 1].page_number + 1
        and all(
            not output.pages[index - 1].error
            and output.pages[index - 1].page_number in complete_pages
            and C.TABLE.search(state.pages[index - 1].markdown)
            for index in (left, left + 1)
        )
    ]
    issues = []
    checked = 0
    renderer = PageRenderer(engine.settings.ovisocr3_browser_executable)
    # Playwright's synchronous browser must stay on its creating thread (also on Windows).
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ovis-render")
    loop = asyncio.get_running_loop()
    try:
        for index, (left, right) in enumerate(pairs):
            page_numbers = [output.pages[n - 1].page_number for n in (left, right)]
            await report(
                {
                    "phase": "reconstructing",
                    "pair_pages": page_numbers,
                    "pairs_done": index,
                    "pairs_total": len(pairs),
                }
            )
            folder = root / f"pair_{page_numbers[0]:04d}_{page_numbers[1]:04d}"
            folder.mkdir(exist_ok=True)
            try:
                pair = await loop.run_in_executor(
                    executor,
                    C._prepare_pair,
                    state.pages[left - 1 : right],
                    images,
                    folder,
                    renderer,
                )
                if not all(C.TABLE.search(md) for md in pair.markdowns):
                    continue
                context = "\n\n".join(
                    f'<INPUT_PAGE index="{n}">\n{md}\n</INPUT_PAGE>'
                    for n, md in enumerate(pair.markdowns, 1)
                )
                prompt = PROMPT + "\n\nInput Markdown:\n\n" + context
                C.save(folder / "input.json", {"source_pages": page_numbers, "prompt": prompt})
                from PIL import Image

                page_images = []
                for path in pair.image_paths:
                    with Image.open(path) as image:
                        page_images.append(image.convert("RGB"))
                raw = await engine._ask(backend, page_images, prompt, require_complete=True)
                (folder / "raw.txt").write_text(raw, encoding="utf-8")
                tables = C.parse_tables(raw)
                if len(tables) > 1:
                    raise ValueError(
                        "Multiple continuations at one boundary need explicit bindings."
                    )
                tables = [
                    dict(table, html=C._restore_image_refs(table["html"], pair)) for table in tables
                ]
                C._apply_merge(state, pair, tables)
                checked += 1
                C.save(
                    folder / "result.json",
                    {
                        "tables": tables,
                        "rendered_pages": [
                            output.pages[n - 1].page_number for n in sorted(pair.rendered_pages)
                        ],
                    },
                )
            except Exception as exc:  # Preserve valid page OCR and other successful boundaries.
                logger.warning("Ovis table reconstruction %s: %s", page_numbers, exc, exc_info=True)
                issue = {"source_pages": page_numbers, "message": str(exc)}
                issues.append(issue)
                C.save(folder / "error.json", issue)
    finally:
        try:
            await loop.run_in_executor(executor, renderer.close)
        finally:
            executor.shutdown(wait=False)

    prefix = "artifacts/crosspage/"
    tables = []
    for group in sorted(state.merged_tables, key=lambda item: item["fragment_ids"]):
        tables.append(
            {
                "source_pages": [output.pages[n - 1].page_number for n in group["source_pages"]],
                "html": C.IMAGE_REF.sub(lambda match: prefix + match[0], group["html"]),
            }
        )
    if tables:
        document = export_document(output, regions_by_page, state.merged_tables)
        crop_errors = C._save_document_crops(document, images, root)
        issues.extend(
            {"source_pages": source.numbers, "message": error["message"]} for error in crop_errors
        )
        output.markdown = C.IMAGE_REF.sub(lambda match: prefix + match[0], document)
    output.meta.update(
        {
            "cross_page_tables": tables,
            "cross_page_reconstruction": {
                "status": "partial" if issues else "completed" if pairs else "not_needed",
                "pairs_total": len(pairs),
                "pairs_checked": checked,
                "issues": issues,
                "seconds": round(time.perf_counter() - started, 3),
                "upstream_revision": UPSTREAM_REVISION,
            },
        }
    )
    C.save(root / "result.json", output.meta)
    (root / "document.md").write_text(output.markdown, encoding="utf-8")
    return output
