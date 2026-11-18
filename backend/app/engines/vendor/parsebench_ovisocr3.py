"""OvisOCR3 prompts and output normalization.

[parsing-arena] Vendored from run-llama/ParseBench (Apache-2.0), commit 0ccc167:
``src/parse_bench/inference/providers/parse/ovisocr3.py`` (prompts, region parsing and
markdown), ``_layout_utils.LABEL_MAP``, ``mistral_ocr._convert_pipe_tables_to_html`` and
``evaluation/metrics/parse/chart_json_to_html.py``. The code is unchanged;
``page_markdown`` restates the per-page part of the provider's ``normalize``.
"""

from __future__ import annotations

import json
import math
import re
from html import escape
from typing import Any

import markdown2

SERVED_MODEL_NAME = "ovisocr3"

# Official prompts, verbatim from prompts/document_structured.txt and
# prompts/chart_parsing.txt in the model repo.
STRUCTURED_PROMPT = (
    "Analyze the document structure and extract text content. Return a single JSON array in natural "
    "reading order.\n\n"
    "Each element follows this schema:\n"
    '`{"bbox": [x_min, y_min, x_max, y_max], "category": "string", "content": "string"}`\n\n'
    "Rules:\n"
    "1. The bbox coordinates are integers scaled to [0, 1000).\n"
    "2. The category is exactly one of:\n"
    '["abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title", '
    '"figure_title", "footer", "footer_image", "footnote", "formula_number", "header", "header_image", '
    '"image", "inline_formula", "number", "paragraph_title", "reference_content", "seal", "table", '
    '"text", "vertical_text", "vision_footnote"].\n'
    "3. The content formatting depends on category:\n"
    '- For "footer_image", "header_image", "image", or "chart": write as an HTML image tag '
    '<img src="images/bbox_{x_min}_{y_min}_{x_max}_{y_max}.jpg" />.\n'
    '- For "display_formula" or "inline_formula": write in LaTeX.\n'
    '- For "table": write as a standard HTML `<table>...</table>`.\n'
    "- Otherwise: write in Markdown.\n"
    "4. Output must be strictly valid JSON. Properly escape quotes and special characters inside content."
)
CHART_PROMPT = (
    "Parse all key information in the chart and output one valid JSON object using exactly this "
    "top-level schema:\n"
    '{"title":"","source":"","x_title":"","y_title":"","values":{}}\n\n'
    "Rules:\n"
    "1. Preserve visible title, source, axis titles, series or category labels, x-axis labels, and "
    "values without explanation or inference.\n"
    '2. Use the literal string "None" when the chart has no visible value for title, source, or an '
    "axis title. For a dual-axis chart, `y_title` may be an array of two strings in visual order; "
    "otherwise it must be one string.\n"
    "3. `values` must be a non-empty object. Preserve the chart hierarchy and encode every scalar data "
    'value as a JSON string. An empty series name may be represented by the key "" when the chart has '
    "no visible series label.\n"
    "4. Output JSON only. Do not output Python syntax, Markdown fences, comments, prose, or extra keys."
)

# Official sampling parameters (SamplingParams(max_tokens=16384, temperature=0.0))
DEFAULT_MAX_TOKENS = 16384

# Region boxes are integers on this grid (model card, "Coordinates and Visual Regions").
COORD_SCALE = 1000.0

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")
_REGION_START_RE = re.compile(r'\{\s*"bbox"')
_IMG_PLACEHOLDER_RE = re.compile(r'^\s*<img src="images/bbox_[^>]*/>\s*$')
_HEADING_MARKER_RE = re.compile(r"^\s*#{1,6}\s")

# _layout_utils.LABEL_MAP
LABEL_MAP: dict[str, str] = {
    "caption": "Caption",
    "code": "Code",
    "document index": "Document Index",
    "footnote": "Footnote",
    "formula": "Formula",
    "list-item": "List-item",
    "list_item": "List-item",
    "page-footer": "Page-footer",
    "page_footer": "Page-footer",
    "page-header": "Page-header",
    "page_header": "Page-header",
    "picture": "Picture",
    "figure": "Picture",
    "section-header": "Section-header",
    "section_header": "Section-header",
    "table": "Table",
    "text": "Text",
    "title": "Title",
}

# OvisOCR3 categories -> keys of the shared _layout_utils.LABEL_MAP. table, text and
# footnote already match; ``number`` (page number) is placed by position below.
_LABEL_ALIASES: dict[str, str] = {
    "doc_title": "title",
    "paragraph_title": "section_header",
    "header": "page_header",
    "footer": "page_footer",
    "figure_title": "caption",
    "vision_footnote": "footnote",
    "image": "picture",
    "chart": "picture",
    "seal": "picture",
    # ParseBench's layout GT labels an image in the header/footer band as Picture.
    "header_image": "picture",
    "footer_image": "picture",
    "display_formula": "formula",
    "inline_formula": "formula",
    "algorithm": "code",
    # PP-DocLayout's ``content`` is a table of contents.
    "content": "document index",
    "abstract": "text",
    "aside_text": "text",
    "formula_number": "text",
    "reference_content": "text",
    "vertical_text": "text",
}


def parse_regions(content: str) -> list[dict[str, Any]]:
    """Decode the structured answer into ``{bbox, category, content}`` regions.

    A response cut off by ``max_tokens`` is not valid JSON; every region completed
    before the cut is still recovered, in emitted order. A region whose own JSON is
    malformed is skipped without losing the rest of the page.
    """
    text = _FENCE_RE.sub("", content.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, list):
        decoder = json.JSONDecoder()
        data = []
        for match in _REGION_START_RE.finditer(text):
            try:
                region, _end = decoder.raw_decode(text, match.start())
            except json.JSONDecodeError:
                continue
            data.append(region)

    regions: list[dict[str, Any]] = []
    for region in data:
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        regions.append(
            {
                "bbox": [x1, y1, x2, y2],
                "category": str(region.get("category") or "text").strip().lower(),
                "content": str(region.get("content") or ""),
            }
        )
    return regions


def _layout_label(category: str, bbox: list[float]) -> str:
    """Canonical17 label for a region category."""
    if category == "number":
        # A printed page number is page furniture on whichever edge it sits.
        return LABEL_MAP["page_header"] if (bbox[1] + bbox[3]) / 2 < COORD_SCALE / 2 else LABEL_MAP["page_footer"]
    return LABEL_MAP.get(_LABEL_ALIASES.get(category, category), "Text")


def _region_markdown(region: dict[str, Any]) -> str:
    """Markdown for one region; image placeholders carry no page text and are dropped."""
    text = str(region.get("chart_html") or region["content"]).strip()
    if not text or _IMG_PLACEHOLDER_RE.match(text):
        return ""
    category = region["category"]
    if category == "display_formula" and not text.startswith("$"):
        return f"$$\n{text}\n$$"
    if category == "inline_formula" and not text.startswith("$"):
        return f"${text}$"
    if category == "doc_title" and not _HEADING_MARKER_RE.match(text):
        return f"# {text}"
    if category == "paragraph_title" and not _HEADING_MARKER_RE.match(text):
        return f"## {text}"
    return text


def _sanitize_html_attributes(markdown: str) -> str:
    """Quote unquoted HTML attributes for XML-based metric parsers."""

    def _quote_attrs(match: re.Match) -> str:
        tag_text = match.group(0)
        return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', tag_text)

    return re.sub(r"<[^>]+>", _quote_attrs, markdown)


def page_markdown(regions: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Page markdown plus layout items (``label``, 0-1000 ``bbox``, ``text``)."""
    items: list[dict[str, Any]] = []
    parts: list[str] = []
    for region in regions:
        text = _region_markdown(region)
        if text:
            # The model is prompted for HTML tables, but convert any pipe
            # tables it emits anyway so GriTS/TEDS can score them.
            text = _sanitize_html_attributes(_convert_pipe_tables_to_html(text))
            parts.append(text)
        items.append(
            {
                "label": _layout_label(region["category"], region["bbox"]),
                "bbox": region["bbox"],
                "text": text,
            }
        )
    return "\n\n".join(parts), items


# --- mistral_ocr.py ----------------------------------------------------------

# Regex for a separator row: only pipes, dashes, colons, and whitespace.
_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _is_pipe_table_line(line: str) -> bool:
    """Return True if *line* looks like a markdown pipe-table row."""
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith("<!--")


def _convert_pipe_tables_to_html(md_content: str) -> str:
    """Replace markdown pipe tables with HTML ``<table>`` elements.

    Non-table content is left untouched so that rule-based evaluation
    (which operates on the raw markdown text) still works.
    """
    if not md_content or "|" not in md_content:
        return md_content

    lines = md_content.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if _is_pipe_table_line(line):
            # Collect consecutive pipe-table lines.
            table_lines: list[str] = [line]
            j = i + 1
            while j < len(lines) and _is_pipe_table_line(lines[j]):
                table_lines.append(lines[j])
                j += 1

            # A valid table needs >= 3 lines (header + separator + data)
            # and must contain a separator row.
            has_separator = any(_SEPARATOR_RE.match(tl.strip()) for tl in table_lines)
            if len(table_lines) >= 3 and has_separator:
                table_md = "\n".join(table_lines)
                rendered = markdown2.markdown(table_md, extras=["tables"])
                if "<table" in rendered.lower():
                    result.append(rendered.strip())
                    i = j
                    continue

            # Not a valid table; keep the first line and advance by one.
            result.append(line)
            i += 1
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


# --- evaluation/metrics/parse/chart_json_to_html.py ---------------------------


def _json_objects(text: str) -> list[dict]:
    """Decode fenced or prefixed JSON, skipping malformed blocks as a whole."""
    if not isinstance(text, str) or not text.strip():
        return []
    blocks = re.findall(r"```(?:json)?[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    objects: list[dict] = []
    if blocks:
        for block in blocks:
            try:
                obj = json.loads(block)
            except (ValueError, RecursionError):
                continue
            if isinstance(obj, dict):
                objects.append(obj)
        return objects

    start = text.find("{")
    if start < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except (ValueError, RecursionError):
        return []
    return [obj] if isinstance(obj, dict) else []


def _is_value(value: Any) -> bool:
    return (
        isinstance(value, (str, int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _fmt(value: Any) -> str:
    if not _is_value(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _title(value: Any) -> str:
    if not isinstance(value, str) or value.strip().lower() in {"", "none", "null", "n/a"}:
        return ""
    return value.strip()


def _series_points(series: Any) -> dict[str, Any]:
    if isinstance(series, dict):
        return {str(key): value for key, value in series.items() if _is_value(value) or value is None}
    return {}


def _render_table(headers: list[str], rows: list[list[str]], titles: list[str]) -> str:
    if not rows:
        return ""
    caption_text = " / ".join(title for title in titles if title)
    caption = f"<caption>{escape(caption_text)}</caption>" if caption_text else ""
    head = "".join(f"<th>{escape(cell)}</th>" for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    # Encode pipes so the Markdown parser does not also interpret these HTML rows as a table.
    return f"<table>{caption}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>".replace("|", "&#124;")


def _panel_to_table(titles: list[str], values: dict[str, Any]) -> str:
    if not values:
        return ""
    if all(_is_value(value) or value is None for value in values.values()):
        return _render_table(["", ""], [[str(key), _fmt(value)] for key, value in values.items()], titles)

    tables = []
    series = {}
    for name, data in values.items():
        if isinstance(data, list):
            rows = [
                [_fmt(point["x"]), _fmt(point["y"])]
                for point in data
                if isinstance(point, dict) and _is_value(point.get("x")) and _is_value(point.get("y"))
            ]
            tables.append(_render_table(["x", "y"], rows, [*titles, str(name)]))
        elif points := _series_points(data):
            series[str(name)] = points
    if series:
        categories = list(dict.fromkeys(category for points in series.values() for category in points))
        rows = [[category, *(_fmt(points.get(category)) for points in series.values())] for category in categories]
        tables.append(_render_table(["", *series], rows, titles))
    return "\n\n".join(table for table in tables if table)


def chart_json_to_html(chart: dict) -> str:
    """Render each panel separately, preserving its title, series and categories."""
    if not isinstance(chart, dict):
        return ""
    figure_title = _title(chart.get("title"))
    additional_info = _title(chart.get("additional_info"))
    figure_context = [figure_title, additional_info]
    tables: list[str] = []
    panels = chart.get("panels")
    if isinstance(panels, dict) and panels:
        for name, panel in panels.items():
            if not isinstance(panel, dict):
                continue
            values = panel.get("values") or panel.get("series")
            if isinstance(values, dict):
                # Each panel retains the figure context in its caption.
                tables.append(_panel_to_table([*figure_context, _title(name)], values))
    else:
        values = chart.get("values")
        if isinstance(values, dict) and values:
            nested_panels = all(
                isinstance(panel, dict) and panel and all(isinstance(series, (dict, list)) for series in panel.values())
                for panel in values.values()
            )
            if nested_panels:
                for name, panel in values.items():
                    tables.append(_panel_to_table([*figure_context, _title(name)], panel))
            else:
                tables.append(_panel_to_table(figure_context, values))
    return "\n\n".join(table for table in tables if table)


def chart_description_to_html(description: str) -> str:
    """Convert all complete chart JSON blocks in one figure description."""
    tables = [chart_json_to_html(obj) for obj in _json_objects(description)]
    return "\n\n".join(table for table in tables if table)
