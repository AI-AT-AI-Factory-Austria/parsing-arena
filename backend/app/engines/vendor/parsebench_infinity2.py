"""Infinity-Parser2 output normalization.

[parsing-arena] Vendored from run-llama/ParseBench (Apache-2.0), commit 0ccc167:
``src/parse_bench/inference/providers/parse/infinity_parser2.py``. The helpers are unchanged;
``reassemble_text`` is the provider's ``_reassemble_text`` and ``normalize_page`` restates the
per-page part of ``_normalize`` (header items first, footer items last) without ParseBench's
layout schema.
"""

import re
from typing import Any

# Infinity-Parser2 category -> Canonical17 label mapping
INFINITY_CATEGORY_MAP: dict[str, str] = {
    "header": "Page-header",
    "title": "Section-header",
    "text": "Text",
    "figure": "Picture",
    "table": "Table",
    "formula": "Formula",
    "figure_caption": "Caption",
    "table_caption": "Caption",
    "formula_caption": "Caption",
    "figure_footnote": "Footnote",
    "table_footnote": "Footnote",
    "page_footnote": "Footnote",
    "footer": "Page-footer",
}


def reassemble_text(label: str, text: str) -> str:
    """Reassemble text content based on label."""
    if not text:
        return ""

    if label == "Section-header":
        return f"# {text.lstrip('# ')}"
    elif label == "Formula":
        stripped = re.sub(r"^[\s$\(\)\[\]]+|[\s$\(\)\[\]]+$", "", text)
        return f"$${stripped}$$"
    elif label == "Picture":
        text = _convert_nonstandard_table(text)
        return text
    elif label == "Table":
        return _convert_table_header(text)
    else:
        return text


def normalize_page(elements: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Page markdown plus ordered items (``category``, pixel ``bbox``, ``md``)."""
    header_items: list[dict] = []
    footer_items: list[dict] = []
    regular_items: list[dict] = []

    for elem in elements:
        raw_cat = elem.get("category", "text").strip().lower()
        norm_cat = INFINITY_CATEGORY_MAP.get(raw_cat, "Text")
        item = {
            "category": norm_cat,
            "bbox": elem.get("bbox", [0, 0, 0, 0]),
            "md": reassemble_text(norm_cat, elem.get("text", "")),
        }
        if norm_cat == "Page-header":
            header_items.append(item)
        elif norm_cat == "Page-footer":
            footer_items.append(item)
        else:
            regular_items.append(item)

    page_items = header_items + regular_items + footer_items
    page_md_parts = [item.get("md", "") for item in page_items if item.get("md")]
    return "\n\n".join(page_md_parts), page_items


# =============================================================================
# Postprocess for chart2table
# =============================================================================


def _is_valid_md_table(table_text: str) -> bool:
    """Check if a markdown table is valid (non-empty)."""
    if not table_text or not table_text.strip():
        return False

    if not all(ch in table_text for ch in ["|", "-", "\n"]):
        return False

    stripped = table_text[table_text.find("|") : table_text.rfind("|") + 1]
    stripped = re.sub(r"^\s*\|[\s\-:|]+\|\s*$", "", stripped, flags=re.MULTILINE)
    if not stripped.replace(" ", "").replace("\n", "").replace("|", ""):
        return False

    return True


def _is_nonstandard_table(text: str) -> bool:
    """Check if text is a non-standard markdown table (no leading '|', contains '&' separators)."""
    if not text:
        return False
    stripped = text.strip()
    if stripped.startswith("|"):
        return False
    return "&" in text


def _find_column_number(text: str) -> int:
    """Find the column number from a nonstandard table.

    Split the text by '&' and count '|' in each segment. The header row always
    has the most pipes (full cells). Column count = max(pipe_counts) + 1.
    """
    if "&" not in text:
        return 0
    raw_segments = text.split("&")
    segments = [s.strip() for s in raw_segments if s.strip()]
    if not segments:
        return 0
    pipe_counts = [s.count("|") for s in segments]
    return max(pipe_counts) + 1


def _find_all_separator_indices(text: str, col_num: int) -> list[int]:
    """Identify which '&' characters are row-group separators based on pipe counts."""
    if col_num == 0:
        return []
    expected_pipes = col_num - 1
    sep_positions = []
    prev_sep = -1
    i = 0
    while i < len(text):
        amp = text.find("&", i)
        if amp == -1:
            break

        # Count pipes between prev_sep+1 and amp-1
        pipe_count = 0
        for j in range(prev_sep + 1, amp):
            if text[j] == "|":
                pipe_count += 1

        if pipe_count == expected_pipes:
            sep_positions.append(amp)
            prev_sep = amp

        i = amp + 1
    return sep_positions


def _convert_nonstandard_table(text: str) -> str:
    """Convert a non-standard markdown table (with '&' row-group separators) to proper markdown table format."""
    if not _is_nonstandard_table(text):
        return text

    col_num = _find_column_number(text)
    if col_num == 0:
        return text

    sep_indices = _find_all_separator_indices(text, col_num)
    if not sep_indices:
        return text

    segments = []
    prev = 0
    for idx in sep_indices:
        segments.append(text[prev:idx].strip())
        prev = idx + 1
    segments.append(text[prev:].strip())

    header = segments[0]
    if not header.startswith("|"):
        header = "| " + header
    if not header.rstrip().endswith("|"):
        header = header.rstrip() + " |"

    separator = "| " + " | ".join(["---"] * col_num) + " |"
    normalized_lines = [header, separator]

    for seg in segments[1:]:
        if not seg:
            continue
        cells = [c.strip() for c in seg.split("|") if c.strip()]
        padded = cells + [""] * max(0, col_num - len(cells))
        row = "| " + " | ".join(padded[:col_num]).rstrip() + " |"
        normalized_lines.append(row)

    return "\n".join(normalized_lines)


# =============================================================================
# Postprocess for HTML table header
# =============================================================================


def _is_year_cell(text: str) -> bool:
    """Return True if text looks like a date/year (yyyy, yyyymm, yyyymmdd, etc.)."""
    text = text.strip()
    return bool(re.fullmatch(r"(19|20)\d{2,4}([-/]?\d{2}([-/]?\d{2})?)?", text))


def _is_gender_cell(text: str) -> bool:
    """Return True if text looks like gender."""
    text = text.strip().lower()
    return text in ("male", "female", "non-binary", "other", "undisclosed")


def _is_pure_text_cell(text: str) -> bool:
    """Return True if text contains no digits at all."""
    text = text.strip()
    return bool(text) and any(c.isalpha() for c in text)


def _is_pure_number_cell(text: str) -> bool:
    """Return True if text looks like a pure numeric value.

    Accepts numbers with commas, decimals, dollar sign, percent sign,
    plus/minus sign, and parentheses (for negative numbers).
    """
    text = text.strip()
    if not text:
        return False
    # Allow: digits, comma, dot, minus, plus, $, %, parentheses
    allowed = set("0123456789,.-+()$% ")
    return all(c in allowed for c in text)


def _determine_header_row_count(rows: list) -> int:
    """Determine how many top rows are header rows (year/gender/value rules + rowspan fallback)."""
    if not rows:
        return 0

    def non_empty_cells(row):
        return [td.get_text(strip=True) for td in row.find_all("td", recursive=False) if td.get_text(strip=True)]

    def stats(row_list):
        """Return (pure_text_count, pure_number_count, total) for a list of rows."""
        text_count = number_count = total = 0
        for row in row_list:
            for cell in non_empty_cells(row):
                total += 1
                if _is_pure_text_cell(cell):
                    text_count += 1
                elif _is_pure_number_cell(cell):
                    number_count += 1
        return text_count, number_count, total

    # Rule 1: Year
    for i, row in enumerate(rows):
        if i > 3:
            break
        cells = non_empty_cells(row)
        if not cells:
            continue
        year_count = sum(1 for c in cells if _is_year_cell(c))
        if year_count / len(cells) >= 0.5:
            return i + 1

    # Rule 2: Gender
    for i, row in enumerate(rows):
        if i > 3:
            break
        cells = non_empty_cells(row)
        if not cells:
            continue
        gender_count = sum(1 for c in cells if _is_gender_cell(c))
        if gender_count / len(cells) >= 0.5:
            return i + 1

    # Rule 3: Value (pure-text header region followed by pure-number data region)
    best_i = -1
    best_score = -1.0
    for i in range(3):
        header_rows = rows[: i + 1]
        data_rows = rows[i + 1 :]
        if not header_rows or not data_rows:
            continue
        header_text, header_num, header_total = stats(header_rows)
        data_text, data_num, data_total = stats(data_rows)
        if header_total == 0 or data_total == 0:
            continue
        if header_text / header_total >= 0.5 and data_num / data_total >= 0.5:
            score = header_text / header_total + data_num / data_total
            if score > best_score:
                best_score = score
                best_i = i
    if best_i >= 0:
        return best_i + 1

    # Rule 4: Fallback - max rowspan in row 0
    first_row = rows[0]
    max_rowspan = 1
    for td in first_row.find_all("td", recursive=False):
        rowspan = int(td.get("rowspan", 1))
        if rowspan > max_rowspan:
            max_rowspan = rowspan
    return max_rowspan


def _convert_table_header(html: str) -> str:
    """Convert <td> tags in HTML table header rows to <th> for TEDS/GriTS evaluation."""
    if not html or "<table" not in html.lower():
        return html

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr", recursive=False)
        if not rows:
            continue

        header_row_count = _determine_header_row_count(rows)

        for i, row in enumerate(rows):
            if i >= header_row_count:
                break
            tds = row.find_all("td", recursive=False)
            for td in tds:
                new_th = soup.new_tag("th")
                for key, value in td.attrs.items():
                    new_th[key] = value
                new_th.string = td.get_text()
                td.replace_with(new_th)

    return str(soup)
