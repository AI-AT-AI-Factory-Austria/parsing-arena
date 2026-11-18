"""MinerU 2.5 Pro output normalization.

[parsing-arena] Vendored from run-llama/ParseBench (Apache-2.0), commit 0ccc167:
``src/parse_bench/inference/providers/parse/mineru2605pro.py`` (static helpers of
``MinerU2605ProProvider``, unchanged). ``normalize_markdown`` applies them in the order of
the provider's ``normalize``.
"""

import re

# MinerU block types (mineru-vl-utils 1.0.x BlockType) -> Canonical17 layout labels
LABEL_MAP: dict[str, str] = {
    "text": "Text",
    "title": "Title",
    "doc_title": "Title",
    "paragraph_title": "Section-header",
    "table": "Table",
    "table_caption": "Caption",
    "table_footnote": "Footnote",
    "figure": "Picture",
    "image": "Picture",
    "image_block": "Picture",
    "image_caption": "Caption",
    "image_footnote": "Footnote",
    "figure_caption": "Caption",
    "formula": "Formula",
    "equation": "Formula",
    "equation_block": "Formula",
    "formula_number": "Formula",
    "display_formula": "Formula",
    "inline_formula": "Formula",
    "header": "Page-header",
    "page_header": "Page-header",
    "footer": "Page-footer",
    "page_footer": "Page-footer",
    "page_number": "Page-footer",
    "footnote": "Footnote",
    "page_footnote": "Footnote",
    "list": "List-item",
    "list_item": "List-item",
    "code": "Code",
    "code_caption": "Caption",
    "algorithm": "Code",
    "aside_text": "Text",
    "ref_text": "Text",
    "phonetic": "Text",
    "chart": "Picture",
    "unknown": "Picture",
}


def _close_unclosed_table_tags(content: str) -> str:
    opens = content.count("<table>")
    closes = content.count("</table>")
    if opens > closes:
        if not content.rstrip().endswith(">"):
            content += "</td></tr>"
        content += "</table>" * (opens - closes)
    return content


def _promote_first_row_to_thead(content: str) -> str:
    """MinerU typically outputs first row as <td> — promote to <thead><th>."""

    def _promote(match: re.Match[str]) -> str:
        table_html = match.group(0)
        if "<thead" in table_html:
            return table_html
        first_tr = re.search(r"<tr>(.*?)</tr>", table_html, re.DOTALL)
        if not first_tr:
            return table_html
        first_tr_full = first_tr.group(0)
        first_tr_inner = first_tr.group(1)
        header_inner = first_tr_inner.replace("<td>", "<th>").replace("</td>", "</th>")
        header_inner = re.sub(r"<td(\s)", r"<th\1", header_inner)
        header_inner = re.sub(r"</td>", "</th>", header_inner)
        thead = f"<thead><tr>{header_inner}</tr></thead>"
        return table_html.replace(first_tr_full, thead, 1)

    return re.sub(r"<table>.*?</table>", _promote, content, flags=re.DOTALL)


def _sanitize_html_attributes(markdown: str) -> str:
    def _quote_attrs(match: re.Match) -> str:
        tag = match.group(0)
        return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', tag)

    return re.sub(r"<[^>]+>", _quote_attrs, markdown)


def normalize_markdown(page_md: str) -> str:
    if page_md:
        page_md = _close_unclosed_table_tags(page_md)
        page_md = _promote_first_row_to_thead(page_md)
        page_md = _sanitize_html_attributes(page_md)
    return page_md
