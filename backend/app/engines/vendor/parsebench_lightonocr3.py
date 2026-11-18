r"""LightOnOCR-3 output normalization.

[parsing-arena] Vendored from run-llama/ParseBench (Apache-2.0), commit 0ccc167:
``src/parse_bench/inference/providers/parse/lightonocr-3.py`` and the two table helpers of
``qwen.py``. The functions are unchanged; ``normalize_page`` restates the provider's
``_normalize_page`` without ParseBench's layout schema.

The model answers each page as blocks separated by blank lines:

    ![<label>](x1,y1,x2,y2) <text content>

Boxes are 0-1000 normalized (left, top, right, bottom). ``table`` text is HTML, ``formula`` is LaTeX,
``title`` text carries its own ``#`` heading, and figure/chart/image blocks carry a description or nothing.
A ``+`` label suffix (``text+``) marks a block continued from the previous page or column.
"""

import re
from typing import Any

SERVED_MODEL_NAME = "loocr-grounding"
DEFAULT_PROMPT = "grounding"

_BLOCK_MARKER_RE = re.compile(r"!\[([a-zA-Z_+]+)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*")

# Raw block label -> ParseBench Canonical17 category. Unlisted labels (list, caption, code) map to Text.
LABEL_MAP: dict[str, str] = {
    "paragraph": "Text",
    "title": "Title",
    "figure_title": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "table": "Table",
    "header": "Page-header",
    "footer": "Page-footer",
    "page_number": "Page-footer",
    "figure": "Picture",
    "chart": "Picture",
    "image": "Picture",
    # ParseBench's layout ground truth labels an image in the header/footer band as Picture.
    "header_image": "Picture",
    "footer_image": "Picture",
}

# The model writes no language tag on code blocks, which ParseBench's code rule needs. Only the languages
# its text-formatting tests use are detected; anything else gets a bare fence.
_CODE_LANG_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("json", re.compile(r"^\s*[\{\[].*[\}\]]\s*$", re.DOTALL)),
    ("python", re.compile(r"\b(def |import |from \w+ import|self\.|print\()")),
    ("cpp", re.compile(r"#include|::|\bstd::|\bvoid\b.*\(|;\s*$", re.MULTILINE)),
    ("fortran", re.compile(r"^\s{6}\S|\bIF\(|\bEND\s*DO\b|\bCALL\b", re.MULTILINE)),
]
_LEADING_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+")


def _base_label(raw_label: str) -> str:
    """``" Text+ "`` -> ``"text"``: the continuation marker and case are not part of the label."""
    return raw_label.strip().lower().rstrip("+")


def _strip_formula_delimiters(text: str) -> str:
    """Remove one outer ``$$`` or ``$`` pair, if the inside has no other ``$``."""
    stripped = text.strip()
    for delim in ("$$", "$"):
        if len(stripped) > 2 * len(delim) and stripped.startswith(delim) and stripped.endswith(delim):
            interior = stripped[len(delim) : -len(delim)]
            if "$" not in interior and interior.strip():
                return interior.strip()
    return text


def _render_block(raw_label: str, text: str) -> str:
    """Markdown for one block. Adds the markup its label implies when the model left it out.

    A title keeps the model's own ``#`` level: the model uses one ``title`` label for titles and section
    headers, so H1 is added only when the text has no heading marker.
    """
    if not text.strip():
        return ""
    label = _base_label(raw_label)
    if label == "code" and not text.lstrip().startswith("```"):
        lang = next((name for name, pattern in _CODE_LANG_HINTS if pattern.search(text)), "")
        return f"```{lang}\n{text}\n```"
    if label == "title" and not _LEADING_HEADING_RE.match(text):
        return f"# {text.strip()}"
    if label == "formula":
        return f"$$\n{_strip_formula_delimiters(text)}\n$$"
    return text


# ---------------------------------------------------------------------------
# Escaped dollar repair
#
# The model sometimes escapes math delimiters (``\$x^2\$``). This pairs them back into ``$x^2$`` when the
# span reads as math, and leaves prices, prose and code alone. Standard library only; behavior matches
# ``fix_escaped_dollars`` in the LightOnOCR repo.
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(
    r"\\(?:[A-Za-z]+|[^\r\n])|(?:\d+(?:\.\d*)?|\.\d+)|[^\W\d_]\w*|[-−–—+*/=<>:;,.'!|_^%&()\[\]{}]", re.UNICODE
)
_WORD_RE = re.compile(r"[^\W\d_]\w*")
_NUMERIC_START_RE = re.compile(r"[+\-−]?\s*(?:\d|[.,]\d)")
_DOLLAR_RE = re.compile(r"(?<![\\$])\\?\$(?!\$)")
_INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\$)((?:\\.|[^$\\\n])*)(?<!\\)\$(?!\$)")

# Spans that must never be edited: code, HTML, links, display math and comments.
_PROTECTED_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[ \t]*(?:\n|$)|\Z)"
    r"|(?P<ticks>`+)[^`\n]*(?P=ticks)"
    r"|^(?: {4}|\t)[^\n]*(?:\n|$)"
    r"|<!--.*?(?:-->|\Z)"
    r"|<(?:pre|code|script|style)\b[^>]*>.*?(?:</(?:pre|code|script|style)\s*>|\Z)"
    r"|<[/!?A-Za-z](?:\"[^\"]*\"|'[^']*'|[^'\">])*>"
    r"|(?:https?://|mailto:|www\.)[^\s<>]+"
    r"|^[ \t]{0,3}\[[^\]\n]+\]:[^\n]*"
    r"|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)",
    re.M | re.S | re.I,
)


def _braces_balanced(body: str) -> bool:
    depth = 0
    for token in re.findall(r"\\.|[{}]", body):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _mask_command_arguments(body: str) -> str | None:
    """Replace the braced arguments of each ``\\command`` with ``{}``. None if one is unclosed."""
    out = []
    pos = 0
    while pos < len(body):
        command = re.match(r"\\[A-Za-z]+", body[pos:])
        if not command:
            out.append(body[pos])
            pos += 1
            continue
        out.append(command[0])
        pos += len(command[0])
        while pos < len(body) and body[pos] == "{":
            depth = 1
            pos += 1
            while pos < len(body) and depth:
                if body[pos] == "\\":
                    pos += 2
                    continue
                depth += (body[pos] == "{") - (body[pos] == "}")
                pos += 1
            if depth:
                return None
            out.append("{}")
    return "".join(out)


def _is_lexical_math(body: str) -> bool:
    """Conservative check that ``body`` holds only identifiers, numbers and operators.

    This is lexical evidence, not proof of mathematical correctness. Adjacent words are not accepted as an
    implicit product, but command arguments may hold text (``\\text{a long description}``).
    """
    body = body.strip()
    if not body or "<td" in body.lower() or "</" in body or not _braces_balanced(body):
        return False
    clean = _mask_command_arguments(body)
    if clean is None:
        return False
    previous_word = None
    pos = 0
    for match in _ATOM_RE.finditer(clean):
        if clean[pos : match.start()].strip():
            return False
        is_word = bool(_WORD_RE.fullmatch(match[0]))
        if is_word and previous_word and (len(match[0]) > 1 or len(previous_word) > 1):
            return False
        previous_word = match[0] if is_word else None
        pos = match.end()
    return not clean[pos:].strip()


def _is_plausible_math(body: str) -> bool:
    stripped = body.strip()
    if not _is_lexical_math(body) or re.fullmatch(r"[^\W\d_]{2,}", stripped):
        return False
    if body != stripped and not re.search(r"[\w\\]", stripped):
        return False
    if len(stripped) > 1 and stripped[0] in ",;":
        return False
    # Escaped delimiters and text arguments do not count toward bracket balance.
    simple = re.sub(r"\\text\{[^{}]*\}", "", stripped)
    level = 0
    for token in re.findall(r"\\.|[()\[\]]", simple):
        if token in ("(", "["):
            level += 1
        elif token in (")", "]"):
            level -= 1
        if level < 0:
            return False
    return level == 0


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Protected spans, plus valid-looking inline math (a ``$`` inside it is not a broken delimiter)."""
    spans = [(m.start(), m.end()) for m in _PROTECTED_RE.finditer(text)]
    # Inline link destinations, including relative URLs.
    for m in re.finditer(r"\]\(", text):
        pos = m.end()
        depth = 1
        while pos < len(text) and text[pos] != "\n" and depth:
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1
        if depth == 0:
            spans.append((m.start(), pos))
    for m in _INLINE_MATH_RE.finditer(text):
        if _is_plausible_math(m[1]):
            spans.append((m.start(), m.end()))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _is_ambiguous_pair(body: str, before: str, after: str, left_escaped: bool, right_escaped: bool) -> bool:
    """Whether an escaped-delimiter pair could be a price, prose or a broken span: leave those alone."""
    numeric = bool(_NUMERIC_START_RE.match(body.strip()))
    is_environment = re.fullmatch(r"\s*\\begin\{([A-Za-z*]+)\}.*\\end\{\1\}\s*", body, re.S)
    # A currency code glued to the dollar sign, e.g. US$, S$, HK$.
    code_currency = left_escaped and bool(re.search(r"(?<!\w)[A-Z]{1,3}$", before))
    return bool(
        ("\n" in body or "\r" in body)
        and not is_environment
        or re.search(r"\n[ \t]*\n", body)
        or code_currency
        or (left_escaped and not right_escaped and numeric)
        or (numeric and _NUMERIC_START_RE.match(after.lstrip(" \t")))
        or re.fullmatch(r"[^\W\d_]{2,}", body.strip())
        or not _is_plausible_math(body)
    )


def _delimiter_edits(segment: str, offset: int) -> list[tuple[int, int, str]]:
    """Edits that pair the dollar delimiters of one unprotected segment. ``offset`` is its start in the text."""
    tokens = list(_DOLLAR_RE.finditer(segment))
    edits = []
    i = 0
    while i + 1 < len(tokens):
        left, right = tokens[i], tokens[i + 1]
        body = segment[left.end() : right.start()]
        left_escaped = left[0].startswith("\\")
        right_escaped = right[0].startswith("\\")
        if not left_escaped and not right_escaped:
            i += 2 if _is_plausible_math(body) else 1
            continue
        if _is_ambiguous_pair(
            body,
            before=segment[: left.start()],
            after=segment[right.end() :],
            left_escaped=left_escaped,
            right_escaped=right_escaped,
        ):
            i += 1
            continue
        edits.append((offset + left.start(), offset + right.end(), f"${body}$"))
        i += 2
    return edits


def _repair_dollar_delimiters(text: str) -> str:
    edits: list[tuple[int, int, str]] = []
    previous = 0
    for start, end in _protected_spans(text):
        edits += _delimiter_edits(text[previous:start], previous)
        previous = end
    edits += _delimiter_edits(text[previous:], previous)
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def parse_blocks(content: str) -> list[dict[str, Any]]:
    """The raw response as ordered ``{label, bbox, text}`` blocks."""
    matches = list(_BLOCK_MARKER_RE.finditer(content))
    ends = [m.start() for m in matches[1:]] + [len(content)] if matches else []
    return [
        {"label": m[1], "bbox": [int(m[i]) for i in range(2, 6)], "text": content[m.end() : end].strip()}
        for m, end in zip(matches, ends, strict=True)
    ]


# --- qwen.py (QwenProvider static helpers) ---------------------------------


def _sanitize_html_attributes(text: str) -> str:
    def _quote_attrs(match: re.Match) -> str:
        tag_text = match.group(0)
        return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', tag_text)

    return re.sub(r"<[^>]+>", _quote_attrs, text)


def _convert_md_tables_to_html(content: str) -> str:
    """Convert markdown pipe tables to HTML <table> elements."""
    import markdown2

    lines = content.split("\n")
    result_parts: list[str] = []
    table_lines: list[str] = []
    in_table = False

    for line in lines:
        is_table_line = "|" in line and line.strip().startswith("|")
        if is_table_line:
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
        else:
            if in_table:
                if len(table_lines) >= 2:
                    table_md = "\n".join(table_lines)
                    html = markdown2.markdown(table_md, extras=["tables"]).strip()
                    if "<table>" in html.lower():
                        result_parts.append(html)
                    else:
                        result_parts.extend(table_lines)
                else:
                    result_parts.extend(table_lines)
                table_lines = []
                in_table = False
            result_parts.append(line)

    if in_table and len(table_lines) >= 2:
        table_md = "\n".join(table_lines)
        html = markdown2.markdown(table_md, extras=["tables"]).strip()
        if "<table>" in html.lower():
            result_parts.append(html)
        else:
            result_parts.extend(table_lines)
    elif in_table:
        result_parts.extend(table_lines)

    return "\n".join(result_parts)


# --- _normalize_page --------------------------------------------------------


def normalize_page(raw_content: str) -> tuple[str, list[dict[str, Any]]]:
    """Markdown for one page plus its blocks (``label``, ``category``, 0-1000 ``bbox``, ``text``)."""
    items = [
        {
            **block,
            "category": LABEL_MAP.get(_base_label(block["label"]), "Text"),
            "text": _repair_dollar_delimiters(str(block.get("text", ""))),
        }
        for block in parse_blocks(raw_content)
    ]
    blocks = [_render_block(str(item.get("label", "")), item["text"]) for item in items]
    markdown = "\n\n".join(block for block in blocks if block)
    if markdown:
        markdown = _sanitize_html_attributes(_convert_md_tables_to_html(markdown))
    return markdown, items
