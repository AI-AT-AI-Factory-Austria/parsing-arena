import io

import pytest

from app import documents
from app.documents import DocumentError, parse_page_selection, to_page_ranges
from tests.conftest import SAMPLE_PDF


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, [1, 2, 3, 4, 5, 6]),
        ("", [1, 2, 3, 4, 5, 6]),
        ("1-3, 5", [1, 2, 3, 5]),
        ("5,1,1", [1, 5]),
        ("4-", [4, 5, 6]),
        ("3-2", [2, 3]),
        ("2-99", [2, 3, 4, 5, 6]),
    ],
)
def test_parse_page_selection(spec, expected):
    assert parse_page_selection(spec, 6) == expected


def test_parse_page_selection_rejects_out_of_range():
    with pytest.raises(DocumentError):
        parse_page_selection("9", 6)


def test_to_page_ranges():
    assert to_page_ranges([1, 2, 3, 5, 7, 8]) == "1-3,5,7-8"
    assert to_page_ranges([4]) == "4"


def test_ingest_pdf_counts_pages_and_writes_thumbnail():
    with SAMPLE_PDF.open("rb") as fh:
        document = documents.ingest(fh, "report.pdf", max_bytes=10 * 1024 * 1024)
    assert document.page_count == 2
    assert document.extension == ".pdf"
    assert document.media_type == "application/pdf"
    from app import storage

    assert storage.thumbnail_path(document.id).is_file()
    page = documents.page_image(document, 2, dpi=72)
    assert page.is_file()


def test_ingest_rejects_unknown_type():
    with pytest.raises(DocumentError, match="Unsupported file type"):
        documents.ingest(io.BytesIO(b"just some text"), "notes.txt", max_bytes=1024)


def test_ingest_enforces_size_limit():
    with SAMPLE_PDF.open("rb") as fh, pytest.raises(DocumentError):
        documents.ingest(fh, "report.pdf", max_bytes=1024)
