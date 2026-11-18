"""Granite integration checks without downloading or running model weights."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.catalog import get_model
from app.config import Settings
from app.engines.backends.transformers_local import LocalModelPool
from app.engines.base import PageSource
from app.engines.granite import GraniteDoclingEngine, GraniteDoclingRuntime
from app.engines.open_engine import PageFailed


@pytest.mark.parametrize("extension,number,doc_page", [(".pdf", 4, 4), (".png", 7, 1)])
def test_conversion_exports_only_the_selected_page(extension, number, doc_page):
    pytest.importorskip("docling")
    runtime = GraniteDoclingRuntime(get_model("granite-docling-258m"), Settings())
    document = SimpleNamespace(pages={}, export_to_markdown=Mock(return_value="Selected page"))
    runtime.converter = Mock()
    runtime.converter.convert.return_value = SimpleNamespace(
        status="success",
        document=document,
        errors=[],
    )
    source = Path(f"document{extension}")
    pages = PageSource(numbers=[number], paths={}, source=source, extension=extension)

    assert runtime.convert_page(pages, number) == ("Selected page", [])
    kwargs = {"page_range": (number, number)} if extension == ".pdf" else {}
    runtime.converter.convert.assert_called_once_with(str(source), raises_on_error=False, **kwargs)
    from docling_core.types.doc import ContentLayer

    document.export_to_markdown.assert_called_once_with(
        page_no=doc_page,
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
        traverse_pictures=True,
        include_picture_classification=False,
    )


def test_truncated_conversion_is_reported_as_a_page_failure():
    pytest.importorskip("docling")
    runtime = GraniteDoclingRuntime(get_model("granite-docling-258m"), Settings())
    runtime.converter = Mock()
    runtime.converter.convert.return_value = SimpleNamespace(
        status="partial_success",
        errors=[SimpleNamespace(error_message="Token limit reached")],
    )
    pages = PageSource(numbers=[1], paths={}, source=Path("document.pdf"), extension=".pdf")

    with pytest.raises(PageFailed, match="Token limit reached"):
        runtime.convert_page(pages, 1)


def test_selected_pages_and_partial_failure_keep_progress_and_page_numbers(tmp_path):
    engine = GraniteDoclingEngine(
        get_model("granite-docling-258m"),
        Settings(),
        LocalModelPool(keep_loaded=False, max_loaded=1),
    )
    backend = Mock()
    backend.convert_page.side_effect = [("First page", []), PageFailed("Token limit reached")]
    pages = PageSource(numbers=[2, 5], paths={}, source=Path("document.pdf"), extension=".pdf")
    updates = []

    async def report(update):
        updates.append(update)

    output = asyncio.run(engine.parse(backend, pages, report=report, artifact_dir=tmp_path))
    assert [call.args[1] for call in backend.convert_page.call_args_list] == [2, 5]
    assert [page.page_number for page in output.pages] == [2, 5]
    assert output.markdown == "First page"
    assert output.pages[0].error is None
    assert output.pages[1].error == "Token limit reached"
    assert all(page.seconds >= 0 for page in output.pages)
    assert updates[-1]["pages_done"] == 2
    assert updates[-1]["page"] == 5


def test_all_failed_pages_do_not_produce_a_successful_run(tmp_path):
    engine = GraniteDoclingEngine(
        get_model("granite-docling-258m"),
        Settings(),
        LocalModelPool(keep_loaded=False, max_loaded=1),
    )
    backend = Mock()
    backend.convert_page.side_effect = PageFailed("Inference failed")
    pages = PageSource(numbers=[1], paths={}, source=Path("document.pdf"), extension=".pdf")

    async def report(update):
        pass

    with pytest.raises(RuntimeError, match="Inference failed"):
        asyncio.run(engine.parse(backend, pages, report=report, artifact_dir=tmp_path))
