import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.catalog import get_model
from app.config import Settings
from app.engines.base import PageSource
from app.engines.library_worker import LibraryRuntime
from app.engines.preset_worker import Marker, OpenDataLoader, artifact_links, odl_elements
from app.engines.presets import DOCTR_CONFIGS, MARKER_CONFIGS, odl_options
from app.engines.registry import EngineRegistry

PRESETS = [*MARKER_CONFIGS, "opendataloader_core", "opendataloader_docling", *DOCTR_CONFIGS]


def test_generated_images_use_run_artifact_links_including_angle_bracket_paths(tmp_path):
    folder = tmp_path / "artifacts" / "page-4"
    images = folder / "report_images"
    images.mkdir(parents=True)
    (images / "image 1.png").write_bytes(b"image")
    (images / "image2.png").write_bytes(b"image")
    markdown = "![](<report_images/image 1.png>)\n![chart](report_images/image2.png)"
    rewritten = artifact_links(markdown, folder, tmp_path)
    assert "![](<artifacts/page-4/report_images/image 1.png>)" in rewritten
    assert "![chart](<artifacts/page-4/report_images/image2.png>)" in rewritten
    assert artifact_links("![](https://example.org/a.png)", folder, tmp_path) == (
        "![](https://example.org/a.png)"
    )


def test_odl_boxes_convert_bottom_left_origin_and_exclude_other_pages():
    data = {
        "kids": [
            {
                "type": "paragraph",
                "page number": 4,
                "bounding box": [10, 100, 90, 180],
                "content": "Selected",
            },
            {
                "type": "paragraph",
                "page number": 5,
                "bounding box": [10, 100, 90, 180],
                "content": "Other page",
            },
        ]
    }
    items = odl_elements(data, 4, width=100, height=200)
    assert len(items) == 1
    assert items[0]["bbox"] == pytest.approx([0.1, 0.1, 0.9, 0.5])
    assert items[0]["content"] == "Selected"


def test_presets_have_distinct_cache_entries_and_no_borrowed_benchmark_scores():
    registry = EngineRegistry(Settings())
    engines = [registry.open_engine(preset) for preset in PRESETS]
    assert len({engine.pool_key for engine in engines}) == 7
    assert all(engine.supports_endpoint is False for engine in engines)
    assert all(get_model(preset).scores is None for preset in PRESETS)


def test_library_selection_preserves_nonconsecutive_page_numbers(tmp_path):
    engine = EngineRegistry(Settings()).open_engine("opendataloader_core")
    backend = Mock()
    backend.convert.side_effect = [
        {"markdown": "Page two", "elements": []},
        {"markdown": "Page five", "elements": []},
    ]
    updates = []

    async def report(update):
        updates.append(update)

    output = asyncio.run(
        engine.parse(
            backend,
            PageSource([2, 5], {}, Path("original.pdf"), ".pdf"),
            report=report,
            artifact_dir=tmp_path,
        )
    )
    assert [call.args[1] for call in backend.convert.call_args_list] == [2, 5]
    assert [page.page_number for page in output.pages] == [2, 5]
    assert output.markdown == "Page two\n\nPage five"
    assert updates[-1]["pages_done"] == 2


@pytest.mark.parametrize("preset", ["opendataloader_core", "opendataloader_docling"])
def test_odl_passes_page_selection_and_fails_instead_of_silent_hybrid_fallback(
    preset,
    monkeypatch,
    tmp_path,
):
    package = pytest.importorskip("opendataloader_pdf")
    runtime = OpenDataLoader.__new__(OpenDataLoader)
    runtime.preset, runtime.options = preset, odl_options(preset)
    runtime.server = (
        SimpleNamespace(url="http://127.0.0.1:12345") if preset.endswith("docling") else None
    )
    convert = Mock(side_effect=RuntimeError("Backend unavailable"))
    monkeypatch.setattr(package, "convert", convert)
    with pytest.raises(RuntimeError, match="Backend unavailable"):
        runtime.convert("source.pdf", 4, tmp_path)
    options = convert.call_args.kwargs
    assert options["pages"] == "4"
    assert options["input_path"] == "source.pdf"
    if runtime.server:
        assert options["hybrid"] == "docling-fast"
        assert options["hybrid_mode"] == "auto"
        assert options["hybrid_fallback"] is False
        assert options["hybrid_url"] == runtime.server.url
    else:
        assert options["hybrid"] == "off"
        assert "hybrid_url" not in options


def test_worker_reports_errors_instead_of_accepting_them_as_output(tmp_path):
    runtime = LibraryRuntime("marker_fast", Settings(data_dir=tmp_path))
    runtime.responses.put({"error": "Model could not load"})
    with pytest.raises(RuntimeError, match="Model could not load"):
        runtime._response()


def test_fast_preset_does_not_start_surya_inference_or_ocr_error_server(monkeypatch):
    import sys

    from app.engines import preset_worker

    monkeypatch.setattr(preset_worker, "configure_surya_cleanup", lambda: None)

    models = {
        name: Mock()
        for name in (
            "inference_manager",
            "fast_layout_model",
            "ocr_error_model",
        )
    }
    monkeypatch.setitem(
        sys.modules,
        "marker.models",
        SimpleNamespace(
            create_model_dict=Mock(return_value=models),
        ),
    )
    marker = Marker("marker_fast")
    models["fast_layout_model"].assert_called_once()
    models["inference_manager"].start.assert_not_called()
    models["ocr_error_model"].assert_not_called()
    assert marker.config["disable_ocr"] is True
    assert marker.config["use_llm"] is False
