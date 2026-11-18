import asyncio
import json

import pytest
from PIL import Image

from app.catalog import ALL_MODELS
from app.config import Settings
from app.engines import unlimited
from app.engines.backends.transformers_local import LocalModelPool
from app.engines.base import PageSource
from app.engines.registry import EngineRegistry


def install_manifest(tmp_path, backend="Vulkan", revision=unlimited.SPEC.revision):
    root = tmp_path / "runtimes/unlimited-ocr2"
    root.mkdir(parents=True, exist_ok=True)
    (root / "llama-server").touch()
    (root / "build-info.json").write_text(
        json.dumps(
            {
                "llama_version": unlimited.LLAMA_VERSION,
                "patch_revision": revision,
                "binary": "llama-server",
                "backend": backend,
            }
        )
    )


def test_page_mapping_headings_tables_and_blank_pages():
    table = "<table><tr><td>A</td><td>17</td></tr></table>"
    raw = (
        "<PAGE><BLOCK>title<LEVEL2>[10, 20, 300, 40]<FORK>Overview\n"
        f"<BLOCK>table[20, 50, 900, 500]<FORK>{table}\n<PAGE>" + unlimited.EOS[0] + "\n"
    )
    first, blank = unlimited.parse_document(raw, [2, 5])
    assert first.page_number == 2
    assert first.markdown == f"## Overview\n\n{table}"
    assert first.elements[0]["bbox"] == [0.01, 0.02, 0.3, 0.04]
    assert first.elements[0]["category"] == "title"
    assert blank.page_number == 5 and blank.markdown == "" and blank.elements == []
    assert first.seconds is None  # Independent per-page timings would be fabricated.


def test_titles_without_levels_keep_heading_semantics():
    raw = (
        "<PAGE><BLOCK>header[0,0,100,10]<FORK>Navigation\n"
        "<BLOCK>title[0,20,100,40]<FORK>Our Code\n"
        "<BLOCK>text[0,50,100,60]<FORK>Body text\n"
        "<BLOCK>title[0,70,100,80]<FORK>## Already formatted\n"
        "<BLOCK>footer[0,90,100,100]<FORK>Footer"
    )
    page = unlimited.parse_document(raw, [1])[0]
    assert page.markdown == (
        "Navigation\n\n# Our Code\n\nBody text\n\n## Already formatted\n\nFooter"
    )
    assert page.elements[1]["content"] == "Our Code"


@pytest.mark.parametrize(
    "raw,numbers",
    [
        ("<PAGE><BLOCK>text[0,0,100,100]<FORK>one", [1, 2]),
        ("<PAGE>one<PAGE>two<PAGE>three", [1, 2]),
        ("unmarked text", [1]),
        ("<PAGE><BLOCK>text[broken]<FORK>text", [1]),
        ("<PAGE><BLOCK>text[200,0,100,10]<FORK>backwards", [1]),
        ("<PAGE><BLOCK>text[0,0,1001,100]<FORK>out of bounds", [1]),
        ("<PAGE><BLOCK>text[0,0,100,100]<FORK>first<BLOCK>incomplete", [1]),
    ],
)
def test_invalid_or_incomplete_output_is_not_reported_as_success(raw, numbers):
    with pytest.raises(RuntimeError):
        unlimited.parse_document(raw, numbers)


def test_dedicated_runtime_and_hardware_dispatch(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path, local_inference=True)
    monkeypatch.setattr(unlimited, "profile_error", lambda *_: None)
    monkeypatch.setattr(unlimited, "runtime_profile", lambda *_: "windows-amd")
    engine = EngineRegistry(settings).open_engine("unlimited-ocr-2")
    assert "dedicated runtime" in engine.unavailable_reason()
    install_manifest(tmp_path)
    assert engine.unavailable_reason() is None
    runtime = engine.build_local()
    assert runtime.device == "Vulkan0" and runtime.spec == unlimited.SPEC
    assert runtime.image_max_tokens == 273
    assert not runtime.flash_attention
    assert {"--special", "--no-cache-prompt", "--no-context-shift"} <= set(runtime.extra_args)
    assert runtime.describe()["sparse_memory_optimized"] is False
    key = engine.pool_key
    monkeypatch.setattr(unlimited, "runtime_profile", lambda *_: "linux-nvidia")
    assert "must be built for CUDA" in engine.unavailable_reason()
    install_manifest(tmp_path, backend="CUDA")
    assert engine.unavailable_reason() is None
    assert engine.build_local().device == "CUDA0" and engine.pool_key != key
    settings.local_device = "cpu"
    assert engine.unavailable_reason() is None
    assert engine.build_local().device == "CPU"
    install_manifest(tmp_path, revision="wrong")
    assert unlimited.installed_runtime(settings) is None


def test_one_request_contains_all_selected_pages_and_respects_limits(tmp_path):
    asyncio.run(_parse_selected_pages(tmp_path))


async def _parse_selected_pages(tmp_path):
    settings = Settings(data_dir=tmp_path, unlimited_ocr_2_max_pages=2)
    engine = unlimited.UnlimitedOcrEngine(
        ALL_MODELS["unlimited-ocr-2"], settings, LocalModelPool(keep_loaded=False, max_loaded=1)
    )
    paths = {}
    for number in (2, 4):
        paths[number] = tmp_path / f"{number}.png"
        Image.new("RGB", (12, 16), (number * 10, 20, 30)).save(paths[number])
    calls, progress = [], []

    class Backend:
        async def complete(self, payload):
            calls.append(payload)
            return (
                "<PAGE><BLOCK>text[0,0,100,100]<FORK>second"
                "<PAGE><BLOCK>text[0,0,100,100]<FORK>fourth"
            )

    async def report(update):
        progress.append(update)

    output = await engine.parse(
        Backend(), PageSource([2, 4], paths), report=report, artifact_dir=tmp_path / "artifacts"
    )
    assert len(calls) == 1
    request = calls[0]
    parts = request["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text", "image_url", "text"]
    assert parts[0]["text"] == parts[2]["text"] == "<PAGE>"
    assert parts[-1]["text"] == "Multi page parsing."
    assert parts[1]["image_url"] != parts[3]["image_url"]
    assert request["cache_prompt"] is False and request["temperature"] == 0
    assert request["max_tokens"] <= settings.unlimited_ocr_2_context_length - 274 * 2 - 128
    assert [p.page_number for p in output.pages] == [2, 4]
    assert output.meta["processing_mode"] == "whole_document"
    assert progress == [
        {"phase": "document", "pages_done": 0},
        {"phase": "document", "pages_done": 2},
    ]
    assert (tmp_path / "artifacts/unlimited-raw.txt").is_file()
    with pytest.raises(RuntimeError, match="1–2 pages"):
        await engine.parse(
            Backend(), PageSource([1, 2, 3], {}), report=report, artifact_dir=tmp_path
        )
    assert len(calls) == 1
