"""Pipeline post-processing and plumbing that needs no model weights."""

import asyncio
import json

import pytest

from app.catalog import MODELS
from app.config import Settings
from app.documents import render_page
from app.engines import infinity
from app.engines.backends.transformers_local import LocalModelPool
from app.engines.base import element
from app.engines.open_engine import settings_prefix
from app.engines.registry import ENGINES
from app.engines.vendor import parsebench_infinity2, parsebench_lightonocr3, parsebench_mineru
from tests.conftest import SAMPLE_PDF

LIGHTON_RAW = """![header](84,61,255,74) Northwind Logistics GmbH
![title](82,99,396,118) ## 4. Monthly shipment volume
![title](82,120,396,130) Outlook
![text](82,422,893,474) Volume rises by \\$x^2\\$ and costs $5 or $7.
![formula](100,500,900,540) $$U = \\lambda / \\mu$$
![table](80,600,920,700) | Month | Value |
| --- | --- |
| Apr | 347 |
"""


def test_every_implemented_model_has_an_engine_and_settings():
    settings = Settings()
    for model in MODELS:
        assert (model.id in ENGINES) == model.implemented, model.id
        if not model.implemented or not ENGINES[model.id].supports_endpoint:
            continue
        prefix = settings_prefix(model.id)
        assert hasattr(settings, f"{prefix}_endpoint_url"), prefix
        assert hasattr(settings, f"{prefix}_served_model"), prefix


def test_rocm_guard_refuses_models_that_crashed_it(monkeypatch):
    from app.engines import falcon, open_engine, tools
    from app.engines.registry import EngineRegistry

    monkeypatch.setattr(falcon, "triton_available", lambda: True)
    monkeypatch.setattr(tools, "find_tesseract", lambda _configured: "tesseract")
    monkeypatch.setattr(open_engine, "local_inference_missing", lambda: None)
    monkeypatch.setattr(
        open_engine, "describe_local_device", lambda *_: {"available": True, "runtime": "ROCm"}
    )
    implemented = [m for m in MODELS if m.implemented]
    registry = EngineRegistry(Settings(local_inference=True))
    blocked = {m.id for m in implemented if not registry.availability(m)["runnable"]}
    assert blocked == {
        "nemotron-3.5-parse",
        "lightonocr-3-0.8b",
        "infinity-parser2-flash",
        "ovisocr3",
    }

    allowed = EngineRegistry(Settings(local_inference=True, allow_unstable_rocm_models=True))
    assert all(allowed.availability(m)["runnable"] for m in implemented)

    monkeypatch.setattr(
        open_engine, "describe_local_device", lambda *_: {"available": True, "runtime": "CUDA"}
    )
    cuda = EngineRegistry(Settings(local_inference=True))
    assert all(cuda.availability(m)["runnable"] for m in implemented)


def test_lightonocr_blocks_to_markdown():
    markdown, items = parsebench_lightonocr3.normalize_page(LIGHTON_RAW)
    assert [i["category"] for i in items] == [
        "Page-header",
        "Title",
        "Title",
        "Text",
        "Formula",
        "Table",
    ]
    assert items[0]["bbox"] == [84, 61, 255, 74]
    assert "## 4. Monthly shipment volume" in markdown
    assert "# Outlook" in markdown
    assert "$x^2$" in markdown and "$5 or $7" in markdown
    assert "$$\nU = \\lambda / \\mu\n$$" in markdown
    assert "<table>" in markdown and "<td>347</td>" in markdown


def test_infinity_json_postprocessing():
    elements = [
        {"bbox": [0, 0, 500, 50], "category": "footer", "text": "Page 1"},
        {"bbox": [0, 100, 1000, 150], "category": "title", "text": "## Results"},
        {
            "bbox": [0, 200, 1000, 400],
            "category": "table",
            "text": (
                "<table><tr><td>Year</td><td>2025</td></tr><tr><td>A</td><td>1</td></tr></table>"
            ),
        },
        {"bbox": [0, 400, 1000, 450], "category": "formula", "text": "$E = mc^2$"},
        {"bbox": [0, 0, 500, 40], "category": "header", "text": "Report"},
    ]
    fenced = "```json\n" + json.dumps(elements) + "\n```"
    loaded = infinity.load_elements(fenced, width=2000, height=3000)
    assert loaded[0]["bbox"] == [0, 0, 1000, 150]

    truncated = json.dumps(elements)[:-30]
    assert len(infinity.load_elements(truncated, 1000, 1000)) == len(elements) - 1

    markdown, items = parsebench_infinity2.normalize_page(loaded)
    assert items[0]["category"] == "Page-header"
    assert items[-1]["category"] == "Page-footer"
    assert "# Results" in markdown
    assert "<th>Year</th>" in markdown
    assert "$$E = mc^2$$" in markdown


def test_infinity_smart_resize_stays_on_grid():
    height, width = infinity.smart_resize(3508, 2480)
    assert height % 32 == 0 and width % 32 == 0
    assert height * width <= infinity.MAX_PIXELS
    assert infinity.smart_resize(10, 10) == (64, 64)


def test_mineru_markdown_normalization():
    md = "Intro\n\n<table><tr><td>Hub</td><td>Q2</td></tr><tr><td>Graz</td><td>4"
    out = parsebench_mineru.normalize_markdown(md)
    assert out.count("<table>") == out.count("</table>") == 1
    assert "<thead><tr><th>Hub</th><th>Q2</th></tr></thead>" in out


def test_element_clamps_boxes():
    el = element("Text", [-0.1, 0.2, 1.4, 0.5], "x", 3)
    assert el["bbox"] == [0.0, 0.2, 1.0, 0.5]
    assert el["layout_order"] == 3


def test_render_page_respects_pixel_cap():
    image = render_page(SAMPLE_PDF, ".pdf", 1, dpi=400, max_pixels=1_000_000)
    assert image.width * image.height <= 1_000_000
    full = render_page(SAMPLE_PDF, ".pdf", 1, dpi=72)
    assert full.width < 700


class FakeModel:
    def __init__(self, name: str, log: list[str]):
        self.name = name
        self.log = log
        self.loaded = False

    def load(self) -> None:
        self.loaded = True
        self.log.append(f"load {self.name}")

    def unload(self) -> None:
        self.loaded = False
        self.log.append(f"unload {self.name}")

    def describe(self) -> dict:
        return {}


@pytest.mark.parametrize(("keep_loaded", "max_loaded"), [(True, 2), (False, 2)])
def test_pool_bounds_loaded_models(keep_loaded, max_loaded):
    log: list[str] = []
    pool = LocalModelPool(keep_loaded=keep_loaded, max_loaded=max_loaded)

    async def scenario():
        for name in ["a", "b", "a", "c"]:
            model, _ = await pool.acquire(name, lambda n=name: FakeModel(n, log))
            assert model.loaded
            await pool.release(name)

    asyncio.run(scenario())
    if keep_loaded:
        assert log == ["load a", "load b", "unload b", "load c"]
        assert pool.is_loaded("a") and pool.is_loaded("c") and not pool.is_loaded("b")
    else:
        assert log == [
            "load a",
            "unload a",
            "load b",
            "unload b",
            "load a",
            "unload a",
            "load c",
            "unload c",
        ]
