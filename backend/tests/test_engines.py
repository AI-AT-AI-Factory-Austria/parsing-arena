"""Engine logic against fake model backends (no weights, no GPU)."""

import asyncio
import json

from PIL import Image

from app.catalog import get_model
from app.config import Settings
from app.engines.backends.transformers_local import LocalModelPool
from app.engines.base import PageSource
from app.engines.falcon import FalconOcrEngine
from app.engines.infinity import DEEP_PARSE_PROMPT, PROMPT_DOC2JSON, InfinityParser2Engine
from app.engines.lightonocr import LightOnOcr3Engine
from app.engines.ovisocr import OvisOcr3Engine
from app.engines.vendor import parsebench_ovisocr3 as ovis


def pages(tmp_path, size=(1000, 1400)) -> PageSource:
    path = tmp_path / "page-0001.png"
    Image.new("RGB", size, "white").save(path)
    return PageSource([1], {1: path})


def engine(cls, model_id: str):
    model = get_model(model_id)
    assert model is not None
    return cls(
        model, Settings(local_inference=True), LocalModelPool(keep_loaded=False, max_loaded=1)
    )


async def ignore(_update: dict) -> None:
    return None


class FakeChat:
    """Answers chat-completion payloads by prompt, recording what was asked."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts: list[str] = []
        self.payloads: list[dict] = []

    async def complete(self, payload: dict) -> str:
        prompt = payload["messages"][0]["content"][1]["text"]
        self.prompts.append(prompt)
        self.payloads.append(payload)
        return self.answer(prompt)


def run(engine_, backend, source, tmp_path):
    return asyncio.run(engine_.parse(backend, source, report=ignore, artifact_dir=tmp_path))


def test_lightonocr_engine(tmp_path):
    raw = "![title](80,70,920,140) Company report\n\n![text](80,175,920,225) Revenue rose 20%."
    chat = FakeChat(lambda _p: raw)
    out = run(engine(LightOnOcr3Engine, "lightonocr-3-0.8b"), chat, pages(tmp_path), tmp_path)
    assert out.pages[0].markdown == "# Company report\n\nRevenue rose 20%."
    assert out.pages[0].elements[0]["bbox"] == [0.08, 0.07, 0.92, 0.14]
    payload = chat.payloads[0]
    assert payload["temperature"] == 0.1 and payload["max_tokens"] == 12288
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_infinity_engine_deep_parses_figures(tmp_path):
    elements = [
        {"bbox": [0, 0, 1000, 40], "category": "header", "text": "Report"},
        {"bbox": [100, 100, 900, 500], "category": "figure", "text": ""},
        {"bbox": [0, 960, 1000, 1000], "category": "footer", "text": "Page 1"},
    ]
    chart_table = "| Month | Value |\n| --- | --- |\n| Apr | 347 |"

    def answer(prompt: str) -> str:
        return (
            "```json\n" + json.dumps(elements) + "\n```"
            if prompt == PROMPT_DOC2JSON
            else chart_table
        )

    chat = FakeChat(answer)
    out = run(
        engine(InfinityParser2Engine, "infinity-parser2-flash"), chat, pages(tmp_path), tmp_path
    )
    assert chat.prompts == [PROMPT_DOC2JSON, DEEP_PARSE_PROMPT]
    assert out.pages[0].markdown == f"Report\n\n{chart_table}\n\nPage 1"
    assert out.pages[0].elements[1]["bbox"] == [0.1, 0.1, 0.9, 0.5]


def test_ovisocr3_engine_reads_charts(tmp_path):
    structured = json.dumps(
        [
            {"bbox": [50, 40, 600, 80], "category": "doc_title", "content": "Report"},
            {
                "bbox": [50, 100, 900, 500],
                "category": "chart",
                "content": '<img src="images/bbox_50_100_900_500.jpg" />',
            },
            {"bbox": [50, 520, 900, 560], "category": "display_formula", "content": "E = mc^2"},
        ]
    )
    chart = json.dumps(
        {
            "title": "Volume",
            "source": "None",
            "x_title": "Month",
            "y_title": "k",
            "values": {"Q2": {"Apr": "347", "May": "358"}},
        }
    )
    chat = FakeChat(lambda p: structured if p == ovis.STRUCTURED_PROMPT else chart)
    out = run(engine(OvisOcr3Engine, "ovisocr3"), chat, pages(tmp_path), tmp_path)
    markdown = out.pages[0].markdown
    assert chat.prompts == [ovis.STRUCTURED_PROMPT, ovis.CHART_PROMPT]
    assert markdown.startswith("# Report")
    assert "<caption>Volume</caption>" in markdown and "<td>347</td>" in markdown
    assert "$$\nE = mc^2\n$$" in markdown
    assert [e["category"] for e in out.pages[0].elements] == ["Title", "Picture", "Formula"]


class FakeTesseract:
    def __init__(self):
        self.calls: list[tuple] = []

    def image_to_string(self, image, *, lang, config):
        self.calls.append((image.size, lang, config))
        return "Northwind Logistics\nQ3 Operations Report\n"


def test_tesseract_engine(tmp_path, monkeypatch):
    from app.engines import tools

    tesseract = engine(tools.TesseractEngine, "tesseract")
    monkeypatch.setattr(tools, "find_tesseract", lambda _configured: None)
    assert "winget install UB-Mannheim.TesseractOCR" in tesseract.unavailable_reason()

    fake = FakeTesseract()
    out = run(tesseract, fake, pages(tmp_path), tmp_path)
    assert out.pages[0].markdown.startswith("Northwind Logistics")
    assert fake.calls == [((1000, 1400), "eng", "")]
    assert tools.TesseractEngine.render.dpi == 300


class FakeFalcon:
    def regions(self, image):
        return [
            {"category": "doc_title", "bbox": [10, 14, 500, 70], "score": 0.9, "text": "# Title"},
            {
                "category": "table",
                "bbox": [10, 140, 900, 560],
                "score": 0.9,
                "text": "| a | b |\n| --- | --- |\n| 1 | 2 |",
            },
            {"category": "image", "bbox": [10, 700, 400, 1120], "score": 0.8, "text": "photo"},
            {"category": "not_a_label", "bbox": [0, 0, 5, 5], "score": 0.5, "text": "skip"},
        ]


def test_falcon_engine_assembles_regions(tmp_path):
    out = run(engine(FalconOcrEngine, "falcon-ocr"), FakeFalcon(), pages(tmp_path), tmp_path)
    page = out.pages[0]
    assert page.markdown.startswith("# Title\n\n<table>")
    assert "photo" not in page.markdown and "skip" not in page.markdown
    assert [e["category"] for e in page.elements] == ["Title", "Table", "Picture"]
    assert page.elements[0]["bbox"] == [0.01, 0.01, 0.5, 0.05]
