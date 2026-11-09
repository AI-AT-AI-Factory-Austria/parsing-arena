import asyncio
import json
from typing import ClassVar

import httpx
import pytest
from PIL import Image

from app.catalog import ALL_MODELS
from app.config import Settings
from app.engines import ovis_crosspage as X
from app.engines.backends.openai_compat import OpenAICompatBackend
from app.engines.backends.transformers_local import LocalModelPool
from app.engines.base import PageSource
from app.engines.ovisocr import OvisOcr3Engine


def table(*rows):
    return "<table>" + "".join(f"<tr><td>{row}</td></tr>" for row in rows) + "</table>"


def page(content, category="table"):
    return json.dumps(
        [
            {"bbox": [10, 10, 900, 40], "category": "text", "content": "Keep this text."},
            {"bbox": [10, 50, 900, 900], "category": category, "content": content},
        ]
    )


def merged(*rows):
    return json.dumps([{"source_pages": [1, 2], "html": table(*rows)}])


class Backend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, payload):
        self.calls.append(payload)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class Renderer:
    calls: ClassVar[list] = []

    def __init__(self, executable):
        pass

    def render(self, markdown, owner, images, folder, name):
        self.calls.append((owner, markdown))
        return X.C.RenderResult(images[owner - 1], markdown, {}, {})

    def close(self):
        pass


def run(tmp_path, responses, numbers=(1, 2), enabled=True):
    images = {}
    for number in numbers:
        images[number] = tmp_path / f"{number}.png"
        Image.new("RGB", (100, 100), "white").save(images[number])
    backend = Backend(responses)
    engine = OvisOcr3Engine(
        ALL_MODELS["ovisocr3"],
        Settings(ovisocr3_crosspage_tables=enabled),
        LocalModelPool(keep_loaded=False, max_loaded=1),
    )
    progress = []

    async def report(update):
        progress.append(update)

    output = asyncio.run(
        engine.parse(
            backend, PageSource(list(numbers), images), report=report, artifact_dir=tmp_path / "run"
        )
    )
    return output, backend, progress


def test_two_pages_join_only_document_and_preserve_page_judging(tmp_path):
    output, backend, progress = run(
        tmp_path,
        [page(table("Alpha")), page(table("Beta")), merged("Alpha", "Beta")],
        numbers=(4, 5),
    )
    assert output.markdown.count("<table>") == 1
    assert "Alpha" in output.markdown and "Beta" in output.markdown
    assert output.markdown.count("Keep this text.") == 2
    assert "Beta" not in output.pages[0].markdown
    assert "Alpha" not in output.pages[1].markdown
    assert all(p.elements and p.seconds is not None for p in output.pages)
    assert output.meta["cross_page_tables"][0]["source_pages"] == [4, 5]
    payload = backend.calls[-1]
    assert len([p for p in payload["messages"][0]["content"] if p["type"] == "image_url"]) == 2
    assert '<INPUT_PAGE index="1">' in payload["messages"][0]["content"][-1]["text"]
    assert payload["arena_require_complete"] is True
    assert any(p.get("phase") == "reconstructing" for p in progress)


def test_three_page_chain_rerenders_and_retains_source_provenance(tmp_path, monkeypatch):
    Renderer.calls = []
    monkeypatch.setattr(X, "PageRenderer", Renderer)
    output, backend, _ = run(
        tmp_path,
        [
            page(table("A")),
            page(table("B")),
            page(table("C")),
            merged("B", "C"),
            merged("A", "B", "C"),
        ],
        numbers=(5, 6, 7),
    )
    assert output.markdown.count("<table>") == 1
    assert output.meta["cross_page_tables"] == [
        {"source_pages": [5, 6, 7], "html": table("A", "B", "C")}
    ]
    assert Renderer.calls == [(2, "Keep this text.\n\n" + table("B", "C"))]
    assert table("B", "C") in backend.calls[-1]["messages"][0]["content"][-1]["text"]


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "[]",
        merged("A")[:-2],
        '[{"source_pages":[1,3],"html":"<table></table>"}]',
        '[{"source_pages":[1,2],"html":"<table><tr><td>broken"}]',
        json.dumps([{"source_pages": [1, 2], "html": table("X")}] * 2),
        RuntimeError("output limit reached"),
    ],
)
def test_bad_reconstruction_keeps_extractions_and_reports_partial(tmp_path, response):
    output, _, _ = run(tmp_path, [page(table("A")), page(table("B")), response])
    assert output.markdown.count("<table>") == 2
    assert output.meta["cross_page_tables"] == []
    assert output.meta["cross_page_reconstruction"]["status"] == "partial"
    assert output.meta["cross_page_reconstruction"]["issues"][0]["source_pages"] == [1, 2]


def test_negative_prediction_does_not_join_unrelated_tables(tmp_path):
    negative = json.dumps([{"source_pages": [], "html": X.C.NEGATIVE}])
    output, _, _ = run(tmp_path, [page(table("A")), page(table("B")), negative])
    assert output.markdown.count("<table>") == 2
    assert output.meta["cross_page_reconstruction"]["pairs_checked"] == 1
    assert output.meta["cross_page_reconstruction"]["issues"] == []


def test_figures_in_joined_tables_are_materialized_with_page_provenance(tmp_path):
    figure = '<img src="page_2/images/bbox_10_50_900_900.jpg" />'
    output, _, _ = run(tmp_path, [page(table("A")), page(table("B")), merged("A", figure)])
    assert 'src="artifacts/crosspage/page_2/images/' in output.markdown
    assert (tmp_path / "run/artifacts/crosspage/page_2/images/bbox_10_50_900_900.jpg").is_file()
    assert output.meta["cross_page_reconstruction"]["issues"] == []


def test_other_tables_and_chart_data_survive_document_merge(tmp_path):
    first = [
        *json.loads(page(table("Unrelated"))),
        {"bbox": [10, 500, 900, 900], "category": "table", "content": table("A")},
        {
            "bbox": [10, 50, 900, 900],
            "category": "chart",
            "content": '<img src="images/bbox_10_50_900_900.jpg" />',
        },
    ]
    output, _, _ = run(
        tmp_path,
        [json.dumps(first), '{"values":{"chart-value":"42"}}', page(table("B")), merged("A", "B")],
    )
    assert "Unrelated" in output.markdown
    assert "chart-value" in output.markdown
    assert output.markdown.count("<table>") == 3


def test_nonconsecutive_pages_are_never_joined(tmp_path):
    output, backend, _ = run(tmp_path, [page(table("A")), page(table("B"))], numbers=(2, 4))
    assert len(backend.calls) == 2
    assert output.meta["cross_page_reconstruction"]["status"] == "not_needed"


def test_disabled_reconstruction_keeps_legacy_pipeline(tmp_path):
    output, backend, _ = run(tmp_path, [page(table("A")), page(table("B"))], enabled=False)
    assert len(backend.calls) == 2
    assert output.meta["cross_page_reconstruction"]["status"] == "disabled"


def test_chart_data_tables_are_not_continuation_candidates(tmp_path):
    chart = '<img src="images/bbox_10_50_900_900.jpg" />'
    chart_json = '{"title":"Chart","values":{"x":"12"}}'
    output, backend, _ = run(
        tmp_path, [page(chart, "chart"), chart_json, page(chart, "chart"), chart_json]
    )
    assert len(backend.calls) == 4
    assert "12" in output.markdown
    assert output.meta["cross_page_reconstruction"]["pairs_total"] == 0


def test_truncated_page_is_not_used_for_reconstruction(tmp_path):
    output, backend, _ = run(tmp_path, [page(table("A"))[:-1], page(table("B"))])
    assert len(backend.calls) == 2
    assert output.meta["cross_page_tables"] == []


def test_endpoint_requires_stop_and_does_not_send_internal_flag():
    async def check():
        def handle(request):
            assert "arena_require_complete" not in json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "length", "message": {"content": merged("A", "B")}}
                    ]
                },
            )

        backend = OpenAICompatBackend("http://test")
        await backend._client.aclose()
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            with pytest.raises(RuntimeError, match="did not finish"):
                await backend.complete({"arena_require_complete": True})
        finally:
            await backend.aclose()

    asyncio.run(check())
