import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.catalog import ALL_MODELS
from app.config import Settings
from app.engines import skyvision
from app.engines.open_engine import PageFailed
from app.engines.registry import EngineRegistry

MODEL_ID = "nemotron-skyvision-cpu"


def test_skyvision_maps_document_preserving_tables_and_reading_order():
    markdown = "# Report\n\n<table><tr><td>A</td></tr></table>"
    result, elements = skyvision.document_output(
        {
            "markdown": markdown,
            "blocks": [
                {"class": "Title", "bbox": [0.1, 0.2, 0.8, 0.3], "content": "Report"},
                {"class": "Table", "bbox": [1.2, 0.8, -0.1, 0.4], "text": "A"},
                {"class": "Text", "bbox": [0, float("nan"), 1, 1]},
                {"class": "Text", "bbox": [0, 0, 1]},
            ],
        }
    )
    assert result == markdown
    assert [e["category"] for e in elements] == ["Title", "Table"]
    assert elements[1]["bbox"] == [0.0, 0.4, 1.0, 0.8]
    assert elements[1]["content"] == "A"
    assert [e["layout_order"] for e in elements] == [0, 1]


@pytest.mark.parametrize(
    "result", [{}, {"markdown": "", "blocks": []}, {"markdown": 3, "blocks": []}]
)
def test_invalid_or_empty_output_is_not_success(result):
    with pytest.raises(PageFailed):
        skyvision.document_output(result)


def test_cpu_option_is_separate_and_requires_installed_runtime(monkeypatch):
    monkeypatch.setattr(skyvision, "skyvision_binary", lambda *_: None)
    registry = EngineRegistry(Settings(local_inference=False))
    model = ALL_MODELS[MODEL_ID]
    assert not registry.availability(model)["runnable"]
    monkeypatch.setattr(skyvision, "skyvision_binary", lambda *_: "vqa-cli")
    monkeypatch.setattr(skyvision, "build_info", lambda *_: {})
    assert not registry.availability(model)["runnable"]
    monkeypatch.setattr(skyvision, "build_info", lambda *_: {"strict_token_limit": True})
    availability = registry.availability(model)
    assert availability["runnable"] and availability["backend"] == "cpu"
    assert availability["runtime_label"] == "CPU · F16 GGUF"
    assert model.scores is None
    assert registry.open_engine(MODEL_ID) is not None
    assert registry.open_engine("nemotron-3.5-parse") is None


def test_tool_error_propagates_instead_of_accepting_partial_page(monkeypatch):
    runtime = skyvision.SkyVisionRuntime(ALL_MODELS[MODEL_ID], Settings())
    monkeypatch.setattr(
        runtime,
        "_rpc",
        lambda *_: {
            "isError": True,
            "content": [{"type": "text", "text": "output token limit reached"}],
        },
    )
    with pytest.raises(RuntimeError, match="output token limit"):
        runtime._tool("parse_document", {})


def test_parse_always_requests_cpu_full_page_json(monkeypatch):
    runtime = skyvision.SkyVisionRuntime(ALL_MODELS[MODEL_ID], Settings())
    runtime.ready = True
    runtime.process = SimpleNamespace(poll=lambda: None)
    requests = []
    monkeypatch.setattr(
        runtime,
        "_tool",
        lambda name, args: (
            requests.append((name, args))
            or {
                "raw_out": "raw model tokens",
                "markdown": "broken upstream table",
            }
        ),
    )
    normalized_inputs = []
    runtime.postprocess = lambda raw: normalized_inputs.append(raw) or {"markdown": "fixed table"}
    result = runtime.parse(Path("page.png"))
    assert result["markdown"] == "fixed table"
    assert result["skyvision_raw_out"] == "raw model tokens"
    assert normalized_inputs == ["raw model tokens"]
    name, args = requests[0]
    assert name == "parse_document"
    assert args["gpu"] is False
    assert args["format"] == "json" and args["max_tokens"] == 8744
    assert args["temperature"] == 0


def test_timeout_stops_worker_so_late_results_cannot_be_reused():
    runtime = skyvision.SkyVisionRuntime(ALL_MODELS[MODEL_ID], Settings(skyvision_timeout_s=0.001))
    runtime.ready = True
    waited = []
    runtime.process = SimpleNamespace(
        poll=lambda: None,
        stdin=io.StringIO(),
        stdout=io.StringIO(),
        wait=lambda **_: waited.append(True),
    )
    with pytest.raises(TimeoutError, match="worker has been stopped"):
        runtime._rpc("tools/call", {})
    assert waited and runtime.process is None and not runtime.loaded


def test_rpc_checks_response_id():
    runtime = skyvision.SkyVisionRuntime(ALL_MODELS[MODEL_ID], Settings())
    stdin = io.StringIO()
    runtime.process = SimpleNamespace(
        poll=lambda: None,
        stdin=stdin,
        stdout=io.StringIO(),
        wait=lambda **_: None,
    )
    runtime.responses.put({"id": 1, "result": {"ok": True}})
    assert runtime._rpc("initialize", {}) == {"ok": True}
    assert json.loads(stdin.getvalue())["method"] == "initialize"
    runtime.responses.put({"id": 1, "result": {"stale": True}})
    with pytest.raises(RuntimeError, match="Unexpected response"):
        runtime._rpc("tools/call", {})
    assert runtime.process is None
