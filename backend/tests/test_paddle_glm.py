"""OCR ordering, rotated crops, output completeness and device/cache boundaries."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from app.catalog import ALL_MODELS
from app.config import Settings
from app.engines.base import PageSource
from app.engines.glm import GlmOcrEngine, decode_generation, patch_merger_as_matmul
from app.engines.open_engine import PageFailed
from app.engines.paddle import (
    PADDLE_PRESETS,
    ordered_polygons,
    text_crop,
    text_output,
    use_native_large_convs,
)
from app.engines.registry import EngineRegistry


def polygon(x, y, w=40, h=10):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_reading_order_handles_jitter_and_confidence_filtering():
    boxes = [polygon(0, 70), polygon(60, 20), polygon(0, 23)]
    assert ordered_polygons(boxes) == [boxes[2], boxes[1], boxes[0]]
    lines = [
        {"text": "First", "score": 0.95, "polygon": polygon(10, 20)},
        {"text": "Noise", "score": 0.1, "polygon": polygon(20, 40)},
        {"text": "", "score": float("nan"), "polygon": polygon(20, 50)},
        {"text": "Last", "score": 0.8, "polygon": polygon(10, 70)},
    ]
    text, boxes = text_output(lines, (100, 200))
    assert text == "First\n\nLast"
    assert boxes[0]["bbox"] == [0.1, 0.1, 0.5, 0.15]
    assert [box["layout_order"] for box in boxes] == [0, 1]
    assert text_output([], (100, 100)) == ("", [])


def test_rotated_text_is_rectified_and_rgb_preserved():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    crop = text_crop(pixels, polygon(10, 10, 20, 60))
    assert crop.size == (60, 20)
    assert crop.getpixel((30, 10)) == (255, 0, 0)


@pytest.mark.parametrize("model_id", [*PADDLE_PRESETS, "glm-ocr"])
def test_new_models_have_separate_cpu_gpu_caches_and_respect_local_inference(model_id):
    auto = EngineRegistry(Settings(local_device="auto"))
    cpu = EngineRegistry(Settings(local_device="cpu"))
    engine = auto.open_engine(model_id)
    assert engine.pool_key != cpu.open_engine(model_id).pool_key
    assert engine.build_local().device_preference == "auto"
    assert cpu.open_engine(model_id).build_local().device_preference == "cpu"
    assert "Local inference is disabled" in engine.unavailable_reason()
    assert ALL_MODELS[model_id].scores is None
    assert ALL_MODELS[model_id].implemented
    if model_id in PADDLE_PRESETS:
        assert engine.build_local().det_revision == ALL_MODELS[model_id].revision


def test_glm_does_not_present_truncated_text_as_a_complete_page():
    processor = Mock()
    processor.decode.return_value = "  # Header\n\nContent  "
    assert decode_generation(processor, [10, 20, 99, 0], {99}) == "# Header\n\nContent"
    assert processor.decode.call_args.args[0] == [10, 20]
    with pytest.raises(PageFailed, match="incomplete"):
        decode_generation(processor, [10, 20], {99})


def test_glm_merger_uses_identical_parameters_and_preserves_projection():
    torch = pytest.importorskip("torch")
    conv = torch.nn.Conv2d(3, 7, kernel_size=2, stride=2).eval()
    model = SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(downsample=conv)))
    patch_merger_as_matmul(model)
    merged = model.model.visual.downsample
    assert merged.weight is conv.weight and merged.bias is conv.bias
    for shape in [(5, 3, 2, 2), (2, 3, 4, 4)]:
        x = torch.randn(shape)
        with torch.inference_mode():
            torch.testing.assert_close(merged(x), conv(x))


def test_large_convolution_fallback_preserves_output_and_restores_backend():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 9, padding=4), torch.nn.Conv2d(3, 2, 1))
    x = torch.randn(1, 2, 12, 12)
    weights = [id(p) for p in model.parameters()]
    enabled = torch.backends.cudnn.enabled
    with torch.inference_mode():
        expected = model(x)
        assert use_native_large_convs(model) == 1
        torch.testing.assert_close(model(x), expected)
    assert [id(p) for p in model.parameters()] == weights
    assert torch.backends.cudnn.enabled == enabled


@pytest.mark.parametrize("model_id", ["paddleocr_small", "glm-ocr"])
def test_selected_page_numbers_outputs_and_artifacts_are_preserved(tmp_path, model_id):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 200)).save(image_path)
    pages = PageSource([2, 5], {2: image_path, 5: image_path})
    engine = EngineRegistry(Settings()).open_engine(model_id)
    backend = Mock()
    if isinstance(engine, GlmOcrEngine):
        backend.extract.return_value = "Recognized page"
    else:
        backend.extract.return_value = [
            {"text": "Recognized page", "score": 0.99, "polygon": polygon(10, 20)}
        ]
    progress = []

    async def report(value):
        progress.append(value)

    artifacts = tmp_path / "new-run" / "artifacts"
    result = asyncio.run(engine.parse(backend, pages, report=report, artifact_dir=artifacts))
    assert [p.page_number for p in result.pages] == [2, 5]
    assert all(p.markdown == "Recognized page" for p in result.pages)
    assert progress[-1]["pages_done"] == 2
    if model_id.startswith("paddleocr"):
        raw = json.loads((artifacts / "paddle-page-5.json").read_text())
        assert raw["lines"][0]["score"] == 0.99
        assert result.pages[0].elements[0]["content"] == "Recognized page"
    else:
        assert (artifacts / "glm-page-5.md").read_text() == "Recognized page"
        assert result.pages[0].elements == []


def test_glm_ragged_tables_preserve_all_cells_and_raw_output(tmp_path):
    from bs4 import BeautifulSoup

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100)).save(image_path)
    raw = "| Hub | Q2 |\n| --- | --- |\n| Hamburg | 412 | 428 | 95.9% |"
    engine = EngineRegistry(Settings()).open_engine("glm-ocr")
    backend = Mock()
    backend.extract.return_value = raw

    async def report(_):
        pass

    result = asyncio.run(
        engine.parse(
            backend, PageSource([1], {1: image_path}), report=report, artifact_dir=tmp_path
        )
    )
    assert (tmp_path / "glm-page-1.md").read_text() == raw
    soup = BeautifulSoup(result.markdown, "html.parser")
    assert [cell.text for cell in soup.select("tbody td")] == ["Hamburg", "412", "428", "95.9%"]
