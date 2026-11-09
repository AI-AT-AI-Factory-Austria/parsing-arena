"""Device selection regressions without downloading or executing model weights."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config import Settings
from app.engines.doctr import DoctrRuntime
from app.engines.registry import EngineRegistry


@pytest.mark.parametrize(
    "preference,device,runtime",
    [
        ("auto", "cuda", "ROCm"),
        ("cuda", "cuda", "ROCm"),
        ("cpu", "cpu", "CPU"),
    ],
)
def test_rocm_is_selected_automatically_but_explicit_cpu_is_respected(
    monkeypatch,
    preference,
    device,
    runtime,
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "AMD Radeon 8060S")
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.version, "hip", "7.13")
    monkeypatch.setattr(torch, "set_num_threads", Mock())
    predictor = Mock()
    predictor.to.return_value = predictor
    predictor.eval.return_value = predictor
    predictor.modules.return_value = []
    factory = Mock(return_value=predictor)
    monkeypatch.setitem(sys.modules, "doctr.models", SimpleNamespace(ocr_predictor=factory))

    model = DoctrRuntime("doctr_large_transformer", device=preference)
    model.load()

    predictor.to.assert_called_once_with(device)
    predictor.cpu.assert_not_called()
    assert model.describe()["runtime"] == runtime
    assert model.describe()["device"] == device
    assert model.describe()["dtype"] == "float32"
    assert factory.call_args.kwargs == {
        "det_arch": "db_resnet50",
        "reco_arch": "vitstr_base",
        "pretrained": True,
        "detect_layout": False,
        "detect_tables": False,
    }
    if device == "cuda":
        assert model.describe()["device_name"] == "AMD Radeon 8060S"
        sync.assert_called_once_with("cuda")
    else:
        assert model.describe()["device_name"] == "CPU"
        sync.assert_not_called()


def test_cpu_and_gpu_presets_cannot_share_a_cached_runtime():
    auto = EngineRegistry(Settings(local_device="auto")).open_engine("doctr_compact")
    cpu = EngineRegistry(Settings(local_device="cpu")).open_engine("doctr_compact")
    assert auto.pool_key != cpu.pool_key
    assert auto.backend_kind == "local"
    assert auto.build_local().device_preference == "auto"
    assert cpu.build_local().device_preference == "cpu"


def test_native_layer_dispatch_preserves_outputs_weights_and_backend_flags():
    torch = pytest.importorskip("torch")
    from app.engines.doctr import use_native_rocm_layers

    torch.manual_seed(9)
    bn = torch.nn.BatchNorm2d(3).eval()
    lstm = torch.nn.LSTM(4, 8, batch_first=True).eval()
    models = torch.nn.ModuleList([bn, lstm])
    image, words = torch.randn(2, 3, 5, 5), torch.randn(2, 5, 4)
    params = [id(p) for p in models.parameters()]
    enabled = torch.backends.cudnn.enabled
    with torch.inference_mode():
        expected_bn, expected_lstm = bn(image), lstm(words)
        assert use_native_rocm_layers(models) == {"batchnorm": 1, "lstm": 1}
        torch.testing.assert_close(bn(image), expected_bn)
        actual_lstm = lstm(words)
        torch.testing.assert_close(actual_lstm[0], expected_lstm[0])
        torch.testing.assert_close(actual_lstm[1], expected_lstm[1])
    assert [id(p) for p in models.parameters()] == params
    assert torch.backends.cudnn.enabled == enabled
