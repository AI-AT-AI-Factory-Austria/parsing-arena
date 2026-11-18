from types import SimpleNamespace

import pytest

from app.api.experiments import build_experiment
from app.config import Settings
from app.engines import execution
from app.engines.registry import EngineRegistry
from app.engines.unlimited import LLAMA_VERSION, SPEC, installed_runtime


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        ("ROCm", "Local AMD GPU · ROCm"),
        ("CUDA", "Local NVIDIA GPU · CUDA"),
        ("MPS", "Local Apple GPU · Metal"),
        ("CPU", "Local CPU"),
    ],
)
def test_pytorch_labels_reflect_detected_hardware(monkeypatch, runtime, expected):
    monkeypatch.setattr(
        execution,
        "describe_local_device",
        lambda _: {
            "available": True,
            "runtime": runtime,
            "device_name": "test device",
        },
    )
    engine = SimpleNamespace(backend_kind="local")
    label, detail = execution.execution_info("glm-ocr", engine, Settings(_env_file=None), True)
    assert label == expected and detail == "PyTorch · test device"


def test_vulkan_and_cpu_are_not_inferred_from_the_torch_gpu(monkeypatch):
    def no_torch(_):
        raise AssertionError("These runtimes do not use PyTorch's selected GPU")

    monkeypatch.setattr(execution, "describe_local_device", no_torch)
    settings = Settings(_env_file=None)
    engine = SimpleNamespace(backend_kind="local", uses_vulkan=True)
    assert (
        execution.execution_info("ovisocr3", engine, settings, True)[0] == "Local AMD GPU · Vulkan"
    )
    assert execution.execution_info("ovisocr3", engine, settings.for_cpu(), True)[0] == "Local CPU"
    engine.backend_kind = "endpoint"
    assert execution.execution_info("ovisocr3", engine, settings, True)[0] == "Remote model server"


def test_marker_balanced_explains_its_mixed_runtime(monkeypatch):
    monkeypatch.setattr(
        execution,
        "describe_local_device",
        lambda _: {
            "available": True,
            "runtime": "ROCm",
        },
    )
    label, detail = execution.execution_info(
        "marker_balanced", SimpleNamespace(backend_kind="local"), Settings(_env_file=None), True
    )
    assert label == "Local AMD GPU · Vulkan"
    assert "auxiliary PyTorch models run on CPU" in detail


def test_unknown_hardware_does_not_claim_cpu(monkeypatch):
    monkeypatch.setattr(execution, "describe_local_device", lambda _: {"available": False})
    assert (
        execution.execution_info(
            "glm-ocr", SimpleNamespace(backend_kind="local"), Settings(_env_file=None), True
        )[0]
        == "Local runtime"
    )


def test_cpu_installation_records_cpu_even_if_client_omits_flag():
    registry = EngineRegistry(Settings(_env_file=None, local_device="cpu", local_inference=False))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(registry=registry)))
    experiment = build_experiment(request, SimpleNamespace(id="test-doc"), ["pypdf"], [1])
    assert experiment.cpu_only is True


def test_readonly_runtimes_can_live_outside_persistent_data(tmp_path):
    import json

    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", runtime_dir=tmp_path / "binaries"
    )
    root = settings.runtime_root / "unlimited-ocr2"
    root.mkdir(parents=True)
    (root / "llama-server").touch()
    (root / "build-info.json").write_text(
        json.dumps(
            {
                "patch_revision": SPEC.revision,
                "llama_version": LLAMA_VERSION,
                "binary": "llama-server",
                "backend": "CPU",
            }
        )
    )
    assert installed_runtime(settings)[0] == root / "llama-server"
    assert not settings.data_dir.exists()
