"""Hardware dispatch, safe profile selection and local Vulkan request handling."""

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from app import runtime
from app.catalog import get_model
from app.config import Settings
from app.engines import falcon, open_engine, qwen35
from app.engines.backends import transformers_local
from app.engines.backends.llama_cpp import LlamaCppRuntime
from app.engines.backends.transformers_local import LocalModelPool, LocalVisionModel
from app.engines.registry import EngineRegistry


def test_cpu_override_does_not_hide_gpu_from_profile_detection(monkeypatch):
    torch = SimpleNamespace(
        __version__="test-rocm",
        version=SimpleNamespace(hip="7"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(name="AMD GPU", total_memory=8e9),
        ),
    )
    monkeypatch.setitem(transformers_local.sys.modules, "torch", torch)
    monkeypatch.setattr(transformers_local, "_device_info", {})
    monkeypatch.setattr(transformers_local, "local_inference_missing", lambda: None)
    assert transformers_local.describe_local_device("cpu")["runtime"] == "CPU"
    assert transformers_local.describe_local_device("auto")["runtime"] == "ROCm"
    # Run metadata must not be able to mutate the cached hardware description.
    info = transformers_local.describe_local_device("auto")
    info["runtime"] = "wrong"
    assert transformers_local.describe_local_device("auto")["runtime"] == "ROCm"


@pytest.mark.parametrize(
    ("platform", "device", "expected"),
    [
        ("win32", "ROCm", "windows-amd"),
        ("linux", "CUDA", "linux-nvidia"),
        ("linux", "ROCm", "native"),
        ("win32", "CPU", "native"),
    ],
)
def test_auto_profile_detects_real_platform_and_accelerator(
    monkeypatch, platform, device, expected
):
    monkeypatch.setattr(runtime.sys, "platform", platform)
    monkeypatch.setattr(runtime, "describe_local_device", lambda *_: {"runtime": device})
    assert runtime.runtime_profile(Settings(runtime_profile="auto")) == expected


def test_explicit_profile_cannot_bypass_hardware_checks(monkeypatch):
    monkeypatch.setattr(runtime, "detected_profile", lambda: "windows-amd")
    registry = EngineRegistry(Settings(local_inference=True, runtime_profile="linux-nvidia"))
    for model_id in ["lightonocr-3-0.8b", "ovisocr3", "unlimited-ocr-2"]:
        reason = registry.open_engine(model_id).unavailable_reason()
        assert "does not match this machine" in reason
    with pytest.raises(ValidationError):
        Settings(runtime_profile="pretend-cuda")


@pytest.mark.parametrize("model_id", list(qwen35.GGUF_MODELS))
def test_windows_amd_uses_pinned_vulkan_but_linux_nvidia_uses_native(monkeypatch, model_id):
    monkeypatch.setattr(runtime, "detected_profile", lambda: "windows-amd")
    monkeypatch.setattr(qwen35, "llama_binary", lambda *_: "llama-server.exe")
    settings = Settings(local_inference=True, runtime_profile="auto")
    engine = EngineRegistry(settings).open_engine(model_id)
    assert engine.unavailable_reason() is None
    assert engine.runtime_label == "AMD Vulkan · F16 GGUF"
    vulkan_key = engine.pool_key
    backend = engine.build_local()
    assert isinstance(backend, LlamaCppRuntime)
    assert len(backend.spec.revision) == 40
    assert "16" in backend.spec.model_file.lower()
    monkeypatch.setattr(runtime, "detected_profile", lambda: "linux-nvidia")
    assert isinstance(engine.build_local(), LocalVisionModel)
    assert engine.pool_key != vulkan_key


def test_remote_endpoint_takes_precedence_over_local_profile(monkeypatch):
    monkeypatch.setattr(runtime, "detected_profile", lambda: "native")
    settings = Settings(
        local_inference=False,
        runtime_profile="linux-nvidia",
        lightonocr_3_0_8b_endpoint_url="http://nvidia-host:8000/v1",
    )
    engine = EngineRegistry(settings).open_engine("lightonocr-3-0.8b")
    assert engine.backend_kind == "endpoint"
    assert engine.unavailable_reason() is None
    assert not engine.uses_vulkan


def test_missing_vulkan_binary_stays_disabled(monkeypatch):
    monkeypatch.setattr(runtime, "detected_profile", lambda: "windows-amd")
    monkeypatch.setattr(qwen35, "llama_binary", lambda *_: None)
    engine = EngineRegistry(Settings(local_inference=True)).open_engine("ovisocr3")
    assert "Install the Vulkan llama-server" in engine.unavailable_reason()


def test_falcon_requires_linux_cuda_triton_and_compiler(monkeypatch):
    monkeypatch.setattr(runtime, "detected_profile", lambda: "native")
    monkeypatch.setattr(open_engine, "local_inference_missing", lambda: None)
    monkeypatch.setattr(falcon, "describe_local_device", lambda *_: {"runtime": "CUDA"})
    monkeypatch.setattr(falcon, "triton_available", lambda: True)
    monkeypatch.setattr(falcon.shutil, "which", lambda *_: "/usr/bin/gcc")
    engine = falcon.FalconOcrEngine(
        get_model("falcon-ocr"),
        Settings(local_inference=True),
        LocalModelPool(keep_loaded=False, max_loaded=1),
    )
    monkeypatch.setattr(falcon.sys, "platform", "win32")
    assert "Linux" in engine.unavailable_reason()
    monkeypatch.setattr(falcon.sys, "platform", "linux")
    assert engine.unavailable_reason() is None
    monkeypatch.setattr(falcon, "triton_available", lambda: False)
    assert "Triton is missing" in engine.unavailable_reason()
    monkeypatch.setattr(falcon, "triton_available", lambda: True)
    monkeypatch.setattr(falcon.shutil, "which", lambda *_: None)
    assert "C compiler" in engine.unavailable_reason()


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_vulkan_preserves_request_and_refuses_truncated_output(finish_reason):
    backend = LlamaCppRuntime("ovisocr3", Settings(), 32768)
    received = []

    def respond(request):
        received.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {"content": "extracted page"},
                    }
                ]
            },
        )

    backend.client = httpx.Client(
        base_url="http://127.0.0.1:12345", transport=httpx.MockTransport(respond)
    )
    backend.process = SimpleNamespace(poll=lambda: 0)
    # A dead process must fail before sending an HTTP request.
    with pytest.raises(RuntimeError, match="not running"):
        backend._complete({})
    backend.process = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda **_: 0)
    payload = {
        "messages": [{"role": "user", "content": "OCR prompt"}],
        "temperature": 0,
        "max_tokens": 16384,
    }
    try:
        if finish_reason == "length":
            with pytest.raises(RuntimeError, match="output limit"):
                backend._complete(payload)
        else:
            assert backend._complete(payload) == "extracted page"
        assert received[0]["messages"] == payload["messages"]
        assert received[0]["max_tokens"] == 16384
        assert received[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert received[0]["top_k"] == 0 and received[0]["repeat_penalty"] == 1.0
    finally:
        backend.unload()
    assert not backend.loaded and backend.client is None and backend.process is None
