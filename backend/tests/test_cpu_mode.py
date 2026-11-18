"""CPU mode must stay local, survive reruns, and never reuse GPU runtimes."""

import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, create_engine, inspect

from app import db
from app.config import Settings
from app.engines import library_worker, preset_worker, qwen35
from app.engines.backends import llama_cpp, transformers_local
from app.engines.cpu_runtime import cpu_subprocess_env
from app.engines.registry import EngineRegistry
from app.main import app


def test_cpu_registry_is_isolated_and_bypasses_remote_gpu_endpoints(monkeypatch):
    monkeypatch.setattr(qwen35, "llama_binary", lambda _: "llama-server")
    settings = Settings(
        local_inference=True,
        local_device="auto",
        local_dtype="bfloat16",
        runtime_profile="linux-nvidia",
        lightonocr_3_0_8b_endpoint_url="http://gpu-server/v1",
        rakedoc_nano_endpoint_url="http://gpu-server/v1",
    )
    registry = EngineRegistry(settings)
    cpu = registry.for_mode(True)
    assert registry.for_mode(False) is registry
    assert cpu is not registry and cpu.pool is not registry.pool
    assert cpu.settings.local_device == "cpu"
    assert cpu.settings.local_dtype == "auto"
    assert cpu.settings.local_max_batch == cpu.settings.kdl_max_concurrent == 1
    assert not cpu.settings.keep_models_loaded
    assert all(not engine.endpoint_url for engine in cpu._open.values())
    engine = cpu.open_engine("lightonocr-3-0.8b")
    assert engine.unavailable_reason() is None
    assert engine.build_local().device == "CPU"
    assert registry.open_engine("lightonocr-3-0.8b").backend_kind == "endpoint"
    assert settings.local_device == "auto" and settings.local_dtype == "bfloat16"
    assert settings.runtime_profile == "linux-nvidia"
    assert cpu.settings.llama_cloud_base_url == settings.llama_cloud_base_url


@pytest.mark.parametrize("native_bf16", [True, False])
@pytest.mark.parametrize("onednn", [True, False])
def test_cpu_precision_requires_hardware_and_kernel_support(monkeypatch, native_bf16, onednn):
    fake_torch = SimpleNamespace(
        cpu=SimpleNamespace(get_capabilities=lambda: {"avx512_bf16": native_bf16}),
        backends=SimpleNamespace(mkldnn=SimpleNamespace(is_available=lambda: onednn, enabled=True)),
        bfloat16="bf16",
        float32="fp32",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert transformers_local.resolve_dtype("auto", "cpu") == (
        "bf16" if native_bf16 and onednn else "fp32"
    )
    assert transformers_local.resolve_dtype("float32", "cpu") == "fp32"
    assert Settings(cpu_dtype="float32").for_cpu().local_dtype == "float32"


def test_eager_attention_workaround_is_only_for_windows_rocm_cpu_bf16(monkeypatch):
    fake_torch = SimpleNamespace(version=SimpleNamespace(hip="7.13"), bfloat16="bf16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(transformers_local.sys, "platform", "win32")
    options = transformers_local.cpu_model_options
    assert options("cpu", "bf16") == {"attn_implementation": "eager"}
    assert options("cpu", "fp32") == options("cuda", "bf16") == {}
    monkeypatch.setattr(transformers_local.sys, "platform", "linux")
    assert options("cpu", "bf16") == {}
    monkeypatch.setattr(transformers_local.sys, "platform", "win32")
    fake_torch.version.hip = None
    assert options("cpu", "bf16") == {}


def test_cpu_subprocess_ignores_inherited_offloading_configuration():
    env = {"LLAMA_ARG_N_GPU_LAYERS": "all", "MTMD_BACKEND_DEVICE": "Vulkan0", "PATH": "bin"}
    result = cpu_subprocess_env(env, 4)
    assert result["PATH"] == "bin" and result["OMP_NUM_THREADS"] == "4"
    assert all(
        result[key] == "-1"
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    )
    assert "LLAMA_ARG_N_GPU_LAYERS" not in result
    assert "MTMD_BACKEND_DEVICE" not in result
    assert env["LLAMA_ARG_N_GPU_LAYERS"] == "all"


def test_cpu_llama_server_disables_every_gpu_offload_and_skips_gpu_discovery(monkeypatch, tmp_path):
    import huggingface_hub

    backend = llama_cpp.LlamaCppRuntime(
        "ovisocr3", Settings(data_dir=tmp_path).for_cpu(), 32768, binary="llama-server"
    )
    find_gpu = Mock(side_effect=AssertionError("CPU mode probed a GPU"))
    monkeypatch.setattr(backend, "_find_gpu", find_gpu)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "model.gguf")
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", popen)
    client = Mock()
    client.get.return_value.status_code = 200
    monkeypatch.setattr(llama_cpp.httpx, "Client", Mock(return_value=client))
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "Vulkan0")
    try:
        backend.load()
        args = popen.call_args.args[0]
        for flag, value in (("--device", "none"), ("--gpu-layers", "0"), ("--flash-attn", "off")):
            assert args[args.index(flag) + 1] == value
        assert {"--no-mmproj-offload", "--no-kv-offload", "--no-op-offload"} <= set(args)
        assert "LLAMA_ARG_DEVICE" not in popen.call_args.kwargs["env"]
        assert backend.describe()["runtime"] == "CPU"
        find_gpu.assert_not_called()
    finally:
        backend.unload()


def test_marker_cpu_worker_overrides_gpu_and_external_server_settings(monkeypatch, tmp_path):
    class Captured(Exception):
        pass

    popen = Mock(side_effect=Captured)
    monkeypatch.setattr(library_worker.subprocess, "Popen", popen)
    monkeypatch.setattr(library_worker, "find_tool", lambda *_: "llama-server")
    for name in (
        "SURYA_INFERENCE_URL",
        "DETECTOR_SERVER_URL",
        "OCR_ERROR_SERVER_URL",
        "FAST_LAYOUT_SERVER_URL",
    ):
        monkeypatch.setenv(name, "http://external-gpu")
    runtime = library_worker.LibraryRuntime(
        "marker_balanced", Settings(data_dir=tmp_path).for_cpu()
    )
    try:
        with pytest.raises(Captured):
            runtime.load()
        env = popen.call_args.kwargs["env"]
        assert env["LLAMA_CPP_NGL"] == "0"
        assert env["LLAMA_CPP_NO_MMPROJ_OFFLOAD"] == "True"
        assert "--device none" in env["LLAMA_CPP_EXTRA_ARGS"]
        assert "--no-op-offload" in env["LLAMA_CPP_EXTRA_ARGS"]
        assert env["TORCH_DEVICE"] == env["FAST_DETECTOR_DEVICE"] == "cpu"
        assert not any(env.get(key) == "http://external-gpu" for key in env)
    finally:
        runtime.log_file.close()


def test_marker_cpu_cannot_attach_to_existing_gpu_server(monkeypatch):
    spawn = SimpleNamespace(
        _sentinel_path=lambda backend: f"/{backend}.json",
        _lock_path=lambda backend: f"/{backend}.lock",
    )
    monkeypatch.setitem(sys.modules, "surya.inference.backends", SimpleNamespace(spawn=spawn))
    monkeypatch.setenv("ARENA_CPU_ONLY", "1")
    preset_worker.configure_surya_cleanup()
    assert spawn._sentinel_path("llamacpp") != "/llamacpp.json"
    assert str(os.getpid()) in spawn._sentinel_path("fast_layout")
    assert "arena-cpu" in spawn._lock_path("detector")


def test_cpu_column_migration_preserves_existing_results_and_is_repeatable(monkeypatch):
    legacy = MetaData()
    for table in db.Base.metadata.sorted_tables:
        table.to_metadata(legacy)
    legacy.tables["experiments"]._columns.remove(legacy.tables["experiments"].c.cpu_only)
    engine = create_engine("sqlite://")
    try:
        legacy.create_all(engine)
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO experiments (id, document_id, status, pages, created_at) "
                "VALUES ('old', 'document', 'completed', '[1]', '2026-11-01 12:00:00')"
            )
        monkeypatch.setattr(db, "get_engine", lambda: engine)
        db.init_db()
        db.init_db()
        assert "cpu_only" in {c["name"] for c in inspect(engine).get_columns("experiments")}
        with engine.connect() as conn:
            assert conn.exec_driver_sql(
                "SELECT id, status, pages, cpu_only FROM experiments"
            ).one() == ("old", "completed", "[1]", 0)
    finally:
        engine.dispose()


def test_api_cpu_run_and_rerun_keep_mode_and_use_cpu_engine(monkeypatch):
    # This is a real pypdf extraction through the API/runner, with no weights or APIs.
    with TestClient(app) as client:
        cpu_models = client.get("/api/models?cpu_only=true").json()
        assert cpu_models["cpu_only"] is True
        pypdf = next(m for m in cpu_models["models"] if m["id"] == "pypdf")
        assert pypdf["availability"]["runtime_label"] == "Local CPU"
        assert pypdf["availability"]["runnable"]
        doc = client.post("/api/documents/sample").json()
        normal_engine = app.state.registry.open_engine("pypdf")
        monkeypatch.setattr(
            normal_engine, "prepare", Mock(side_effect=AssertionError("GPU registry"))
        )
        ids = []
        try:
            result = client.post(
                "/api/experiments",
                json={
                    "document_id": doc["id"],
                    "models": ["pypdf"],
                    "pages": "1",
                    "cpu_only": True,
                },
            )
            assert result.status_code == 201, result.text
            for iteration in range(2):
                experiment = result.json()
                ids.append(experiment["id"])
                assert experiment["cpu_only"] is True
                deadline = time.monotonic() + 10
                while experiment["status"] in ("queued", "running"):
                    assert time.monotonic() < deadline, experiment
                    time.sleep(0.02)
                    experiment = client.get(f"/api/experiments/{experiment['id']}").json()
                run = experiment["runs"][0]
                assert run["status"] == "completed", run
                assert run["meta"]["cpu_only"] is True and run["markdown"]
                if iteration == 0:
                    result = client.post(f"/api/experiments/{experiment['id']}/rerun")
                    assert result.status_code == 201, result.text
        finally:
            for experiment_id in ids:
                client.delete(f"/api/experiments/{experiment_id}")
