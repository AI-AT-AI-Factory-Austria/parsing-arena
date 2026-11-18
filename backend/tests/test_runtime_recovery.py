"""Saved results remain accessible when optional native libraries cannot load."""

import builtins
import importlib.util
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import documents, samples, storage
from app.config import Settings
from app.engines.backends import transformers_local
from app.engines.registry import EngineRegistry
from app.main import app


def block_import(monkeypatch, name):
    original = builtins.__import__

    def guarded(module, *args, **kwargs):
        if module == name:
            raise OSError("Application control blocked a native dependency")
        return original(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_documents_import_without_pdfium(monkeypatch):
    block_import(monkeypatch, "pypdfium2")
    spec = importlib.util.spec_from_file_location("app._recovery_documents", documents.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_page_selection("1", 2) == [1]


def test_cached_pdf_preview_does_not_load_pdfium(monkeypatch):
    block_import(monkeypatch, "pypdfium2")
    document = SimpleNamespace(id="cached-preview", extension=".pdf")
    target = storage.page_image_path(document.id, 1, 144, None)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"already-rendered-preview")
    assert documents.page_image(document, 1, dpi=144) == target


def test_pdfium_load_failure_is_a_server_dependency_error(monkeypatch, tmp_path):
    block_import(monkeypatch, "pypdfium2")
    with pytest.raises(documents.DocumentRuntimeUnavailable, match="PDF rendering library"):
        documents.count_pages(tmp_path / "test.pdf", ".pdf")


def test_upload_reports_unavailable_renderer_as_503(monkeypatch):
    block_import(monkeypatch, "pypdfium2")
    with TestClient(app) as client:
        result = client.post(
            "/api/documents", files={"file": ("test.pdf", b"%PDF-1.4\n", "application/pdf")}
        )
        assert result.status_code == 503
        assert "PDF rendering library" in result.json()["detail"]
        assert client.get("/api/health").status_code == 200


def test_unrenderable_new_sample_does_not_break_gallery(monkeypatch, tmp_path):
    block_import(monkeypatch, "pypdfium2")
    folder = tmp_path / "sample-docs"
    folder.mkdir()
    (folder / "new.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        samples, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path, max_upload_mb=100)
    )
    with TestClient(app) as client:
        result = client.get("/api/samples")
        assert result.status_code == 200
        assert result.json() == {"samples": []}


def test_blocked_torch_disables_models_without_breaking_catalog(monkeypatch):
    block_import(monkeypatch, "torch")
    monkeypatch.setattr(transformers_local, "_device_info", {})
    monkeypatch.setattr(transformers_local, "_torch_import_error", None)
    # This regression covers a present-but-unloadable runtime, independently of
    # whether this test environment has the optional packages installed.
    original_missing = transformers_local.local_inference_missing
    monkeypatch.setattr(transformers_local, "local_inference_missing", lambda: None)
    info = transformers_local.describe_local_device()
    assert info["available"] is False
    assert "could not be loaded" in info["detail"]
    assert original_missing() == info["detail"]
    info["detail"] = "modified caller copy"
    assert transformers_local.describe_local_device()["detail"] != info["detail"]

    registry = EngineRegistry(Settings(local_inference=True, runtime_profile="auto"))
    assert "could not be loaded" in registry.open_engine("rakedoc-nano").unavailable_reason()
    assert registry.open_engine("pypdf").unavailable_reason() is None
