import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llamaparse_configured"] is False
    assert body["local_device"] is None


def test_models_catalog(client):
    body = client.get("/api/models").json()
    ids = [m["id"] for m in body["models"]]
    assert ids[:2] == ["rakedoc-nano", "aeddix-alpine-ocr"]
    assert len(ids) == 24
    assert "surya-ocr-2" not in ids and "teleocr" not in ids
    assert all(m["implemented"] and m["specialty"] for m in body["models"])
    assert "nemotron-3.5-parse" not in ids and "falcon-ocr" not in ids
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["nemotron-skyvision-cpu"]["name"] == "Nemotron SkyVision CPU"
    assert by_id["nemotron-skyvision-cpu"]["scores"] is None
    assert by_id["nemotron-skyvision-cpu"]["availability"]["backend"] == "cpu"
    assert by_id["docling"]["name"] == "Docling Standard"
    assert by_id["granite-docling-258m"]["name"] == "Granite Docling 258M"
    assert by_id["granite-docling-258m"]["scores"] is None
    assert by_id["granite-docling-258m"]["availability"]["runnable"] is False
    tools = (
        "pypdf",
        "tesseract",
        "docling",
        "marker_fast",
        "marker_balanced",
        "opendataloader_core",
        "opendataloader_docling",
        "doctr_compact",
        "doctr_default",
        "doctr_large_transformer",
    )
    vlms = [m for m in body["models"] if m["implemented"] and m["id"] not in tools]
    assert all(m["revision"] for m in vlms if m["id"] != "rakedoc-nano")
    # CPU tools stay available when local GPU inference is off.
    assert by_id["pypdf"]["availability"] == {
        "runnable": True,
        "backend": "cpu",
        "reason": None,
        "loaded": None,
        "runtime_label": None,
        "max_pages": None,
    }
    rakedoc = body["models"][0]
    assert rakedoc["scores"]["overall"] == pytest.approx(77.23)
    assert rakedoc["availability"]["runnable"] is False
    assert "Local inference is disabled" in rakedoc["availability"]["reason"]
    assert body["reference"]["id"] == "llamaparse-agentic-plus"
    assert body["reference"]["availability"]["reason"].startswith("Add your LLAMA_CLOUD_API_KEY")


def test_document_lifecycle_and_validation(client):
    res = client.post("/api/documents/sample")
    assert res.status_code == 201
    doc = res.json()
    assert doc["page_count"] == 2

    thumb = client.get(doc["thumbnail_url"])
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/webp"
    page = client.get(f"/api/documents/{doc['id']}/pages/2")
    assert page.status_code == 200
    assert page.headers["content-type"] == "image/png"
    assert client.get(f"/api/documents/{doc['id']}/pages/3").status_code == 404

    def create(**body):
        return client.post("/api/experiments", json={"document_id": doc["id"], **body})

    assert create(models=["nope"]).status_code == 422
    unavailable = create(models=["rakedoc-nano"])
    assert unavailable.status_code == 422
    assert "cannot run" in unavailable.json()["detail"]
    assert create(models=["rakedoc-nano"], pages="7").status_code == 422
    assert create(models=[]).status_code == 422


def test_upload_rejects_unsupported_file(client):
    res = client.post("/api/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 422
    assert "Unsupported file type" in res.json()["detail"]


def test_missing_resources(client):
    assert client.get("/api/experiments/unknown").status_code == 404
    assert client.delete("/api/experiments/unknown").status_code == 404
    assert client.get("/api/runs/unknown/files/../../.env").status_code == 404
    assert client.get("/api/experiments").json() == {"experiments": []}
