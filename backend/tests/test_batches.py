import io
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from sqlalchemy import MetaData, create_engine, inspect

from app import db
from app.batches import aggregate_models
from app.judge import RUBRIC_VERSION
from app.main import app
from tests.conftest import SAMPLE_PDF


def pdf_bytes(pages: int, title: str) -> bytes:
    reader = PdfReader(SAMPLE_PDF)
    writer = PdfWriter()
    for i in range(pages):
        writer.add_page(reader.pages[i % len(reader.pages)])
    writer.add_metadata({"/Title": title})
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def upload(client, pages=3, title="first"):
    response = client.post("/api/documents", files={
        "file": (f"{title}.pdf", pdf_bytes(pages, title), "application/pdf")
    })
    assert response.status_code == 201, response.text
    return response.json()


def finished(client, batch):
    deadline = time.monotonic() + 15
    while batch["status"] in ("queued", "running"):
        assert time.monotonic() < deadline, batch
        time.sleep(0.03)
        batch = client.get(f"/api/batches/{batch['id']}").json()
    return batch


def test_batch_parses_separate_documents_preserves_pages_and_reruns():
    with TestClient(app) as client:
        a, b = upload(client), upload(client, 2, "second")
        response = client.post("/api/batches", json={
            "documents": [
                {"document_id": a["id"], "pages": [3, 1]},
                {"document_id": b["id"], "pages": [2]},
            ], "models": ["pypdf"], "cpu_only": True,
        })
        assert response.status_code == 201, response.text
        batch = finished(client, response.json())
        assert batch["status"] == "completed" and batch["cpu_only"]
        assert batch["page_count"] == 3 and len(batch["experiments"]) == 2
        for experiment, doc, pages in zip(batch["experiments"], [a, b], [[1, 3], [2]], strict=True):
            assert experiment["document"]["id"] == doc["id"]
            assert experiment["batch_id"] == batch["id"]
            assert experiment["pages"] == pages and experiment["cpu_only"]
            assert [p["page_number"] for p in experiment["runs"][0]["pages"]] == pages
            assert experiment["runs"][0]["markdown"]
        row = batch["aggregates"][0]
        assert row["pages_completed"] == row["timed_pages"] == row["pages_total"] == 3
        assert row["pass_rate"] is None and row["judged_pages"] == 0
        assert row["seconds_per_page"] == pytest.approx(row["parse_seconds"] / 3, abs=0.001)
        assert client.get(f"/api/documents/{a['id']}/pages/3?thumbnail=true").status_code == 200
        history = client.get("/api/experiments?standalone_only=true").json()["experiments"]
        assert not any(e["batch_id"] == batch["id"] for e in history)
        assert any(x["id"] == batch["id"] for x in client.get("/api/batches").json()["batches"])
        assert client.delete(f"/api/experiments/{batch['experiments'][0]['id']}").status_code == 409

        response = client.post(f"/api/batches/{batch['id']}/rerun")
        assert response.status_code == 201, response.text
        rerun = finished(client, response.json())
        assert rerun["id"] != batch["id"] and rerun["page_count"] == 3 and rerun["cpu_only"]
        assert client.delete(f"/api/batches/{batch['id']}").status_code == 204
        # Shared source documents must survive deleting the first batch.
        assert client.get(a["file_url"]).status_code == 200
        assert client.get(f"/api/batches/{rerun['id']}").status_code == 200
        assert client.delete(f"/api/batches/{rerun['id']}").status_code == 204
        assert client.get(f"/api/batches/{batch['id']}").status_code == 404


def test_batch_deduplicates_content_and_pages_even_across_uploads():
    with TestClient(app) as client:
        a, duplicate = upload(client), upload(client)
        assert a["id"] != duplicate["id"] and a["sha256"] == duplicate["sha256"]
        response = client.post("/api/batches", json={
            "documents": [
                {"document_id": a["id"], "pages": [1, 1, 2]},
                {"document_id": duplicate["id"], "pages": [2, 3]},
            ], "models": ["pypdf"],
        })
        assert response.status_code == 201, response.text
        batch = finished(client, response.json())
        assert batch["page_count"] == 3 and len(batch["experiments"]) == 1
        assert batch["experiments"][0]["pages"] == [1, 2, 3]
        assert client.delete(f"/api/batches/{batch['id']}").status_code == 204


def test_invalid_batch_is_atomic_and_enforces_combined_limit(monkeypatch):
    with TestClient(app) as client:
        a, b = upload(client, 7), upload(client, 6, "second")
        calls = []
        monkeypatch.setattr(app.state.runner, "submit", calls.append)
        before = client.get("/api/batches").json()
        bodies = [
            [{"document_id": a["id"], "pages": list(range(1, 8))},
             {"document_id": b["id"], "pages": list(range(1, 7))}],
            [{"document_id": a["id"], "pages": [1]},
             {"document_id": "missing", "pages": [1]}],
            [{"document_id": a["id"], "pages": [1]},
             {"document_id": b["id"], "pages": [7]}],
            [{"document_id": a["id"], "pages": []}],
            [{"document_id": a["id"], "pages": [True]}],
        ]
        for documents in bodies:
            response = client.post(
                "/api/batches", json={"documents": documents, "models": ["pypdf"]}
            )
            assert response.status_code in (422, 404), response.text
            assert client.get("/api/batches").json() == before
            assert calls == []
        unavailable = client.post("/api/batches", json={
            "documents": [{"document_id": a["id"], "pages": [1]}],
            "models": ["infinity-parser2-flash"], "cpu_only": True,
        })
        assert unavailable.status_code == 422 and "Disabled in CPU mode" in unavailable.text
        assert calls == []


def fake_experiment(pages, seconds, verdicts, status="completed", errors=(), rubric=RUBRIC_VERSION):
    return SimpleNamespace(pages=pages, runs=[SimpleNamespace(
        model_id="pypdf", status=status, parse_seconds=seconds,
        page_results=[{"page_number": n, "error": "failed" if n in errors else None}
                      for n in pages] if status == "completed" else [],
        judgment=SimpleNamespace(rubric_version=rubric, page_results=[
            {"page_number": n, "verdict": v} for n, v in verdicts.items()
        ]) if verdicts else None,
    )])


def test_aggregates_weight_pages_not_documents_and_expose_failures_and_missing_checks():
    row = aggregate_models([
        fake_experiment([1], 10, {1: "excellent"}),
        fake_experiment([1, 2, 3], 90, {1: "pass", 2: "fail"}),
        fake_experiment([1, 2], None, {}, status="failed"),
    ])[0]
    assert row.seconds_per_page == 25  # Not the unweighted document mean of 20.
    assert row.pages_total == 6 and row.pages_completed == 4 and row.pages_failed == 2
    assert row.timed_pages == 4 and row.parse_seconds == 100
    assert (row.excellent, row.passed, row.failed, row.judged_pages) == (1, 1, 1, 3)
    assert row.pass_rate == 66.7  # Missing checks and failed parsing are not quality passes/fails.


def test_partial_page_errors_and_old_judgments_do_not_inflate_quality():
    row = aggregate_models([
        fake_experiment([1, 2], 6, {1: "excellent", 2: "excellent"}, errors=[2]),
        fake_experiment([1], 9, {1: "excellent"}, rubric="old"),
    ])[0]
    assert row.pages_completed == 2 and row.pages_failed == 1
    assert row.timed_pages == 3 and row.seconds_per_page == 5
    assert row.judged_pages == 1 and row.excellent == 1


def test_batch_migration_preserves_legacy_runs_and_is_repeatable(monkeypatch):
    legacy = MetaData()
    for table in db.Base.metadata.sorted_tables:
        table.to_metadata(legacy)
    table = legacy.tables["experiments"]
    for index in list(table.indexes):
        if "batch_id" in index.columns:
            table.indexes.remove(index)
    for constraint in list(table.constraints):
        if "batch_id" in constraint.columns:
            table.constraints.remove(constraint)
    table._columns.remove(table.c.batch_id)
    engine = create_engine("sqlite://")
    try:
        legacy.create_all(engine)
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO experiments (id, document_id, status, pages, created_at) "
                "VALUES ('saved', 'document', 'completed', '[1, 3]', '2026-11-01 12:00:00')"
            )
        monkeypatch.setattr(db, "get_engine", lambda: engine)
        db.init_db()
        db.init_db()
        assert "batch_id" in {c["name"] for c in inspect(engine).get_columns("experiments")}
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT id, status, pages, batch_id FROM experiments"
            ).one() == ("saved", "completed", "[1, 3]", None)
    finally:
        engine.dispose()
