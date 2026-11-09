from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from app import samples
from app.config import get_settings
from app.db import init_db, session_scope
from app.engines.registry import EngineRegistry
from app.main import app
from app.models import Experiment, SampleDocument


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    folder = tmp_path / "sample-docs"
    folder.mkdir()
    source = Path(__file__).resolve().parents[1] / "samples" / "sample-report.pdf"
    (folder / "Example-Report_p2.pdf").write_bytes(source.read_bytes())
    Image.new("RGB", (64, 64), "white").save(folder / "Example-Invoice.png")
    (folder / "ignored.txt").write_text("Not a supported document")
    monkeypatch.setattr(
        samples, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path, max_upload_mb=100)
    )
    init_db()
    runner = SimpleNamespace(submit=Mock(), queue_position=lambda _: None)
    monkeypatch.setattr(app.state, "runner", runner, raising=False)
    monkeypatch.setattr(app.state, "registry", EngineRegistry(get_settings()), raising=False)
    with TestClient(app) as client:
        # Lifespan creates the actual runner; replace it before any requests submit work.
        actual_runner = app.state.runner
        monkeypatch.setattr(actual_runner, "submit", runner.submit)
        yield client, folder, runner
    with session_scope() as session:
        for sample in session.scalars(select(SampleDocument)):
            document = sample.document
            session.delete(sample)
            session.delete(document)


def test_discovery_imports_once_and_never_runs_models(gallery):
    client, _, runner = gallery
    first = client.get("/api/samples").json()["samples"]
    second = client.get("/api/samples").json()["samples"]
    assert len(first) == 2
    assert [s["document"]["id"] for s in first] == [s["document"]["id"] for s in second]
    assert all(s["status"] == "unprepared" and s["experiment_id"] is None for s in first)
    assert first[1]["title"] == "Example Report · Page 2"
    assert client.get(first[0]["document"]["thumbnail_url"]).status_code == 200
    assert client.get("/api/samples/random").status_code == 409
    runner.submit.assert_not_called()


def test_preparation_is_idempotent_and_random_only_opens_saved_results(gallery):
    client, _, runner = gallery
    sample = client.get("/api/samples").json()["samples"][1]
    url = f"/api/samples/{sample['id']}/prepare"
    first = client.post(url, json={"models": ["pypdf"]})
    assert first.status_code == 200
    eid = first.json()["id"]
    with session_scope() as session:
        experiment = session.get(Experiment, eid)
        experiment.status = "completed"
        experiment.runs[0].status = "completed"
    # Fresh HTTP/DB sessions and repeated visits must reuse persisted IDs.
    for _ in range(3):
        assert client.post(url, json={"models": ["pypdf"]}).json()["id"] == eid
        chosen = client.get("/api/samples/random")
        assert chosen.json()["experiment_id"] == eid
        assert chosen.headers["cache-control"] == "no-store"
    runner.submit.assert_called_once_with(eid)
    refreshed = client.post(url, json={"models": ["pypdf"], "refresh": True})
    assert refreshed.status_code == 200
    assert refreshed.json()["id"] != eid
    assert runner.submit.call_count == 2


def test_concurrent_preparation_submits_only_one_run(gallery):
    client, _, runner = gallery
    sample = client.get("/api/samples").json()["samples"][1]

    def prepare(_):
        response = client.post(f"/api/samples/{sample['id']}/prepare", json={"models": ["pypdf"]})
        assert response.status_code == 200
        return response.json()["id"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        ids = list(pool.map(prepare, range(3)))
    assert len(set(ids)) == 1
    runner.submit.assert_called_once()


def test_changed_or_removed_files_do_not_reuse_stale_results(gallery):
    client, folder, runner = gallery
    before = client.get("/api/samples").json()["samples"]
    old_image = before[0]
    prepared = client.post(
        f"/api/samples/{old_image['id']}/prepare", json={"models": ["pypdf"]}
    ).json()
    Image.new("RGB", (64, 64), "black").save(folder / "Example-Invoice.png")
    after = client.get("/api/samples").json()["samples"]
    assert after[0]["id"] != old_image["id"]
    assert after[0]["experiment_id"] is None
    assert client.get(f"/api/experiments/{prepared['id']}").status_code == 200
    (folder / "Example-Invoice.png").unlink()
    assert len(client.get("/api/samples").json()["samples"]) == 1
    runner.submit.assert_called_once()


def test_existing_exact_comparison_is_adopted_without_rerun(gallery):
    client, _, runner = gallery
    sample = client.get("/api/samples").json()["samples"][1]
    existing = client.post(
        "/api/experiments", json={"document_id": sample["document"]["id"], "models": ["pypdf"]}
    ).json()
    saved = client.post(f"/api/samples/{sample['id']}/prepare", json={"models": ["pypdf"]})
    assert saved.json()["id"] == existing["id"]
    runner.submit.assert_called_once()


def test_missing_sample_and_invalid_model_are_rejected(gallery):
    client, _, runner = gallery
    assert (
        client.post("/api/samples/missing/prepare", json={"models": ["pypdf"]}).status_code == 404
    )
    sample = client.get("/api/samples").json()["samples"][1]
    url = f"/api/samples/{sample['id']}/prepare"
    assert client.post(url, json={"models": []}).status_code == 422
    assert client.post(url, json={"models": ["unknown"]}).status_code == 422
    runner.submit.assert_not_called()
