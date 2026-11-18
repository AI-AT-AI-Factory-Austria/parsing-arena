"""Credential setup must persist without exposing keys or starting processing."""

import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import credentials
from app.api.settings import router
from app.config import get_settings
from app.engines.registry import EngineRegistry
from app.judging import JudgeService

HEADERS = {"X-Parsing-Arena-Setup": "1"}
KEYS = {"llama_cloud_api_key": "test-llama-secret", "mistral_api_key": "test-mistral-secret"}


@pytest.fixture
def setup_client(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "llama_cloud_api_key", "")
    monkeypatch.setattr(settings, "mistral_api_key", SecretStr(""))
    monkeypatch.setattr(settings, "llama_cloud_base_url", credentials.REGIONS["us"])
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.registry = EngineRegistry(settings)
    app.state.judge = JudgeService(settings)
    # No runner or judge worker is started: saving credentials must not need one.
    with TestClient(app) as client:
        yield client, settings


def test_save_persist_restore_and_update_both_runtimes(setup_client, monkeypatch):
    client, settings = setup_client
    status = client.get("/api/settings/judging")
    assert status.headers["cache-control"] == "no-store"
    assert status.json() == {
        "llamaparse_configured": False,
        "judge_configured": False,
        "llamaparse_region": "us",
    }
    response = client.post(
        "/api/settings/judging",
        json={**KEYS, "llamaparse_region": "eu"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "llamaparse_configured": True,
        "judge_configured": True,
        "llamaparse_region": "eu",
    }
    assert all(key not in response.text for key in KEYS.values())
    for cpu_only in (False, True):
        live = client.app.state.registry.for_mode(cpu_only).settings
        assert live.llama_cloud_api_key == KEYS["llama_cloud_api_key"]
        assert live.llama_cloud_base_url == credentials.REGIONS["eu"]
        assert live.mistral_api_key.get_secret_value() == KEYS["mistral_api_key"]
    assert client.app.state.judge.evaluator.settings.judge_configured
    path = credentials.credentials_path(settings)
    assert json.loads(path.read_text()) == {**KEYS, "llamaparse_region": "eu"}
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    # A new startup loads persisted keys with no environment secrets.
    restored = get_settings.__wrapped__()
    assert restored.llama_cloud_api_key == KEYS["llama_cloud_api_key"]
    assert restored.mistral_api_key.get_secret_value() == KEYS["mistral_api_key"]
    assert restored.llama_cloud_base_url == credentials.REGIONS["eu"]


def test_partial_save_and_environment_precedence(setup_client, monkeypatch):
    client, settings = setup_client
    for name, key in KEYS.items():
        response = client.post("/api/settings/judging", json={name: key}, headers=HEADERS)
        assert response.status_code == 200
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "env-llama")
    monkeypatch.setenv("MISTRAL", "env-mistral")
    restored = get_settings.__wrapped__()
    assert restored.llama_cloud_api_key == "env-llama"
    assert restored.mistral_api_key.get_secret_value() == "env-mistral"
    before = credentials.credentials_path(settings).read_bytes()
    for name in KEYS:
        response = client.post("/api/settings/judging", json={name: "overwrite"}, headers=HEADERS)
        assert response.status_code == 409
    assert credentials.credentials_path(settings).read_bytes() == before


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"mistral_api_key": ""},
        {"mistral_api_key": "secret with spaces"},
        {"mistral_api_key": "secret\ninside"},
        {"mistral_api_key": ["secret"]},
        {"mistral_api_key": "x" * 4097},
        {"unknown-secret-field": "secret"},
        {"llama_cloud_api_key": "secret", "llamaparse_region": "invalid"},
        {"mistral_api_key": "secret", "llamaparse_region": "eu"},
        ["secret"],
    ],
)
def test_invalid_body_does_not_echo_secrets(setup_client, body):
    client, settings = setup_client
    response = client.post("/api/settings/judging", json=body, headers=HEADERS)
    assert response.status_code == 422
    assert "secret" not in response.text
    assert not credentials.credentials_path(settings).exists()


def test_invalid_json_and_large_request(setup_client):
    client, _ = setup_client
    headers = {**HEADERS, "Content-Type": "application/json"}
    for body, expected in ((b'{"secret":', 422), (b"s" * 16_385, 413)):
        response = client.post("/api/settings/judging", content=body, headers=headers)
        assert response.status_code == expected
        assert "secret" not in response.text


def test_require_header_and_allowed_origin(setup_client):
    client, settings = setup_client
    assert client.post("/api/settings/judging", json=KEYS).status_code == 403
    response = client.post(
        "/api/settings/judging",
        json=KEYS,
        headers={**HEADERS, "Origin": "https://untrusted.example"},
    )
    assert response.status_code == 403
    assert not credentials.credentials_path(settings).exists()
    assert client.post("/api/settings/judging", data=KEYS, headers=HEADERS).status_code == 415
    response = client.post(
        "/api/settings/judging",
        json=KEYS,
        headers={**HEADERS, "Origin": settings.cors_origins[0]},
    )
    assert response.status_code == 200


def test_failed_save_is_atomic_and_can_be_retried(setup_client, monkeypatch):
    client, settings = setup_client
    assert (
        client.post(
            "/api/settings/judging",
            json={"mistral_api_key": KEYS["mistral_api_key"]},
            headers=HEADERS,
        ).status_code
        == 200
    )
    path = credentials.credentials_path(settings)
    before = path.read_bytes()
    with monkeypatch.context() as patch:

        def fail_replace(*args):
            raise PermissionError("test error with secret")

        patch.setattr(credentials.os, "replace", fail_replace)
        response = client.post(
            "/api/settings/judging",
            json={"llama_cloud_api_key": KEYS["llama_cloud_api_key"]},
            headers=HEADERS,
        )
        assert response.status_code == 500
        assert "secret" not in response.text
    assert path.read_bytes() == before
    assert not settings.llamaparse_configured
    assert not list(settings.data_dir.glob(".judging-keys-*"))
    assert (
        client.post(
            "/api/settings/judging",
            json={"llama_cloud_api_key": KEYS["llama_cloud_api_key"]},
            headers=HEADERS,
        ).status_code
        == 200
    )


def test_corrupt_file_does_not_block_local_startup(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "judging-keys.json").write_text('{"secret"')
    settings = get_settings.__wrapped__()
    assert not settings.llamaparse_configured and not settings.judge_configured
    assert "Saved API keys could not be loaded" in caplog.text
    assert "secret" not in caplog.text


def test_environment_keys_do_not_get_written_to_file(setup_client, monkeypatch):
    client, settings = setup_client
    monkeypatch.setattr(settings, "llama_cloud_api_key", "env-private-key")
    response = client.post(
        "/api/settings/judging",
        json={"mistral_api_key": KEYS["mistral_api_key"]},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert "env-private-key" not in credentials.credentials_path(settings).read_text()
    assert "env-private-key" not in client.get("/api/settings/judging").text
    assert settings.llama_cloud_api_key == "env-private-key"
