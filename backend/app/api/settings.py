"""Write-only setup for missing installation keys; never return stored secrets."""

import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from ..config import get_settings
from ..credentials import (
    REGIONS,
    STORE_LOCK,
    CredentialStoreError,
    apply_saved_credentials,
    write_credentials,
)

router = APIRouter(prefix="/settings", tags=["settings"])


class JudgingKeyStatus(BaseModel):
    llamaparse_configured: bool
    judge_configured: bool
    llamaparse_region: Literal["us", "eu"]


class JudgingKeysInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llama_cloud_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    llamaparse_region: Literal["us", "eu"] | None = None


def _status() -> JudgingKeyStatus:
    settings = get_settings()
    return JudgingKeyStatus(
        llamaparse_configured=settings.llamaparse_configured,
        judge_configured=settings.judge_configured,
        llamaparse_region="eu" if settings.llama_cloud_base_url == REGIONS["eu"] else "us",
    )


@router.get("/judging", response_model=JudgingKeyStatus)
def judging_key_status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return _status()


@router.post("/judging", response_model=JudgingKeyStatus)
async def save_judging_keys(request: Request, response: Response):
    settings = get_settings()
    # Custom header + JSON prevent cross-site forms from changing installation
    # credentials. Deployed origins must match the configured CORS allowlist.
    origin = request.headers.get("origin")
    allowed = {*settings.cors_origins, str(request.base_url).rstrip("/")}
    if request.headers.get("x-parsing-arena-setup") != "1" or (
        origin is not None and origin not in allowed
    ):
        raise HTTPException(403, "API keys must be saved from this installation's settings dialog.")
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise HTTPException(415, "Send API keys as JSON.")
    # Parse here so Pydantic/FastAPI cannot echo submitted secrets in 422 errors.
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 16_384:
            raise HTTPException(413, "API key settings are too large.")
    try:
        body = JudgingKeysInput.model_validate(json.loads(raw))
    except (ValueError, ValidationError):
        raise HTTPException(422, "Provide API keys and a valid LlamaParse region.") from None
    values = {}
    for name in ("llama_cloud_api_key", "mistral_api_key"):
        secret = getattr(body, name)
        if secret is not None:
            value = secret.get_secret_value().strip()
            if not value or len(value) > 4096 or any(not 33 <= ord(c) <= 126 for c in value):
                raise HTTPException(
                    422, "API keys must be nonempty and contain no spaces or control characters."
                )
            values[name] = value
    if not values:
        raise HTTPException(422, "Enter at least one missing API key.")
    if body.llamaparse_region and "llama_cloud_api_key" not in values:
        raise HTTPException(422, "Set the LlamaParse region together with its missing API key.")
    if "llama_cloud_api_key" in values:
        values["llamaparse_region"] = body.llamaparse_region or _status().llamaparse_region
    try:
        with STORE_LOCK:
            apply_saved_credentials(settings)
            if ("llama_cloud_api_key" in values and settings.llamaparse_configured) or (
                "mistral_api_key" in values and settings.judge_configured
            ):
                raise HTTPException(
                    409,
                    "An API key is already configured. Reopen the dialog to refresh its status.",
                )
            write_credentials(settings, values)
            apply_saved_credentials(settings)
            # CPU mode has its own Settings copy; make saved keys usable there
            # immediately too, without restarting or changing inference options.
            registry = request.app.state.registry
            for cpu_only in (False, True):
                apply_saved_credentials(registry.for_mode(cpu_only).settings)
    except CredentialStoreError as exc:
        raise HTTPException(500, str(exc)) from None
    response.headers["Cache-Control"] = "no-store"
    return _status()
