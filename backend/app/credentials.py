"""Installation credentials: server-only, atomic storage, never served as content."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import SecretStr

if TYPE_CHECKING:
    from .config import Settings

REGIONS = {
    "us": "https://api.cloud.llamaindex.ai",
    "eu": "https://api.cloud.eu.llamaindex.ai",
}
STORE_LOCK = threading.Lock()


class CredentialStoreError(RuntimeError):
    pass


def credentials_path(settings: Settings) -> Path:
    return settings.data_dir / "judging-keys.json"


def read_credentials(settings: Settings) -> dict[str, str]:
    try:
        raw = json.loads(credentials_path(settings).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or any(not isinstance(v, str) for v in raw.values()):
            raise ValueError("Invalid credential store")
        return raw
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        raise CredentialStoreError(
            "Saved API keys could not be read. Check the server's key file."
        ) from None


def apply_saved_credentials(settings: Settings) -> None:
    saved = read_credentials(settings)
    # Environment/.env settings always take precedence over UI configuration.
    if not settings.llamaparse_configured and saved.get("llama_cloud_api_key"):
        settings.llama_cloud_api_key = saved["llama_cloud_api_key"]
        region = saved.get("llamaparse_region")
        if region in REGIONS:
            settings.llama_cloud_base_url = REGIONS[region]
    if not settings.judge_configured and saved.get("mistral_api_key"):
        settings.mistral_api_key = SecretStr(saved["mistral_api_key"])


def write_credentials(settings: Settings, values: dict[str, str]) -> None:
    """Caller holds STORE_LOCK. Keep existing entries and replace atomically."""
    saved = read_credentials(settings)
    saved.update(values)
    temporary = None
    try:
        path = credentials_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp creates a private 0600 file on POSIX. On Windows the service
        # account's directory ACL controls access; no keys go into git or logs.
        fd, temporary = tempfile.mkstemp(prefix=".judging-keys-", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        raise CredentialStoreError(
            "API keys could not be saved. Check server storage permissions."
        ) from None
    finally:
        if temporary:
            with suppress(OSError):
                os.unlink(temporary)
