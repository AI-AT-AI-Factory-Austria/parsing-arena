"""Deployment profiles select implementations; they never pretend hardware exists."""

import sys

from .config import Settings
from .engines.backends.transformers_local import describe_local_device


def detected_profile() -> str:
    device = describe_local_device("auto")
    if sys.platform == "win32" and device.get("runtime") == "ROCm":
        return "windows-amd"
    if sys.platform == "linux" and device.get("runtime") == "CUDA":
        return "linux-nvidia"
    return "native"


def runtime_profile(settings: Settings) -> str:
    return detected_profile() if settings.runtime_profile == "auto" else settings.runtime_profile


def profile_error(settings: Settings) -> str | None:
    if settings.runtime_profile != "auto" and settings.runtime_profile != detected_profile():
        return (
            f"RUNTIME_PROFILE={settings.runtime_profile} does not match this machine "
            f"({detected_profile()}). Use auto or install the matching GPU runtime."
        )
    return None
