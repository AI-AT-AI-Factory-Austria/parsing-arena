"""Install pinned portable Windows x64 tools used by the local parser presets.

Run with the backend Python. Files stay inside data/runtimes; no PATH/system changes.
Marker's Python environment is installed separately with uv sync --project runtimes/marker.
"""

import hashlib
import io
import os
import platform
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(
    os.environ.get(
        "RUNTIME_DIR",
        Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data")) / "runtimes",
    )
)
TOOLS = {
    "java": (
        "https://github.com/adoptium/temurin21-binaries/releases/download/"
        "jdk-21.0.13%2B9/OpenJDK21U-jre_x64_windows_hotspot_21.0.13_9.zip",
        "2332c0b4cff6bed722da3567c91d5e47568bcba863fa4945d0a8d6fc76158628",
    ),
    "llama": (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        "b12110/llama-b12110-bin-win-vulkan-x64.zip",
        "9732a5e48887ca54d0bfe4af4324dc85fd56cd8cda19b52df3542fa6ebe6874f",
    ),
}


def main():
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit(
            "This installer is for Windows x64. Install Java 11+ and llama.cpp locally."
        )
    for name, (url, checksum) in TOOLS.items():
        target = ROOT / name
        marker = target / ".archive-sha256"
        if marker.exists() and marker.read_text() == checksum:
            print(f"{name}: already installed", flush=True)
            continue
        print(f"Downloading {name}…", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise RuntimeError(f"Checksum mismatch for {name}")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.infolist():
                if not (target / entry.filename).resolve().is_relative_to(target.resolve()):
                    raise RuntimeError("Unsafe archive path")
            archive.extractall(target)
        marker.write_text(checksum)
        print(f"Installed {name} in {target}", flush=True)


if __name__ == "__main__":
    main()
