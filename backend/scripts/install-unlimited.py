# /// script
# requires-python = ">=3.11"
# dependencies = ["cmake==4.1.2", "ninja==1.13.0", "huggingface-hub>=0.36"]
# ///
"""Install Unlimited-OCR 2's separate, pinned llama.cpp runtime.

Windows: the author's patched DLLs over the official b11913 Vulkan release.
Linux: build the same patch with CUDA, or CPU-only with --cpu (no CUDA Toolkit).
Does not replace the runtime used by the other models or the AMD PyTorch install.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "RX5950XT/Unlimited-OCR-2-GGUF"
REVISION = "1e3e5f3e82435578cd1e3c191673e1444b617f52"
VERSION = "b11913"
ARCHIVE_SHA = "d9b27dfe8addb0782c59c194604dbab69ccef8493c8964a435d762c4be45e176"
OVERLAY_SHA = "931f4842d9c261d0aa57ce6469f4f8297da4f9a7eb02852464478d8754a1d06d"
ROOT = Path(__file__).resolve().parents[2] / "data/runtimes/unlimited-ocr2"


def download(url: str, path: Path, sha: str | None = None) -> Path:
    if not path.exists():
        print(f"Downloading {path.name}...", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(path)
    if sha:
        with path.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != sha:
                raise RuntimeError(f"Checksum mismatch: {path}")
    return path


def extract(archive: Path, target: Path, strip_root: bool = False):
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            parts = Path(item.filename).parts[1:] if strip_root else Path(item.filename).parts
            relative = Path(*parts)
            destination = (target / relative).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise RuntimeError("Unsafe archive path")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-model", action="store_true", help="Cache BF16 + F16 weights (~6.7 GB)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Build without CUDA on Linux. Existing GPU builds also support CPU inference.",
    )
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "build-info.json"
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info.get("patch_revision") != REVISION or info.get("llama_version") != VERSION:
            raise RuntimeError(
                "A different Unlimited runtime is installed. Use a separate directory."
            )
        binary = ROOT / info["binary"]
        if not binary.is_file():
            raise RuntimeError("Incomplete runtime installation: remove build-info.json and rerun.")
    else:
        if os.name == "nt":
            archive = download(
                f"https://github.com/ggml-org/llama.cpp/releases/download/{VERSION}/"
                f"llama-{VERSION}-bin-win-vulkan-x64.zip",
                ROOT / "base.zip",
                ARCHIVE_SHA,
            )
            extract(archive, ROOT / "bin")
            binary = next((ROOT / "bin").rglob("llama-server.exe"))
            overlay = Path(
                hf_hub_download(
                    REPO,
                    f"llama.cpp-bin/llama-{VERSION}-unlimited-ocr2-win-x64.zip",
                    revision=REVISION,
                )
            )
            with overlay.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != OVERLAY_SHA:
                    raise RuntimeError("Patched DLL checksum mismatch")
            with zipfile.ZipFile(overlay) as source:
                for name in ("llama.dll", "mtmd.dll"):
                    entry = next(
                        item for item in source.infolist() if Path(item.filename).name == name
                    )
                    with source.open(entry) as src, (binary.parent / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            backend = "Vulkan"
        else:
            import cmake
            import ninja

            source = ROOT / "source"
            archive = download(
                f"https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/{VERSION}",
                ROOT / "source.zip",
            )
            if not (source / ".arena-patched").exists():
                extract(archive, source, strip_root=True)
                # Keep git apply relative to this source tree, not the parent arena repo.
                subprocess.run(["git", "init", "--quiet", str(source)], check=True)
                patch = hf_hub_download(
                    REPO, "llama.cpp-patch/unlimited-ocr2.patch", revision=REVISION
                )
                subprocess.run(["git", "apply", "--check", patch], cwd=source, check=True)
                subprocess.run(["git", "apply", patch], cwd=source, check=True)
                (source / ".arena-patched").write_text(REVISION)
            build = source / "build"
            cmake_bin = str(Path(cmake.CMAKE_BIN_DIR) / "cmake")
            subprocess.run(
                [
                    cmake_bin,
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    f"-DCMAKE_MAKE_PROGRAM={Path(ninja.BIN_DIR) / 'ninja'}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DGGML_CUDA={'OFF' if args.cpu else 'ON'}",
                    "-DGGML_VULKAN=OFF",
                    "-DLLAMA_CURL=OFF",
                ],
                check=True,
            )
            subprocess.run(
                [cmake_bin, "--build", str(build), "--target", "llama-server", "-j", "4"],
                check=True,
            )
            binary = build / "bin/llama-server"
            backend = "CPU" if args.cpu else "CUDA"
        manifest.write_text(
            json.dumps(
                {
                    "llama_version": VERSION,
                    "patch_repo": REPO,
                    "patch_revision": REVISION,
                    "backend": backend,
                    "binary": binary.relative_to(ROOT).as_posix(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    subprocess.run([str(binary), "--list-devices"], check=True)
    if args.download_model:
        for filename in ("Unlimited-OCR-2-BF16.gguf", "mmproj-Unlimited-OCR-2-F16.gguf"):
            print(f"Caching {filename}...", flush=True)
            hf_hub_download(REPO, filename, revision=REVISION)
    print(f"Installed: {binary}", flush=True)


if __name__ == "__main__":
    main()
