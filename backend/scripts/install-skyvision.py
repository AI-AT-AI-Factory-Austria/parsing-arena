# /// script
# requires-python = ">=3.11"
# dependencies = ["cmake==4.1.2", "ninja==1.13.0"]
# ///
"""Build the pinned SkyVision CPU runtime without changing system tools or PyTorch.

Run: uv run --no-project --script backend/scripts/install-skyvision.py
Windows downloads a portable LLVM/MinGW compiler; Linux uses the installed C/C++ compiler.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import cmake
import ninja

SOURCE_REV = "a4822e5065a08f1198b9d6b1cffc274626172bd1"
GGML_REV = "57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234"
COMPILER_VERSION = "20261104"
COMPILER_SHA = "552191b5754c49a5a20723276a6fc56541ea1922c4c50b7b669022333fc1b40f"
ROOT = Path(__file__).resolve().parents[2] / "data" / "runtimes"


def download_zip(url: str, target: Path, checksum: str | None = None):
    if (target / ".extracted").exists():
        return
    archive = target.with_suffix(".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {target.name}...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    if checksum and hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
        raise RuntimeError(f"Checksum mismatch: {archive}")
    with zipfile.ZipFile(archive) as source:
        # GitHub source and compiler archives have a single top-level directory.
        for item in source.infolist():
            relative = Path(*Path(item.filename).parts[1:])
            destination = (target / relative).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise RuntimeError("Unsafe archive path")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    (target / ".extracted").write_text(url, encoding="utf-8")


def main():
    target = ROOT / "skyvision"
    source = target / f"source-{SOURCE_REV[:12]}"
    download_zip(f"https://codeload.github.com/SkYContact/SkyVision.cpp/zip/{SOURCE_REV}", source)
    download_zip(f"https://codeload.github.com/ggml-org/ggml/zip/{GGML_REV}", source / "ggml")
    # Upstream relies on transitive C library includes; libc++ needs this explicit header.
    for name in ("src/core/text/tokenizer.cpp", "src/core/engine/weight_load.hpp"):
        file = source / name
        text = file.read_text(encoding="utf-8")
        if "#include <cstdlib>" not in text:
            file.write_text("#include <cstdlib>\n" + text, encoding="utf-8")
    # Return a failed page on token exhaustion, rather than a successful partial extraction.
    model_source = source / "src/models/nemotron_parse/nemotron_parse_model.cpp"
    text = model_source.read_text(encoding="utf-8")
    marker = "Parsing Arena: output token limit reached; page is incomplete."
    # Upgrade sources patched before the project was renamed without applying
    # the token-cap patch a second time.
    legacy_marker = "OCR Arena: output token limit reached; page is incomplete."
    if legacy_marker in text:
        text = text.replace(legacy_marker, marker)
        model_source.write_text(text, encoding="utf-8")
    if marker not in text:
        old = (
            "    if (generated >= max_new && cur != id_eos)\n"
            '        fprintf(stderr, "nemotron_parse: hit the %d token cap '
            '(raise --max-tokens for long "\n'
            '                        "pages; a full page is often 2-8k tokens)\\n", max_new);'
        )
        new = (
            "    if (generated >= max_new && cur != id_eos) {\n"
            f'        out_text = "{marker}";\n'
            "        return false;\n"
            "    }"
        )
        if old not in text:
            raise RuntimeError("Pinned Nemotron token-cap patch no longer matches.")
        model_source.write_text(text.replace(old, new, 1), encoding="utf-8")
    env = os.environ.copy()
    args = []
    if os.name == "nt":
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            raise SystemExit("The portable compiler requires Windows x64.")
        compiler = ROOT / "build-tools" / f"llvm-mingw-{COMPILER_VERSION}"
        download_zip(
            f"https://github.com/mstorsjo/llvm-mingw/releases/download/{COMPILER_VERSION}/"
            f"llvm-mingw-{COMPILER_VERSION}-ucrt-x86_64.zip",
            compiler,
            COMPILER_SHA,
        )
        env["PATH"] = str(compiler / "bin") + os.pathsep + env.get("PATH", "")
        args += [
            f"-DCMAKE_C_COMPILER={(compiler / 'bin/clang.exe').as_posix()}",
            f"-DCMAKE_CXX_COMPILER={(compiler / 'bin/clang++.exe').as_posix()}",
            "-DCMAKE_EXE_LINKER_FLAGS=-static",
            "-DCMAKE_CXX_FLAGS=-include cstdlib",
        ]
        # Upstream links Winsock with an MSVC pragma; MinGW needs an explicit library.
        cmake_file = source / "CMakeLists.txt"
        cmake_text = cmake_file.read_text(encoding="utf-8")
        patch = "\nif(MINGW)\n  target_link_libraries(vqa-cli PRIVATE ws2_32)\nendif()\n"
        if patch not in cmake_text:
            cmake_file.write_text(cmake_text + patch, encoding="utf-8")
    cmake_binary = str(Path(cmake.CMAKE_BIN_DIR) / ("cmake.exe" if os.name == "nt" else "cmake"))
    ninja_binary = str(Path(ninja.BIN_DIR) / ("ninja.exe" if os.name == "nt" else "ninja"))
    build = source / "build-cpu"
    command = [
        cmake_binary,
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja_binary}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_BACKEND_DL=OFF",
        "-DGGML_CUDA=OFF",
        "-DGGML_BUILD_EXAMPLES=OFF",
        "-DGGML_BUILD_TESTS=OFF",
        "-DGGML_OPENMP=OFF",
        "-DGGML_NATIVE=OFF",
        "-DGGML_AVX=ON",
        "-DGGML_AVX2=ON",
        "-DGGML_FMA=ON",
        "-DGGML_F16C=ON",
        *args,
    ]
    subprocess.run(command, env=env, check=True)
    subprocess.run(
        [
            cmake_binary,
            "--build",
            str(build),
            "--target",
            "vqa-cli",
            "np_markup_test",
            "--parallel",
            "8",
        ],
        env=env,
        check=True,
    )
    suffix = ".exe" if os.name == "nt" else ""
    subprocess.run([str(build / f"np_markup_test{suffix}")], env=env, check=True)
    binary = target / f"vqa-cli{suffix}"
    shutil.copy2(build / binary.name, binary)
    specs = subprocess.run(
        [str(binary), "--specs"], capture_output=True, text=True, encoding="utf-8", check=True
    )
    assert "nemotron_parse" in {s["arch"] for s in json.loads(specs.stdout)}
    manifest = {
        "source_revision": SOURCE_REV,
        "ggml_revision": GGML_REV,
        "device": "cpu",
        "build": "AVX2/FMA/F16C; static; no CUDA",
        "mingw_winsock_patch": os.name == "nt",
        "strict_token_limit": True,
        "explicit_cstdlib": True,
    }
    (target / "build-info.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Installed and checked: {binary}", flush=True)


if __name__ == "__main__":
    main()
