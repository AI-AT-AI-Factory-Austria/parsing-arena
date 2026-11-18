"""Persistent local subprocesses for libraries with incompatible dependencies.

Only paths/configuration and extraction results cross the pipe. No remote APIs.
"""

import asyncio
import atexit
import importlib.util
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..config import BACKEND_DIR
from .base import PageOutput, PageSource, ParseOutput, Report
from .cpu_runtime import cpu_llama_args, cpu_subprocess_env
from .tools import CpuEngine

logger = logging.getLogger(__name__)
PREFIX = "ARENA_RESULT:"


def marker_python(settings) -> Path:
    return (
        Path(settings.marker_python)
        if settings.marker_python
        else (
            BACKEND_DIR
            / "runtimes"
            / "marker"
            / ".venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
    )


def find_tool(settings, name: str) -> str | None:
    configured = getattr(settings, "java_cmd" if name == "java" else "llama_cpp_binary")
    if configured:
        return shutil.which(configured)
    system = shutil.which(name if name == "java" else "llama-server")
    if system:
        return system
    root = settings.runtime_root / ("java" if name == "java" else "llama")
    filename = ("java" if name == "java" else "llama-server") + (".exe" if os.name == "nt" else "")
    return next((str(p) for p in root.rglob(filename) if p.is_file()), None)


class LibraryRuntime:
    def __init__(self, preset, settings):
        self.preset, self.settings = preset, settings
        self.process = None
        self.responses = queue.Queue()
        self.info = {}
        self.log_file = None

    @property
    def loaded(self):
        return self.process is not None and self.process.poll() is None and bool(self.info)

    def _read(self, stream):
        for line in stream:
            if line.startswith(PREFIX):
                self.responses.put(json.loads(line[len(PREFIX) :]))
            else:
                logger.debug("%s: %s", self.preset, line.rstrip())
        self.responses.put({"error": f"{self.preset} worker exited; see its runtime log."})

    def load(self):
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", HF_HUB_DISABLE_PROGRESS_BARS="1", OMP_NUM_THREADS="4")
        if self.settings.local_device == "cpu":
            env = cpu_subprocess_env(env, self.settings.cpu_threads)
        python = sys.executable
        if self.preset.startswith("marker_"):
            python = str(marker_python(self.settings))
            env.update(
                SURYA_MODEL_CHECKPOINT="datalab-to/surya-ocr-2.5",
                SURYA_GGUF_REPO="datalab-to/surya-ocr-2.5-gguf",
                SURYA_GGUF_MODEL_FILE="surya-2.5.gguf",
                SURYA_GGUF_MMPROJ_FILE="surya-2.5-mmproj.gguf",
                SURYA_INFERENCE_BACKEND="llamacpp",
                SURYA_INFERENCE_PARALLEL="1",
                TORCH_DEVICE="cpu",
                SURYA_INFERENCE_KEEP_ALIVE="False",
            )
            for override in (
                "SURYA_INFERENCE_URL",
                "SURYA_GGUF_LOCAL_MODEL_PATH",
                "SURYA_GGUF_LOCAL_MMPROJ_PATH",
            ):
                env.pop(override, None)
            binary = find_tool(self.settings, "llama")
            if binary:
                env["LLAMA_CPP_BINARY"] = binary
            if self.settings.local_device == "cpu":
                env.update(
                    LLAMA_CPP_NGL="0",
                    LLAMA_CPP_NO_MMPROJ_OFFLOAD="True",
                    LLAMA_CPP_EXTRA_ARGS=" ".join(cpu_llama_args(self.settings.cpu_threads)),
                    FAST_DETECTOR_DEVICE="cpu",
                )
                for name in (
                    "DETECTOR_SERVER_URL",
                    "OCR_ERROR_SERVER_URL",
                    "FAST_LAYOUT_SERVER_URL",
                ):
                    env.pop(name, None)
        else:
            java = find_tool(self.settings, "java")
            if java:
                env["PATH"] = str(Path(java).parent) + os.pathsep + env.get("PATH", "")
        log_dir = self.settings.data_dir / "runtime-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (log_dir / f"{self.preset}.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [python, "-u", str(Path(__file__).with_name("preset_worker.py")), self.preset],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            text=True,
            encoding="utf-8",
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        threading.Thread(target=self._read, args=(self.process.stdout,), daemon=True).start()
        atexit.register(self.unload)
        self.info = self._response()

    def _response(self):
        try:
            response = self.responses.get(timeout=1200)
        except queue.Empty as exc:
            self.unload()  # Never reuse a timed-out worker with an outstanding reply.
            raise TimeoutError(
                f"{self.preset} exceeded 20 minutes; see data/runtime-logs."
            ) from exc
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def convert(self, source: Path, number: int, artifact_dir: Path):
        self.process.stdin.write(
            json.dumps({"source": str(source), "page": number, "output": str(artifact_dir)}) + "\n"
        )
        self.process.stdin.flush()
        return self._response()

    def describe(self):
        return self.info

    def unload(self):
        process, self.process = self.process, None
        self.info = {}
        if process is not None:
            import psutil

            # Surya's auxiliary servers deliberately outlive their client. Track
            # only our own children before the worker exits and they are reparented.
            try:
                children = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            if process.poll() is None:
                try:
                    process.stdin.close()  # EOF lets upstream atexit clean up model servers.
                    process.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=10)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(children, timeout=10)
            if process.stdout:
                process.stdout.close()
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        atexit.unregister(self.unload)


class LibraryEngine(CpuEngine):
    @property
    def pool_key(self):
        return type(self).__name__, self.model.id

    @property
    def backend_kind(self):
        return (
            "local"
            if self.model.id == "marker_balanced" and self.settings.local_device != "cpu"
            else "cpu"
        )

    def unavailable_reason(self):
        if self.model.id == "marker_balanced" and not self.settings.local_inference:
            return "Local inference is disabled."
        if self.model.id.startswith("marker_"):
            if not marker_python(self.settings).is_file():
                return "Install Marker with: uv sync --project runtimes/marker (from backend)."
            if self.model.id == "marker_balanced" and not find_tool(self.settings, "llama"):
                return "Marker Balanced needs llama-server. Run scripts/install-preset-tools.py."
        else:
            if importlib.util.find_spec("opendataloader_pdf") is None:
                return "OpenDataLoader is not installed (uv sync --extra local --inexact)."
            if not find_tool(self.settings, "java"):
                return "OpenDataLoader needs Java 11+. Run scripts/install-preset-tools.py."
        return None

    def build_local(self):
        return LibraryRuntime(self.model.id, self.settings)

    async def parse(
        self, backend: LibraryRuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        if pages.source is None:
            raise RuntimeError("The original document is required.")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        source = pages.source
        if pages.extension != ".pdf" and self.model.id.startswith("opendataloader_"):
            # Preserve image-only input; Core still has no OCR and returns no text.
            from PIL import Image

            source = artifact_dir / "image-input.pdf"
            with Image.open(pages.source) as image:
                image.convert("RGB").save(source, "PDF")
        outputs = []
        for index, number in enumerate(pages.numbers):
            await report({"phase": "parsing", "page": number, "page_index": index})
            started = time.perf_counter()
            result = await asyncio.to_thread(backend.convert, source, number, artifact_dir)
            outputs.append(
                PageOutput(
                    number,
                    result["markdown"],
                    time.perf_counter() - started,
                    result.get("elements", []),
                )
            )
            await report({"pages_done": index + 1, "page": number})
        return ParseOutput("\n\n".join(p.markdown for p in outputs), outputs)
