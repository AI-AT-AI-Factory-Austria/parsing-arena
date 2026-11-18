"""Nemotron F16 with SkyVision's full-page parser on the CPU (local stdio only)."""

import asyncio
import atexit
import json
import math
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from ..config import Settings
from .base import PageSource, ParseOutput, RenderSpec, Report, element
from .nemotron import SDK_FILES, load_sdk
from .open_engine import PageFailed, parse_page_by_page
from .tools import CpuEngine

MODEL_FILE = "nemotron-parse-f16.gguf"
MAX_TOKENS = 8744
CONTEXT_LENGTH = 24576
NORMALIZER_REPO = "nvidia/Nemotron-3.5-Parse"
NORMALIZER_REVISION = "4035ffb1d8f8bafcdb867dfaf6f7514f101464f9"


def skyvision_binary(settings: Settings) -> str | None:
    if settings.skyvision_binary:
        return shutil.which(settings.skyvision_binary)
    path = (
        settings.data_dir / "runtimes/skyvision" / ("vqa-cli.exe" if os.name == "nt" else "vqa-cli")
    )
    return str(path) if path.is_file() else shutil.which("vqa-cli")


def build_info(binary: str) -> dict:
    try:
        return json.loads(Path(binary).with_name("build-info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def document_output(document: dict) -> tuple[str, list[dict]]:
    """SkyVision JSON uses the original Nemotron class names and normalized boxes."""
    markdown = document.get("markdown")
    blocks = document.get("blocks")
    if not isinstance(markdown, str) or not isinstance(blocks, list):
        raise PageFailed("SkyVision returned an invalid document response.")
    if not markdown.strip() and not blocks:
        raise PageFailed("SkyVision returned no document content.")
    layout = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        bbox = block.get("bbox")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, int | float) and math.isfinite(v) for v in bbox)
        ):
            continue
        x1, y1, x2, y2 = bbox
        layout.append(
            element(
                str(block.get("class") or "Text"),
                [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                str(block.get("content") or block.get("text") or ""),
                len(layout),
            )
        )
    return markdown, layout


class SkyVisionRuntime:
    def __init__(self, model, settings: Settings):
        self.model, self.settings = model, settings
        self.process: subprocess.Popen | None = None
        self.responses: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.log_file = None
        self.weights = ""
        self.ready = False
        self.sequence = 0
        self.build = {}
        self.postprocess = None

    @property
    def loaded(self):
        return self.ready and self.process is not None and self.process.poll() is None

    def _read(self, stream):
        try:
            for line in stream:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put(RuntimeError("Invalid JSON from the SkyVision worker."))
                    return
        finally:
            self.responses.put(RuntimeError("SkyVision exited; see its runtime log."))

    def _rpc(self, method: str, params: dict) -> dict:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("The SkyVision worker is not running.")
            self.sequence += 1
            request = {"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params}
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            try:
                response = self.responses.get(timeout=self.settings.skyvision_timeout_s)
            except queue.Empty as exc:
                self.unload()
                raise TimeoutError("SkyVision timed out; its worker has been stopped.") from exc
            if isinstance(response, Exception):
                raise response
            if not isinstance(response, dict) or response.get("id") != self.sequence:
                self.unload()
                raise RuntimeError("Unexpected response from the SkyVision worker.")
            if "error" in response:
                raise RuntimeError(str(response["error"]).replace("OCR Arena:", "Parsing Arena:"))
            return response["result"]

    def _tool(self, name: str, arguments: dict) -> dict:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(
            c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
        )
        if result.get("isError"):
            # Older installed workers can still emit the previous project name.
            raise RuntimeError(
                text.replace("OCR Arena:", "Parsing Arena:") or "SkyVision inference failed."
            )
        document = json.loads(text)
        if not isinstance(document, dict):
            raise RuntimeError("SkyVision returned an invalid document.")
        return document

    def load(self):
        from huggingface_hub import snapshot_download

        binary = skyvision_binary(self.settings)
        if not binary:
            raise RuntimeError(
                "SkyVision is not installed. Run backend/scripts/install-skyvision.py."
            )
        self.build = build_info(binary)
        if not self.build.get("strict_token_limit"):
            raise RuntimeError(
                "Rebuild SkyVision with the arena installer for complete-page checks."
            )
        root = Path(
            snapshot_download(
                self.model.hf_repo,
                revision=self.model.revision,
                allow_patterns=[MODEL_FILE, "tokenizer*.json", "special_tokens_map.json"],
            )
        )
        self.weights = str(root / MODEL_FILE)
        # SkyVision's current table renderer drops merged-cell text. Normalize its
        # untouched token stream with NVIDIA's text-only helpers (no model/GPU load).
        normalizer_root = Path(
            snapshot_download(
                NORMALIZER_REPO,
                revision=NORMALIZER_REVISION,
                allow_patterns=SDK_FILES,
            )
        )
        self.postprocess = load_sdk(normalizer_root).postprocess_document
        env = os.environ.copy()
        # Preserve the checkpoint's native image resolution and CPU math defaults.
        for name in list(env):
            if name.startswith(("NEMOTRON_PARSE_", "VQA_")):
                env.pop(name)
        log_dir = self.settings.data_dir / "runtime-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (log_dir / "nemotron-skyvision-cpu.log").open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [
                    binary,
                    "--mcp",
                    "--models-dir",
                    str(root),
                    "--model",
                    self.weights,
                    "--tokenizer",
                    str(root),
                    "--cpu",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log_file,
                text=True,
                encoding="utf-8",
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            atexit.register(self.unload)
            threading.Thread(target=self._read, args=(self.process.stdout,), daemon=True).start()
            self._rpc(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "parsing-arena", "version": "1.0"},
                },
            )
            result = self._tool(
                "load_model",
                {
                    "model": self.weights,
                    "gpu": False,
                    "n_ctx": CONTEXT_LENGTH,
                    "threads": self.settings.skyvision_threads,
                },
            )
            if result.get("arch") != "nemotron_parse":
                raise RuntimeError("SkyVision loaded the wrong model architecture.")
            self.ready = True
        except BaseException:
            self.unload()
            raise

    def parse(self, image_path: Path) -> dict:
        if not self.loaded:
            raise RuntimeError("The SkyVision model is not loaded.")
        document = self._tool(
            "parse_document",
            {
                "model": self.weights,
                "image": str(image_path.resolve()),
                "format": "json",
                "gpu": False,
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
            },
        )
        raw = document.get("raw_out")
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("SkyVision returned no raw model output.")
        assert self.postprocess is not None
        normalized = self.postprocess(raw)
        normalized["skyvision_raw_out"] = raw
        return normalized

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "SkyVision.cpp",
            "device": "cpu",
            "runtime": "CPU",
            "format": "GGUF F16",
            "dtype": "float16",
            "weights_repo": self.model.hf_repo,
            "weights_revision": self.model.revision,
            "model_file": MODEL_FILE,
            "mode": "full-page",
            "max_tokens": MAX_TOKENS,
            "context_length": CONTEXT_LENGTH,
            "threads": self.settings.skyvision_threads or "auto (physical cores)",
            "normalizer": "NVIDIA SDK postprocess_document (text only)",
            "normalizer_repo": NORMALIZER_REPO,
            "normalizer_revision": NORMALIZER_REVISION,
            "build": self.build,
        }

    def unload(self):
        atexit.unregister(self.unload)
        self.ready = False
        self.postprocess = None
        process, self.process = self.process, None
        if process:
            if process.poll() is None:
                try:
                    process.stdin.close()
                    process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=10)
            if process.stdout:
                process.stdout.close()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        if self.log_file:
            self.log_file.close()
            self.log_file = None


class NemotronSkyVisionEngine(CpuEngine):
    render = RenderSpec(dpi=150)
    runtime_label = "CPU · F16 GGUF"

    @property
    def pool_key(self):
        return (*super().pool_key, self.settings.skyvision_binary, self.settings.skyvision_threads)

    def unavailable_reason(self):
        binary = skyvision_binary(self.settings)
        if not binary or not build_info(binary).get("strict_token_limit"):
            return (
                "Install the SkyVision CPU runtime: uv run --no-project --script "
                "backend/scripts/install-skyvision.py (from the repo root)."
            )
        return None

    def build_local(self):
        return SkyVisionRuntime(self.model, self.settings)

    async def parse(
        self, backend: SkyVisionRuntime, pages: PageSource, *, report: Report, artifact_dir: Path
    ) -> ParseOutput:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        async def parse_one(number, _image):
            try:
                document = await asyncio.to_thread(backend.parse, pages.paths[number])
            except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
                raise PageFailed(f"{type(exc).__name__}: {exc}") from exc
            (artifact_dir / f"skyvision-page-{number}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return document_output(document)

        return await parse_page_by_page(pages, report, parse_one)
