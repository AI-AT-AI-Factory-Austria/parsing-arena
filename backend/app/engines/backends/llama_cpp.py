"""Pinned full-precision GGUF conversions, served locally on the AMD Vulkan GPU.

The engine still owns prompts, page rendering, crops and output normalization.
Only model inference changes. No document leaves the loopback interface.
"""

import asyncio
import atexit
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ...config import Settings


@dataclass(frozen=True)
class GgufSpec:
    repo: str
    revision: str
    model_file: str
    projector_file: str


GGUF_MODELS = {
    "lightonocr-3-0.8b": GgufSpec(
        "mradermacher/LightOnOCR-3-0.8B-GGUF",
        "d76992384e501892b4e37bd55020d762d30b5b37",
        "LightOnOCR-3-0.8B.f16.gguf",
        "LightOnOCR-3-0.8B.mmproj-f16.gguf",
    ),
    "ovisocr3": GgufSpec(
        "mradermacher/OvisOCR3-GGUF",
        "041e883c46e3fe2bbe0a8d7d9bdc73571f6f769a",
        "OvisOCR3.f16.gguf",
        "OvisOCR3.mmproj-f16.gguf",
    ),
    "infinity-parser2-flash": GgufSpec(
        "prithivMLmods/Infinity-Parser2-Flash-GGUF",
        "4611cfe753931a1f2f730a68254fe42a842558f3",
        "Infinity-Parser2-Flash.F16.gguf",
        "Infinity-Parser2-Flash.mmproj-f16.gguf",
    ),
}


def llama_binary(settings: Settings) -> str | None:
    # Lazy import: library_worker shares the portable-tool discovery code.
    from ..library_worker import find_tool

    return find_tool(settings, "llama")


class LlamaCppRuntime:
    def __init__(
        self,
        model_id: str,
        settings: Settings,
        context_length: int,
        *,
        spec: GgufSpec | None = None,
        binary: str | None = None,
        device: str = "Vulkan0",
        flash_attention: bool = True,
        image_max_tokens: int = 16384,
        extra_args: tuple[str, ...] = (),
    ):
        self.model_id, self.settings = model_id, settings
        self.spec = spec or GGUF_MODELS[model_id]
        self.binary, self.device = binary, device
        self.flash_attention, self.image_max_tokens = flash_attention, image_max_tokens
        self.extra_args = extra_args
        self.context_length = context_length
        self.process: subprocess.Popen | None = None
        self.client: httpx.Client | None = None
        self.log_file = None
        self.device_name = ""

    @property
    def loaded(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.client is not None

    def load(self) -> None:
        from huggingface_hub import hf_hub_download

        atexit.register(self.unload)
        binary = self.binary or llama_binary(self.settings)
        if not binary:
            raise RuntimeError("Install the Vulkan llama-server or set LLAMA_CPP_BINARY.")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        devices = subprocess.run(
            [binary, "--list-devices"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=flags,
            check=True,
        )
        lines = (devices.stdout + devices.stderr).splitlines()
        self.device_name = next((line.strip() for line in lines if f"{self.device}:" in line), "")
        if not self.device_name:
            raise RuntimeError(f"This llama-server cannot find GPU {self.device}.")
        model, projector = [
            hf_hub_download(self.spec.repo, name, revision=self.spec.revision)
            for name in (self.spec.model_file, self.spec.projector_file)
        ]
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        log_dir = self.settings.data_dir / "runtime-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.model_id}-{self.device.rstrip('0123456789').lower()}.log"
        self.log_file = log_path.open("w", encoding="utf-8")
        command = [
            binary,
            "--model",
            model,
            "--mmproj",
            projector,
            "--alias",
            self.model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parallel",
            "1",
            "--ctx-size",
            str(self.context_length),
            "--fit",
            "off",
            "--device",
            self.device,
            "--mmproj-device",
            self.device,
            "--gpu-layers",
            "all",
            "--flash-attn",
            "on" if self.flash_attention else "off",
            "--jinja",
            "--reasoning",
            "off",
            "--image-max-tokens",
            str(self.image_max_tokens),
            *self.extra_args,
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            self.client = httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=self.settings.endpoint_timeout_s,
                trust_env=False,
            )
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Local model server exited; see {log_path}.")
                try:
                    if self.client.get("/health", timeout=2).status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                time.sleep(0.25)
            raise TimeoutError(f"Local model loading timed out; see {log_path}.")
        except BaseException:
            self.unload()
            raise

    def unload(self) -> None:
        atexit.unregister(self.unload)
        if self.client:
            self.client.close()
            self.client = None
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.process = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "llama.cpp",
            "runtime": self.device.rstrip("0123456789"),
            "device": self.device,
            "device_name": self.device_name,
            "dtype": "float16",
            "format": "GGUF F16",
            "weights_repo": self.spec.repo,
            "weights_revision": self.spec.revision,
            "model_file": self.spec.model_file,
            "projector_file": self.spec.projector_file,
            "context_length": self.context_length,
        }

    def _complete(self, payload: dict) -> str:
        payload = dict(payload)
        require_complete = payload.pop("arena_require_complete", False)
        if not self.loaded:
            raise RuntimeError("The local model server is not running.")
        assert self.client
        response = self.client.post(
            "/v1/chat/completions",
            json={
                **payload,
                "model": self.model_id,
                "chat_template_kwargs": {"enable_thinking": False},
                # llama.cpp has restrictive sampling defaults; preserve the engine's policy.
                "top_k": payload.get("top_k", 0),
                "min_p": 0,
                "top_p": payload.get("top_p", 1.0),
                "repeat_penalty": 1.0,
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if require_complete and choice.get("finish_reason") != "stop":
            raise RuntimeError("Table reconstruction did not finish; page OCR was retained.")
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "The model reached its output limit before finishing the extraction."
            )
        return choice["message"].get("content") or ""

    async def complete(self, payload: dict) -> str:
        return await asyncio.to_thread(self._complete, payload)
