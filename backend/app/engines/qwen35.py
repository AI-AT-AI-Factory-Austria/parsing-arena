"""Qwen3.5 OCR via local GGUF on Windows/AMD or CPU, with native PyTorch fallback."""

from ..runtime import profile_error, runtime_profile
from .backends.llama_cpp import GGUF_MODELS, LlamaCppRuntime, llama_binary
from .open_engine import OpenModelEngine


class Qwen35Engine(OpenModelEngine):
    @property
    def uses_llama_cpp(self) -> bool:
        return self.uses_vulkan or (
            self.settings.local_inference
            and not self.endpoint_url
            and self.settings.local_device == "cpu"
            and bool(llama_binary(self.settings))
        )

    @property
    def uses_vulkan(self) -> bool:
        return (
            self.settings.local_inference
            and not self.endpoint_url
            and self.settings.local_device != "cpu"
            and runtime_profile(self.settings) == "windows-amd"
        )

    @property
    def runtime_label(self) -> str | None:
        if self.uses_llama_cpp and self.settings.local_device == "cpu":
            return "CPU · F16 GGUF"
        return "AMD Vulkan · F16 GGUF" if self.uses_vulkan else None

    @property
    def pool_key(self):
        variant = GGUF_MODELS[self.model.id] if self.uses_llama_cpp else None
        return (*super().pool_key, variant, self.settings.local_device, self.settings.local_dtype)

    def unavailable_reason(self) -> str | None:
        if self.uses_llama_cpp:
            if not self.settings.local_inference:
                return "Local inference is disabled."
            if error := profile_error(self.settings):
                return error
            if not llama_binary(self.settings):
                return "Install the Vulkan llama-server (backend/scripts/install-preset-tools.py)."
            return None
        return super().unavailable_reason()

    def vulkan_backend(self, context_length: int) -> LlamaCppRuntime:
        return LlamaCppRuntime(self.model.id, self.settings, context_length)
