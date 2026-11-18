from typing import Any

from ..catalog import LLAMAPARSE_ID, MODELS, ArenaModel
from ..config import Settings
from .backends.transformers_local import LocalModelPool, describe_local_device
from .doctr import DoctrEngine
from .execution import execution_info
from .glm import GlmOcrEngine
from .granite import GraniteDoclingEngine
from .infinity import InfinityParser2Engine
from .kdl.engine import KdlEngine
from .library_worker import LibraryEngine
from .lightonocr import LightOnOcr3Engine
from .llamaparse import LlamaParseEngine
from .mineru import MinerUEngine
from .open_engine import OpenModelEngine
from .ovisocr import OvisOcr3Engine
from .paddle import PaddleOcrEngine
from .skyvision import NemotronSkyVisionEngine
from .tools import DoclingEngine, PypdfEngine, TesseractEngine
from .unlimited import UnlimitedOcrEngine

ENGINES: dict[str, type[OpenModelEngine]] = {
    "rakedoc-nano": KdlEngine,
    "aeddix-alpine-ocr": KdlEngine,
    "florin-parser-nano": KdlEngine,
    "nemotron-skyvision-cpu": NemotronSkyVisionEngine,
    "lightonocr-3-0.8b": LightOnOcr3Engine,
    "mineru2.5-pro-2605": MinerUEngine,
    "infinity-parser2-flash": InfinityParser2Engine,
    "pypdf": PypdfEngine,
    "tesseract": TesseractEngine,
    "docling": DoclingEngine,
    "granite-docling-258m": GraniteDoclingEngine,
    "marker_fast": LibraryEngine,
    "marker_balanced": LibraryEngine,
    "opendataloader_core": LibraryEngine,
    "opendataloader_docling": LibraryEngine,
    "doctr_compact": DoctrEngine,
    "doctr_default": DoctrEngine,
    "doctr_large_transformer": DoctrEngine,
    "paddleocr_tiny": PaddleOcrEngine,
    "paddleocr_small": PaddleOcrEngine,
    "paddleocr_medium": PaddleOcrEngine,
    "glm-ocr": GlmOcrEngine,
    "ovisocr3": OvisOcr3Engine,
    "unlimited-ocr-2": UnlimitedOcrEngine,
}

# Excluded from CPU selection due to long processing times. Florin and docTR
# Large were added at the user's request after the batch run. GPU availability is unchanged.
CPU_DISABLED_MODELS = frozenset({
    "infinity-parser2-flash",
    "aeddix-alpine-ocr",
    "glm-ocr",
    "florin-parser-nano",
    "doctr_large_transformer",
})


class EngineRegistry:
    def __init__(self, settings: Settings):
        if settings.local_device == "cpu":
            settings = settings.for_cpu()
        self.settings = settings
        self.pool = LocalModelPool(
            keep_loaded=settings.keep_models_loaded, max_loaded=settings.max_loaded_models
        )
        self.llamaparse = LlamaParseEngine(settings)
        self._open: dict[str, OpenModelEngine] = {
            m.id: ENGINES[m.id](m, settings, self.pool) for m in MODELS if m.id in ENGINES
        }
        self._cpu = EngineRegistry(settings.for_cpu()) if settings.local_device != "cpu" else None

    def for_mode(self, cpu_only: bool = False) -> "EngineRegistry":
        return self._cpu if cpu_only and self._cpu is not None else self

    async def close(self) -> None:
        await self.pool.close()
        if self._cpu is not None:
            await self._cpu.close()

    def open_engine(self, model_id: str) -> OpenModelEngine | None:
        return self._open.get(model_id)

    def availability(self, model: ArenaModel) -> dict[str, Any]:
        """How (and whether) this server can run the model right now."""
        if model.id == LLAMAPARSE_ID:
            reason = self.llamaparse.unavailable_reason()
            return {
                "runnable": reason is None, "backend": "api", "reason": reason,
                "execution_label": "Hosted API",
                "execution_detail": "Documents are sent to the LlamaParse API when selected.",
            }
        engine = self._open.get(model.id)
        if engine is None:
            return {"runnable": False, "backend": None, "reason": "Not integrated yet."}
        reason = (
            "Disabled in CPU mode due to long processing times."
            if self.settings.local_device == "cpu" and model.id in CPU_DISABLED_MODELS
            else engine.unavailable_reason()
        )
        info: dict[str, Any] = {
            "runnable": reason is None,
            "backend": engine.backend_kind,
            "reason": reason,
            "runtime_label": getattr(engine, "runtime_label", None),
            "max_pages": getattr(engine, "max_pages", None),
        }
        if self.settings.local_device == "cpu":
            info["runtime_label"] = getattr(engine, "runtime_label", None) or "Local CPU"
        info["execution_label"], info["execution_detail"] = execution_info(
            model.id, engine, self.settings, reason is None
        )
        if engine.backend_kind == "local" and reason is None:
            info["loaded"] = engine.is_loaded()
        return info

    def local_device(self) -> dict[str, Any]:
        return describe_local_device(self.settings.local_device)
