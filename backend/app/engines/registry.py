from typing import Any

from ..catalog import LLAMAPARSE_ID, MODELS, ArenaModel
from ..config import Settings
from .backends.transformers_local import LocalModelPool, describe_local_device
from .falcon import FalconOcrEngine
from .infinity import InfinityParser2Engine
from .kdl.engine import KdlEngine
from .lightonocr import LightOnOcr3Engine
from .llamaparse import LlamaParseEngine
from .mineru import MinerUEngine
from .nemotron import NemotronEngine
from .open_engine import OpenModelEngine
from .ovisocr import OvisOcr3Engine
from .tools import DoclingEngine, PypdfEngine, TesseractEngine

ENGINES: dict[str, type[OpenModelEngine]] = {
    "rakedoc-nano": KdlEngine,
    "aeddix-alpine-ocr": KdlEngine,
    "florin-parser-nano": KdlEngine,
    "nemotron-3.5-parse": NemotronEngine,
    "lightonocr-3-0.8b": LightOnOcr3Engine,
    "mineru2.5-pro-2605": MinerUEngine,
    "infinity-parser2-flash": InfinityParser2Engine,
    "pypdf": PypdfEngine,
    "tesseract": TesseractEngine,
    "docling": DoclingEngine,
    "falcon-ocr": FalconOcrEngine,
    "ovisocr3": OvisOcr3Engine,
}


class EngineRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool = LocalModelPool(
            keep_loaded=settings.keep_models_loaded, max_loaded=settings.max_loaded_models
        )
        self.llamaparse = LlamaParseEngine(settings)
        self._open: dict[str, OpenModelEngine] = {
            m.id: ENGINES[m.id](m, settings, self.pool) for m in MODELS if m.id in ENGINES
        }

    def open_engine(self, model_id: str) -> OpenModelEngine | None:
        return self._open.get(model_id)

    def availability(self, model: ArenaModel) -> dict[str, Any]:
        """How (and whether) this server can run the model right now."""
        if model.id == LLAMAPARSE_ID:
            reason = self.llamaparse.unavailable_reason()
            return {"runnable": reason is None, "backend": "api", "reason": reason}
        engine = self._open.get(model.id)
        if engine is None:
            return {"runnable": False, "backend": None, "reason": "Not integrated yet."}
        reason = engine.unavailable_reason()
        info: dict[str, Any] = {
            "runnable": reason is None,
            "backend": engine.backend_kind,
            "reason": reason,
        }
        if engine.backend_kind == "local" and reason is None:
            info["loaded"] = engine.is_loaded()
        return info

    def local_device(self) -> dict[str, Any]:
        return describe_local_device(self.settings.local_device)
