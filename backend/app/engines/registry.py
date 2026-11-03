from typing import Any

from ..catalog import LLAMAPARSE_ID, MODELS, ArenaModel
from ..config import Settings
from .backends.transformers_local import LocalModelPool, describe_local_device
from .kdl.engine import VARIANTS, KdlEngine
from .llamaparse import LlamaParseEngine


class EngineRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool = LocalModelPool(
            device=settings.local_device,
            dtype=settings.local_dtype,
            max_batch=settings.local_max_batch,
            max_model_len=settings.local_max_model_len,
            keep_loaded=settings.keep_models_loaded,
        )
        self.llamaparse = LlamaParseEngine(settings)
        self._open: dict[str, KdlEngine] = {
            m.id: KdlEngine(m, settings, self.pool) for m in MODELS if m.id in VARIANTS
        }

    def open_engine(self, model_id: str) -> KdlEngine | None:
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
            info["loaded"] = self.pool.is_loaded(model.hf_repo or "", model.revision)
        return info

    def local_device(self) -> dict[str, Any]:
        return describe_local_device(self.settings.local_device)
