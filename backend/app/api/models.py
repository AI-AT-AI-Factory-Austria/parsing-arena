from fastapi import APIRouter, Request

from .. import __version__
from ..catalog import LLAMAPARSE_CREDITS_PER_PAGE, LLAMAPARSE_USD_PER_CREDIT, MODELS, REFERENCE
from ..config import get_settings
from ..schemas import HealthOut, ModelsOut
from .serializers import model_out

router = APIRouter(tags=["models"])


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    settings = get_settings()
    registry = request.app.state.registry
    return HealthOut(
        status="ok",
        version=__version__,
        llamaparse_configured=settings.llamaparse_configured,
        local_device=registry.local_device() if settings.local_inference else None,
        max_upload_mb=settings.max_upload_mb,
        max_pages_per_experiment=settings.max_pages_per_experiment,
    )


@router.get("/models", response_model=ModelsOut)
def list_models(request: Request) -> ModelsOut:
    registry = request.app.state.registry
    return ModelsOut(
        models=[model_out(m, registry) for m in MODELS],
        reference=model_out(REFERENCE, registry),
        llamaparse_credits_per_page=LLAMAPARSE_CREDITS_PER_PAGE,
        llamaparse_usd_per_credit=LLAMAPARSE_USD_PER_CREDIT,
        source="ParseBench leaderboard (github.com/run-llama/ParseBench)",
    )
