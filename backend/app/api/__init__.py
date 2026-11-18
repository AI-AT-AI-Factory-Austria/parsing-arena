from fastapi import APIRouter

from . import batches, documents, experiments, models, samples, settings

router = APIRouter(prefix="/api")
router.include_router(models.router)
router.include_router(documents.router)
router.include_router(experiments.router)
router.include_router(samples.router)
router.include_router(batches.router)
router.include_router(settings.router)
