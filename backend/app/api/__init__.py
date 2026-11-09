from fastapi import APIRouter

from . import documents, experiments, models, samples

router = APIRouter(prefix="/api")
router.include_router(models.router)
router.include_router(documents.router)
router.include_router(experiments.router)
router.include_router(samples.router)
