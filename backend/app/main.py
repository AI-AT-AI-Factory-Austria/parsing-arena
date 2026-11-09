import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import router
from .config import get_settings
from .db import init_db
from .engines.registry import EngineRegistry
from .judging import JudgeService
from .runner import Runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    registry = EngineRegistry(settings)
    judge = JudgeService(settings)
    runner = Runner(settings, registry, judge)
    app.state.registry = registry
    app.state.runner = runner
    app.state.judge = judge
    await judge.start()
    await runner.start()
    if settings.local_inference:
        # Importing torch takes a few seconds; do it before the first request needs it.
        asyncio.get_running_loop().run_in_executor(None, registry.local_device)
    yield
    await runner.stop()
    await judge.stop()
    await registry.pool.close()


app = FastAPI(title="OCR Arena API", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
