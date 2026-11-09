"""Background execution of experiments.

One experiment runs at a time. Within an experiment the open models run one
after another so their timings do not compete for the same accelerator, while
LlamaParse (remote) runs concurrently with them. Model loading is timed
separately from parsing.
"""

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import asdict
from typing import Any

from sqlalchemy import select

from . import documents, storage
from .catalog import LLAMAPARSE_ID
from .config import Settings
from .db import session_scope
from .engines.base import EngineUnavailable, PageSource, ParseOutput, RenderSpec, Report
from .engines.registry import EngineRegistry
from .judging import JudgeService
from .models import Document, Experiment, ModelRun, utcnow

logger = logging.getLogger(__name__)

INTERRUPTED = "Interrupted because the server restarted."
PROGRESS_FLUSH_S = 0.5


def _update_run(run_id: str, **fields: Any) -> None:
    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        if run is None:
            return
        for key, value in fields.items():
            setattr(run, key, value)


class Runner:
    def __init__(
        self, settings: Settings, registry: EngineRegistry, judge: JudgeService | None = None
    ):
        self.settings = settings
        self.registry = registry
        self.judge = judge
        self._pending: deque[str] = deque()
        self._wakeup: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self.current: str | None = None

    # ----------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        for experiment_id in self._recover():
            self._enqueue(experiment_id)
        self._task = asyncio.create_task(self._work(), name="experiment-runner")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def submit(self, experiment_id: str) -> None:
        """Thread-safe: queue an experiment for execution."""
        assert self._loop is not None, "runner not started"
        self._loop.call_soon_threadsafe(self._enqueue, experiment_id)

    def queue_position(self, experiment_id: str) -> int | None:
        """0 = next in line. None when running or not queued."""
        try:
            return list(self._pending).index(experiment_id)
        except ValueError:
            return None

    def _enqueue(self, experiment_id: str) -> None:
        if experiment_id not in self._pending and experiment_id != self.current:
            self._pending.append(experiment_id)
            assert self._wakeup is not None
            self._wakeup.set()

    def _recover(self) -> list[str]:
        queued: list[str] = []
        with session_scope() as session:
            stmt = (
                select(Experiment)
                .where(Experiment.status.in_(["queued", "running"]))
                .order_by(Experiment.created_at)
            )
            for experiment in session.scalars(stmt):
                if experiment.status == "queued":
                    queued.append(experiment.id)
                    continue
                now = utcnow()
                for run in experiment.runs:
                    if run.status not in ("completed", "failed"):
                        run.status, run.error, run.finished_at = "failed", INTERRUPTED, now
                done = any(r.status == "completed" for r in experiment.runs)
                experiment.status = "completed" if done else "failed"
                experiment.error = INTERRUPTED
                experiment.finished_at = now
        return queued

    async def _work(self) -> None:
        assert self._wakeup is not None
        while True:
            while not self._pending:
                self._wakeup.clear()
                await self._wakeup.wait()
            experiment_id = self._pending.popleft()
            self.current = experiment_id
            try:
                await self._run_experiment(experiment_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("experiment %s failed", experiment_id)
                self._fail_experiment(experiment_id, str(exc))
            finally:
                self.current = None

    # ---------------------------------------------------------- experiment
    async def _run_experiment(self, experiment_id: str) -> None:
        with session_scope() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment is None or experiment.status != "queued":
                return
            experiment.status = "running"
            experiment.started_at = utcnow()
            document = experiment.document
            session.expunge(document)
            pages = list(experiment.pages)
            runs = [(r.id, r.model_id) for r in experiment.runs]

        reference = [run_id for run_id, model_id in runs if model_id == LLAMAPARSE_ID]
        open_runs = [(run_id, model_id) for run_id, model_id in runs if model_id != LLAMAPARSE_ID]

        reference_task = None
        if reference:
            reference_task = asyncio.create_task(self._run_reference(reference[0], document, pages))
        try:
            for run_id, model_id in open_runs:
                await self._run_open_model(experiment_id, run_id, model_id, document, pages)
        finally:
            if reference_task is not None:
                await reference_task

        with session_scope() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment is None:
                return
            succeeded = any(r.status == "completed" for r in experiment.runs)
            experiment.status = "completed" if succeeded else "failed"
            if not succeeded:
                experiment.error = "No model produced an output."
            experiment.finished_at = utcnow()

    @staticmethod
    def _render_pages(document: Document, pages: list[int], spec: RenderSpec | None) -> PageSource:
        paths = {}
        if spec is not None:
            paths = {
                n: documents.page_image(document, n, dpi=spec.dpi, max_pixels=spec.max_pixels)
                for n in pages
            }
        return PageSource(
            numbers=list(pages),
            paths=paths,
            source=storage.original_path(document.id, document.extension),
            extension=document.extension,
        )

    def _fail_experiment(self, experiment_id: str, message: str) -> None:
        with session_scope() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment is None:
                return
            now = utcnow()
            for run in experiment.runs:
                if run.status not in ("completed", "failed"):
                    run.status, run.error, run.finished_at = "failed", message, now
            experiment.status = "failed"
            experiment.error = message
            experiment.finished_at = now

    # ---------------------------------------------------------------- runs
    def _reporter(self, run_id: str) -> Report:
        state: dict[str, Any] = {}
        last_flush = 0.0

        async def report(update: dict[str, Any]) -> None:
            nonlocal last_flush
            state.update(update)
            now = time.monotonic()
            if "phase" in update or "pages_done" in update or now - last_flush >= PROGRESS_FLUSH_S:
                last_flush = now
                fields: dict[str, Any] = {"progress": dict(state)}
                if "pages_done" in update:
                    fields["pages_done"] = update["pages_done"]
                _update_run(run_id, **fields)

        return report

    async def _run_open_model(
        self,
        experiment_id: str,
        run_id: str,
        model_id: str,
        document: Document,
        pages: list[int],
    ) -> None:
        engine = self.registry.open_engine(model_id)
        if engine is None:
            _update_run(
                run_id,
                status="failed",
                error="This model is not integrated yet.",
                finished_at=utcnow(),
            )
            return
        _update_run(run_id, status="loading", started_at=utcnow(), progress={"phase": "loading"})
        try:
            # Rendering happens before the parse timer so every model is timed on inference.
            source = await asyncio.to_thread(self._render_pages, document, pages, engine.render)
            backend, load_seconds = await engine.prepare()
            _update_run(
                run_id,
                status="running",
                load_seconds=round(load_seconds, 3),
                meta=engine.describe(backend),
                progress={"phase": "layout"},
            )
            started = time.perf_counter()
            output = await engine.parse(
                backend,
                source,
                report=self._reporter(run_id),
                artifact_dir=storage.run_dir(experiment_id, run_id),
            )
            elapsed = time.perf_counter() - started
            self._complete(run_id, output, elapsed)
        except EngineUnavailable as exc:
            _update_run(run_id, status="failed", error=str(exc), finished_at=utcnow())
        except Exception as exc:
            logger.exception("model run %s (%s) failed", run_id, model_id)
            _update_run(
                run_id, status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=utcnow()
            )
        finally:
            await engine.release()

    async def _run_reference(self, run_id: str, document: Document, pages: list[int]) -> None:
        engine = self.registry.llamaparse
        _update_run(
            run_id,
            status="running",
            started_at=utcnow(),
            meta=engine.describe(),
            progress={"phase": "uploading"},
        )
        started = time.perf_counter()
        try:
            output = await engine.parse(
                storage.original_path(document.id, document.extension),
                filename=document.filename,
                media_type=document.media_type,
                page_numbers=pages,
                page_count=document.page_count,
                report=self._reporter(run_id),
            )
            self._complete(run_id, output, time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, EngineUnavailable):
                logger.warning("LlamaParse run %s failed: %s", run_id, exc)
            _update_run(run_id, status="failed", error=str(exc), finished_at=utcnow())

    def _complete(self, run_id: str, output: ParseOutput, elapsed: float) -> None:
        with session_scope() as session:
            run = session.get(ModelRun, run_id)
            if run is None:
                return
            run.status = "completed"
            run.parse_seconds = round(elapsed, 3)
            run.markdown = output.markdown
            run.page_results = [asdict(page) for page in output.pages]
            run.pages_done = len(output.pages)
            run.meta = {**(run.meta or {}), **output.meta}
            run.progress = {"phase": "done"}
            run.finished_at = utcnow()
            experiment_id = run.experiment_id
        if self.judge:
            self.judge.submit_experiment(experiment_id)
