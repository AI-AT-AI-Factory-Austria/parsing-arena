"""Durable background judging, independent of OCR scheduling and timing."""

import asyncio
import contextlib
import logging

from sqlalchemy import select

from .catalog import LLAMAPARSE_ID
from .config import Settings
from .db import session_scope
from .judge import RUBRIC_VERSION, JudgeError, MistralJudge, input_hash, overall_verdict
from .models import Experiment, Judgment, ModelRun, utcnow

logger = logging.getLogger(__name__)


class JudgeService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.evaluator = MistralJudge(settings)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._scheduled: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    async def start(self):
        self._loop = asyncio.get_running_loop()
        with session_scope() as session:
            pending = list(
                session.scalars(select(Judgment).where(Judgment.status.in_(["queued", "running"])))
            )
            for judgment in pending:
                if self.settings.judge_configured:
                    judgment.status = "queued"
                    self._enqueue(judgment.run_id)
                else:
                    judgment.status = "failed"
                    judgment.error = "Add MISTRAL to .env, then retry the quality check."
        self._task = asyncio.create_task(self._work(), name="quality-judge")

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _enqueue(self, run_id: str):
        if run_id not in self._scheduled:
            self._scheduled.add(run_id)
            self._queue.put_nowait(run_id)

    def submit_experiment(self, experiment_id: str, *, retry_failed: bool = False) -> int:
        if not self.settings.judge_configured or self._loop is None:
            return 0
        queued = []
        with session_scope() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment is None:
                return 0
            reference = next((r for r in experiment.runs if r.model_id == LLAMAPARSE_ID), None)
            if reference is None or reference.status != "completed":
                return 0
            for run in experiment.runs:
                if run.id == reference.id or run.status != "completed":
                    continue
                fingerprint = input_hash(
                    reference.id,
                    experiment.pages,
                    reference.page_results or [],
                    run.page_results or [],
                    self.settings.judge_model,
                )
                judgment = run.judgment
                if judgment:
                    if judgment.status in ("queued", "running"):
                        continue
                    if judgment.input_hash == fingerprint:
                        if judgment.status == "completed" or not retry_failed:
                            continue
                else:
                    judgment = Judgment(run_id=run.id)
                    session.add(judgment)
                # Preserve completed pages on retries/restarts, but never across changed inputs.
                if judgment.input_hash != fingerprint:
                    judgment.page_results = []
                judgment.reference_run_id = reference.id
                judgment.input_hash = fingerprint
                judgment.model = self.settings.judge_model
                judgment.rubric_version = RUBRIC_VERSION
                judgment.pages_total = len(experiment.pages)
                judgment.status = "queued"
                judgment.verdict = None
                judgment.error = None
                judgment.finished_at = None
                queued.append(run.id)
        for run_id in queued:
            self._loop.call_soon_threadsafe(self._enqueue, run_id)
        return len(queued)

    async def _work(self):
        while True:
            run_id = await self._queue.get()
            try:
                await self._evaluate(run_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the worker alive, redact provider details
                # Do not log document text, provider response bodies or credentials.
                message = (
                    str(exc)
                    if isinstance(exc, JudgeError)
                    else "Quality check interrupted. Retry it."
                )
                logger.warning("Judgment %s failed (%s)", run_id, type(exc).__name__)
                with session_scope() as session:
                    judgment = session.get(Judgment, run_id)
                    if judgment:
                        judgment.status = "failed"
                        judgment.verdict = None
                        judgment.error = message
                        judgment.finished_at = utcnow()
            finally:
                self._scheduled.discard(run_id)
                self._queue.task_done()

    async def _evaluate(self, run_id: str):
        with session_scope() as session:
            judgment = session.get(Judgment, run_id)
            if judgment is None or judgment.status != "queued":
                return
            run = session.get(ModelRun, run_id)
            reference = session.get(ModelRun, judgment.reference_run_id)
            if run is None or reference is None:
                raise JudgeError("The candidate or reference output is no longer available.")
            pages = list(run.experiment.pages)
            fingerprint = input_hash(
                reference.id,
                pages,
                reference.page_results or [],
                run.page_results or [],
                self.settings.judge_model,
            )
            if fingerprint != judgment.input_hash:
                raise JudgeError("Outputs or judge settings changed. Run the quality check again.")
            reference_pages = {p["page_number"]: p for p in reference.page_results or []}
            candidate_pages = {p["page_number"]: p for p in run.page_results or []}
            results = list(judgment.page_results or [])
            judgment.status = "running"

        # Validate all reference pages before making any paid request.
        for number in pages:
            baseline = reference_pages.get(number)
            if not baseline or baseline.get("error") or not baseline.get("markdown", "").strip():
                raise JudgeError(f"LlamaParse has no usable reference for page {number}.")

        for number in pages:
            if any(p["page_number"] == number for p in results):
                continue
            baseline = reference_pages[number]
            candidate = candidate_pages.get(number)
            if not candidate or candidate.get("error") or not candidate.get("markdown", "").strip():
                result = {
                    "page_number": number,
                    "verdict": "fail",
                    "issues": [
                        {
                            "severity": "major",
                            "category": "content",
                            "message": "No usable text was extracted for this page.",
                            "reference_quote": baseline["markdown"].strip()[:200],
                            "candidate_quote": "",
                        }
                    ],
                }
            else:
                result = await self.evaluator.compare(
                    number, baseline["markdown"], candidate["markdown"]
                )
            results.append(result)
            with session_scope() as session:
                judgment = session.get(Judgment, run_id)
                if judgment is None or judgment.input_hash != fingerprint:
                    return
                judgment.page_results = list(results)

        with session_scope() as session:
            judgment = session.get(Judgment, run_id)
            if judgment is None or judgment.input_hash != fingerprint:
                return
            judgment.status = "completed"
            judgment.verdict = overall_verdict(results)
            judgment.finished_at = utcnow()
