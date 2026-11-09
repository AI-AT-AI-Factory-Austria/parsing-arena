from ..catalog import ArenaModel, get_model
from ..config import get_settings
from ..engines.registry import EngineRegistry
from ..models import Document, Experiment, ModelRun
from ..runner import Runner
from ..schemas import (
    AvailabilityOut,
    DocumentOut,
    ExperimentDetailOut,
    ExperimentOut,
    JudgmentOut,
    ModelOut,
    PageResultOut,
    RunDetailOut,
    RunOut,
    ScoresOut,
)


def model_out(model: ArenaModel, registry: EngineRegistry) -> ModelOut:
    return ModelOut(
        id=model.id,
        name=model.name,
        publisher=model.publisher,
        params=model.params,
        params_b=model.params_b,
        license=model.license,
        kind=model.kind,
        implemented=model.implemented,
        hf_repo=model.hf_repo,
        hf_url=f"https://huggingface.co/{model.hf_repo}" if model.hf_repo else None,
        revision=model.revision,
        verified=model.verified,
        lineage=model.lineage,
        pipeline=model.pipeline,
        scores=ScoresOut(**vars(model.scores)) if model.scores else None,
        availability=AvailabilityOut(**registry.availability(model)),
    )


def document_out(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        media_type=document.media_type,
        size_bytes=document.size_bytes,
        page_count=document.page_count,
        created_at=document.created_at,
        thumbnail_url=f"/api/documents/{document.id}/thumbnail",
        file_url=f"/api/documents/{document.id}/file",
    )


def _run_fields(run: ModelRun) -> dict:
    model = get_model(run.model_id)
    per_page = None
    if run.parse_seconds is not None and run.pages_done:
        per_page = round(run.parse_seconds / run.pages_done, 3)
    return dict(
        id=run.id,
        model_id=run.model_id,
        model_name=model.name if model else run.model_id,
        kind=model.kind if model else "open",
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        load_seconds=run.load_seconds,
        parse_seconds=run.parse_seconds,
        seconds_per_page=per_page,
        pages_done=run.pages_done,
        pages_total=run.pages_total,
        progress=run.progress or {},
        meta=run.meta or {},
        error=run.error,
        judgment=(
            JudgmentOut(
                status=run.judgment.status,
                verdict=run.judgment.verdict,
                model=run.judgment.model,
                rubric_version=run.judgment.rubric_version,
                reference_run_id=run.judgment.reference_run_id,
                pages_done=len(run.judgment.page_results),
                pages_total=run.judgment.pages_total,
                pages=run.judgment.page_results,
                finished_at=run.judgment.finished_at,
                error=run.judgment.error,
            )
            if run.judgment
            else None
        ),
    )


def run_out(run: ModelRun) -> RunOut:
    return RunOut(**_run_fields(run))


def run_detail_out(run: ModelRun) -> RunDetailOut:
    return RunDetailOut(
        **_run_fields(run),
        markdown=run.markdown,
        pages=[PageResultOut(**page) for page in run.page_results or []],
    )


def experiment_out(experiment: Experiment, runner: Runner) -> ExperimentOut:
    return ExperimentOut(
        id=experiment.id,
        status=experiment.status,
        created_at=experiment.created_at,
        started_at=experiment.started_at,
        finished_at=experiment.finished_at,
        pages=list(experiment.pages),
        error=experiment.error,
        queue_position=runner.queue_position(experiment.id),
        document=document_out(experiment.document),
        runs=[run_out(run) for run in experiment.runs],
        judge_configured=get_settings().judge_configured,
        judge_model=get_settings().judge_model,
    )


def experiment_detail_out(experiment: Experiment, runner: Runner) -> ExperimentDetailOut:
    summary = experiment_out(experiment, runner)
    return ExperimentDetailOut(
        **summary.model_dump(exclude={"runs", "document"}),
        document=summary.document,
        runs=[run_detail_out(run) for run in experiment.runs],
    )
