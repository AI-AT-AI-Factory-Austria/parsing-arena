import shutil

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from .. import documents, storage
from ..catalog import ALL_MODELS, LLAMAPARSE_ID, MODELS
from ..config import get_settings
from ..db import get_session
from ..models import Document, Experiment, ModelRun
from ..schemas import ExperimentCreate, ExperimentDetailOut, ExperimentList
from .serializers import experiment_detail_out, experiment_out

router = APIRouter(tags=["experiments"])

CATALOG_ORDER = {m.id: i for i, m in enumerate(MODELS)}


def _load(session: Session, experiment_id: str) -> Experiment:
    experiment = session.get(
        Experiment,
        experiment_id,
        options=[selectinload(Experiment.runs), selectinload(Experiment.document)],
    )
    if experiment is None:
        raise HTTPException(404, "Experiment not found.")
    return experiment


def _create(
    request: Request,
    session: Session,
    document: Document,
    model_ids: list[str],
    pages: list[int],
    cpu_only: bool = False,
) -> Experiment:
    experiment = build_experiment(request, document, model_ids, pages, cpu_only)
    session.add(experiment)
    session.commit()
    request.app.state.runner.submit(experiment.id)
    return _load(session, experiment.id)


def build_experiment(
    request: Request,
    document: Document,
    model_ids: list[str],
    pages: list[int],
    cpu_only: bool = False,
) -> Experiment:
    """Validate and construct without committing or starting any work."""
    registry = request.app.state.registry.for_mode(cpu_only)
    selected: list[str] = []
    for model_id in dict.fromkeys(model_ids):
        model = ALL_MODELS.get(model_id)
        if model is None:
            raise HTTPException(422, f"Unknown model '{model_id}'.")
        availability = registry.availability(model)
        if not availability["runnable"]:
            raise HTTPException(422, f"{model.name} cannot run: {availability['reason']}")
        max_pages = availability.get("max_pages")
        if max_pages is not None and len(pages) > max_pages:
            raise HTTPException(
                422, f"{model.name} accepts at most {max_pages} pages per run. Select fewer pages."
            )
        selected.append(model_id)
    # Open models first (catalog order), the reference parser last.
    selected.sort(key=lambda m: (m == LLAMAPARSE_ID, CATALOG_ORDER.get(m, 0)))

    experiment = Experiment(
        document_id=document.id, pages=pages, status="queued", cpu_only=cpu_only
    )
    for position, model_id in enumerate(selected):
        experiment.runs.append(
            ModelRun(
                model_id=model_id,
                position=position,
                pages_total=len(pages),
                progress={"phase": "queued"},
                meta={},
            )
        )
    return experiment


@router.post("/experiments", response_model=ExperimentDetailOut, status_code=201)
def create_experiment(
    body: ExperimentCreate, request: Request, session: Session = Depends(get_session)
) -> ExperimentDetailOut:
    document = session.get(Document, body.document_id)
    if document is None:
        raise HTTPException(404, "Document not found.")
    try:
        pages = documents.parse_page_selection(body.pages, document.page_count)
    except (documents.DocumentError, ValueError) as exc:
        raise HTTPException(422, f"Invalid page selection: {exc}") from exc
    limit = get_settings().max_pages_per_experiment
    if len(pages) > limit:
        raise HTTPException(422, f"Select at most {limit} pages per experiment.")
    experiment = _create(request, session, document, body.models, pages, body.cpu_only)
    return experiment_detail_out(experiment, request.app.state.runner)


@router.post(
    "/experiments/{experiment_id}/rerun", response_model=ExperimentDetailOut, status_code=201
)
def rerun_experiment(
    experiment_id: str, request: Request, session: Session = Depends(get_session)
) -> ExperimentDetailOut:
    previous = _load(session, experiment_id)
    experiment = _create(
        request,
        session,
        previous.document,
        [r.model_id for r in previous.runs],
        list(previous.pages),
        previous.cpu_only,
    )
    return experiment_detail_out(experiment, request.app.state.runner)


@router.get("/experiments", response_model=ExperimentList)
def list_experiments(
    request: Request, limit: int = 50, session: Session = Depends(get_session),
    standalone_only: bool = False,
) -> ExperimentList:
    stmt = (
        select(Experiment)
        .order_by(Experiment.created_at.desc())
        .limit(min(max(limit, 1), 200))
        .options(selectinload(Experiment.runs), selectinload(Experiment.document))
    )
    if standalone_only:
        stmt = stmt.where(Experiment.batch_id.is_(None))
    runner = request.app.state.runner
    return ExperimentList(experiments=[experiment_out(e, runner) for e in session.scalars(stmt)])


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetailOut)
def get_experiment(
    experiment_id: str, request: Request, session: Session = Depends(get_session)
) -> ExperimentDetailOut:
    return experiment_detail_out(_load(session, experiment_id), request.app.state.runner)


@router.post(
    "/experiments/{experiment_id}/judge", response_model=ExperimentDetailOut, status_code=202
)
async def judge_experiment(
    experiment_id: str, request: Request, session: Session = Depends(get_session)
) -> ExperimentDetailOut:
    experiment = _load(session, experiment_id)
    if not get_settings().judge_configured:
        raise HTTPException(409, "Add MISTRAL to .env to enable quality checks.")
    reference = next((r for r in experiment.runs if r.model_id == LLAMAPARSE_ID), None)
    if reference is None or reference.status != "completed":
        raise HTTPException(409, "A completed LlamaParse reference is required for judging.")
    if not any(r.model_id != LLAMAPARSE_ID and r.status == "completed" for r in experiment.runs):
        raise HTTPException(409, "Wait for a model to finish before judging its output.")
    request.app.state.judge.submit_experiment(experiment_id, retry_failed=True)
    session.expire_all()
    return experiment_detail_out(_load(session, experiment_id), request.app.state.runner)


@router.delete("/experiments/{experiment_id}", status_code=204)
def delete_experiment(
    experiment_id: str, request: Request, session: Session = Depends(get_session)
) -> Response:
    experiment = _load(session, experiment_id)
    if experiment.batch_id:
        raise HTTPException(409, "This document belongs to a batch. Delete the batch instead.")
    if experiment.status in ("queued", "running"):
        raise HTTPException(409, "Wait until the experiment has finished before deleting it.")
    if any(r.judgment and r.judgment.status in ("queued", "running") for r in experiment.runs):
        raise HTTPException(
            409, "Wait until the quality checks finish before deleting this experiment."
        )
    document = experiment.document
    session.delete(experiment)
    session.flush()
    shutil.rmtree(storage.experiment_dir(experiment_id), ignore_errors=True)
    others = session.scalar(
        select(func.count()).select_from(Experiment).where(Experiment.document_id == document.id)
    )
    if not others:
        session.delete(document)
        shutil.rmtree(storage.document_dir(document.id), ignore_errors=True)
    return Response(status_code=204)


@router.get("/runs/{run_id}/files/{path:path}")
def get_run_file(run_id: str, path: str, session: Session = Depends(get_session)) -> FileResponse:
    run = session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(404, "Run not found.")
    target = storage.safe_child(storage.run_dir(run.experiment_id, run.id), path)
    if target is None or not target.is_file():
        raise HTTPException(404, "File not found.")
    return FileResponse(target)
