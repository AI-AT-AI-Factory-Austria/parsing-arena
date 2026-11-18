import shutil

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from .. import storage
from ..batches import aggregate_models, combined_status
from ..config import get_settings
from ..db import get_session
from ..models import Batch, Document, Experiment, ModelRun, SampleDocument
from ..schemas import (
    MAX_BATCH_PAGES,
    BatchCreate,
    BatchDetailOut,
    BatchDocumentCreate,
    BatchList,
    BatchOut,
)
from .experiments import build_experiment
from .serializers import document_out, experiment_detail_out

router = APIRouter(tags=["batches"])


def _query():
    return select(Batch).options(
        selectinload(Batch.experiments).selectinload(Experiment.document),
        selectinload(Batch.experiments).selectinload(Experiment.runs)
        .selectinload(ModelRun.judgment),
    )


def _load(session: Session, batch_id: str) -> Batch:
    batch = session.scalar(_query().where(Batch.id == batch_id))
    if batch is None:
        raise HTTPException(404, "Batch not found.")
    return batch


def batch_out(batch: Batch) -> BatchOut:
    experiments = batch.experiments
    runs = [r for e in experiments for r in e.runs]
    return BatchOut(
        id=batch.id, created_at=batch.created_at,
        status=combined_status([e.status for e in experiments]),
        cpu_only=experiments[0].cpu_only,
        documents=[document_out(e.document) for e in experiments],
        page_count=sum(len(e.pages) for e in experiments),
        model_count=len({r.model_id for r in runs}),
        completed_runs=sum(r.status in ("completed", "failed") for r in runs),
        total_runs=len(runs),
        judging=any(r.judgment and r.judgment.status in ("queued", "running") for r in runs),
    )


def batch_detail_out(batch: Batch, request: Request) -> BatchDetailOut:
    return BatchDetailOut(
        **batch_out(batch).model_dump(),
        experiments=[experiment_detail_out(e, request.app.state.runner) for e in batch.experiments],
        aggregates=aggregate_models(batch.experiments),
    )


def _create(body: BatchCreate, request: Request, session: Session) -> Batch:
    batch = Batch()
    seen: set[tuple[str, int]] = set()
    by_hash: dict[str, tuple[Document, set[int]]] = {}
    for item in body.documents:
        document = session.get(Document, item.document_id)
        if document is None:
            raise HTTPException(404, "Document not found.")
        if any(n > document.page_count for n in item.pages):
            raise HTTPException(422, f"Invalid page number for {document.filename}.")
        pages = set(item.pages)
        keys = {(document.sha256, n) for n in pages}
        seen.update(keys)
        # Repeated uploads/selections of the same source never run a page twice.
        if document.sha256 in by_hash:
            by_hash[document.sha256][1].update(pages)
        else:
            by_hash[document.sha256] = (document, pages)
    limit = min(MAX_BATCH_PAGES, get_settings().max_pages_per_experiment)
    if len(seen) > limit:
        raise HTTPException(422, f"Select at most {limit} unique pages across all documents.")
    for document, pages in by_hash.values():
        batch.experiments.append(build_experiment(
            request, document, body.models, sorted(pages), body.cpu_only
        ))
    # Validate the whole selection before committing or queueing any document.
    session.add(batch)
    session.commit()
    for experiment in batch.experiments:
        request.app.state.runner.submit(experiment.id)
    return _load(session, batch.id)


@router.post("/batches", response_model=BatchDetailOut, status_code=201)
def create_batch(
    body: BatchCreate, request: Request, session: Session = Depends(get_session)
) -> BatchDetailOut:
    return batch_detail_out(_create(body, request, session), request)


@router.get("/batches", response_model=BatchList)
def list_batches(limit: int = 50, session: Session = Depends(get_session)) -> BatchList:
    batches = session.scalars(
        _query().order_by(Batch.created_at.desc()).limit(min(max(limit, 1), 200))
    )
    return BatchList(batches=[batch_out(b) for b in batches])


@router.get("/batches/{batch_id}", response_model=BatchDetailOut)
def get_batch(
    batch_id: str, request: Request, session: Session = Depends(get_session)
) -> BatchDetailOut:
    return batch_detail_out(_load(session, batch_id), request)


@router.post("/batches/{batch_id}/rerun", response_model=BatchDetailOut, status_code=201)
def rerun_batch(
    batch_id: str, request: Request, session: Session = Depends(get_session)
) -> BatchDetailOut:
    previous = _load(session, batch_id)
    body = BatchCreate(
        documents=[BatchDocumentCreate(document_id=e.document_id, pages=e.pages)
                   for e in previous.experiments],
        models=[r.model_id for r in previous.experiments[0].runs],
        cpu_only=previous.experiments[0].cpu_only,
    )
    return batch_detail_out(_create(body, request, session), request)


@router.delete("/batches/{batch_id}", status_code=204)
def delete_batch(batch_id: str, session: Session = Depends(get_session)) -> Response:
    batch = _load(session, batch_id)
    if batch_out(batch).status in ("queued", "running") or batch_out(batch).judging:
        raise HTTPException(409, "Wait until parsing and quality checks finish before deleting.")
    documents = {e.document_id: e.document for e in batch.experiments}
    experiment_ids = [e.id for e in batch.experiments]
    session.delete(batch)
    session.flush()
    for experiment_id in experiment_ids:
        shutil.rmtree(storage.experiment_dir(experiment_id), ignore_errors=True)
    for document_id, document in documents.items():
        remaining = session.scalar(select(func.count()).select_from(Experiment).where(
            Experiment.document_id == document_id
        ))
        sample = session.scalar(select(SampleDocument.id).where(
            SampleDocument.document_id == document_id
        ))
        if not remaining and sample is None:
            session.delete(document)
            shutil.rmtree(storage.document_dir(document_id), ignore_errors=True)
    return Response(status_code=204)
