import random

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..models import SampleDocument
from ..samples import reusable_experiment, sample_lock, sample_title, sync_samples
from ..schemas import ExperimentDetailOut, SampleList, SampleOut, SamplePrepare
from .experiments import _create
from .serializers import document_out, experiment_detail_out

router = APIRouter(tags=["samples"])


def sample_out(sample: SampleDocument) -> SampleOut:
    experiment = sample.experiment
    runs = experiment.runs if experiment else []
    return SampleOut(
        id=sample.id,
        title=sample_title(sample.filename),
        document=document_out(sample.document),
        experiment_id=sample.experiment_id,
        status=experiment.status if experiment else "unprepared",
        model_count=len(runs),
        completed_models=sum(r.status == "completed" for r in runs),
        judging=any(r.judgment and r.judgment.status in ("queued", "running") for r in runs),
    )


@router.get("/samples", response_model=SampleList)
def list_samples(session: Session = Depends(get_session)) -> SampleList:
    return SampleList(samples=[sample_out(sample) for sample in sync_samples(session)])


@router.get("/samples/random", response_model=SampleOut)
def random_sample(response: Response, session: Session = Depends(get_session)) -> SampleOut:
    response.headers["Cache-Control"] = "no-store"
    samples = [sample_out(sample) for sample in sync_samples(session)]
    ready = [s for s in samples if s.status == "completed"]
    available = ready or [s for s in samples if s.experiment_id]
    if not available:
        raise HTTPException(409, "No saved sample results are available yet.")
    return random.choice(available)


@router.post("/samples/{sample_id}/prepare", response_model=ExperimentDetailOut)
def prepare_sample(
    sample_id: str, body: SamplePrepare, request: Request, session: Session = Depends(get_session)
) -> ExperimentDetailOut:
    # Explicit preparation only. Opening a gallery card never starts/restarts a run.
    with sample_lock:
        sample = next((s for s in sync_samples(session) if s.id == sample_id), None)
        if sample is None:
            raise HTTPException(404, "Sample document not found.")
        experiment = sample.experiment
        if body.refresh and experiment and experiment.status in ("queued", "running"):
            raise HTTPException(409, "This sample is already being prepared.")
        if body.refresh:
            experiment = None
        elif experiment is None:
            experiment = reusable_experiment(session, sample, body.models)
        if experiment is None:
            pages = list(range(1, sample.document.page_count + 1))
            if len(pages) > get_settings().max_pages_per_experiment:
                raise HTTPException(422, "This sample exceeds the page limit per experiment.")
            experiment = _create(request, session, sample.document, body.models, pages)
        sample.experiment = experiment
        session.commit()
        return experiment_detail_out(experiment, request.app.state.runner)
