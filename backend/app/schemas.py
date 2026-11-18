from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field


def _as_utc(value: datetime | None) -> datetime | None:
    # SQLite drops tzinfo; every timestamp is stored in UTC.
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class ScoresOut(BaseModel):
    overall: float
    tables: float
    charts: float
    content: float
    formatting: float
    grounding: float


class AvailabilityOut(BaseModel):
    runnable: bool
    backend: str | None = None
    reason: str | None = None
    loaded: bool | None = None
    runtime_label: str | None = None
    max_pages: int | None = None


class ModelOut(BaseModel):
    id: str
    name: str
    publisher: str
    params: str
    params_b: float | None
    license: str
    kind: str
    implemented: bool
    hf_repo: str | None
    hf_url: str | None
    revision: str | None
    verified: bool
    lineage: str
    pipeline: str
    specialty: str
    scores: ScoresOut | None
    availability: AvailabilityOut


class ModelsOut(BaseModel):
    cpu_only: bool = False
    models: list[ModelOut]
    reference: ModelOut
    llamaparse_credits_per_page: int
    llamaparse_usd_per_credit: float
    source: str


class HealthOut(BaseModel):
    status: str
    version: str
    llamaparse_configured: bool
    judge_configured: bool
    local_device: dict[str, Any] | None
    runtime_profile: str = "native"
    runtime_profile_error: str | None = None
    max_upload_mb: int
    max_pages_per_experiment: int


class DocumentOut(BaseModel):
    id: str
    filename: str
    media_type: str
    size_bytes: int
    page_count: int
    sha256: str
    created_at: UtcDatetime
    thumbnail_url: str
    file_url: str


class PageResultOut(BaseModel):
    page_number: int
    markdown: str
    seconds: float | None = None
    elements: list[dict[str, Any]] | None = None
    error: str | None = None


Verdict = Literal["excellent", "pass", "fail"]


class JudgeIssueOut(BaseModel):
    severity: Literal["minor", "major"]
    category: str
    message: str
    reference_quote: str
    candidate_quote: str


class PageJudgmentOut(BaseModel):
    page_number: int
    verdict: Verdict
    issues: list[JudgeIssueOut]


class JudgmentOut(BaseModel):
    status: Literal["queued", "running", "completed", "failed"]
    verdict: Verdict | None
    model: str
    rubric_version: str
    outdated: bool = False
    reference_run_id: str
    pages_done: int
    pages_total: int
    pages: list[PageJudgmentOut]
    finished_at: UtcDatetime | None
    error: str | None


class RunOut(BaseModel):
    id: str
    model_id: str
    model_name: str
    kind: str
    status: str
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None
    load_seconds: float | None
    parse_seconds: float | None
    seconds_per_page: float | None
    pages_done: int
    pages_total: int
    progress: dict[str, Any]
    meta: dict[str, Any]
    error: str | None
    judgment: JudgmentOut | None


class RunDetailOut(RunOut):
    markdown: str | None
    pages: list[PageResultOut]


class ExperimentOut(BaseModel):
    id: str
    batch_id: str | None = None
    status: str
    created_at: UtcDatetime
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None
    pages: list[int]
    cpu_only: bool = False
    error: str | None
    queue_position: int | None
    document: DocumentOut
    runs: list[RunOut]
    judge_configured: bool
    judge_model: str


class ExperimentDetailOut(ExperimentOut):
    runs: list[RunDetailOut]  # type: ignore[assignment]


class ExperimentList(BaseModel):
    experiments: list[ExperimentOut]


class ExperimentCreate(BaseModel):
    document_id: str
    models: list[str] = Field(min_length=1)
    pages: str | None = Field(default=None, description="e.g. '1-3,5'; empty = all pages")
    cpu_only: bool = False


MAX_BATCH_PAGES = 12


class BatchDocumentCreate(BaseModel):
    document_id: str
    pages: list[Annotated[int, Field(strict=True, ge=1)]] = Field(
        min_length=1, max_length=MAX_BATCH_PAGES
    )


class BatchCreate(BaseModel):
    documents: list[BatchDocumentCreate] = Field(min_length=1, max_length=MAX_BATCH_PAGES)
    models: list[str] = Field(min_length=1)
    cpu_only: bool = False


class ModelAggregateOut(BaseModel):
    model_id: str
    model_name: str
    kind: str
    status: str
    pages_total: int
    pages_completed: int
    pages_failed: int
    timed_pages: int
    parse_seconds: float | None
    seconds_per_page: float | None
    judged_pages: int
    excellent: int
    passed: int
    failed: int
    pass_rate: float | None


class BatchOut(BaseModel):
    id: str
    created_at: UtcDatetime
    status: str
    cpu_only: bool
    documents: list[DocumentOut]
    page_count: int
    model_count: int
    completed_runs: int
    total_runs: int
    judging: bool


class BatchDetailOut(BatchOut):
    experiments: list[ExperimentDetailOut]
    aggregates: list[ModelAggregateOut]


class BatchList(BaseModel):
    batches: list[BatchOut]


class SampleOut(BaseModel):
    id: str
    title: str
    document: DocumentOut
    experiment_id: str | None
    status: Literal["unprepared", "queued", "running", "completed", "failed"]
    model_count: int
    completed_models: int
    judging: bool


class SampleList(BaseModel):
    samples: list[SampleOut]


class SamplePrepare(BaseModel):
    models: list[str] = Field(min_length=1)
    refresh: bool = False
