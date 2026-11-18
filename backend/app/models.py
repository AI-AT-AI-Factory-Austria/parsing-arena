import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(512))
    media_type: Mapped[str] = mapped_column(String(128))
    extension: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    page_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Batch(Base):
    """A saved comparison of selected pages from several separate documents."""

    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", order_by="Experiment.created_at"
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    batch_id: Mapped[str | None] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), index=True
    )
    # queued | running | completed | failed
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    pages: Mapped[list[int]] = mapped_column(JSON)  # selected 1-based page numbers
    cpu_only: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="experiments")
    batch: Mapped[Batch | None] = relationship(back_populates="experiments")
    runs: Mapped[list["ModelRun"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="ModelRun.position",
    )


class SampleDocument(Base):
    """A content-addressed sample and its reusable comparison."""

    __tablename__ = "sample_documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # file SHA-256
    filename: Mapped[str] = mapped_column(String(512))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL")
    )
    document: Mapped[Document] = relationship(lazy="selectin")
    experiment: Mapped[Experiment | None] = relationship(lazy="selectin")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    model_id: Mapped[str] = mapped_column(String(64))
    position: Mapped[int] = mapped_column(Integer, default=0)
    # queued | loading | running | completed | failed
    status: Mapped[str] = mapped_column(String(16), default="queued")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # One-off model load/warm-up, kept out of the parse time for a fair comparison.
    load_seconds: Mapped[float | None] = mapped_column(Float)
    # Wall-clock time spent parsing the selected pages.
    parse_seconds: Mapped[float | None] = mapped_column(Float)
    pages_done: Mapped[int] = mapped_column(Integer, default=0)
    pages_total: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    markdown: Mapped[str | None] = mapped_column(Text)
    # [{"page_number", "markdown", "seconds", "elements": [...]}]
    page_results: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    judgment: Mapped["Judgment | None"] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False, lazy="selectin"
    )


class Judgment(Base):
    """Separate table so existing databases need no destructive schema migration."""

    __tablename__ = "judgments"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), primary_key=True
    )
    reference_run_id: Mapped[str] = mapped_column(String(32))
    input_hash: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    rubric_version: Mapped[str] = mapped_column(String(64))
    # API/validation failures are distinct from a genuine extraction verdict of fail.
    status: Mapped[str] = mapped_column(String(16), default="queued")
    verdict: Mapped[str | None] = mapped_column(String(16))
    pages_total: Mapped[int] = mapped_column(Integer)
    page_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    run: Mapped[ModelRun] = relationship(back_populates="judgment")
