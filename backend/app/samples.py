"""Discover sample files and retain their imported documents and saved comparisons.

Discovery never runs models or calls external APIs. Content hashes distinguish
updated files; removing a source file only hides it from the gallery.
"""

import hashlib
import logging
import re
import threading
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from . import documents, storage
from .config import get_settings
from .models import Document, Experiment, SampleDocument

logger = logging.getLogger(__name__)
sample_lock = threading.RLock()
EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def sample_title(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"_+p(\d+)$", r" · Page \1", name, flags=re.IGNORECASE)
    return re.sub(r"[-_]+", " ", name).strip()


def sync_samples(session: Session) -> list[SampleDocument]:
    """Import supported files once, including ones added while the server runs."""
    folder = (get_settings().data_dir / "sample-docs").resolve()
    if not folder.is_dir():
        return []
    found = []
    seen: set[str] = set()
    with sample_lock:
        for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
            if (
                not path.is_file()
                or path.suffix.lower() not in EXTENSIONS
                or not path.resolve().is_relative_to(folder)
            ):
                continue
            try:
                if path.stat().st_size > get_settings().max_upload_mb * 1024 * 1024:
                    logger.warning("Skipping oversized sample %s", path.name)
                    continue
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                sample = session.get(SampleDocument, digest)
                if sample is None:
                    # Reuse an earlier upload of exactly these bytes when available.
                    matches = session.scalars(
                        select(Document)
                        .where(Document.sha256 == digest)
                        .order_by(Document.created_at.desc())
                    )
                    document = next(
                        (d for d in matches if storage.original_path(d.id, d.extension).is_file()),
                        None,
                    )
                    if document is None:
                        with path.open("rb") as stream:
                            document = documents.ingest(
                                stream,
                                path.name,
                                max_bytes=get_settings().max_upload_mb * 1024 * 1024,
                            )
                        session.add(document)
                        session.flush()
                    sample = SampleDocument(id=digest, filename=path.name, document=document)
                    session.add(sample)
                else:
                    sample.filename = path.name
                found.append(sample)
            except (OSError, documents.DocumentError, documents.DocumentRuntimeUnavailable) as exc:
                logger.warning("Skipping unreadable sample %s: %s", path.name, exc)
        # Commit under the lock so simultaneous discovery cannot import duplicates.
        session.commit()
    return found


def reusable_experiment(
    session: Session, sample: SampleDocument, model_ids: list[str]
) -> Experiment | None:
    pages = list(range(1, sample.document.page_count + 1))
    candidates = session.scalars(
        select(Experiment)
        .join(Document)
        .where(Document.sha256 == sample.id)
        .order_by(Experiment.created_at.desc())
        .options(selectinload(Experiment.runs))
    )
    return next(
        (
            e
            for e in candidates
            if e.pages == pages and {r.model_id for r in e.runs} == set(model_ids)
        ),
        None,
    )
