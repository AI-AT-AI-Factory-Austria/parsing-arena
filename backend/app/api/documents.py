from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import documents, storage
from ..config import BACKEND_DIR, get_settings
from ..db import get_session
from ..models import Document
from ..schemas import DocumentOut
from .serializers import document_out

router = APIRouter(tags=["documents"])

SAMPLE_PATH = BACKEND_DIR / "samples" / "sample-report.pdf"
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _get_document(session: Session, document_id: str) -> Document:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found.")
    return document


@router.post("/documents", response_model=DocumentOut, status_code=201)
def upload_document(file: UploadFile, session: Session = Depends(get_session)) -> DocumentOut:
    settings = get_settings()
    try:
        document = documents.ingest(
            file.file, file.filename or "", max_bytes=settings.max_upload_mb * 1024 * 1024
        )
    except documents.DocumentError as exc:
        raise HTTPException(422, str(exc)) from exc
    except documents.DocumentRuntimeUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    session.add(document)
    session.flush()
    return document_out(document)


@router.post("/documents/sample", response_model=DocumentOut, status_code=201)
def use_sample_document(session: Session = Depends(get_session)) -> DocumentOut:
    try:
        with SAMPLE_PATH.open("rb") as fh:
            document = documents.ingest(fh, SAMPLE_PATH.name, max_bytes=50 * 1024 * 1024)
    except documents.DocumentRuntimeUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    session.add(document)
    session.flush()
    return document_out(document)


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, session: Session = Depends(get_session)) -> DocumentOut:
    return document_out(_get_document(session, document_id))


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str, session: Session = Depends(get_session)) -> FileResponse:
    document = _get_document(session, document_id)
    return FileResponse(
        storage.original_path(document.id, document.extension),
        media_type=document.media_type,
        filename=document.filename,
        content_disposition_type="inline",
    )


@router.get("/documents/{document_id}/thumbnail")
def get_thumbnail(document_id: str, session: Session = Depends(get_session)) -> FileResponse:
    document = _get_document(session, document_id)
    return FileResponse(
        storage.thumbnail_path(document.id), media_type="image/webp", headers=IMMUTABLE
    )


@router.get("/documents/{document_id}/pages/{page_number}")
def get_page_image(
    document_id: str, page_number: int, session: Session = Depends(get_session),
    thumbnail: bool = False,
) -> FileResponse:
    document = _get_document(session, document_id)
    if not 1 <= page_number <= document.page_count:
        raise HTTPException(404, "Page not found.")
    try:
        path = documents.page_image(
            document, page_number, dpi=48 if thumbnail else get_settings().render_dpi,
            max_pixels=160_000 if thumbnail else None,
        )
    except documents.DocumentRuntimeUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return FileResponse(path, media_type="image/png", headers=IMMUTABLE)
