from pathlib import Path

from .config import get_settings


def data_dir() -> Path:
    return get_settings().data_dir


def document_dir(document_id: str) -> Path:
    return data_dir() / "documents" / document_id


def original_path(document_id: str, extension: str) -> Path:
    return document_dir(document_id) / f"original{extension}"


def page_image_path(document_id: str, page_number: int, dpi: int) -> Path:
    return document_dir(document_id) / "pages" / f"{dpi}dpi" / f"page-{page_number:04d}.png"


def thumbnail_path(document_id: str) -> Path:
    return document_dir(document_id) / "thumbnail.webp"


def run_dir(experiment_id: str, run_id: str) -> Path:
    return data_dir() / "experiments" / experiment_id / run_id


def experiment_dir(experiment_id: str) -> Path:
    return data_dir() / "experiments" / experiment_id


def safe_child(root: Path, relative: str) -> Path | None:
    """Resolve `relative` inside `root`, refusing anything that escapes it."""
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents:
        return None
    return candidate
