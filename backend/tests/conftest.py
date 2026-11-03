import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Isolate every test run from the developer's data, models and API key.
_DATA_DIR = Path(tempfile.mkdtemp(prefix="ocr-arena-tests-"))
os.environ["DATA_DIR"] = str(_DATA_DIR)
os.environ["LOCAL_INFERENCE"] = "false"
os.environ["LLAMA_CLOUD_API_KEY"] = ""

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "samples" / "sample-report.pdf"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_data_dir():
    yield
    from app import db

    if db._engine is not None:
        db._engine.dispose()
    shutil.rmtree(_DATA_DIR, ignore_errors=True)
