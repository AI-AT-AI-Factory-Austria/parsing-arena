from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

        @event.listens_for(_engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        _session_factory = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    from . import models  # noqa: F401  (registers tables)

    Base.metadata.create_all(get_engine())
    # Additive migration for existing SQLite installations. Historical runs keep
    # their original execution mode and all documents/results stay untouched.
    engine = get_engine()
    if "cpu_only" not in {c["name"] for c in inspect(engine).get_columns("experiments")}:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "ALTER TABLE experiments ADD COLUMN cpu_only BOOLEAN NOT NULL DEFAULT 0"
            )
    if "batch_id" not in {c["name"] for c in inspect(engine).get_columns("experiments")}:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "ALTER TABLE experiments ADD COLUMN batch_id VARCHAR(32) "
                "REFERENCES batches(id) ON DELETE CASCADE"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_experiments_batch_id ON experiments (batch_id)"
            )


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session
