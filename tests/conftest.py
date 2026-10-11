import os
from pathlib import Path
import tempfile
import pytest

from app.database import initialize_database, seed_database

_SESSION_TEMP_DIR = None
_ORIG_ENV_DB = None


@pytest.fixture(scope="session", autouse=True)
def isolate_session_database():
    """
    Session-level pytest fixture ensuring tests never read or write to the live demo database.
    Initializes a temporary SQLite database seeded with TechNova sample knowledge.
    """
    global _SESSION_TEMP_DIR, _ORIG_ENV_DB
    _SESSION_TEMP_DIR = tempfile.TemporaryDirectory()
    temp_db_path = Path(_SESSION_TEMP_DIR.name) / "pytest_session.db"
    _ORIG_ENV_DB = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = str(temp_db_path)

    initialize_database()
    seed_database()

    yield str(temp_db_path)

    if _ORIG_ENV_DB is not None:
        os.environ["DATABASE_PATH"] = _ORIG_ENV_DB
    elif "DATABASE_PATH" in os.environ:
        del os.environ["DATABASE_PATH"]

    if _SESSION_TEMP_DIR:
        _SESSION_TEMP_DIR.cleanup()
