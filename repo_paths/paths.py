import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_ENV = REPO_ROOT / "providers.env"
DOCS_DIR = REPO_ROOT / "docs"
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
DATA_DIR = REPO_ROOT / "data"


def get_db_path() -> Path:
    """Resolve active SQLite database path at runtime, prioritizing ZAMP_DB_PATH env var.

    - Absolute ZAMP_DB_PATH: resolved directly as an absolute path.
    - Relative ZAMP_DB_PATH: resolved relative to REPO_ROOT (independent of current working directory).
    - Default: DATA_DIR / 'zamp.db'.
    """
    env_path = os.environ.get("ZAMP_DB_PATH")
    if env_path:
        p = Path(env_path)
        return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()
    return DATA_DIR / "zamp.db"


DB_PATH = get_db_path()

