from __future__ import annotations

import os
from pathlib import Path

from repo_paths import DATA_DIR, DB_PATH, DOCS_DIR, PROVIDERS_ENV, REPO_ROOT, SRC_DIR, TESTS_DIR
from src.config.settings import GROQ_LARGE, GROQ_QWEN, GROQ_SMALL, init_env


def test_repo_paths():
    assert REPO_ROOT.is_dir()
    assert (REPO_ROOT / "src").resolve() == SRC_DIR.resolve()
    assert (REPO_ROOT / "docs").resolve() == DOCS_DIR.resolve()
    assert (REPO_ROOT / "tests").resolve() == TESTS_DIR.resolve()
    assert (REPO_ROOT / "data").resolve() == DATA_DIR.resolve()
    assert PROVIDERS_ENV.name == "providers.env"
    assert DB_PATH == DATA_DIR / "zamp.db"


def test_config_settings_and_init_env():
    init_env()
    # GROQ_API_KEY should be loaded into os.environ
    assert "GROQ_API_KEY" in os.environ
    assert len(os.environ["GROQ_API_KEY"]) > 0

    assert GROQ_LARGE == "groq_gpt_oss_120b"
    assert GROQ_SMALL == "groq_gpt_oss_20b"
    assert GROQ_QWEN == "groq_qwen3_8_27b"
