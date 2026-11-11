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
    # When data/zamp.db has 0 rows and open_inquiry.db does not exist, default is open_inquiry.db
    assert DB_PATH == DATA_DIR / "open_inquiry.db"


def test_config_settings_and_init_env():
    init_env()
    # GROQ_API_KEY should be loaded into os.environ
    assert "GROQ_API_KEY" in os.environ
    assert len(os.environ["GROQ_API_KEY"]) > 0

    assert GROQ_LARGE == "groq_gpt_oss_120b"
    assert GROQ_SMALL == "groq_gpt_oss_20b"
    assert GROQ_QWEN == "groq_qwen3_8_27b"


def test_db_path_precedence_and_compatibility(monkeypatch, tmp_path):
    import sqlite3
    import warnings
    from repo_paths import get_db_path

    # Clean env
    monkeypatch.delenv("OPEN_INQUIRY_DB_PATH", raising=False)
    monkeypatch.delenv("ZAMP_DB_PATH", raising=False)

    # 1. OPEN_INQUIRY_DB_PATH precedence
    custom_oi = tmp_path / "custom_oi.db"
    monkeypatch.setenv("OPEN_INQUIRY_DB_PATH", str(custom_oi))
    assert get_db_path() == custom_oi.resolve()

    # 2. ZAMP_DB_PATH backward compatibility with warning
    monkeypatch.delenv("OPEN_INQUIRY_DB_PATH")
    custom_legacy = tmp_path / "legacy.db"
    monkeypatch.setenv("ZAMP_DB_PATH", str(custom_legacy))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        resolved = get_db_path()
        assert resolved == custom_legacy.resolve()
        assert len(recorded) >= 1
        assert issubclass(recorded[-1].category, DeprecationWarning)
        assert "OPEN_INQUIRY_DB_PATH" in str(recorded[-1].message)

    # 3. OPEN_INQUIRY_DB_PATH takes precedence over ZAMP_DB_PATH
    monkeypatch.setenv("OPEN_INQUIRY_DB_PATH", str(custom_oi))
    monkeypatch.setenv("ZAMP_DB_PATH", str(custom_legacy))
    assert get_db_path() == custom_oi.resolve()

    # 4. Explicit db_path argument takes precedence over environment
    explicit_db = tmp_path / "explicit.db"
    assert get_db_path(explicit_db) == explicit_db.resolve()

    # 5. :memory: handling
    assert str(get_db_path(":memory:")) == ":memory:"


def test_db_path_legacy_preservation_and_preference(monkeypatch, tmp_path):
    import sqlite3
    from repo_paths import paths

    monkeypatch.delenv("OPEN_INQUIRY_DB_PATH", raising=False)
    monkeypatch.delenv("ZAMP_DB_PATH", raising=False)

    fake_data = tmp_path / "fake_data"
    fake_data.mkdir()
    monkeypatch.setattr(paths, "DATA_DIR", fake_data)

    # Case A: Neither file exists -> defaults to open_inquiry.db
    assert paths.get_db_path() == fake_data / "open_inquiry.db"

    # Case B: Empty legacy zamp.db exists -> defaults to open_inquiry.db
    legacy_file = fake_data / "zamp.db"
    conn = sqlite3.connect(legacy_file)
    conn.execute("CREATE TABLE prospects (prospect_id TEXT PRIMARY KEY);")
    conn.close()
    assert paths.get_db_path() == fake_data / "open_inquiry.db"

    # Case C: Populated legacy zamp.db exists -> preserved and returned
    conn = sqlite3.connect(legacy_file)
    conn.execute("INSERT INTO prospects (prospect_id) VALUES ('p-123');")
    conn.commit()
    conn.close()
    assert paths.get_db_path() == legacy_file

    # Case D: Both files exist (even if legacy is populated) -> prefers new open_inquiry.db
    new_file = fake_data / "open_inquiry.db"
    new_file.touch()
    assert paths.get_db_path() == new_file

