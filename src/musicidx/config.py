"""Configuration helpers for MusicIdx."""

from __future__ import annotations

import os
from pathlib import Path

DB_PATH_ENV_VAR = "MUSICIDX_DB_PATH"
MODELS_PATH_ENV_VAR = "MUSICIDX_MODELS_PATH"
DEFAULT_DB_FILENAME = "musicidx.sqlite"
DEFAULT_MODELS_DIRNAME = ".musicidx-models"


def default_db_path() -> Path:
    """Return the default project-local database path.

    By default the database lives in the current working directory so the MVP is
    easy to inspect during development. It can be overridden with
    MUSICIDX_DB_PATH or with per-command ``--db`` options.
    """
    configured = os.environ.get(DB_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / DEFAULT_DB_FILENAME


def resolve_db_path(path: Path | str | None = None) -> Path:
    """Resolve an explicit DB path or fall back to the configured default."""
    if path is None:
        return default_db_path()
    return Path(path).expanduser()


def default_models_path() -> Path:
    """Return the default local model directory path."""
    configured = os.environ.get(MODELS_PATH_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / DEFAULT_MODELS_DIRNAME


def resolve_models_path(path: Path | str | None = None) -> Path:
    """Resolve an explicit models path or fall back to the configured default."""
    if path is None:
        return default_models_path()
    return Path(path).expanduser()
