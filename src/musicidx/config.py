"""Configuration helpers for MusicIdx."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DB_PATH_ENV_VAR = "MUSICIDX_DB_PATH"
MODELS_PATH_ENV_VAR = "MUSICIDX_MODELS_PATH"
FFPROBE_PATH_ENV_VAR = "MUSICIDX_FFPROBE_PATH"
FPCALC_PATH_ENV_VAR = "MUSICIDX_FPCALC_PATH"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_MODEL_ENV_VAR = "MUSICIDX_GEMINI_MODEL"
GEMINI_BASE_URL_ENV_VAR = "MUSICIDX_GEMINI_BASE_URL"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_MODEL_ENV_VAR = "MUSICIDX_OPENAI_MODEL"
OPENAI_BASE_URL_ENV_VAR = "MUSICIDX_OPENAI_BASE_URL"
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


def resolve_executable(default_name: str, env_var: str) -> str | None:
    """Resolve an executable from an env override or PATH.

    Env overrides may be absolute paths, relative paths, or command names.
    Missing path overrides return None so callers can report a clear diagnostic.
    """
    configured = os.environ.get(env_var)
    if configured:
        if "/" not in configured:
            return shutil.which(configured)
        configured_path = Path(configured).expanduser()
        return str(configured_path) if configured_path.exists() else None
    return shutil.which(default_name)
