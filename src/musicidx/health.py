"""Index readiness/health checks for MusicIdx databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from musicidx.analyzer.embeddings import EMBEDDING_KIND
from musicidx.db import table_exists

DEBUG_MODELS_PATH_MARKER = "desktop/src-tauri/target/debug/resources/models"
DERIVED_TAG_SOURCE = "derived:features"


def build_index_health(
    conn: sqlite3.Connection,
    *,
    db_path: Path | str,
    models_path: Path | str,
    semantic_model: str,
) -> dict[str, Any]:
    """Return a compact index-readiness report for CLI/UI diagnostics."""
    db_path_obj = Path(db_path).expanduser().resolve()
    models_path_obj = Path(models_path).expanduser().resolve()

    tracks = _track_counts(conn)
    active_tracks = int(tracks["active"])
    audio_features = _coverage(
        conn,
        "audio_features",
        "track_id",
        active_tracks=active_tracks,
    )
    derived_tags = _derived_tag_health(conn, active_tracks=active_tracks)
    context_fit = _context_fit_health(conn, active_tracks=active_tracks)
    profiles = _profile_health(conn, active_tracks=active_tracks)
    embeddings = _embedding_health(
        conn,
        active_tracks=active_tracks,
        semantic_model=semantic_model,
    )

    config = {
        "db_path": str(db_path_obj),
        "models_path": str(models_path_obj),
        "semantic_model": semantic_model,
        "db_path_inside_models_path": _path_inside(db_path_obj, models_path_obj),
        "models_path_debug_resources": DEBUG_MODELS_PATH_MARKER in str(models_path_obj),
        "models_path_exists": models_path_obj.exists(),
    }
    warnings = _warnings(
        config, tracks, audio_features, derived_tags, context_fit, profiles, embeddings
    )
    return {
        "db_path": str(db_path_obj),
        "models_path": str(models_path_obj),
        "semantic_model": semantic_model,
        "ready": not any(warning["severity"] == "error" for warning in warnings),
        "config": config,
        "tracks": tracks,
        "audio_features": audio_features,
        "derived_tags": derived_tags,
        "context_fit": context_fit,
        "profiles": profiles,
        "embeddings": embeddings,
        "warnings": warnings,
        "recommended_actions": _recommended_actions(warnings),
    }


def _track_counts(conn: sqlite3.Connection) -> dict[str, int]:
    if not table_exists(conn, "tracks"):
        return {
            "total": 0,
            "active": 0,
            "missing": 0,
            "failed": 0,
            "quarantined": 0,
        }
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN missing_at IS NULL AND quarantined_at IS NULL THEN 1 ELSE 0 END)
                AS active,
            SUM(CASE WHEN missing_at IS NOT NULL THEN 1 ELSE 0 END) AS missing,
            SUM(CASE WHEN last_error IS NOT NULL OR error_count > 0 THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN quarantined_at IS NOT NULL THEN 1 ELSE 0 END) AS quarantined
        FROM tracks
        """
    ).fetchone()
    return {
        key: int(row[key] or 0) for key in ["total", "active", "missing", "failed", "quarantined"]
    }


def _coverage(
    conn: sqlite3.Connection,
    table_name: str,
    track_id_column: str,
    *,
    active_tracks: int,
) -> dict[str, Any]:
    if active_tracks <= 0 or not table_exists(conn, table_name):
        return {"count": 0, "tracks": 0, "coverage": 0.0, "missing": active_tracks}
    count = int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT x.{track_id_column}) AS tracks
        FROM {table_name} x
        JOIN tracks t ON t.id = x.{track_id_column}
        WHERE t.missing_at IS NULL AND t.quarantined_at IS NULL
        """
    ).fetchone()
    tracks = int(row["tracks"] or 0)
    return {
        "count": count,
        "tracks": tracks,
        "coverage": _ratio(tracks, active_tracks),
        "missing": max(0, active_tracks - tracks),
    }


def _derived_tag_health(conn: sqlite3.Connection, *, active_tracks: int) -> dict[str, Any]:
    if active_tracks <= 0 or not table_exists(conn, "track_tags"):
        return {
            "count": 0,
            "tracks": 0,
            "coverage": 0.0,
            "missing": active_tracks,
            "source": DERIVED_TAG_SOURCE,
        }
    row = conn.execute(
        """
        SELECT COUNT(*) AS count, COUNT(DISTINCT tt.track_id) AS tracks
        FROM track_tags tt
        JOIN tracks t ON t.id = tt.track_id
        WHERE tt.source = ?
          AND t.missing_at IS NULL
          AND t.quarantined_at IS NULL
        """,
        (DERIVED_TAG_SOURCE,),
    ).fetchone()
    tracks = int(row["tracks"] or 0)
    return {
        "source": DERIVED_TAG_SOURCE,
        "count": int(row["count"] or 0),
        "tracks": tracks,
        "coverage": _ratio(tracks, active_tracks),
        "missing": max(0, active_tracks - tracks),
    }


def _context_fit_health(conn: sqlite3.Connection, *, active_tracks: int) -> dict[str, Any]:
    return _coverage(conn, "track_context_fit", "track_id", active_tracks=active_tracks)


def _profile_health(conn: sqlite3.Connection, *, active_tracks: int) -> dict[str, Any]:
    if active_tracks <= 0 or not table_exists(conn, "track_profiles"):
        return {
            "count": 0,
            "tracks": 0,
            "coverage": 0.0,
            "missing": active_tracks,
            "schema_v2": 0,
            "with_embedding_text": 0,
            "with_source_fingerprint": 0,
        }
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS count,
            COUNT(DISTINCT p.track_id) AS tracks,
            SUM(CASE WHEN p.profile_schema_version = 2 THEN 1 ELSE 0 END) AS schema_v2,
            SUM(CASE WHEN COALESCE(p.embedding_text, '') != '' THEN 1 ELSE 0 END)
                AS with_embedding_text,
            SUM(CASE WHEN COALESCE(p.source_fingerprint, '') != '' THEN 1 ELSE 0 END)
                AS with_source_fingerprint
        FROM track_profiles p
        JOIN tracks t ON t.id = p.track_id
        WHERE t.missing_at IS NULL AND t.quarantined_at IS NULL
        """
    ).fetchone()
    tracks = int(row["tracks"] or 0)
    return {
        "count": int(row["count"] or 0),
        "tracks": tracks,
        "coverage": _ratio(tracks, active_tracks),
        "missing": max(0, active_tracks - tracks),
        "schema_v2": int(row["schema_v2"] or 0),
        "schema_v2_coverage": _ratio(int(row["schema_v2"] or 0), active_tracks),
        "with_embedding_text": int(row["with_embedding_text"] or 0),
        "with_source_fingerprint": int(row["with_source_fingerprint"] or 0),
    }


def _embedding_health(
    conn: sqlite3.Connection,
    *,
    active_tracks: int,
    semantic_model: str,
) -> dict[str, Any]:
    if (
        active_tracks <= 0
        or not table_exists(conn, "embeddings")
        or not table_exists(conn, "track_profiles")
    ):
        return {
            "model": semantic_model,
            "count": 0,
            "tracks": 0,
            "coverage": 0.0,
            "current": 0,
            "stale": 0,
            "missing": active_tracks,
            "available_models": [],
        }
    available_models = [
        str(row["model"])
        for row in conn.execute(
            "SELECT DISTINCT model FROM embeddings WHERE kind = ? ORDER BY model",
            (EMBEDDING_KIND,),
        ).fetchall()
    ]
    selected_model = _select_model(semantic_model, available_models)
    if selected_model is None:
        return {
            "model": semantic_model,
            "selected_model": None,
            "count": 0,
            "tracks": 0,
            "coverage": 0.0,
            "current": 0,
            "stale": 0,
            "missing": active_tracks,
            "available_models": available_models,
        }
    row = conn.execute(
        """
        SELECT
            COUNT(e.track_id) AS count,
            COUNT(DISTINCT e.track_id) AS tracks,
            SUM(CASE WHEN e.text = COALESCE(p.embedding_text, p.profile_text) THEN 1 ELSE 0 END)
                AS current,
            SUM(CASE WHEN e.text != COALESCE(p.embedding_text, p.profile_text) THEN 1 ELSE 0 END)
                AS stale
        FROM track_profiles p
        JOIN tracks t ON t.id = p.track_id
        LEFT JOIN embeddings e
          ON e.track_id = p.track_id
         AND e.kind = ?
         AND e.model = ?
        WHERE t.missing_at IS NULL AND t.quarantined_at IS NULL
        """,
        (EMBEDDING_KIND, selected_model),
    ).fetchone()
    tracks = int(row["tracks"] or 0)
    current = int(row["current"] or 0)
    stale = int(row["stale"] or 0)
    return {
        "model": semantic_model,
        "selected_model": selected_model,
        "count": int(row["count"] or 0),
        "tracks": tracks,
        "coverage": _ratio(tracks, active_tracks),
        "current": current,
        "current_coverage": _ratio(current, active_tracks),
        "stale": stale,
        "missing": max(0, active_tracks - tracks),
        "available_models": available_models,
    }


def _warnings(
    config: dict[str, Any],
    tracks: dict[str, int],
    audio_features: dict[str, Any],
    derived_tags: dict[str, Any],
    context_fit: dict[str, Any],
    profiles: dict[str, Any],
    embeddings: dict[str, Any],
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if config["db_path_inside_models_path"]:
        warnings.append(
            _warning(
                "db_path_inside_models_path", "error", "DB path is inside the models directory."
            )
        )
    if config["models_path_debug_resources"]:
        warnings.append(
            _warning(
                "models_path_debug_resources",
                "warning",
                "Models path points to Tauri debug resources.",
            )
        )
    if not config["models_path_exists"]:
        warnings.append(_warning("models_path_missing", "warning", "Models path does not exist."))
    if tracks["total"] <= 0:
        warnings.append(_warning("no_tracks", "error", "No tracks are indexed."))
    if tracks["failed"] > 0 or tracks["quarantined"] > 0:
        warnings.append(
            _warning(
                "failed_or_quarantined_tracks", "warning", "Some tracks failed or are quarantined."
            )
        )
    if audio_features["missing"] > 0:
        warnings.append(
            _warning(
                "missing_audio_features",
                "warning",
                "Some active tracks are missing audio features.",
            )
        )
    if derived_tags["missing"] > 0:
        warnings.append(
            _warning(
                "missing_derived_tags",
                "warning",
                "Some active tracks are missing derived feature tags.",
            )
        )
    if context_fit["missing"] > 0:
        warnings.append(
            _warning(
                "missing_context_fit",
                "warning",
                "Some active tracks are missing context-fit scores.",
            )
        )
    if profiles["missing"] > 0 or profiles["schema_v2"] < tracks["active"]:
        warnings.append(
            _warning(
                "profiles_not_v2", "warning", "Some active profiles are missing or not schema v2."
            )
        )
    if profiles["with_embedding_text"] < profiles["tracks"]:
        warnings.append(
            _warning(
                "missing_embedding_text", "warning", "Some profiles lack semantic embedding text."
            )
        )
    if embeddings["missing"] > 0:
        warnings.append(
            _warning(
                "missing_embeddings",
                "warning",
                "Some active profiles are missing embeddings for the selected model.",
            )
        )
    if embeddings["stale"] > 0:
        warnings.append(_warning("stale_embeddings", "warning", "Some embeddings are stale."))
    if embeddings.get("selected_model") is None and tracks["active"] > 0:
        warnings.append(
            _warning(
                "semantic_model_not_indexed",
                "warning",
                "Selected semantic model has no stored embeddings.",
            )
        )
    return warnings


def _recommended_actions(warnings: list[dict[str, str]]) -> list[str]:
    actions_by_code = {
        "db_path_inside_models_path": (
            "Move the SQLite DB outside the models directory and update Settings."
        ),
        "models_path_debug_resources": (
            "Use the repo .musicidx-models directory for local development."
        ),
        "models_path_missing": "Choose an existing models directory in Settings.",
        "no_tracks": "Run Scan files and full indexing.",
        "failed_or_quarantined_tracks": (
            "Inspect Failed tracks; fix/remove bad files or ignore them."
        ),
        "missing_audio_features": "Run Audio features.",
        "missing_derived_tags": "Run Derived tags + context fit.",
        "missing_context_fit": "Run Derived tags + context fit.",
        "profiles_not_v2": "Run Rebuild profiles.",
        "missing_embedding_text": "Run Rebuild profiles.",
        "missing_embeddings": "Run Profile embeddings.",
        "stale_embeddings": "Run Profile embeddings.",
        "semantic_model_not_indexed": "Run Profile embeddings for the selected semantic model.",
    }
    output: list[str] = []
    for warning in warnings:
        action = actions_by_code.get(warning["code"])
        if action and action not in output:
            output.append(action)
    return output


def _warning(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _select_model(requested_model: str, available_models: list[str]) -> str | None:
    if requested_model in available_models:
        return requested_model
    requested_leaf = Path(requested_model).name
    for model in available_models:
        if Path(model).name == requested_leaf:
            return model
    return available_models[0] if len(available_models) == 1 else None


def _path_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _ratio(value: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(value / denominator, 6)
