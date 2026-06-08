"""SQLite connection and migration helpers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from musicidx.config import resolve_db_path
from musicidx.migrations import MIGRATIONS

CORE_TABLES = [
    "library_roots",
    "tracks",
    "audio_features",
    "track_tags",
    "track_profiles",
    "tracks_fts",
    "embeddings",
    "search_events",
    "feedback",
]


def utc_now() -> str:
    """Return a stable UTC timestamp string for database rows."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect_db(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection and enable local-index pragmas."""
    db_path = resolve_db_path(path)
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    apply_migrations(conn)


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending migrations exactly once."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied_versions = {
        int(row["version"])
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }

    for version, name, sql in MIGRATIONS:
        if version in applied_versions:
            continue
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (version, name, utc_now()),
        )
    conn.commit()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True if a table or virtual table exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_counts(conn: sqlite3.Connection) -> dict[str, int | None]:
    """Return row counts for known tables, using None for missing tables."""
    counts: dict[str, int | None] = {}
    for table in CORE_TABLES:
        if not table_exists(conn, table):
            counts[table] = None
            continue
        counts[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    return counts


def db_info(conn: sqlite3.Connection, path: Path | str) -> dict[str, Any]:
    """Build a small, JSON-serializable database information payload."""
    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    foreign_keys = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    migration_rows = conn.execute(
        "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall() if table_exists(conn, "schema_migrations") else []

    return {
        "db_path": str(resolve_db_path(path)),
        "sqlite_version": sqlite3.sqlite_version,
        "journal_mode": journal_mode,
        "foreign_keys": foreign_keys,
        "tables": table_counts(conn),
        "migrations": [dict(row) for row in migration_rows],
    }
