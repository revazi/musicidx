"""Failure tracking and quarantine helpers for repeated bad tracks."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from musicidx.db import utc_now

DEFAULT_QUARANTINE_THRESHOLD = 3


@dataclass(slots=True)
class FailedTrack:
    """One failed or quarantined track row."""

    id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    missing_at: str | None
    last_error: str | None
    error_count: int
    last_error_at: str | None
    quarantined_at: str | None
    quarantine_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_track_error(
    conn: sqlite3.Connection,
    track_id: str,
    error: str,
    *,
    threshold: int = DEFAULT_QUARANTINE_THRESHOLD,
) -> None:
    """Record a track error and quarantine after repeated failures."""
    now = utc_now()
    row = conn.execute(
        "SELECT error_count, quarantined_at FROM tracks WHERE id = ?",
        (track_id,),
    ).fetchone()
    if row is None:
        return

    next_count = int(row["error_count"] or 0) + 1
    should_quarantine = threshold > 0 and next_count >= threshold
    quarantined_at = row["quarantined_at"] or (now if should_quarantine else None)
    quarantine_reason = error if should_quarantine else None

    conn.execute(
        """
        UPDATE tracks
        SET last_error = ?, last_error_at = ?, error_count = ?,
            quarantined_at = ?, quarantine_reason = COALESCE(?, quarantine_reason)
        WHERE id = ?
        """,
        (error, now, next_count, quarantined_at, quarantine_reason, track_id),
    )


def clear_track_failure(conn: sqlite3.Connection, track_id: str) -> None:
    """Clear all failure/quarantine state for a track after success or retry."""
    conn.execute(
        """
        UPDATE tracks
        SET last_error = NULL,
            last_error_at = NULL,
            error_count = 0,
            quarantined_at = NULL,
            quarantine_reason = NULL
        WHERE id = ?
        """,
        (track_id,),
    )


def reset_failed_tracks(conn: sqlite3.Connection, *, track_id: str | None = None) -> int:
    """Clear failure/quarantine state for one track or all failed tracks."""
    if track_id is not None:
        before = conn.total_changes
        clear_track_failure(conn, track_id)
        conn.commit()
        return conn.total_changes - before

    before = conn.total_changes
    conn.execute(
        """
        UPDATE tracks
        SET last_error = NULL,
            last_error_at = NULL,
            error_count = 0,
            quarantined_at = NULL,
            quarantine_reason = NULL
        WHERE last_error IS NOT NULL
           OR error_count > 0
           OR quarantined_at IS NOT NULL
        """
    )
    conn.commit()
    return conn.total_changes - before


def list_failed_tracks(
    conn: sqlite3.Connection,
    *,
    include_missing: bool = False,
    quarantined_only: bool = False,
) -> list[FailedTrack]:
    """List tracks with failures or quarantine state."""
    clauses = [
        "(last_error IS NOT NULL OR error_count > 0 OR quarantined_at IS NOT NULL)",
    ]
    if not include_missing:
        clauses.append("missing_at IS NULL")
    if quarantined_only:
        clauses.append("quarantined_at IS NOT NULL")

    rows = conn.execute(
        f"""
        SELECT id, path, title, artist, album, missing_at, last_error, error_count,
               last_error_at, quarantined_at, quarantine_reason
        FROM tracks
        WHERE {' AND '.join(clauses)}
        ORDER BY quarantined_at IS NULL, error_count DESC, path
        """
    ).fetchall()
    return [
        FailedTrack(
            id=str(row["id"]),
            path=str(row["path"]),
            title=row["title"],
            artist=row["artist"],
            album=row["album"],
            missing_at=row["missing_at"],
            last_error=row["last_error"],
            error_count=int(row["error_count"] or 0),
            last_error_at=row["last_error_at"],
            quarantined_at=row["quarantined_at"],
            quarantine_reason=row["quarantine_reason"],
        )
        for row in rows
    ]
