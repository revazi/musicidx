"""Missing-track listing and pruning helpers."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MissingTrack:
    """One track marked missing from disk."""

    id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    root_path: str | None
    missing_at: str
    last_error: str | None
    quarantined_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_missing_tracks(
    conn: sqlite3.Connection,
    *,
    root_path: str | None = None,
) -> list[MissingTrack]:
    """List tracks marked missing from disk."""
    clauses = ["t.missing_at IS NOT NULL"]
    params: list[Any] = []
    if root_path is not None:
        clauses.append("lr.path = ?")
        params.append(root_path)

    rows = conn.execute(
        f"""
        SELECT t.id, t.path, t.title, t.artist, t.album, t.missing_at,
               t.last_error, t.quarantined_at, lr.path AS root_path
        FROM tracks t
        LEFT JOIN library_roots lr ON lr.id = t.root_id
        WHERE {' AND '.join(clauses)}
        ORDER BY t.missing_at DESC, t.path
        """,
        params,
    ).fetchall()
    return [
        MissingTrack(
            id=str(row["id"]),
            path=str(row["path"]),
            title=row["title"],
            artist=row["artist"],
            album=row["album"],
            root_path=row["root_path"],
            missing_at=str(row["missing_at"]),
            last_error=row["last_error"],
            quarantined_at=row["quarantined_at"],
        )
        for row in rows
    ]


def prune_missing_tracks(conn: sqlite3.Connection, *, track_id: str | None = None) -> int:
    """Delete missing-track database rows only; never delete files from disk."""
    if track_id is not None:
        rows = conn.execute(
            "SELECT id FROM tracks WHERE id = ? AND missing_at IS NOT NULL",
            (track_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT id FROM tracks WHERE missing_at IS NOT NULL").fetchall()

    track_ids = [str(row["id"]) for row in rows]
    if not track_ids:
        conn.commit()
        return 0

    placeholders = ", ".join("?" for _ in track_ids)
    conn.execute(f"DELETE FROM tracks_fts WHERE track_id IN ({placeholders})", track_ids)
    conn.execute(f"DELETE FROM tracks WHERE id IN ({placeholders})", track_ids)
    conn.commit()
    return len(track_ids)
