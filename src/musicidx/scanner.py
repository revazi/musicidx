"""Directory scanner for local music libraries."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from musicidx.db import utc_now

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".wav",
    ".aiff",
    ".aif",
    ".ogg",
    ".opus",
    ".alac",
    ".wv",
}


@dataclass(slots=True)
class ScanSummary:
    """Summary counters produced by a scanner run."""

    root_path: str
    added: int = 0
    unchanged: int = 0
    modified: int = 0
    missing: int = 0
    skipped: int = 0
    errors: int = 0
    dry_run: bool = False
    root_missing: bool = False

    @property
    def total_seen(self) -> int:
        return self.added + self.unchanged + self.modified

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["total_seen"] = self.total_seen
        return payload


def scan_library(
    root_path: Path,
    conn: sqlite3.Connection,
    *,
    full_hash: bool = False,
    follow_symlinks: bool = False,
    dry_run: bool = False,
) -> ScanSummary:
    """Scan a music directory and upsert supported audio files.

    Scanning is path-idempotent. Re-scanning unchanged files does not create
    duplicate tracks. Files that disappeared from the scanned root are marked
    with ``missing_at`` instead of being deleted.
    """
    root = root_path.expanduser().resolve()
    summary = ScanSummary(root_path=str(root), dry_run=dry_run)
    now = utc_now()
    root_id = _get_library_root_id(conn, root)

    if not root.exists():
        if root_id is None:
            raise FileNotFoundError(f"Directory does not exist: {root}")
        summary.root_missing = True
        try:
            _mark_missing_tracks(conn, root_id, set(), now, summary, dry_run=dry_run)
            if not dry_run:
                conn.execute("UPDATE library_roots SET updated_at = ? WHERE id = ?", (now, root_id))
                conn.commit()
        except Exception:
            if not dry_run:
                conn.rollback()
            raise
        return summary

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    if root_id is None and not dry_run:
        root_id = _insert_library_root(conn, root, now)
    elif root_id is not None and not dry_run:
        conn.execute("UPDATE library_roots SET updated_at = ? WHERE id = ?", (now, root_id))

    seen_paths: set[str] = set()

    try:
        for file_path in _iter_audio_files(root, follow_symlinks=follow_symlinks):
            try:
                path = file_path.resolve()
                stat = path.stat()
            except OSError:
                summary.errors += 1
                continue

            path_str = str(path)
            seen_paths.add(path_str)
            extension = path.suffix.lower()
            path_hash = _hash_text(path_str)
            content_hash = _hash_file(path) if full_hash else None

            existing = conn.execute("SELECT * FROM tracks WHERE path = ?", (path_str,)).fetchone()
            if existing is None:
                summary.added += 1
                if not dry_run:
                    conn.execute(
                        """
                        INSERT INTO tracks (
                            id, root_id, path, path_hash, extension, file_size,
                            file_mtime_ns, content_hash, indexed_at, missing_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        (
                            str(uuid.uuid4()),
                            root_id,
                            path_str,
                            path_hash,
                            extension,
                            stat.st_size,
                            stat.st_mtime_ns,
                            content_hash,
                            now,
                        ),
                    )
                continue

            stored_content_hash = existing["content_hash"]
            new_content_hash = content_hash if full_hash else stored_content_hash
            file_metadata_changed = (
                existing["extension"] != extension
                or existing["file_size"] != stat.st_size
                or existing["file_mtime_ns"] != stat.st_mtime_ns
            )
            metadata_changed = (
                existing["root_id"] != root_id
                or existing["path_hash"] != path_hash
                or file_metadata_changed
            )
            hash_changed = (
                full_hash
                and stored_content_hash is not None
                and content_hash is not None
                and stored_content_hash != content_hash
            )
            hash_backfilled = full_hash and stored_content_hash is None and content_hash is not None
            was_missing = existing["missing_at"] is not None
            stale_analysis = file_metadata_changed or hash_changed
            row_changed = metadata_changed or hash_changed or hash_backfilled or was_missing

            if metadata_changed or hash_changed or was_missing:
                summary.modified += 1
            else:
                summary.unchanged += 1

            if row_changed and not dry_run:
                conn.execute(
                    """
                    UPDATE tracks
                    SET root_id = ?, path_hash = ?, extension = ?, file_size = ?,
                        file_mtime_ns = ?, content_hash = ?, indexed_at = ?, missing_at = NULL,
                        last_error = NULL, last_error_at = NULL, error_count = 0,
                        quarantined_at = NULL, quarantine_reason = NULL
                    WHERE id = ?
                    """,
                    (
                        root_id,
                        path_hash,
                        extension,
                        stat.st_size,
                        stat.st_mtime_ns,
                        new_content_hash,
                        now,
                        existing["id"],
                    ),
                )
                if stale_analysis:
                    _clear_stale_track_outputs(conn, existing["id"])

        if root_id is not None:
            _mark_missing_tracks(conn, root_id, seen_paths, now, summary, dry_run=dry_run)

        if not dry_run:
            conn.commit()
    except Exception:
        if not dry_run:
            conn.rollback()
        raise

    return summary


def _iter_audio_files(root: Path, *, follow_symlinks: bool) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root, followlinks=follow_symlinks):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def _get_library_root_id(conn: sqlite3.Connection, root: Path) -> int | None:
    row = conn.execute("SELECT id FROM library_roots WHERE path = ?", (str(root),)).fetchone()
    return int(row["id"]) if row else None


def _insert_library_root(conn: sqlite3.Connection, root: Path, now: str) -> int:
    cursor = conn.execute(
        "INSERT INTO library_roots (path, created_at, updated_at) VALUES (?, ?, ?)",
        (str(root), now, now),
    )
    return int(cursor.lastrowid)


def _clear_stale_track_outputs(conn: sqlite3.Connection, track_id: str) -> None:
    """Invalidate derived data after an existing file changes on disk."""
    conn.execute(
        """
        UPDATE tracks
        SET chromaprint = NULL,
            fingerprint_duration = NULL,
            title = NULL,
            artist = NULL,
            album = NULL,
            album_artist = NULL,
            genre = NULL,
            date = NULL,
            track_number = NULL,
            disc_number = NULL,
            duration_sec = NULL,
            codec = NULL,
            sample_rate = NULL,
            bit_rate = NULL,
            channels = NULL,
            analysis_version = 0,
            analyzed_at = NULL
        WHERE id = ?
        """,
        (track_id,),
    )
    conn.execute("DELETE FROM audio_features WHERE track_id = ?", (track_id,))
    conn.execute("DELETE FROM track_tags WHERE track_id = ?", (track_id,))
    conn.execute("DELETE FROM track_context_fit WHERE track_id = ?", (track_id,))
    conn.execute("DELETE FROM track_profiles WHERE track_id = ?", (track_id,))
    conn.execute("DELETE FROM embeddings WHERE track_id = ?", (track_id,))
    conn.execute("DELETE FROM tracks_fts WHERE track_id = ?", (track_id,))


def _mark_missing_tracks(
    conn: sqlite3.Connection,
    root_id: int,
    seen_paths: set[str],
    now: str,
    summary: ScanSummary,
    *,
    dry_run: bool,
) -> None:
    rows = conn.execute(
        "SELECT id, path FROM tracks WHERE root_id = ? AND missing_at IS NULL",
        (root_id,),
    ).fetchall()
    for row in rows:
        if row["path"] in seen_paths:
            continue
        summary.missing += 1
        if not dry_run:
            conn.execute("UPDATE tracks SET missing_at = ? WHERE id = ?", (now, row["id"]))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
