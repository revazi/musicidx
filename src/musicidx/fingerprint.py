"""Audio fingerprinting and duplicate candidate detection."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musicidx.config import FPCALC_PATH_ENV_VAR, resolve_executable
from musicidx.db import utc_now
from musicidx.failures import clear_track_failure, record_track_error


class FingerprintError(RuntimeError):
    """Raised when a track cannot be fingerprinted."""


@dataclass(slots=True)
class TrackFingerprint:
    """Chromaprint fingerprint output from fpcalc."""

    chromaprint: str
    duration_sec: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FingerprintSummary:
    """Summary counters for a fingerprint run."""

    processed: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class DuplicateTrack:
    """Track information included in a duplicate group."""

    track_id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    duration_sec: float | None
    fingerprint_duration: float | None
    missing: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DuplicateGroup:
    """A group of duplicate or possible moved-file candidates."""

    kind: str
    reason: str
    key: str
    tracks: list[DuplicateTrack]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "key": self.key,
            "tracks": [track.as_dict() for track in self.tracks],
        }


def is_fpcalc_available() -> bool:
    """Return True when fpcalc is available on PATH or configured explicitly."""
    return resolve_executable("fpcalc", FPCALC_PATH_ENV_VAR) is not None


def fingerprint_path(path: Path) -> TrackFingerprint:
    """Fingerprint one audio file with fpcalc JSON output."""
    fpcalc = resolve_executable("fpcalc", FPCALC_PATH_ENV_VAR)
    if fpcalc is None:
        raise FingerprintError(
            "fpcalc not found; install with `brew install chromaprint` "
            f"or set {FPCALC_PATH_ENV_VAR}"
        )
    command = [fpcalc, "-json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FingerprintError(
            "fpcalc not found; install with `brew install chromaprint` "
            f"or set {FPCALC_PATH_ENV_VAR}"
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or "fpcalc returned a non-zero exit code"
        raise FingerprintError(detail)

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FingerprintError("fpcalc returned invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise FingerprintError("fpcalc JSON root was not an object")

    fingerprint = parsed.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise FingerprintError("fpcalc did not return a fingerprint")

    return TrackFingerprint(
        chromaprint=fingerprint.strip(),
        duration_sec=_parse_float(parsed.get("duration")),
    )


def process_fingerprints(
    conn: sqlite3.Connection,
    *,
    track_id: str | None = None,
    missing_only: bool = False,
) -> FingerprintSummary:
    """Fingerprint selected non-missing tracks and persist results."""
    summary = FingerprintSummary()
    rows = _select_tracks_for_fingerprinting(conn, track_id=track_id, missing_only=missing_only)

    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            summary.skipped += 1
            record_track_error(conn, row["id"], "file is missing on disk")
            continue

        summary.processed += 1
        try:
            fingerprint = fingerprint_path(path)
            save_track_fingerprint(conn, row["id"], fingerprint)
            summary.updated += 1
        except FingerprintError as exc:
            summary.errors += 1
            record_track_error(conn, row["id"], str(exc))

    conn.commit()
    return summary


def save_track_fingerprint(
    conn: sqlite3.Connection,
    track_id: str,
    fingerprint: TrackFingerprint,
) -> None:
    """Persist fingerprint information for one track."""
    conn.execute(
        """
        UPDATE tracks
        SET chromaprint = ?, fingerprint_duration = ?, indexed_at = ?
        WHERE id = ?
        """,
        (fingerprint.chromaprint, fingerprint.duration_sec, utc_now(), track_id),
    )
    clear_track_failure(conn, track_id)


def find_duplicate_groups(
    conn: sqlite3.Connection,
    *,
    include_missing: bool = True,
    duration_tolerance_sec: float = 3.0,
) -> list[DuplicateGroup]:
    """Find duplicate and possible moved-file candidate groups.

    This is intentionally conservative and read-only. It never deletes rows or
    rewrites paths.
    """
    groups: list[DuplicateGroup] = []
    seen_signatures: set[tuple[str, tuple[str, ...]]] = set()

    for group in _same_content_hash_groups(conn, include_missing=include_missing):
        _append_group(groups, seen_signatures, group)

    for group in _same_chromaprint_groups(
        conn,
        include_missing=include_missing,
        duration_tolerance_sec=duration_tolerance_sec,
    ):
        _append_group(groups, seen_signatures, group)

    for group in _same_metadata_groups(
        conn,
        include_missing=include_missing,
        duration_tolerance_sec=duration_tolerance_sec,
    ):
        _append_group(groups, seen_signatures, group)

    return groups


def _select_tracks_for_fingerprinting(
    conn: sqlite3.Connection,
    *,
    track_id: str | None,
    missing_only: bool,
) -> list[sqlite3.Row]:
    clauses = ["missing_at IS NULL", "quarantined_at IS NULL"]
    params: list[Any] = []
    if track_id is not None:
        clauses.append("id = ?")
        params.append(track_id)
    if missing_only:
        clauses.append("chromaprint IS NULL")

    return conn.execute(
        f"SELECT id, path FROM tracks WHERE {' AND '.join(clauses)} ORDER BY path",
        params,
    ).fetchall()


def _same_content_hash_groups(
    conn: sqlite3.Connection,
    *,
    include_missing: bool,
) -> list[DuplicateGroup]:
    missing_clause = "" if include_missing else "AND missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT content_hash
        FROM tracks
        WHERE content_hash IS NOT NULL
          AND content_hash != ''
          {missing_clause}
        GROUP BY content_hash
        HAVING COUNT(*) > 1
        ORDER BY content_hash
        """
    ).fetchall()

    groups: list[DuplicateGroup] = []
    for row in rows:
        tracks = _tracks_for_clause(
            conn,
            "content_hash = ?",
            [row["content_hash"]],
            include_missing,
        )
        groups.append(
            _make_group(
                kind=_kind_for_tracks(tracks, default="exact_duplicate"),
                reason="same content hash",
                key=row["content_hash"],
                tracks=tracks,
            )
        )
    return groups


def _same_chromaprint_groups(
    conn: sqlite3.Connection,
    *,
    include_missing: bool,
    duration_tolerance_sec: float,
) -> list[DuplicateGroup]:
    missing_clause = "" if include_missing else "AND missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT chromaprint
        FROM tracks
        WHERE chromaprint IS NOT NULL
          AND chromaprint != ''
          {missing_clause}
        GROUP BY chromaprint
        HAVING COUNT(*) > 1
        ORDER BY chromaprint
        """
    ).fetchall()

    groups: list[DuplicateGroup] = []
    for row in rows:
        tracks = _tracks_for_clause(conn, "chromaprint = ?", [row["chromaprint"]], include_missing)
        for cluster in _cluster_by_duration(tracks, duration_tolerance_sec):
            if len(cluster) < 2:
                continue
            groups.append(
                _make_group(
                    kind=_kind_for_tracks(cluster, default="audio_duplicate"),
                    reason="same chromaprint and similar duration",
                    key=row["chromaprint"],
                    tracks=cluster,
                )
            )
    return groups


def _same_metadata_groups(
    conn: sqlite3.Connection,
    *,
    include_missing: bool,
    duration_tolerance_sec: float,
) -> list[DuplicateGroup]:
    missing_clause = "" if include_missing else "AND missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT lower(trim(artist)) AS artist_key, lower(trim(title)) AS title_key
        FROM tracks
        WHERE artist IS NOT NULL
          AND title IS NOT NULL
          AND duration_sec IS NOT NULL
          AND trim(artist) != ''
          AND trim(title) != ''
          {missing_clause}
        GROUP BY artist_key, title_key
        HAVING COUNT(*) > 1
        ORDER BY artist_key, title_key
        """
    ).fetchall()

    groups: list[DuplicateGroup] = []
    for row in rows:
        tracks = _tracks_for_clause(
            conn,
            "lower(trim(artist)) = ? AND lower(trim(title)) = ?",
            [row["artist_key"], row["title_key"]],
            include_missing,
        )
        for cluster in _cluster_by_duration(tracks, duration_tolerance_sec):
            if len(cluster) < 2:
                continue
            groups.append(
                _make_group(
                    kind=_kind_for_tracks(cluster, default="possible_duplicate"),
                    reason="same artist/title and similar duration",
                    key=f"{row['artist_key']}::{row['title_key']}",
                    tracks=cluster,
                )
            )
    return groups


def _tracks_for_clause(
    conn: sqlite3.Connection,
    clause: str,
    params: list[Any],
    include_missing: bool,
) -> list[DuplicateTrack]:
    missing_clause = "" if include_missing else "AND missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT
            id, path, title, artist, album, duration_sec, fingerprint_duration, missing_at
        FROM tracks
        WHERE {clause}
          {missing_clause}
        ORDER BY coalesce(artist, ''), coalesce(album, ''), coalesce(title, ''), path
        """,
        params,
    ).fetchall()
    return [_duplicate_track_from_row(row) for row in rows]


def _duplicate_track_from_row(row: sqlite3.Row) -> DuplicateTrack:
    return DuplicateTrack(
        track_id=row["id"],
        path=row["path"],
        title=row["title"],
        artist=row["artist"],
        album=row["album"],
        duration_sec=row["duration_sec"],
        fingerprint_duration=row["fingerprint_duration"],
        missing=row["missing_at"] is not None,
    )


def _cluster_by_duration(
    tracks: list[DuplicateTrack],
    tolerance_sec: float,
) -> list[list[DuplicateTrack]]:
    if len(tracks) < 2:
        return [tracks]

    sorted_tracks = sorted(tracks, key=lambda track: _duration_for_track(track) or -1.0)
    unknown_duration = [track for track in sorted_tracks if _duration_for_track(track) is None]
    known_duration = [track for track in sorted_tracks if _duration_for_track(track) is not None]

    clusters: list[list[DuplicateTrack]] = []
    current: list[DuplicateTrack] = []
    cluster_start: float | None = None

    for track in known_duration:
        duration = _duration_for_track(track)
        if duration is None:
            continue
        if cluster_start is None or duration - cluster_start <= tolerance_sec:
            current.append(track)
            if cluster_start is None:
                cluster_start = duration
        else:
            clusters.append(current)
            current = [track]
            cluster_start = duration

    if current:
        clusters.append(current)

    if unknown_duration:
        clusters.append(unknown_duration)

    return clusters


def _duration_for_track(track: DuplicateTrack) -> float | None:
    if track.fingerprint_duration is not None:
        return track.fingerprint_duration
    return track.duration_sec


def _make_group(kind: str, reason: str, key: str, tracks: list[DuplicateTrack]) -> DuplicateGroup:
    return DuplicateGroup(kind=kind, reason=reason, key=key, tracks=tracks)


def _kind_for_tracks(tracks: list[DuplicateTrack], *, default: str) -> str:
    has_missing = any(track.missing for track in tracks)
    has_present = any(not track.missing for track in tracks)
    if has_missing and has_present:
        return "possible_move"
    return default


def _append_group(
    groups: list[DuplicateGroup],
    seen_signatures: set[tuple[str, tuple[str, ...]]],
    group: DuplicateGroup,
) -> None:
    signature = (group.kind, tuple(sorted(track.track_id for track in group.tracks)))
    if signature in seen_signatures:
        return
    seen_signatures.add(signature)
    groups.append(group)


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
