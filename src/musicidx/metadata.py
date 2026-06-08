"""Metadata extraction and text-profile persistence."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musicidx.config import FFPROBE_PATH_ENV_VAR, resolve_executable
from musicidx.db import utc_now

TAG_ALIASES = {
    "title": ["title", "tracktitle", "name"],
    "artist": ["artist", "artists", "performer"],
    "album": ["album"],
    "album_artist": ["album_artist", "albumartist", "album artist", "albumartists"],
    "genre": ["genre", "style"],
    "date": ["date", "year", "originaldate", "originalyear", "release_date"],
    "track_number": ["track", "tracknumber", "track_number", "track no", "trackno"],
    "disc_number": ["disc", "discnumber", "disc_number", "disk", "disknumber"],
}


class MetadataExtractionError(RuntimeError):
    """Raised when metadata cannot be extracted from a file."""


@dataclass(slots=True)
class TrackMetadata:
    """Common metadata and technical audio fields extracted from ffprobe."""

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    album_artist: str | None = None
    genre: str | None = None
    date: str | None = None
    track_number: str | None = None
    disc_number: str | None = None
    duration_sec: float | None = None
    codec: str | None = None
    sample_rate: int | None = None
    bit_rate: int | None = None
    channels: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetadataSummary:
    """Summary counters for a metadata extraction run."""

    processed: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class TextSearchResult:
    """One full-text search result."""

    track_id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    profile_text: str | None
    score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_ffprobe_available() -> bool:
    """Return True when ffprobe is available on PATH or configured explicitly."""
    return resolve_executable("ffprobe", FFPROBE_PATH_ENV_VAR) is not None


def extract_metadata(path: Path) -> TrackMetadata:
    """Extract metadata from an audio file using ffprobe JSON output."""
    data = run_ffprobe(path)
    return metadata_from_ffprobe_json(data)


def run_ffprobe(path: Path) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output."""
    ffprobe = resolve_executable("ffprobe", FFPROBE_PATH_ENV_VAR)
    if ffprobe is None:
        raise MetadataExtractionError(
            "ffprobe not found; install with `brew install ffmpeg` "
            f"or set {FFPROBE_PATH_ENV_VAR}"
        )

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise MetadataExtractionError(
            "ffprobe not found; install with `brew install ffmpeg` "
            f"or set {FFPROBE_PATH_ENV_VAR}"
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or "ffprobe returned a non-zero exit code"
        raise MetadataExtractionError(detail)

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MetadataExtractionError("ffprobe returned invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise MetadataExtractionError("ffprobe JSON root was not an object")
    return parsed


def metadata_from_ffprobe_json(data: dict[str, Any]) -> TrackMetadata:
    """Convert ffprobe JSON into normalized TrackMetadata."""
    format_info = _object_or_empty(data.get("format"))
    audio_stream = _first_audio_stream(data.get("streams"))
    tags = _collect_tags(format_info, audio_stream)

    return TrackMetadata(
        title=_first_tag(tags, "title"),
        artist=_first_tag(tags, "artist"),
        album=_first_tag(tags, "album"),
        album_artist=_first_tag(tags, "album_artist"),
        genre=_first_tag(tags, "genre"),
        date=_first_tag(tags, "date"),
        track_number=_first_tag(tags, "track_number"),
        disc_number=_first_tag(tags, "disc_number"),
        duration_sec=_parse_float(format_info.get("duration") or audio_stream.get("duration")),
        codec=_clean_value(audio_stream.get("codec_name")),
        sample_rate=_parse_int(audio_stream.get("sample_rate")),
        bit_rate=_parse_int(format_info.get("bit_rate") or audio_stream.get("bit_rate")),
        channels=_parse_int(audio_stream.get("channels")),
    )


def process_metadata(
    conn: sqlite3.Connection,
    *,
    track_id: str | None = None,
    missing_only: bool = False,
) -> MetadataSummary:
    """Extract and persist metadata for selected tracks."""
    summary = MetadataSummary()
    rows = _select_tracks_for_metadata(conn, track_id=track_id, missing_only=missing_only)

    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            summary.skipped += 1
            _record_track_error(conn, row["id"], "file is missing on disk")
            continue

        summary.processed += 1
        try:
            metadata = extract_metadata(path)
            profile_text, profile_json = build_track_profile(metadata, path)
            save_track_metadata(conn, row["id"], metadata, profile_text, profile_json)
            summary.updated += 1
        except MetadataExtractionError as exc:
            summary.errors += 1
            _record_track_error(conn, row["id"], str(exc))

    conn.commit()
    return summary


def build_track_profile(metadata: TrackMetadata, path: Path) -> tuple[str, str]:
    """Build deterministic profile text and JSON for a track."""
    parts: list[str] = []
    if metadata.artist:
        parts.append(f"Artist: {metadata.artist}.")
    if metadata.title:
        parts.append(f"Title: {metadata.title}.")
    else:
        parts.append(f"Filename: {path.stem}.")
    if metadata.album:
        parts.append(f"Album: {metadata.album}.")
    if metadata.album_artist and metadata.album_artist != metadata.artist:
        parts.append(f"Album artist: {metadata.album_artist}.")
    if metadata.genre:
        parts.append(f"Genre: {metadata.genre}.")
    if metadata.date:
        parts.append(f"Date: {metadata.date}.")
    if metadata.duration_sec is not None:
        parts.append(f"Duration: {_format_duration(metadata.duration_sec)}.")

    technical = _technical_profile(metadata)
    if technical:
        parts.append(f"Technical: {technical}.")

    profile = {
        "path": str(path),
        "metadata": metadata.as_dict(),
    }
    return " ".join(parts), json.dumps(profile, sort_keys=True)


def save_track_metadata(
    conn: sqlite3.Connection,
    track_id: str,
    metadata: TrackMetadata,
    profile_text: str,
    profile_json: str,
) -> None:
    """Persist metadata, profile text, and FTS rows for one track."""
    now = utc_now()
    conn.execute(
        """
        UPDATE tracks
        SET title = ?, artist = ?, album = ?, album_artist = ?, genre = ?, date = ?,
            track_number = ?, disc_number = ?, duration_sec = ?, codec = ?, sample_rate = ?,
            bit_rate = ?, channels = ?, indexed_at = ?, last_error = NULL
        WHERE id = ?
        """,
        (
            metadata.title,
            metadata.artist,
            metadata.album,
            metadata.album_artist,
            metadata.genre,
            metadata.date,
            metadata.track_number,
            metadata.disc_number,
            metadata.duration_sec,
            metadata.codec,
            metadata.sample_rate,
            metadata.bit_rate,
            metadata.channels,
            now,
            track_id,
        ),
    )
    conn.execute(
        """
        INSERT INTO track_profiles (track_id, profile_text, profile_json, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            profile_text = excluded.profile_text,
            profile_json = excluded.profile_json,
            updated_at = excluded.updated_at
        """,
        (track_id, profile_text, profile_json, now),
    )
    conn.execute("DELETE FROM tracks_fts WHERE track_id = ?", (track_id,))
    conn.execute(
        """
        INSERT INTO tracks_fts (track_id, title, artist, album, genre, profile_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (track_id, metadata.title, metadata.artist, metadata.album, metadata.genre, profile_text),
    )


def search_text(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
    include_missing: bool = False,
) -> list[TextSearchResult]:
    """Search indexed track text through SQLite FTS5."""
    fts_query = _normalize_fts_query(query)
    if not fts_query:
        return []

    missing_clause = "" if include_missing else "AND t.missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT
            t.id AS track_id,
            t.path AS path,
            t.title AS title,
            t.artist AS artist,
            t.album AS album,
            t.genre AS genre,
            p.profile_text AS profile_text,
            bm25(tracks_fts) AS score
        FROM tracks_fts
        JOIN tracks t ON t.id = tracks_fts.track_id
        LEFT JOIN track_profiles p ON p.track_id = t.id
        WHERE tracks_fts MATCH ?
          {missing_clause}
        ORDER BY score
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()

    return [
        TextSearchResult(
            track_id=row["track_id"],
            path=row["path"],
            title=row["title"],
            artist=row["artist"],
            album=row["album"],
            genre=row["genre"],
            profile_text=row["profile_text"],
            score=float(row["score"]),
        )
        for row in rows
    ]


def _select_tracks_for_metadata(
    conn: sqlite3.Connection,
    *,
    track_id: str | None,
    missing_only: bool,
) -> list[sqlite3.Row]:
    clauses = ["missing_at IS NULL"]
    params: list[Any] = []
    if track_id is not None:
        clauses.append("id = ?")
        params.append(track_id)
    if missing_only:
        clauses.append(
            """
            (
                title IS NULL
                OR duration_sec IS NULL
                OR id NOT IN (SELECT track_id FROM track_profiles)
            )
            """
        )

    return conn.execute(
        f"SELECT id, path FROM tracks WHERE {' AND '.join(clauses)} ORDER BY path",
        params,
    ).fetchall()


def _record_track_error(conn: sqlite3.Connection, track_id: str, error: str) -> None:
    conn.execute("UPDATE tracks SET last_error = ? WHERE id = ?", (error, track_id))


def _collect_tags(format_info: dict[str, Any], audio_stream: dict[str, Any]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for source in (
        _object_or_empty(format_info.get("tags")),
        _object_or_empty(audio_stream.get("tags")),
    ):
        for key, value in source.items():
            normalized_key = _normalize_tag_key(str(key))
            if normalized_key not in tags:
                cleaned = _clean_value(value)
                if cleaned is not None:
                    tags[normalized_key] = cleaned
    return tags


def _first_tag(tags: dict[str, str], canonical_name: str) -> str | None:
    for alias in TAG_ALIASES[canonical_name]:
        value = tags.get(_normalize_tag_key(alias))
        if value:
            return value
    return None


def _first_audio_stream(streams: Any) -> dict[str, Any]:
    if not isinstance(streams, list):
        return {}
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    return {}


def _object_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_tag_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _clean_value(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _parse_int(value: Any) -> int | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}"


def _technical_profile(metadata: TrackMetadata) -> str | None:
    parts: list[str] = []
    if metadata.codec:
        parts.append(f"codec {metadata.codec}")
    if metadata.sample_rate:
        parts.append(f"sample rate {metadata.sample_rate} Hz")
    if metadata.bit_rate:
        parts.append(f"bit rate {metadata.bit_rate} bps")
    if metadata.channels:
        parts.append(f"channels {metadata.channels}")
    return ", ".join(parts) if parts else None


def _normalize_fts_query(query: str) -> str:
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    return " ".join(terms)
