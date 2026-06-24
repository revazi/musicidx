"""Metadata extraction and text-profile persistence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musicidx.config import FFPROBE_PATH_ENV_VAR, resolve_executable
from musicidx.db import utc_now
from musicidx.failures import clear_track_failure, record_track_error
from musicidx.profile_documents import (
    build_profile_document,
    embedding_text_from_profile_json,
    normalize_text,
    profile_json_text,
    profile_source_fingerprint,
)

TEXT_METADATA_FIELDS = (
    "title",
    "artist",
    "album",
    "album_artist",
    "genre",
    "date",
    "track_number",
    "disc_number",
)

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
class MetadataClaim:
    """One provenance/confidence claim for a display metadata field."""

    field_name: str
    value_text: str
    source: str
    confidence: float
    source_detail: str | None = None
    selected: bool = False
    license: str | None = None
    raw_json: dict[str, Any] | None = None

    @property
    def value_norm(self) -> str | None:
        return normalize_metadata_value(self.value_text)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetadataExtractionResult:
    """Selected metadata plus raw provenance claims."""

    metadata: TrackMetadata
    claims: list[MetadataClaim]


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
    """Extract selected metadata from an audio file using ffprobe JSON output."""
    return extract_metadata_result(path).metadata


def extract_metadata_result(path: Path) -> MetadataExtractionResult:
    """Extract metadata and provenance claims from ffprobe and local fallbacks."""
    data = run_ffprobe(path)
    ffprobe_metadata = metadata_from_ffprobe_json(data)
    return build_metadata_extraction_result(ffprobe_metadata, path)


def build_metadata_extraction_result(
    ffprobe_metadata: TrackMetadata,
    path: Path,
) -> MetadataExtractionResult:
    """Build selected metadata and claims from raw ffprobe metadata plus filename hints."""
    metadata = apply_filename_fallback(ffprobe_metadata, path)
    claims = _metadata_claims(ffprobe_metadata, metadata, path)
    return MetadataExtractionResult(metadata=metadata, claims=claims)


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
            record_track_error(conn, row["id"], "file is missing on disk")
            continue

        summary.processed += 1
        try:
            extraction = extract_metadata_result(path)
            metadata = extraction.metadata
            profile_text, profile_json = build_track_profile(metadata, path)
            save_track_metadata(
                conn,
                row["id"],
                metadata,
                profile_text,
                profile_json,
                claims=extraction.claims,
            )
            summary.updated += 1
        except MetadataExtractionError as exc:
            summary.errors += 1
            record_track_error(conn, row["id"], str(exc))

    conn.commit()
    return summary


def apply_filename_fallback(metadata: TrackMetadata, path: Path) -> TrackMetadata:
    """Fill missing title/artist from a readable filename when tags are absent."""
    fallback = infer_metadata_from_filename(path)
    parsed_title = infer_metadata_from_text(metadata.title) if metadata.title else None
    return TrackMetadata(
        title=(
            parsed_title.title
            if parsed_title is not None and not metadata.artist
            else metadata.title or fallback.title
        ),
        artist=(
            metadata.artist
            or (parsed_title.artist if parsed_title else None)
            or fallback.artist
        ),
        album=metadata.album,
        album_artist=metadata.album_artist,
        genre=metadata.genre,
        date=metadata.date,
        track_number=metadata.track_number,
        disc_number=metadata.disc_number,
        duration_sec=metadata.duration_sec,
        codec=metadata.codec,
        sample_rate=metadata.sample_rate,
        bit_rate=metadata.bit_rate,
        channels=metadata.channels,
    )


def infer_metadata_from_filename(path: Path) -> TrackMetadata:
    """Infer minimal metadata from common `Artist - Title.ext` filenames."""
    return infer_metadata_from_text(path.stem)


def infer_metadata_from_text(value: str) -> TrackMetadata:
    decoded = urllib.parse.unquote(value).replace("_", " ")
    cleaned = re.sub(r"\s+", " ", decoded).strip(" ._-—–")
    cleaned = re.sub(r"^\d{1,3}\s*[-_.]\s*", "", cleaned).strip()
    if not cleaned:
        return TrackMetadata()

    parts = re.split(r"\s+[-–—]\s+", cleaned, maxsplit=1)
    if len(parts) == 2:
        artist = _clean_filename_part(parts[0])
        title = _clean_filename_part(parts[1])
        if artist and title:
            return TrackMetadata(title=title, artist=artist)
    return TrackMetadata(title=_clean_filename_part(cleaned))


def _clean_filename_part(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value).strip(" ._-—–")
    return cleaned or None


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

    profile_text = " ".join(parts)
    document = build_profile_document(
        metadata=metadata.as_dict(),
        path=path,
        profile_text=profile_text,
    )
    return profile_text, profile_json_text(document)


def save_track_metadata(
    conn: sqlite3.Connection,
    track_id: str,
    metadata: TrackMetadata,
    profile_text: str,
    profile_json: str,
    *,
    claims: list[MetadataClaim] | None = None,
) -> None:
    """Persist metadata, profile text, FTS rows, and metadata provenance claims."""
    now = utc_now()
    claims = claims if claims is not None else _selected_metadata_claims(metadata)
    track_row = conn.execute(
        "SELECT id, path, missing_at, external_match_confidence FROM tracks WHERE id = ?",
        (track_id,),
    ).fetchone()
    confidence = _metadata_confidence(metadata, claims)
    title_norm = normalize_metadata_value(metadata.title)
    artist_norm = normalize_metadata_value(metadata.artist)
    album_norm = normalize_metadata_value(metadata.album)
    artist_title_norm = normalize_metadata_value(
        " ".join(part for part in [metadata.artist, metadata.title] if part)
    )
    conn.execute(
        """
        UPDATE tracks
        SET title = ?, artist = ?, album = ?, album_artist = ?, genre = ?, date = ?,
            track_number = ?, disc_number = ?, duration_sec = ?, codec = ?, sample_rate = ?,
            bit_rate = ?, channels = ?, title_norm = ?, artist_norm = ?, album_norm = ?,
            artist_title_norm = ?, metadata_confidence = ?, indexed_at = ?
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
            title_norm,
            artist_norm,
            album_norm,
            artist_title_norm,
            confidence,
            now,
            track_id,
        ),
    )
    _save_metadata_claims(conn, track_id, claims, created_at=now)
    profile_json, embedding_text, source_fingerprint = _profile_storage_values(
        track_id=track_id,
        track_row=track_row,
        metadata=metadata,
        profile_text=profile_text,
        profile_json=profile_json,
        claims=claims,
        metadata_confidence=confidence,
        normalized={
            "title_norm": title_norm,
            "artist_norm": artist_norm,
            "album_norm": album_norm,
            "artist_title_norm": artist_title_norm,
        },
        generated_at=now,
    )
    conn.execute(
        """
        INSERT INTO track_profiles (
            track_id, profile_text, embedding_text, profile_json, profile_schema_version,
            source_fingerprint, updated_at
        ) VALUES (?, ?, ?, ?, 2, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            profile_text = excluded.profile_text,
            embedding_text = excluded.embedding_text,
            profile_json = excluded.profile_json,
            profile_schema_version = excluded.profile_schema_version,
            source_fingerprint = excluded.source_fingerprint,
            updated_at = excluded.updated_at
        """,
        (track_id, profile_text, embedding_text, profile_json, source_fingerprint, now),
    )
    clear_track_failure(conn, track_id)
    conn.execute("DELETE FROM tracks_fts WHERE track_id = ?", (track_id,))
    conn.execute(
        """
        INSERT INTO tracks_fts (track_id, title, artist, album, genre, profile_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (track_id, metadata.title, metadata.artist, metadata.album, metadata.genre, profile_text),
    )


def normalize_metadata_value(value: str | None) -> str | None:
    """Normalize metadata for stable matching/index columns."""
    return normalize_text(value)


def _profile_storage_values(
    *,
    track_id: str,
    track_row: sqlite3.Row | None,
    metadata: TrackMetadata,
    profile_text: str,
    profile_json: str,
    claims: list[MetadataClaim],
    metadata_confidence: float,
    normalized: dict[str, str | None],
    generated_at: str,
) -> tuple[str, str, str]:
    if track_row is None:
        embedding_text = embedding_text_from_profile_json(profile_json, fallback_text=profile_text)
        document = json.loads(profile_json)
    else:
        document = build_profile_document(
            track_id=track_id,
            metadata=metadata.as_dict(),
            path=Path(track_row["path"]),
            profile_text=profile_text,
            generated_at=generated_at,
            normalized=normalized,
            metadata_confidence=metadata_confidence,
            external_match_confidence=track_row["external_match_confidence"],
            field_confidence=_selected_claim_confidence(claims),
            provenance=_selected_claim_provenance(claims),
            missing=track_row["missing_at"] is not None,
        )
        profile_json = profile_json_text(document)
        embedding_text = document["search_text"]["embedding_text"]
    return profile_json, embedding_text, profile_source_fingerprint(document)


def _selected_claim_confidence(claims: list[MetadataClaim]) -> dict[str, float]:
    return {
        claim.field_name: round(max(0.0, min(1.0, claim.confidence)), 6)
        for claim in claims
        if claim.selected
    }


def _selected_claim_provenance(claims: list[MetadataClaim]) -> dict[str, str]:
    output: dict[str, str] = {}
    for claim in claims:
        if not claim.selected:
            continue
        detail = f":{claim.source_detail}" if claim.source_detail else ""
        output[claim.field_name] = f"{claim.source}{detail}"
    return output


def _metadata_claims(
    ffprobe_metadata: TrackMetadata,
    selected_metadata: TrackMetadata,
    path: Path,
) -> list[MetadataClaim]:
    claims: list[MetadataClaim] = []
    for field_name in TEXT_METADATA_FIELDS:
        value = getattr(ffprobe_metadata, field_name)
        if value:
            claims.append(
                MetadataClaim(
                    field_name=field_name,
                    value_text=value,
                    source="ffprobe",
                    source_detail="format/stream tags",
                    confidence=_ffprobe_claim_confidence(field_name),
                )
            )

    parsed_title = (
        infer_metadata_from_text(ffprobe_metadata.title) if ffprobe_metadata.title else None
    )
    if parsed_title and not ffprobe_metadata.artist:
        if parsed_title.artist:
            claims.append(
                MetadataClaim(
                    field_name="artist",
                    value_text=parsed_title.artist,
                    source="derived",
                    source_detail="artist-title pattern in title tag",
                    confidence=0.72,
                )
            )
        if parsed_title.title:
            claims.append(
                MetadataClaim(
                    field_name="title",
                    value_text=parsed_title.title,
                    source="derived",
                    source_detail="artist-title pattern in title tag",
                    confidence=0.72,
                )
            )

    filename_metadata = infer_metadata_from_filename(path)
    filename_confidence = 0.58 if filename_metadata.artist and filename_metadata.title else 0.45
    if filename_metadata.artist:
        claims.append(
            MetadataClaim(
                field_name="artist",
                value_text=filename_metadata.artist,
                source="filename_parser",
                source_detail="artist - title filename pattern",
                confidence=filename_confidence,
            )
        )
    if filename_metadata.title:
        claims.append(
            MetadataClaim(
                field_name="title",
                value_text=filename_metadata.title,
                source="filename_parser",
                source_detail="filename stem",
                confidence=filename_confidence,
            )
        )

    _mark_selected_claims(claims, selected_metadata)
    _ensure_selected_claims(claims, selected_metadata)
    return claims


def _ffprobe_claim_confidence(field_name: str) -> float:
    if field_name in {"title", "artist", "album_artist"}:
        return 0.86
    if field_name == "album":
        return 0.82
    if field_name == "genre":
        return 0.70
    return 0.75


def _mark_selected_claims(claims: list[MetadataClaim], metadata: TrackMetadata) -> None:
    selected_norms = {
        field_name: normalize_metadata_value(getattr(metadata, field_name))
        for field_name in TEXT_METADATA_FIELDS
    }
    selected_fields: set[str] = set()
    for claim in sorted(claims, key=lambda item: item.confidence, reverse=True):
        if claim.field_name in selected_fields:
            continue
        if claim.value_norm and claim.value_norm == selected_norms.get(claim.field_name):
            claim.selected = True
            selected_fields.add(claim.field_name)


def _ensure_selected_claims(claims: list[MetadataClaim], metadata: TrackMetadata) -> None:
    selected_fields = {claim.field_name for claim in claims if claim.selected}
    for field_name in TEXT_METADATA_FIELDS:
        value = getattr(metadata, field_name)
        if not value or field_name in selected_fields:
            continue
        claims.append(
            MetadataClaim(
                field_name=field_name,
                value_text=value,
                source="derived",
                source_detail="selected metadata fallback",
                confidence=0.40,
                selected=True,
            )
        )


def _selected_metadata_claims(metadata: TrackMetadata) -> list[MetadataClaim]:
    return [
        MetadataClaim(
            field_name=field_name,
            value_text=value,
            source="derived",
            source_detail="selected metadata supplied by caller",
            confidence=0.70,
            selected=True,
        )
        for field_name in TEXT_METADATA_FIELDS
        if (value := getattr(metadata, field_name))
    ]


def _metadata_confidence(metadata: TrackMetadata, claims: list[MetadataClaim]) -> float:
    confidences: list[float] = []
    for field_name in ("title", "artist", "album", "genre"):
        selected_value = normalize_metadata_value(getattr(metadata, field_name))
        if not selected_value:
            continue
        field_confidence = max(
            (
                claim.confidence
                for claim in claims
                if claim.field_name == field_name
                and claim.selected
                and claim.value_norm == selected_value
            ),
            default=0.0,
        )
        confidences.append(field_confidence)
    if not confidences:
        return 0.0
    return round(sum(confidences) / len(confidences), 6)


def _save_metadata_claims(
    conn: sqlite3.Connection,
    track_id: str,
    claims: list[MetadataClaim],
    *,
    created_at: str,
) -> None:
    if not claims:
        return
    placeholders = ", ".join("?" for _ in TEXT_METADATA_FIELDS)
    conn.execute(
        f"""
        UPDATE track_metadata_claims
        SET selected = 0
        WHERE track_id = ?
          AND field_name IN ({placeholders})
        """,
        (track_id, *TEXT_METADATA_FIELDS),
    )
    for claim in claims:
        claim_id = _metadata_claim_id(track_id, claim)
        raw_json = json.dumps(claim.raw_json, sort_keys=True) if claim.raw_json else None
        conn.execute(
            """
            INSERT INTO track_metadata_claims (
                id, track_id, field_name, value_text, value_norm, source, source_detail,
                confidence, selected, license, raw_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                confidence = excluded.confidence,
                selected = excluded.selected,
                license = excluded.license,
                raw_json = excluded.raw_json,
                created_at = excluded.created_at
            """,
            (
                claim_id,
                track_id,
                claim.field_name,
                claim.value_text,
                claim.value_norm,
                claim.source,
                claim.source_detail,
                max(0.0, min(1.0, claim.confidence)),
                1 if claim.selected else 0,
                claim.license,
                raw_json,
                created_at,
            ),
        )


def _metadata_claim_id(track_id: str, claim: MetadataClaim) -> str:
    fingerprint = "\0".join(
        [
            track_id,
            claim.field_name,
            claim.source,
            claim.source_detail or "",
            claim.value_norm or "",
        ]
    )
    return "claim_" + hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:32]


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
    clauses = ["missing_at IS NULL", "quarantined_at IS NULL"]
    params: list[Any] = []
    if track_id is not None:
        clauses.append("id = ?")
        params.append(track_id)
    if missing_only:
        clauses.append(
            """
            (
                duration_sec IS NULL
                OR codec IS NULL
                OR title IS NULL
                OR (artist IS NULL AND path LIKE '% - %')
                OR id NOT IN (SELECT track_id FROM track_profiles)
            )
            """
        )

    return conn.execute(
        f"SELECT id, path FROM tracks WHERE {' AND '.join(clauses)} ORDER BY path",
        params,
    ).fetchall()


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
