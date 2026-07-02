"""Track profile text/JSON rebuild helpers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musicidx.db import utc_now
from musicidx.metadata import TrackMetadata, build_track_profile
from musicidx.profile_documents import (
    PROFILE_SCHEMA_VERSION,
    build_profile_document,
    profile_source_fingerprint,
)
from musicidx.profile_documents import (
    profile_json_text as serialize_profile_json,
)
from musicidx.tempo import perceived_tempo_bpm, tempo_descriptors_from_metadata_and_tags


@dataclass(slots=True)
class ProfileRebuildSummary:
    """Summary counters for profile regeneration."""

    processed: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    schema_version: int = PROFILE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def rebuild_track_profiles(
    conn: sqlite3.Connection,
    *,
    track_id: str | None = None,
    include_missing: bool = False,
) -> ProfileRebuildSummary:
    """Regenerate materialized profiles from current metadata/features/tags."""
    summary = ProfileRebuildSummary()
    rows = _select_profile_tracks(conn, track_id=track_id, include_missing=include_missing)
    for row in rows:
        summary.processed += 1
        try:
            rebuild_track_profile(conn, row["id"])
        except ValueError:
            summary.errors += 1
            continue
        summary.updated += 1
    conn.commit()
    return summary


def rebuild_track_profile(
    conn: sqlite3.Connection,
    track_id: str,
    *,
    updated_at: str | None = None,
) -> tuple[str, str]:
    """Rebuild profile text/JSON from metadata, audio features, and tags."""
    updated_at = updated_at or utc_now()
    track = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
    if track is None:
        raise ValueError(f"track not found: {track_id}")

    metadata = TrackMetadata(
        title=track["title"],
        artist=track["artist"],
        album=track["album"],
        album_artist=track["album_artist"],
        genre=track["genre"],
        date=track["date"],
        track_number=track["track_number"],
        disc_number=track["disc_number"],
        duration_sec=track["duration_sec"],
        codec=track["codec"],
        sample_rate=track["sample_rate"],
        bit_rate=track["bit_rate"],
        channels=track["channels"],
    )
    base_text, _base_json = build_track_profile(metadata, Path(track["path"]))

    tag_rows = conn.execute(
        """
        SELECT source, tag, score
        FROM track_tags
        WHERE track_id = ?
        ORDER BY score DESC, tag ASC, source ASC
        """,
        (track_id,),
    ).fetchall()
    tags: list[dict[str, Any]] = [
        {
            "source": row["source"],
            "tag": row["tag"],
            "score": float(row["score"]),
        }
        for row in tag_rows
    ]
    tempo_descriptors = tempo_descriptors_from_metadata_and_tags(metadata.as_dict(), tags)

    audio_features: dict[str, Any] | None = None
    audio_row = conn.execute(
        "SELECT * FROM audio_features WHERE track_id = ?",
        (track_id,),
    ).fetchone()
    if audio_row is not None:
        audio_text = describe_audio_feature_row(audio_row, tempo_descriptors=tempo_descriptors)
        if audio_text:
            base_text = f"{base_text} Audio: {audio_text}".strip()
        audio_features = _audio_features_dict(audio_row)

    if tag_rows:
        tag_text = ", ".join(
            f"{row['tag']} {float(row['score']):.2f}" for row in tag_rows[:20]
        )
        base_text = f"{base_text} Tags: {tag_text}.".strip()

    context_fit = _context_fit(conn, track_id)

    claim_confidence, claim_provenance = _metadata_claim_summary(conn, track_id)
    profile_text = base_text.strip()
    document = build_profile_document(
        track_id=track_id,
        metadata=metadata.as_dict(),
        path=Path(track["path"]),
        profile_text=profile_text,
        generated_at=updated_at,
        analysis_version=track["analysis_version"],
        normalized={
            "title_norm": track["title_norm"],
            "artist_norm": track["artist_norm"],
            "album_norm": track["album_norm"],
            "artist_title_norm": track["artist_title_norm"],
        },
        metadata_confidence=track["metadata_confidence"],
        external_match_confidence=track["external_match_confidence"],
        field_confidence=claim_confidence,
        provenance=claim_provenance,
        audio_features=audio_features,
        tags=tags,
        context_fit=context_fit,
        missing=track["missing_at"] is not None,
    )
    profile_json = serialize_profile_json(document)
    source_fingerprint = profile_source_fingerprint(document)
    embedding_text = document["search_text"]["embedding_text"]

    conn.execute(
        """
        INSERT INTO track_profiles (
            track_id, profile_text, embedding_text, profile_json, profile_schema_version,
            source_fingerprint, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            profile_text = excluded.profile_text,
            embedding_text = excluded.embedding_text,
            profile_json = excluded.profile_json,
            profile_schema_version = excluded.profile_schema_version,
            source_fingerprint = excluded.source_fingerprint,
            updated_at = excluded.updated_at
        """,
        (
            track_id,
            profile_text,
            embedding_text,
            profile_json,
            PROFILE_SCHEMA_VERSION,
            source_fingerprint,
            updated_at,
        ),
    )
    conn.execute("DELETE FROM tracks_fts WHERE track_id = ?", (track_id,))
    conn.execute(
        """
        INSERT INTO tracks_fts (track_id, title, artist, album, genre, profile_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            track["title"],
            track["artist"],
            track["album"],
            track["genre"],
            profile_text,
        ),
    )
    return profile_text, profile_json


def describe_audio_feature_row(
    row: sqlite3.Row,
    *,
    tempo_descriptors: list[str] | None = None,
) -> str:
    """Create human-readable audio descriptors from an audio_features row."""
    parts: list[str] = []
    if row["energy"] is not None:
        parts.append(f"{_band(float(row['energy']))} energy")
    if row["bpm"] is not None:
        bpm = perceived_tempo_bpm(row["bpm"], descriptors=tempo_descriptors)
        if bpm is not None:
            parts.append(f"tempo around {bpm:.0f} BPM")
    if row["brightness"] is not None:
        parts.append(f"{_band(float(row['brightness']))} brightness")
    if row["danceability"] is not None:
        parts.append(f"{_band(float(row['danceability']))} danceability")
    if row["aggression"] is not None:
        parts.append(f"{_band(float(row['aggression']))} aggression")
    if row["key_name"] and row["mode"]:
        parts.append(f"rough key {row['key_name']} {row['mode']}")
    return ", ".join(parts) + "." if parts else ""


def _audio_features_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "bpm": row["bpm"],
        "key_name": row["key_name"],
        "mode": row["mode"],
        "dynamic_range": row["dynamic_range"],
        "energy": row["energy"],
        "valence": row["valence"],
        "danceability": row["danceability"],
        "acousticness": row["acousticness"],
        "instrumentalness": row["instrumentalness"],
        "vocalness": row["vocalness"],
        "speechiness": row["speechiness"],
        "aggression": row["aggression"],
        "brightness": row["brightness"],
        "spectral_centroid_mean": row["spectral_centroid_mean"],
        "spectral_centroid_std": row["spectral_centroid_std"],
        "spectral_flatness_mean": row["spectral_flatness_mean"],
        "spectral_rolloff_mean": row["spectral_rolloff_mean"],
        "zero_crossing_rate_mean": row["zero_crossing_rate_mean"],
        "mfcc_mean": _parse_json(row["mfcc_mean_json"], default=[]),
        "mfcc_std": _parse_json(row["mfcc_std_json"], default=[]),
        "raw_features": _parse_json(row["raw_features_json"], default={}),
    }


def _context_fit(conn: sqlite3.Connection, track_id: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT context, score
        FROM track_context_fit
        WHERE track_id = ?
        ORDER BY score DESC, context ASC
        """,
        (track_id,),
    ).fetchall()
    return {row["context"]: float(row["score"] or 0.0) for row in rows}


def _metadata_claim_summary(
    conn: sqlite3.Connection,
    track_id: str,
) -> tuple[dict[str, float], dict[str, str]]:
    rows = conn.execute(
        """
        SELECT field_name, source, source_detail, confidence
        FROM track_metadata_claims
        WHERE track_id = ? AND selected = 1
        ORDER BY confidence DESC, created_at DESC
        """,
        (track_id,),
    ).fetchall()
    confidence: dict[str, float] = {}
    provenance: dict[str, str] = {}
    for row in rows:
        field_name = row["field_name"]
        if field_name in confidence:
            continue
        confidence[field_name] = float(row["confidence"] or 0.0)
        detail = f":{row['source_detail']}" if row["source_detail"] else ""
        provenance[field_name] = f"{row['source']}{detail}"
    return confidence, provenance


def _select_profile_tracks(
    conn: sqlite3.Connection,
    *,
    track_id: str | None,
    include_missing: bool,
) -> list[sqlite3.Row]:
    clauses = ["quarantined_at IS NULL"]
    params: list[Any] = []
    if not include_missing:
        clauses.append("missing_at IS NULL")
    if track_id is not None:
        clauses.append("id = ?")
        params.append(track_id)
    return conn.execute(
        f"SELECT id FROM tracks WHERE {' AND '.join(clauses)} ORDER BY path",
        params,
    ).fetchall()


def _parse_json(value: str | None, *, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _band(value: float) -> str:
    if value < 0.33:
        return "low"
    if value < 0.66:
        return "medium"
    return "high"
