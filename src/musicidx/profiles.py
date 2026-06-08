"""Track profile text/JSON rebuild helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from musicidx.db import utc_now
from musicidx.metadata import TrackMetadata, build_track_profile


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
    base_text, base_json = build_track_profile(metadata, Path(track["path"]))
    profile_json = json.loads(base_json)
    profile_json["analysis_version"] = track["analysis_version"]

    audio_row = conn.execute(
        "SELECT * FROM audio_features WHERE track_id = ?",
        (track_id,),
    ).fetchone()
    if audio_row is not None:
        audio_text = describe_audio_feature_row(audio_row)
        if audio_text:
            base_text = f"{base_text} Audio: {audio_text}".strip()
        profile_json["audio_features"] = _audio_features_dict(audio_row)

    tag_rows = conn.execute(
        """
        SELECT source, tag, score
        FROM track_tags
        WHERE track_id = ?
        ORDER BY score DESC, tag ASC, source ASC
        """,
        (track_id,),
    ).fetchall()
    if tag_rows:
        tag_text = ", ".join(
            f"{row['tag']} {float(row['score']):.2f}" for row in tag_rows[:20]
        )
        base_text = f"{base_text} Tags: {tag_text}.".strip()
        profile_json["tags"] = [
            {
                "source": row["source"],
                "tag": row["tag"],
                "score": float(row["score"]),
            }
            for row in tag_rows
        ]

    profile_text = base_text.strip()
    profile_json_text = json.dumps(profile_json, sort_keys=True)

    conn.execute(
        """
        INSERT INTO track_profiles (track_id, profile_text, profile_json, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            profile_text = excluded.profile_text,
            profile_json = excluded.profile_json,
            updated_at = excluded.updated_at
        """,
        (track_id, profile_text, profile_json_text, updated_at),
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
    return profile_text, profile_json_text


def describe_audio_feature_row(row: sqlite3.Row) -> str:
    """Create human-readable audio descriptors from an audio_features row."""
    parts: list[str] = []
    if row["energy"] is not None:
        parts.append(f"{_band(float(row['energy']))} energy")
    if row["bpm"] is not None:
        parts.append(f"tempo around {float(row['bpm']):.0f} BPM")
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
        "danceability": row["danceability"],
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
