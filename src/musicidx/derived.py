"""Derived feature tags and listening context-fit scores."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from musicidx.db import utc_now
from musicidx.profiles import rebuild_track_profile

DERIVED_TAG_SOURCE = "derived:features"

CONTEXTS = [
    "background",
    "lounge_bar",
    "warm_lounge",
    "dinner",
    "cooking",
    "focus",
    "dark_ambient",
    "no_vocals_background",
    "party",
    "club",
    "workout",
    "driving",
]


@dataclass(slots=True)
class DerivedTag:
    tag: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContextFit:
    context: str
    score: float
    confidence: float
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DerivedSummary:
    processed: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    tags_written: int = 0
    contexts_written: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def rebuild_derived_signals(
    conn: sqlite3.Connection,
    *,
    track_id: str | None = None,
    include_missing: bool = False,
) -> DerivedSummary:
    """Recompute local feature-derived tags and context fit rows."""
    summary = DerivedSummary()
    rows = _select_audio_feature_rows(conn, track_id=track_id, include_missing=include_missing)
    now = utc_now()
    for row in rows:
        summary.processed += 1
        try:
            tags = derive_feature_tags(row)
            contexts = derive_context_fit(row)
            _save_derived_tags(conn, row["track_id"], tags, updated_at=now)
            _save_context_fit(conn, row["track_id"], contexts, updated_at=now)
            rebuild_track_profile(conn, row["track_id"], updated_at=now)
        except Exception:  # pragma: no cover - defensive per-track isolation
            summary.errors += 1
            continue
        summary.updated += 1
        summary.tags_written += len(tags)
        summary.contexts_written += len(contexts)
    conn.commit()
    return summary


def derive_feature_tags(row: sqlite3.Row | dict[str, Any]) -> list[DerivedTag]:
    """Create search tags from deterministic audio-feature thresholds."""
    bpm = _value(row, "bpm")
    energy = _value(row, "energy")
    danceability = _value(row, "danceability")
    aggression = _value(row, "aggression")
    brightness = _value(row, "brightness")
    vocalness = _value(row, "vocalness")
    instrumentalness = _value(row, "instrumentalness")
    has_vocal_evidence = vocalness is not None or instrumentalness is not None
    contexts = {fit.context: fit.score for fit in derive_context_fit(row)}

    tags: list[DerivedTag] = []
    _append(tags, "low_energy", _max_score(energy, 0.38, softness=0.35))
    _append(tags, "medium_energy", _range_score(energy, 0.30, 0.70, softness=0.30))
    _append(tags, "high_energy", _min_score(energy, 0.62, softness=0.35))
    _append(tags, "danceable", _min_score(danceability, 0.58, softness=0.35))
    _append(tags, "not_danceable", _max_score(danceability, 0.42, softness=0.35))
    _append(tags, "low_aggression", _max_score(aggression, 0.35, softness=0.35))
    _append(tags, "aggressive", _min_score(aggression, 0.62, softness=0.35), threshold=0.75)
    _append(tags, "dark", _max_score(brightness, 0.42, softness=0.35))
    _append(tags, "bright", _min_score(brightness, 0.58, softness=0.35))
    _append(tags, "slow", _max_score(bpm, 95.0, softness=45.0))
    _append(tags, "midtempo", _range_score(bpm, 85.0, 125.0, softness=45.0))
    _append(tags, "fast", _min_score(bpm, 125.0, softness=45.0))
    _append(tags, "dance_tempo", _range_score(bpm, 110.0, 132.0, softness=35.0))

    for context, tag, threshold in [
        ("background", "background_friendly", 0.85),
        ("lounge_bar", "lounge_friendly", 0.70),
        ("warm_lounge", "warm_lounge", 0.70),
        ("club", "club_friendly", 0.70),
        ("focus", "focus_friendly", 0.85),
    ]:
        _append(tags, tag, contexts.get(context, 0.0), threshold=threshold)
    if has_vocal_evidence:
        _append(
            tags,
            "no_vocals_background",
            contexts.get("no_vocals_background", 0.0),
            threshold=0.78,
        )

    return sorted(tags, key=lambda item: (item.score, item.tag), reverse=True)


def derive_context_fit(row: sqlite3.Row | dict[str, Any]) -> list[ContextFit]:
    """Score transparent listening contexts from basic audio features."""
    bpm = _value(row, "bpm")
    energy = _value(row, "energy")
    danceability = _value(row, "danceability")
    aggression = _value(row, "aggression")
    brightness = _value(row, "brightness")
    vocalness = _value(row, "vocalness")
    instrumentalness = _value(row, "instrumentalness")

    low_aggression = _max_score(aggression, 0.35, softness=0.35)
    very_low_aggression = _max_score(aggression, 0.22, softness=0.30)
    medium_energy = _range_score(energy, 0.25, 0.68, softness=0.30)
    low_mid_energy = _range_score(energy, 0.18, 0.58, softness=0.30)
    high_energy = _min_score(energy, 0.62, softness=0.35)
    dance = _min_score(danceability, 0.58, softness=0.35)
    groove = _range_score(danceability, 0.42, 0.82, softness=0.30)
    calm_dance = _range_score(danceability, 0.20, 0.62, softness=0.35)
    dark = _max_score(brightness, 0.42, softness=0.35)
    balanced_brightness = _range_score(brightness, 0.30, 0.68, softness=0.30)
    bright = _min_score(brightness, 0.58, softness=0.35)
    warm = _range_score(brightness, 0.28, 0.58, softness=0.35)
    dance_tempo = _range_score(bpm, 110.0, 132.0, softness=35.0)
    mid_tempo = _range_score(bpm, 80.0, 124.0, softness=45.0)
    fast_tempo = _min_score(bpm, 125.0, softness=45.0)
    driving_tempo = _range_score(bpm, 85.0, 150.0, softness=45.0)
    has_vocal_evidence = vocalness is not None or instrumentalness is not None
    low_vocal = _max_score(vocalness, 0.20, softness=0.30) if vocalness is not None else 0.0
    instrumental = (
        _min_score(instrumentalness, 0.70, softness=0.30)
        if instrumentalness is not None
        else 0.0
    )
    vocal_or_instrumental = max(low_vocal, instrumental)
    no_vocal_prior = vocal_or_instrumental if has_vocal_evidence else 0.35

    scores: dict[str, tuple[float, dict[str, float], float]] = {
        "background": _weighted(
            {
                "low_aggression": low_aggression,
                "background_energy": _range_score(energy, 0.14, 0.52, softness=0.25),
                "not_too_bright": _max_score(brightness, 0.68, softness=0.30),
                "not_too_danceable": _max_score(danceability, 0.68, softness=0.35),
            },
            confidence=0.68,
        ),
        "lounge_bar": _weighted(
            {
                "low_aggression": low_aggression,
                "medium_energy": medium_energy,
                "warmth": warm,
                "groove": groove,
                "mid_tempo": mid_tempo,
            },
            confidence=0.74,
        ),
        "warm_lounge": _weighted(
            {
                "warmth": warm,
                "low_aggression": low_aggression,
                "low_mid_energy": low_mid_energy,
                "groove": groove,
                "balanced_brightness": balanced_brightness,
            },
            confidence=0.74,
        ),
        "dinner": _weighted(
            {
                "low_aggression": low_aggression,
                "low_mid_energy": low_mid_energy,
                "warmth": warm,
                "background": low_mid_energy * low_aggression,
            },
            confidence=0.70,
        ),
        "cooking": _weighted(
            {
                "medium_energy": medium_energy,
                "low_aggression": low_aggression,
                "groove": groove,
                "balanced_brightness": balanced_brightness,
            },
            confidence=0.70,
        ),
        "focus": _weighted(
            {
                "very_low_aggression": very_low_aggression,
                "low_mid_energy": low_mid_energy,
                "calm_dance": calm_dance,
                "low_vocal_or_instrumental": no_vocal_prior,
            },
            confidence=0.56 if not has_vocal_evidence else 0.78,
        ),
        "dark_ambient": _weighted(
            {
                "dark": dark,
                "low_aggression": low_aggression,
                "low_mid_energy": low_mid_energy,
                "not_too_danceable": calm_dance,
            },
            confidence=0.72,
        ),
        "no_vocals_background": _no_vocals_context_score(
            {
                "low_vocal_or_instrumental": no_vocal_prior,
                "low_aggression": low_aggression,
                "background_energy": _range_score(energy, 0.14, 0.52, softness=0.25),
                "background": low_mid_energy * low_aggression,
            },
            has_vocal_evidence=has_vocal_evidence,
        ),
        "party": _weighted(
            {
                "high_energy": high_energy,
                "danceable": dance,
                "bright": bright,
                "not_too_aggressive": low_aggression,
            },
            confidence=0.72,
        ),
        "club": _weighted(
            {
                "danceable": dance,
                "dance_tempo": dance_tempo,
                "high_energy": high_energy,
                "not_too_aggressive": low_aggression,
            },
            confidence=0.76,
        ),
        "workout": _weighted(
            {
                "high_energy": high_energy,
                "fast_tempo": fast_tempo,
                "danceable": dance,
                "aggression": _min_score(aggression, 0.45, softness=0.45),
            },
            confidence=0.72,
        ),
        "driving": _weighted(
            {
                "medium_or_high_energy": max(medium_energy, high_energy),
                "driving_tempo": driving_tempo,
                "groove": groove,
                "not_too_aggressive": low_aggression,
            },
            confidence=0.72,
        ),
    }

    output = [
        ContextFit(
            context=context,
            score=round(max(0.0, min(1.0, score)), 6),
            confidence=confidence,
            evidence=evidence,
        )
        for context, (score, evidence, confidence) in scores.items()
        if score >= 0.05
    ]
    return sorted(output, key=lambda item: (item.score, item.context), reverse=True)


def _select_audio_feature_rows(
    conn: sqlite3.Connection,
    *,
    track_id: str | None,
    include_missing: bool,
) -> list[sqlite3.Row]:
    clauses = ["t.quarantined_at IS NULL"]
    params: list[Any] = []
    if not include_missing:
        clauses.append("t.missing_at IS NULL")
    if track_id is not None:
        clauses.append("t.id = ?")
        params.append(track_id)
    return conn.execute(
        f"""
        SELECT
            t.id AS track_id,
            af.bpm,
            af.energy,
            af.danceability,
            af.aggression,
            af.brightness,
            af.vocalness,
            af.instrumentalness
        FROM audio_features af
        JOIN tracks t ON t.id = af.track_id
        WHERE {' AND '.join(clauses)}
        ORDER BY t.path
        """,
        params,
    ).fetchall()


def _save_derived_tags(
    conn: sqlite3.Connection,
    track_id: str,
    tags: list[DerivedTag],
    *,
    updated_at: str,
) -> None:
    conn.execute(
        "DELETE FROM track_tags WHERE track_id = ? AND source = ?",
        (track_id, DERIVED_TAG_SOURCE),
    )
    for tag in tags:
        conn.execute(
            """
            INSERT INTO track_tags (track_id, source, tag, score, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(track_id, source, tag) DO UPDATE SET
                score = excluded.score,
                updated_at = excluded.updated_at
            """,
            (track_id, DERIVED_TAG_SOURCE, tag.tag, tag.score, updated_at),
        )


def _save_context_fit(
    conn: sqlite3.Connection,
    track_id: str,
    contexts: list[ContextFit],
    *,
    updated_at: str,
) -> None:
    conn.execute("DELETE FROM track_context_fit WHERE track_id = ?", (track_id,))
    for fit in contexts:
        conn.execute(
            """
            INSERT INTO track_context_fit (
                track_id, context, score, confidence, evidence_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id, context) DO UPDATE SET
                score = excluded.score,
                confidence = excluded.confidence,
                evidence_json = excluded.evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                track_id,
                fit.context,
                fit.score,
                fit.confidence,
                json.dumps(fit.evidence, sort_keys=True),
                updated_at,
            ),
        )


def _weighted(
    parts: dict[str, float],
    *,
    confidence: float,
) -> tuple[float, dict[str, float], float]:
    if not parts:
        return 0.0, {}, confidence
    score = sum(parts.values()) / len(parts)
    evidence = {key: round(max(0.0, min(1.0, value)), 6) for key, value in parts.items()}
    return score, evidence, round(max(0.0, min(1.0, confidence)), 6)


def _no_vocals_context_score(
    parts: dict[str, float],
    *,
    has_vocal_evidence: bool,
) -> tuple[float, dict[str, float], float]:
    score, evidence, confidence = _weighted(
        parts,
        confidence=0.82 if has_vocal_evidence else 0.34,
    )
    if not has_vocal_evidence:
        evidence["vocal_evidence_missing"] = 1.0
        score = min(score, 0.45)
    return score, evidence, confidence


def _append(tags: list[DerivedTag], tag: str, score: float, *, threshold: float = 0.35) -> None:
    if score >= threshold:
        tags.append(DerivedTag(tag=tag, score=round(max(0.0, min(1.0, score)), 6)))


def _range_score(value: float | None, low: float, high: float, *, softness: float) -> float:
    if value is None:
        return 0.0
    if low <= value <= high:
        return 1.0
    distance = min(abs(value - low), abs(value - high))
    return max(0.0, 1.0 - distance / softness)


def _max_score(value: float | None, maximum: float, *, softness: float) -> float:
    if value is None:
        return 0.0
    if value <= maximum:
        return 1.0
    return max(0.0, 1.0 - (value - maximum) / softness)


def _min_score(value: float | None, minimum: float, *, softness: float) -> float:
    if value is None:
        return 0.0
    if value >= minimum:
        return 1.0
    return max(0.0, 1.0 - (minimum - value) / softness)


def _value(row: sqlite3.Row | dict[str, Any], key: str) -> float | None:
    try:
        value = row[key]
    except (KeyError, IndexError):
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
