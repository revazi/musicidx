"""Deterministic track matching reports.

Fingerprints and content hashes are authoritative identity evidence. Metadata and duration
are useful supporting/advisory evidence, but they never prove duplicate/same-recording
identity on their own. Optional audio embeddings are similarity-only evidence; semantic
embeddings are intentionally not used here.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from musicidx.analyzer.embeddings import EmbeddingError, blob_to_vector, normalize_vector
from musicidx.chromaprint_frames import row_frames


class TrackMatchError(RuntimeError):
    """Raised when a requested match report cannot be built."""


class TrackNotFoundError(TrackMatchError):
    """Raised when a requested track ID is not present in the index."""


@dataclass(frozen=True, slots=True)
class MatchTrack:
    track_id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    duration_sec: float | None
    fingerprint_duration: float | None
    missing: bool
    identity: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchEvidence:
    source: str
    role: str
    status: str
    score: float
    decisive: bool
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchReport:
    schema_version: int
    track_a: MatchTrack
    track_b: MatchTrack
    decision: str
    identity_decision: str
    confidence: str
    confidence_score: float
    candidate_score: float
    candidate_kind: str
    candidate_strength: str
    candidate_summary: str
    candidate_reasons: list[str]
    candidate_scores: dict[str, float]
    reasons: list[str]
    evidence: list[MatchEvidence]
    warnings: list[str]
    policy: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "track_a": self.track_a.as_dict(),
            "track_b": self.track_b.as_dict(),
            "decision": self.decision,
            "identity_decision": self.identity_decision,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "candidate_score": self.candidate_score,
            "candidate_kind": self.candidate_kind,
            "candidate_strength": self.candidate_strength,
            "candidate_summary": self.candidate_summary,
            "candidate_reasons": self.candidate_reasons,
            "candidate_scores": self.candidate_scores,
            "reasons": self.reasons,
            "evidence": [item.as_dict() for item in self.evidence],
            "warnings": self.warnings,
            "policy": self.policy,
        }


VERSION_CONFLICT_ALIASES = {
    "acoustic": "acoustic",
    "bootleg": "bootleg",
    "club mix": "club_mix",
    "cover": "cover",
    "edit": "edit",
    "extended mix": "extended_mix",
    "instrumental": "instrumental",
    "live": "live",
    "radio edit": "radio_edit",
    "remaster": "remaster",
    "remastered": "remaster",
    "remix": "remix",
    "rework": "rework",
    "version": "version",
    "vip": "vip",
}
VERSION_CONFLICT_PHRASES = sorted(VERSION_CONFLICT_ALIASES, key=len, reverse=True)
AUDIO_EMBEDDING_KIND = "audio_clap"
AUDIO_SIMILARITY_THRESHOLD = 0.85
CANDIDATE_SCORE_THRESHOLD = 0.03
FEATURE_SIMILARITY_THRESHOLD = 0.74
FINGERPRINT_RELATED_THRESHOLD = 0.08
FINGERPRINT_SIMILARITY_THRESHOLD = 0.42
FINGERPRINT_MAX_ALIGN_OFFSET = 120
FINGERPRINT_MAX_BIT_ERROR = 2
FINGERPRINT_ANCHOR_TOP_OFFSETS = 32
FEATURE_SIMILARITY_FIELDS = (
    "bpm",
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "vocalness",
    "speechiness",
    "aggression",
    "brightness",
)
CANDIDATE_SOURCE_PRIORITY = (
    "content_hash",
    "chromaprint",
    "fingerprint_similarity",
    "name",
    "duration",
    "artist_similarity",
    "artist_title_norm",
    "album_similarity",
    "filename_stem",
    "feature_similarity",
    "audio_embedding",
)
MATCH_NAME_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "baby",
    "feat",
    "ft",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "the",
    "to",
    "u",
    "you",
    "your",
}

MATCH_POLICY = {
    "identity_authority": ["content_hash", "chromaprint"],
    "candidate_ranking": list(CANDIDATE_SOURCE_PRIORITY),
    "candidate_score_role": "closest-track-ranking-only_not_identity",
    "metadata_role": "advisory_only",
    "filename_role": "advisory_only",
    "duration_role": "supporting_only",
    "fingerprint_similarity_role": (
        "soundwave_candidate_ranking_only_exact_chromaprint_still_required_for_identity"
    ),
    "features_role": "similarity_only_not_identity",
    "semantic_embeddings": "not_used_for_identity",
    "audio_embeddings": "similarity_only_not_identity",
    "llm": "not_used",
    "version_conflicts": "related_version_not_duplicate_without_authoritative_identity_match",
    "audio_embedding_kind": AUDIO_EMBEDDING_KIND,
    "audio_similarity_threshold": AUDIO_SIMILARITY_THRESHOLD,
}


def compare_tracks(
    conn: sqlite3.Connection,
    track_a_id: str,
    track_b_id: str,
    *,
    duration_tolerance_sec: float = 3.0,
    audio_embedding_kind: str = AUDIO_EMBEDDING_KIND,
    audio_similarity_threshold: float = AUDIO_SIMILARITY_THRESHOLD,
) -> MatchReport:
    """Compare two indexed tracks and return a deterministic MatchReport."""
    left = _get_track(conn, track_a_id)
    right = _get_track(conn, track_b_id)
    evidence = _match_evidence(
        conn,
        left,
        right,
        duration_tolerance_sec=duration_tolerance_sec,
        audio_embedding_kind=audio_embedding_kind,
        audio_similarity_threshold=audio_similarity_threshold,
    )
    decision, identity_decision, confidence, confidence_score, reasons, warnings = _decision(
        evidence
    )
    candidate_score, candidate_reasons, candidate_scores = _candidate_rank(evidence)
    candidate_kind, candidate_strength, candidate_summary = _candidate_classification(
        decision=decision,
        candidate_score=candidate_score,
        candidate_scores=candidate_scores,
    )
    return MatchReport(
        schema_version=1,
        track_a=_match_track(left),
        track_b=_match_track(right),
        decision=decision,
        identity_decision=identity_decision,
        confidence=confidence,
        confidence_score=confidence_score,
        candidate_score=candidate_score,
        candidate_kind=candidate_kind,
        candidate_strength=candidate_strength,
        candidate_summary=candidate_summary,
        candidate_reasons=candidate_reasons,
        candidate_scores=candidate_scores,
        reasons=reasons,
        evidence=evidence,
        warnings=warnings,
        policy=MATCH_POLICY,
    )


def find_track_matches(
    conn: sqlite3.Connection,
    track_id: str,
    *,
    limit: int = 10,
    include_missing: bool = True,
    duration_tolerance_sec: float = 3.0,
    audio_embedding_kind: str = AUDIO_EMBEDDING_KIND,
    audio_similarity_threshold: float = AUDIO_SIMILARITY_THRESHOLD,
) -> list[MatchReport]:
    """Find deterministic local match candidates for one indexed track."""
    source = _get_track(conn, track_id)
    candidates = _candidate_tracks_for_source(
        conn,
        source,
        include_missing=include_missing,
        audio_embedding_kind=audio_embedding_kind,
        duration_tolerance_sec=duration_tolerance_sec,
    )
    reports = [
        compare_tracks(
            conn,
            track_id,
            row["id"],
            duration_tolerance_sec=duration_tolerance_sec,
            audio_embedding_kind=audio_embedding_kind,
            audio_similarity_threshold=audio_similarity_threshold,
        )
        for row in candidates
    ]
    reports.sort(key=_candidate_sort_key)
    max_results = max(1, limit)
    minimum_results = min(5, max_results, len(reports))
    selected = [report for report in reports if _is_informative_match_report(report)]
    if len(selected) < minimum_results:
        selected_ids = {report.track_b.track_id for report in selected}
        for report in reports:
            if report.track_b.track_id in selected_ids:
                continue
            selected.append(report)
            selected_ids.add(report.track_b.track_id)
            if len(selected) >= minimum_results:
                break
    return selected[:max_results]


def _get_track(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT
            id, path, title, artist, album, content_hash, chromaprint,
            chromaprint_algorithm, chromaprint_frames, chromaprint_frame_count,
            duration_sec, fingerprint_duration, artist_title_norm, missing_at
        FROM tracks
        WHERE id = ?
        """,
        (track_id,),
    ).fetchone()
    if row is None:
        raise TrackNotFoundError(f"track not found: {track_id}")
    return row


def _candidate_tracks_for_source(
    conn: sqlite3.Connection,
    source: sqlite3.Row,
    *,
    include_missing: bool,
    audio_embedding_kind: str,
    duration_tolerance_sec: float,
) -> list[sqlite3.Row]:
    del audio_embedding_kind, duration_tolerance_sec
    clauses = ["id != ?"]
    params: list[Any] = [source["id"]]
    if not include_missing:
        clauses.append("missing_at IS NULL")
    return conn.execute(
        f"""
        SELECT id
        FROM tracks
        WHERE {' AND '.join(clauses)}
        ORDER BY coalesce(title, ''), coalesce(artist, ''), path
        """,
        params,
    ).fetchall()


def _match_track(row: sqlite3.Row) -> MatchTrack:
    return MatchTrack(
        track_id=row["id"],
        path=row["path"],
        title=row["title"],
        artist=row["artist"],
        album=row["album"],
        duration_sec=_optional_float(row["duration_sec"]),
        fingerprint_duration=_optional_float(row["fingerprint_duration"]),
        missing=row["missing_at"] is not None,
        identity={
            "content_hash": bool(_clean(row["content_hash"])),
            "chromaprint": bool(_clean(row["chromaprint"])),
            "chromaprint_frames": bool(row_frames(row)),
            "duration_sec": row["duration_sec"] is not None,
            "fingerprint_duration": row["fingerprint_duration"] is not None,
            "artist_title_norm": bool(_clean(row["artist_title_norm"])),
            "filename_stem": bool(_filename_stem(row["path"])),
        },
    )


def _match_evidence(
    conn: sqlite3.Connection,
    left: sqlite3.Row,
    right: sqlite3.Row,
    *,
    duration_tolerance_sec: float,
    audio_embedding_kind: str,
    audio_similarity_threshold: float,
) -> list[MatchEvidence]:
    evidence = [
        _name_evidence(left, right),
        _string_identity_evidence(
            "content_hash",
            left["content_hash"],
            right["content_hash"],
            role="identity_authority",
            match_score=1.0,
            decisive_on_match=True,
        ),
        _string_identity_evidence(
            "chromaprint",
            left["chromaprint"],
            right["chromaprint"],
            role="identity_authority",
            match_score=0.96,
            decisive_on_match=True,
        ),
        _fingerprint_similarity_evidence(left, right),
        _duration_evidence(left, right, duration_tolerance_sec=duration_tolerance_sec),
        _metadata_text_similarity_evidence("artist_similarity", left["artist"], right["artist"]),
        _string_identity_evidence(
            "artist_title_norm",
            left["artist_title_norm"],
            right["artist_title_norm"],
            role="metadata_advisory",
            match_score=0.55,
            decisive_on_match=False,
        ),
        _metadata_text_similarity_evidence("album_similarity", left["album"], right["album"]),
        _filename_stem_evidence(left, right),
        _feature_similarity_evidence(conn, left["id"], right["id"]),
        _version_conflict_evidence(left, right),
        _audio_embedding_evidence(
            conn,
            left["id"],
            right["id"],
            kind=audio_embedding_kind,
            threshold=audio_similarity_threshold,
        ),
    ]
    return evidence


def _string_identity_evidence(
    source: str,
    left_value: Any,
    right_value: Any,
    *,
    role: str,
    match_score: float,
    decisive_on_match: bool,
) -> MatchEvidence:
    left = _clean(left_value)
    right = _clean(right_value)
    if not left or not right:
        return MatchEvidence(
            source=source,
            role=role,
            status="missing",
            score=0.0,
            decisive=False,
            details={"left_available": bool(left), "right_available": bool(right)},
        )
    matched = left == right
    details: dict[str, Any] = {"left_key": _short_key(left), "right_key": _short_key(right)}
    return MatchEvidence(
        source=source,
        role=role,
        status="match" if matched else "mismatch",
        score=match_score if matched else 0.0,
        decisive=decisive_on_match and matched,
        details=details,
    )


def _name_evidence(left: sqlite3.Row, right: sqlite3.Row) -> MatchEvidence:
    left_names = _track_name_candidates(left)
    right_names = _track_name_candidates(right)
    if not left_names or not right_names:
        return MatchEvidence(
            source="name",
            role="candidate_ranking_only",
            status="missing",
            score=0.0,
            decisive=False,
            details={"left_available": bool(left_names), "right_available": bool(right_names)},
        )
    best: tuple[float, str, str] = (0.0, left_names[0], right_names[0])
    for left_name in left_names:
        for right_name in right_names:
            score = _name_similarity(left_name, right_name)
            if score > best[0]:
                best = (score, left_name, right_name)
    score, left_name, right_name = best
    if score >= 0.88:
        status = "match"
    elif score >= 0.45:
        status = "partial"
    else:
        status = "mismatch"
    return MatchEvidence(
        source="name",
        role="candidate_ranking_only",
        status=status,
        score=round(score, 6),
        decisive=False,
        details={"left_name": _short_key(left_name), "right_name": _short_key(right_name)},
    )


def _duration_evidence(
    left: sqlite3.Row,
    right: sqlite3.Row,
    *,
    duration_tolerance_sec: float,
) -> MatchEvidence:
    left_duration = _duration_for_row(left)
    right_duration = _duration_for_row(right)
    if left_duration is None or right_duration is None:
        return MatchEvidence(
            source="duration",
            role="supporting_only",
            status="missing",
            score=0.0,
            decisive=False,
            details={
                "left_sec": left_duration,
                "right_sec": right_duration,
                "tolerance_sec": duration_tolerance_sec,
            },
        )
    delta = abs(left_duration - right_duration)
    matched = delta <= duration_tolerance_sec
    score = max(0.0, 1.0 - (delta / max(duration_tolerance_sec * 4.0, 12.0)))
    return MatchEvidence(
        source="duration",
        role="supporting_only",
        status="match" if matched else "mismatch",
        score=round(score, 6),
        decisive=False,
        details={
            "left_sec": round(left_duration, 6),
            "right_sec": round(right_duration, 6),
            "delta_sec": round(delta, 6),
            "tolerance_sec": duration_tolerance_sec,
        },
    )


def _audio_embedding_models_for_track(
    conn: sqlite3.Connection,
    track_id: str,
    *,
    kind: str,
) -> list[str]:
    rows = conn.execute(
        """
        SELECT model
        FROM embeddings
        WHERE track_id = ?
          AND kind = ?
        ORDER BY model
        """,
        (track_id, kind),
    ).fetchall()
    return [row["model"] for row in rows]


def _audio_embedding_evidence(
    conn: sqlite3.Connection,
    left_track_id: str,
    right_track_id: str,
    *,
    kind: str,
    threshold: float,
) -> MatchEvidence:
    rows = conn.execute(
        """
        SELECT track_id, model, dim, vector
        FROM embeddings
        WHERE kind = ?
          AND track_id IN (?, ?)
        ORDER BY model, track_id
        """,
        (kind, left_track_id, right_track_id),
    ).fetchall()
    by_model: dict[str, dict[str, sqlite3.Row]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["track_id"]] = row
    shared_models = [
        model
        for model, model_rows in by_model.items()
        if left_track_id in model_rows and right_track_id in model_rows
    ]
    if not shared_models:
        return MatchEvidence(
            source="audio_embedding",
            role="similarity_only",
            status="missing",
            score=0.0,
            decisive=False,
            details={"kind": kind, "shared_models": []},
        )

    best: tuple[str, float] | None = None
    errors: list[str] = []
    for model in shared_models:
        try:
            left_row = by_model[model][left_track_id]
            right_row = by_model[model][right_track_id]
            left_vector = normalize_vector(
                blob_to_vector(left_row["vector"], int(left_row["dim"]))
            )
            right_vector = normalize_vector(
                blob_to_vector(right_row["vector"], int(right_row["dim"]))
            )
            if left_vector.shape != right_vector.shape:
                errors.append(f"{model}:dimension_mismatch")
                continue
            similarity = float(np.dot(left_vector, right_vector))
        except (EmbeddingError, ValueError) as exc:
            errors.append(f"{model}:{exc}")
            continue
        if best is None or similarity > best[1]:
            best = (model, similarity)

    if best is None:
        return MatchEvidence(
            source="audio_embedding",
            role="similarity_only",
            status="error",
            score=0.0,
            decisive=False,
            details={"kind": kind, "shared_models": shared_models, "errors": errors[:3]},
        )

    model, similarity = best
    similar = similarity >= threshold
    details: dict[str, Any] = {
        "kind": kind,
        "model": model,
        "threshold": threshold,
        "shared_models": shared_models,
    }
    if errors:
        details["errors"] = errors[:3]
    return MatchEvidence(
        source="audio_embedding",
        role="similarity_only",
        status="similar" if similar else "dissimilar",
        score=round(max(-1.0, min(1.0, similarity)), 6),
        decisive=False,
        details=details,
    )


def _metadata_text_similarity_evidence(
    source: str,
    left_value: Any,
    right_value: Any,
) -> MatchEvidence:
    left = _normalized_text(left_value)
    right = _normalized_text(right_value)
    if not left or not right:
        return MatchEvidence(
            source=source,
            role="metadata_advisory",
            status="missing",
            score=0.0,
            decisive=False,
            details={"left_available": bool(left), "right_available": bool(right)},
        )
    score = _name_similarity(left, right)
    if score >= 0.92:
        status = "match"
    elif score >= 0.70:
        status = "partial"
    else:
        status = "mismatch"
    return MatchEvidence(
        source=source,
        role="metadata_advisory",
        status=status,
        score=round(score, 6),
        decisive=False,
        details={"left_value": _short_key(left), "right_value": _short_key(right)},
    )


def _fingerprint_similarity_evidence(left: sqlite3.Row, right: sqlite3.Row) -> MatchEvidence:
    left_fingerprint = _clean(left["chromaprint"])
    right_fingerprint = _clean(right["chromaprint"])
    left_frames = row_frames(left)
    right_frames = row_frames(right)
    if not left_frames or not right_frames:
        return MatchEvidence(
            source="fingerprint_similarity",
            role="soundwave_candidate_ranking_only",
            status="missing",
            score=0.0,
            decisive=False,
            details={
                "left_available": bool(left_frames),
                "right_available": bool(right_frames),
                "left_encoded_available": bool(left_fingerprint),
                "right_encoded_available": bool(right_fingerprint),
            },
        )
    score, details = _chromaprint_similarity(left_frames, right_frames)
    if left_fingerprint == right_fingerprint:
        status = "same"
    elif score >= FINGERPRINT_SIMILARITY_THRESHOLD:
        status = "similar"
    elif score >= FINGERPRINT_RELATED_THRESHOLD:
        status = "related"
    else:
        status = "dissimilar"
    return MatchEvidence(
        source="fingerprint_similarity",
        role="soundwave_candidate_ranking_only",
        status=status,
        score=round(score, 6),
        decisive=False,
        details={
            "related_threshold": FINGERPRINT_RELATED_THRESHOLD,
            "similar_threshold": FINGERPRINT_SIMILARITY_THRESHOLD,
            "left_key": _short_key(left_fingerprint),
            "right_key": _short_key(right_fingerprint),
            **details,
        },
    )


def _filename_stem_evidence(left: sqlite3.Row, right: sqlite3.Row) -> MatchEvidence:
    left_stem = _filename_stem(left["path"])
    right_stem = _filename_stem(right["path"])
    if not left_stem or not right_stem:
        return MatchEvidence(
            source="filename_stem",
            role="metadata_advisory",
            status="missing",
            score=0.0,
            decisive=False,
            details={"left_available": bool(left_stem), "right_available": bool(right_stem)},
        )
    matched = left_stem == right_stem
    return MatchEvidence(
        source="filename_stem",
        role="metadata_advisory",
        status="match" if matched else "mismatch",
        score=0.35 if matched else 0.0,
        decisive=False,
        details={"left_stem": _short_key(left_stem), "right_stem": _short_key(right_stem)},
    )


def _feature_similarity_evidence(
    conn: sqlite3.Connection,
    left_track_id: str,
    right_track_id: str,
) -> MatchEvidence:
    rows = conn.execute(
        f"""
        SELECT track_id, {', '.join(FEATURE_SIMILARITY_FIELDS)}
        FROM audio_features
        WHERE track_id IN (?, ?)
        """,
        (left_track_id, right_track_id),
    ).fetchall()
    by_track = {row["track_id"]: row for row in rows}
    left = by_track.get(left_track_id)
    right = by_track.get(right_track_id)
    if left is None or right is None:
        return MatchEvidence(
            source="feature_similarity",
            role="similarity_only",
            status="missing",
            score=0.0,
            decisive=False,
            details={"left_available": left is not None, "right_available": right is not None},
        )

    scores: dict[str, float] = {}
    for field in FEATURE_SIMILARITY_FIELDS:
        left_value = left[field]
        right_value = right[field]
        if left_value is None or right_value is None:
            continue
        scores[field] = _feature_value_similarity(field, float(left_value), float(right_value))
    if not scores:
        return MatchEvidence(
            source="feature_similarity",
            role="similarity_only",
            status="missing",
            score=0.0,
            decisive=False,
            details={"shared_fields": []},
        )
    score = sum(scores.values()) / len(scores)
    return MatchEvidence(
        source="feature_similarity",
        role="similarity_only",
        status="similar" if score >= FEATURE_SIMILARITY_THRESHOLD else "dissimilar",
        score=round(score, 6),
        decisive=False,
        details={
            "threshold": FEATURE_SIMILARITY_THRESHOLD,
            "shared_fields": sorted(scores),
            "field_scores": {key: round(value, 6) for key, value in sorted(scores.items())},
        },
    )


def _version_conflict_evidence(left: sqlite3.Row, right: sqlite3.Row) -> MatchEvidence:
    left_artist = _normalized_text(left["artist"])
    right_artist = _normalized_text(right["artist"])
    left_title = _clean(left["title"])
    right_title = _clean(right["title"])
    if not left_artist or not right_artist or not left_title or not right_title:
        return MatchEvidence(
            source="version_conflict",
            role="metadata_conflict_advisory",
            status="missing",
            score=0.0,
            decisive=False,
            details={
                "left_artist_available": bool(left_artist),
                "right_artist_available": bool(right_artist),
                "left_title_available": bool(left_title),
                "right_title_available": bool(right_title),
            },
        )

    same_artist = left_artist == right_artist
    left_base, left_tokens = _base_title_and_version_tokens(left_title)
    right_base, right_tokens = _base_title_and_version_tokens(right_title)
    base_title_match = bool(left_base and right_base and left_base == right_base)
    has_conflict = same_artist and base_title_match and left_tokens != right_tokens and bool(
        left_tokens or right_tokens
    )
    return MatchEvidence(
        source="version_conflict",
        role="metadata_conflict_advisory",
        status="conflict" if has_conflict else "none",
        score=1.0 if has_conflict else 0.0,
        decisive=False,
        details={
            "same_artist": same_artist,
            "base_title_match": base_title_match,
            "left_base_title": left_base,
            "right_base_title": right_base,
            "left_tokens": sorted(left_tokens),
            "right_tokens": sorted(right_tokens),
        },
    )


def _base_title_and_version_tokens(title: str) -> tuple[str, set[str]]:
    normalized = _normalized_text(title)
    tokens: set[str] = set()
    base = f" {normalized} "
    for phrase in VERSION_CONFLICT_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, base):
            tokens.add(VERSION_CONFLICT_ALIASES[phrase])
            base = re.sub(pattern, " ", base)
    base = re.sub(r"\b\d{4}\b", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base, tokens


def _track_name_candidates(row: sqlite3.Row) -> list[str]:
    title = _normalized_text(row["title"])
    artist_title = _normalized_text(
        " ".join(part for part in [row["artist"], row["title"]] if part)
    )
    return _unique_non_empty([title, artist_title])


def _name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    substring_score = 0.0
    if left in right or right in left:
        substring_score = min(len(left), len(right)) / max(len(left), len(right), 1)
        substring_score = max(0.72, min(0.96, substring_score))
    token_score = _token_similarity(left, right)
    sequence_score = difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
    if substring_score == 0.0 and token_score == 0.0 and sequence_score < 0.62:
        sequence_score = 0.0
    return max(substring_score, token_score, sequence_score)


def _token_similarity(left: str, right: str) -> float:
    left_tokens = _content_name_tokens(left)
    right_tokens = _content_name_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap_tokens = left_tokens.intersection(right_tokens)
    if not overlap_tokens:
        return 0.0
    overlap = len(overlap_tokens)
    union = len(left_tokens.union(right_tokens))
    containment = overlap / min(len(left_tokens), len(right_tokens))
    jaccard = overlap / union
    score = max(jaccard, containment * 0.92)
    if overlap == 1:
        score = min(score, 0.42)
    return score


def _content_name_tokens(value: str) -> set[str]:
    return {
        token
        for token in value.split()
        if len(token) > 1 and token not in MATCH_NAME_STOP_WORDS
    }


def _chromaprint_similarity(
    left_frames: tuple[int, ...],
    right_frames: tuple[int, ...],
) -> tuple[float, dict[str, Any]]:
    if not left_frames or not right_frames:
        return 0.0, {"method": "decoded_chromaprint_alignment", "decoded": False}
    if left_frames == right_frames:
        return 1.0, {
            "method": "decoded_chromaprint_alignment",
            "decoded": True,
            "left_frames": len(left_frames),
            "right_frames": len(right_frames),
            "best_offset": 0,
            "best_matches": min(len(left_frames), len(right_frames)),
        }
    score, best_offset, best_matches = _fingerprint_alignment_score(left_frames, right_frames)
    return round(score, 6), {
        "method": "decoded_chromaprint_alignment",
        "decoded": True,
        "left_frames": len(left_frames),
        "right_frames": len(right_frames),
        "max_bit_error": FINGERPRINT_MAX_BIT_ERROR,
        "best_offset": best_offset,
        "best_matches": best_matches,
    }


def _fingerprint_alignment_score(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[float, int, int]:
    if not left or not right:
        return 0.0, 0, 0
    candidate_offsets = set(_local_fingerprint_offsets(left, right))
    candidate_offsets.update(_anchor_fingerprint_offsets(left, right))
    candidate_offsets.add(0)
    best_score = 0.0
    best_offset = 0
    best_matches = 0
    denominator = max(1, min(len(left), len(right)))
    for offset in candidate_offsets:
        matches = _fingerprint_matches_at_offset(left, right, offset)
        score = matches / denominator
        if score > best_score or (score == best_score and matches > best_matches):
            best_score = score
            best_offset = offset
            best_matches = matches
    return min(1.0, best_score), best_offset, best_matches


def _local_fingerprint_offsets(left: tuple[int, ...], right: tuple[int, ...]) -> list[int]:
    counts: Counter[int] = Counter()
    for left_index, left_frame in enumerate(left):
        right_begin = max(0, left_index - FINGERPRINT_MAX_ALIGN_OFFSET)
        right_end = min(len(right), left_index + FINGERPRINT_MAX_ALIGN_OFFSET)
        for right_index in range(right_begin, right_end):
            if (left_frame ^ right[right_index]).bit_count() <= FINGERPRINT_MAX_BIT_ERROR:
                counts[left_index - right_index] += 1
    return [offset for offset, _count in counts.most_common(FINGERPRINT_ANCHOR_TOP_OFFSETS)]


def _anchor_fingerprint_offsets(left: tuple[int, ...], right: tuple[int, ...]) -> list[int]:
    positions: dict[int, list[int]] = defaultdict(list)
    for right_index, right_frame in enumerate(right):
        if len(positions[right_frame]) < 64:
            positions[right_frame].append(right_index)
    counts: Counter[int] = Counter()
    step = max(1, len(left) // 1500)
    for left_index in range(0, len(left), step):
        for right_index in positions.get(left[left_index], []):
            counts[left_index - right_index] += 1
    return [offset for offset, _count in counts.most_common(FINGERPRINT_ANCHOR_TOP_OFFSETS)]


def _fingerprint_matches_at_offset(
    left: tuple[int, ...],
    right: tuple[int, ...],
    offset: int,
) -> int:
    left_start = max(0, offset)
    left_end = min(len(left), len(right) + offset)
    if left_end <= left_start:
        return 0
    matches = 0
    for left_index in range(left_start, left_end):
        right_index = left_index - offset
        if (left[left_index] ^ right[right_index]).bit_count() <= FINGERPRINT_MAX_BIT_ERROR:
            matches += 1
    return matches


def _feature_value_similarity(field: str, left: float, right: float) -> float:
    delta = abs(left - right)
    if field == "bpm":
        return max(0.0, 1.0 - (delta / 80.0))
    return max(0.0, 1.0 - min(1.0, delta))


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _decision(
    evidence: list[MatchEvidence],
) -> tuple[str, str, str, float, list[str], list[str]]:
    by_source = {item.source: item for item in evidence}
    content = by_source["content_hash"]
    chromaprint = by_source["chromaprint"]
    duration = by_source["duration"]
    artist_title = by_source["artist_title_norm"]
    filename_stem = by_source["filename_stem"]
    version_conflict = by_source["version_conflict"]
    audio_embedding = by_source["audio_embedding"]
    warnings: list[str] = []

    if content.status == "match":
        return (
            "exact_duplicate",
            "same",
            "high",
            1.0,
            ["same content hash"],
            warnings,
        )
    if chromaprint.status == "match" and duration.status in {"match", "missing"}:
        if duration.status == "missing":
            warnings.append("duration_missing_for_fingerprint_match")
        return (
            "same_recording",
            "same",
            "high",
            0.96,
            ["same chromaprint"],
            warnings,
        )
    if chromaprint.status == "match" and duration.status == "mismatch":
        warnings.append("same_chromaprint_duration_mismatch")
        return (
            "possible_recording_match",
            "possible",
            "medium",
            0.72,
            ["same chromaprint but duration differs"],
            warnings,
        )

    if version_conflict.status == "conflict":
        warnings.append("version_conflict_blocks_metadata_identity")
        return (
            "related_version_not_duplicate",
            "unknown",
            "medium",
            0.4,
            ["artist/title metadata indicates a live/remix/edit/version conflict"],
            warnings,
        )

    authoritative_mismatch = content.status == "mismatch" or chromaprint.status == "mismatch"
    if authoritative_mismatch:
        reasons = [
            f"{item.source} mismatch"
            for item in (content, chromaprint)
            if item.status == "mismatch"
        ]
        if artist_title.status == "match":
            reasons.append("artist/title metadata matches but is advisory only")
            warnings.append("metadata_match_blocked_by_identity_mismatch")
        if filename_stem.status == "match":
            reasons.append("filename stem matches but is advisory only")
            warnings.append("filename_match_blocked_by_identity_mismatch")
        return (
            "no_identity_match",
            "unknown",
            "medium",
            0.0,
            reasons,
            warnings,
        )

    if artist_title.status == "match" and duration.status == "match":
        warnings.append("metadata_duration_match_is_advisory_only")
        return (
            "possible_metadata_match",
            "possible",
            "medium",
            0.68,
            ["same artist/title metadata and similar duration"],
            warnings,
        )
    if artist_title.status == "match":
        warnings.append("metadata_match_is_advisory_only")
        return (
            "possible_metadata_match",
            "possible",
            "low",
            0.45,
            ["same artist/title metadata"],
            warnings,
        )
    if filename_stem.status == "match" and duration.status == "match":
        warnings.append("filename_duration_match_is_advisory_only")
        return (
            "possible_metadata_match",
            "possible",
            "low",
            0.52,
            ["same filename stem and similar duration"],
            warnings,
        )
    if filename_stem.status == "match":
        warnings.append("filename_match_is_advisory_only")
        return (
            "possible_metadata_match",
            "possible",
            "low",
            0.25,
            ["same filename stem"],
            warnings,
        )

    if audio_embedding.status == "similar":
        warnings.append("audio_embedding_similarity_only")
        return (
            "sound_similar_only",
            "unknown",
            "low",
            round(max(0.0, float(audio_embedding.score or 0.0)) * 0.5, 6),
            ["audio embedding similarity only; not identity evidence"],
            warnings,
        )

    missing_authority = content.status == "missing" and chromaprint.status == "missing"
    if missing_authority:
        warnings.append("missing_identity_evidence")
    return (
        "insufficient_evidence",
        "unknown",
        "low",
        0.0,
        ["no authoritative identity match found"],
        warnings,
    )


def _candidate_rank(
    evidence: list[MatchEvidence],
) -> tuple[float, list[str], dict[str, float]]:
    by_source = {item.source: item for item in evidence}
    candidate_scores = {
        source: _candidate_source_score(by_source.get(source))
        for source in CANDIDATE_SOURCE_PRIORITY
    }
    weights = {
        "content_hash": 0.20,
        "chromaprint": 0.20,
        "fingerprint_similarity": 0.25,
        "name": 0.16,
        "duration": 0.07,
        "artist_similarity": 0.05,
        "artist_title_norm": 0.04,
        "album_similarity": 0.02,
        "filename_stem": 0.02,
        "feature_similarity": 0.03,
        "audio_embedding": 0.02,
    }
    score = min(
        1.0,
        sum(candidate_scores[source] * weights[source] for source in CANDIDATE_SOURCE_PRIORITY),
    )
    reasons = [
        _candidate_reason(source, by_source[source], candidate_scores[source])
        for source in CANDIDATE_SOURCE_PRIORITY
        if source in by_source and candidate_scores[source] > 0.0
    ]
    return (
        round(score, 6),
        reasons[:6],
        {source: round(value, 6) for source, value in candidate_scores.items()},
    )


def _candidate_classification(
    *,
    decision: str,
    candidate_score: float,
    candidate_scores: dict[str, float],
) -> tuple[str, str, str]:
    name = candidate_scores.get("name", 0.0)
    content = candidate_scores.get("content_hash", 0.0)
    chromaprint = candidate_scores.get("chromaprint", 0.0)
    fingerprint = candidate_scores.get("fingerprint_similarity", 0.0)
    duration = candidate_scores.get("duration", 0.0)
    artist = candidate_scores.get("artist_similarity", 0.0)
    artist_title = candidate_scores.get("artist_title_norm", 0.0)
    album = candidate_scores.get("album_similarity", 0.0)
    filename = candidate_scores.get("filename_stem", 0.0)
    features = candidate_scores.get("feature_similarity", 0.0)
    audio = candidate_scores.get("audio_embedding", 0.0)

    if decision == "exact_duplicate" or content >= 1.0:
        return ("exact_duplicate", "strong", "Exact duplicate: same content hash")
    if decision == "same_recording" or chromaprint >= 0.95:
        return ("same_recording", "strong", "Same recording: exact fingerprint match")
    if decision == "possible_recording_match" or fingerprint >= FINGERPRINT_SIMILARITY_THRESHOLD:
        return ("possible_recording_match", "strong", "Possible recording match: close fingerprint")
    if fingerprint >= FINGERPRINT_RELATED_THRESHOLD:
        return ("soundwave_related", "medium", "Soundwave-related: partial fingerprint overlap")
    if decision == "related_version_not_duplicate":
        return ("related_version", "medium", "Related version/remix/live/edit, not a duplicate")
    if name >= 0.88 and artist >= 0.92:
        strength = "strong" if duration >= 0.75 or features >= 0.90 else "medium"
        return ("same_title_artist", strength, "Same/very close title and artist")
    if name >= 0.88 and artist_title > 0.0:
        return ("same_title_metadata", "medium", "Same/very close title with metadata support")
    if name >= 0.88:
        return ("same_title", "medium", "Same/very close title")
    if name >= 0.70:
        return ("close_title", "medium", "Close title match")
    if artist >= 0.92 and (duration >= 0.75 or album >= 0.80):
        return ("same_artist_context", "medium", "Same artist with duration/album support")
    if filename >= 0.30:
        return ("filename_match", "medium", "Same/close filename")
    if audio >= AUDIO_SIMILARITY_THRESHOLD:
        return ("audio_similar", "weak", "Audio embedding similarity only")
    if features >= 0.90:
        return ("feature_similar", "weak", "Feature-similar fallback")
    if duration >= 0.75:
        return ("duration_similar", "weak", "Duration-similar fallback")
    if candidate_score > 0.0:
        return ("nearest_fallback", "weak", "Weak nearest fallback")
    return ("no_nearby_signal", "weak", "No strong nearby signal")


def _candidate_source_score(evidence: MatchEvidence | None) -> float:
    if evidence is None:
        return 0.0
    if evidence.source == "fingerprint_similarity":
        if evidence.status not in {"same", "similar", "related"}:
            return 0.0
        return max(0.0, min(1.0, float(evidence.score or 0.0)))
    raw_similarity_sources = {"duration", "feature_similarity", "audio_embedding"}
    if evidence.source in raw_similarity_sources and evidence.status not in {"missing", "error"}:
        return max(0.0, min(1.0, float(evidence.score or 0.0)))
    if evidence.status not in {"match", "partial", "same", "similar"}:
        return 0.0
    return max(0.0, min(1.0, float(evidence.score or 0.0)))


def _candidate_reason(source: str, evidence: MatchEvidence, score: float) -> str:
    labels = {
        "name": "name",
        "content_hash": "content hash",
        "chromaprint": "exact fingerprint",
        "fingerprint_similarity": "fingerprint similarity",
        "duration": "duration",
        "artist_similarity": "artist",
        "artist_title_norm": "artist/title metadata",
        "album_similarity": "album",
        "filename_stem": "filename",
        "feature_similarity": "features",
        "audio_embedding": "audio embedding",
    }
    return f"{labels.get(source, source)} {evidence.status} {score:.2f}"


def _candidate_sort_key(report: MatchReport) -> tuple[Any, ...]:
    scores = report.candidate_scores
    exact_soundwave_score = max(
        scores.get("content_hash", 0.0),
        scores.get("chromaprint", 0.0),
    )
    fuzzy_soundwave_score = scores.get("fingerprint_similarity", 0.0)
    return (
        -report.candidate_score,
        -exact_soundwave_score,
        -fuzzy_soundwave_score,
        -scores.get("name", 0.0),
        -scores.get("duration", 0.0),
        -scores.get("artist_similarity", 0.0),
        -scores.get("artist_title_norm", 0.0),
        -scores.get("album_similarity", 0.0),
        -scores.get("filename_stem", 0.0),
        -scores.get("feature_similarity", 0.0),
        -scores.get("audio_embedding", 0.0),
        -_decision_priority(report.decision),
        report.track_b.artist or "",
        report.track_b.title or "",
        report.track_b.path,
    )


def _is_informative_match_report(report: MatchReport) -> bool:
    if report.candidate_score >= CANDIDATE_SCORE_THRESHOLD:
        return True
    if report.confidence_score > 0.0:
        return True
    if report.decision != "no_identity_match":
        return False
    informative_warnings = {
        "metadata_match_blocked_by_identity_mismatch",
        "filename_match_blocked_by_identity_mismatch",
    }
    return bool(informative_warnings.intersection(report.warnings))


def _decision_priority(decision: str) -> int:
    priorities = {
        "exact_duplicate": 5,
        "same_recording": 4,
        "possible_recording_match": 3,
        "possible_metadata_match": 2,
        "related_version_not_duplicate": 2,
        "sound_similar_only": 1,
        "no_identity_match": 1,
        "insufficient_evidence": 0,
    }
    return priorities.get(decision, 0)


def _duration_for_row(row: sqlite3.Row) -> float | None:
    value = row["fingerprint_duration"] if row["fingerprint_duration"] is not None else None
    if value is None:
        value = row["duration_sec"]
    return _optional_float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _filename_stem(path: Any) -> str:
    decoded_path = urllib.parse.unquote(str(path or ""))
    stem = Path(decoded_path).stem
    return _normalized_text(stem)


def _unique_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _short_key(value: str) -> str:
    if len(value) <= 24:
        return value
    return f"{value[:12]}…{value[-8:]}"
