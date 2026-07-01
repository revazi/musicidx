"""Deterministic track matching reports.

Fingerprints and content hashes are authoritative identity evidence. Metadata and duration
are useful supporting/advisory evidence, but they never prove duplicate/same-recording
identity on their own. Optional audio embeddings are similarity-only evidence; semantic
embeddings are intentionally not used here.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from musicidx.analyzer.embeddings import EmbeddingError, blob_to_vector, normalize_vector


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

MATCH_POLICY = {
    "identity_authority": ["content_hash", "chromaprint"],
    "metadata_role": "advisory_only",
    "duration_role": "supporting_only",
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
    return MatchReport(
        schema_version=1,
        track_a=_match_track(left),
        track_b=_match_track(right),
        decision=decision,
        identity_decision=identity_decision,
        confidence=confidence,
        confidence_score=confidence_score,
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
    reports = [report for report in reports if report.confidence_score > 0.0]
    reports.sort(
        key=lambda report: (
            -report.confidence_score,
            -_decision_priority(report.decision),
            report.track_b.artist or "",
            report.track_b.title or "",
            report.track_b.path,
        )
    )
    return reports[: max(1, limit)]


def _get_track(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT
            id, path, title, artist, album, content_hash, chromaprint,
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
) -> list[sqlite3.Row]:
    clauses = ["id != ?"]
    params: list[Any] = [source["id"]]
    evidence_clauses: list[str] = []
    for field_name in ("content_hash", "chromaprint", "artist_title_norm"):
        value = _clean(source[field_name])
        if value:
            evidence_clauses.append(f"{field_name} = ?")
            params.append(value)
    artist = _clean(source["artist"]).casefold()
    if artist:
        evidence_clauses.append("lower(trim(artist)) = ?")
        params.append(artist)
    audio_models = _audio_embedding_models_for_track(
        conn,
        source["id"],
        kind=audio_embedding_kind,
    )
    if audio_models:
        placeholders = ", ".join("?" for _ in audio_models)
        evidence_clauses.append(
            "id IN ("
            "SELECT track_id FROM embeddings "
            f"WHERE kind = ? AND model IN ({placeholders})"
            ")"
        )
        params.extend([audio_embedding_kind, *audio_models])
    if not evidence_clauses:
        return []
    clauses.append("(" + " OR ".join(evidence_clauses) + ")")
    if not include_missing:
        clauses.append("missing_at IS NULL")
    return conn.execute(
        f"""
        SELECT id
        FROM tracks
        WHERE {' AND '.join(clauses)}
        ORDER BY coalesce(artist, ''), coalesce(title, ''), path
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
            "duration_sec": row["duration_sec"] is not None,
            "fingerprint_duration": row["fingerprint_duration"] is not None,
            "artist_title_norm": bool(_clean(row["artist_title_norm"])),
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
        _duration_evidence(left, right, duration_tolerance_sec=duration_tolerance_sec),
        _string_identity_evidence(
            "artist_title_norm",
            left["artist_title_norm"],
            right["artist_title_norm"],
            role="metadata_advisory",
            match_score=0.55,
            decisive_on_match=False,
        ),
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
    score = max(0.0, 1.0 - (delta / max(duration_tolerance_sec, 1.0))) if matched else 0.0
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


def _short_key(value: str) -> str:
    if len(value) <= 24:
        return value
    return f"{value[:12]}…{value[-8:]}"
