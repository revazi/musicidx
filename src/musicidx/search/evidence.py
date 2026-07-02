"""Candidate evidence summaries for search results.

These helpers make retrieval/evidence provenance inspectable without changing ranking.
The ranker still computes scores; this module only converts a score breakdown into a
small, stable JSON-friendly evidence object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateEvidenceSource:
    """One evidence source that contributed to a candidate/result."""

    source: str
    role: str
    score: float
    matched: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Compact evidence-source summary for a search result."""

    retrieved_by: tuple[str, ...]
    sources: tuple[CandidateEvidenceSource, ...]
    identity: dict[str, bool]
    semantic_only: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieved_by": list(self.retrieved_by),
            "sources": [source.as_dict() for source in self.sources],
            "identity": dict(self.identity),
            "semantic_only": self.semantic_only,
        }


def build_candidate_evidence(breakdown: Mapping[str, Any]) -> dict[str, Any]:
    """Build a stable candidate-evidence summary from a search score breakdown."""
    sources = _sources_from_breakdown(breakdown)
    retrieved_by = tuple(source.source for source in sources if source.matched)
    identity = _identity_summary(breakdown.get("identity") or {})
    semantic_only = bool((breakdown.get("evidence") or {}).get("semantic_only"))
    return CandidateEvidence(
        retrieved_by=retrieved_by,
        sources=tuple(sources),
        identity=identity,
        semantic_only=semantic_only,
    ).as_dict()


def _sources_from_breakdown(breakdown: Mapping[str, Any]) -> list[CandidateEvidenceSource]:
    sources = [
        _source(
            "metadata",
            role="candidate_evidence",
            score=_score(breakdown, "metadata_score"),
            details={"matches": _count(breakdown.get("metadata_matches"))},
        ),
        _source(
            "profile_text",
            role="candidate_evidence",
            score=_score(breakdown, "text_score"),
            details=_direct_content_details(breakdown, "profile"),
        ),
        _source(
            "tags",
            role="candidate_evidence",
            score=_score(breakdown, "tag_score"),
            details={"matches": _count(breakdown.get("matched_tags"))},
        ),
        _source(
            "context_fit",
            role="candidate_evidence",
            score=_score(breakdown, "context_score"),
            details={"matches": _count(breakdown.get("matched_contexts"))},
        ),
        _source(
            "audio_features",
            role="candidate_evidence",
            score=_score(breakdown, "feature_score"),
            details={"matches": _count(breakdown.get("feature_reasons"))},
        ),
        _source(
            "semantic_profile",
            role="candidate_evidence",
            score=_score(breakdown, "semantic_score"),
            details={},
        ),
        _source(
            "feedback",
            role="rerank_adjustment",
            score=abs(_score(breakdown, "feedback_score")),
            details={"direction": _feedback_direction(_score(breakdown, "feedback_score"))},
        ),
    ]
    return [source for source in sources if source.matched]


def _source(
    source: str,
    *,
    role: str,
    score: float,
    details: dict[str, Any],
) -> CandidateEvidenceSource:
    return CandidateEvidenceSource(
        source=source,
        role=role,
        score=round(score, 6),
        matched=score > 0.0,
        details={key: value for key, value in details.items() if value not in {None, 0, ""}},
    )


def _score(breakdown: Mapping[str, Any], key: str) -> float:
    return float(breakdown.get(key) or 0.0)


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list | tuple) else 0


def _direct_content_details(breakdown: Mapping[str, Any], source: str) -> dict[str, Any]:
    direct = breakdown.get("direct_content_evidence") or {}
    matched_by_source = direct.get("matched_by_source") or {}
    matches = matched_by_source.get(source) or []
    return {"matches": len(matches)} if isinstance(matches, list | tuple | set) else {}


def _feedback_direction(score: float) -> str | None:
    if score > 0.0:
        return "positive"
    if score < 0.0:
        return "negative"
    return None


def _identity_summary(identity: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "content_hash": bool(identity.get("content_hash")),
        "chromaprint": bool(identity.get("chromaprint")),
        "duration_sec": identity.get("duration_sec") is not None,
        "artist_title_norm": bool(identity.get("artist_title_norm")),
    }
