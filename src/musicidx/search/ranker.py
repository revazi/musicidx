"""Hybrid local search and ranking."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from musicidx.analyzer.embeddings import EmbeddingError, search_semantic
from musicidx.search.explain import build_explanation
from musicidx.search.intent import (
    IntentHints,
    SearchIntent,
    normalize_tag_terms,
    normalize_terms,
    parse_intent_dynamic,
)


@dataclass(slots=True)
class SearchResult:
    track_id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    score: float
    breakdown: dict[str, Any]
    explanation: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResponse:
    query: str
    intent: SearchIntent
    results: list[SearchResult]
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.as_dict(),
            "results": [result.as_dict() for result in self.results],
            "diagnostics": self.diagnostics,
        }


def search_music(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int | None = None,
    include_missing: bool = False,
    semantic_model: str | None = None,
    explain: bool = False,
    llm_hints: IntentHints | None = None,
    parser: str = "dynamic",
    llm_error: str | None = None,
) -> SearchResponse:
    """Search local tracks with dynamically parsed intent and hybrid ranking."""
    intent = parse_intent_dynamic(
        query,
        conn,
        limit=limit,
        include_missing=include_missing,
        semantic_model=semantic_model or "sentence-transformers/all-MiniLM-L6-v2",
        llm_hints=llm_hints,
        parser=parser,
        llm_error=llm_error,
    )
    track_rows = _select_track_candidates(conn, include_missing=include_missing)
    tags_by_track = _load_tags(conn, include_missing=include_missing)
    fts_scores = _fts_scores(conn, query, include_missing=include_missing)
    semantic_scores, semantic_error = _semantic_scores(
        conn,
        intent,
        include_missing=include_missing,
    )
    feedback_scores = _feedback_scores(conn, query, include_missing=include_missing)

    weights = _weights(use_semantic=bool(semantic_scores), use_feedback=bool(feedback_scores))
    scored_results: list[SearchResult] = []
    for row in track_rows:
        track_id = row["track_id"]
        track_tags = tags_by_track.get(track_id, [])
        breakdown = _score_track(
            row,
            track_tags,
            intent,
            fts_score=fts_scores.get(track_id, 0.0),
            semantic_score=semantic_scores.get(track_id, 0.0),
            feedback_score=feedback_scores.get(track_id, 0.0),
            weights=weights,
        )
        scored_results.append(
            SearchResult(
                track_id=track_id,
                path=row["path"],
                title=row["title"],
                artist=row["artist"],
                album=row["album"],
                genre=row["genre"],
                score=round(float(breakdown["final_score"]), 6),
                breakdown=breakdown,
                explanation=build_explanation(breakdown) if explain else [],
            )
        )

    ranked = sorted(scored_results, key=lambda result: result.score, reverse=True)
    diversified = _apply_diversity(ranked, intent)
    diagnostics = {
        "candidate_count": len(track_rows),
        "fts_candidate_count": len(fts_scores),
        "semantic_candidate_count": len(semantic_scores),
        "feedback_candidate_count": len(feedback_scores),
        "semantic_error": semantic_error,
        "weights": weights,
    }
    return SearchResponse(
        query=query,
        intent=intent,
        results=diversified[: intent.limit],
        diagnostics=diagnostics,
    )


def _select_track_candidates(
    conn: sqlite3.Connection,
    *,
    include_missing: bool,
) -> list[sqlite3.Row]:
    missing_clause = "" if include_missing else "WHERE t.missing_at IS NULL"
    return conn.execute(
        f"""
        SELECT
            t.id AS track_id,
            t.path,
            t.title,
            t.artist,
            t.album,
            t.genre,
            t.missing_at,
            p.profile_text,
            af.bpm,
            af.energy,
            af.danceability,
            af.aggression,
            af.brightness
        FROM tracks t
        LEFT JOIN track_profiles p ON p.track_id = t.id
        LEFT JOIN audio_features af ON af.track_id = t.id
        {missing_clause}
        ORDER BY t.path
        """
    ).fetchall()


def _load_tags(
    conn: sqlite3.Connection,
    *,
    include_missing: bool,
) -> dict[str, list[dict[str, Any]]]:
    missing_clause = "" if include_missing else "AND tr.missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT tt.track_id, tt.source, tt.tag, tt.score
        FROM track_tags tt
        JOIN tracks tr ON tr.id = tt.track_id
        WHERE 1 = 1
          {missing_clause}
        ORDER BY tt.track_id, tt.score DESC
        """
    ).fetchall()
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(row["track_id"], []).append(
            {
                "source": row["source"],
                "tag": row["tag"],
                "score": float(row["score"]),
            }
        )
    return output


def _fts_scores(
    conn: sqlite3.Connection,
    query: str,
    *,
    include_missing: bool,
) -> dict[str, float]:
    terms = normalize_terms(query)
    if not terms:
        return {}
    fts_query = " OR ".join(terms)
    missing_clause = "" if include_missing else "AND t.missing_at IS NULL"
    try:
        rows = conn.execute(
            f"""
            SELECT tracks_fts.track_id, bm25(tracks_fts) AS bm25_score
            FROM tracks_fts
            JOIN tracks t ON t.id = tracks_fts.track_id
            WHERE tracks_fts MATCH ?
              {missing_clause}
            ORDER BY bm25_score
            LIMIT 500
            """,
            (fts_query,),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    return {row["track_id"]: 1.0 / (index + 1) for index, row in enumerate(rows)}


def _semantic_scores(
    conn: sqlite3.Connection,
    intent: SearchIntent,
    *,
    include_missing: bool,
) -> tuple[dict[str, float], str | None]:
    if not intent.use_semantic or intent.semantic_model is None:
        return {}, None
    try:
        results = search_semantic(
            conn,
            intent.query,
            model_name=intent.semantic_model,
            limit=500,
            include_missing=include_missing,
        )
    except EmbeddingError as exc:
        return {}, str(exc)
    except Exception as exc:  # pragma: no cover - dependency/runtime errors vary
        return {}, f"semantic search unavailable: {exc}"
    return {
        result.track_id: max(0.0, min(1.0, (result.score + 1.0) / 2.0))
        for result in results
    }, None


def _score_track(
    row: sqlite3.Row,
    track_tags: list[dict[str, Any]],
    intent: SearchIntent,
    *,
    fts_score: float,
    semantic_score: float,
    feedback_score: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    tag_score, matched_tags, avoided_tags = _tag_score(track_tags, intent)
    feature_score, feature_reasons = _feature_score(row, intent)
    text_score = max(fts_score, _profile_term_score(row["profile_text"] or "", intent.query_terms))

    final_score = (
        weights["semantic"] * semantic_score
        + weights["tags"] * tag_score
        + weights["features"] * feature_score
        + weights["text"] * text_score
        + weights.get("feedback", 0.0) * feedback_score
    )

    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "tag_score": tag_score,
        "feature_score": feature_score,
        "text_score": text_score,
        "feedback_score": feedback_score,
        "matched_tags": matched_tags,
        "avoided_tags": avoided_tags,
        "feature_reasons": feature_reasons,
    }


def _tag_score(
    track_tags: list[dict[str, Any]],
    intent: SearchIntent,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    prefer = intent.prefer_tags + intent.prefer_tag_concepts
    avoid = intent.avoid_tags + intent.avoid_tag_concepts
    if not prefer and not avoid:
        return 0.5, [], []

    positive = 0.0
    negative = 0.0
    matched_tags: list[dict[str, Any]] = []
    avoided_tags: list[dict[str, Any]] = []
    for tag in track_tags:
        if _tag_matches(tag["tag"], prefer):
            contribution = 0.15 + 0.85 * float(tag["score"])
            positive += contribution
            matched_tags.append(tag)
        if _tag_matches(tag["tag"], avoid):
            contribution = 0.15 + 0.85 * float(tag["score"])
            negative += contribution
            avoided_tags.append(tag)

    return max(0.0, min(1.0, positive - negative)), matched_tags, avoided_tags


def _tag_matches(tag: str, concepts: list[str]) -> bool:
    tag_terms = set(normalize_tag_terms(tag))
    tag_text = " ".join(tag_terms)
    for concept in concepts:
        concept_terms = set(normalize_tag_terms(concept))
        if not concept_terms:
            continue
        concept_text = " ".join(concept_terms)
        if concept_terms.issubset(tag_terms) or concept_text in tag_text:
            return True
    return False


def _feature_score(row: sqlite3.Row, intent: SearchIntent) -> tuple[float, list[str]]:
    if not intent.feature_ranges:
        return 0.5, []

    scores: list[float] = []
    reasons: list[str] = []
    for field_name, feature_range in intent.feature_ranges.items():
        column = "bpm" if field_name == "tempo_bpm" else field_name
        value = row[column]
        softness = 30.0 if field_name == "tempo_bpm" else 0.20
        score = range_score(value, feature_range.low, feature_range.high, softness=softness)
        scores.append(score)
        if value is not None:
            reasons.append(
                f"{field_name} {float(value):.2f} scored {score:.2f} "
                f"for range [{feature_range.low:.2f}, {feature_range.high:.2f}]"
            )
    return sum(scores) / len(scores), reasons


def range_score(value: Any, low: float, high: float, *, softness: float = 0.15) -> float:
    """Score a value against a soft target range."""
    if value is None:
        return 0.4
    numeric = float(value)
    if low <= numeric <= high:
        return 1.0
    distance = min(abs(numeric - low), abs(numeric - high))
    return max(0.0, 1.0 - distance / softness)


def _profile_term_score(profile_text: str, query_terms: list[str]) -> float:
    if not profile_text or not query_terms:
        return 0.0
    normalized = profile_text.lower()
    matched = sum(1 for term in query_terms if re.search(rf"\b{re.escape(term)}\b", normalized))
    return matched / len(query_terms)


def _feedback_scores(
    conn: sqlite3.Connection,
    query: str,
    *,
    include_missing: bool,
) -> dict[str, float]:
    """Return small query-aware feedback boosts/penalties in the range [-1, 1]."""
    missing_clause = "" if include_missing else "AND t.missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT
            f.track_id,
            AVG(
                CASE
                    WHEN LOWER(COALESCE(se.query, '')) = LOWER(?) THEN f.rating
                    ELSE f.rating * 0.25
                END
            ) AS feedback_score
        FROM feedback f
        JOIN tracks t ON t.id = f.track_id
        LEFT JOIN search_events se ON se.id = f.search_event_id
        WHERE 1 = 1
          {missing_clause}
        GROUP BY f.track_id
        """,
        (query,),
    ).fetchall()
    return {
        row["track_id"]: max(-1.0, min(1.0, float(row["feedback_score"] or 0.0)))
        for row in rows
    }


def _weights(*, use_semantic: bool, use_feedback: bool) -> dict[str, float]:
    if use_semantic:
        weights = {"semantic": 0.30, "tags": 0.30, "features": 0.25, "text": 0.15}
    else:
        weights = {"semantic": 0.0, "tags": 0.40, "features": 0.35, "text": 0.25}
    if use_feedback:
        weights["features"] = max(0.0, weights["features"] - 0.03)
        weights["text"] = max(0.0, weights["text"] - 0.02)
        weights["feedback"] = 0.05
    else:
        weights["feedback"] = 0.0
    return weights


def _apply_diversity(results: list[SearchResult], intent: SearchIntent) -> list[SearchResult]:
    max_per_artist = intent.diversity.get("max_tracks_per_artist", 2)
    artist_counts: dict[str, int] = {}
    selected: list[SearchResult] = []
    for result in results:
        artist_key = result.artist.strip().lower() if result.artist else f"track:{result.track_id}"
        if artist_counts.get(artist_key, 0) >= max_per_artist:
            continue
        artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
        selected.append(result)
    return selected
