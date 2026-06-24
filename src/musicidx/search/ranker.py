"""Hybrid local search and ranking."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from musicidx.analyzer.embeddings import EmbeddingError, search_semantic
from musicidx.search.explain import build_explanation
from musicidx.search.feedback import latest_feedback_for_query
from musicidx.search.intent import (
    IntentHints,
    SearchIntent,
    normalize_tag_terms,
    normalize_terms,
    parse_intent_dynamic,
)

MIN_RANKING_TAG_SCORE = 0.20
MIN_RESULT_SCORE = 0.05
WEAK_TOP_RESULT_SCORE = 0.12
SEMANTIC_FLOOR = 0.15
SEMANTIC_CEILING = 0.75
GENERIC_EXPLANATION_TAGS = {
    "background_friendly",
    "focus_friendly",
    "no_vocals_background",
}

METADATA_STOP_WORDS = {
    "a",
    "an",
    "am",
    "and",
    "background",
    "for",
    "give",
    "i",
    "im",
    "in",
    "m",
    "me",
    "music",
    "please",
    "recommend",
    "show",
    "some",
    "song",
    "songs",
    "track",
    "tracks",
    "with",
}


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
    context_by_track = _load_context_fit(conn, include_missing=include_missing)
    fts_scores = _fts_scores(conn, query, include_missing=include_missing)
    semantic_scores, semantic_error = _semantic_scores(
        conn,
        intent,
        include_missing=include_missing,
    )
    feedback_scores = _feedback_scores(conn, query, include_missing=include_missing)

    weights = _weights(
        intent,
        use_semantic=bool(semantic_scores),
        use_feedback=bool(feedback_scores),
        use_context=bool(context_by_track),
    )
    scored_results: list[SearchResult] = []
    for row in track_rows:
        track_id = row["track_id"]
        track_tags = tags_by_track.get(track_id, [])
        breakdown = _score_track(
            row,
            track_tags,
            context_by_track.get(track_id, {}),
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

    artist_focused = any(_has_strong_artist_match(result) for result in scored_results)
    ranked = sorted(
        scored_results,
        key=lambda result: (
            _has_strong_artist_match(result) if artist_focused else False,
            result.score,
        ),
        reverse=True,
    )
    filtered = _filter_weak_results(ranked, intent)
    sorted_results = _apply_explicit_sort(filtered, intent)
    deduplicated = _suppress_near_duplicates(sorted_results)
    diversified = deduplicated if intent.sort_by else _apply_diversity(deduplicated, intent)
    limited_results = _with_saved_feedback(
        conn,
        diversified[: intent.limit],
        query=query,
    )
    display_results = _with_display_scores(limited_results)
    top_raw_score = _top_raw_score(limited_results)
    diagnostics = {
        "candidate_count": len(track_rows),
        "scored_candidate_count": len(scored_results),
        "filtered_candidate_count": len(filtered),
        "minimum_result_score": MIN_RESULT_SCORE,
        "minimum_ranking_tag_score": MIN_RANKING_TAG_SCORE,
        "sort_by": [sort_spec.as_dict() for sort_spec in intent.sort_by],
        "fts_candidate_count": len(fts_scores),
        "semantic_candidate_count": len(semantic_scores),
        "feedback_candidate_count": len(feedback_scores),
        "context_candidate_count": len(context_by_track),
        "semantic_error": semantic_error,
        "weights": weights,
        "score_normalization": "none",
        "score_calibration": "weighted_components_divided_by_active_weight_budget",
        "top_raw_score": top_raw_score,
        "weak_top_result_score": WEAK_TOP_RESULT_SCORE,
        "score_warnings": _score_warnings(top_raw_score, limited_results),
        "duplicate_suppressed_count": max(0, len(sorted_results) - len(deduplicated)),
    }
    return SearchResponse(
        query=query,
        intent=intent,
        results=display_results,
        diagnostics=diagnostics,
    )


def _with_saved_feedback(
    conn: sqlite3.Connection,
    results: list[SearchResult],
    *,
    query: str,
) -> list[SearchResult]:
    ratings = latest_feedback_for_query(
        conn,
        query=query,
        track_ids=[result.track_id for result in results],
    )
    if not ratings:
        return results
    output: list[SearchResult] = []
    for result in results:
        breakdown = dict(result.breakdown)
        if result.track_id in ratings:
            breakdown["saved_feedback_rating"] = ratings[result.track_id]
        output.append(
            SearchResult(
                track_id=result.track_id,
                path=result.path,
                title=result.title,
                artist=result.artist,
                album=result.album,
                genre=result.genre,
                score=result.score,
                breakdown=breakdown,
                explanation=result.explanation,
            )
        )
    return output


def _with_display_scores(results: list[SearchResult]) -> list[SearchResult]:
    """Attach score annotations without re-scaling the externally visible score."""
    output: list[SearchResult] = []
    for result in results:
        raw_score = _raw_score(result)
        breakdown = dict(result.breakdown)
        breakdown["raw_score"] = raw_score
        breakdown["display_score"] = raw_score
        breakdown["score_normalization"] = "none"
        output.append(
            SearchResult(
                track_id=result.track_id,
                path=result.path,
                title=result.title,
                artist=result.artist,
                album=result.album,
                genre=result.genre,
                score=round(raw_score, 6),
                breakdown=breakdown,
                explanation=result.explanation,
            )
        )
    return output


def _top_raw_score(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    return max(_raw_score(result) for result in results)


def _raw_score(result: SearchResult) -> float:
    return float(result.breakdown.get("final_score", result.score) or 0.0)


def _score_warnings(top_raw_score: float, results: list[SearchResult]) -> list[str]:
    warnings: list[str] = []
    if results and top_raw_score < WEAK_TOP_RESULT_SCORE:
        warnings.append("weak_top_score")
    if results and all(
        result.breakdown.get("evidence", {}).get("semantic_only") for result in results
    ):
        warnings.append("semantic_only_results")
    return warnings


def _suppress_near_duplicates(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    output: list[SearchResult] = []
    for result in results:
        keys = _duplicate_keys(result)
        if keys and any(key in seen for key in keys):
            continue
        seen.update(keys)
        output.append(result)
    return output


def _duplicate_keys(result: SearchResult) -> list[str]:
    identity = result.breakdown.get("identity") or {}
    keys: list[str] = []
    content_hash = str(identity.get("content_hash") or "").strip()
    if content_hash:
        keys.append(f"content:{content_hash}")
    chromaprint = str(identity.get("chromaprint") or "").strip()
    if chromaprint:
        keys.append(f"chromaprint:{chromaprint}")
    artist_title_norm = str(identity.get("artist_title_norm") or "").strip()
    if artist_title_norm:
        keys.append(f"artist_title:{artist_title_norm}")

    artist = _normalize_identity_text(result.artist or "")
    title = _normalize_identity_text(result.title or "")
    if artist and title:
        keys.append(f"metadata:{artist}|{title}")

    stem = _normalize_identity_text(Path(urllib.parse.unquote(result.path)).stem)
    if stem:
        keys.append(f"path:{stem}")
    return _unique_strings(keys)


def _normalize_identity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", urllib.parse.unquote(value)).casefold()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", normalized)
    normalized = re.sub(
        r"\b(official|video|audio|original|mix|remaster(?:ed)?|extended)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


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
            t.album_artist,
            t.genre,
            t.content_hash,
            t.chromaprint,
            t.duration_sec,
            t.artist_title_norm,
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


def _load_context_fit(
    conn: sqlite3.Connection,
    *,
    include_missing: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    missing_clause = "" if include_missing else "AND tr.missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT cf.track_id, cf.context, cf.score, cf.confidence, cf.evidence_json
        FROM track_context_fit cf
        JOIN tracks tr ON tr.id = cf.track_id
        WHERE 1 = 1
          {missing_clause}
        ORDER BY cf.track_id, cf.score DESC
        """
    ).fetchall()
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(row["track_id"], {})[row["context"]] = {
            "context": row["context"],
            "score": float(row["score"] or 0.0),
            "confidence": float(row["confidence"] or 0.0),
            "evidence": _parse_json(row["evidence_json"], default={}),
        }
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
    return {result.track_id: _semantic_relevance(result.score) for result in results}, None


def _score_track(
    row: sqlite3.Row,
    track_tags: list[dict[str, Any]],
    context_fit: dict[str, dict[str, Any]],
    intent: SearchIntent,
    *,
    fts_score: float,
    semantic_score: float,
    feedback_score: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    tag_score, matched_tags, avoided_tags = _tag_score(track_tags, intent)
    feature_score, feature_reasons = _feature_score(row, intent)
    context_score, matched_contexts = _context_score(context_fit, intent)
    metadata_score, metadata_matches = _metadata_score(row, intent)
    text_score = max(fts_score, _profile_term_score(row["profile_text"] or "", intent.query_terms))

    weighted_score = (
        weights["semantic"] * semantic_score
        + weights["metadata"] * metadata_score
        + weights["tags"] * tag_score
        + weights.get("context", 0.0) * context_score
        + weights["features"] * feature_score
        + weights["text"] * text_score
        + weights.get("feedback", 0.0) * feedback_score
    )
    final_score = _calibrated_score(weighted_score, weights)
    evidence = _evidence_summary(
        semantic_score=semantic_score,
        metadata_score=metadata_score,
        tag_score=tag_score,
        context_score=context_score,
        feature_score=feature_score,
        text_score=text_score,
        feedback_score=feedback_score,
    )
    confidence, confidence_warnings = _confidence_label(final_score, evidence)

    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "tag_score": tag_score,
        "metadata_score": metadata_score,
        "feature_score": feature_score,
        "context_score": context_score,
        "text_score": text_score,
        "feedback_score": feedback_score,
        "confidence": confidence,
        "confidence_warnings": confidence_warnings,
        "evidence": evidence,
        "matched_tags": matched_tags,
        "avoided_tags": avoided_tags,
        "metadata_matches": metadata_matches,
        "feature_reasons": feature_reasons,
        "matched_contexts": matched_contexts,
        "identity": {
            "content_hash": row["content_hash"],
            "chromaprint": row["chromaprint"],
            "duration_sec": row["duration_sec"],
            "artist_title_norm": row["artist_title_norm"],
        },
        "sort_by": [sort_spec.as_dict() for sort_spec in intent.sort_by],
        "sort_values": _sort_values(row),
    }


def _calibrated_score(weighted_score: float, weights: dict[str, float]) -> float:
    normalizer = sum(max(0.0, float(value or 0.0)) for value in weights.values())
    if normalizer <= 0.0:
        return 0.0
    return max(0.0, min(1.0, weighted_score / normalizer))


def _evidence_summary(
    *,
    semantic_score: float,
    metadata_score: float,
    tag_score: float,
    context_score: float,
    feature_score: float,
    text_score: float,
    feedback_score: float,
) -> dict[str, Any]:
    signals = {
        "semantic": semantic_score > 0.0,
        "metadata": metadata_score > 0.0,
        "tags": tag_score > 0.0,
        "context": context_score > 0.0,
        "features": feature_score > 0.0,
        "text": text_score > 0.0,
        "feedback": feedback_score != 0.0,
    }
    non_semantic_signals = [
        name for name, present in signals.items() if present and name not in {"semantic"}
    ]
    semantic_only = signals["semantic"] and not non_semantic_signals
    if semantic_only:
        category = "semantic_only"
    elif signals["metadata"]:
        category = "metadata"
    elif signals["tags"] or signals["context"] or signals["text"]:
        category = "hybrid"
    elif signals["features"]:
        category = "feature_only"
    elif signals["semantic"]:
        category = "semantic_plus_weak"
    else:
        category = "weak"
    return {
        "category": category,
        "signals": signals,
        "semantic_only": semantic_only,
        "non_semantic_signal_count": len(non_semantic_signals),
    }


def _confidence_label(final_score: float, evidence: dict[str, Any]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if evidence.get("semantic_only"):
        warnings.append("semantic_only")
    if final_score < WEAK_TOP_RESULT_SCORE:
        warnings.append("weak_score")

    non_semantic_count = int(evidence.get("non_semantic_signal_count") or 0)
    category = str(evidence.get("category") or "weak")
    if final_score >= 0.45 and non_semantic_count >= 2:
        confidence = "high"
    elif final_score >= 0.25 and category in {"metadata", "hybrid"}:
        confidence = "medium"
    elif final_score >= 0.35 and non_semantic_count >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    return confidence, warnings


def _sort_values(row: sqlite3.Row) -> dict[str, float | None]:
    return {
        "tempo_bpm": _safe_row_float(row, "bpm"),
        "energy": _safe_row_float(row, "energy"),
        "danceability": _safe_row_float(row, "danceability"),
        "aggression": _safe_row_float(row, "aggression"),
        "brightness": _safe_row_float(row, "brightness"),
    }


def _safe_row_float(row: sqlite3.Row, key: str) -> float | None:
    value = row[key]
    return None if value is None else float(value)


def _context_score(
    context_fit: dict[str, dict[str, Any]],
    intent: SearchIntent,
) -> tuple[float, list[dict[str, Any]]]:
    desired_contexts = _desired_context_fit_names(intent)
    if not desired_contexts:
        return 0.0, []
    matches: list[dict[str, Any]] = []
    for context in desired_contexts:
        fit = context_fit.get(context)
        if not fit:
            continue
        score = float(fit.get("score") or 0.0)
        confidence = float(fit.get("confidence") or 0.0)
        adjusted = score * (0.65 + 0.35 * confidence)
        matches.append(
            {
                "context": context,
                "score": round(score, 6),
                "confidence": round(confidence, 6),
                "adjusted_score": round(adjusted, 6),
                "evidence": fit.get("evidence") or {},
            }
        )
    if not matches:
        return 0.0, []
    matches.sort(key=lambda item: float(item["adjusted_score"]), reverse=True)
    top_scores = [float(item["adjusted_score"]) for item in matches[:3]]
    return max(top_scores), matches[:5]


def _desired_context_fit_names(intent: SearchIntent) -> list[str]:
    mapping = {
        "bar": ["lounge_bar", "warm_lounge", "background"],
        "chill": ["background", "lounge_bar", "focus"],
        "background": ["background", "no_vocals_background"],
        "cooking": ["cooking", "dinner", "background"],
        "dinner": ["dinner", "warm_lounge", "lounge_bar", "background"],
        "focus": ["focus", "background", "no_vocals_background"],
        "ambient": ["dark_ambient", "background", "focus"],
        "dark": ["dark_ambient"],
        "no_vocals_background": ["no_vocals_background", "background", "focus"],
        "party": ["party", "club"],
        "wedding": ["party", "dinner", "warm_lounge", "lounge_bar"],
        "workout": ["workout", "club"],
        "shower": ["party", "club", "driving"],
        "driving": ["driving", "club"],
        "sleep": ["background", "dark_ambient"],
    }
    desired: list[str] = []
    for context in intent.contexts:
        desired.extend(mapping.get(context, [context]))
    query_text = intent.query.casefold()
    if "no vocal" in query_text or "without vocal" in query_text or "instrumental" in query_text:
        desired.extend(["no_vocals_background", "background"])
    return _unique_strings(desired)


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _metadata_score(row: sqlite3.Row, intent: SearchIntent) -> tuple[float, list[dict[str, Any]]]:
    query_terms = {term for term in intent.query_terms if term not in METADATA_STOP_WORDS}
    if not query_terms:
        return 0.0, []
    query_text = " ".join(query_terms)
    fields = [
        ("artist", row["artist"], 1.0),
        ("album_artist", row["album_artist"], 0.95),
        ("title", row["title"], 0.75),
        ("album", row["album"], 0.55),
        ("genre", row["genre"], 0.45),
    ]
    matches: list[dict[str, Any]] = []
    for field_name, value, field_weight in fields:
        score = _metadata_field_score(
            str(value) if value else None,
            query_terms=query_terms,
            query_text=query_text,
            field_name=field_name,
        )
        if score <= 0.0:
            continue
        matches.append(
            {
                "field": field_name,
                "value": value,
                "score": round(min(1.0, score * field_weight), 6),
            }
        )
    matches.sort(key=lambda item: float(item["score"]), reverse=True)
    return (float(matches[0]["score"]) if matches else 0.0), matches[:5]


def _metadata_field_score(
    value: str | None,
    *,
    query_terms: set[str],
    query_text: str,
    field_name: str,
) -> float:
    if not value:
        return 0.0
    field_terms = set(normalize_tag_terms(value))
    if not field_terms:
        return 0.0
    field_text = " ".join(field_terms)
    if re.search(rf"\b{re.escape(field_text)}\b", query_text):
        return 1.0
    overlap = field_terms.intersection(query_terms)
    if len(field_terms) > 1 and field_terms.issubset(query_terms):
        return 1.0
    if not overlap:
        return 0.0
    if field_name in {"artist", "album_artist"}:
        return 0.85 if any(len(term) >= 5 for term in overlap) else 0.55
    if field_name == "title":
        return 0.55 if len(field_terms) == 1 and any(len(term) >= 5 for term in overlap) else 0.35
    return 0.30


def _tag_score(
    track_tags: list[dict[str, Any]],
    intent: SearchIntent,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    prefer = intent.prefer_tags + intent.prefer_tag_concepts
    avoid = intent.avoid_tags + intent.avoid_tag_concepts
    if not prefer and not avoid:
        return 0.0, [], []

    positive = 0.0
    negative = 0.0
    matched_tags: list[dict[str, Any]] = []
    avoided_tags: list[dict[str, Any]] = []
    for tag in track_tags:
        score = float(tag["score"])
        if score < MIN_RANKING_TAG_SCORE:
            continue
        contribution = (score - MIN_RANKING_TAG_SCORE) / (1.0 - MIN_RANKING_TAG_SCORE)
        if _tag_matches(tag["tag"], prefer):
            positive += contribution
            if _should_explain_matched_tag(tag):
                matched_tags.append(tag)
        if _tag_matches_avoid(tag["tag"], avoid):
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
        if concept_terms.issubset(tag_terms):
            return True
        if any(len(term) < 3 for term in concept_terms):
            continue
        if concept_text in tag_text:
            return True
    return False


def _should_explain_matched_tag(tag: dict[str, Any]) -> bool:
    tag_name = str(tag.get("tag") or "")
    source = str(tag.get("source") or "")
    score = float(tag.get("score") or 0.0)
    if source == "derived:features" and tag_name in GENERIC_EXPLANATION_TAGS:
        return False
    if source == "derived:features" and tag_name.endswith("_friendly") and score < 0.85:
        return False
    return True


def _tag_matches_avoid(tag: str, concepts: list[str]) -> bool:
    tag_terms = set(normalize_tag_terms(tag))
    if _is_negated_positive_tag(tag_terms):
        return False
    return _tag_matches(tag, concepts)


def _is_negated_positive_tag(tag_terms: set[str]) -> bool:
    if tag_terms.intersection({"not", "no", "without", "low"}):
        return True
    if "instrumental" in tag_terms:
        return True
    return False


def _feature_score(row: sqlite3.Row, intent: SearchIntent) -> tuple[float, list[str]]:
    if not intent.feature_ranges:
        return 0.0, []

    scores: list[float] = []
    reasons: list[str] = []
    for field_name, feature_range in intent.feature_ranges.items():
        column = "bpm" if field_name == "tempo_bpm" else field_name
        value = row[column]
        softness = 30.0 if field_name == "tempo_bpm" else 0.20
        score = range_score(value, feature_range.low, feature_range.high, softness=softness)
        scores.append(score)
        if value is not None and _should_explain_feature(field_name, feature_range, score):
            reasons.append(_feature_reason(field_name, float(value), feature_range, score))
    return sum(scores) / len(scores), reasons


def _should_explain_feature(field_name: str, feature_range: Any, score: float) -> bool:
    width = float(feature_range.high) - float(feature_range.low)
    broad_width = 80.0 if field_name == "tempo_bpm" else 0.55
    if width >= broad_width and score >= 0.95:
        return False
    return score >= 0.25


def _feature_reason(field_name: str, value: float, feature_range: Any, score: float) -> str:
    labels = {
        "tempo_bpm": "BPM",
        "energy": "energy",
        "danceability": "danceability",
        "aggression": "aggression",
        "brightness": "brightness",
    }
    target = _feature_target_label(str(feature_range.source))
    label = labels.get(field_name, field_name)
    if field_name == "tempo_bpm":
        value_text = f"{value:.0f}"
    else:
        value_text = f"{value:.2f}"
    if score >= 0.75:
        return f"{label} {value_text} fits {target} target"
    return f"{label} {value_text} partially fits {target} target ({score:.2f})"


def _feature_target_label(source: str) -> str:
    for prefix in ["library_percentile:", "fallback:"]:
        source = source.replace(prefix, "")
    source = source.replace("union:", "").replace("+", "/")
    parts: list[str] = []
    for part in source.split("/"):
        cleaned = part.strip()
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    return "/".join(parts).replace("_", " ") or "requested"


def range_score(value: Any, low: float, high: float, *, softness: float = 0.15) -> float:
    """Score a value against a soft target range."""
    if value is None:
        return 0.0
    numeric = float(value)
    if low <= numeric <= high:
        return 1.0
    distance = min(abs(numeric - low), abs(numeric - high))
    return max(0.0, 1.0 - distance / softness)


def _profile_term_score(profile_text: str, query_terms: list[str]) -> float:
    if not profile_text or not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[a-z0-9]+", profile_text.lower()))
    matched = sum(1 for term in query_terms if term in text_terms)
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


def _parse_json(value: str | None, *, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _semantic_relevance(cosine_score: float) -> float:
    if cosine_score <= SEMANTIC_FLOOR:
        return 0.0
    if cosine_score >= SEMANTIC_CEILING:
        return 1.0
    return (cosine_score - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR)


def _filter_weak_results(results: list[SearchResult], intent: SearchIntent) -> list[SearchResult]:
    if intent.sort_by:
        return [result for result in results if _has_sort_value(result, intent)]
    meaningful = [result for result in results if result.score >= MIN_RESULT_SCORE]
    if meaningful and _has_subjective_intent(intent):
        evidence_results = [result for result in meaningful if _has_query_evidence(result)]
        if evidence_results:
            minimum_count = min(intent.limit, 5)
            if len(evidence_results) >= minimum_count or not _allow_feature_backfill(intent):
                return evidence_results
            evidence_ids = {result.track_id for result in evidence_results}
            feature_backfill = [
                result
                for result in meaningful
                if result.track_id not in evidence_ids and not _has_avoid_match(result)
            ]
            return [*evidence_results, *feature_backfill[: minimum_count - len(evidence_results)]]
    if meaningful:
        return meaningful
    return [result for result in results if result.score > 0.0]


def _has_subjective_intent(intent: SearchIntent) -> bool:
    return bool(
        intent.contexts
        or intent.prefer_tag_concepts
        or intent.prefer_tags
        or intent.avoid_tag_concepts
        or intent.avoid_tags
    )


def _allow_feature_backfill(intent: SearchIntent) -> bool:
    return bool({"party", "wedding", "workout", "shower"}.intersection(intent.contexts))


def _has_query_evidence(result: SearchResult) -> bool:
    breakdown = result.breakdown
    return bool(
        float(breakdown.get("tag_score") or 0.0) > 0.0
        or float(breakdown.get("metadata_score") or 0.0) > 0.0
        or float(breakdown.get("context_score") or 0.0) > 0.0
        or float(breakdown.get("text_score") or 0.0) > 0.0
        or float(breakdown.get("semantic_score") or 0.0) > 0.0
        or float(breakdown.get("feedback_score") or 0.0) > 0.0
    )


def _has_avoid_match(result: SearchResult) -> bool:
    return bool(result.breakdown.get("avoided_tags"))


def _has_strong_artist_match(result: SearchResult) -> bool:
    for match in result.breakdown.get("metadata_matches") or []:
        if not isinstance(match, dict):
            continue
        is_artist_field = match.get("field") in {"artist", "album_artist"}
        if is_artist_field and float(match.get("score") or 0.0) >= 0.8:
            return True
    return False


def _has_sort_value(result: SearchResult, intent: SearchIntent) -> bool:
    sort_values = result.breakdown.get("sort_values") or {}
    return any(sort_values.get(sort_spec.field) is not None for sort_spec in intent.sort_by)


def _apply_explicit_sort(results: list[SearchResult], intent: SearchIntent) -> list[SearchResult]:
    if not intent.sort_by:
        return results

    def key(result: SearchResult) -> tuple[Any, ...]:
        sort_values = result.breakdown.get("sort_values") or {}
        parts: list[Any] = []
        for sort_spec in intent.sort_by:
            value = sort_values.get(sort_spec.field)
            parts.append(value is None)
            numeric = float(value) if value is not None else 0.0
            parts.append(-numeric if sort_spec.direction == "desc" else numeric)
        parts.append(-result.score)
        return tuple(parts)

    return sorted(results, key=key)


def _weights(
    intent: SearchIntent,
    *,
    use_semantic: bool,
    use_feedback: bool,
    use_context: bool,
) -> dict[str, float]:
    has_mood_or_feature_intent = bool(intent.contexts or intent.feature_ranges)
    if use_semantic:
        if has_mood_or_feature_intent:
            weights = {
                "semantic": 0.42,
                "metadata": 0.55,
                "context": 0.26 if use_context else 0.0,
                "tags": 0.24,
                "features": 0.18,
                "text": 0.08,
            }
        else:
            weights = {
                "semantic": 0.52,
                "metadata": 0.55,
                "context": 0.0,
                "tags": 0.22,
                "features": 0.16,
                "text": 0.10,
            }
    elif has_mood_or_feature_intent:
        weights = {
            "semantic": 0.0,
            "metadata": 0.55,
            "context": 0.32 if use_context else 0.0,
            "tags": 0.38,
            "features": 0.34,
            "text": 0.22,
        }
    else:
        weights = {
            "semantic": 0.0,
            "metadata": 0.55,
            "context": 0.0,
            "tags": 0.34,
            "features": 0.18,
            "text": 0.48,
        }
    if use_feedback:
        weights["features"] = max(0.0, weights["features"] - 0.04)
        weights["text"] = max(0.0, weights["text"] - 0.03)
        weights["feedback"] = 0.18
    else:
        weights["feedback"] = 0.0
    return weights


def _apply_diversity(results: list[SearchResult], intent: SearchIntent) -> list[SearchResult]:
    if any(_has_strong_artist_match(result) for result in results):
        return results
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
