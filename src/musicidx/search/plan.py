"""Inspectable search planning helpers.

The plan is diagnostic: it explains how the current parser interprets a query before
ranking. Later search-optimisation phases can make the ranker execute this plan more
strictly.
"""

from __future__ import annotations

import re
from typing import Any

from musicidx.search.intent import SearchIntent, negated_query_terms
from musicidx.search.taxonomy import load_search_taxonomy

FEATURE_ALIASES = {
    "aggressive",
    "aggression",
    "bpm",
    "bright",
    "brightness",
    "dance",
    "danceability",
    "danceable",
    "danciest",
    "dark",
    "darkest",
    "energetic",
    "energy",
    "fast",
    "faster",
    "fastest",
    "hardest",
    "high",
    "highest",
    "least",
    "low",
    "lowest",
    "max",
    "maximum",
    "min",
    "minimum",
    "most",
    "quickest",
    "slow",
    "slower",
    "slowest",
    "softest",
    "tempo",
    "uptempo",
}

DIRECTIVE_TERMS = FEATURE_ALIASES | {
    "asc",
    "ascending",
    "by",
    "desc",
    "descending",
    "order",
    "sort",
}

OCCASION_CONTEXTS = set(load_search_taxonomy().occasion_contexts)

FEATURE_LABELS = {
    "aggression": "aggression",
    "brightness": "brightness",
    "danceability": "danceability",
    "energy": "energy",
    "tempo_bpm": "perceived BPM",
}


def build_search_plan(intent: SearchIntent) -> dict[str, Any]:
    """Build a ranking-oriented diagnostic plan from parsed intent."""
    mode = classify_search_mode(intent)
    content_terms = content_query_terms(intent)
    directive_terms = directive_query_terms(intent)
    semantic_role = semantic_role_for_intent(intent, mode=mode, content_terms=content_terms)
    feature_ranges = {
        field_name: feature_range.as_dict()
        for field_name, feature_range in intent.feature_ranges.items()
    }
    return {
        "schema_version": 1,
        "query": intent.query,
        "parser": intent.parser,
        "mode": mode,
        "terms": {
            "content": content_terms,
            "directives": directive_terms,
            "negated": negated_query_terms(intent.query),
            "raw_query_terms": intent.query_terms,
        },
        "entities": {"artists": [], "titles": [], "albums": []},
        "must": _must_clauses(intent, mode=mode, content_terms=content_terms),
        "should": _should_clauses(intent, mode=mode),
        "avoid": _avoid_clauses(intent),
        "hard_filters": _hard_filters(intent),
        "sort": [sort_spec.as_dict() for sort_spec in intent.sort_by],
        "semantic": {
            "enabled": intent.use_semantic,
            "model": intent.semantic_model,
            "role": semantic_role,
        },
        "llm": _llm_plan(intent, mode=mode),
        "diagnostics": {
            "candidate_source_plan": _candidate_source_plan(intent, mode=mode),
            "notes": _plan_notes(intent, mode=mode, semantic_role=semantic_role),
            "contexts": intent.contexts,
            "prefer_tag_concepts": intent.prefer_tag_concepts,
            "avoid_tag_concepts": intent.avoid_tag_concepts,
            "prefer_tags": intent.prefer_tags,
            "avoid_tags": intent.avoid_tags,
            "feature_ranges": feature_ranges,
        },
        "warnings": _plan_warnings(intent),
    }


def classify_search_mode(intent: SearchIntent) -> str:
    """Classify the query for diagnostics and future mode-specific ranking."""
    if intent.sort_by:
        return "feature_sort"
    if any(context in OCCASION_CONTEXTS for context in intent.contexts):
        return "occasion"
    if _contextual_query_should_use_context_mode(intent):
        return "context_vibe"
    if _looks_like_tag_or_genre_query(intent):
        return "tag_genre"
    if intent.contexts:
        return "context_vibe"
    if intent.feature_ranges:
        return "feature_filter"
    if content_query_terms(intent):
        return "metadata_exact"
    if intent.use_semantic:
        return "fallback_semantic"
    return "metadata_exact"


def content_query_terms(intent: SearchIntent) -> list[str]:
    """Return query terms that look like content, not feature/sort directives."""
    output: list[str] = []
    for term in intent.query_terms:
        normalized = term.strip().lower()
        if not normalized or normalized in DIRECTIVE_TERMS:
            continue
        output.append(normalized)
    return _unique(output)


def directive_query_terms(intent: SearchIntent) -> list[str]:
    """Return raw query tokens that look like feature/sort directives."""
    tokens = re.findall(r"[a-z0-9]+", intent.query.lower())
    directives = [token for token in tokens if token in DIRECTIVE_TERMS]
    for sort_spec in intent.sort_by:
        directives.extend([sort_spec.field, sort_spec.direction])
    return _unique(directives)


def semantic_role_for_intent(
    intent: SearchIntent,
    *,
    mode: str | None = None,
    content_terms: list[str] | None = None,
) -> str:
    """Describe how semantic similarity should be interpreted for this query."""
    if not intent.use_semantic:
        return "disabled"
    resolved_mode = mode or classify_search_mode(intent)
    resolved_content_terms = (
        content_terms if content_terms is not None else content_query_terms(intent)
    )
    if resolved_mode == "feature_sort":
        return "tie_breaker" if resolved_content_terms else "supporting_tie_breaker"
    if resolved_mode == "metadata_exact":
        return "fallback"
    if resolved_mode in {"tag_genre", "feature_filter"}:
        return "supporting"
    if resolved_mode in {"context_vibe", "occasion", "fallback_semantic"}:
        return "primary_evidence"
    return "supporting"


def _contextual_query_should_use_context_mode(intent: SearchIntent) -> bool:
    if not intent.contexts:
        return False
    if not _looks_like_tag_or_genre_query(intent):
        return True
    if negated_query_terms(intent.query):
        return True
    content_terms = content_query_terms(intent)
    if len(content_terms) != 1:
        return True
    return not _is_direct_single_tag_or_genre_query(intent, content_terms[0])


def _is_direct_single_tag_or_genre_query(intent: SearchIntent, term: str) -> bool:
    significant_terms = [
        token
        for token in re.findall(r"[a-z0-9]+", intent.query.lower())
        if token not in DIRECTIVE_TERMS
        and token not in {"music", "song", "songs", "track", "tracks"}
    ]
    return significant_terms == [term]


def _looks_like_tag_or_genre_query(intent: SearchIntent) -> bool:
    content_terms = content_query_terms(intent)
    if not content_terms:
        return False
    if not intent.prefer_tags:
        return False
    content_text = " ".join(content_terms)
    normalized_tags = [
        " ".join(re.findall(r"[a-z0-9]+", tag.lower().replace("---", " ").replace("_", " ")))
        for tag in intent.prefer_tags
    ]
    return any(
        term in tag_text or content_text in tag_text
        for term in content_terms
        for tag_text in normalized_tags
    )


def _must_clauses(
    intent: SearchIntent,
    *,
    mode: str,
    content_terms: list[str],
) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    if intent.sort_by:
        for sort_spec in intent.sort_by:
            clauses.append(
                {
                    "type": "feature_present",
                    "field": sort_spec.field,
                    "label": FEATURE_LABELS.get(sort_spec.field, sort_spec.field),
                    "source": sort_spec.source or "local_parser",
                }
            )
    if mode == "feature_sort" and content_terms:
        clauses.extend(_content_clauses(content_terms, kind="tag_or_text"))
    elif mode == "tag_genre":
        clauses.extend(_content_clauses(content_terms, kind="tag_or_text"))
    elif mode == "metadata_exact":
        clauses.extend(_content_clauses(content_terms, kind="metadata_or_text", phrase=True))
    return clauses


def _should_clauses(intent: SearchIntent, *, mode: str) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for context in intent.contexts:
        clauses.append({"type": "context", "concept": context, "source": "local_parser"})
    for concept in intent.prefer_tag_concepts:
        clauses.append({"type": "tag_concept", "concept": concept, "source": "local_parser"})
    for tag in intent.prefer_tags[:20]:
        clauses.append({"type": "library_tag", "tag": tag, "source": "library_profile"})
    for field_name, feature_range in intent.feature_ranges.items():
        clauses.append(
            {
                "type": "feature_range",
                "field": field_name,
                "label": FEATURE_LABELS.get(field_name, field_name),
                "low": feature_range.low,
                "high": feature_range.high,
                "source": feature_range.source,
            }
        )
    if mode in {"context_vibe", "occasion"} and intent.use_semantic:
        clauses.append(
            {"type": "semantic_similarity", "role": "primary_evidence", "source": "embeddings"}
        )
    elif intent.use_semantic:
        clauses.append(
            {"type": "semantic_similarity", "role": "supporting", "source": "embeddings"}
        )
    return _dedupe_dicts(clauses)


def _avoid_clauses(intent: SearchIntent) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for concept in intent.avoid_tag_concepts:
        clauses.append({"type": "tag_concept", "concept": concept, "source": "local_parser"})
    for tag in intent.avoid_tags[:20]:
        clauses.append({"type": "library_tag", "tag": tag, "source": "library_profile"})
    return _dedupe_dicts(clauses)


def _content_clauses(
    terms: list[str],
    *,
    kind: str,
    phrase: bool = False,
) -> list[dict[str, str]]:
    if not terms:
        return []
    if phrase and len(terms) > 1:
        return [{"type": kind, "concept": " ".join(terms), "source": "local_parser"}]
    return [{"type": kind, "concept": term, "source": "local_parser"} for term in terms]


def _hard_filters(intent: SearchIntent) -> dict[str, Any]:
    filters: dict[str, Any] = {"missing": bool(intent.include_missing)}
    for sort_spec in intent.sort_by:
        filters.setdefault(sort_spec.field, {})["not_null"] = True
    return filters


def _candidate_source_plan(intent: SearchIntent, *, mode: str) -> list[dict[str, Any]]:
    sources = [
        {
            "source": "metadata_text_fts",
            "enabled": bool(content_query_terms(intent)),
            "role": "primary" if mode == "metadata_exact" else "supporting",
        },
        {
            "source": "tags",
            "enabled": bool(intent.prefer_tags or intent.avoid_tags),
            "role": "primary" if mode == "tag_genre" else "supporting",
        },
        {
            "source": "features",
            "enabled": bool(intent.feature_ranges or intent.sort_by),
            "role": "primary" if mode in {"feature_sort", "feature_filter"} else "supporting",
        },
        {
            "source": "context_fit",
            "enabled": bool(intent.contexts),
            "role": "primary" if mode in {"context_vibe", "occasion"} else "supporting",
        },
        {
            "source": "semantic_profile",
            "enabled": intent.use_semantic,
            "role": semantic_role_for_intent(intent, mode=mode),
        },
        {
            "source": "audio_embedding",
            "enabled": False,
            "role": "not_configured_similarity_only",
        },
        {
            "source": "fingerprint",
            "enabled": False,
            "role": "identity_matching_not_search_ranking",
        },
        {
            "source": "feedback",
            "enabled": True,
            "role": "rerank_adjustment",
        },
    ]
    return sources


def _plan_notes(intent: SearchIntent, *, mode: str, semantic_role: str) -> list[str]:
    notes = [f"mode={mode}", f"semantic_role={semantic_role}"]
    if intent.sort_by:
        notes.append("explicit sort is authoritative; relevance is tie-breaker")
    if intent.llm_rejected_hints:
        notes.append("some or all LLM hints were rejected by local policy")
    return notes


def _llm_plan(intent: SearchIntent, *, mode: str) -> dict[str, Any]:
    policy = intent.llm_policy or {"used": False, "role": "none"}
    ignored_because = policy.get("ignored_because") or intent.llm_error
    return {
        "used": bool(policy.get("used")),
        "role": policy.get("role") or "none",
        "mode": mode,
        "accepted_hints": intent.llm_hints.as_dict() if intent.llm_hints else None,
        "rejected_hints": (
            intent.llm_rejected_hints.as_dict() if intent.llm_rejected_hints else None
        ),
        "accepted_fields": policy.get("accepted_fields") or [],
        "rejected_fields": policy.get("rejected_fields") or [],
        "provider": _llm_provider_from_parser(intent.parser),
        "model": None,
        "error": intent.llm_error,
        "ignored_because": ignored_because,
    }


def _llm_provider_from_parser(parser: str) -> str | None:
    prefix = "dynamic+"
    if parser.startswith(prefix):
        return parser[len(prefix) :] or None
    return None


def _plan_warnings(intent: SearchIntent) -> list[str]:
    warnings: list[str] = []
    if intent.llm_error:
        warnings.append("llm_hints_unavailable_or_ignored")
    if intent.llm_rejected_hints:
        warnings.append("llm_hints_partially_or_fully_rejected")
    if intent.sort_by and not content_query_terms(intent):
        warnings.append("sort_only_query")
    if not intent.use_semantic:
        warnings.append("semantic_model_not_available")
    if not intent.prefer_tags and intent.prefer_tag_concepts:
        warnings.append("no_library_tags_matched_preferred_concepts")
    return warnings


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = repr(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output
