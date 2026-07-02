"""Dynamic, library-aware query intent parsing."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from musicidx.analyzer.embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_KIND
from musicidx.search.taxonomy import load_search_taxonomy
from musicidx.tempo import perceived_tempo_bpm, tempo_descriptors_from_metadata_and_tags

FEATURE_FIELDS = ["energy", "danceability", "aggression", "brightness", "tempo_bpm"]

_SEARCH_TAXONOMY = load_search_taxonomy()

DEFAULT_FEATURE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    field_name: dict(levels)
    for field_name, levels in _SEARCH_TAXONOMY.feature_ranges.items()
}

CONTEXT_PRIORS: dict[str, dict[str, Any]] = {
    name: {
        "keywords": list(entry.keywords),
        "prefer": list(entry.prefer),
        "avoid": list(entry.avoid),
        "features": dict(entry.features),
    }
    for name, entry in _SEARCH_TAXONOMY.contexts.items()
}

QUERY_PRIORS: dict[str, dict[str, Any]] = {
    name: {
        "keywords": list(entry.keywords),
        "prefer": list(entry.prefer),
        "avoid": list(entry.avoid),
        "features": dict(entry.features),
    }
    for name, entry in _SEARCH_TAXONOMY.query_priors.items()
}

QUERY_CONCEPT_STOP_WORDS = set(_SEARCH_TAXONOMY.query_concept_stop_words)


@dataclass(slots=True)
class TagStat:
    tag: str
    count: int
    avg_score: float
    max_score: float
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeatureRange:
    low: float
    high: float
    source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SortSpec:
    field: str
    direction: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LibraryProfile:
    total_tracks: int
    tag_stats: dict[str, TagStat]
    feature_percentiles: dict[str, dict[str, float]]
    embedding_models: list[str]

    def as_dict(self) -> dict[str, Any]:
        top_tags = sorted(
            self.tag_stats.values(),
            key=lambda stat: (stat.count, stat.avg_score),
            reverse=True,
        )[:50]
        return {
            "total_tracks": self.total_tracks,
            "top_tags": [tag.as_dict() for tag in top_tags],
            "feature_percentiles": self.feature_percentiles,
            "embedding_models": self.embedding_models,
        }


@dataclass(slots=True)
class IntentHints:
    """Optional high-level hints from an LLM parser."""

    contexts: list[str] = field(default_factory=list)
    prefer_tag_concepts: list[str] = field(default_factory=list)
    avoid_tag_concepts: list[str] = field(default_factory=list)
    feature_preferences: dict[str, str] = field(default_factory=dict)
    sort_by: list[SortSpec] = field(default_factory=list)
    limit: int | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchIntent:
    query: str
    limit: int
    include_missing: bool
    parser: str
    llm_hints: IntentHints | None
    llm_rejected_hints: IntentHints | None
    llm_policy: dict[str, Any]
    llm_error: str | None
    contexts: list[str]
    query_terms: list[str]
    prefer_tag_concepts: list[str]
    avoid_tag_concepts: list[str]
    prefer_tags: list[str]
    avoid_tags: list[str]
    feature_ranges: dict[str, FeatureRange]
    sort_by: list[SortSpec]
    semantic_model: str | None
    use_semantic: bool
    diversity: dict[str, int]
    library_profile: LibraryProfile

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "limit": self.limit,
            "include_missing": self.include_missing,
            "parser": self.parser,
            "llm_hints": self.llm_hints.as_dict() if self.llm_hints else None,
            "llm_rejected_hints": (
                self.llm_rejected_hints.as_dict() if self.llm_rejected_hints else None
            ),
            "llm_policy": self.llm_policy,
            "llm_error": self.llm_error,
            "contexts": self.contexts,
            "query_terms": self.query_terms,
            "prefer_tag_concepts": self.prefer_tag_concepts,
            "avoid_tag_concepts": self.avoid_tag_concepts,
            "prefer_tags": self.prefer_tags,
            "avoid_tags": self.avoid_tags,
            "feature_ranges": {
                field_name: feature_range.as_dict()
                for field_name, feature_range in self.feature_ranges.items()
            },
            "sort_by": [sort_spec.as_dict() for sort_spec in self.sort_by],
            "semantic_model": self.semantic_model,
            "use_semantic": self.use_semantic,
            "diversity": self.diversity,
            "library_profile": self.library_profile.as_dict(),
        }


def build_library_profile(
    conn: sqlite3.Connection,
    *,
    include_missing: bool = False,
) -> LibraryProfile:
    """Analyze the current local library for dynamic query parsing."""
    missing_clause = "" if include_missing else "WHERE missing_at IS NULL"
    total_tracks = int(conn.execute(f"SELECT COUNT(*) FROM tracks {missing_clause}").fetchone()[0])

    tag_missing_clause = "" if include_missing else "AND tr.missing_at IS NULL"
    tag_rows = conn.execute(
        f"""
        SELECT tt.tag, COUNT(*) AS count, AVG(tt.score) AS avg_score,
               MAX(tt.score) AS max_score, GROUP_CONCAT(DISTINCT tt.source) AS sources
        FROM track_tags tt
        JOIN tracks tr ON tr.id = tt.track_id
        WHERE 1 = 1
          {tag_missing_clause}
        GROUP BY tt.tag
        ORDER BY count DESC, avg_score DESC, tt.tag ASC
        """
    ).fetchall()
    tag_stats = {
        row["tag"]: TagStat(
            tag=row["tag"],
            count=int(row["count"]),
            avg_score=float(row["avg_score"] or 0.0),
            max_score=float(row["max_score"] or 0.0),
            sources=sorted(str(row["sources"] or "").split(",")) if row["sources"] else [],
        )
        for row in tag_rows
    }

    feature_percentiles = _feature_percentiles(conn, include_missing=include_missing)
    embedding_missing_clause = "" if include_missing else "AND tr.missing_at IS NULL"
    embedding_rows = conn.execute(
        f"""
        SELECT DISTINCT e.model
        FROM embeddings e
        JOIN tracks tr ON tr.id = e.track_id
        WHERE e.kind = ?
          {embedding_missing_clause}
        ORDER BY e.model
        """,
        (EMBEDDING_KIND,),
    ).fetchall()

    return LibraryProfile(
        total_tracks=total_tracks,
        tag_stats=tag_stats,
        feature_percentiles=feature_percentiles,
        embedding_models=[row["model"] for row in embedding_rows],
    )


def parse_intent_dynamic(
    query: str,
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    include_missing: bool = False,
    semantic_model: str = DEFAULT_EMBEDDING_MODEL,
    llm_hints: IntentHints | None = None,
    parser: str = "dynamic",
    llm_error: str | None = None,
) -> SearchIntent:
    """Parse a query using common listening priors plus the actual library vocabulary."""
    library_profile = build_library_profile(conn, include_missing=include_missing)
    negated_terms = _negated_query_terms(query)
    query_terms = [term for term in normalize_terms(query) if term not in negated_terms]
    local_contexts = _detect_contexts(query)
    parsed_limit = limit or _parse_limit(query) or 10

    prefer_concepts: list[str] = []
    avoid_concepts: list[str] = []
    requested_feature_preferences: dict[str, list[str]] = {}
    sort_by = _parse_sort_specs(query)
    prior_prefer, prior_avoid, prior_features = _detect_query_priors(query)
    accepted_llm_hints, rejected_llm_hints, llm_policy = _apply_llm_advisory_policy(
        query_terms=query_terms,
        local_contexts=local_contexts,
        local_sort_by=sort_by,
        local_prior_features=prior_features,
        library_profile=library_profile,
        llm_hints=llm_hints,
    )
    detected_contexts = _unique(
        local_contexts + (accepted_llm_hints.contexts if accepted_llm_hints else [])
    )
    prefer_concepts.extend(prior_prefer)
    avoid_concepts.extend(prior_avoid)
    for field_name, levels in prior_features.items():
        requested_feature_preferences.setdefault(field_name, []).extend(levels)

    for context in detected_contexts:
        if context not in CONTEXT_PRIORS:
            continue
        prior = CONTEXT_PRIORS[context]
        prefer_concepts.extend(prior.get("prefer", []))
        avoid_concepts.extend(prior.get("avoid", []))
        for field_name, level in prior.get("features", {}).items():
            requested_feature_preferences.setdefault(field_name, []).append(level)

    if accepted_llm_hints is not None:
        prefer_concepts.extend(accepted_llm_hints.prefer_tag_concepts)
        avoid_concepts.extend(accepted_llm_hints.avoid_tag_concepts)

    for sort_spec in sort_by:
        if sort_spec.field in DEFAULT_FEATURE_RANGES:
            requested_feature_preferences.setdefault(sort_spec.field, []).append(
                "very_high" if sort_spec.direction == "desc" else "very_low"
            )

    # Unknown or specific query words still influence tag matching dynamically, but
    # remove command/sort filler so words like "highest" or "show" do not pollute ranking.
    prefer_concepts.extend(
        term
        for term in query_terms
        if term not in QUERY_CONCEPT_STOP_WORDS and term not in avoid_concepts
    )

    prefer_concepts = _unique(prefer_concepts)
    avoid_concepts = _unique(avoid_concepts)
    prefer_tags = match_library_tags(prefer_concepts, library_profile.tag_stats)
    avoid_tags = match_library_tags(avoid_concepts, library_profile.tag_stats)

    feature_ranges: dict[str, FeatureRange] = {}
    for field_name, levels in requested_feature_preferences.items():
        for level in levels:
            feature_range = dynamic_feature_range(library_profile, field_name, level)
            feature_ranges[field_name] = _merge_feature_range(
                feature_ranges.get(field_name),
                feature_range,
            )

    available_semantic_model = _select_available_semantic_model(
        semantic_model,
        library_profile.embedding_models,
    )

    return SearchIntent(
        query=query,
        limit=max(1, min(100, parsed_limit)),
        include_missing=include_missing,
        parser=parser,
        llm_hints=accepted_llm_hints,
        llm_rejected_hints=rejected_llm_hints,
        llm_policy=llm_policy,
        llm_error=llm_error,
        contexts=detected_contexts,
        query_terms=query_terms,
        prefer_tag_concepts=prefer_concepts,
        avoid_tag_concepts=avoid_concepts,
        prefer_tags=prefer_tags,
        avoid_tags=avoid_tags,
        feature_ranges=feature_ranges,
        sort_by=sort_by,
        semantic_model=available_semantic_model,
        use_semantic=available_semantic_model is not None,
        diversity={"max_tracks_per_artist": 2},
        library_profile=library_profile,
    )


def _apply_llm_advisory_policy(
    *,
    query_terms: list[str],
    local_contexts: list[str],
    local_sort_by: list[SortSpec],
    local_prior_features: dict[str, list[str]],
    library_profile: LibraryProfile,
    llm_hints: IntentHints | None,
) -> tuple[IntentHints | None, IntentHints | None, dict[str, Any]]:
    """Accept only advisory LLM context/tag expansion.

    The deterministic parser owns objective intent: exact terms, limits, feature ranges,
    and sort directives. LLM hints can add contextual/tag concepts only for vague or
    contextual queries.
    """
    if llm_hints is None:
        return None, None, {"used": False, "role": "none"}

    objective_reason = _objective_llm_ignore_reason(
        query_terms=query_terms,
        local_contexts=local_contexts,
        local_sort_by=local_sort_by,
        local_prior_features=local_prior_features,
        library_profile=library_profile,
    )
    if objective_reason:
        return (
            None,
            llm_hints,
            {
                "used": True,
                "role": "ignored_for_objective_query",
                "accepted_fields": [],
                "rejected_fields": _non_empty_hint_fields(llm_hints),
                "ignored_because": objective_reason,
            },
        )

    accepted = IntentHints(
        contexts=llm_hints.contexts,
        prefer_tag_concepts=llm_hints.prefer_tag_concepts,
        avoid_tag_concepts=llm_hints.avoid_tag_concepts,
        notes=llm_hints.notes,
    )
    rejected = IntentHints(
        feature_preferences=llm_hints.feature_preferences,
        sort_by=llm_hints.sort_by,
        limit=llm_hints.limit,
    )
    accepted_fields = _non_empty_hint_fields(accepted)
    rejected_fields = _non_empty_hint_fields(rejected)
    return (
        accepted if accepted_fields else None,
        rejected if rejected_fields else None,
        {
            "used": True,
            "role": "advisory_context_expansion",
            "accepted_fields": accepted_fields,
            "rejected_fields": rejected_fields,
            "ignored_because": None,
        },
    )


def _objective_llm_ignore_reason(
    *,
    query_terms: list[str],
    local_contexts: list[str],
    local_sort_by: list[SortSpec],
    local_prior_features: dict[str, list[str]],
    library_profile: LibraryProfile,
) -> str | None:
    content_terms = [term for term in query_terms if term not in QUERY_CONCEPT_STOP_WORDS]
    direct_tag_matches = match_library_tags(content_terms, library_profile.tag_stats)
    if local_sort_by:
        return "explicit_sort"
    if direct_tag_matches and len(content_terms) <= 3:
        return "direct_library_tag_or_genre"
    if local_prior_features and not local_contexts:
        return "explicit_feature_query"
    if content_terms and not local_contexts:
        return "metadata_or_text_query"
    return None


def _non_empty_hint_fields(hints: IntentHints) -> list[str]:
    fields: list[str] = []
    if hints.contexts:
        fields.append("contexts")
    if hints.prefer_tag_concepts:
        fields.append("prefer_tag_concepts")
    if hints.avoid_tag_concepts:
        fields.append("avoid_tag_concepts")
    if hints.feature_preferences:
        fields.append("feature_preferences")
    if hints.sort_by:
        fields.append("sort_by")
    if hints.limit is not None:
        fields.append("limit")
    return fields


def _select_available_semantic_model(
    requested_model: str,
    available_models: list[str],
) -> str | None:
    if requested_model in available_models:
        return requested_model
    requested_leaf = requested_model.rstrip("/").split("/")[-1]
    for model in available_models:
        if model.rstrip("/").split("/")[-1] == requested_leaf:
            return model
    if requested_model == DEFAULT_EMBEDDING_MODEL and available_models:
        return available_models[0]
    return None


def _parse_sort_specs(query: str) -> list[SortSpec]:
    normalized = " ".join(query.lower().split())
    specs: list[SortSpec] = []

    def add(field: str, direction: str, source: str) -> None:
        specs.append(SortSpec(field=field, direction=direction, source=source))

    high_tempo_pattern = (
        r"\b("
        r"highest|max(?:imum)?|fastest|quickest|"
        r"highest\s+(?:bpm|tempo)|most\s+(?:bpm|tempo)|"
        r"high\s+(?:bpm|tempo)|fast\s+(?:bpm|tempo)"
        r")\b"
    )
    if re.search(high_tempo_pattern, normalized):
        if re.search(r"\b(bpm|tempo|fastest|quickest)\b", normalized):
            add("tempo_bpm", "desc", "natural_language")
    low_tempo_pattern = (
        r"\b("
        r"lowest|min(?:imum)?|slowest|"
        r"least\s+(?:bpm|tempo)|low\s+(?:bpm|tempo)|slow\s+(?:bpm|tempo)"
        r")\b"
    )
    if re.search(low_tempo_pattern, normalized):
        if re.search(r"\b(bpm|tempo|slowest)\b", normalized):
            add("tempo_bpm", "asc", "natural_language")

    feature_patterns = {
        "energy": {
            "desc": [
                r"\bmost energetic\b",
                r"\bhighest energy\b",
                r"\bhigh energy\b",
                r"\bmax(?:imum)? energy\b",
            ],
            "asc": [
                r"\bleast energetic\b",
                r"\blowest energy\b",
                r"\blow energy\b",
                r"\bminimum energy\b",
            ],
        },
        "danceability": {
            "desc": [
                r"\bmost danceable\b",
                r"\bdanciest\b",
                r"\bhighest danceability\b",
                r"\bhigh danceability\b",
            ],
            "asc": [r"\bleast danceable\b", r"\blowest danceability\b", r"\blow danceability\b"],
        },
        "aggression": {
            "desc": [
                r"\bmost aggressive\b",
                r"\bhardest\b",
                r"\bhighest aggression\b",
                r"\bhigh aggression\b",
            ],
            "asc": [
                r"\bleast aggressive\b",
                r"\bsoftest\b",
                r"\blowest aggression\b",
                r"\blow aggression\b",
            ],
        },
        "brightness": {
            "desc": [
                r"\bbrightest\b",
                r"\bhighest brightness\b",
                r"\bhigh brightness\b",
                r"\bmost bright\b",
            ],
            "asc": [
                r"\bdarkest\b",
                r"\blowest brightness\b",
                r"\blow brightness\b",
                r"\bleast bright\b",
            ],
        },
    }
    for field_name, directions in feature_patterns.items():
        for direction, patterns in directions.items():
            if any(re.search(pattern, normalized) for pattern in patterns):
                add(field_name, direction, "natural_language")

    explicit = re.search(
        r"\b(?:sort|order)\s+by\s+(bpm|tempo|energy|danceability|dance|aggression|brightness)\s*(asc|ascending|desc|descending)?\b",
        normalized,
    )
    if explicit:
        field = _normalize_sort_field(explicit.group(1))
        direction = "asc" if explicit.group(2) in {"asc", "ascending"} else "desc"
        add(field, direction, "natural_language")

    return _merge_sort_specs([], specs)


def _normalize_sort_field(field_name: str) -> str:
    normalized = field_name.strip().lower()
    if normalized in {"bpm", "tempo"}:
        return "tempo_bpm"
    if normalized == "dance":
        return "danceability"
    return normalized


def _merge_sort_specs(*groups: list[SortSpec]) -> list[SortSpec]:
    output: list[SortSpec] = []
    seen: set[str] = set()
    for group in groups:
        for spec in group:
            field_name = _normalize_sort_field(spec.field)
            direction = spec.direction if spec.direction in {"asc", "desc"} else "desc"
            if field_name not in DEFAULT_FEATURE_RANGES or field_name in seen:
                continue
            seen.add(field_name)
            output.append(SortSpec(field=field_name, direction=direction, source=spec.source))
    return output[:3]


def normalize_terms(text: str) -> list[str]:
    """Normalize text into search terms."""
    stop_words = {
        "a",
        "an",
        "am",
        "and",
        "are",
        "bar",
        "bpm",
        "for",
        "give",
        "i",
        "im",
        "in",
        "is",
        "m",
        "me",
        "music",
        "my",
        "of",
        "play",
        "songs",
        "that",
        "the",
        "to",
        "track",
        "tracks",
        "want",
        "with",
        "without",
        *QUERY_CONCEPT_STOP_WORDS,
    }
    return [
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in stop_words and not term.isdigit()
    ]


def match_library_tags(concepts: list[str], tag_stats: dict[str, TagStat]) -> list[str]:
    """Select actual local-library tags matching query/context concepts."""
    matches: list[tuple[str, float]] = []
    for tag, stat in tag_stats.items():
        tag_terms = set(normalize_tag_terms(tag))
        normalized_tag = " ".join(tag_terms)
        for concept in concepts:
            concept_terms = set(normalize_tag_terms(concept))
            if not concept_terms or _is_opposite_polarity_tag(tag_terms, concept_terms):
                continue
            concept_text = " ".join(concept_terms)
            if _concept_matches_tag_text(concept_terms, concept_text, tag_terms, normalized_tag):
                matches.append((tag, stat.count + stat.avg_score))
                break
    matches.sort(key=lambda item: item[1], reverse=True)
    return _unique([tag for tag, _ in matches])


def _concept_matches_tag_text(
    concept_terms: set[str],
    concept_text: str,
    tag_terms: set[str],
    normalized_tag: str,
) -> bool:
    if concept_terms.issubset(tag_terms):
        return True
    if any(len(term) < 3 for term in concept_terms):
        return False
    return concept_text in normalized_tag


def _is_opposite_polarity_tag(tag_terms: set[str], concept_terms: set[str]) -> bool:
    polarity_terms = {"low", "not", "no", "without"}
    return bool(tag_terms.intersection(polarity_terms)) and not bool(
        concept_terms.intersection(polarity_terms)
    )


def normalize_tag_terms(value: str) -> list[str]:
    """Normalize tag labels such as electronic---ambient into tokens."""
    return re.findall(r"[a-z0-9]+", value.lower().replace("---", " ").replace("_", " "))


def dynamic_feature_range(
    profile: LibraryProfile, field_name: str, level: str) -> FeatureRange:
    """Build a feature range from library percentiles when enough data exists."""
    percentiles = profile.feature_percentiles.get(field_name)
    if not percentiles or int(percentiles.get("count", 0)) < 3:
        low, high = DEFAULT_FEATURE_RANGES[field_name][level]
        return FeatureRange(low=low, high=high, source=f"fallback:{level}")

    if level == "very_low":
        low, high = percentiles["min"], percentiles["p25"]
    elif level == "low":
        low, high = percentiles["min"], percentiles["p35"]
    elif level == "low_mid":
        low, high = percentiles["min"], percentiles["p55"]
    elif level == "mid":
        low, high = percentiles["p25"], percentiles["p75"]
    elif level == "mid_high":
        low, high = percentiles["p45"], percentiles["max"]
    elif level == "high":
        low, high = percentiles["p65"], percentiles["max"]
    elif level == "very_high":
        low, high = percentiles["p75"], percentiles["max"]
    else:
        low, high = DEFAULT_FEATURE_RANGES[field_name]["mid"]
        return FeatureRange(low=low, high=high, source="fallback:mid")

    return FeatureRange(
        low=round(low, 6),
        high=round(high, 6),
        source=f"library_percentile:{level}",
    )


def _feature_percentiles(
    conn: sqlite3.Connection, *, include_missing: bool) -> dict[str, dict[str, float]]:
    missing_clause = "" if include_missing else "AND tr.missing_at IS NULL"
    field_sql = {
        "energy": "af.energy",
        "danceability": "af.danceability",
        "aggression": "af.aggression",
        "brightness": "af.brightness",
    }
    output: dict[str, dict[str, float]] = {}
    for field_name, sql_expr in field_sql.items():
        rows = conn.execute(
            f"""
            SELECT {sql_expr} AS value
            FROM audio_features af
            JOIN tracks tr ON tr.id = af.track_id
            WHERE {sql_expr} IS NOT NULL
              {missing_clause}
            ORDER BY value
            """
        ).fetchall()
        values = [float(row["value"]) for row in rows]
        if not values:
            continue
        output[field_name] = {
            "count": float(len(values)),
            "min": round(min(values), 6),
            "p25": round(_quantile(values, 0.25), 6),
            "p35": round(_quantile(values, 0.35), 6),
            "p45": round(_quantile(values, 0.45), 6),
            "p55": round(_quantile(values, 0.55), 6),
            "p65": round(_quantile(values, 0.65), 6),
            "p70": round(_quantile(values, 0.70), 6),
            "p75": round(_quantile(values, 0.75), 6),
            "max": round(max(values), 6),
        }

    tempo_rows = conn.execute(
        f"""
        SELECT
            af.bpm AS value,
            tr.title,
            tr.artist,
            tr.album,
            tr.genre,
            GROUP_CONCAT(tt.tag) AS tags,
            GROUP_CONCAT(tt.source) AS sources
        FROM audio_features af
        JOIN tracks tr ON tr.id = af.track_id
        LEFT JOIN track_tags tt ON tt.track_id = tr.id
        WHERE af.bpm IS NOT NULL
          {missing_clause}
        GROUP BY tr.id
        ORDER BY value
        """
    ).fetchall()
    tempo_values = []
    for row in tempo_rows:
        descriptors = tempo_descriptors_from_metadata_and_tags(
            {
                "title": row["title"],
                "artist": row["artist"],
                "album": row["album"],
                "genre": row["genre"],
            },
            [
                {"tag": tag, "source": source}
                for tag, source in zip(
                    str(row["tags"] or "").split(","),
                    str(row["sources"] or "").split(","),
                    strict=False,
                )
                if tag or source
            ],
        )
        value = perceived_tempo_bpm(row["value"], descriptors=descriptors)
        if value is not None:
            tempo_values.append(value)
    if tempo_values:
        output["tempo_bpm"] = {
            "count": float(len(tempo_values)),
            "min": round(min(tempo_values), 6),
            "p25": round(_quantile(tempo_values, 0.25), 6),
            "p35": round(_quantile(tempo_values, 0.35), 6),
            "p45": round(_quantile(tempo_values, 0.45), 6),
            "p55": round(_quantile(tempo_values, 0.55), 6),
            "p65": round(_quantile(tempo_values, 0.65), 6),
            "p70": round(_quantile(tempo_values, 0.70), 6),
            "p75": round(_quantile(tempo_values, 0.75), 6),
            "max": round(max(tempo_values), 6),
        }
    return output


def _detect_query_priors(query: str) -> tuple[list[str], list[str], dict[str, list[str]]]:
    prefer: list[str] = []
    avoid: list[str] = []
    features: dict[str, list[str]] = {}
    matched_names = [
        name
        for name, prior in QUERY_PRIORS.items()
        if any(_keyword_present(query, keyword) for keyword in prior.get("keywords", []))
    ]
    if "not_aggressive" in matched_names:
        matched_names = [name for name in matched_names if name != "aggressive"]
    if "instrumental" in matched_names:
        matched_names = [name for name in matched_names if name != "vocal"]

    for name in matched_names:
        prior = QUERY_PRIORS[name]
        prefer.extend(prior.get("prefer", []))
        avoid.extend(prior.get("avoid", []))
        for field_name, level in prior.get("features", {}).items():
            if field_name in DEFAULT_FEATURE_RANGES and level in DEFAULT_FEATURE_RANGES[field_name]:
                features.setdefault(field_name, []).append(level)
    return _unique(prefer), _unique(avoid), features


def _detect_contexts(query: str) -> list[str]:
    contexts: list[str] = []
    for context, prior in CONTEXT_PRIORS.items():
        for keyword in prior["keywords"]:
            if _keyword_present(query, keyword):
                contexts.append(context)
                break
    return contexts


def _keyword_present(query: str, keyword: str) -> bool:
    normalized_query = query.lower().replace("-", " ")
    normalized_keyword = keyword.lower().replace("-", " ")
    escaped = re.escape(normalized_keyword).replace(r"\ ", r"\s+")
    return re.search(rf"\b{escaped}\b", normalized_query) is not None


def negated_query_terms(query: str) -> list[str]:
    """Return normalized query terms that are explicitly negated."""
    return sorted(_negated_query_terms(query))


def _negated_query_terms(query: str) -> set[str]:
    negated: set[str] = set()
    if any(
        _keyword_present(query, phrase)
        for phrase in [
            "not aggressive",
            "less aggressive",
            "low aggression",
            "non aggressive",
            "non-aggressive",
        ]
    ):
        negated.update({"aggressive", "aggression"})
    if any(
        _keyword_present(query, phrase)
        for phrase in ["no vocals", "without vocals", "vocal free", "vocal-free", "no singing"]
    ):
        negated.update({"vocal", "vocals", "singing", "singer"})
    if any(_keyword_present(query, phrase) for phrase in ["not fast", "not too fast"]):
        negated.update({"fast", "faster", "uptempo"})
    if any(_keyword_present(query, phrase) for phrase in ["not slow", "not too slow"]):
        negated.update({"slow", "slower", "downtempo"})
    return negated


def _parse_limit(query: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s*(tracks?|songs?|results?)?\b", query.lower())
    if not match:
        return None
    return int(match.group(1))


def _merge_feature_range(existing: FeatureRange | None, new: FeatureRange) -> FeatureRange:
    if existing is None:
        return new
    low = max(existing.low, new.low)
    high = min(existing.high, new.high)
    if low <= high:
        return FeatureRange(low=low, high=high, source=f"{existing.source}+{new.source}")
    return FeatureRange(
        low=min(existing.low, new.low),
        high=max(existing.high, new.high),
        source=f"union:{existing.source}+{new.source}",
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output
