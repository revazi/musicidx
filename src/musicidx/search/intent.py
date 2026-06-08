"""Dynamic, library-aware query intent parsing."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from musicidx.analyzer.embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_KIND

FEATURE_FIELDS = ["energy", "danceability", "aggression", "brightness", "tempo_bpm"]

DEFAULT_FEATURE_RANGES = {
    "energy": {
        "very_low": (0.0, 0.25),
        "low": (0.0, 0.40),
        "low_mid": (0.0, 0.65),
        "mid": (0.25, 0.75),
        "mid_high": (0.40, 1.0),
        "high": (0.55, 1.0),
        "very_high": (0.70, 1.0),
    },
    "danceability": {
        "very_low": (0.0, 0.25),
        "low": (0.0, 0.40),
        "low_mid": (0.0, 0.60),
        "mid": (0.30, 0.75),
        "mid_high": (0.45, 1.0),
        "high": (0.60, 1.0),
        "very_high": (0.75, 1.0),
    },
    "aggression": {
        "very_low": (0.0, 0.20),
        "low": (0.0, 0.35),
        "low_mid": (0.0, 0.55),
        "mid": (0.25, 0.70),
        "mid_high": (0.40, 1.0),
        "high": (0.60, 1.0),
        "very_high": (0.75, 1.0),
    },
    "brightness": {
        "very_low": (0.0, 0.25),
        "low": (0.0, 0.40),
        "low_mid": (0.0, 0.60),
        "mid": (0.25, 0.75),
        "mid_high": (0.40, 1.0),
        "high": (0.60, 1.0),
        "very_high": (0.75, 1.0),
    },
    "tempo_bpm": {
        "very_low": (45.0, 80.0),
        "low": (50.0, 95.0),
        "low_mid": (60.0, 115.0),
        "mid": (75.0, 125.0),
        "mid_high": (90.0, 150.0),
        "high": (105.0, 180.0),
        "very_high": (130.0, 210.0),
    },
}

CONTEXT_PRIORS: dict[str, dict[str, Any]] = {
    "chill": {
        "keywords": ["chill", "relax", "relaxed", "calm", "laid back", "laid-back"],
        "prefer": ["relaxing", "calm", "ambient", "downtempo", "background", "soft"],
        "avoid": ["aggressive", "hardcore", "metal", "chaotic", "heavy"],
        "features": {"energy": "low_mid", "aggression": "low", "brightness": "low_mid"},
    },
    "bar": {
        "keywords": ["bar", "lounge", "cocktail", "cafe", "restaurant"],
        "prefer": ["background", "relaxing", "downtempo", "ambient", "jazz", "soul", "house"],
        "avoid": ["aggressive", "hardcore", "metal", "chaotic", "very loud"],
        "features": {
            "energy": "low_mid",
            "aggression": "low",
            "danceability": "mid",
            "tempo_bpm": "low_mid",
        },
    },
    "shower": {
        "keywords": ["shower", "morning"],
        "prefer": ["happy", "upbeat", "energetic", "fun", "pop", "dance", "synth-pop"],
        "avoid": ["sleep", "sad", "dark", "drone", "meditative"],
        "features": {"energy": "high", "danceability": "mid_high", "tempo_bpm": "mid_high"},
    },
    "melancholic": {
        "keywords": ["sad", "melancholic", "melancholy", "blue", "heartbreak"],
        "prefer": ["sad", "melancholic", "emotional", "dark", "romantic", "meditative"],
        "avoid": ["party", "very energetic", "aggressive"],
        "features": {"energy": "low_mid", "brightness": "low", "aggression": "low"},
    },
    "party": {
        "keywords": ["party", "club", "dancefloor", "celebration"],
        "prefer": ["party", "happy", "dance", "energetic", "house", "disco", "pop"],
        "avoid": ["sleep", "sad", "meditative"],
        "features": {"energy": "high", "danceability": "high", "tempo_bpm": "mid_high"},
    },
    "workout": {
        "keywords": ["workout", "gym", "run", "running", "exercise", "training"],
        "prefer": ["energetic", "powerful", "sport", "upbeat", "fast"],
        "avoid": ["sleep", "soft", "meditative"],
        "features": {"energy": "very_high", "tempo_bpm": "high", "aggression": "mid_high"},
    },
    "focus": {
        "keywords": ["focus", "study", "work", "coding", "concentrate", "reading"],
        "prefer": ["background", "ambient", "meditative", "calm", "minimal", "deep"],
        "avoid": ["aggressive", "party", "very energetic"],
        "features": {"energy": "low_mid", "aggression": "low", "brightness": "low_mid"},
    },
    "sleep": {
        "keywords": ["sleep", "bed", "night", "nap", "dream"],
        "prefer": ["relaxing", "calm", "meditative", "ambient", "soft", "slow", "soundscape"],
        "avoid": ["party", "energetic", "aggressive", "fast"],
        "features": {
            "energy": "very_low",
            "aggression": "very_low",
            "brightness": "low",
            "tempo_bpm": "low",
        },
    },
    "ambient": {
        "keywords": ["ambient", "atmospheric", "spacey", "soundscape"],
        "prefer": ["ambient", "background", "soundscape", "space", "meditative", "deep"],
        "avoid": ["aggressive", "hardcore"],
        "features": {"aggression": "low", "brightness": "low_mid"},
    },
    "romantic": {
        "keywords": ["romantic", "love", "date"],
        "prefer": ["romantic", "love", "emotional", "soft", "relaxing"],
        "avoid": ["aggressive", "chaotic"],
        "features": {"energy": "low_mid", "aggression": "low"},
    },
    "happy": {
        "keywords": ["happy", "feel good", "uplifting", "positive"],
        "prefer": ["happy", "positive", "uplifting", "fun", "upbeat"],
        "avoid": ["sad", "dark", "melancholic"],
        "features": {"energy": "mid_high", "danceability": "mid_high"},
    },
    "dark": {
        "keywords": ["dark", "moody", "noir"],
        "prefer": ["dark", "deep", "emotional", "ambient"],
        "avoid": ["happy", "party"],
        "features": {"brightness": "low", "aggression": "low_mid"},
    },
}


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
    limit: int | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchIntent:
    query: str
    limit: int
    parser: str
    llm_hints: IntentHints | None
    llm_error: str | None
    contexts: list[str]
    query_terms: list[str]
    prefer_tag_concepts: list[str]
    avoid_tag_concepts: list[str]
    prefer_tags: list[str]
    avoid_tags: list[str]
    feature_ranges: dict[str, FeatureRange]
    semantic_model: str | None
    use_semantic: bool
    diversity: dict[str, int]
    library_profile: LibraryProfile

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "limit": self.limit,
            "parser": self.parser,
            "llm_hints": self.llm_hints.as_dict() if self.llm_hints else None,
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
    query_terms = normalize_terms(query)
    detected_contexts = _unique(_detect_contexts(query) + (llm_hints.contexts if llm_hints else []))
    parsed_limit = limit or (llm_hints.limit if llm_hints else None) or _parse_limit(query) or 10

    prefer_concepts: list[str] = []
    avoid_concepts: list[str] = []
    requested_feature_preferences: dict[str, list[str]] = {}

    for context in detected_contexts:
        if context not in CONTEXT_PRIORS:
            continue
        prior = CONTEXT_PRIORS[context]
        prefer_concepts.extend(prior.get("prefer", []))
        avoid_concepts.extend(prior.get("avoid", []))
        for field_name, level in prior.get("features", {}).items():
            requested_feature_preferences.setdefault(field_name, []).append(level)

    if llm_hints is not None:
        prefer_concepts.extend(llm_hints.prefer_tag_concepts)
        avoid_concepts.extend(llm_hints.avoid_tag_concepts)
        for field_name, level in llm_hints.feature_preferences.items():
            if field_name in DEFAULT_FEATURE_RANGES and level in DEFAULT_FEATURE_RANGES[field_name]:
                requested_feature_preferences.setdefault(field_name, []).append(level)

    # Unknown or specific query words still influence tag matching dynamically.
    prefer_concepts.extend(query_terms)

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

    available_semantic_model = (
        semantic_model if semantic_model in library_profile.embedding_models else None
    )

    return SearchIntent(
        query=query,
        limit=max(1, min(100, parsed_limit)),
        parser=parser,
        llm_hints=llm_hints,
        llm_error=llm_error,
        contexts=detected_contexts,
        query_terms=query_terms,
        prefer_tag_concepts=prefer_concepts,
        avoid_tag_concepts=avoid_concepts,
        prefer_tags=prefer_tags,
        avoid_tags=avoid_tags,
        feature_ranges=feature_ranges,
        semantic_model=available_semantic_model,
        use_semantic=available_semantic_model is not None,
        diversity={"max_tracks_per_artist": 2},
        library_profile=library_profile,
    )


def normalize_terms(text: str) -> list[str]:
    """Normalize text into search terms."""
    stop_words = {
        "a",
        "an",
        "and",
        "for",
        "give",
        "i",
        "me",
        "music",
        "of",
        "play",
        "songs",
        "the",
        "to",
        "track",
        "tracks",
        "want",
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
            if not concept_terms:
                continue
            concept_text = " ".join(concept_terms)
            if concept_terms.issubset(tag_terms) or concept_text in normalized_tag:
                matches.append((tag, stat.count + stat.avg_score))
                break
    matches.sort(key=lambda item: item[1], reverse=True)
    return _unique([tag for tag, _ in matches])


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
        low, high = percentiles["min"], percentiles["p35"]
    elif level == "low":
        low, high = percentiles["min"], percentiles["p45"]
    elif level == "low_mid":
        low, high = percentiles["min"], percentiles["p65"]
    elif level == "mid":
        low, high = percentiles["p25"], percentiles["p75"]
    elif level == "mid_high":
        low, high = percentiles["p35"], percentiles["max"]
    elif level == "high":
        low, high = percentiles["p55"], percentiles["max"]
    elif level == "very_high":
        low, high = percentiles["p70"], percentiles["max"]
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
        "tempo_bpm": "af.bpm",
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
    return output


def _detect_contexts(query: str) -> list[str]:
    normalized = query.lower()
    contexts: list[str] = []
    for context, prior in CONTEXT_PRIORS.items():
        for keyword in prior["keywords"]:
            if re.search(rf"\b{re.escape(keyword)}\b", normalized):
                contexts.append(context)
                break
    return contexts


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
