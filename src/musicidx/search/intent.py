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
        "avoid": ["aggressive", "hardcore", "metal", "chaotic", "heavy", "party", "workout"],
        "features": {"energy": "low", "aggression": "low", "brightness": "low_mid"},
    },
    "bar": {
        "keywords": ["bar", "lounge", "cocktail", "cafe", "restaurant"],
        "prefer": ["background", "relaxing", "downtempo", "ambient", "jazz", "soul", "lounge"],
        "avoid": ["aggressive", "hardcore", "metal", "chaotic", "very loud", "party"],
        "features": {
            "energy": "low",
            "aggression": "low",
            "brightness": "low_mid",
            "tempo_bpm": "low_mid",
        },
    },
    "background": {
        "keywords": ["background", "background music", "in the background"],
        "prefer": ["background", "ambient", "calm", "instrumental", "soft"],
        "avoid": ["aggressive", "chaotic", "very energetic", "vocal", "speech"],
        "features": {"energy": "low_mid", "aggression": "low"},
    },
    "cooking": {
        "keywords": ["cooking", "cook", "kitchen"],
        "prefer": ["cooking", "dinner", "background", "warm", "groovy", "pleasant"],
        "avoid": ["aggressive", "chaotic", "harsh", "sleep"],
        "features": {"energy": "mid", "aggression": "low", "danceability": "mid"},
    },
    "dinner": {
        "keywords": ["dinner", "supper", "dining"],
        "prefer": ["dinner", "warm", "lounge", "background", "soft", "pleasant"],
        "avoid": ["aggressive", "chaotic", "workout", "hardcore", "very energetic"],
        "features": {"energy": "low_mid", "aggression": "low", "brightness": "low_mid"},
    },
    "no_vocals_background": {
        "keywords": ["no vocals", "without vocals", "instrumental background", "no voice"],
        "prefer": ["instrumental", "background", "ambient", "calm", "no vocals"],
        "avoid": ["vocal", "vocals", "speech", "singer", "rap"],
        "features": {"energy": "low_mid", "aggression": "low"},
    },
    "driving": {
        "keywords": ["driving", "drive", "road trip", "car music"],
        "prefer": ["driving", "groove", "energetic", "steady", "upbeat"],
        "avoid": ["sleep", "meditative", "chaotic"],
        "features": {"energy": "mid_high", "danceability": "mid_high", "tempo_bpm": "mid_high"},
    },
    "shower": {
        "keywords": ["shower", "morning"],
        "prefer": ["happy", "upbeat", "energetic", "fun", "pop", "dance", "synth-pop"],
        "avoid": ["sleep", "sad", "dark", "drone", "meditative"],
        "features": {"energy": "high", "danceability": "mid_high", "tempo_bpm": "mid_high"},
    },
    "melancholic": {
        "keywords": [
            "sad",
            "melancholic",
            "melancholy",
            "blue",
            "heartbreak",
            "reflective",
            "introspective",
        ],
        "prefer": [
            "sad",
            "melancholic",
            "emotional",
            "dark",
            "romantic",
            "meditative",
            "reflective",
            "introspective",
        ],
        "avoid": ["party", "very energetic", "aggressive"],
        "features": {"energy": "low", "brightness": "low", "aggression": "low"},
    },
    "party": {
        "keywords": ["party", "club", "dancefloor", "celebration"],
        "prefer": ["party", "happy", "dance", "energetic", "house", "disco", "pop"],
        "avoid": ["sleep", "sad", "meditative"],
        "features": {"energy": "high", "danceability": "high", "tempo_bpm": "mid_high"},
    },
    "wedding": {
        "keywords": ["wedding", "weddings", "wedding reception", "reception"],
        "prefer": [
            "romantic",
            "love",
            "happy",
            "uplifting",
            "feel good",
            "warm",
            "party",
            "disco",
            "pop",
            "soul",
        ],
        "avoid": [
            "aggressive",
            "hardcore",
            "hard techno",
            "dark",
            "chaotic",
            "workout",
            "sleep",
            "sad",
        ],
        "features": {
            "energy": "mid",
            "danceability": "mid_high",
            "aggression": "low",
            "tempo_bpm": "mid",
        },
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
        "features": {"energy": "low", "aggression": "low", "brightness": "low_mid"},
    },
    "sleep": {
        "keywords": ["sleep", "sleepy", "bed", "night", "nap", "dream"],
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
        "keywords": ["happy", "feel good", "feel-good", "uplifting", "positive", "upbeat"],
        "prefer": ["happy", "positive", "uplifting", "fun", "upbeat", "energetic"],
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

QUERY_PRIORS: dict[str, dict[str, Any]] = {
    "energetic": {
        "keywords": ["energetic", "high energy", "high-energy", "pumped", "pumping"],
        "prefer": ["energetic", "energy", "upbeat", "powerful", "intense"],
        "avoid": ["sleep", "sleepy", "quiet", "calm"],
        "features": {"energy": "high"},
    },
    "low_energy": {
        "keywords": ["low energy", "low-energy", "quiet", "mellow", "soft", "gentle"],
        "prefer": ["calm", "mellow", "soft", "gentle", "relaxing"],
        "avoid": ["aggressive", "hardcore", "chaotic", "very energetic"],
        "features": {"energy": "low_mid", "aggression": "low"},
    },
    "danceable": {
        "keywords": ["dance", "danceable", "dancy", "danciest", "groove", "groovy"],
        "prefer": ["dance", "danceable", "danceability", "groove", "groovy", "disco", "house"],
        "avoid": ["drone", "ambient sleep"],
        "features": {"danceability": "high", "energy": "mid_high"},
    },
    "fast": {
        "keywords": ["fast", "faster", "uptempo", "up-tempo", "high bpm", "high-bpm"],
        "prefer": ["fast", "uptempo", "energetic"],
        "avoid": ["slow", "sleepy"],
        "features": {"tempo_bpm": "high", "energy": "mid_high"},
    },
    "slow": {
        "keywords": ["slow", "slower", "downtempo", "down-tempo", "low bpm", "low-bpm"],
        "prefer": ["slow", "downtempo", "calm", "relaxing"],
        "avoid": ["fast", "hardcore"],
        "features": {"tempo_bpm": "low", "energy": "low_mid"},
    },
    "aggressive": {
        "keywords": ["aggressive", "hard", "heavy", "intense", "brutal", "chaotic"],
        "prefer": ["aggressive", "hard", "heavy", "intense", "powerful"],
        "avoid": ["sleep", "soft", "calm"],
        "features": {"aggression": "high", "energy": "mid_high"},
    },
    "not_aggressive": {
        "keywords": [
            "not aggressive",
            "less aggressive",
            "low aggression",
            "non aggressive",
            "non-aggressive",
        ],
        "prefer": ["soft", "calm", "gentle", "relaxing", "low aggression"],
        "avoid": ["aggressive", "hardcore", "heavy", "chaotic"],
        "features": {"aggression": "low", "energy": "low_mid"},
    },
    "bright": {
        "keywords": ["bright", "brighter", "sparkly", "shimmering", "shiny"],
        "prefer": ["bright", "sparkly", "uplifting", "shimmering"],
        "avoid": ["dark", "muddy"],
        "features": {"brightness": "high"},
    },
    "dark": {
        "keywords": ["dark", "darker", "moody", "noir", "shadowy"],
        "prefer": ["dark", "moody", "deep", "noir"],
        "avoid": ["bright", "happy"],
        "features": {"brightness": "low"},
    },
    "instrumental": {
        "keywords": ["instrumental", "no vocals", "without vocals", "vocal-free", "vocal free"],
        "prefer": ["instrumental", "background", "ambient", "minimal"],
        "avoid": ["vocal", "vocals", "singer", "singalong"],
        "features": {},
    },
    "vocal": {
        "keywords": ["vocal", "vocals", "singing", "singer", "singalong"],
        "prefer": ["vocal", "vocals", "singing", "singer", "song"],
        "avoid": ["instrumental"],
        "features": {},
    },
    "lofi": {
        "keywords": ["lofi", "lo-fi", "lo fi"],
        "prefer": ["lofi", "lo-fi", "hip hop", "beats", "chill", "background"],
        "avoid": ["aggressive", "party"],
        "features": {"energy": "low_mid", "aggression": "low"},
    },
}

QUERY_CONCEPT_STOP_WORDS = {
    "best",
    "by",
    "fastest",
    "find",
    "good",
    "great",
    "highest",
    "library",
    "list",
    "local",
    "lowest",
    "most",
    "least",
    "nice",
    "no",
    "not",
    "order",
    "please",
    "recommend",
    "recommendations",
    "show",
    "slowest",
    "some",
    "sort",
    "stuff",
    "top",
    "vibe",
    "vibes",
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
    sort_by: list[SortSpec]
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
    detected_contexts = _unique(_detect_contexts(query) + (llm_hints.contexts if llm_hints else []))
    parsed_limit = limit or (llm_hints.limit if llm_hints else None) or _parse_limit(query) or 10

    prefer_concepts: list[str] = []
    avoid_concepts: list[str] = []
    requested_feature_preferences: dict[str, list[str]] = {}
    sort_by = _parse_sort_specs(query)
    prior_prefer, prior_avoid, prior_features = _detect_query_priors(query)
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

    if llm_hints is not None:
        prefer_concepts.extend(llm_hints.prefer_tag_concepts)
        avoid_concepts.extend(llm_hints.avoid_tag_concepts)
        for field_name, level in llm_hints.feature_preferences.items():
            if field_name in DEFAULT_FEATURE_RANGES and level in DEFAULT_FEATURE_RANGES[field_name]:
                requested_feature_preferences.setdefault(field_name, []).append(level)
        sort_by = _merge_sort_specs(sort_by, llm_hints.sort_by)

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
        sort_by=sort_by,
        semantic_model=available_semantic_model,
        use_semantic=available_semantic_model is not None,
        diversity={"max_tracks_per_artist": 2},
        library_profile=library_profile,
    )


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

    high_tempo_pattern = r"\b(highest|max(?:imum)?|fastest|quickest|highest\s+bpm|most\s+bpm)\b"
    if re.search(high_tempo_pattern, normalized):
        if re.search(r"\b(bpm|tempo|fastest|quickest)\b", normalized):
            add("tempo_bpm", "desc", "natural_language")
    if re.search(r"\b(lowest|min(?:imum)?|slowest|least\s+bpm)\b", normalized):
        if re.search(r"\b(bpm|tempo|slowest)\b", normalized):
            add("tempo_bpm", "asc", "natural_language")

    feature_patterns = {
        "energy": {
            "desc": [r"\bmost energetic\b", r"\bhighest energy\b", r"\bmax(?:imum)? energy\b"],
            "asc": [r"\bleast energetic\b", r"\blowest energy\b", r"\bminimum energy\b"],
        },
        "danceability": {
            "desc": [r"\bmost danceable\b", r"\bdanciest\b", r"\bhighest danceability\b"],
            "asc": [r"\bleast danceable\b", r"\blowest danceability\b"],
        },
        "aggression": {
            "desc": [r"\bmost aggressive\b", r"\bhardest\b", r"\bhighest aggression\b"],
            "asc": [r"\bleast aggressive\b", r"\bsoftest\b", r"\blowest aggression\b"],
        },
        "brightness": {
            "desc": [r"\bbrightest\b", r"\bhighest brightness\b", r"\bmost bright\b"],
            "asc": [r"\bdarkest\b", r"\blowest brightness\b", r"\bleast bright\b"],
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
