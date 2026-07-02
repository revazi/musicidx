"""Declarative search taxonomy loading and validation.

The taxonomy file captures parser concepts (contexts, query priors, feature range names,
semantic/source policies) in data. The current parser still owns runtime behavior; this module
makes the vocabulary inspectable and gives later phases a safe source to migrate toward.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

try:  # pragma: no cover - dependency availability is environment-specific.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

TAXONOMY_RESOURCE = "taxonomy.yaml"
REQUIRED_FEATURE_FIELDS = {"aggression", "brightness", "danceability", "energy", "tempo_bpm"}
REQUIRED_FEATURE_LEVELS = {
    "very_low",
    "low",
    "low_mid",
    "mid",
    "mid_high",
    "high",
    "very_high",
}


@dataclass(frozen=True, slots=True)
class TaxonomyEntry:
    """Context/query-prior taxonomy entry."""

    name: str
    keywords: tuple[str, ...]
    prefer: tuple[str, ...]
    avoid: tuple[str, ...]
    features: dict[str, str]


@dataclass(frozen=True, slots=True)
class SearchTaxonomy:
    """Validated search taxonomy data."""

    schema_version: int
    feature_ranges: dict[str, dict[str, tuple[float, float]]]
    contexts: dict[str, TaxonomyEntry]
    query_priors: dict[str, TaxonomyEntry]
    occasion_contexts: tuple[str, ...]
    query_concept_stop_words: tuple[str, ...]
    semantic_roles: dict[str, str]
    identity_policy: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_ranges": {
                field: {level: list(bounds) for level, bounds in levels.items()}
                for field, levels in self.feature_ranges.items()
            },
            "contexts": {
                name: _entry_as_dict(entry) for name, entry in self.contexts.items()
            },
            "query_priors": {
                name: _entry_as_dict(entry) for name, entry in self.query_priors.items()
            },
            "occasion_contexts": list(self.occasion_contexts),
            "query_concept_stop_words": list(self.query_concept_stop_words),
            "semantic_roles": dict(self.semantic_roles),
            "identity_policy": dict(self.identity_policy),
        }


@lru_cache(maxsize=1)
def load_search_taxonomy() -> SearchTaxonomy:
    """Load and validate the bundled search taxonomy."""
    with resources.files(__package__).joinpath(TAXONOMY_RESOURCE).open(
        "r", encoding="utf-8"
    ) as handle:
        data = _load_yaml(handle.read())
    return parse_search_taxonomy(data)


def load_search_taxonomy_from_path(path: str | Path) -> SearchTaxonomy:
    """Load and validate a taxonomy file from disk."""
    data = _load_yaml(Path(path).read_text(encoding="utf-8"))
    return parse_search_taxonomy(data)


def parse_search_taxonomy(data: Any) -> SearchTaxonomy:
    """Validate raw taxonomy data and return a typed representation."""
    if not isinstance(data, dict):
        raise ValueError("taxonomy root must be a mapping")

    schema_version = _required_int(data, "schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported taxonomy schema_version: {schema_version}")

    feature_ranges = _parse_feature_ranges(_required_mapping(data, "feature_ranges"))
    contexts = _parse_entries(_required_mapping(data, "contexts"), section="contexts")
    query_priors = _parse_entries(
        _required_mapping(data, "query_priors"), section="query_priors"
    )
    occasion_contexts = tuple(_required_string_list(data, "occasion_contexts"))
    for context in occasion_contexts:
        if context not in contexts:
            raise ValueError(f"occasion context {context!r} is not defined in contexts")

    query_concept_stop_words = tuple(_required_string_list(data, "query_concept_stop_words"))
    semantic_roles = _string_mapping(_required_mapping(data, "semantic_roles"), "semantic_roles")
    identity_policy = _string_mapping(_required_mapping(data, "identity_policy"), "identity_policy")

    _validate_entry_features(contexts, feature_ranges, section="contexts")
    _validate_entry_features(query_priors, feature_ranges, section="query_priors")

    return SearchTaxonomy(
        schema_version=schema_version,
        feature_ranges=feature_ranges,
        contexts=contexts,
        query_priors=query_priors,
        occasion_contexts=occasion_contexts,
        query_concept_stop_words=query_concept_stop_words,
        semantic_roles=semantic_roles,
        identity_policy=identity_policy,
    )


def _load_yaml(text: str) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load the search taxonomy")
    return yaml.safe_load(text)


def _parse_feature_ranges(raw: dict[str, Any]) -> dict[str, dict[str, tuple[float, float]]]:
    fields = set(raw)
    missing_fields = REQUIRED_FEATURE_FIELDS - fields
    if missing_fields:
        raise ValueError(f"feature_ranges missing fields: {sorted(missing_fields)}")

    parsed: dict[str, dict[str, tuple[float, float]]] = {}
    for field_name, levels_raw in raw.items():
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("feature range field names must be non-empty strings")
        if not isinstance(levels_raw, dict):
            raise ValueError(f"feature_ranges.{field_name} must be a mapping")
        levels: dict[str, tuple[float, float]] = {}
        missing_levels = REQUIRED_FEATURE_LEVELS - set(levels_raw)
        if missing_levels:
            raise ValueError(
                f"feature_ranges.{field_name} missing levels: {sorted(missing_levels)}"
            )
        for level_name, bounds_raw in levels_raw.items():
            bounds = _number_pair(bounds_raw, f"feature_ranges.{field_name}.{level_name}")
            if bounds[0] > bounds[1]:
                raise ValueError(f"feature range low > high for {field_name}.{level_name}")
            levels[level_name] = bounds
        parsed[field_name] = levels
    return parsed


def _parse_entries(raw: dict[str, Any], *, section: str) -> dict[str, TaxonomyEntry]:
    parsed: dict[str, TaxonomyEntry] = {}
    for name, item in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{section} names must be non-empty strings")
        if not isinstance(item, dict):
            raise ValueError(f"{section}.{name} must be a mapping")
        keywords = tuple(_string_list(item.get("keywords"), f"{section}.{name}.keywords"))
        prefer = tuple(_string_list(item.get("prefer", []), f"{section}.{name}.prefer"))
        avoid = tuple(_string_list(item.get("avoid", []), f"{section}.{name}.avoid"))
        features = _string_mapping(item.get("features", {}), f"{section}.{name}.features")
        if not keywords:
            raise ValueError(f"{section}.{name}.keywords must not be empty")
        parsed[name] = TaxonomyEntry(
            name=name,
            keywords=keywords,
            prefer=prefer,
            avoid=avoid,
            features=features,
        )
    return parsed


def _validate_entry_features(
    entries: dict[str, TaxonomyEntry],
    feature_ranges: dict[str, dict[str, tuple[float, float]]],
    *,
    section: str,
) -> None:
    for name, entry in entries.items():
        for field_name, level_name in entry.features.items():
            if field_name not in feature_ranges:
                raise ValueError(f"{section}.{name} uses unknown feature field {field_name!r}")
            if level_name not in feature_ranges[field_name]:
                raise ValueError(
                    f"{section}.{name} uses unknown feature level {field_name}.{level_name}"
                )


def _entry_as_dict(entry: TaxonomyEntry) -> dict[str, Any]:
    return {
        "keywords": list(entry.keywords),
        "prefer": list(entry.prefer),
        "avoid": list(entry.avoid),
        "features": dict(entry.features),
    }


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"taxonomy.{key} must be a mapping")
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"taxonomy.{key} must be an integer")
    return value


def _required_string_list(data: dict[str, Any], key: str) -> list[str]:
    return _string_list(data.get(key), f"taxonomy.{key}")


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path} entries must be non-empty strings")
        normalized = item.strip().lower()
        if normalized in seen:
            raise ValueError(f"{path} contains duplicate entry {item!r}")
        seen.add(normalized)
        output.append(item)
    return output


def _string_mapping(value: Any, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path}.{key} must be a non-empty string")
        output[key] = item
    return output


def _number_pair(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{path} must be a two-number list")
    low, high = value
    if not isinstance(low, int | float) or not isinstance(high, int | float):
        raise ValueError(f"{path} must contain numbers")
    return float(low), float(high)
