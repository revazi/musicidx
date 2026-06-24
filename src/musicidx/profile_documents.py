"""Versioned track-profile document helpers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION = 2
PROFILE_VERSION = "2026-06-23.1"

IDENTITY_FIELDS = (
    "title",
    "artist",
    "album",
    "album_artist",
    "genre",
    "date",
    "track_number",
    "disc_number",
)

FEATURE_COMPLETENESS_FIELDS = (
    "bpm",
    "energy",
    "danceability",
    "aggression",
    "brightness",
)


def normalize_text(value: str | None) -> str | None:
    """Normalize user/display metadata for stable comparison and search fields."""
    if not value:
        return None
    text = unicodedata.normalize("NFKD", value).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def build_profile_document(
    *,
    metadata: Mapping[str, Any],
    path: Path | str,
    profile_text: str,
    track_id: str | None = None,
    generated_at: str | None = None,
    analysis_version: int | None = None,
    normalized: Mapping[str, Any] | None = None,
    metadata_confidence: float | None = None,
    external_match_confidence: float | None = None,
    field_confidence: Mapping[str, float] | None = None,
    provenance: Mapping[str, str] | None = None,
    audio_features: Mapping[str, Any] | None = None,
    tags: list[Mapping[str, Any]] | None = None,
    context_fit: Mapping[str, float] | None = None,
    missing: bool = False,
) -> dict[str, Any]:
    """Build a versioned materialized track profile document."""
    metadata_dict = {field: metadata.get(field) for field in IDENTITY_FIELDS}
    normalized_dict = _normalized_identity(metadata_dict, normalized)
    confidence_dict = _confidence_summary(
        metadata_dict,
        metadata_confidence=metadata_confidence,
        field_confidence=field_confidence,
    )
    tag_rows = [_tag_dict(tag) for tag in (tags or [])]
    musical = _musical_dict(audio_features or {})
    context_scores = {
        key: _clamp_score(value)
        for key, value in (context_fit or {}).items()
        if value is not None
    }
    embedding_text = build_embedding_text(
        metadata=metadata_dict,
        audio_features=audio_features or {},
        tags=tag_rows,
        context_fit=context_scores,
        profile_text=profile_text,
    )
    warnings = _quality_warnings(
        metadata_dict,
        confidence_dict,
        audio_features=audio_features,
        tags=tag_rows,
        missing=missing,
    )
    profile_path = Path(path)
    document: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_version": PROFILE_VERSION,
        "track_id": track_id,
        "generated_at": generated_at,
        "analysis_version": analysis_version,
        # Legacy-compatible top-level fields used by existing tests/debug tooling.
        "path": str(profile_path),
        "metadata": metadata_dict,
        "audio_features": dict(audio_features or {}),
        "tags": tag_rows,
        "identity": {
            "display": {
                "title": metadata_dict.get("title"),
                "artist": metadata_dict.get("artist"),
                "album": metadata_dict.get("album"),
                "album_artist": metadata_dict.get("album_artist"),
                "genre": metadata_dict.get("genre"),
                "release_date": metadata_dict.get("date"),
                "track_number": metadata_dict.get("track_number"),
                "disc_number": metadata_dict.get("disc_number"),
            },
            "normalized": normalized_dict,
            "confidence": confidence_dict,
            "provenance": dict(provenance or {}),
        },
        "external_ids": {},
        "file": {
            "path": str(profile_path),
            "extension": profile_path.suffix.lower(),
            "duration_sec": metadata.get("duration_sec"),
            "codec": metadata.get("codec"),
            "sample_rate": metadata.get("sample_rate"),
            "bit_rate": metadata.get("bit_rate"),
            "channels": metadata.get("channels"),
            "missing": bool(missing),
        },
        "musical": musical,
        "tag_groups": _tag_groups(tag_rows),
        "context_fit": context_scores,
        "search_text": {
            "profile_text": profile_text,
            "embedding_text": embedding_text,
            "fts_boost_terms": _fts_boost_terms(metadata_dict, tag_rows),
        },
        "quality": {
            "metadata_complete": _metadata_complete_score(metadata_dict),
            "audio_features_complete": _audio_features_complete_score(audio_features),
            "tag_coverage": _tag_coverage_score(tag_rows),
            "external_match_confidence": _clamp_score(external_match_confidence),
            "needs_review": bool(warnings),
            "warnings": warnings,
        },
    }
    return _without_none(document)


def build_embedding_text(
    *,
    metadata: Mapping[str, Any],
    audio_features: Mapping[str, Any],
    tags: list[Mapping[str, Any]],
    context_fit: Mapping[str, float] | None = None,
    profile_text: str,
) -> str:
    """Build text optimized for semantic search over profile facts."""
    parts: list[str] = []
    artist_title = " - ".join(
        str(value) for value in [metadata.get("artist"), metadata.get("title")] if value
    )
    if artist_title:
        parts.append(artist_title)
    elif metadata.get("title"):
        parts.append(str(metadata["title"]))
    if metadata.get("album"):
        parts.append(f"Album {metadata['album']}")
    if metadata.get("genre"):
        parts.append(f"Genre {metadata['genre']}")

    tag_terms = [str(tag.get("tag")) for tag in tags[:16] if tag.get("tag")]
    if tag_terms:
        parts.append("Tags " + ", ".join(tag_terms))

    context_terms = [
        context.replace("_", " ")
        for context, score in sorted(
            (context_fit or {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        if score >= 0.55
    ]
    if context_terms:
        parts.append("Fits contexts " + ", ".join(context_terms))

    feature_text = _feature_embedding_text(audio_features)
    if feature_text:
        parts.append(feature_text)

    if not parts and profile_text:
        parts.append(profile_text)
    return ". ".join(part.strip(" .") for part in parts if part).strip() + "."


def profile_json_text(document: Mapping[str, Any]) -> str:
    """Serialize profile JSON deterministically."""
    return json.dumps(document, sort_keys=True)


def profile_source_fingerprint(document: Mapping[str, Any]) -> str:
    """Hash source facts that should trigger profile/embedding refresh."""
    stable = _fingerprint_payload(document)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def embedding_text_from_profile_json(profile_json: str, *, fallback_text: str) -> str:
    """Return optimized embedding text stored inside profile JSON, if present."""
    try:
        document = json.loads(profile_json)
    except json.JSONDecodeError:
        return fallback_text
    value = (
        document.get("search_text", {}).get("embedding_text")
        if isinstance(document, dict)
        else None
    )
    return str(value).strip() if value else fallback_text


def _normalized_identity(
    metadata: Mapping[str, Any],
    normalized: Mapping[str, Any] | None,
) -> dict[str, str | None]:
    return {
        "title_norm": _first_text(normalized, "title_norm")
        or normalize_text(_string(metadata.get("title"))),
        "artist_norm": _first_text(normalized, "artist_norm")
        or normalize_text(_string(metadata.get("artist"))),
        "album_norm": _first_text(normalized, "album_norm")
        or normalize_text(_string(metadata.get("album"))),
        "artist_title_norm": _first_text(normalized, "artist_title_norm")
        or normalize_text(
            " ".join(
                str(value) for value in [metadata.get("artist"), metadata.get("title")] if value
            )
        ),
    }


def _confidence_summary(
    metadata: Mapping[str, Any],
    *,
    metadata_confidence: float | None,
    field_confidence: Mapping[str, float] | None,
) -> dict[str, float]:
    by_field = {key: _clamp_score(value) for key, value in (field_confidence or {}).items()}
    for field in ("title", "artist", "album", "genre"):
        if metadata.get(field) and field not in by_field:
            by_field[field] = _clamp_score(metadata_confidence) or 0.0
    by_field["overall"] = _clamp_score(metadata_confidence)
    return by_field


def _musical_dict(audio_features: Mapping[str, Any]) -> dict[str, Any]:
    if not audio_features:
        return {}
    bpm = _float_or_none(audio_features.get("bpm"))
    key_name = audio_features.get("key_name")
    mode = audio_features.get("mode")
    return _without_none(
        {
            "bpm": {
                "value": bpm,
                "bucket": _bpm_bucket(bpm),
            },
            "key": {
                "name": " ".join(str(part) for part in [key_name, mode] if part) or None,
                "tonic": key_name,
                "mode": mode,
            },
            "energy": _float_or_none(audio_features.get("energy")),
            "valence": _float_or_none(audio_features.get("valence")),
            "danceability": _float_or_none(audio_features.get("danceability")),
            "acousticness": _float_or_none(audio_features.get("acousticness")),
            "instrumentalness": _float_or_none(audio_features.get("instrumentalness")),
            "vocalness": _float_or_none(audio_features.get("vocalness")),
            "speechiness": _float_or_none(audio_features.get("speechiness")),
            "aggression": _float_or_none(audio_features.get("aggression")),
            "brightness": _float_or_none(audio_features.get("brightness")),
            "dynamic_range": _float_or_none(audio_features.get("dynamic_range")),
        }
    )


def _feature_embedding_text(audio_features: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    bpm = _float_or_none(audio_features.get("bpm"))
    if bpm is not None:
        parts.append(f"around {bpm:.0f} BPM")
    for field in ("energy", "danceability", "aggression", "brightness"):
        value = _float_or_none(audio_features.get(field))
        if value is not None:
            parts.append(f"{_band(value)} {field}")
    key_name = audio_features.get("key_name")
    mode = audio_features.get("mode")
    if key_name and mode:
        parts.append(f"rough key {key_name} {mode}")
    return ", ".join(parts) if parts else None


def _tag_dict(tag: Mapping[str, Any]) -> dict[str, Any]:
    return _without_none(
        {
            "source": tag.get("source"),
            "tag": tag.get("tag"),
            "score": _float_or_none(tag.get("score")),
        }
    )


def _tag_groups(tags: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for tag in tags:
        tag_name = str(tag.get("tag") or "")
        if not tag_name:
            continue
        category = _tag_category(tag_name, str(tag.get("source") or ""))
        grouped.setdefault(category, []).append(dict(tag))
    return grouped


def _tag_category(tag: str, source: str) -> str:
    text = f"{source} {tag}".casefold()
    if "genre" in text:
        return "genre"
    if "vocal" in text or "instrumental" in text or "speech" in text:
        return "vocal"
    if any(term in text for term in ["party", "dance", "background", "focus", "workout"]):
        return "function"
    if any(term in text for term in ["guitar", "piano", "drum", "synth", "bass"]):
        return "instrument"
    return "mood"


def _fts_boost_terms(metadata: Mapping[str, Any], tags: list[Mapping[str, Any]]) -> list[str]:
    terms: list[str] = []
    for field in ("artist", "title", "album", "genre"):
        value = metadata.get(field)
        if value:
            terms.append(str(value))
    terms.extend(str(tag.get("tag")) for tag in tags[:10] if tag.get("tag"))
    seen: set[str] = set()
    output: list[str] = []
    for term in terms:
        normalized = normalize_text(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(term)
    return output


def _quality_warnings(
    metadata: Mapping[str, Any],
    confidence: Mapping[str, float],
    *,
    audio_features: Mapping[str, Any] | None,
    tags: list[Mapping[str, Any]],
    missing: bool,
) -> list[str]:
    warnings: list[str] = []
    if missing:
        warnings.append("missing_file")
    if not metadata.get("title"):
        warnings.append("missing_title")
    if not metadata.get("artist"):
        warnings.append("missing_artist")
    if (confidence.get("overall") or 0.0) < 0.60:
        warnings.append("low_metadata_confidence")
    if not audio_features:
        warnings.append("missing_audio_features")
    if not tags:
        warnings.append("missing_tags")
    return warnings


def _metadata_complete_score(metadata: Mapping[str, Any]) -> float:
    fields = ("title", "artist", "album", "genre")
    return round(sum(1 for field in fields if metadata.get(field)) / len(fields), 6)


def _audio_features_complete_score(audio_features: Mapping[str, Any] | None) -> float:
    if not audio_features:
        return 0.0
    return round(
        sum(1 for field in FEATURE_COMPLETENESS_FIELDS if audio_features.get(field) is not None)
        / len(FEATURE_COMPLETENESS_FIELDS),
        6,
    )


def _tag_coverage_score(tags: list[Mapping[str, Any]]) -> float:
    return round(min(1.0, len(tags) / 8.0), 6)


def _bpm_bucket(bpm: float | None) -> str | None:
    if bpm is None:
        return None
    if bpm < 80:
        return "slow"
    if bpm < 110:
        return "midtempo"
    if bpm < 132:
        return "dance"
    if bpm < 165:
        return "fast"
    return "very_fast"


def _band(value: float) -> str:
    if value < 0.33:
        return "low"
    if value < 0.66:
        return "medium"
    return "high"


def _fingerprint_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": document.get("schema_version"),
        "identity": document.get("identity"),
        "file": document.get("file"),
        "musical": document.get("musical"),
        "tags": document.get("tags"),
        "tag_groups": document.get("tag_groups"),
        "context_fit": document.get("context_fit"),
        "search_text": document.get("search_text"),
        "analysis_version": document.get("analysis_version"),
    }


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _first_text(mapping: Mapping[str, Any] | None, key: str) -> str | None:
    if not mapping:
        return None
    value = mapping.get(key)
    return str(value) if value else None


def _string(value: Any) -> str | None:
    return str(value) if value else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_score(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, numeric)), 6)
