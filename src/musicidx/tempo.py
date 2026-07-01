"""Tempo normalization helpers.

Audio beat trackers often return double-time or half-time tempo estimates. MusicIdx keeps the
stored feature simple, but search/profile layers should prefer a human/perceived tempo for broad
queries such as "high BPM" unless a track appears to be from a genre where 170+ BPM is expected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

HIGH_TEMPO_STYLE_TERMS = {
    "breakcore",
    "dnb",
    "drum and bass",
    "drum n bass",
    "drum bass",
    "drum&bass",
    "gabber",
    "happy hardcore",
    "hardcore",
    "jungle",
    "speedcore",
}


def perceived_tempo_bpm(
    bpm: Any,
    *,
    descriptors: Iterable[str | None] | None = None,
) -> float | None:
    """Return a human/perceived BPM for broad search and profile text.

    Examples:
    - 172 BPM rock/pop double-time -> 86 BPM.
    - 174 BPM drum-and-bass/jungle -> 174 BPM when descriptors say so.
    - 48 BPM half-time estimate -> 96 BPM.
    """
    value = _float_or_none(bpm)
    if value is None or value <= 0:
        return None

    if value >= 168.0 and not _has_high_tempo_style(descriptors):
        value = value / 2.0
    elif value < 62.0:
        value = value * 2.0

    return round(value, 6)


def raw_and_perceived_tempo(
    bpm: Any,
    *,
    descriptors: Iterable[str | None] | None = None,
) -> dict[str, float | bool | None]:
    """Return raw/perceived tempo info for diagnostics/profile JSON."""
    raw = _float_or_none(bpm)
    perceived = perceived_tempo_bpm(raw, descriptors=descriptors)
    return {
        "raw": raw,
        "perceived": perceived,
        "adjusted": raw is not None and perceived is not None and abs(raw - perceived) > 0.01,
    }


def tempo_descriptors_from_metadata_and_tags(
    metadata: Mapping[str, Any] | None,
    tags: Iterable[Mapping[str, Any]] | None,
) -> list[str]:
    """Collect metadata/tag text used to decide whether high raw BPM is expected."""
    descriptors: list[str] = []
    for field in ("genre", "title", "album", "artist"):
        value = (metadata or {}).get(field)
        if value:
            descriptors.append(str(value))
    for tag in tags or []:
        for key in ("tag", "source"):
            value = tag.get(key)
            if value:
                descriptors.append(str(value))
    return descriptors


def _has_high_tempo_style(descriptors: Iterable[str | None] | None) -> bool:
    if not descriptors:
        return False
    text = " ".join(str(item).casefold().replace("---", " ") for item in descriptors if item)
    text = text.replace("&", " and ").replace("-", " ").replace("_", " ")
    compact = " ".join(text.split())
    return any(term in compact for term in HIGH_TEMPO_STYLE_TERMS)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None
