"""Search result explanations."""

from __future__ import annotations

from typing import Any


def build_explanation(breakdown: dict[str, Any]) -> list[str]:
    """Create concise explanation lines from a score breakdown."""
    lines: list[str] = []

    semantic_score = breakdown.get("semantic_score")
    evidence = breakdown.get("evidence") or {}
    confidence = breakdown.get("confidence")
    confidence_warnings = breakdown.get("confidence_warnings") or []
    if semantic_score is not None and semantic_score > 0:
        lines.append(f"semantic profile similarity {semantic_score:.2f}")
    if evidence.get("semantic_only"):
        lines.append("semantic-only match; no metadata, tag, context, feature, or text evidence")
    if confidence == "low" and confidence_warnings:
        labels = ", ".join(str(item).replace("_", " ") for item in confidence_warnings[:3])
        lines.append(f"low confidence: {labels}")

    metadata_matches = breakdown.get("metadata_matches") or []
    if metadata_matches:
        summary = ", ".join(
            f"{item['field']}={item['value']}" for item in metadata_matches[:3]
        )
        lines.append(f"metadata match: {summary}")

    matched_contexts = [
        item
        for item in (breakdown.get("matched_contexts") or [])
        if float(item.get("score") or 0.0) >= 0.55
        or float(item.get("confidence") or 0.0) >= 0.60
    ]
    if matched_contexts:
        context_text = ", ".join(
            f"{item['context'].replace('_', ' ')} {item['score']:.2f}"
            for item in matched_contexts[:3]
        )
        lines.append(f"context fit: {context_text}")

    matched_tags = breakdown.get("matched_tags") or []
    if matched_tags:
        tag_text = ", ".join(
            f"{item['tag']} {item['score']:.2f}" for item in matched_tags[:4]
        )
        lines.append(f"matched tags: {tag_text}")

    avoided_tags = breakdown.get("avoided_tags") or []
    if avoided_tags:
        tag_text = ", ".join(
            f"{item['tag']} {item['score']:.2f}" for item in avoided_tags[:3]
        )
        lines.append(f"penalized avoided tags: {tag_text}")

    feature_reasons = breakdown.get("feature_reasons") or []
    lines.extend(feature_reasons[:5])

    text_score = breakdown.get("text_score")
    if text_score is not None and text_score > 0:
        lines.append(f"text/profile match {text_score:.2f}")

    sort_lines = _sort_explanations(breakdown)
    lines.extend(sort_lines)

    feedback_score = breakdown.get("feedback_score")
    if feedback_score is not None and feedback_score > 0:
        lines.append(f"positive feedback boost {feedback_score:.2f}")
    elif feedback_score is not None and feedback_score < 0:
        lines.append(f"negative feedback penalty {feedback_score:.2f}")

    if not lines:
        lines.append("included as a local-library candidate")
    return lines


def _sort_explanations(breakdown: dict[str, Any]) -> list[str]:
    sort_specs = breakdown.get("sort_by") or []
    sort_values = breakdown.get("sort_values") or {}
    output: list[str] = []
    labels = {
        "tempo_bpm": "BPM",
        "energy": "energy",
        "danceability": "danceability",
        "aggression": "aggression",
        "brightness": "brightness",
    }
    for spec in sort_specs[:2]:
        if not isinstance(spec, dict):
            continue
        field = spec.get("field")
        direction = spec.get("direction")
        if not isinstance(field, str) or not isinstance(direction, str):
            continue
        value = sort_values.get(field)
        label = labels.get(field, field)
        descriptor = "highest" if direction == "desc" else "lowest"
        if isinstance(value, int | float):
            output.append(f"sorted by {descriptor} {label}: {value:.2f}")
        else:
            output.append(f"sorted by {descriptor} {label}")
    return output
