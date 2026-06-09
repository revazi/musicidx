"""Search result explanations."""

from __future__ import annotations

from typing import Any


def build_explanation(breakdown: dict[str, Any]) -> list[str]:
    """Create concise explanation lines from a score breakdown."""
    lines: list[str] = []

    semantic_score = breakdown.get("semantic_score")
    if semantic_score is not None and semantic_score > 0:
        lines.append(f"semantic profile similarity {semantic_score:.2f}")

    matched_tags = breakdown.get("matched_tags") or []
    if matched_tags:
        tag_text = ", ".join(
            f"{item['tag']} {item['score']:.2f}" for item in matched_tags[:5]
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

    feedback_score = breakdown.get("feedback_score")
    if feedback_score is not None and feedback_score > 0:
        lines.append(f"positive feedback boost {feedback_score:.2f}")
    elif feedback_score is not None and feedback_score < 0:
        lines.append(f"negative feedback penalty {feedback_score:.2f}")

    if not lines:
        lines.append("included as a local-library candidate")
    return lines
