"""Search quality evaluation helpers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from musicidx.search.intent import normalize_tag_terms
from musicidx.search.ranker import SearchResponse


@dataclass(slots=True)
class EvalQuery:
    id: str
    text: str
    expected_tags: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "expected_tags": self.expected_tags,
            "avoid_tags": self.avoid_tags,
        }


def load_eval_queries(path: Path) -> list[EvalQuery]:
    """Load a JSON eval query file.

    Supported shape:

    ```json
    {"queries": [{"id": "chill_bar", "text": "chill bar"}]}
    ```
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read eval file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON eval file: {exc}") from exc

    raw_queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(raw_queries, list):
        raise ValueError("eval file must contain a JSON object with a 'queries' list")

    queries: list[EvalQuery] = []
    for index, item in enumerate(raw_queries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"query #{index} must be an object")
        text = str(item.get("text") or "").strip()
        if not text:
            raise ValueError(f"query #{index} is missing text")
        query_id = str(item.get("id") or f"query_{index}").strip()
        queries.append(
            EvalQuery(
                id=query_id,
                text=text,
                expected_tags=_string_list(item.get("expected_tags")),
                avoid_tags=_string_list(item.get("avoid_tags")),
            )
        )
    return queries


def evaluate_response(
    conn: sqlite3.Connection,
    eval_query: EvalQuery,
    response: SearchResponse,
) -> dict[str, Any]:
    """Evaluate one search response against subjective tag expectations."""
    result_ids = [result.track_id for result in response.results]
    evidence_by_track = _track_evidence(conn, result_ids)
    expected_seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    good_count = 0
    avoid_count = 0
    artist_keys: set[str] = set()

    for rank, result in enumerate(response.results, start=1):
        evidence = evidence_by_track.get(result.track_id, "")
        matched_expected = [
            tag for tag in eval_query.expected_tags if _concept_matches_text(tag, evidence)
        ]
        matched_avoid = [
            tag for tag in eval_query.avoid_tags if _concept_matches_text(tag, evidence)
        ]
        expected_seen.update(matched_expected)
        is_good = bool(matched_expected) if eval_query.expected_tags else not matched_avoid
        is_bad = bool(matched_avoid)
        if is_good and not is_bad:
            good_count += 1
        if is_bad:
            avoid_count += 1
        artist_key = (
            result.artist.strip().lower() if result.artist else f"track:{result.track_id}"
        )
        artist_keys.add(artist_key)
        rows.append(
            {
                "rank": rank,
                "track_id": result.track_id,
                "title": result.title,
                "artist": result.artist,
                "score": result.score,
                "matched_expected": matched_expected,
                "matched_avoid": matched_avoid,
            }
        )

    result_count = len(response.results)
    expected_count = len(eval_query.expected_tags)
    precision = good_count / result_count if result_count else 0.0
    avoid_rate = avoid_count / result_count if result_count else 0.0
    tag_coverage = len(expected_seen) / expected_count if expected_count else 1.0
    diversity_score = len(artist_keys) / result_count if result_count else 0.0
    duplicate_rate = 1.0 - diversity_score if result_count else 0.0

    return {
        "id": eval_query.id,
        "query": eval_query.text,
        "expected_tags": eval_query.expected_tags,
        "avoid_tags": eval_query.avoid_tags,
        "result_count": result_count,
        "precision_at_k": round(precision, 6),
        "avoid_rate": round(avoid_rate, 6),
        "tag_coverage": round(tag_coverage, 6),
        "diversity_score": round(diversity_score, 6),
        "duplicate_rate": round(duplicate_rate, 6),
        "top_results": rows,
    }


def aggregate_eval_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query eval metrics."""
    if not results:
        return {
            "query_count": 0,
            "avg_precision_at_k": 0.0,
            "avg_avoid_rate": 0.0,
            "avg_tag_coverage": 0.0,
            "avg_diversity_score": 0.0,
            "avg_duplicate_rate": 0.0,
        }

    def avg(field: str) -> float:
        return round(sum(float(result[field]) for result in results) / len(results), 6)

    return {
        "query_count": len(results),
        "avg_precision_at_k": avg("precision_at_k"),
        "avg_avoid_rate": avg("avoid_rate"),
        "avg_tag_coverage": avg("tag_coverage"),
        "avg_diversity_score": avg("diversity_score"),
        "avg_duplicate_rate": avg("duplicate_rate"),
    }


def _track_evidence(conn: sqlite3.Connection, track_ids: list[str]) -> dict[str, str]:
    if not track_ids:
        return {}
    placeholders = ",".join("?" for _ in track_ids)
    rows = conn.execute(
        f"""
        SELECT
            t.id AS track_id,
            t.title,
            t.artist,
            t.album,
            t.genre,
            p.profile_text,
            GROUP_CONCAT(tt.tag, ' ') AS tags
        FROM tracks t
        LEFT JOIN track_profiles p ON p.track_id = t.id
        LEFT JOIN track_tags tt ON tt.track_id = t.id
        WHERE t.id IN ({placeholders})
        GROUP BY t.id
        """,
        track_ids,
    ).fetchall()
    output: dict[str, str] = {}
    for row in rows:
        values = [
            row["title"],
            row["artist"],
            row["album"],
            row["genre"],
            row["profile_text"],
            row["tags"],
        ]
        output[row["track_id"]] = " ".join(str(value) for value in values if value).lower()
    return output


def _concept_matches_text(concept: str, text: str) -> bool:
    concept_terms = set(normalize_tag_terms(concept))
    if not concept_terms:
        return False
    text_terms = set(normalize_tag_terms(text))
    concept_text = " ".join(sorted(concept_terms))
    text_text = " ".join(sorted(text_terms))
    return concept_terms.issubset(text_terms) or concept_text in text_text


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if str(item).strip()]
