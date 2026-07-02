"""Search quality evaluation helpers."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from musicidx.search.intent import normalize_tag_terms
from musicidx.search.plan import classify_search_mode
from musicidx.search.ranker import SearchResponse


@dataclass(slots=True)
class EvalQuery:
    id: str
    text: str
    expected_tags: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)
    category: str | None = None
    expectations: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "expected_tags": self.expected_tags,
            "avoid_tags": self.avoid_tags,
            "category": self.category,
            "expectations": self.expectations,
        }


def load_eval_queries(path: Path) -> list[EvalQuery]:
    """Load a JSON eval query file.

    Supported shape:

    ```json
    {"queries": [{"id": "chill_bar", "text": "chill bar"}]}
    ```

    Regression expectations can be provided under `expectations`:

    ```json
    {
      "queries": [
        {
          "id": "highest_bpm_techno",
          "text": "highest BPM techno",
          "expectations": {
            "expected_mode": "feature_sort",
            "must_match_any": ["techno"],
            "top_k": 5,
            "sort_by": {"field": "tempo_bpm", "direction": "desc"}
          }
        }
      ]
    }
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
        text = str(item.get("text") or item.get("query") or "").strip()
        if not text:
            raise ValueError(f"query #{index} is missing text/query")
        query_id = str(item.get("id") or f"query_{index}").strip()
        category = str(item.get("category") or "").strip() or None
        queries.append(
            EvalQuery(
                id=query_id,
                text=text,
                expected_tags=_string_list(item.get("expected_tags")),
                avoid_tags=_string_list(item.get("avoid_tags")),
                category=category,
                expectations=_expectations(item),
            )
        )
    return queries


def evaluate_response(
    conn: sqlite3.Connection,
    eval_query: EvalQuery,
    response: SearchResponse,
) -> dict[str, Any]:
    """Evaluate one search response against subjective and structured expectations."""
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
    structured_checks = _evaluate_structured_expectations(
        eval_query,
        response,
        evidence_by_track,
    )
    structured_failures = [check for check in structured_checks if not check["passed"]]
    structured_pass_count = len(structured_checks) - len(structured_failures)
    structured_pass_rate = (
        structured_pass_count / len(structured_checks) if structured_checks else None
    )
    issue_types = sorted(
        {
            str(check["issue_type"])
            for check in structured_failures
            if check.get("issue_type")
        }
    )

    return {
        "id": eval_query.id,
        "query": eval_query.text,
        "category": eval_query.category,
        "expected_tags": eval_query.expected_tags,
        "avoid_tags": eval_query.avoid_tags,
        "expectations": eval_query.expectations,
        "result_count": result_count,
        "precision_at_k": round(precision, 6),
        "avoid_rate": round(avoid_rate, 6),
        "tag_coverage": round(tag_coverage, 6),
        "diversity_score": round(diversity_score, 6),
        "duplicate_rate": round(duplicate_rate, 6),
        "structured_check_count": len(structured_checks),
        "structured_pass_rate": (
            round(structured_pass_rate, 6) if structured_pass_rate is not None else None
        ),
        "structured_passed": None if not structured_checks else not structured_failures,
        "issue_types": issue_types,
        "structured_checks": structured_checks,
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
            "structured_query_count": 0,
            "avg_structured_pass_rate": None,
            "structured_issue_counts": {},
            "structured_failure_category_counts": {},
        }

    def avg(field: str) -> float:
        return round(sum(float(result[field]) for result in results) / len(results), 6)

    structured_results = [
        result for result in results if int(result.get("structured_check_count") or 0) > 0
    ]
    structured_issue_counts: Counter[str] = Counter()
    structured_failure_category_counts: Counter[str] = Counter()
    for result in results:
        structured_issue_counts.update(str(issue) for issue in result.get("issue_types", []))
        for check in result.get("structured_checks", []):
            if check.get("passed"):
                continue
            category = check.get("failure_category") or check.get("issue_type")
            if category:
                structured_failure_category_counts.update([str(category)])
    avg_structured_pass_rate = None
    if structured_results:
        avg_structured_pass_rate = round(
            sum(float(result["structured_pass_rate"] or 0.0) for result in structured_results)
            / len(structured_results),
            6,
        )

    return {
        "query_count": len(results),
        "avg_precision_at_k": avg("precision_at_k"),
        "avg_avoid_rate": avg("avoid_rate"),
        "avg_tag_coverage": avg("tag_coverage"),
        "avg_diversity_score": avg("diversity_score"),
        "avg_duplicate_rate": avg("duplicate_rate"),
        "structured_query_count": len(structured_results),
        "avg_structured_pass_rate": avg_structured_pass_rate,
        "structured_issue_counts": dict(sorted(structured_issue_counts.items())),
        "structured_failure_category_counts": dict(
            sorted(structured_failure_category_counts.items())
        ),
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


def _evaluate_structured_expectations(
    eval_query: EvalQuery,
    response: SearchResponse,
    evidence_by_track: dict[str, str],
) -> list[dict[str, Any]]:
    expectations = eval_query.expectations
    if not expectations:
        return []

    checks: list[dict[str, Any]] = []
    expected_mode = _optional_string(expectations.get("expected_mode"))
    if expected_mode:
        observed_mode = _observed_mode(response)
        checks.append(
            _check(
                name="expected_mode",
                passed=observed_mode == expected_mode,
                issue_type="intent",
                failure_category="planning",
                expected=expected_mode,
                actual=observed_mode,
            )
        )

    expected_semantic_role = _optional_string(expectations.get("semantic_role"))
    if expected_semantic_role:
        actual_semantic_role = _optional_string(
            response.diagnostics.get("semantic_role") if response.diagnostics else None
        )
        checks.append(
            _check(
                name="semantic_role",
                passed=actual_semantic_role == expected_semantic_role,
                issue_type="intent",
                failure_category="planning",
                expected=expected_semantic_role,
                actual=actual_semantic_role,
            )
        )

    sort_expectation = _sort_expectation(expectations.get("sort_by"))
    if sort_expectation:
        field = sort_expectation["field"]
        direction = sort_expectation["direction"]
        actual_sort = [sort_spec.as_dict() for sort_spec in response.intent.sort_by]
        intent_has_sort = any(
            sort_spec.field == field and sort_spec.direction == direction
            for sort_spec in response.intent.sort_by
        )
        checks.append(
            _check(
                name="intent_sort_by",
                passed=intent_has_sort,
                issue_type="intent",
                failure_category="planning",
                expected=sort_expectation,
                actual=actual_sort,
            )
        )
        values = [_result_sort_value(result, field) for result in response.results]
        missing_ranks = [index for index, value in enumerate(values, start=1) if value is None]
        checks.append(
            _check(
                name="sort_feature_present",
                passed=not missing_ranks,
                issue_type="feature_quality",
                expected={"field": field, "missing_ranks": []},
                actual={"field": field, "missing_ranks": missing_ranks},
            )
        )
        comparable_values = [value for value in values if value is not None]
        checks.append(
            _check(
                name="result_sort_order",
                passed=_is_sorted(comparable_values, direction),
                issue_type="sort",
                expected={"field": field, "direction": direction},
                actual={"values": comparable_values},
            )
        )

    top_k = _top_k(expectations.get("top_k"), fallback=len(response.results))
    must_match_any = _string_list(expectations.get("must_match_any"))
    if must_match_any:
        failures = _content_match_failures(
            response,
            evidence_by_track,
            concepts=must_match_any,
            mode="any",
            top_k=top_k,
        )
        checks.append(
            _check(
                name="must_match_any",
                passed=not failures and bool(response.results),
                issue_type="retrieval" if not response.results else "ranking",
                expected={"concepts": must_match_any, "top_k": top_k},
                actual={"failed_results": failures},
            )
        )

    must_match_all = _string_list(expectations.get("must_match_all"))
    if must_match_all:
        failures = _content_match_failures(
            response,
            evidence_by_track,
            concepts=must_match_all,
            mode="all",
            top_k=top_k,
        )
        checks.append(
            _check(
                name="must_match_all",
                passed=not failures and bool(response.results),
                issue_type="retrieval" if not response.results else "ranking",
                expected={"concepts": must_match_all, "top_k": top_k},
                actual={"failed_results": failures},
            )
        )

    must_not_top = _string_list(expectations.get("must_not_top"))
    if must_not_top:
        top_n = _top_k(expectations.get("top_n"), fallback=1)
        failures = _identity_forbidden_failures(response, must_not_top, top_n=top_n)
        checks.append(
            _check(
                name="must_not_top",
                passed=not failures,
                issue_type="ranking",
                expected={"forbidden": must_not_top, "top_n": top_n},
                actual={"failed_results": failures},
            )
        )

    must_not_warnings_top = _string_list(expectations.get("must_not_warnings_top"))
    if must_not_warnings_top:
        top_n = _top_k(expectations.get("top_n"), fallback=1)
        failures = _warning_forbidden_failures(
            response,
            must_not_warnings_top,
            top_n=top_n,
        )
        checks.append(
            _check(
                name="must_not_warnings_top",
                passed=not failures,
                issue_type="ranking",
                expected={"forbidden": must_not_warnings_top, "top_n": top_n},
                actual={"failed_results": failures},
            )
        )

    rank_reason_primary = _optional_string(expectations.get("rank_reason_primary_top"))
    if rank_reason_primary:
        top_n = _top_k(expectations.get("top_n"), fallback=1)
        failures = _rank_reason_primary_failures(
            response,
            rank_reason_primary,
            top_n=top_n,
        )
        checks.append(
            _check(
                name="rank_reason_primary_top",
                passed=not failures and bool(response.results),
                issue_type="ranking",
                expected={"primary": rank_reason_primary, "top_n": top_n},
                actual={"failed_results": failures},
            )
        )

    must_have_evidence_sources = _string_list(
        expectations.get("must_have_evidence_sources_top")
    )
    if must_have_evidence_sources:
        top_n = _top_k(expectations.get("top_n"), fallback=1)
        failures = _evidence_source_failures(
            response,
            must_have_evidence_sources,
            top_n=top_n,
        )
        checks.append(
            _check(
                name="must_have_evidence_sources_top",
                passed=not failures and bool(response.results),
                issue_type="retrieval" if not response.results else "ranking",
                expected={"sources": must_have_evidence_sources, "top_n": top_n},
                actual={"failed_results": failures},
            )
        )

    return checks


def _concept_matches_text(concept: str, text: str) -> bool:
    concept_terms = set(normalize_tag_terms(concept))
    if not concept_terms:
        return False
    text_terms = set(normalize_tag_terms(text))
    concept_text = " ".join(sorted(concept_terms))
    text_text = " ".join(sorted(text_terms))
    return concept_terms.issubset(text_terms) or concept_text in text_text


def _content_match_failures(
    response: SearchResponse,
    evidence_by_track: dict[str, str],
    *,
    concepts: list[str],
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for rank, result in enumerate(response.results[:top_k], start=1):
        evidence = evidence_by_track.get(result.track_id, "")
        matched = [concept for concept in concepts if _concept_matches_text(concept, evidence)]
        passed = bool(matched) if mode == "any" else len(matched) == len(concepts)
        if passed:
            continue
        failures.append(
            {
                "rank": rank,
                "track_id": result.track_id,
                "title": result.title,
                "artist": result.artist,
                "matched": matched,
            }
        )
    return failures


def _identity_forbidden_failures(
    response: SearchResponse,
    forbidden: list[str],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    normalized_forbidden = [
        " ".join(normalize_tag_terms(item)) for item in forbidden if normalize_tag_terms(item)
    ]
    for rank, result in enumerate(response.results[:top_n], start=1):
        identity = " ".join(
            part
            for part in [result.track_id, result.artist or "", result.title or ""]
            if part
        )
        normalized_identity = " ".join(normalize_tag_terms(identity))
        matched = [item for item in normalized_forbidden if item in normalized_identity]
        if not matched:
            continue
        failures.append(
            {
                "rank": rank,
                "track_id": result.track_id,
                "title": result.title,
                "artist": result.artist,
                "matched": matched,
            }
        )
    return failures


def _warning_forbidden_failures(
    response: SearchResponse,
    forbidden: list[str],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    forbidden_set = {warning.strip().lower() for warning in forbidden if warning.strip()}
    failures: list[dict[str, Any]] = []
    for rank, result in enumerate(response.results[:top_n], start=1):
        warnings = _result_warning_codes(result)
        matched = sorted(forbidden_set.intersection(warnings))
        if not matched:
            continue
        failures.append(
            {
                "rank": rank,
                "track_id": result.track_id,
                "title": result.title,
                "artist": result.artist,
                "matched_warnings": matched,
            }
        )
    return failures


def _rank_reason_primary_failures(
    response: SearchResponse,
    expected_primary: str,
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for rank, result in enumerate(response.results[:top_n], start=1):
        rank_reason = result.breakdown.get("rank_reason") or {}
        actual = str(rank_reason.get("primary") or "").strip().lower()
        if actual == expected_primary:
            continue
        failures.append(
            {
                "rank": rank,
                "track_id": result.track_id,
                "title": result.title,
                "artist": result.artist,
                "actual_primary": actual,
            }
        )
    return failures


def _evidence_source_failures(
    response: SearchResponse,
    expected_sources: list[str],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    expected = {source.strip().lower() for source in expected_sources if source.strip()}
    failures: list[dict[str, Any]] = []
    for rank, result in enumerate(response.results[:top_n], start=1):
        candidate_evidence = result.breakdown.get("candidate_evidence") or {}
        actual = {
            str(source).strip().lower()
            for source in candidate_evidence.get("retrieved_by") or []
            if str(source).strip()
        }
        missing = sorted(expected.difference(actual))
        if not missing:
            continue
        failures.append(
            {
                "rank": rank,
                "track_id": result.track_id,
                "title": result.title,
                "artist": result.artist,
                "missing_sources": missing,
                "actual_sources": sorted(actual),
            }
        )
    return failures


def _result_warning_codes(result: Any) -> set[str]:
    breakdown = result.breakdown or {}
    warnings = {str(item).strip().lower() for item in breakdown.get("confidence_warnings") or []}
    evidence = breakdown.get("evidence") or {}
    if evidence.get("semantic_only"):
        warnings.add("semantic_only")
        warnings.add("semantic_only_match")
    if "weak_score" in warnings:
        warnings.add("weak_match")
    return {warning for warning in warnings if warning}


def _observed_mode(response: SearchResponse) -> str:
    return classify_search_mode(response.intent)


def _sort_expectation(value: Any) -> dict[str, str] | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, dict):
        return None
    field_name = _optional_string(value.get("field"))
    direction = _optional_string(value.get("direction"))
    if field_name in {"bpm", "tempo"}:
        field_name = "tempo_bpm"
    if field_name == "dance":
        field_name = "danceability"
    if not field_name or direction not in {"asc", "desc"}:
        return None
    return {"field": field_name, "direction": direction}


def _result_sort_value(result: Any, field_name: str) -> float | None:
    sort_values = result.breakdown.get("sort_values")
    if not isinstance(sort_values, dict):
        return None
    return _float_or_none(sort_values.get(field_name))


def _is_sorted(values: list[float], direction: str) -> bool:
    if len(values) <= 1:
        return True
    tolerance = 1e-9
    if direction == "asc":
        return all(
            left <= right + tolerance for left, right in zip(values, values[1:], strict=False)
        )
    return all(
        left + tolerance >= right for left, right in zip(values, values[1:], strict=False)
    )


def _check(
    *,
    name: str,
    passed: bool,
    issue_type: str,
    expected: Any,
    actual: Any,
    failure_category: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "issue_type": None if passed else issue_type,
        "failure_category": None if passed else (failure_category or issue_type),
        "expected": expected,
        "actual": actual,
    }


def _expectations(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("expectations")
    output = dict(value) if isinstance(value, dict) else {}
    for key in (
        "expected_mode",
        "must_match_any",
        "must_match_all",
        "must_not_top",
        "must_not_warnings_top",
        "rank_reason_primary_top",
        "must_have_evidence_sources_top",
        "sort_by",
        "semantic_role",
        "top_k",
        "top_n",
    ):
        if key in item and key not in output:
            output[key] = item[key]
    return output


def _top_k(value: Any, *, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(1, min(100, parsed))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if str(item).strip()]


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
