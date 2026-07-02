"""Repeatable evaluation helpers for deterministic MatchReport behavior."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from musicidx.search.matching import compare_tracks


@dataclass(slots=True)
class MatchEvalCase:
    id: str
    track_a: str
    track_b: str
    category: str | None = None
    expectations: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "track_a": self.track_a,
            "track_b": self.track_b,
            "category": self.category,
            "expectations": self.expectations,
        }


def load_match_eval_cases(path: Path) -> list[MatchEvalCase]:
    """Load a JSON match-eval file.

    Supported shape:

    ```json
    {"matches": [{"id": "same", "track_a": "a", "track_b": "b"}]}
    ```
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read match eval file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON match eval file: {exc}") from exc

    raw_cases = data.get("matches") if isinstance(data, dict) else None
    if not isinstance(raw_cases, list):
        raise ValueError("match eval file must contain a JSON object with a 'matches' list")

    cases: list[MatchEvalCase] = []
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"match #{index} must be an object")
        track_a = str(item.get("track_a") or item.get("track_a_id") or "").strip()
        track_b = str(item.get("track_b") or item.get("track_b_id") or "").strip()
        if not track_a or not track_b:
            raise ValueError(f"match #{index} is missing track_a/track_b")
        case_id = str(item.get("id") or f"match_{index}").strip()
        category = str(item.get("category") or "").strip() or None
        expectations = item.get("expectations") or {}
        if not isinstance(expectations, dict):
            raise ValueError(f"match #{index} expectations must be an object")
        cases.append(
            MatchEvalCase(
                id=case_id,
                track_a=track_a,
                track_b=track_b,
                category=category,
                expectations=expectations,
            )
        )
    return cases


def evaluate_match_case(
    conn: sqlite3.Connection,
    case: MatchEvalCase,
    *,
    duration_tolerance_sec: float = 3.0,
) -> dict[str, Any]:
    """Compare one pair of tracks and check expected MatchReport fields."""
    report = compare_tracks(
        conn,
        case.track_a,
        case.track_b,
        duration_tolerance_sec=duration_tolerance_sec,
    )
    checks = _structured_checks(case.expectations, report.as_dict())
    failures = [check for check in checks if not check["passed"]]
    pass_rate = (len(checks) - len(failures)) / len(checks) if checks else None
    issue_types = sorted(
        {
            str(check["issue_type"])
            for check in failures
            if check.get("issue_type")
        }
    )
    return {
        "id": case.id,
        "category": case.category,
        "track_a": case.track_a,
        "track_b": case.track_b,
        "expectations": case.expectations,
        "decision": report.decision,
        "identity_decision": report.identity_decision,
        "confidence": report.confidence,
        "confidence_score": report.confidence_score,
        "candidate_score": report.candidate_score,
        "candidate_kind": report.candidate_kind,
        "candidate_strength": report.candidate_strength,
        "candidate_summary": report.candidate_summary,
        "candidate_reasons": report.candidate_reasons,
        "structured_check_count": len(checks),
        "structured_pass_rate": round(pass_rate, 6) if pass_rate is not None else None,
        "structured_passed": None if not checks else not failures,
        "issue_types": issue_types,
        "structured_checks": checks,
        "report": report.as_dict(),
    }


def aggregate_match_eval_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-pair match eval metrics."""
    if not results:
        return {
            "match_count": 0,
            "structured_match_count": 0,
            "avg_structured_pass_rate": None,
            "structured_issue_counts": {},
        }
    structured = [
        result for result in results if int(result.get("structured_check_count") or 0) > 0
    ]
    issue_counts: Counter[str] = Counter()
    for result in results:
        issue_counts.update(str(issue) for issue in result.get("issue_types", []))
    avg_pass_rate = None
    if structured:
        avg_pass_rate = round(
            sum(float(result["structured_pass_rate"] or 0.0) for result in structured)
            / len(structured),
            6,
        )
    return {
        "match_count": len(results),
        "structured_match_count": len(structured),
        "avg_structured_pass_rate": avg_pass_rate,
        "structured_issue_counts": dict(sorted(issue_counts.items())),
    }


def _structured_checks(
    expectations: dict[str, Any],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key, issue_type in [
        ("decision", "matching_decision"),
        ("identity_decision", "matching_decision"),
        ("confidence", "matching_confidence"),
    ]:
        if key in expectations:
            checks.append(
                _check(
                    name=key,
                    passed=report.get(key) == expectations[key],
                    issue_type=issue_type,
                    expected=expectations[key],
                    actual=report.get(key),
                )
            )
    if "min_confidence_score" in expectations:
        expected = float(expectations["min_confidence_score"])
        actual = float(report.get("confidence_score") or 0.0)
        checks.append(
            _check(
                name="min_confidence_score",
                passed=actual >= expected,
                issue_type="matching_confidence",
                expected=expected,
                actual=actual,
            )
        )
    if "must_have_evidence" in expectations:
        expected = _string_list(expectations["must_have_evidence"])
        actual = {item.get("source") for item in report.get("evidence") or []}
        missing = [source for source in expected if source not in actual]
        checks.append(
            _check(
                name="must_have_evidence",
                passed=not missing,
                issue_type="matching_evidence",
                expected=expected,
                actual={"missing": missing, "sources": sorted(str(item) for item in actual)},
            )
        )
    if "must_have_warnings" in expectations:
        expected = _string_list(expectations["must_have_warnings"])
        actual = {str(item) for item in report.get("warnings") or []}
        missing = [warning for warning in expected if warning not in actual]
        checks.append(
            _check(
                name="must_have_warnings",
                passed=not missing,
                issue_type="matching_policy",
                expected=expected,
                actual={"missing": missing, "warnings": sorted(actual)},
            )
        )
    if "must_not_decide_identity" in expectations:
        must_not = bool(expectations["must_not_decide_identity"])
        actual = report.get("identity_decision")
        checks.append(
            _check(
                name="must_not_decide_identity",
                passed=(not must_not) or actual != "same",
                issue_type="matching_policy",
                expected={"identity_decision_not": "same"},
                actual=actual,
            )
        )
    return checks


def _check(
    *,
    name: str,
    passed: bool,
    issue_type: str,
    expected: Any,
    actual: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "issue_type": None if passed else issue_type,
        "expected": expected,
        "actual": actual,
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
