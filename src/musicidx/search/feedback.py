"""Feedback persistence helpers for search evaluation."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from musicidx.db import utc_now


def save_search_event(conn: sqlite3.Connection, response: Any) -> str:
    """Persist a search event and return its ID."""
    event_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO search_events (id, query, parsed_intent_json, result_track_ids_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            event_id,
            response.query,
            json.dumps(response.intent.as_dict(), sort_keys=True),
            json.dumps([result.track_id for result in response.results]),
            utc_now(),
        ),
    )
    conn.commit()
    return event_id


def save_feedback_event(
    conn: sqlite3.Connection,
    *,
    query: str,
    track_ids: list[str] | None = None,
) -> str:
    """Persist a lightweight search event for non-interactive feedback."""
    event_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO search_events (id, query, parsed_intent_json, result_track_ids_json, created_at)
        VALUES (?, ?, NULL, ?, ?)
        """,
        (event_id, query, json.dumps(track_ids or []), utc_now()),
    )
    conn.commit()
    return event_id


def save_track_feedback(
    conn: sqlite3.Connection,
    *,
    search_event_id: str | None,
    track_id: str,
    rating: int,
    note: str | None = None,
) -> str:
    """Persist a single track judgment.

    Ratings are intentionally small integers:
    - `1` good match
    - `0` neutral/skip
    - `-1` bad match
    """
    feedback_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO feedback (id, search_event_id, track_id, rating, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (feedback_id, search_event_id, track_id, max(-1, min(1, int(rating))), note, utc_now()),
    )
    conn.commit()
    return feedback_id


def feedback_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return compact feedback counts for diagnostics/UI use."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) AS negative,
            SUM(CASE WHEN rating = 0 THEN 1 ELSE 0 END) AS neutral,
            COUNT(DISTINCT track_id) AS tracks,
            COUNT(DISTINCT search_event_id) AS search_events
        FROM feedback
        """
    ).fetchone()
    return {
        "total": int(row["total"] or 0),
        "positive": int(row["positive"] or 0),
        "negative": int(row["negative"] or 0),
        "neutral": int(row["neutral"] or 0),
        "tracks": int(row["tracks"] or 0),
        "search_events": int(row["search_events"] or 0),
    }
