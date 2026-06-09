from __future__ import annotations

import json

from typer.testing import CliRunner

from musicidx.cli import _search_payload, app
from musicidx.db import connect_db, init_db
from musicidx.search.intent import parse_intent_dynamic
from musicidx.search.ranker import search_music


def test_parse_intent_uses_library_tags_and_feature_ranges(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "chill-track",
            tmp_path / "chill.mp3",
            profile_text="relaxing ambient background",
            tags=[
                ("essentia:mood", "relaxing", 0.9),
                ("essentia:genre", "electronic---ambient", 0.7),
            ],
            energy=0.30,
            aggression=0.10,
            danceability=0.50,
            bpm=90.0,
        )

        intent = parse_intent_dynamic("Give me 5 tracks for a chill bar", conn)

        assert intent.limit == 5
        assert "chill" in intent.contexts
        assert "bar" in intent.contexts
        assert "relaxing" in intent.prefer_tags
        assert "electronic---ambient" in intent.prefer_tags
        assert "energy" in intent.feature_ranges
        assert "aggression" in intent.feature_ranges
    finally:
        conn.close()


def test_search_music_ranks_library_aware_chill_match_first(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "chill-track",
            tmp_path / "chill.mp3",
            title="Calm Room",
            artist="Artist A",
            profile_text="relaxing ambient background downtempo",
            tags=[
                ("essentia:mood", "relaxing", 0.9),
                ("essentia:genre", "electronic---ambient", 0.7),
            ],
            energy=0.30,
            aggression=0.10,
            danceability=0.50,
            bpm=90.0,
        )
        _insert_track(
            conn,
            "party-track",
            tmp_path / "party.mp3",
            title="Big Night",
            artist="Artist B",
            profile_text="party energetic fast dance",
            tags=[("essentia:mood", "party", 0.9), ("essentia:mood", "energetic", 0.8)],
            energy=0.90,
            aggression=0.70,
            danceability=0.95,
            bpm=135.0,
        )

        response = search_music(conn, "chill bar", explain=True)

        assert response.results[0].track_id == "chill-track"
        assert response.results[0].score > response.results[1].score
        assert response.results[0].explanation

        payload = _search_payload(response, db_path="index.sqlite", concise=True)
        assert payload["db_path"] == "index.sqlite"
        assert payload["intent"]["contexts"] == ["chill", "bar"]
        assert "library_profile" not in payload["intent"]
        assert payload["results"][0]["track_id"] == "chill-track"
        assert "breakdown" not in payload["results"][0]
        assert payload["results"][0]["why"]
    finally:
        conn.close()


def test_export_command_writes_m3u_playlist(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        track_path = tmp_path / "chill.mp3"
        _insert_track(
            conn,
            "chill-track",
            track_path,
            title="Calm Room",
            artist="Artist A",
            profile_text="relaxing ambient background downtempo",
            tags=[("essentia:mood", "relaxing", 0.9)],
            energy=0.30,
            aggression=0.10,
            danceability=0.50,
            bpm=90.0,
        )
    finally:
        conn.close()

    out = tmp_path / "exports" / "chill.m3u"
    result = CliRunner().invoke(
        app,
        [
            "export",
            "chill bar",
            "--db",
            str(db_path),
            "--out",
            str(out),
            "--limit",
            "1",
            "--absolute-paths",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    content = out.read_text()
    assert content.startswith("#EXTM3U")
    assert "#EXTINF:-1,Artist A - Calm Room" in content
    assert str(track_path.resolve()) in content


def test_eval_command_reports_search_quality_metrics(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(
            conn,
            "chill-track",
            tmp_path / "chill.mp3",
            title="Calm Room",
            artist="Artist A",
            profile_text="relaxing ambient background downtempo",
            tags=[("essentia:mood", "relaxing", 0.9)],
            energy=0.30,
            aggression=0.10,
            danceability=0.50,
            bpm=90.0,
        )
        _insert_track(
            conn,
            "party-track",
            tmp_path / "party.mp3",
            title="Big Night",
            artist="Artist B",
            profile_text="party energetic fast dance",
            tags=[("essentia:mood", "party", 0.9)],
            energy=0.90,
            aggression=0.70,
            danceability=0.95,
            bpm=135.0,
        )
    finally:
        conn.close()

    eval_file = tmp_path / "queries.json"
    eval_file.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "chill_bar",
                        "text": "chill bar",
                        "expected_tags": ["relaxing"],
                        "avoid_tags": ["party"],
                    }
                ]
            }
        )
    )

    result = CliRunner().invoke(
        app,
        ["eval", str(eval_file), "--db", str(db_path), "--limit", "2", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["query_count"] == 1
    assert payload["results"][0]["precision_at_k"] == 0.5
    assert payload["results"][0]["avoid_rate"] == 0.5
    assert payload["results"][0]["tag_coverage"] == 1.0


def test_feedback_command_persists_query_aware_feedback(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(
            conn,
            "track-1",
            tmp_path / "track.mp3",
            title="Feedback Track",
            artist="Artist A",
            profile_text="relaxing ambient background",
            tags=[("essentia:mood", "relaxing", 0.5)],
            energy=0.30,
            aggression=0.10,
            danceability=0.50,
            bpm=90.0,
        )
    finally:
        conn.close()

    result = CliRunner().invoke(
        app,
        [
            "feedback",
            "--db",
            str(db_path),
            "--track-id",
            "track-1",
            "--query",
            "chill",
            "--rating",
            "good",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["track_id"] == "track-1"
    assert payload["rating"] == 1
    assert payload["search_event_id"]
    conn = connect_db(db_path)
    try:
        response = search_music(conn, "chill", limit=1, explain=True)
        assert response.results[0].breakdown["feedback_score"] == 1.0
    finally:
        conn.close()


def test_judge_command_persists_feedback_and_feedback_affects_ranking(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(
            conn,
            "bad-track",
            tmp_path / "a-bad.mp3",
            title="Same Score Bad",
            artist="Artist A",
            profile_text="relaxing ambient background",
            tags=[("essentia:mood", "relaxing", 0.5)],
            energy=0.30,
            aggression=0.10,
            danceability=0.50,
            bpm=90.0,
        )
        _insert_track(
            conn,
            "good-track",
            tmp_path / "b-good.mp3",
            title="Same Score Good",
            artist="Artist B",
            profile_text="relaxing ambient background",
            tags=[("essentia:mood", "relaxing", 0.5)],
            energy=0.30,
            aggression=0.10,
            danceability=0.50,
            bpm=90.0,
        )
    finally:
        conn.close()

    result = CliRunner().invoke(
        app,
        ["judge", "chill", "--db", str(db_path), "--limit", "2"],
        input="n\ny\n",
    )

    assert result.exit_code == 0, result.output
    conn = connect_db(db_path)
    try:
        rows = conn.execute("SELECT track_id, rating FROM feedback").fetchall()
        ratings = {row["track_id"]: row["rating"] for row in rows}
        assert ratings == {"bad-track": -1, "good-track": 1}
        response = search_music(conn, "chill", limit=2, explain=True)
        assert response.results[0].track_id == "good-track"
        assert response.results[0].breakdown["feedback_score"] > 0
        assert "positive feedback boost" in "; ".join(response.results[0].explanation)
    finally:
        conn.close()


def _insert_track(
    conn,
    track_id: str,
    path,
    *,
    title: str | None = None,
    artist: str | None = None,
    profile_text: str,
    tags: list[tuple[str, str, float]],
    energy: float,
    aggression: float,
    danceability: float,
    bpm: float,
) -> None:
    path.write_bytes(b"audio")
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns,
            title, artist, indexed_at
        ) VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?)
        """,
        (track_id, str(path), f"hash-{track_id}", path.suffix.lower(), title, artist, now),
    )
    conn.execute(
        """
        INSERT INTO audio_features (
            track_id, bpm, energy, aggression, danceability, brightness, updated_at
        ) VALUES (?, ?, ?, ?, ?, 0.3, ?)
        """,
        (track_id, bpm, energy, aggression, danceability, now),
    )
    conn.execute(
        """
        INSERT INTO track_profiles (track_id, profile_text, profile_json, updated_at)
        VALUES (?, ?, '{}', ?)
        """,
        (track_id, profile_text, now),
    )
    conn.execute(
        """
        INSERT INTO tracks_fts (track_id, title, artist, album, genre, profile_text)
        VALUES (?, ?, ?, NULL, NULL, ?)
        """,
        (track_id, title, artist, profile_text),
    )
    for source, tag, score in tags:
        conn.execute(
            """
            INSERT INTO track_tags (track_id, source, tag, score, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (track_id, source, tag, score, now),
        )
    conn.commit()
