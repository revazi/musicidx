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


def test_parse_intent_detects_highest_bpm_sort(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "fast-track",
            tmp_path / "fast.mp3",
            profile_text="fast dance",
            tags=[("essentia:mood", "energetic", 0.9)],
            energy=0.80,
            aggression=0.40,
            danceability=0.90,
            bpm=150.0,
        )

        intent = parse_intent_dynamic("give me the 3 highest BPM tracks", conn)

        assert intent.limit == 3
        assert intent.sort_by[0].field == "tempo_bpm"
        assert intent.sort_by[0].direction == "desc"
        assert intent.feature_ranges["tempo_bpm"].source.endswith("very_high")
    finally:
        conn.close()


def test_search_music_sorts_by_highest_bpm(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "slow-track",
            tmp_path / "slow.mp3",
            title="Slow",
            artist="Artist A",
            profile_text="slow song",
            tags=[("essentia:mood", "calm", 0.9)],
            energy=0.20,
            aggression=0.10,
            danceability=0.20,
            bpm=80.0,
        )
        _insert_track(
            conn,
            "mid-track",
            tmp_path / "mid.mp3",
            title="Mid",
            artist="Artist B",
            profile_text="mid tempo",
            tags=[("essentia:mood", "upbeat", 0.9)],
            energy=0.50,
            aggression=0.20,
            danceability=0.60,
            bpm=120.0,
        )
        _insert_track(
            conn,
            "fast-track",
            tmp_path / "fast.mp3",
            title="Fast",
            artist="Artist C",
            profile_text="fast dance",
            tags=[("essentia:mood", "energetic", 0.9)],
            energy=0.90,
            aggression=0.30,
            danceability=0.95,
            bpm=155.0,
        )

        response = search_music(conn, "highest BPM", explain=True)

        assert [result.track_id for result in response.results] == [
            "fast-track",
            "mid-track",
            "slow-track",
        ]
        assert response.diagnostics["sort_by"] == [
            {"field": "tempo_bpm", "direction": "desc", "source": "natural_language"}
        ]
        assert "sorted by highest BPM" in "; ".join(response.results[0].explanation)
    finally:
        conn.close()


def test_search_music_prioritizes_explicit_artist_plus_vibe_query(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        for index, title in enumerate(["Singularity", "Zulu", "Powers Of Ten"], start=1):
            _insert_track(
                conn,
                f"bodzin-{index}",
                tmp_path / f"bodzin-{index}.mp3",
                title=title,
                artist="Stephan Bodzin",
                profile_text=f"Artist: Stephan Bodzin. Title: {title}. medium danceability",
                tags=[("essentia:genre", "electronic---techno", 0.45)],
                energy=0.55,
                aggression=0.30,
                danceability=0.82,
                bpm=123.0,
            )
        _insert_track(
            conn,
            "dance-competitor",
            tmp_path / "competitor.mp3",
            title="Peak Dance Floor",
            artist="Other Artist",
            profile_text="dance party energetic house floor",
            tags=[("essentia:mood", "party", 0.9), ("essentia:mood", "energetic", 0.8)],
            energy=0.95,
            aggression=0.55,
            danceability=0.96,
            bpm=128.0,
        )

        response = search_music(conn, "Stephan Bodzin, dance", limit=4, explain=True)

        assert [result.artist for result in response.results[:3]] == [
            "Stephan Bodzin",
            "Stephan Bodzin",
            "Stephan Bodzin",
        ]
        assert response.results[0].breakdown["metadata_score"] == 1.0
        assert response.results[0].score == 1.0
        assert "metadata match" in "; ".join(response.results[0].explanation)
    finally:
        conn.close()


def test_search_music_returns_normalized_relevance_score(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "strong-track",
            tmp_path / "strong.mp3",
            title="Strong",
            artist="Artist A",
            profile_text="relaxing ambient background",
            tags=[("essentia:mood", "relaxing", 0.9)],
            energy=0.25,
            aggression=0.10,
            danceability=0.40,
            bpm=90.0,
        )
        _insert_track(
            conn,
            "weaker-track",
            tmp_path / "weak.mp3",
            title="Weak",
            artist="Artist B",
            profile_text="relaxing",
            tags=[("essentia:mood", "relaxing", 0.45)],
            energy=0.40,
            aggression=0.20,
            danceability=0.40,
            bpm=100.0,
        )

        response = search_music(conn, "chill", explain=True)

        assert response.results[0].score == 1.0
        assert response.results[0].breakdown["raw_score"] < 1.0
        assert response.results[0].breakdown["display_score"] == 1.0
        assert response.diagnostics["score_normalization"] == "relative_to_top_result"
        assert response.diagnostics["top_raw_score"] == response.results[0].breakdown["raw_score"]
        assert (
            response.results[0].breakdown["raw_score"]
            >= response.results[1].breakdown["raw_score"]
        )
    finally:
        conn.close()


def test_parse_intent_expands_upbeat_dance_without_llm(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "dance-track",
            tmp_path / "dance.mp3",
            profile_text="high energy high danceability",
            tags=[("essentia:genre", "electronic---house", 0.85)],
            energy=0.85,
            aggression=0.30,
            danceability=0.90,
            bpm=126.0,
        )

        intent = parse_intent_dynamic("upbeat dance music", conn)

        assert "happy" in intent.contexts
        assert "energy" in intent.feature_ranges
        assert "danceability" in intent.feature_ranges
        assert "electronic---house" in intent.prefer_tags
    finally:
        conn.close()


def test_search_music_uses_feature_priors_for_upbeat_without_exact_tags(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "upbeat-track",
            tmp_path / "upbeat.mp3",
            title="Movement",
            artist="Artist A",
            profile_text="high energy high danceability tempo around 126 bpm",
            tags=[("essentia:genre", "electronic---house", 0.85)],
            energy=0.88,
            aggression=0.35,
            danceability=0.92,
            bpm=126.0,
        )
        _insert_track(
            conn,
            "slow-track",
            tmp_path / "slow.mp3",
            title="Still",
            artist="Artist B",
            profile_text="low energy low danceability slow acoustic ballad",
            tags=[("essentia:mood", "sad", 0.85)],
            energy=0.18,
            aggression=0.10,
            danceability=0.15,
            bpm=72.0,
        )

        response = search_music(conn, "upbeat dance music", explain=True)

        assert [result.track_id for result in response.results] == ["upbeat-track"]
        assert response.results[0].breakdown["feature_score"] > 0.8
        assert response.diagnostics["weights"]["features"] > 0.30
    finally:
        conn.close()


def test_parse_intent_handles_not_aggressive_as_negative_feature(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "soft-track",
            tmp_path / "soft.mp3",
            profile_text="soft calm",
            tags=[("essentia:mood", "calm", 0.8)],
            energy=0.25,
            aggression=0.10,
            danceability=0.30,
            bpm=82.0,
        )

        intent = parse_intent_dynamic("not aggressive background", conn)

        assert "aggressive" in intent.avoid_tag_concepts
        assert "aggressive" not in intent.prefer_tag_concepts
        assert intent.feature_ranges["aggression"].source.endswith("low")
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

        assert [result.track_id for result in response.results] == ["chill-track"]
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


def test_search_music_prefers_mood_evidence_over_feature_only_matches(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "feature-only-track",
            tmp_path / "feature-only.mp3",
            title="Feature Only",
            artist="Artist A",
            profile_text="low energy low brightness low aggression",
            tags=[("essentia:mood", "neutral", 0.9)],
            energy=0.10,
            aggression=0.10,
            danceability=0.20,
            bpm=75.0,
        )
        _insert_track(
            conn,
            "sad-evidence-track",
            tmp_path / "sad.mp3",
            title="Sad Evidence",
            artist="Artist B",
            profile_text="sad reflective melancholic song",
            tags=[("essentia:mood", "sad", 0.85)],
            energy=0.60,
            aggression=0.30,
            danceability=0.40,
            bpm=90.0,
        )

        response = search_music(conn, "sad reflective melancholic songs", explain=True)

        assert [result.track_id for result in response.results] == ["sad-evidence-track"]
        assert response.results[0].breakdown["text_score"] > 0
    finally:
        conn.close()


def test_search_music_filters_unmatched_local_library_candidates(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "unrelated-track",
            tmp_path / "unrelated.mp3",
            title="Unrelated",
            artist="Artist A",
            profile_text="slow acoustic ballad",
            tags=[("essentia:mood", "sad", 0.8)],
            energy=0.20,
            aggression=0.10,
            danceability=0.10,
            bpm=70.0,
        )

        response = search_music(conn, "galactic pirate polka", explain=True)

        assert response.results == []
        assert response.diagnostics["filtered_candidate_count"] == 0
    finally:
        conn.close()


def test_search_music_discounts_low_confidence_best_guess_tags(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "weak-tag-track",
            tmp_path / "weak.mp3",
            title="Weak Guess",
            artist="Artist A",
            profile_text="unrelated filler",
            tags=[("essentia:mood", "energetic", 0.05)],
            energy=0.10,
            aggression=0.10,
            danceability=0.10,
            bpm=70.0,
        )
        _insert_track(
            conn,
            "strong-tag-track",
            tmp_path / "strong.mp3",
            title="Strong Match",
            artist="Artist B",
            profile_text="energetic dance floor party",
            tags=[("essentia:mood", "energetic", 0.85)],
            energy=0.90,
            aggression=0.50,
            danceability=0.95,
            bpm=128.0,
        )

        response = search_music(conn, "energetic dance", explain=True)

        assert [result.track_id for result in response.results] == ["strong-tag-track"]
        assert response.diagnostics["minimum_ranking_tag_score"] == 0.20
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
    assert payload["results"][0]["precision_at_k"] == 1.0
    assert payload["results"][0]["avoid_rate"] == 0.0
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
