from __future__ import annotations

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
