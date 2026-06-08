from __future__ import annotations

from musicidx.db import connect_db, init_db
from musicidx.metadata import (
    TrackMetadata,
    build_track_profile,
    metadata_from_ffprobe_json,
    process_metadata,
    save_track_metadata,
    search_text,
)


def test_metadata_from_ffprobe_json_normalizes_tags_and_technical_fields():
    data = {
        "format": {
            "duration": "428.4",
            "bit_rate": "880000",
            "tags": {
                "TITLE": "La Femme d'Argent",
                "ARTIST": "Air",
                "ALBUM": "Moon Safari",
                "Album Artist": "Air",
                "GENRE": "Downtempo/Electronic",
                "YEAR": "1998",
                "TRACKNUMBER": "1/10",
                "DiscNumber": "1/1",
            },
        },
        "streams": [
            {"codec_type": "video", "codec_name": "mjpeg"},
            {
                "codec_type": "audio",
                "codec_name": "flac",
                "sample_rate": "44100",
                "channels": 2,
            },
        ],
    }

    metadata = metadata_from_ffprobe_json(data)

    assert metadata.title == "La Femme d'Argent"
    assert metadata.artist == "Air"
    assert metadata.album == "Moon Safari"
    assert metadata.album_artist == "Air"
    assert metadata.genre == "Downtempo/Electronic"
    assert metadata.date == "1998"
    assert metadata.track_number == "1/10"
    assert metadata.disc_number == "1/1"
    assert metadata.duration_sec == 428.4
    assert metadata.codec == "flac"
    assert metadata.sample_rate == 44100
    assert metadata.bit_rate == 880000
    assert metadata.channels == 2


def test_save_metadata_updates_tracks_profiles_and_fts(tmp_path):
    db_path = tmp_path / "index.sqlite"
    track_path = tmp_path / "pink_moon.mp3"
    track_path.write_bytes(b"audio")
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)
        metadata = TrackMetadata(
            title="Pink Moon",
            artist="Nick Drake",
            album="Pink Moon",
            genre="Folk",
            duration_sec=149.0,
            codec="mp3",
            sample_rate=44100,
            bit_rate=192000,
            channels=2,
        )
        profile_text, profile_json = build_track_profile(metadata, track_path)

        save_track_metadata(conn, "track-1", metadata, profile_text, profile_json)
        conn.commit()

        row = conn.execute(
            "SELECT title, artist, codec FROM tracks WHERE id = 'track-1'"
        ).fetchone()
        assert dict(row) == {"title": "Pink Moon", "artist": "Nick Drake", "codec": "mp3"}
        profile_row = conn.execute(
            "SELECT profile_text FROM track_profiles WHERE track_id = 'track-1'"
        ).fetchone()
        assert "Artist: Nick Drake." in profile_row["profile_text"]

        results = search_text(conn, "Nick Drake")
        assert len(results) == 1
        assert results[0].track_id == "track-1"
        assert results[0].title == "Pink Moon"
    finally:
        conn.close()


def test_process_metadata_uses_extractor_and_populates_search(monkeypatch, tmp_path):
    track_path = tmp_path / "song.flac"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)

        def fake_extract_metadata(path):
            assert path == track_path
            return TrackMetadata(
                title="Svefn-g-englar",
                artist="Sigur Ros",
                album="Agaetis byrjun",
                genre="Post-rock",
                duration_sec=600.0,
                codec="flac",
            )

        monkeypatch.setattr("musicidx.metadata.extract_metadata", fake_extract_metadata)
        summary = process_metadata(conn)

        assert summary.processed == 1
        assert summary.updated == 1
        assert summary.errors == 0
        results = search_text(conn, "Sigur Ros")
        assert [result.track_id for result in results] == ["track-1"]
    finally:
        conn.close()


def _insert_track(conn, track_id: str, path) -> None:
    conn.execute(
        """
        INSERT INTO tracks (id, path, path_hash, extension, file_size, file_mtime_ns, indexed_at)
        VALUES (?, ?, 'hash', ?, 1, 1, '2026-01-01T00:00:00+00:00')
        """,
        (track_id, str(path), path.suffix.lower()),
    )
    conn.commit()
