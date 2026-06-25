from __future__ import annotations

import json

from typer.testing import CliRunner

from musicidx.cli import app
from musicidx.db import connect_db, init_db
from musicidx.metadata import (
    MetadataExtractionResult,
    TrackMetadata,
    apply_filename_fallback,
    build_metadata_extraction_result,
    build_track_profile,
    infer_metadata_from_filename,
    metadata_from_ffprobe_json,
    normalize_metadata_value,
    process_metadata,
    repair_metadata_from_duplicate_candidates,
    repair_metadata_from_filename,
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


def test_filename_fallback_infers_artist_title_from_decoded_name(tmp_path):
    path = tmp_path / "David%20August%20-%20Ingrid.mp3"

    metadata = infer_metadata_from_filename(path)

    assert metadata.artist == "David August"
    assert metadata.title == "Ingrid"


def test_apply_filename_fallback_preserves_real_tags_and_fills_missing_artist(tmp_path):
    path = tmp_path / "Stephan Bodzin - Singularity.mp3"

    filled = apply_filename_fallback(TrackMetadata(title="Stephan Bodzin - Singularity"), path)

    assert filled.artist == "Stephan Bodzin"
    assert filled.title == "Singularity"

    preserved = apply_filename_fallback(
        TrackMetadata(title="Custom Title", artist="Tagged Artist"),
        path,
    )
    assert preserved.artist == "Tagged Artist"
    assert preserved.title == "Custom Title"


def test_metadata_extraction_result_tracks_claims_and_selected_values(tmp_path):
    path = tmp_path / "Stephan Bodzin - Singularity.mp3"
    result = build_metadata_extraction_result(
        TrackMetadata(title="Stephan Bodzin - Singularity", genre="Techno"),
        path,
    )

    assert result.metadata.artist == "Stephan Bodzin"
    assert result.metadata.title == "Singularity"
    assert normalize_metadata_value("Stéphan & Bodzin") == "stephan and bodzin"
    selected = {
        (claim.field_name, claim.source): claim
        for claim in result.claims
        if claim.selected
    }
    assert selected[("artist", "derived")].value_text == "Stephan Bodzin"
    assert selected[("title", "derived")].value_text == "Singularity"
    assert selected[("genre", "ffprobe")].value_text == "Techno"


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
            """
            SELECT title, artist, codec, title_norm, artist_norm, artist_title_norm,
                   metadata_confidence
            FROM tracks WHERE id = 'track-1'
            """
        ).fetchone()
        assert row["title"] == "Pink Moon"
        assert row["artist"] == "Nick Drake"
        assert row["codec"] == "mp3"
        assert row["title_norm"] == "pink moon"
        assert row["artist_norm"] == "nick drake"
        assert row["artist_title_norm"] == "nick drake pink moon"
        assert row["metadata_confidence"] == 0.7
        profile_row = conn.execute(
            """
            SELECT profile_text, embedding_text, profile_json, profile_schema_version,
                   source_fingerprint
            FROM track_profiles WHERE track_id = 'track-1'
            """
        ).fetchone()
        assert "Artist: Nick Drake." in profile_row["profile_text"]
        assert "Nick Drake - Pink Moon" in profile_row["embedding_text"]
        profile_doc = json.loads(profile_row["profile_json"])
        assert profile_doc["schema_version"] == 2
        assert profile_doc["identity"]["normalized"]["artist_norm"] == "nick drake"
        assert profile_doc["identity"]["confidence"]["overall"] == 0.7
        assert profile_doc["search_text"]["embedding_text"] == profile_row["embedding_text"]
        assert profile_row["profile_schema_version"] == 2
        assert profile_row["source_fingerprint"]

        claim_rows = conn.execute(
            """
            SELECT field_name, value_text, source, selected
            FROM track_metadata_claims
            WHERE track_id = 'track-1'
            ORDER BY field_name
            """
        ).fetchall()
        claims = {row["field_name"]: row for row in claim_rows}
        assert claims["artist"]["value_text"] == "Nick Drake"
        assert claims["artist"]["source"] == "derived"
        assert claims["artist"]["selected"] == 1

        results = search_text(conn, "Nick Drake")
        assert len(results) == 1
        assert results[0].track_id == "track-1"
        assert results[0].title == "Pink Moon"
    finally:
        conn.close()


def test_metadata_missing_only_retries_titleless_profiled_tracks(monkeypatch, tmp_path):
    track_path = tmp_path / "untitled.mp3"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)
        metadata = TrackMetadata(duration_sec=120.0, codec="mp3", sample_rate=44100, channels=2)
        profile_text, profile_json = build_track_profile(metadata, track_path)
        save_track_metadata(conn, "track-1", metadata, profile_text, profile_json)
        conn.commit()

        def fake_extract_metadata_result(path):
            assert path == track_path
            metadata = TrackMetadata(
                title="Untitled",
                duration_sec=120.0,
                codec="mp3",
                sample_rate=44100,
                channels=2,
            )
            return MetadataExtractionResult(metadata=metadata, claims=[])

        monkeypatch.setattr(
            "musicidx.metadata.extract_metadata_result",
            fake_extract_metadata_result,
        )

        summary = process_metadata(conn, missing_only=True)

        assert summary.processed == 1
        assert summary.updated == 1
        row = conn.execute("SELECT title FROM tracks WHERE id = 'track-1'").fetchone()
        assert row["title"] == "Untitled"
    finally:
        conn.close()


def test_repair_metadata_from_filename_fills_missing_artist_and_updates_profile(tmp_path):
    track_path = tmp_path / "Donna Summer - Love To Love You Baby.mp3"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)
        metadata = TrackMetadata(title="Love To Love You Baby", duration_sec=255.0)
        profile_text, profile_json = build_track_profile(metadata, track_path)
        save_track_metadata(conn, "track-1", metadata, profile_text, profile_json)
        conn.commit()

        summary = repair_metadata_from_filename(conn, missing_only=True)

        assert summary.processed == 1
        assert summary.updated == 1
        assert summary.repairs is not None
        assert summary.repairs[0].changed_fields == ["artist"]
        row = conn.execute(
            """
            SELECT title, artist, artist_title_norm
            FROM tracks WHERE id = 'track-1'
            """
        ).fetchone()
        assert row["title"] == "Love To Love You Baby"
        assert row["artist"] == "Donna Summer"
        assert row["artist_title_norm"] == "donna summer love to love you baby"
        profile = conn.execute(
            "SELECT embedding_text FROM track_profiles WHERE track_id = 'track-1'"
        ).fetchone()
        assert "Donna Summer - Love To Love You Baby" in profile["embedding_text"]
    finally:
        conn.close()


def test_repair_metadata_from_duplicate_candidates_fills_missing_artist(tmp_path):
    known_path = tmp_path / "Julio Iglesias - Cu Cu Rru Cu Cu Paloma.mp3"
    unknown_path = tmp_path / "Cu Cu Rru Cu Cu Paloma.mp3"
    known_path.write_bytes(b"audio")
    unknown_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "known", known_path)
        save_track_metadata(
            conn,
            "known",
            TrackMetadata(
                title="Cu Cu Rru Cu Cu Paloma",
                artist="Julio Iglesias",
                duration_sec=208.50,
            ),
            *build_track_profile(
                TrackMetadata(
                    title="Cu Cu Rru Cu Cu Paloma",
                    artist="Julio Iglesias",
                    duration_sec=208.50,
                ),
                known_path,
            ),
        )
        _insert_track(conn, "unknown", unknown_path)
        save_track_metadata(
            conn,
            "unknown",
            TrackMetadata(title="Cu Cu Rru Cu Cu Paloma", duration_sec=208.48),
            *build_track_profile(
                TrackMetadata(title="Cu Cu Rru Cu Cu Paloma", duration_sec=208.48),
                unknown_path,
            ),
        )
        conn.commit()

        summary = repair_metadata_from_duplicate_candidates(conn, missing_only=True)

        assert summary.updated == 1
        assert summary.repairs is not None
        assert summary.repairs[0].track_id == "unknown"
        assert summary.repairs[0].after["artist"] == "Julio Iglesias"
        row = conn.execute("SELECT artist FROM tracks WHERE id = 'unknown'").fetchone()
        assert row["artist"] == "Julio Iglesias"
    finally:
        conn.close()


def test_repair_metadata_command_json_dry_run(tmp_path):
    db_path = tmp_path / "index.sqlite"
    track_path = tmp_path / "Prince - International Lover.mp3"
    track_path.write_bytes(b"audio")
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)
        metadata = TrackMetadata(title="International Lover")
        profile_text, profile_json = build_track_profile(metadata, track_path)
        save_track_metadata(conn, "track-1", metadata, profile_text, profile_json)
        conn.commit()
    finally:
        conn.close()

    result = CliRunner().invoke(
        app,
        [
            "repair-metadata",
            "--db",
            str(db_path),
            "--from-filename",
            "--from-duplicates",
            "--missing-only",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["updated"] == 1
    assert payload["repairs"][0]["after"]["artist"] == "Prince"
    conn = connect_db(db_path)
    try:
        row = conn.execute("SELECT artist FROM tracks WHERE id = 'track-1'").fetchone()
        assert row["artist"] is None
    finally:
        conn.close()


def test_process_metadata_uses_extractor_and_populates_search(monkeypatch, tmp_path):
    track_path = tmp_path / "song.flac"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)

        def fake_extract_metadata_result(path):
            assert path == track_path
            metadata = TrackMetadata(
                title="Svefn-g-englar",
                artist="Sigur Ros",
                album="Agaetis byrjun",
                genre="Post-rock",
                duration_sec=600.0,
                codec="flac",
            )
            return MetadataExtractionResult(metadata=metadata, claims=[])

        monkeypatch.setattr(
            "musicidx.metadata.extract_metadata_result",
            fake_extract_metadata_result,
        )
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
