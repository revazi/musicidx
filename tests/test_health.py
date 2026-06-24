from __future__ import annotations

from typer.testing import CliRunner

from musicidx.analyzer.embeddings import save_profile_embedding
from musicidx.cli import app
from musicidx.db import connect_db, init_db
from musicidx.health import build_index_health


def test_build_index_health_reports_ready_index(tmp_path):
    db_path = tmp_path / "index.sqlite"
    models_path = tmp_path / ".musicidx-models"
    models_path.mkdir()
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_ready_track(conn, "track-1", tmp_path / "one.mp3", model="model-a")
        _insert_ready_track(conn, "track-2", tmp_path / "two.mp3", model="model-a")
        conn.commit()

        payload = build_index_health(
            conn,
            db_path=db_path,
            models_path=models_path,
            semantic_model="model-a",
        )

        assert payload["ready"] is True
        assert payload["tracks"]["active"] == 2
        assert payload["audio_features"]["coverage"] == 1.0
        assert payload["derived_tags"]["coverage"] == 1.0
        assert payload["context_fit"]["coverage"] == 1.0
        assert payload["profiles"]["schema_v2"] == 2
        assert payload["embeddings"]["current"] == 2
        assert payload["warnings"] == []
    finally:
        conn.close()


def test_index_health_warns_for_missing_context_and_stale_embeddings(tmp_path):
    db_path = tmp_path / "models" / "index.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(conn, "track-1", tmp_path / "one.mp3")
        now = "2026-01-01T00:00:00+00:00"
        conn.execute(
            """
            INSERT INTO track_profiles (
                track_id, profile_text, embedding_text, profile_json,
                profile_schema_version, source_fingerprint, updated_at
            ) VALUES ('track-1', 'profile', 'fresh text', '{}', 2, 'fp', ?)
            """,
            (now,),
        )
        save_profile_embedding(conn, "track-1", "stale text", [1.0, 0.0], model_name="model-a")
        conn.commit()

        payload = build_index_health(
            conn,
            db_path=db_path,
            models_path=models_path,
            semantic_model="model-a",
        )

        codes = {warning["code"] for warning in payload["warnings"]}
        assert "db_path_inside_models_path" in codes
        assert "missing_audio_features" in codes
        assert "missing_derived_tags" in codes
        assert "missing_context_fit" in codes
        assert "stale_embeddings" in codes
        assert payload["ready"] is False
    finally:
        conn.close()


def test_index_health_command_json(tmp_path):
    db_path = tmp_path / "index.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    conn = connect_db(db_path)
    try:
        init_db(conn)
    finally:
        conn.close()

    result = CliRunner().invoke(
        app,
        [
            "index-health",
            "--db",
            str(db_path),
            "--models-path",
            str(models_path),
            "--semantic-model",
            "model-a",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"tracks"' in result.output
    assert '"no_tracks"' in result.output


def _insert_ready_track(conn, track_id: str, path, *, model: str) -> None:
    _insert_track(conn, track_id, path)
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        """
        INSERT INTO audio_features (
            track_id, bpm, energy, danceability, aggression, brightness, updated_at
        ) VALUES (?, 100.0, 0.4, 0.5, 0.2, 0.4, ?)
        """,
        (track_id, now),
    )
    conn.execute(
        """
        INSERT INTO track_tags (track_id, source, tag, score, updated_at)
        VALUES (?, 'derived:features', 'low_aggression', 0.9, ?)
        """,
        (track_id, now),
    )
    conn.execute(
        """
        INSERT INTO track_context_fit (
            track_id, context, score, confidence, evidence_json, updated_at
        ) VALUES (?, 'background', 0.8, 0.7, '{}', ?)
        """,
        (track_id, now),
    )
    conn.execute(
        """
        INSERT INTO track_profiles (
            track_id, profile_text, embedding_text, profile_json,
            profile_schema_version, source_fingerprint, updated_at
        ) VALUES (?, 'profile', 'embedding text', '{}', 2, 'fp', ?)
        """,
        (track_id, now),
    )
    save_profile_embedding(conn, track_id, "embedding text", [1.0, 0.0], model_name=model)


def _insert_track(conn, track_id: str, path) -> None:
    path.write_bytes(b"audio")
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns,
            title, artist, indexed_at
        ) VALUES (?, ?, ?, ?, 1, 1, 'Title', 'Artist', '2026-01-01T00:00:00+00:00')
        """,
        (track_id, str(path), f"hash-{track_id}", path.suffix.lower()),
    )
