from __future__ import annotations

import numpy as np

from musicidx.analyzer.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    blob_to_vector,
    process_embeddings,
    save_profile_embedding,
    search_semantic,
)
from musicidx.db import connect_db, init_db


def test_process_embeddings_stores_profile_vectors_and_skips_current(monkeypatch, tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_profile(conn, "track-1", tmp_path / "a.mp3", "calm ambient relaxing")
        _insert_profile(conn, "track-2", tmp_path / "b.mp3", "fast party energetic")

        def fake_embed_texts(texts, *, model_name=DEFAULT_EMBEDDING_MODEL):
            assert model_name == DEFAULT_EMBEDDING_MODEL
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(texts)]

        monkeypatch.setattr("musicidx.analyzer.embeddings.embed_texts", fake_embed_texts)

        first = process_embeddings(conn)
        second = process_embeddings(conn)

        assert first.processed == 2
        assert first.updated == 2
        assert first.errors == 0
        assert second.processed == 0
        assert second.skipped == 2

        rows = conn.execute(
            "SELECT track_id, dim, vector FROM embeddings ORDER BY track_id"
        ).fetchall()
        assert [row["track_id"] for row in rows] == ["track-1", "track-2"]
        assert rows[0]["dim"] == 2
        assert blob_to_vector(rows[0]["vector"], 2).tolist() == [1.0, 0.0]
    finally:
        conn.close()


def test_process_embeddings_prefers_embedding_text(monkeypatch, tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_profile(
            conn,
            "track-1",
            tmp_path / "a.mp3",
            "display profile text",
            embedding_text="semantic optimized text",
        )

        seen_texts = []

        def fake_embed_texts(texts, *, model_name=DEFAULT_EMBEDDING_MODEL):
            seen_texts.extend(texts)
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr("musicidx.analyzer.embeddings.embed_texts", fake_embed_texts)

        summary = process_embeddings(conn)

        assert summary.updated == 1
        assert seen_texts == ["semantic optimized text"]
        row = conn.execute("SELECT text FROM embeddings WHERE track_id = 'track-1'").fetchone()
        assert row["text"] == "semantic optimized text"
    finally:
        conn.close()


def test_search_semantic_ranks_by_query_similarity(monkeypatch, tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_profile(conn, "track-1", tmp_path / "ambient.mp3", "deep ambient relaxing")
        _insert_profile(conn, "track-2", tmp_path / "party.mp3", "bright party dance")
        save_profile_embedding(conn, "track-1", "deep ambient relaxing", [1.0, 0.0])
        save_profile_embedding(conn, "track-2", "bright party dance", [0.0, 1.0])
        conn.commit()

        def fake_embed_texts(texts, *, model_name=DEFAULT_EMBEDDING_MODEL):
            assert texts == ["chill atmospheric music"]
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr("musicidx.analyzer.embeddings.embed_texts", fake_embed_texts)

        results = search_semantic(conn, "chill atmospheric music", limit=2)

        assert [result.track_id for result in results] == ["track-1", "track-2"]
        assert results[0].score > results[1].score
        assert results[0].profile_text == "deep ambient relaxing"
    finally:
        conn.close()


def _insert_profile(
    conn,
    track_id: str,
    path,
    profile_text: str,
    *,
    embedding_text: str | None = None,
) -> None:
    path.write_bytes(b"audio")
    conn.execute(
        """
        INSERT INTO tracks (id, path, path_hash, extension, file_size, file_mtime_ns, indexed_at)
        VALUES (?, ?, ?, ?, 1, 1, '2026-01-01T00:00:00+00:00')
        """,
        (track_id, str(path), f"hash-{track_id}", path.suffix.lower()),
    )
    conn.execute(
        """
        INSERT INTO track_profiles (
            track_id, profile_text, embedding_text, profile_json, updated_at
        ) VALUES (?, ?, ?, '{}', '2026-01-01T00:00:00+00:00')
        """,
        (track_id, profile_text, embedding_text),
    )
    conn.commit()
