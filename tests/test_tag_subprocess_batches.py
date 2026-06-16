from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from musicidx.cli import _run_tag_subprocess_batches
from musicidx.db import connect_db, init_db


def test_run_tag_subprocess_batches_splits_track_ids(monkeypatch, tmp_path):
    db_path = tmp_path / "index.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    seen_batches: list[list[str]] = []

    def fake_run(command, capture_output, text, check):
        assert capture_output is True
        assert text is True
        assert check is False
        assert "--no-subprocess-batches" in command
        assert command[command.index("--workers") + 1] == "1"
        ids_path = Path(command[command.index("--track-id-file") + 1])
        ids = json.loads(ids_path.read_text())
        seen_batches.append(ids)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "processed": len(ids),
                    "updated": len(ids),
                    "skipped": 0,
                    "errors": 0,
                    "model_count": 2,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("musicidx.cli.subprocess.run", fake_run)

    summary, batches = _run_tag_subprocess_batches(
        db_path=db_path,
        models_path=models_path,
        track_ids=["track-1", "track-2", "track-3"],
        missing_only=True,
        min_score=0.1,
        batch_size=2,
    )

    assert batches == 2
    assert seen_batches == [["track-1", "track-2"], ["track-3"]]
    assert summary.processed == 3
    assert summary.updated == 3
    assert summary.errors == 0
    assert summary.model_count == 2


def test_run_tag_subprocess_batches_records_failed_batch(monkeypatch, tmp_path):
    db_path = tmp_path / "index.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(conn, "track-1", tmp_path / "one.mp3")
        _insert_track(conn, "track-2", tmp_path / "two.mp3")
        conn.commit()
    finally:
        conn.close()

    def fake_run(command, capture_output, text, check):
        return SimpleNamespace(returncode=9, stdout="", stderr="boom")

    monkeypatch.setattr("musicidx.cli.subprocess.run", fake_run)

    summary, batches = _run_tag_subprocess_batches(
        db_path=db_path,
        models_path=models_path,
        track_ids=["track-1", "track-2"],
        missing_only=False,
        min_score=0.2,
        batch_size=5,
    )

    assert batches == 1
    assert summary.errors == 2

    conn = connect_db(db_path)
    try:
        rows = conn.execute("SELECT id, last_error FROM tracks ORDER BY id").fetchall()
        assert [(row["id"], row["last_error"]) for row in rows] == [
            ("track-1", "tag subprocess batch failed: boom"),
            ("track-2", "tag subprocess batch failed: boom"),
        ]
    finally:
        conn.close()


def _insert_track(conn, track_id: str, path) -> None:
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns, indexed_at
        ) VALUES (?, ?, ?, ?, 1, 1, '2026-01-01T00:00:00+00:00')
        """,
        (track_id, str(path), f"hash-{track_id}", path.suffix),
    )
