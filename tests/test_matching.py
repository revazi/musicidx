from __future__ import annotations

import array
import json
from pathlib import Path

import chromaprint
from typer.testing import CliRunner

from musicidx.analyzer.embeddings import vector_to_blob
from musicidx.cli import app
from musicidx.db import connect_db, init_db
from musicidx.search.match_evaluation import (
    aggregate_match_eval_results,
    evaluate_match_case,
    load_match_eval_cases,
)
from musicidx.search.matching import compare_tracks, find_track_matches


def test_compare_tracks_exact_content_hash_is_authoritative(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "track-a",
            tmp_path / "a.mp3",
            content_hash="hash-1",
            chromaprint="fp-1",
            title="Song",
            artist="Artist",
            duration_sec=120.0,
            fingerprint_duration=120.0,
        )
        _insert_track(
            conn,
            "track-b",
            tmp_path / "b.mp3",
            content_hash="hash-1",
            chromaprint="fp-1",
            title="Song",
            artist="Artist",
            duration_sec=120.5,
            fingerprint_duration=120.5,
        )

        report = compare_tracks(conn, "track-a", "track-b")

        assert report.schema_version == 1
        assert report.decision == "exact_duplicate"
        assert report.identity_decision == "same"
        assert report.confidence == "high"
        assert report.confidence_score == 1.0
        evidence = {item.source: item for item in report.evidence}
        assert evidence["content_hash"].status == "match"
        assert evidence["content_hash"].decisive is True
        assert report.candidate_kind == "exact_duplicate"
        assert report.candidate_strength == "strong"
        assert report.policy["semantic_embeddings"] == "not_used_for_identity"
        assert report.policy["llm"] == "not_used"
    finally:
        conn.close()


def test_compare_tracks_metadata_duration_match_is_only_possible(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "track-a",
            tmp_path / "a.mp3",
            title="Song",
            artist="Artist",
            duration_sec=120.0,
        )
        _insert_track(
            conn,
            "track-b",
            tmp_path / "b.mp3",
            title="Song",
            artist="Artist",
            duration_sec=121.0,
        )

        report = compare_tracks(conn, "track-a", "track-b")

        assert report.decision == "possible_metadata_match"
        assert report.identity_decision == "possible"
        assert report.confidence == "medium"
        assert "metadata_duration_match_is_advisory_only" in report.warnings
    finally:
        conn.close()


def test_compare_tracks_fingerprint_mismatch_blocks_metadata_identity(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "track-a",
            tmp_path / "a.mp3",
            chromaprint="fp-a",
            title="Song",
            artist="Artist",
            duration_sec=120.0,
        )
        _insert_track(
            conn,
            "track-b",
            tmp_path / "b.mp3",
            chromaprint="fp-b",
            title="Song",
            artist="Artist",
            duration_sec=120.2,
        )

        report = compare_tracks(conn, "track-a", "track-b")

        assert report.decision == "no_identity_match"
        assert report.identity_decision == "unknown"
        assert "metadata_match_blocked_by_identity_mismatch" in report.warnings
        evidence = {item.source: item for item in report.evidence}
        assert evidence["chromaprint"].status == "mismatch"
        assert evidence["artist_title_norm"].status == "match"
    finally:
        conn.close()


def test_audio_embedding_only_is_sound_similar_not_identity(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-a", tmp_path / "a.mp3", title="Alpha", artist="Artist A")
        _insert_track(conn, "track-b", tmp_path / "b.mp3", title="Beta", artist="Artist B")
        _insert_audio_embedding(conn, "track-a", [1.0, 0.0])
        _insert_audio_embedding(conn, "track-b", [0.98, 0.02])

        report = compare_tracks(conn, "track-a", "track-b")

        assert report.decision == "sound_similar_only"
        assert report.identity_decision == "unknown"
        assert report.confidence == "low"
        assert "audio_embedding_similarity_only" in report.warnings
        evidence = {item.source: item for item in report.evidence}
        assert evidence["audio_embedding"].status == "similar"
        assert evidence["audio_embedding"].role == "similarity_only"
        assert evidence["audio_embedding"].decisive is False
        assert report.policy["audio_embeddings"] == "similarity_only_not_identity"
    finally:
        conn.close()


def test_audio_embedding_cannot_override_fingerprint_mismatch(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "track-a",
            tmp_path / "a.mp3",
            chromaprint="fp-a",
            title="Alpha",
            artist="Artist A",
        )
        _insert_track(
            conn,
            "track-b",
            tmp_path / "b.mp3",
            chromaprint="fp-b",
            title="Beta",
            artist="Artist B",
        )
        _insert_audio_embedding(conn, "track-a", [1.0, 0.0])
        _insert_audio_embedding(conn, "track-b", [1.0, 0.0])

        report = compare_tracks(conn, "track-a", "track-b")

        assert report.decision == "no_identity_match"
        assert report.identity_decision == "unknown"
        evidence = {item.source: item for item in report.evidence}
        assert evidence["chromaprint"].status == "mismatch"
        assert evidence["audio_embedding"].status == "similar"
        assert "audio_embedding_similarity_only" not in report.warnings
    finally:
        conn.close()


def test_match_track_finds_audio_embedding_similarity_candidates(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-a", tmp_path / "a.mp3", title="Alpha", artist="Artist A")
        _insert_track(conn, "track-b", tmp_path / "b.mp3", title="Beta", artist="Artist B")
        _insert_audio_embedding(conn, "track-a", [1.0, 0.0])
        _insert_audio_embedding(conn, "track-b", [0.99, 0.01])

        reports = find_track_matches(conn, "track-a")

        assert [report.track_b.track_id for report in reports] == ["track-b"]
        assert reports[0].decision == "sound_similar_only"
        assert reports[0].candidate_scores["audio_embedding"] > 0.9
    finally:
        conn.close()


def test_compare_tracks_version_conflict_is_related_not_duplicate(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "studio",
            tmp_path / "studio.mp3",
            title="Song",
            artist="Artist",
            duration_sec=240.0,
        )
        _insert_track(
            conn,
            "live",
            tmp_path / "live.mp3",
            title="Song (Live)",
            artist="Artist",
            duration_sec=241.0,
        )

        report = compare_tracks(conn, "studio", "live")

        assert report.decision == "related_version_not_duplicate"
        assert report.identity_decision == "unknown"
        assert "version_conflict_blocks_metadata_identity" in report.warnings
        evidence = {item.source: item for item in report.evidence}
        assert evidence["version_conflict"].status == "conflict"
        assert evidence["version_conflict"].details["left_tokens"] == []
        assert evidence["version_conflict"].details["right_tokens"] == ["live"]
    finally:
        conn.close()


def test_authoritative_identity_overrides_version_conflict(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "studio",
            tmp_path / "studio.mp3",
            title="Song",
            artist="Artist",
            content_hash="hash-1",
            duration_sec=240.0,
        )
        _insert_track(
            conn,
            "live-label",
            tmp_path / "live-label.mp3",
            title="Song (Live)",
            artist="Artist",
            content_hash="hash-1",
            duration_sec=241.0,
        )

        report = compare_tracks(conn, "studio", "live-label")

        assert report.decision == "exact_duplicate"
        assert report.identity_decision == "same"
        evidence = {item.source: item for item in report.evidence}
        assert evidence["content_hash"].decisive is True
        assert evidence["version_conflict"].status == "conflict"
    finally:
        conn.close()


def test_find_track_matches_includes_related_version_candidates(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "studio",
            tmp_path / "studio.mp3",
            title="Song",
            artist="Artist",
            duration_sec=240.0,
        )
        _insert_track(
            conn,
            "remix",
            tmp_path / "remix.mp3",
            title="Song - Remix",
            artist="Artist",
            duration_sec=242.0,
        )
        _insert_track(
            conn,
            "other",
            tmp_path / "other.mp3",
            title="Unrelated",
            artist="Artist",
            duration_sec=240.0,
        )

        reports = find_track_matches(conn, "studio")

        assert reports[0].track_b.track_id == "remix"
        assert reports[0].decision == "related_version_not_duplicate"
        assert "other" in {report.track_b.track_id for report in reports}
    finally:
        conn.close()


def test_find_track_matches_returns_deterministic_candidates(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "source", tmp_path / "source.mp3", content_hash="hash-1")
        _insert_track(conn, "exact", tmp_path / "exact.mp3", content_hash="hash-1")
        _insert_track(
            conn,
            "metadata",
            tmp_path / "metadata.mp3",
            title="Shared",
            artist="Artist",
            duration_sec=200.0,
        )
        _insert_track(
            conn,
            "metadata-source",
            tmp_path / "metadata-source.mp3",
            title="Shared",
            artist="Artist",
            duration_sec=201.0,
        )

        reports = find_track_matches(conn, "source")
        metadata_reports = find_track_matches(conn, "metadata-source")

        assert reports[0].track_b.track_id == "exact"
        assert reports[0].decision == "exact_duplicate"
        assert metadata_reports[0].track_b.track_id == "metadata"
        assert metadata_reports[0].decision == "possible_metadata_match"
    finally:
        conn.close()


def test_find_track_matches_includes_filename_stem_candidates_as_advisory(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "source",
            tmp_path / "Track A.mp3",
            duration_sec=180.0,
        )
        _insert_track(
            conn,
            "copy",
            tmp_path / "Track%20A.mp3",
            duration_sec=181.0,
        )

        reports = find_track_matches(conn, "source")

        assert [report.track_b.track_id for report in reports] == ["copy"]
        assert reports[0].decision == "possible_metadata_match"
        assert reports[0].identity_decision == "possible"
        assert "filename_duration_match_is_advisory_only" in reports[0].warnings
        evidence = {item.source: item for item in reports[0].evidence}
        assert evidence["filename_stem"].status == "match"
        assert evidence["filename_stem"].decisive is False
    finally:
        conn.close()


def test_filename_stem_match_cannot_override_fingerprint_mismatch(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "source",
            tmp_path / "Track A.mp3",
            chromaprint="fp-a",
            duration_sec=180.0,
        )
        _insert_track(
            conn,
            "copy",
            tmp_path / "Track%20A.mp3",
            chromaprint="fp-b",
            duration_sec=180.5,
        )

        report = compare_tracks(conn, "source", "copy")
        reports = find_track_matches(conn, "source")

        assert report.decision == "no_identity_match"
        assert "filename_match_blocked_by_identity_mismatch" in report.warnings
        assert [item.track_b.track_id for item in reports] == ["copy"]
        assert reports[0].decision == "no_identity_match"
        assert reports[0].confidence_score == 0.0
    finally:
        conn.close()


def test_name_similarity_ignores_generic_token_overlap(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "source",
            tmp_path / "source.mp3",
            title="Love To Love You Baby",
            artist="Donna Summer",
        )
        _insert_track(
            conn,
            "generic-overlap",
            tmp_path / "generic.mp3",
            title="Only You",
            artist="Steve Monite",
        )

        report = compare_tracks(conn, "source", "generic-overlap")

        evidence = {item.source: item for item in report.evidence}
        assert evidence["name"].status == "mismatch"
        assert evidence["name"].score < 0.45
    finally:
        conn.close()


def test_name_similarity_caps_single_shared_content_word(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "source",
            tmp_path / "source.mp3",
            title="Love To Love You Baby",
            artist="Donna Summer",
        )
        _insert_track(
            conn,
            "single-word-overlap",
            tmp_path / "single.mp3",
            title="Water Of Love",
            artist="Dire Straits",
        )

        report = compare_tracks(conn, "source", "single-word-overlap")

        evidence = {item.source: item for item in report.evidence}
        assert evidence["name"].score <= 0.42
        assert evidence["name"].status == "mismatch"
    finally:
        conn.close()


def test_find_track_matches_pads_to_five_nearest_candidates(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "source", tmp_path / "source.mp3", title="Source Song")
        for index in range(1, 7):
            _insert_track(
                conn,
                f"candidate-{index}",
                tmp_path / f"candidate-{index}.mp3",
                title=f"Candidate {index}",
                duration_sec=120.0 + index,
            )

        reports = find_track_matches(conn, "source", limit=5)

        assert len(reports) == 5
        scores = [report.candidate_score for report in reports]
        assert scores == sorted(scores, reverse=True)
    finally:
        conn.close()


def test_dissimilar_fingerprint_noise_does_not_rank_candidate(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "source",
            tmp_path / "source.mp3",
            title="Source",
            chromaprint=_encoded_fingerprint([index for index in range(300)]),
        )
        _insert_track(
            conn,
            "different-fp",
            tmp_path / "different.mp3",
            title="Different",
            chromaprint=_encoded_fingerprint([index + 10_000 for index in range(300)]),
        )

        report = compare_tracks(conn, "source", "different-fp")

        evidence = {item.source: item for item in report.evidence}
        assert evidence["fingerprint_similarity"].status == "dissimilar"
        assert report.candidate_scores["fingerprint_similarity"] == 0.0
    finally:
        conn.close()


def test_find_track_matches_uses_artist_metadata_after_title(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "source",
            tmp_path / "source.mp3",
            title="Shared Song",
            artist="Artist A",
            duration_sec=180.0,
        )
        _insert_track(
            conn,
            "same-title-same-artist",
            tmp_path / "same-artist.mp3",
            title="Shared Song",
            artist="Artist A",
            duration_sec=180.0,
        )
        _insert_track(
            conn,
            "same-title-different-artist",
            tmp_path / "different-artist.mp3",
            title="Shared Song",
            artist="Artist B",
            duration_sec=180.0,
        )

        reports = find_track_matches(conn, "source", limit=5)

        assert reports[0].track_b.track_id == "same-title-same-artist"
        assert reports[0].candidate_kind == "same_title_artist"
        assert reports[0].candidate_strength == "strong"
        assert reports[0].candidate_scores["artist_similarity"] == 1.0
        assert reports[1].track_b.track_id == "same-title-different-artist"
        assert reports[1].candidate_scores["artist_similarity"] < 1.0
    finally:
        conn.close()


def test_find_track_matches_uses_feature_similarity_after_name_fingerprint(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "source", tmp_path / "source.mp3", title="Alpha", artist="A")
        _insert_track(conn, "feature-near", tmp_path / "near.mp3", title="Beta", artist="B")
        _insert_track(conn, "feature-far", tmp_path / "far.mp3", title="Gamma", artist="C")
        _insert_audio_features(conn, "source", bpm=120.0, energy=0.8, danceability=0.7)
        _insert_audio_features(conn, "feature-near", bpm=121.0, energy=0.78, danceability=0.72)
        _insert_audio_features(conn, "feature-far", bpm=80.0, energy=0.1, danceability=0.2)

        reports = find_track_matches(conn, "source", limit=5)
        by_id = {report.track_b.track_id: report for report in reports}

        assert by_id["feature-near"].candidate_kind == "feature_similar"
        assert by_id["feature-near"].candidate_strength == "weak"
        assert by_id["feature-near"].candidate_scores["feature_similarity"] > 0.9
        assert (
            by_id["feature-near"].candidate_scores["feature_similarity"]
            > by_id["feature-far"].candidate_scores["feature_similarity"]
        )
    finally:
        conn.close()


def test_find_track_matches_prioritizes_soundwave_before_name(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        source_fp = _encoded_fingerprint([index for index in range(360)])
        similar_fp = _encoded_fingerprint([index for index in range(20, 320)])
        different_fp = _encoded_fingerprint([index + 10_000 for index in range(360)])
        _insert_track(
            conn,
            "source",
            tmp_path / "source.mp3",
            title="Shared Song",
            artist="Artist",
            chromaprint=source_fp,
            duration_sec=180.0,
        )
        _insert_track(
            conn,
            "same-name-similar-fp",
            tmp_path / "same-name-similar.mp3",
            title="Shared Song",
            artist="Artist",
            chromaprint=similar_fp,
            duration_sec=180.5,
        )
        _insert_track(
            conn,
            "same-name-different-fp",
            tmp_path / "same-name-different.mp3",
            title="Shared Song",
            artist="Artist",
            chromaprint=different_fp,
            duration_sec=180.5,
        )
        _insert_track(
            conn,
            "different-name-exact-fp",
            tmp_path / "different-name-exact.mp3",
            title="Different Song",
            artist="Artist",
            chromaprint=source_fp,
            duration_sec=180.5,
        )

        reports = find_track_matches(conn, "source", limit=5)

        assert reports[0].track_b.track_id == "different-name-exact-fp"
        assert reports[0].decision == "same_recording"
        assert reports[1].track_b.track_id == "same-name-similar-fp"
        assert reports[1].candidate_kind == "possible_recording_match"
        assert reports[1].candidate_scores["fingerprint_similarity"] > 0.0
        assert (
            reports[1].candidate_scores["fingerprint_similarity"]
            > reports[2].candidate_scores["fingerprint_similarity"]
        )
    finally:
        conn.close()


def test_match_eval_loader_and_evaluator(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(conn, "track-a", tmp_path / "a.mp3", content_hash="hash-1")
        _insert_track(conn, "track-b", tmp_path / "b.mp3", content_hash="hash-1")
    finally:
        conn.close()
    eval_file = tmp_path / "matches.json"
    eval_file.write_text(
        json.dumps(
            {
                "matches": [
                    {
                        "id": "exact",
                        "track_a": "track-a",
                        "track_b": "track-b",
                        "expectations": {
                            "decision": "exact_duplicate",
                            "identity_decision": "same",
                            "confidence": "high",
                            "min_confidence_score": 1.0,
                            "must_have_evidence": ["content_hash", "audio_embedding"],
                        },
                    }
                ]
            }
        )
    )

    cases = load_match_eval_cases(eval_file)
    conn = connect_db(db_path)
    try:
        result = evaluate_match_case(conn, cases[0])
        summary = aggregate_match_eval_results([result])
    finally:
        conn.close()

    assert cases[0].id == "exact"
    assert result["structured_passed"] is True
    assert summary["avg_structured_pass_rate"] == 1.0
    assert summary["structured_issue_counts"] == {}


def test_eval_matches_cli_json_reports_structured_failures(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(conn, "track-a", tmp_path / "a.mp3", content_hash="hash-1")
        _insert_track(conn, "track-b", tmp_path / "b.mp3", content_hash="hash-1")
    finally:
        conn.close()
    eval_file = tmp_path / "matches.json"
    eval_file.write_text(
        json.dumps(
            {
                "matches": [
                    {
                        "id": "wrong_expectation",
                        "track_a": "track-a",
                        "track_b": "track-b",
                        "expectations": {
                            "decision": "insufficient_evidence",
                            "must_not_decide_identity": True,
                        },
                    }
                ]
            }
        )
    )

    result = CliRunner().invoke(
        app,
        ["eval-matches", str(eval_file), "--db", str(db_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["match_count"] == 1
    assert payload["summary"]["avg_structured_pass_rate"] == 0.0
    assert payload["summary"]["structured_issue_counts"] == {
        "matching_decision": 1,
        "matching_policy": 1,
    }
    assert payload["results"][0]["structured_passed"] is False


def test_compare_tracks_and_match_track_cli_json(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        _insert_track(conn, "track-a", tmp_path / "a.mp3", content_hash="hash-1")
        _insert_track(conn, "track-b", tmp_path / "b.mp3", content_hash="hash-1")
    finally:
        conn.close()

    compare_result = CliRunner().invoke(
        app,
        [
            "compare-tracks",
            "--db",
            str(db_path),
            "--track-a",
            "track-a",
            "--track-b",
            "track-b",
            "--json",
        ],
    )
    assert compare_result.exit_code == 0, compare_result.output
    compare_payload = json.loads(compare_result.output)
    assert compare_payload["report"]["decision"] == "exact_duplicate"

    match_result = CliRunner().invoke(
        app,
        [
            "match-track",
            "--db",
            str(db_path),
            "--track-id",
            "track-a",
            "--against-library",
            "--json",
        ],
    )
    assert match_result.exit_code == 0, match_result.output
    match_payload = json.loads(match_result.output)
    assert match_payload["against_library"] is True
    assert match_payload["count"] == 1
    assert match_payload["reports"][0]["track_b"]["track_id"] == "track-b"


def _encoded_fingerprint(frames: list[int]) -> str:
    payload = array.array("I", frames)
    return chromaprint.encode_fingerprint(payload, 1).decode("ascii")


def _insert_audio_features(
    conn,
    track_id: str,
    *,
    bpm: float | None = None,
    energy: float | None = None,
    danceability: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audio_features (track_id, bpm, energy, danceability, updated_at)
        VALUES (?, ?, ?, ?, '2026-01-01T00:00:00+00:00')
        """,
        (track_id, bpm, energy, danceability),
    )
    conn.commit()


def _insert_audio_embedding(
    conn,
    track_id: str,
    vector: list[float],
    *,
    model: str = "test-audio-model",
) -> None:
    conn.execute(
        """
        INSERT INTO embeddings (track_id, kind, model, dim, vector, text, updated_at)
        VALUES (?, 'audio_clap', ?, ?, ?, NULL, '2026-01-01T00:00:00+00:00')
        """,
        (track_id, model, len(vector), vector_to_blob(vector)),
    )
    conn.commit()


def _insert_track(
    conn,
    track_id: str,
    path: Path,
    *,
    title: str | None = None,
    artist: str | None = None,
    album: str | None = None,
    content_hash: str | None = None,
    chromaprint: str | None = None,
    duration_sec: float | None = None,
    fingerprint_duration: float | None = None,
    missing: bool = False,
) -> None:
    path.write_bytes(b"audio")
    artist_title_norm = f"{artist or ''} {title or ''}".strip().lower() or None
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns,
            title, artist, album, content_hash, chromaprint, duration_sec,
            fingerprint_duration, artist_title_norm, indexed_at, missing_at
        ) VALUES (?, ?, ?, '.mp3', 1, 1, ?, ?, ?, ?, ?, ?, ?, ?,
                  '2026-01-01T00:00:00+00:00', ?)
        """,
        (
            track_id,
            str(path),
            f"hash-{track_id}",
            title,
            artist,
            album,
            content_hash,
            chromaprint,
            duration_sec,
            fingerprint_duration,
            artist_title_norm,
            "2026-01-02T00:00:00+00:00" if missing else None,
        ),
    )
    conn.commit()
