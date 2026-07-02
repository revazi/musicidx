from __future__ import annotations

from musicidx.profile_documents import build_profile_document


def test_profile_document_uses_perceived_bpm_for_double_time_pop_rock(tmp_path):
    document = build_profile_document(
        metadata={
            "title": "Walk Of Life",
            "artist": "Dire Straits",
            "album": "Brothers In Arms",
            "genre": "Rock",
        },
        path=tmp_path / "walk-of-life.mp3",
        profile_text="Artist: Dire Straits. Title: Walk Of Life. Genre: Rock.",
        audio_features={
            "bpm": 172.0,
            "energy": 0.7,
            "danceability": 0.7,
            "aggression": 0.2,
            "brightness": 0.4,
        },
        tags=[{"source": "derived:features", "tag": "fast", "score": 1.0}],
    )

    assert document["musical"]["bpm"] == {
        "value": 86.0,
        "raw_value": 172.0,
        "bucket": "midtempo",
    }
    assert "around 86 BPM" in document["search_text"]["embedding_text"]
    assert "around 172 BPM" not in document["search_text"]["embedding_text"]


def test_profile_document_preserves_high_bpm_for_drum_and_bass(tmp_path):
    document = build_profile_document(
        metadata={
            "title": "Jungle Runner",
            "artist": "Artist A",
            "genre": "Electronic",
        },
        path=tmp_path / "jungle-runner.mp3",
        profile_text="Artist: Artist A. Title: Jungle Runner. Genre: Electronic.",
        audio_features={"bpm": 174.0},
        tags=[{"source": "essentia:genre", "tag": "electronic---drum n bass", "score": 0.9}],
    )

    assert document["musical"]["bpm"] == {"value": 174.0, "bucket": "very_fast"}
    assert "around 174 BPM" in document["search_text"]["embedding_text"]
