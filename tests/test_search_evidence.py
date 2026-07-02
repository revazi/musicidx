from __future__ import annotations

from musicidx.search.evidence import build_candidate_evidence


def test_build_candidate_evidence_summarizes_sources_and_identity():
    evidence = build_candidate_evidence(
        {
            "metadata_score": 0.8,
            "metadata_matches": [{"field": "title", "value": "Blue"}],
            "tag_score": 0.7,
            "matched_tags": [{"tag": "ambient", "score": 0.9}],
            "context_score": 0.0,
            "feature_score": 0.4,
            "feature_reasons": ["energy in range"],
            "text_score": 0.5,
            "direct_content_evidence": {
                "matched_by_source": {"profile": ["blue"], "metadata": ["blue"]}
            },
            "semantic_score": 0.3,
            "feedback_score": -1.0,
            "evidence": {"semantic_only": False},
            "identity": {
                "content_hash": "abc",
                "chromaprint": None,
                "duration_sec": 123.0,
                "artist_title_norm": "artist title",
            },
        }
    )

    assert evidence["retrieved_by"] == [
        "metadata",
        "profile_text",
        "tags",
        "audio_features",
        "semantic_profile",
        "feedback",
    ]
    assert evidence["identity"] == {
        "content_hash": True,
        "chromaprint": False,
        "duration_sec": True,
        "artist_title_norm": True,
    }
    assert evidence["semantic_only"] is False
    sources = {source["source"]: source for source in evidence["sources"]}
    assert sources["metadata"]["details"] == {"matches": 1}
    assert sources["profile_text"]["details"] == {"matches": 1}
    assert sources["feedback"]["role"] == "rerank_adjustment"
    assert sources["feedback"]["details"] == {"direction": "negative"}


def test_build_candidate_evidence_marks_semantic_only():
    evidence = build_candidate_evidence(
        {
            "semantic_score": 0.6,
            "evidence": {"semantic_only": True},
            "identity": {},
        }
    )

    assert evidence["retrieved_by"] == ["semantic_profile"]
    assert evidence["semantic_only"] is True
    assert evidence["sources"] == [
        {
            "source": "semantic_profile",
            "role": "candidate_evidence",
            "score": 0.6,
            "matched": True,
            "details": {},
        }
    ]
