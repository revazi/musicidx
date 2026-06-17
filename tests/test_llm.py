from __future__ import annotations

import io
import json
import urllib.error

from musicidx.search.intent import LibraryProfile
from musicidx.search.llm import intent_hints_from_json, parse_intent_gemini, parse_intent_openai


def test_intent_hints_from_json_sanitizes_invalid_values():
    hints = intent_hints_from_json(
        {
            "limit": 500,
            "contexts": ["focus", "unknown"],
            "prefer_tag_concepts": ["Ambient", "ambient", "Deep"],
            "avoid_tag_concepts": ["Aggressive"],
            "feature_preferences": {
                "energy": "low_mid",
                "tempo_bpm": "mid",
                "unknown": "high",
                "brightness": "invalid",
            },
            "sort_by": [
                {"field": "bpm", "direction": "desc"},
                {"field": "unknown", "direction": "asc"},
                {"field": "energy", "direction": "sideways"},
            ],
            "notes": "library-aware intent",
        }
    )

    assert hints.limit == 100
    assert hints.contexts == ["focus"]
    assert hints.prefer_tag_concepts == ["ambient", "deep"]
    assert hints.avoid_tag_concepts == ["aggressive"]
    assert hints.feature_preferences == {"energy": "low_mid", "tempo_bpm": "mid"}
    assert [sort.as_dict() for sort in hints.sort_by] == [
        {"field": "tempo_bpm", "direction": "desc", "source": "llm"}
    ]
    assert hints.notes == "library-aware intent"


def test_parse_intent_gemini_uses_generate_content(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            content = json.dumps(
                {
                    "limit": 8,
                    "contexts": ["focus"],
                    "prefer_tag_concepts": ["ambient", "background"],
                    "avoid_tag_concepts": ["aggressive"],
                    "feature_preferences": {"energy": "low_mid", "aggression": "low"},
                    "sort_by": [{"field": "tempo_bpm", "direction": "desc"}],
                    "notes": "gemini test",
                }
            )
            payload = {"candidates": [{"content": {"parts": [{"text": content}]}}]}
            return json.dumps(payload).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr("musicidx.search.llm.urllib.request.urlopen", fake_urlopen)

    profile = LibraryProfile(
        total_tracks=0,
        tag_stats={},
        feature_percentiles={},
        embedding_models=[],
    )
    hints = parse_intent_gemini("focus music", profile, model="gemini-test", timeout_sec=6)

    assert "models/gemini-test:generateContent" in captured["url"]
    assert "key=test-gemini-key" in captured["url"]
    assert captured["timeout"] == 6
    system_prompt = captured["body"]["systemInstruction"]["parts"][0]["text"]
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "Always produce a usable music search intent" in system_prompt
    assert (
        "Do not return empty hints unless the user explicitly asks not to search"
        in system_prompt
    )
    assert hints.limit == 8
    assert hints.contexts == ["focus"]
    assert hints.prefer_tag_concepts == ["ambient", "background"]
    assert hints.feature_preferences == {"energy": "low_mid", "aggression": "low"}
    assert [sort.as_dict() for sort in hints.sort_by] == [
        {"field": "tempo_bpm", "direction": "desc", "source": "llm"}
    ]


def test_parse_intent_gemini_falls_back_when_configured_model_is_retired(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    seen_urls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            content = json.dumps({"contexts": ["focus"], "prefer_tag_concepts": ["ambient"]})
            payload = {"candidates": [{"content": {"parts": [{"text": content}]}}]}
            return json.dumps(payload).encode()

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        if "gemini-1.5-flash" in request.full_url:
            raise urllib.error.HTTPError(
                request.full_url,
                404,
                "not found",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"retired"}}'),
            )
        return FakeResponse()

    monkeypatch.setattr("musicidx.search.llm.urllib.request.urlopen", fake_urlopen)

    profile = LibraryProfile(
        total_tracks=0,
        tag_stats={},
        feature_percentiles={},
        embedding_models=[],
    )
    hints = parse_intent_gemini("focus music", profile, model="gemini-1.5-flash")

    assert any("models/gemini-1.5-flash:generateContent" in url for url in seen_urls)
    assert any("models/gemini-2.0-flash:generateContent" in url for url in seen_urls)
    assert hints.contexts == ["focus"]
    assert hints.prefer_tag_concepts == ["ambient"]


def test_parse_intent_openai_uses_chat_completions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            content = json.dumps(
                {
                    "limit": 7,
                    "contexts": ["chill"],
                    "prefer_tag_concepts": ["relaxing", "ambient"],
                    "avoid_tag_concepts": ["aggressive"],
                    "feature_preferences": {"energy": "low_mid", "aggression": "low"},
                    "notes": "test",
                }
            )
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())
        captured["auth"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setattr("musicidx.search.llm.urllib.request.urlopen", fake_urlopen)

    profile = LibraryProfile(
        total_tracks=0,
        tag_stats={},
        feature_percentiles={},
        embedding_models=[],
    )
    hints = parse_intent_openai("chill music", profile, model="gpt-test", timeout_sec=5)

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["timeout"] == 5
    system_prompt = captured["body"]["messages"][0]["content"]
    assert captured["body"]["model"] == "gpt-test"
    assert "Always produce a usable music search intent" in system_prompt
    assert (
        "Do not return empty hints unless the user explicitly asks not to search"
        in system_prompt
    )
    assert captured["auth"] == "Bearer test-key"
    assert hints.limit == 7
    assert hints.contexts == ["chill"]
    assert hints.prefer_tag_concepts == ["relaxing", "ambient"]
    assert hints.feature_preferences == {"energy": "low_mid", "aggression": "low"}
