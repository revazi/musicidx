"""Optional cloud LLM-backed search intent parsing."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from musicidx.config import (
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_BASE_URL_ENV_VAR,
    GEMINI_MODEL_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_BASE_URL_ENV_VAR,
    OPENAI_MODEL_ENV_VAR,
)
from musicidx.search.intent import (
    CONTEXT_PRIORS,
    DEFAULT_FEATURE_RANGES,
    IntentHints,
    LibraryProfile,
)

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
SUPPORTED_LLM_PROVIDERS = {"gemini", "openai"}


class LLMIntentError(RuntimeError):
    """Raised when LLM intent parsing is unavailable or invalid."""


def is_gemini_configured() -> bool:
    """Return True when a Gemini API key is configured."""
    return bool(os.environ.get(GEMINI_API_KEY_ENV_VAR))


def is_openai_configured() -> bool:
    """Return True when an OpenAI API key is configured."""
    return bool(os.environ.get(OPENAI_API_KEY_ENV_VAR))


def default_gemini_model() -> str:
    """Return the configured Gemini model name."""
    return os.environ.get(GEMINI_MODEL_ENV_VAR, DEFAULT_GEMINI_MODEL)


def default_openai_model() -> str:
    """Return the configured OpenAI model name."""
    return os.environ.get(OPENAI_MODEL_ENV_VAR, DEFAULT_OPENAI_MODEL)


def parse_intent_llm(
    query: str,
    library_profile: LibraryProfile,
    *,
    provider: str = "gemini",
    model: str | None = None,
    timeout_sec: float = 30.0,
) -> IntentHints:
    """Dispatch LLM intent parsing to a supported provider."""
    normalized_provider = provider.strip().lower()
    if normalized_provider == "gemini":
        return parse_intent_gemini(
            query,
            library_profile,
            model=model,
            timeout_sec=timeout_sec,
        )
    if normalized_provider == "openai":
        return parse_intent_openai(
            query,
            library_profile,
            model=model,
            timeout_sec=timeout_sec,
        )
    raise LLMIntentError(
        f"unsupported LLM provider: {provider}; expected one of {sorted(SUPPORTED_LLM_PROVIDERS)}"
    )


def parse_intent_gemini(
    query: str,
    library_profile: LibraryProfile,
    *,
    model: str | None = None,
    timeout_sec: float = 30.0,
) -> IntentHints:
    """Ask Gemini to produce structured intent hints."""
    api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
    if not api_key:
        raise LLMIntentError(f"{GEMINI_API_KEY_ENV_VAR} is not set")

    selected_model = model or default_gemini_model()
    payload = {
        "systemInstruction": {"parts": [{"text": _system_prompt()}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": json.dumps(_intent_request(query, library_profile))}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    request = urllib.request.Request(
        _gemini_generate_content_url(selected_model, api_key),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMIntentError(f"Gemini request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMIntentError(f"Gemini request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMIntentError("Gemini request timed out") from exc

    try:
        data = json.loads(body)
        parts = data["candidates"][0]["content"]["parts"]
        content = "".join(part.get("text", "") for part in parts)
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMIntentError("Gemini returned invalid intent JSON") from exc

    return intent_hints_from_json(parsed)


def parse_intent_openai(
    query: str,
    library_profile: LibraryProfile,
    *,
    model: str | None = None,
    timeout_sec: float = 30.0,
) -> IntentHints:
    """Ask OpenAI to produce structured intent hints."""
    api_key = os.environ.get(OPENAI_API_KEY_ENV_VAR)
    if not api_key:
        raise LLMIntentError(f"{OPENAI_API_KEY_ENV_VAR} is not set")

    selected_model = model or default_openai_model()
    payload = {
        "model": selected_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": json.dumps(_intent_request(query, library_profile))},
        ],
    }

    request = urllib.request.Request(
        _openai_chat_completions_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMIntentError(f"OpenAI request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMIntentError(f"OpenAI request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMIntentError("OpenAI request timed out") from exc

    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMIntentError("OpenAI returned invalid intent JSON") from exc

    return intent_hints_from_json(parsed)


def intent_hints_from_json(data: dict[str, Any]) -> IntentHints:
    """Validate and sanitize LLM JSON into IntentHints."""
    if not isinstance(data, dict):
        raise LLMIntentError("LLM intent output must be an object")

    contexts = [
        context
        for context in _string_list(data.get("contexts"))
        if context in CONTEXT_PRIORS
    ]
    prefer = _string_list(data.get("prefer_tag_concepts"))
    avoid = _string_list(data.get("avoid_tag_concepts"))
    feature_preferences = _feature_preferences(data.get("feature_preferences"))
    limit = _safe_limit(data.get("limit"))
    notes = data.get("notes") if isinstance(data.get("notes"), str) else None

    return IntentHints(
        contexts=contexts,
        prefer_tag_concepts=prefer,
        avoid_tag_concepts=avoid,
        feature_preferences=feature_preferences,
        limit=limit,
        notes=notes,
    )


def _intent_request(query: str, library_profile: LibraryProfile) -> dict[str, Any]:
    return {
        "query": query,
        "library_profile": library_profile.as_dict(),
        "allowed_contexts": sorted(CONTEXT_PRIORS.keys()),
        "allowed_feature_fields": sorted(DEFAULT_FEATURE_RANGES.keys()),
        "allowed_feature_levels": sorted(
            {level for ranges in DEFAULT_FEATURE_RANGES.values() for level in ranges}
        ),
    }


def _system_prompt() -> str:
    return """
You are a music search intent parser.

Return only JSON. Do not recommend tracks. Do not invent artists, albums, or songs.
Only translate the user query into search hints for a local music database.

Use this schema:
{
  "limit": integer or null,
  "contexts": [string],
  "prefer_tag_concepts": [string],
  "avoid_tag_concepts": [string],
  "feature_preferences": {
    "energy": "very_low|low|low_mid|mid|mid_high|high|very_high",
    "danceability": "very_low|low|low_mid|mid|mid_high|high|very_high",
    "aggression": "very_low|low|low_mid|mid|mid_high|high|very_high",
    "brightness": "very_low|low|low_mid|mid|mid_high|high|very_high",
    "tempo_bpm": "very_low|low|low_mid|mid|mid_high|high|very_high"
  },
  "notes": string
}

Prefer tag concepts that exist or are semantically close to the provided library tags.
If unsure, produce broad mood/genre concepts rather than track recommendations.
""".strip()


def _gemini_generate_content_url(model: str, api_key: str) -> str:
    base_url = os.environ.get(GEMINI_BASE_URL_ENV_VAR, DEFAULT_GEMINI_BASE_URL).rstrip("/")
    encoded_model = urllib.parse.quote(model, safe="")
    query = urllib.parse.urlencode({"key": api_key})
    return f"{base_url}/models/{encoded_model}:generateContent?{query}"


def _openai_chat_completions_url() -> str:
    base_url = os.environ.get(OPENAI_BASE_URL_ENV_VAR, DEFAULT_OPENAI_BASE_URL).rstrip("/")
    return f"{base_url}/chat/completions"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output[:30]


def _feature_preferences(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, str] = {}
    for field_name, level in value.items():
        if not isinstance(field_name, str) or not isinstance(level, str):
            continue
        normalized_field = field_name.strip().lower()
        normalized_level = level.strip().lower()
        if normalized_field not in DEFAULT_FEATURE_RANGES:
            continue
        if normalized_level not in DEFAULT_FEATURE_RANGES[normalized_field]:
            continue
        output[normalized_field] = normalized_level
    return output


def _safe_limit(value: Any) -> int | None:
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, min(100, limit))
