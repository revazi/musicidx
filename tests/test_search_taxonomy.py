from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from musicidx.cli import app
from musicidx.search.intent import CONTEXT_PRIORS, DEFAULT_FEATURE_RANGES, QUERY_PRIORS
from musicidx.search.taxonomy import load_search_taxonomy, parse_search_taxonomy


def test_search_taxonomy_command_outputs_bundled_taxonomy_json():
    result = CliRunner().invoke(app, ["search-taxonomy", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["counts"]["contexts"] >= 1
    assert "wedding" in payload["taxonomy"]["contexts"]
    assert payload["taxonomy"]["identity_policy"]["llm"] == "advisory_context_expansion_only"


def test_bundled_search_taxonomy_loads_and_exposes_core_policy():
    taxonomy = load_search_taxonomy()

    assert taxonomy.schema_version == 1
    assert set(taxonomy.feature_ranges) >= {
        "energy",
        "danceability",
        "aggression",
        "brightness",
        "tempo_bpm",
    }
    assert {"wedding", "workout", "driving", "party", "shower"}.issubset(
        taxonomy.occasion_contexts
    )
    assert taxonomy.identity_policy["fingerprint"] == "identity_matching_not_search_ranking"
    assert taxonomy.identity_policy["audio_embedding"] == "similarity_only_not_identity"
    assert taxonomy.identity_policy["llm"] == "advisory_context_expansion_only"


def test_bundled_search_taxonomy_has_phase1_context_vocabulary():
    taxonomy = load_search_taxonomy()

    assert "bar" in taxonomy.contexts
    assert "lounge" in taxonomy.contexts["bar"].keywords
    assert "cooking" in taxonomy.contexts
    assert "dinner" in taxonomy.contexts
    assert "no_vocals_background" in taxonomy.contexts
    assert "no vocals" in taxonomy.contexts["no_vocals_background"].keywords
    assert "ambient" in taxonomy.contexts
    assert "dark" in taxonomy.contexts
    assert "not_aggressive" in taxonomy.query_priors
    assert "aggressive" in taxonomy.query_priors["not_aggressive"].avoid


def test_bundled_search_taxonomy_matches_current_parser_constants():
    taxonomy = load_search_taxonomy()

    assert taxonomy.feature_ranges == {
        field: {level: tuple(bounds) for level, bounds in levels.items()}
        for field, levels in DEFAULT_FEATURE_RANGES.items()
    }
    assert set(taxonomy.contexts) == set(CONTEXT_PRIORS)
    assert set(taxonomy.query_priors) == set(QUERY_PRIORS)

    for name, expected in CONTEXT_PRIORS.items():
        entry = taxonomy.contexts[name]
        assert list(entry.keywords) == expected["keywords"]
        assert list(entry.prefer) == expected["prefer"]
        assert list(entry.avoid) == expected["avoid"]
        assert entry.features == expected["features"]

    for name, expected in QUERY_PRIORS.items():
        entry = taxonomy.query_priors[name]
        assert list(entry.keywords) == expected["keywords"]
        assert list(entry.prefer) == expected["prefer"]
        assert list(entry.avoid) == expected["avoid"]
        assert entry.features == expected["features"]


def test_search_taxonomy_validation_rejects_unknown_feature_level():
    data = load_search_taxonomy().as_dict()
    data["contexts"]["wedding"]["features"]["energy"] = "extreme"

    with pytest.raises(ValueError, match="unknown feature level"):
        parse_search_taxonomy(data)


def test_search_taxonomy_validation_rejects_duplicate_keywords():
    data = load_search_taxonomy().as_dict()
    data["contexts"]["bar"]["keywords"] = ["bar", "BAR"]

    with pytest.raises(ValueError, match="duplicate entry"):
        parse_search_taxonomy(data)
