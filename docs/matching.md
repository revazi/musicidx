# MusicIdx track matching

MusicIdx matching is deterministic and local-first. SQLite remains the source of truth. The matching helpers do not call an LLM and do not use semantic profile embeddings for identity.

## Commands

```bash
musicidx compare-tracks --track-a <track-id-a> --track-b <track-id-b> --json
musicidx match-track --track-id <track-id> --against-library --json
musicidx duplicates --json
```

`match-track` already searches against indexed library candidates by default; `--against-library` is an explicit compatibility/readability flag.

## MatchReport policy

Authoritative identity evidence:

- `content_hash`
- `chromaprint`

Supporting/advisory evidence:

- `duration`
- `artist_title_norm`
- `version_conflict`
- optional `audio_embedding` with `embeddings.kind = "audio_clap"`

Never identity evidence:

- semantic profile embeddings
- audio embeddings alone
- LLM output

## Decisions

| Decision | Meaning |
| --- | --- |
| `exact_duplicate` | Same content hash. This is an authoritative same-file/content decision. |
| `same_recording` | Same chromaprint with similar or missing duration. This is an authoritative same-recording decision. |
| `possible_recording_match` | Same chromaprint but duration differs enough to require review. |
| `possible_metadata_match` | Metadata/duration look related, but no content hash or chromaprint proves identity. |
| `related_version_not_duplicate` | Same artist/base title but live/remix/remaster/edit/version tokens conflict, unless authoritative identity evidence overrides it. |
| `sound_similar_only` | Optional audio embedding similarity is high, but it is similarity-only and not identity evidence. |
| `no_identity_match` | Available authoritative identity evidence mismatches. |
| `insufficient_evidence` | No decisive or useful matching evidence was available. |

## Important JSON fields

```json
{
  "schema_version": 1,
  "decision": "exact_duplicate",
  "identity_decision": "same",
  "confidence": "high",
  "confidence_score": 1.0,
  "reasons": ["same content hash"],
  "warnings": [],
  "evidence": [],
  "policy": {
    "identity_authority": ["content_hash", "chromaprint"],
    "audio_embeddings": "similarity_only_not_identity",
    "semantic_embeddings": "not_used_for_identity",
    "llm": "not_used"
  }
}
```

`identity_decision` is intentionally conservative:

- `same`: only content hash or chromaprint can produce this.
- `possible`: useful but non-authoritative evidence exists.
- `unknown`: identity is not proven.

## Audio embeddings

Audio embeddings are optional and experimental. If stored under `embeddings.kind = "audio_clap"`, matching computes cosine similarity for shared models and may return `sound_similar_only`.

Audio embeddings cannot:

- create `exact_duplicate`
- create `same_recording`
- override `content_hash` mismatch
- override `chromaprint` mismatch
- override a version/remix/live conflict

## Regression eval

Use match eval files to lock down decisions:

```bash
musicidx eval-matches <match-eval.json> --json
```

Example shape:

```json
{
  "matches": [
    {
      "id": "exact_duplicate",
      "track_a": "track-a",
      "track_b": "track-b",
      "expectations": {
        "decision": "exact_duplicate",
        "identity_decision": "same",
        "must_have_evidence": ["content_hash"]
      }
    }
  ]
}
```

See `eval/matching_regressions.example.json` for a placeholder template.
