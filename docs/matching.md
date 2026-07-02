# MusicIdx track matching

MusicIdx matching is deterministic and local-first. SQLite remains the source of truth. The matching helpers do not call an LLM and do not use semantic profile embeddings for identity.

## Commands

```bash
musicidx compare-tracks --track-a <track-id-a> --track-b <track-id-b> --json
musicidx match-track --track-id <track-id> --against-library --json
musicidx duplicates --json
```

`match-track` already searches against indexed library candidates by default; `--against-library` is an explicit compatibility/readability flag. It returns a closest-candidate list, not only identity matches. The desktop shows the top 3–5 candidates inline under the clicked result card.

## MatchReport policy

Authoritative identity evidence:

- `content_hash`
- `chromaprint`

Closest candidates are returned sorted by `candidate_score` descending. The score is built from these signals:

1. `content_hash` / exact `chromaprint` / decoded `fingerprint_similarity`
2. `name` (stop-word-aware; generic single-word overlap is capped)
3. `duration`
4. `artist_similarity` / `artist_title_norm` / `album_similarity` / `filename_stem`
5. `feature_similarity`
6. optional `audio_embedding` with `embeddings.kind = "audio_clap"`

MusicIdx stores both the encoded `chromaprint` string and decoded Chromaprint frame data in SQLite. Running indexing/fingerprinting on an existing DB backfills decoded frames from existing fingerprints, so closest-track matching can do local soundwave alignment without rereading audio files.

Supporting/advisory evidence:

- `name`
- `fingerprint_similarity`
- `duration`
- `artist_similarity`
- `artist_title_norm`
- `album_similarity`
- `filename_stem`
- `feature_similarity`
- `version_conflict`
- optional `audio_embedding` with `embeddings.kind = "audio_clap"`

Never identity evidence:

- name/title matching
- fuzzy fingerprint similarity
- duration alone
- features
- semantic profile embeddings
- audio embeddings alone
- LLM output

## Decisions

| Decision | Meaning |
| --- | --- |
| `exact_duplicate` | Same content hash. This is an authoritative same-file/content decision. |
| `same_recording` | Same chromaprint with similar or missing duration. This is an authoritative same-recording decision. |
| `possible_recording_match` | Same chromaprint but duration differs enough to require review. |
| `possible_metadata_match` | Metadata, filename stem, and/or duration look related, but no content hash or chromaprint proves identity. |
| `related_version_not_duplicate` | Same artist/base title but live/remix/remaster/edit/version tokens conflict, unless authoritative identity evidence overrides it. |
| `sound_similar_only` | Optional audio embedding similarity is high, but it is similarity-only and not identity evidence. |
| `no_identity_match` | Available authoritative identity evidence mismatches. `match-track` may still return this if advisory evidence such as artist/title or filename stem made the pair worth inspecting. |
| `insufficient_evidence` | No decisive or useful matching evidence was available. |

## Important JSON fields

```json
{
  "schema_version": 1,
  "decision": "exact_duplicate",
  "identity_decision": "same",
  "confidence": "high",
  "confidence_score": 1.0,
  "candidate_score": 1.0,
  "candidate_kind": "exact_duplicate",
  "candidate_strength": "strong",
  "candidate_summary": "Exact duplicate: same content hash",
  "candidate_reasons": ["name match 1.00", "content hash match 1.00"],
  "candidate_scores": {"name": 1.0, "content_hash": 1.0},
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

`candidate_kind`, `candidate_strength`, and `candidate_score` rank/label nearest tracks for inspection. They do not prove identity. `identity_decision` is intentionally conservative:

- `same`: only content hash or chromaprint can produce this.
- `possible`: useful but non-authoritative evidence exists.
- `unknown`: identity is not proven.

## Audio embeddings

Filename-stem, artist/album similarity, feature-similarity, and decoded fingerprint-similarity evidence are advisory/candidate-ranking-only. They help the Matches button surface copied/renamed files, close soundwave candidates, remixes/mixtapes/edits with partial overlap, and musically similar tracks. Decoded fingerprint similarity can make a candidate `soundwave_related` or `possible_recording_match`, but only exact content hash or exact `chromaprint` can create `exact_duplicate` / `same_recording`. Dissimilar fuzzy fingerprint scores are ignored for candidate ranking so tiny fingerprint noise does not dominate secondary matches.

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
