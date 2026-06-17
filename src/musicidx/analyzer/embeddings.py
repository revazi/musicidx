"""Text embeddings over enriched track profile text."""

from __future__ import annotations

import importlib.util
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from musicidx.db import utc_now

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_KIND = "profile_text"


class EmbeddingError(RuntimeError):
    """Raised when profile embedding cannot be completed."""


@dataclass(slots=True)
class EmbeddingSummary:
    """Summary counters for an embedding run."""

    processed: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    model: str = DEFAULT_EMBEDDING_MODEL
    kind: str = EMBEDDING_KIND
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SemanticSearchResult:
    """One semantic profile search result."""

    track_id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    profile_text: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_sentence_transformers_available() -> bool:
    """Return True when sentence-transformers is importable."""
    return importlib.util.find_spec("sentence_transformers") is not None


def embed_texts(texts: list[str], *, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Embed text with sentence-transformers and return normalized float32 vectors."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError("sentence-transformers is not installed") from exc

    try:
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # pragma: no cover - model loading/runtime errors vary
        raise EmbeddingError(f"failed to embed text with {model_name}: {exc}") from exc

    return _ensure_2d_float32(vectors)


def process_embeddings(
    conn: sqlite3.Connection,
    *,
    track_id: str | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    refresh: bool = False,
) -> EmbeddingSummary:
    """Embed enriched track profiles and persist vectors."""
    summary = EmbeddingSummary(model=model_name)
    rows = _select_profiles_for_embedding(conn, track_id=track_id)
    pending: list[sqlite3.Row] = []

    for row in rows:
        is_current = _embedding_is_current(
            conn,
            row["track_id"],
            model_name,
            row["profile_text"],
        )
        if not refresh and is_current:
            summary.skipped += 1
            continue
        pending.append(row)

    for batch in _chunks(pending, max(1, batch_size)):
        texts = [row["profile_text"] for row in batch]
        try:
            vectors = embed_texts(texts, model_name=model_name)
            if len(vectors) != len(batch):
                raise EmbeddingError("embedding model returned unexpected vector count")
        except EmbeddingError as exc:
            summary.errors += len(batch)
            summary.last_error = str(exc)
            continue

        for row, vector in zip(batch, vectors, strict=True):
            summary.processed += 1
            save_profile_embedding(
                conn,
                row["track_id"],
                row["profile_text"],
                vector,
                model_name=model_name,
            )
            summary.updated += 1

    conn.commit()
    return summary


def save_profile_embedding(
    conn: sqlite3.Connection,
    track_id: str,
    profile_text: str,
    vector: Any,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """Persist a profile-text embedding vector."""
    normalized = normalize_vector(vector)
    now = utc_now()
    conn.execute(
        """
        INSERT INTO embeddings (track_id, kind, model, dim, vector, text, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(track_id, kind, model) DO UPDATE SET
            dim = excluded.dim,
            vector = excluded.vector,
            text = excluded.text,
            updated_at = excluded.updated_at
        """,
        (
            track_id,
            EMBEDDING_KIND,
            model_name,
            int(normalized.shape[0]),
            vector_to_blob(normalized),
            profile_text,
            now,
        ),
    )


def search_semantic(
    conn: sqlite3.Connection,
    query: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 10,
    include_missing: bool = False,
) -> list[SemanticSearchResult]:
    """Search profile embeddings with brute-force cosine similarity."""
    rows = _select_embeddings_for_search(
        conn,
        model_name=model_name,
        include_missing=include_missing,
    )
    if not rows:
        return []

    query_vector = normalize_vector(embed_texts([query], model_name=model_name)[0])
    matrix = np.vstack([blob_to_vector(row["vector"], row["dim"]) for row in rows])
    matrix = normalize_matrix(matrix)
    scores = matrix @ query_vector
    ranked_indices = np.argsort(scores)[::-1][:limit]

    results: list[SemanticSearchResult] = []
    for index in ranked_indices:
        row = rows[int(index)]
        results.append(
            SemanticSearchResult(
                track_id=row["track_id"],
                path=row["path"],
                title=row["title"],
                artist=row["artist"],
                album=row["album"],
                genre=row["genre"],
                profile_text=row["profile_text"],
                score=round(float(scores[int(index)]), 6),
            )
        )
    return results


def vector_to_blob(vector: Any) -> bytes:
    """Serialize a vector as float32 bytes."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize a float32 vector blob."""
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.shape[0] != dim:
        raise EmbeddingError(
            f"stored vector dimension mismatch: expected {dim}, got {vector.shape[0]}"
        )
    return vector


def normalize_vector(vector: Any) -> np.ndarray:
    """Return a finite, unit-normalized float32 vector."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise EmbeddingError("embedding vector was empty")
    if not np.all(np.isfinite(array)):
        raise EmbeddingError("embedding vector contained non-finite values")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise EmbeddingError("embedding vector had zero norm")
    return (array / norm).astype(np.float32)


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows of a matrix for cosine scoring."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def _select_profiles_for_embedding(
    conn: sqlite3.Connection,
    *,
    track_id: str | None,
) -> list[sqlite3.Row]:
    clauses = ["t.missing_at IS NULL", "t.quarantined_at IS NULL"]
    params: list[Any] = []
    if track_id is not None:
        clauses.append("t.id = ?")
        params.append(track_id)

    return conn.execute(
        f"""
        SELECT t.id AS track_id, p.profile_text
        FROM track_profiles p
        JOIN tracks t ON t.id = p.track_id
        WHERE {' AND '.join(clauses)}
        ORDER BY t.path
        """,
        params,
    ).fetchall()


def _select_embeddings_for_search(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    include_missing: bool,
) -> list[sqlite3.Row]:
    missing_clause = "" if include_missing else "AND t.missing_at IS NULL"
    return conn.execute(
        f"""
        SELECT
            e.track_id,
            e.vector,
            e.dim,
            t.path,
            t.title,
            t.artist,
            t.album,
            t.genre,
            p.profile_text
        FROM embeddings e
        JOIN tracks t ON t.id = e.track_id
        JOIN track_profiles p ON p.track_id = e.track_id
        WHERE e.kind = ?
          AND e.model = ?
          {missing_clause}
        ORDER BY t.path
        """,
        (EMBEDDING_KIND, model_name),
    ).fetchall()


def _embedding_is_current(
    conn: sqlite3.Connection,
    track_id: str,
    model_name: str,
    profile_text: str,
) -> bool:
    row = conn.execute(
        """
        SELECT text
        FROM embeddings
        WHERE track_id = ? AND kind = ? AND model = ?
        """,
        (track_id, EMBEDDING_KIND, model_name),
    ).fetchone()
    return row is not None and row["text"] == profile_text


def _chunks(rows: list[sqlite3.Row], size: int) -> list[list[sqlite3.Row]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _ensure_2d_float32(vectors: Any) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise EmbeddingError("embedding model returned an invalid vector shape")
    return np.asarray([normalize_vector(vector) for vector in array], dtype=np.float32)

