"""Optional Essentia ML tag analysis using local model files."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from musicidx.db import utc_now
from musicidx.failures import clear_track_failure, record_track_error
from musicidx.profiles import rebuild_track_profile

MANIFEST_FILENAME = "manifest.json"
DEFAULT_MIN_SCORE = 0.20
DEFAULT_TOP_K = 10
SUPPORTED_PROFILES = {"musicnn_classifier", "effnet_classifier", "direct_2d"}


class EssentiaModelError(RuntimeError):
    """Raised when Essentia model analysis cannot be completed."""


@dataclass(slots=True)
class TrackTag:
    """One tag predicted for a track."""

    source: str
    tag: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EssentiaModelSpec:
    """Manifest description for one local Essentia model pipeline."""

    name: str
    kind: str
    profile: str
    labels: list[str]
    sample_rate: int = 16000
    top_k: int = DEFAULT_TOP_K
    min_score: float | None = None
    embedding_model: Path | None = None
    embedding_output: str | None = None
    classifier_model: Path | None = None
    classifier_input: str | None = None
    classifier_output: str | None = None
    model: Path | None = None
    input: str | None = None
    output: str | None = None

    @property
    def source(self) -> str:
        return f"essentia:{self.name}"

    def model_files(self) -> list[Path]:
        files: list[Path] = []
        if self.embedding_model is not None:
            files.append(self.embedding_model)
        if self.classifier_model is not None:
            files.append(self.classifier_model)
        if self.model is not None:
            files.append(self.model)
        return files

    def missing_files(self) -> list[Path]:
        return [path for path in self.model_files() if not path.exists()]

    def is_available(self) -> bool:
        return bool(self.labels) and self.profile in SUPPORTED_PROFILES and not self.missing_files()

    def as_status_dict(self) -> dict[str, Any]:
        missing_files = [str(path) for path in self.missing_files()]
        return {
            "name": self.name,
            "kind": self.kind,
            "profile": self.profile,
            "source": self.source,
            "available": self.is_available(),
            "labels": len(self.labels),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "model_files": [str(path) for path in self.model_files()],
            "missing_files": missing_files,
        }


@dataclass(slots=True)
class TagAnalysisSummary:
    """Summary counters for an ML tag analysis run."""

    processed: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    model_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class StoredTrackTag:
    """Stored tag row for CLI display."""

    track_id: str
    source: str
    tag: str
    score: float
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelManifestStatus:
    """Status information for local model manifests."""

    models_path: str
    manifest_path: str
    manifest_exists: bool
    essentia_available: bool
    models: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_essentia_available() -> bool:
    """Return True when the Essentia Python package is importable."""
    return importlib.util.find_spec("essentia") is not None


def model_manifest_status(models_path: Path) -> ModelManifestStatus:
    """Return status for the local model manifest and known models."""
    manifest_path = models_path / MANIFEST_FILENAME
    errors: list[str] = []
    specs: list[EssentiaModelSpec] = []
    if manifest_path.exists():
        try:
            specs = load_model_specs(models_path)
        except EssentiaModelError as exc:
            errors.append(str(exc))

    return ModelManifestStatus(
        models_path=str(models_path),
        manifest_path=str(manifest_path),
        manifest_exists=manifest_path.exists(),
        essentia_available=is_essentia_available(),
        models=[spec.as_status_dict() for spec in specs],
        errors=errors,
    )


def load_model_specs(models_path: Path) -> list[EssentiaModelSpec]:
    """Load local Essentia model specs from manifest.json."""
    manifest_path = models_path / MANIFEST_FILENAME
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise EssentiaModelError(f"invalid model manifest JSON: {manifest_path}") from exc

    model_entries = manifest.get("models") if isinstance(manifest, dict) else manifest
    if not isinstance(model_entries, list):
        raise EssentiaModelError("model manifest must contain a 'models' list")

    specs: list[EssentiaModelSpec] = []
    for entry in model_entries:
        if not isinstance(entry, dict):
            raise EssentiaModelError("each model manifest entry must be an object")
        specs.append(_model_spec_from_manifest_entry(models_path, entry))
    return specs


def available_model_specs(models_path: Path) -> list[EssentiaModelSpec]:
    """Return model specs with all required local files present."""
    return [spec for spec in load_model_specs(models_path) if spec.is_available()]


def analyze_essentia_tags(
    path: Path,
    specs: list[EssentiaModelSpec],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[TrackTag]:
    """Run available Essentia model specs and return predicted tags."""
    if not specs:
        raise EssentiaModelError("no available local Essentia model specs found")

    try:
        import essentia.standard as es  # type: ignore[import-untyped]
        import numpy as np
    except ImportError as exc:
        raise EssentiaModelError("essentia is not installed") from exc

    tags: list[TrackTag] = []
    for spec in specs:
        try:
            predictions = _run_model_spec(es, path, spec)
            scores = _scores_from_predictions(predictions, np=np)
            tags.extend(_tags_from_scores(spec, scores, min_score=min_score))
        except EssentiaModelError:
            raise
        except Exception as exc:  # pragma: no cover - Essentia exceptions vary by version/model
            raise EssentiaModelError(f"{spec.name}: model inference failed: {exc}") from exc

    return _deduplicate_tags(tags)


def process_tags(
    conn: sqlite3.Connection,
    *,
    models_path: Path,
    track_id: str | None = None,
    track_ids: list[str] | None = None,
    missing_only: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
    workers: int = 1,
) -> TagAnalysisSummary:
    """Analyze selected tracks with local ML tag models."""
    rows = _select_tracks_for_tags(
        conn,
        track_id=track_id,
        track_ids=track_ids,
        missing_only=missing_only,
    )
    specs = available_model_specs(models_path)
    summary = TagAnalysisSummary(model_count=len(specs))

    if not specs:
        summary.errors = len(rows)
        for row in rows:
            record_track_error(conn, row["id"], "no available local Essentia model specs found")
        conn.commit()
        return summary

    if workers <= 1:
        for row in rows:
            _process_one_tag_job(conn, row, specs, summary, min_score=min_score)
        conn.commit()
        return summary

    pending_jobs: list[sqlite3.Row] = []
    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            summary.skipped += 1
            record_track_error(conn, row["id"], "file is missing on disk")
            continue
        pending_jobs.append(row)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {
            executor.submit(
                analyze_essentia_tags,
                Path(row["path"]),
                specs,
                min_score=min_score,
            ): row
            for row in pending_jobs
        }
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            summary.processed += 1
            try:
                tags = future.result()
                save_track_tags(conn, row["id"], tags, sources=[spec.source for spec in specs])
                summary.updated += 1
            except EssentiaModelError as exc:
                summary.errors += 1
                record_track_error(conn, row["id"], str(exc))
            except Exception as exc:  # pragma: no cover - defensive safety net
                summary.errors += 1
                record_track_error(conn, row["id"], f"unexpected tag analysis error: {exc}")

    conn.commit()
    return summary


def save_track_tags(
    conn: sqlite3.Connection,
    track_id: str,
    tags: list[TrackTag],
    *,
    sources: list[str] | None = None,
) -> None:
    """Persist ML tags for one track and refresh profile text."""
    now = utc_now()
    source_names = sorted(set(sources or [tag.source for tag in tags]))
    for source in source_names:
        conn.execute("DELETE FROM track_tags WHERE track_id = ? AND source = ?", (track_id, source))

    for tag in tags:
        conn.execute(
            """
            INSERT INTO track_tags (track_id, source, tag, score, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(track_id, source, tag) DO UPDATE SET
                score = excluded.score,
                updated_at = excluded.updated_at
            """,
            (track_id, tag.source, tag.tag, tag.score, now),
        )

    clear_track_failure(conn, track_id)
    rebuild_track_profile(conn, track_id, updated_at=now)


def list_track_tags(conn: sqlite3.Connection, *, track_id: str) -> list[StoredTrackTag]:
    """List stored tags for one track."""
    rows = conn.execute(
        """
        SELECT track_id, source, tag, score, updated_at
        FROM track_tags
        WHERE track_id = ?
        ORDER BY score DESC, tag ASC, source ASC
        """,
        (track_id,),
    ).fetchall()
    return [
        StoredTrackTag(
            track_id=row["track_id"],
            source=row["source"],
            tag=row["tag"],
            score=float(row["score"]),
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def _model_spec_from_manifest_entry(models_path: Path, entry: dict[str, Any]) -> EssentiaModelSpec:
    name = _required_str(entry, "name")
    profile = _required_str(entry, "profile")
    if profile not in SUPPORTED_PROFILES:
        raise EssentiaModelError(f"unsupported Essentia model profile for {name}: {profile}")

    labels = _load_labels(models_path, entry)
    return EssentiaModelSpec(
        name=name,
        kind=str(entry.get("kind") or "tag"),
        profile=profile,
        labels=labels,
        sample_rate=int(entry.get("sample_rate") or 16000),
        top_k=int(entry.get("top_k") or DEFAULT_TOP_K),
        min_score=_optional_float(entry.get("min_score")),
        embedding_model=_optional_path(models_path, entry.get("embedding_model")),
        embedding_output=_optional_str(entry.get("embedding_output")),
        classifier_model=_optional_path(models_path, entry.get("classifier_model")),
        classifier_input=_optional_str(entry.get("classifier_input")),
        classifier_output=_optional_str(entry.get("classifier_output")),
        model=_optional_path(models_path, entry.get("model")),
        input=_optional_str(entry.get("input")),
        output=_optional_str(entry.get("output")),
    )


def _load_labels(models_path: Path, entry: dict[str, Any]) -> list[str]:
    labels = entry.get("labels")
    if isinstance(labels, list):
        return [_normalize_tag_name(str(label)) for label in labels if str(label).strip()]

    labels_file = entry.get("labels_file")
    if not isinstance(labels_file, str) or not labels_file.strip():
        return []

    path = _resolve_model_path(models_path, labels_file)
    if not path.exists():
        return []

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [_normalize_tag_name(str(label)) for label in data if str(label).strip()]
        if isinstance(data, dict):
            for key in ["classes", "labels", "class_names", "classes_names"]:
                value = data.get(key)
                if isinstance(value, list):
                    return [
                        _normalize_tag_name(str(label))
                        for label in value
                        if str(label).strip()
                    ]
        return []

    return [_normalize_tag_name(line) for line in path.read_text().splitlines() if line.strip()]


def _run_model_spec(es: Any, path: Path, spec: EssentiaModelSpec) -> Any:
    audio = es.MonoLoader(
        filename=str(path),
        sampleRate=spec.sample_rate,
        resampleQuality=4,
    )()

    if spec.profile == "musicnn_classifier":
        if spec.embedding_model is None or spec.classifier_model is None:
            raise EssentiaModelError(f"{spec.name}: missing embedding/classifier model path")
        embeddings = es.TensorflowPredictMusiCNN(
            **_algorithm_kwargs(spec.embedding_model, output=spec.embedding_output)
        )(audio)
        return es.TensorflowPredict2D(
            **_algorithm_kwargs(
                spec.classifier_model,
                input_name=spec.classifier_input,
                output=spec.classifier_output,
            )
        )(embeddings)

    if spec.profile == "effnet_classifier":
        if spec.embedding_model is None or spec.classifier_model is None:
            raise EssentiaModelError(f"{spec.name}: missing embedding/classifier model path")
        embeddings = es.TensorflowPredictEffnetDiscogs(
            **_algorithm_kwargs(spec.embedding_model, output=spec.embedding_output)
        )(audio)
        return es.TensorflowPredict2D(
            **_algorithm_kwargs(
                spec.classifier_model,
                input_name=spec.classifier_input,
                output=spec.classifier_output,
            )
        )(embeddings)

    if spec.profile == "direct_2d":
        if spec.model is None:
            raise EssentiaModelError(f"{spec.name}: missing model path")
        return es.TensorflowPredict2D(
            **_algorithm_kwargs(spec.model, input_name=spec.input, output=spec.output)
        )(audio)

    raise EssentiaModelError(f"{spec.name}: unsupported profile {spec.profile}")


def _algorithm_kwargs(
    graph_filename: Path,
    *,
    input_name: str | None = None,
    output: str | None = None,
) -> dict[str, str]:
    kwargs = {"graphFilename": str(graph_filename)}
    if input_name:
        kwargs["input"] = input_name
    if output:
        kwargs["output"] = output
    return kwargs


def _scores_from_predictions(predictions: Any, *, np: Any) -> list[float]:
    array = np.asarray(predictions, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        array = array.mean(axis=tuple(range(array.ndim - 1)))
    return [float(value) for value in array.tolist()]


def _tags_from_scores(
    spec: EssentiaModelSpec,
    scores: list[float],
    *,
    min_score: float,
) -> list[TrackTag]:
    threshold = spec.min_score if spec.min_score is not None else min_score
    pairs = [
        (label, score)
        for label, score in zip(spec.labels, scores, strict=False)
        if score >= threshold
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)

    return [
        TrackTag(source=spec.source, tag=label, score=round(float(score), 6))
        for label, score in pairs[: spec.top_k]
    ]


def _deduplicate_tags(tags: list[TrackTag]) -> list[TrackTag]:
    best: dict[tuple[str, str], TrackTag] = {}
    for tag in tags:
        key = (tag.source, tag.tag)
        existing = best.get(key)
        if existing is None or tag.score > existing.score:
            best[key] = tag
    return sorted(best.values(), key=lambda tag: (-tag.score, tag.source, tag.tag))


def _select_tracks_for_tags(
    conn: sqlite3.Connection,
    *,
    track_id: str | None,
    track_ids: list[str] | None = None,
    missing_only: bool,
) -> list[sqlite3.Row]:
    clauses = ["missing_at IS NULL", "quarantined_at IS NULL"]
    params: list[Any] = []
    if track_id is not None:
        clauses.append("id = ?")
        params.append(track_id)
    if track_ids is not None:
        if not track_ids:
            return []
        placeholders = ", ".join("?" for _ in track_ids)
        clauses.append(f"id IN ({placeholders})")
        params.extend(track_ids)
    if missing_only:
        clauses.append("id NOT IN (SELECT track_id FROM track_tags WHERE source LIKE 'essentia:%')")

    return conn.execute(
        f"SELECT id, path FROM tracks WHERE {' AND '.join(clauses)} ORDER BY path",
        params,
    ).fetchall()


def _process_one_tag_job(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    specs: list[EssentiaModelSpec],
    summary: TagAnalysisSummary,
    *,
    min_score: float,
) -> None:
    path = Path(row["path"])
    if not path.exists():
        summary.skipped += 1
        record_track_error(conn, row["id"], "file is missing on disk")
        return

    summary.processed += 1
    try:
        tags = analyze_essentia_tags(path, specs, min_score=min_score)
        save_track_tags(conn, row["id"], tags, sources=[spec.source for spec in specs])
        summary.updated += 1
    except EssentiaModelError as exc:
        summary.errors += 1
        record_track_error(conn, row["id"], str(exc))
    except Exception as exc:  # pragma: no cover - defensive safety net
        summary.errors += 1
        record_track_error(conn, row["id"], f"unexpected tag analysis error: {exc}")


def _required_str(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EssentiaModelError(f"model manifest entry missing required string: {key}")
    return value.strip()


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_path(models_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _resolve_model_path(models_path, value)


def _resolve_model_path(models_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else models_path / path


def _normalize_tag_name(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())
