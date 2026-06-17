"""Typer command-line interface for MusicIdx."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from musicidx import __version__
from musicidx.analyzer.basic_features import is_librosa_available, process_basic_analysis
from musicidx.analyzer.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingError,
    is_sentence_transformers_available,
    process_embeddings,
    search_semantic,
)
from musicidx.analyzer.essentia_models import (
    DEFAULT_MIN_SCORE,
    TagAnalysisSummary,
    available_model_specs,
    is_essentia_available,
    list_track_tags,
    model_manifest_status,
    process_tags,
)
from musicidx.config import (
    DB_PATH_ENV_VAR,
    DEFAULT_DB_FILENAME,
    DEFAULT_MODELS_DIRNAME,
    FFPROBE_PATH_ENV_VAR,
    FPCALC_PATH_ENV_VAR,
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_MODEL_ENV_VAR,
    MODELS_PATH_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_MODEL_ENV_VAR,
    resolve_db_path,
    resolve_executable,
    resolve_models_path,
)
from musicidx.db import CORE_TABLES, connect_db, db_info, init_db
from musicidx.failures import list_failed_tracks, record_track_error, reset_failed_tracks
from musicidx.fingerprint import find_duplicate_groups, is_fpcalc_available, process_fingerprints
from musicidx.metadata import is_ffprobe_available, process_metadata, search_text
from musicidx.missing import list_missing_tracks, prune_missing_tracks
from musicidx.resources import (
    RuntimeTimer,
    recommend_indexing_plan,
    resolve_basic_chunk_sec,
    resolve_embedding_batch_size,
    resolve_tag_batch_size,
    resolve_worker_count,
    with_runtime_diagnostics,
)
from musicidx.scanner import scan_library
from musicidx.search.evaluation import (
    aggregate_eval_results,
    evaluate_response,
    load_eval_queries,
)
from musicidx.search.feedback import (
    feedback_summary,
    save_feedback_event,
    save_search_event,
    save_track_feedback,
)
from musicidx.search.intent import build_library_profile, parse_intent_dynamic
from musicidx.search.llm import (
    LLMIntentError,
    default_gemini_model,
    is_gemini_configured,
    is_openai_configured,
    parse_intent_llm,
)
from musicidx.search.ranker import search_music

app = typer.Typer(help="Local-first music library index CLI.")
models_app = typer.Typer(help="Manage local ML model files.")
app.add_typer(models_app, name="models")
console = Console()

DB_OPTION_HELP = (
    f"SQLite database path. Defaults to ./{DEFAULT_DB_FILENAME} or {DB_PATH_ENV_VAR}."
)
MODELS_OPTION_HELP = (
    f"Local model directory. Defaults to ./{DEFAULT_MODELS_DIRNAME} or {MODELS_PATH_ENV_VAR}."
)
DbOption = Annotated[Path | None, typer.Option("--db", help=DB_OPTION_HELP)]
ModelsPathOption = Annotated[
    Path | None,
    typer.Option("--models-path", help=MODELS_OPTION_HELP),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]
DirectoryArg = Annotated[Path, typer.Argument(help="Music directory to scan.")]
FullHashOption = Annotated[bool, typer.Option("--full-hash", help="Compute SHA-256 file hashes.")]
FollowSymlinksOption = Annotated[
    bool,
    typer.Option("--follow-symlinks", help="Follow symbolic links while scanning."),
]
DryRunOption = Annotated[
    bool,
    typer.Option("--dry-run", help="Report changes without writing them."),
]
TrackIdOption = Annotated[str | None, typer.Option("--track-id", help="Process one track ID.")]
RequiredTrackIdOption = Annotated[str, typer.Option("--track-id", help="Track ID to show.")]
MissingOnlyOption = Annotated[
    bool,
    typer.Option("--missing-only", help="Only process tracks without stored metadata/profile."),
]
TagMissingOnlyOption = Annotated[
    bool,
    typer.Option("--missing-only", help="Only process tracks without stored Essentia tags."),
]
FingerprintMissingOnlyOption = Annotated[
    bool,
    typer.Option("--missing-only", help="Only process tracks without stored fingerprints."),
]
SearchQueryArg = Annotated[str, typer.Argument(help="Full-text query to search for.")]
EvalFileArg = Annotated[Path, typer.Argument(help="JSON eval query file.")]
LimitOption = Annotated[
    int,
    typer.Option("--limit", min=1, max=100, help="Maximum number of results."),
]
OptionalLimitOption = Annotated[
    int | None,
    typer.Option("--limit", min=1, max=100, help="Maximum number of results."),
]
IncludeMissingOption = Annotated[
    bool,
    typer.Option("--include-missing", help="Include tracks marked missing from disk."),
]
DurationToleranceOption = Annotated[
    float,
    typer.Option(
        "--duration-tolerance",
        min=0.0,
        help="Duration tolerance in seconds for duplicate grouping.",
    ),
]
DuplicatesIncludeMissingOption = Annotated[
    bool,
    typer.Option(
        "--include-missing/--exclude-missing",
        help="Include tracks marked missing from disk for move detection.",
    ),
]
QuickOption = Annotated[
    bool,
    typer.Option("--quick", help="Analyze only the first 120 seconds of each track."),
]
ChunkedOption = Annotated[
    bool,
    typer.Option("--chunked", help="Analyze audio in sequential chunks to reduce peak RAM."),
]
ChunkSecOption = Annotated[
    str,
    typer.Option("--chunk-sec", help="Chunk duration in seconds, or 'auto'."),
]
MaxChunksOption = Annotated[
    int | None,
    typer.Option(
        "--max-chunks",
        min=1,
        max=10000,
        help="Maximum chunks to sample for chunked analysis. Default: all chunks.",
    ),
]
WorkersOption = Annotated[
    str,
    typer.Option("--workers", help="Number of analysis worker threads, or 'auto'."),
]
MinScoreOption = Annotated[
    float,
    typer.Option("--min-score", min=0.0, max=1.0, help="Minimum tag score to store."),
]
TagSubprocessBatchesOption = Annotated[
    bool,
    typer.Option(
        "--subprocess-batches/--no-subprocess-batches",
        help="Run tag analysis in child-process batches so TensorFlow memory is reclaimed.",
    ),
]
TagBatchSizeOption = Annotated[
    str,
    typer.Option("--batch-size", help="Tracks per tag subprocess batch, or 'auto'."),
]
TrackIdFileOption = Annotated[
    Path | None,
    typer.Option(
        "--track-id-file",
        help="Read tag-analysis track IDs from a JSON file.",
        hidden=True,
    ),
]
EmbeddingModelOption = Annotated[
    str,
    typer.Option("--model", help="Sentence-transformers model name for profile embeddings."),
]
BatchSizeOption = Annotated[
    str,
    typer.Option("--batch-size", help="Embedding batch size, or 'auto'."),
]
RefreshOption = Annotated[
    bool,
    typer.Option("--refresh", help="Recompute embeddings even when stored text is current."),
]
ResourceProfileOption = Annotated[
    str,
    typer.Option(
        "--resource-profile",
        help="Indexing resource profile: auto, low, balanced, full.",
    ),
]
SemanticModelOption = Annotated[
    str,
    typer.Option("--semantic-model", help="Embedding model name/path to use if indexed."),
]
ExplainOption = Annotated[bool, typer.Option("--explain", help="Include match explanations.")]
SearchFormatOption = Annotated[
    str,
    typer.Option("--format", help="Output format: table, json, or m3u."),
]
UseLlmOption = Annotated[
    bool,
    typer.Option("--llm/--no-llm", help="Use an LLM provider to add intent parsing hints."),
]
LlmProviderOption = Annotated[
    str,
    typer.Option("--llm-provider", help="LLM provider: gemini or openai."),
]
LlmModelOption = Annotated[
    str | None,
    typer.Option(
        "--llm-model",
        help=f"LLM model. Defaults to {GEMINI_MODEL_ENV_VAR} or {OPENAI_MODEL_ENV_VAR}.",
    ),
]
LlmTimeoutOption = Annotated[
    float,
    typer.Option("--llm-timeout", min=1.0, max=120.0, help="LLM timeout in seconds."),
]
ConciseOption = Annotated[
    bool,
    typer.Option("--concise", help="Use shorter JSON output for search results."),
]
FeedbackRatingOption = Annotated[
    str,
    typer.Option("--rating", help="Feedback rating: good, bad, neutral, 1, 0, or -1."),
]
FeedbackQueryOption = Annotated[
    str | None,
    typer.Option("--query", help="Original search query for query-aware feedback."),
]
FeedbackNoteOption = Annotated[
    str | None,
    typer.Option("--note", help="Optional feedback note."),
]
FailedQuarantinedOnlyOption = Annotated[
    bool,
    typer.Option("--quarantined-only", help="Show only quarantined failed tracks."),
]
RetryAllOption = Annotated[
    bool,
    typer.Option("--all", help="Reset all failed/quarantined tracks."),
]
PruneMissingAllOption = Annotated[
    bool,
    typer.Option("--all", help="Prune all tracks marked missing from disk."),
]
ExportFormatOption = Annotated[
    str,
    typer.Option("--format", help="Export format: m3u, json, or csv."),
]
OutputPathOption = Annotated[
    Path | None,
    typer.Option("--out", "-o", help="Write export output to a file instead of stdout."),
]
AbsolutePathsOption = Annotated[
    bool,
    typer.Option("--absolute-paths", help="Export absolute track paths."),
]
RelativePathsOption = Annotated[
    bool,
    typer.Option("--relative-paths", help="Export paths relative to the output file or cwd."),
]


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _status(available: bool) -> str:
    return "available" if available else "missing"


def _resolve_workers(value: str | int, *, kind: str, resource_profile: str) -> int:
    try:
        return resolve_worker_count(value, kind=kind, profile=resource_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_embedding_batch_size(value: str | int, *, resource_profile: str) -> int:
    try:
        return resolve_embedding_batch_size(value, profile=resource_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_tag_batch_size(value: str | int, *, resource_profile: str) -> int:
    try:
        return resolve_tag_batch_size(value, profile=resource_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_basic_chunk_sec(value: str | float, *, resource_profile: str) -> float:
    try:
        return resolve_basic_chunk_sec(value, profile=resource_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _select_tag_track_ids(
    conn: sqlite3.Connection,
    *,
    track_id: str | None,
    missing_only: bool,
) -> list[str]:
    clauses = ["missing_at IS NULL"]
    params: list[Any] = []
    if track_id is not None:
        clauses.append("id = ?")
        params.append(track_id)
    if missing_only:
        clauses.append("id NOT IN (SELECT track_id FROM track_tags WHERE source LIKE 'essentia:%')")
    rows = conn.execute(
        f"SELECT id FROM tracks WHERE {' AND '.join(clauses)} ORDER BY path",
        params,
    ).fetchall()
    return [str(row["id"]) for row in rows]


def _load_track_id_file(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise typer.BadParameter(f"could not read track ID file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid track ID file JSON: {exc}") from exc
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise typer.BadParameter("track ID file must contain a JSON array of strings")
    return payload


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _json_from_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        loaded = json.loads(stripped[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("expected JSON object")
    return loaded


def _summary_from_payload(payload: dict[str, Any]) -> TagAnalysisSummary:
    return TagAnalysisSummary(
        processed=int(payload.get("processed", 0)),
        updated=int(payload.get("updated", 0)),
        skipped=int(payload.get("skipped", 0)),
        errors=int(payload.get("errors", 0)),
        model_count=int(payload.get("model_count", 0)),
    )


def _record_tag_batch_error(conn: sqlite3.Connection, track_ids: list[str], error: str) -> None:
    for track_id in track_ids:
        record_track_error(conn, track_id, error)
    conn.commit()


def _tag_subprocess_base_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "musicidx.cli"]


def _run_tag_subprocess_batches(
    *,
    db_path: Path,
    models_path: Path,
    track_ids: list[str],
    missing_only: bool,
    min_score: float,
    batch_size: int,
) -> tuple[TagAnalysisSummary, int]:
    summary = TagAnalysisSummary()
    batches = _chunks(track_ids, batch_size)
    if not batches:
        try:
            summary.model_count = len(available_model_specs(models_path))
        except Exception:
            summary.model_count = 0
        return summary, 0

    with tempfile.TemporaryDirectory(prefix="musicidx-tags-") as temp_dir:
        temp_path = Path(temp_dir)
        for batch_index, batch in enumerate(batches, start=1):
            ids_path = temp_path / f"batch-{batch_index}.json"
            ids_path.write_text(json.dumps(batch))
            command = [
                *_tag_subprocess_base_command(),
                "analyze-tags",
                "--db",
                str(db_path),
                "--models-path",
                str(models_path),
                "--min-score",
                str(min_score),
                "--workers",
                "1",
                "--no-subprocess-batches",
                "--track-id-file",
                str(ids_path),
                "--json",
            ]
            if missing_only:
                command.insert(-1, "--missing-only")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                summary.errors += len(batch)
                summary.model_count = max(summary.model_count, 0)
                error = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
                conn = connect_db(db_path)
                try:
                    _record_tag_batch_error(
                        conn,
                        batch,
                        f"tag subprocess batch failed: {error[:500]}",
                    )
                finally:
                    conn.close()
                continue
            try:
                child_summary = _summary_from_payload(_json_from_output(result.stdout))
            except Exception as exc:
                summary.errors += len(batch)
                conn = connect_db(db_path)
                try:
                    _record_tag_batch_error(
                        conn,
                        batch,
                        f"tag subprocess returned invalid JSON: {exc}",
                    )
                finally:
                    conn.close()
                continue
            summary.processed += child_summary.processed
            summary.updated += child_summary.updated
            summary.skipped += child_summary.skipped
            summary.errors += child_summary.errors
            summary.model_count = max(summary.model_count, child_summary.model_count)
    return summary, len(batches)


@app.command("doctor")
def doctor_command(json_output: JsonOption = False) -> None:
    """Check local dependencies and runtime capabilities."""
    ffprobe_path = resolve_executable("ffprobe", FFPROBE_PATH_ENV_VAR)
    fpcalc_path = resolve_executable("fpcalc", FPCALC_PATH_ENV_VAR)
    resource_plan = recommend_indexing_plan()
    checks = [
        {
            "name": "SQLite",
            "status": "available",
            "detail": f"sqlite {sqlite3.sqlite_version}",
        },
        {
            "name": "SQLite FTS5",
            "status": _status(_sqlite_fts5_available()),
            "detail": "required for text search",
        },
        {
            "name": "FFmpeg/ffprobe",
            "status": _status(ffprobe_path is not None),
            "detail": ffprobe_path
            or (
                "missing; install with `brew install ffmpeg` "
                f"or set {FFPROBE_PATH_ENV_VAR}"
            ),
        },
        {
            "name": "fpcalc",
            "status": _status(fpcalc_path is not None),
            "detail": fpcalc_path
            or (
                "missing; install with `brew install chromaprint` "
                f"or set {FPCALC_PATH_ENV_VAR}"
            ),
        },
        {
            "name": "Ollama",
            "status": _status(shutil.which("ollama") is not None),
            "detail": shutil.which("ollama") or "ollama not found on PATH",
        },
        {
            "name": "Gemini API key",
            "status": _status(is_gemini_configured()),
            "detail": f"configured via {GEMINI_API_KEY_ENV_VAR}"
            if is_gemini_configured()
            else f"missing; set {GEMINI_API_KEY_ENV_VAR} to use --llm",
        },
        {
            "name": "OpenAI API key",
            "status": _status(is_openai_configured()),
            "detail": f"configured via {OPENAI_API_KEY_ENV_VAR}"
            if is_openai_configured()
            else f"optional fallback; set {OPENAI_API_KEY_ENV_VAR}",
        },
        {
            "name": "librosa",
            "status": _status(is_librosa_available()),
            "detail": "python module for basic audio analysis",
        },
        {
            "name": "Essentia",
            "status": _status(is_essentia_available()),
            "detail": "python module for ML mood/genre tags",
        },
        {
            "name": "Essentia model manifest",
            "status": _status(model_manifest_status(resolve_models_path()).manifest_exists),
            "detail": str(resolve_models_path() / "manifest.json"),
        },
        {
            "name": "Embedding model support",
            "status": _status(is_sentence_transformers_available()),
            "detail": "sentence-transformers module",
        },
        {
            "name": "Adaptive indexing",
            "status": "available",
            "detail": (
                f"{resource_plan.effective_profile}: basic workers "
                f"{resource_plan.basic_workers}, chunk {resource_plan.basic_chunk_sec:g}s, "
                f"tag workers {resource_plan.tag_workers}, "
                f"tag batch {resource_plan.tag_batch_size}, "
                f"embed batch {resource_plan.embedding_batch_size}"
            ),
        },
    ]

    payload = {"version": __version__, "checks": checks, "resources": resource_plan.as_dict()}
    if json_output:
        _print_json(payload)
        return

    table = Table(title="MusicIdx doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in checks:
        style = "green" if check["status"] == "available" else "yellow"
        table.add_row(
            str(check["name"]),
            f"[{style}]{check['status']}[/{style}]",
            str(check["detail"]),
        )
    console.print(table)


@app.command("resources")
def resources_command(
    resource_profile: ResourceProfileOption = "auto",
    json_output: JsonOption = False,
) -> None:
    """Show adaptive indexing recommendations for this machine."""
    try:
        plan = recommend_indexing_plan(resource_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = plan.as_dict()
    if json_output:
        _print_json(payload)
        return

    table = Table(title="Adaptive indexing resources")
    table.add_column("Setting")
    table.add_column("Value")
    resources = payload["resources"]
    table.add_row("Requested profile", payload["requested_profile"])
    table.add_row("Effective profile", payload["effective_profile"])
    table.add_row("CPU count", str(resources["cpu_count"]))
    table.add_row("Total memory", f"{resources['total_memory_gb']} GB")
    table.add_row("Basic workers", str(payload["basic_workers"]))
    table.add_row("Basic chunk seconds", str(payload["basic_chunk_sec"]))
    table.add_row("Tag workers", str(payload["tag_workers"]))
    table.add_row("Embedding batch size", str(payload["embedding_batch_size"]))
    table.add_row("Tag batch size", str(payload["tag_batch_size"]))
    table.add_row("Reason", str(payload["reason"]))
    if payload["warning"]:
        table.add_row("Warning", str(payload["warning"]))
    console.print(table)


@app.command("init")
def init_command(db: DbOption = None, json_output: JsonOption = False) -> None:
    """Initialize the local SQLite database."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        payload = {"initialized": True, "db_path": str(db_path)}
    finally:
        conn.close()

    if json_output:
        _print_json(payload)
        return
    console.print(f"[green]Initialized database:[/green] {db_path}")


@app.command("db-info")
def db_info_command(db: DbOption = None, json_output: JsonOption = False) -> None:
    """Show database path, pragmas, migrations, and row counts."""
    db_path = resolve_db_path(db)
    if not db_path.exists():
        payload = {"exists": False, "db_path": str(db_path)}
        if json_output:
            _print_json(payload)
        else:
            console.print(f"[red]Database does not exist:[/red] {db_path}")
            console.print("Run `musicidx init` first.")
        raise typer.Exit(1)

    conn = connect_db(db_path)
    try:
        init_db(conn)
        payload = {"exists": True, **db_info(conn, db_path)}
    finally:
        conn.close()

    if json_output:
        _print_json(payload)
        return

    console.print(f"[bold]Database:[/bold] {payload['db_path']}")
    console.print(f"SQLite: {payload['sqlite_version']}")
    console.print(f"Journal mode: {payload['journal_mode']}")
    console.print(f"Foreign keys: {payload['foreign_keys']}")

    table = Table(title="Table row counts")
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    for table_name in CORE_TABLES:
        count = payload["tables"].get(table_name)
        table.add_row(table_name, "missing" if count is None else str(count))
    console.print(table)


@app.command("failed")
def failed_command(
    db: DbOption = None,
    include_missing: IncludeMissingOption = False,
    quarantined_only: FailedQuarantinedOnlyOption = False,
    json_output: JsonOption = False,
) -> None:
    """List tracks with repeated failures or quarantine state."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        failed_tracks = list_failed_tracks(
            conn,
            include_missing=include_missing,
            quarantined_only=quarantined_only,
        )
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "count": len(failed_tracks),
        "failed": [track.as_dict() for track in failed_tracks],
    }
    if json_output:
        _print_json(payload)
        return

    if not failed_tracks:
        console.print("[green]No failed or quarantined tracks.[/green]")
        return

    table = Table(title="Failed/quarantined tracks")
    table.add_column("Count", justify="right")
    table.add_column("Quarantined")
    table.add_column("Artist")
    table.add_column("Title")
    table.add_column("Path")
    table.add_column("Error")
    for track in failed_tracks:
        table.add_row(
            str(track.error_count),
            "yes" if track.quarantined_at else "no",
            track.artist or "",
            track.title or "",
            track.path,
            track.last_error or track.quarantine_reason or "",
        )
    console.print(table)


@app.command("retry-failed")
def retry_failed_command(
    db: DbOption = None,
    track_id: TrackIdOption = None,
    all_tracks: RetryAllOption = False,
    json_output: JsonOption = False,
) -> None:
    """Clear failure/quarantine state so failed tracks can be retried."""
    if track_id is None and not all_tracks:
        raise typer.BadParameter("pass --track-id <id> or --all")
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        reset_count = reset_failed_tracks(conn, track_id=track_id if not all_tracks else None)
    finally:
        conn.close()

    payload = {"db_path": str(db_path), "reset": reset_count, "track_id": track_id}
    if json_output:
        _print_json(payload)
        return
    console.print(f"[green]Reset failed track state:[/green] {reset_count}")


@app.command("missing")
def missing_command(
    db: DbOption = None,
    json_output: JsonOption = False,
) -> None:
    """List tracks marked missing from disk."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        missing_tracks = list_missing_tracks(conn)
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "count": len(missing_tracks),
        "missing": [track.as_dict() for track in missing_tracks],
    }
    if json_output:
        _print_json(payload)
        return

    if not missing_tracks:
        console.print("[green]No missing tracks.[/green]")
        return

    table = Table(title="Missing tracks")
    table.add_column("Missing at")
    table.add_column("Artist")
    table.add_column("Title")
    table.add_column("Path")
    table.add_column("Root")
    for track in missing_tracks:
        table.add_row(
            track.missing_at,
            track.artist or "",
            track.title or "",
            track.path,
            track.root_path or "",
        )
    console.print(table)


@app.command("prune-missing")
def prune_missing_command(
    db: DbOption = None,
    track_id: TrackIdOption = None,
    all_tracks: PruneMissingAllOption = False,
    json_output: JsonOption = False,
) -> None:
    """Delete missing-track database rows only; never delete files."""
    if track_id is None and not all_tracks:
        raise typer.BadParameter("pass --track-id <id> or --all")
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        pruned_count = prune_missing_tracks(conn, track_id=track_id if not all_tracks else None)
    finally:
        conn.close()

    payload = {"db_path": str(db_path), "pruned": pruned_count, "track_id": track_id}
    if json_output:
        _print_json(payload)
        return
    console.print(f"[green]Pruned missing track rows:[/green] {pruned_count}")


@app.command("scan")
def scan_command(
    directory: DirectoryArg,
    db: DbOption = None,
    full_hash: FullHashOption = False,
    follow_symlinks: FollowSymlinksOption = False,
    dry_run: DryRunOption = False,
    json_output: JsonOption = False,
) -> None:
    """Recursively scan a directory for supported audio files."""
    timer = RuntimeTimer()
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = scan_library(
            directory,
            conn,
            full_hash=full_hash,
            follow_symlinks=follow_symlinks,
            dry_run=dry_run,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        if json_output:
            _print_json({"error": str(exc), "db_path": str(db_path)})
        else:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        conn.close()

    payload = with_runtime_diagnostics({"db_path": str(db_path), **summary.as_dict()}, timer)
    if json_output:
        _print_json(payload)
        return

    table = Table(title="Scan summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key in ["added", "unchanged", "modified", "missing", "skipped", "errors", "total_seen"]:
        table.add_row(key.replace("_", " "), str(payload[key]))
    console.print(f"[bold]Root:[/bold] {summary.root_path}")
    console.print(f"[bold]Database:[/bold] {db_path}")
    if dry_run:
        console.print("[yellow]Dry run:[/yellow] no database changes were written.")
    console.print(table)


@app.command("metadata")
def metadata_command(
    db: DbOption = None,
    track_id: TrackIdOption = None,
    missing_only: MissingOnlyOption = False,
    json_output: JsonOption = False,
) -> None:
    """Extract metadata for scanned tracks using ffprobe."""
    timer = RuntimeTimer()
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_metadata(conn, track_id=track_id, missing_only=missing_only)
    finally:
        conn.close()

    payload = with_runtime_diagnostics(
        {
            "db_path": str(db_path),
            "ffprobe_available": is_ffprobe_available(),
            **summary.as_dict(),
        },
        timer,
        include_child_peak=True,
    )
    if json_output:
        _print_json(payload)
        return

    if not payload["ffprobe_available"]:
        console.print(
            "[yellow]ffprobe is not available; install with `brew install ffmpeg`.[/yellow]"
        )

    table = Table(title="Metadata summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key in ["processed", "updated", "skipped", "errors"]:
        table.add_row(key, str(payload[key]))
    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(table)


@app.command("search-text")
def search_text_command(
    query: SearchQueryArg,
    db: DbOption = None,
    limit: LimitOption = 10,
    include_missing: IncludeMissingOption = False,
    json_output: JsonOption = False,
) -> None:
    """Search extracted metadata/profile text with SQLite FTS5."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        results = search_text(conn, query, limit=limit, include_missing=include_missing)
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "query": query,
        "limit": limit,
        "results": [result.as_dict() for result in results],
    }
    if json_output:
        _print_json(payload)
        return

    table = Table(title=f"Text search: {query}")
    table.add_column("#", justify="right")
    table.add_column("Title")
    table.add_column("Artist")
    table.add_column("Album")
    table.add_column("Path")
    for index, result in enumerate(results, start=1):
        table.add_row(
            str(index),
            result.title or "",
            result.artist or "",
            result.album or "",
            result.path,
        )
    console.print(table)


@app.command("fingerprint")
def fingerprint_command(
    db: DbOption = None,
    track_id: TrackIdOption = None,
    missing_only: FingerprintMissingOnlyOption = False,
    json_output: JsonOption = False,
) -> None:
    """Fingerprint scanned tracks with fpcalc/chromaprint."""
    timer = RuntimeTimer()
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_fingerprints(conn, track_id=track_id, missing_only=missing_only)
    finally:
        conn.close()

    payload = with_runtime_diagnostics(
        {
            "db_path": str(db_path),
            "fpcalc_available": is_fpcalc_available(),
            **summary.as_dict(),
        },
        timer,
        include_child_peak=True,
    )
    if json_output:
        _print_json(payload)
        return

    if not payload["fpcalc_available"]:
        console.print(
            "[yellow]fpcalc is not available; install with `brew install chromaprint`.[/yellow]"
        )

    table = Table(title="Fingerprint summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key in ["processed", "updated", "skipped", "errors"]:
        table.add_row(key, str(payload[key]))
    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(table)


@app.command("duplicates")
def duplicates_command(
    db: DbOption = None,
    include_missing: DuplicatesIncludeMissingOption = True,
    duration_tolerance: DurationToleranceOption = 3.0,
    json_output: JsonOption = False,
) -> None:
    """Show duplicate and possible moved-file candidates."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        groups = find_duplicate_groups(
            conn,
            include_missing=include_missing,
            duration_tolerance_sec=duration_tolerance,
        )
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "count": len(groups),
        "groups": [group.as_dict() for group in groups],
    }
    if json_output:
        _print_json(payload)
        return

    if not groups:
        console.print("[green]No duplicate candidates found.[/green]")
        return

    for group_index, group in enumerate(groups, start=1):
        table = Table(title=f"Group {group_index}: {group.kind} — {group.reason}")
        table.add_column("#", justify="right")
        table.add_column("Missing")
        table.add_column("Artist")
        table.add_column("Title")
        table.add_column("Album")
        table.add_column("Duration", justify="right")
        table.add_column("Path")
        for track_index, track in enumerate(group.tracks, start=1):
            duration = track.fingerprint_duration or track.duration_sec
            table.add_row(
                str(track_index),
                "yes" if track.missing else "no",
                track.artist or "",
                track.title or "",
                track.album or "",
                f"{duration:.1f}" if duration is not None else "",
                track.path,
            )
        console.print(table)


@app.command("analyze-basic")
def analyze_basic_command(
    db: DbOption = None,
    track_id: TrackIdOption = None,
    quick: QuickOption = False,
    chunked: ChunkedOption = False,
    chunk_sec: ChunkSecOption = "auto",
    max_chunks: MaxChunksOption = None,
    workers: WorkersOption = "auto",
    resource_profile: ResourceProfileOption = "auto",
    json_output: JsonOption = False,
) -> None:
    """Analyze basic audio features for scanned tracks."""
    timer = RuntimeTimer()
    resolved_workers = _resolve_workers(workers, kind="basic", resource_profile=resource_profile)
    resolved_chunk_sec = _resolve_basic_chunk_sec(chunk_sec, resource_profile=resource_profile)
    resource_plan = recommend_indexing_plan(resource_profile)
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_basic_analysis(
            conn,
            track_id=track_id,
            quick=quick,
            workers=resolved_workers,
            chunked=chunked,
            chunk_sec=resolved_chunk_sec,
            max_chunks=max_chunks,
        )
    finally:
        conn.close()

    payload = with_runtime_diagnostics(
        {
            "db_path": str(db_path),
            "librosa_available": is_librosa_available(),
            "quick": quick,
            "chunked": chunked,
            "chunk_sec": resolved_chunk_sec,
            "chunk_sec_requested": str(chunk_sec),
            "max_chunks": max_chunks,
            "workers": resolved_workers,
            "workers_requested": str(workers),
            "resource_profile": resource_profile,
            "resource_plan": resource_plan.as_dict(),
            **summary.as_dict(),
        },
        timer,
    )
    if json_output:
        _print_json(payload)
        return

    if not payload["librosa_available"]:
        console.print(
            "[yellow]librosa is not available; basic audio analysis failed.[/yellow]"
        )

    table = Table(title="Basic audio analysis summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key in ["processed", "updated", "skipped", "errors", "analysis_version"]:
        table.add_row(key, str(payload[key]))
    console.print(f"[bold]Database:[/bold] {db_path}")
    if quick:
        console.print("[yellow]Quick mode:[/yellow] analyzed at most first 120 seconds.")
    if chunked:
        max_chunks_text = "all chunks" if max_chunks is None else str(max_chunks)
        console.print(
            f"[yellow]Chunked mode:[/yellow] {resolved_chunk_sec:g}s chunks, max {max_chunks_text}."
        )
    console.print(table)


@app.command("analyze-tags")
def analyze_tags_command(
    db: DbOption = None,
    models_path: ModelsPathOption = None,
    track_id: TrackIdOption = None,
    missing_only: TagMissingOnlyOption = False,
    min_score: MinScoreOption = DEFAULT_MIN_SCORE,
    workers: WorkersOption = "auto",
    resource_profile: ResourceProfileOption = "auto",
    subprocess_batches: TagSubprocessBatchesOption = True,
    batch_size: TagBatchSizeOption = "auto",
    track_id_file: TrackIdFileOption = None,
    json_output: JsonOption = False,
) -> None:
    """Analyze ML mood/genre tags with local Essentia models."""
    timer = RuntimeTimer()
    resolved_workers = _resolve_workers(workers, kind="tags", resource_profile=resource_profile)
    resolved_batch_size = _resolve_tag_batch_size(batch_size, resource_profile=resource_profile)
    resource_plan = recommend_indexing_plan(resource_profile)
    db_path = resolve_db_path(db)
    resolved_models_path = resolve_models_path(models_path)
    track_ids = _load_track_id_file(track_id_file) if track_id_file is not None else None
    use_subprocess_batches = subprocess_batches and track_id_file is None
    batches = 0

    if use_subprocess_batches:
        conn = connect_db(db_path)
        try:
            init_db(conn)
            selected_track_ids = _select_tag_track_ids(
                conn,
                track_id=track_id,
                missing_only=missing_only,
            )
        finally:
            conn.close()
        summary, batches = _run_tag_subprocess_batches(
            db_path=db_path,
            models_path=resolved_models_path,
            track_ids=selected_track_ids,
            missing_only=missing_only,
            min_score=min_score,
            batch_size=resolved_batch_size,
        )
    else:
        conn = connect_db(db_path)
        try:
            init_db(conn)
            summary = process_tags(
                conn,
                models_path=resolved_models_path,
                track_id=track_id,
                track_ids=track_ids,
                missing_only=missing_only,
                min_score=min_score,
                workers=resolved_workers,
            )
        finally:
            conn.close()

    payload = with_runtime_diagnostics(
        {
            "db_path": str(db_path),
            "models_path": str(resolved_models_path),
            "essentia_available": is_essentia_available(),
            "min_score": min_score,
            "workers": resolved_workers,
            "workers_requested": str(workers),
            "resource_profile": resource_profile,
            "resource_plan": resource_plan.as_dict(),
            "subprocess_batches": use_subprocess_batches,
            "batch_size": resolved_batch_size,
            "batch_size_requested": str(batch_size),
            "batches": batches,
            **summary.as_dict(),
        },
        timer,
        include_child_peak=use_subprocess_batches,
    )
    if json_output:
        _print_json(payload)
        return

    if not payload["essentia_available"]:
        console.print("[yellow]Essentia is not installed; tag analysis failed.[/yellow]")
    if payload["model_count"] == 0:
        console.print(
            "[yellow]No available local Essentia model specs found. "
            "Run `musicidx models path` and add a manifest.json.[/yellow]"
        )

    table = Table(title="ML tag analysis summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key in [
        "processed",
        "updated",
        "skipped",
        "errors",
        "model_count",
        "subprocess_batches",
        "batch_size",
        "batches",
    ]:
        table.add_row(key, str(payload[key]))
    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]Models:[/bold] {resolved_models_path}")
    console.print(table)


@app.command("tags")
def tags_command(
    track_id: RequiredTrackIdOption,
    db: DbOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show stored tags for a track."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        tags = list_track_tags(conn, track_id=track_id)
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "track_id": track_id,
        "tags": [tag.as_dict() for tag in tags],
    }
    if json_output:
        _print_json(payload)
        return

    table = Table(title=f"Tags for {track_id}")
    table.add_column("Source")
    table.add_column("Tag")
    table.add_column("Score", justify="right")
    for tag in tags:
        table.add_row(tag.source, tag.tag, f"{tag.score:.3f}")
    console.print(table)


@app.command("embed")
def embed_command(
    db: DbOption = None,
    track_id: TrackIdOption = None,
    model: EmbeddingModelOption = DEFAULT_EMBEDDING_MODEL,
    batch_size: BatchSizeOption = "auto",
    resource_profile: ResourceProfileOption = "auto",
    refresh: RefreshOption = False,
    json_output: JsonOption = False,
) -> None:
    """Embed enriched track profile text for semantic search."""
    timer = RuntimeTimer()
    resolved_batch_size = _resolve_embedding_batch_size(
        batch_size,
        resource_profile=resource_profile,
    )
    resource_plan = recommend_indexing_plan(resource_profile)
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_embeddings(
            conn,
            track_id=track_id,
            model_name=model,
            batch_size=resolved_batch_size,
            refresh=refresh,
        )
    finally:
        conn.close()

    payload = with_runtime_diagnostics(
        {
            "db_path": str(db_path),
            "sentence_transformers_available": is_sentence_transformers_available(),
            "batch_size": resolved_batch_size,
            "batch_size_requested": str(batch_size),
            "resource_profile": resource_profile,
            "resource_plan": resource_plan.as_dict(),
            "refresh": refresh,
            **summary.as_dict(),
        },
        timer,
    )
    if json_output:
        _print_json(payload)
        return

    if not payload["sentence_transformers_available"]:
        console.print(
            "[yellow]sentence-transformers is not installed; embedding failed.[/yellow]"
        )

    table = Table(title="Profile embedding summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key in ["processed", "updated", "skipped", "errors", "kind", "model"]:
        table.add_row(key, str(payload[key]))
    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(table)


@app.command("search-semantic")
def search_semantic_command(
    query: SearchQueryArg,
    db: DbOption = None,
    model: EmbeddingModelOption = DEFAULT_EMBEDDING_MODEL,
    limit: LimitOption = 10,
    include_missing: IncludeMissingOption = False,
    json_output: JsonOption = False,
) -> None:
    """Search enriched profile embeddings semantically."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        results = search_semantic(
            conn,
            query,
            model_name=model,
            limit=limit,
            include_missing=include_missing,
        )
    except EmbeddingError as exc:
        if json_output:
            _print_json({"db_path": str(db_path), "query": query, "error": str(exc)})
        else:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "query": query,
        "model": model,
        "limit": limit,
        "results": [result.as_dict() for result in results],
    }
    if json_output:
        _print_json(payload)
        return

    table = Table(title=f"Semantic search: {query}")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Title")
    table.add_column("Artist")
    table.add_column("Album")
    table.add_column("Path")
    for index, result in enumerate(results, start=1):
        table.add_row(
            str(index),
            f"{result.score:.3f}",
            result.title or "",
            result.artist or "",
            result.album or "",
            result.path,
        )
    console.print(table)


@app.command("parse")
def parse_command(
    query: SearchQueryArg,
    db: DbOption = None,
    limit: OptionalLimitOption = None,
    semantic_model: SemanticModelOption = DEFAULT_EMBEDDING_MODEL,
    include_missing: IncludeMissingOption = False,
    use_llm: UseLlmOption = False,
    llm_provider: LlmProviderOption = "gemini",
    llm_model: LlmModelOption = None,
    llm_timeout: LlmTimeoutOption = 30.0,
    json_output: JsonOption = False,
) -> None:
    """Parse a query into dynamic, library-aware search intent."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        llm_hints, parser, llm_error = _maybe_parse_with_llm(
            conn,
            query,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
            timeout_sec=llm_timeout,
            include_missing=include_missing,
        )
        intent = parse_intent_dynamic(
            query,
            conn,
            limit=limit,
            include_missing=include_missing,
            semantic_model=semantic_model,
            llm_hints=llm_hints,
            parser=parser,
            llm_error=llm_error,
        )
    finally:
        conn.close()

    payload = {"db_path": str(db_path), "intent": intent.as_dict()}
    if json_output:
        _print_json(payload)
        return
    _print_json(payload)


@app.command("search")
def search_command(
    query: SearchQueryArg,
    db: DbOption = None,
    limit: OptionalLimitOption = None,
    semantic_model: SemanticModelOption = DEFAULT_EMBEDDING_MODEL,
    include_missing: IncludeMissingOption = False,
    explain: ExplainOption = False,
    output_format: SearchFormatOption = "table",
    use_llm: UseLlmOption = False,
    llm_provider: LlmProviderOption = "gemini",
    llm_model: LlmModelOption = None,
    llm_timeout: LlmTimeoutOption = 30.0,
    concise: ConciseOption = False,
    json_output: JsonOption = False,
) -> None:
    """Search the local library with dynamic hybrid ranking."""
    if json_output:
        output_format = "json"
    if output_format not in {"table", "json", "m3u"}:
        console.print("[red]Error:[/red] --format must be one of: table, json, m3u")
        raise typer.Exit(1)

    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        llm_hints, parser, llm_error = _maybe_parse_with_llm(
            conn,
            query,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
            timeout_sec=llm_timeout,
            include_missing=include_missing,
        )
        response = search_music(
            conn,
            query,
            limit=limit,
            include_missing=include_missing,
            semantic_model=semantic_model,
            explain=explain,
            llm_hints=llm_hints,
            parser=parser,
            llm_error=llm_error,
        )
    finally:
        conn.close()

    payload = _search_payload(response, db_path=str(db_path), concise=concise)
    if output_format == "json":
        _print_json(payload)
        return
    if output_format == "m3u":
        _print_m3u(response.results)
        return

    table = Table(title=f"Search: {query}")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Title")
    table.add_column("Artist")
    table.add_column("Album")
    table.add_column("Why" if explain else "Path")
    for index, result in enumerate(response.results, start=1):
        table.add_row(
            str(index),
            f"{result.score:.3f}",
            result.title or "",
            result.artist or "",
            result.album or "",
            "; ".join(result.explanation) if explain else result.path,
        )
    console.print(table)


@app.command("eval")
def eval_command(
    eval_file: EvalFileArg,
    db: DbOption = None,
    limit: LimitOption = 10,
    semantic_model: SemanticModelOption = DEFAULT_EMBEDDING_MODEL,
    include_missing: IncludeMissingOption = False,
    use_llm: UseLlmOption = False,
    llm_provider: LlmProviderOption = "gemini",
    llm_model: LlmModelOption = None,
    llm_timeout: LlmTimeoutOption = 30.0,
    json_output: JsonOption = False,
) -> None:
    """Run a repeatable search-quality eval query set."""
    try:
        eval_queries = load_eval_queries(eval_file)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        results = []
        for eval_query in eval_queries:
            llm_hints, parser, llm_error = _maybe_parse_with_llm(
                conn,
                eval_query.text,
                use_llm=use_llm,
                llm_provider=llm_provider,
                llm_model=llm_model,
                timeout_sec=llm_timeout,
                include_missing=include_missing,
            )
            response = search_music(
                conn,
                eval_query.text,
                limit=limit,
                include_missing=include_missing,
                semantic_model=semantic_model,
                explain=False,
                llm_hints=llm_hints,
                parser=parser,
                llm_error=llm_error,
            )
            results.append(evaluate_response(conn, eval_query, response))
        payload = {
            "db_path": str(db_path),
            "eval_file": str(eval_file),
            "limit": limit,
            "summary": aggregate_eval_results(results),
            "feedback_summary": feedback_summary(conn),
            "results": results,
        }
    finally:
        conn.close()

    if json_output:
        _print_json(payload)
        return

    summary = payload["summary"]
    console.print(
        "[bold]Eval summary:[/bold] "
        f"queries={summary['query_count']} "
        f"precision@{limit}={summary['avg_precision_at_k']:.3f} "
        f"avoid={summary['avg_avoid_rate']:.3f} "
        f"coverage={summary['avg_tag_coverage']:.3f} "
        f"diversity={summary['avg_diversity_score']:.3f}"
    )
    table = Table(title="Search-quality eval")
    table.add_column("ID")
    table.add_column("Query")
    table.add_column(f"P@{limit}", justify="right")
    table.add_column("Avoid", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Diversity", justify="right")
    table.add_column("Top result")
    for result in results:
        top = result["top_results"][0] if result["top_results"] else {}
        title = top.get("title") or top.get("track_id") or ""
        artist = top.get("artist") or ""
        table.add_row(
            result["id"],
            result["query"],
            f"{result['precision_at_k']:.3f}",
            f"{result['avoid_rate']:.3f}",
            f"{result['tag_coverage']:.3f}",
            f"{result['diversity_score']:.3f}",
            " - ".join(part for part in [artist, title] if part),
        )
    console.print(table)


@app.command("judge")
def judge_command(
    query: SearchQueryArg,
    db: DbOption = None,
    limit: LimitOption = 10,
    semantic_model: SemanticModelOption = DEFAULT_EMBEDDING_MODEL,
    include_missing: IncludeMissingOption = False,
    use_llm: UseLlmOption = False,
    llm_provider: LlmProviderOption = "gemini",
    llm_model: LlmModelOption = None,
    llm_timeout: LlmTimeoutOption = 30.0,
) -> None:
    """Interactively mark search results as good or bad matches."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    saved = 0
    skipped = 0
    try:
        init_db(conn)
        llm_hints, parser, llm_error = _maybe_parse_with_llm(
            conn,
            query,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
            timeout_sec=llm_timeout,
            include_missing=include_missing,
        )
        response = search_music(
            conn,
            query,
            limit=limit,
            include_missing=include_missing,
            semantic_model=semantic_model,
            explain=True,
            llm_hints=llm_hints,
            parser=parser,
            llm_error=llm_error,
        )
        event_id = save_search_event(conn, response)
        console.print(f"[bold]Judging search:[/bold] {query}")
        console.print("Choose [green]y[/green]=good, [red]n[/red]=bad, s=skip, q=quit.")
        for index, result in enumerate(response.results, start=1):
            console.print()
            console.print(
                f"[bold]{index}/{len(response.results)}[/bold] "
                f"score={result.score:.3f} track_id={result.track_id}"
            )
            console.print(
                " - ".join(part for part in [result.artist, result.title] if part)
                or result.path
            )
            if result.album:
                console.print(f"Album: {result.album}")
            console.print(f"Path: {result.path}")
            if result.explanation:
                console.print("Why: " + "; ".join(result.explanation))
            choice = typer.prompt("Good match? [y/n/s/q]", default="s", show_default=False)
            normalized = choice.strip().lower()
            if normalized in {"q", "quit"}:
                break
            if normalized in {"y", "yes", "+", "1"}:
                save_track_feedback(
                    conn,
                    search_event_id=event_id,
                    track_id=result.track_id,
                    rating=1,
                )
                saved += 1
            elif normalized in {"n", "no", "-", "-1"}:
                save_track_feedback(
                    conn,
                    search_event_id=event_id,
                    track_id=result.track_id,
                    rating=-1,
                )
                saved += 1
            else:
                skipped += 1
        summary = feedback_summary(conn)
    finally:
        conn.close()

    console.print(
        f"[green]Saved feedback:[/green] {saved}; skipped: {skipped}; "
        f"total feedback rows: {summary['total']}"
    )


@app.command("feedback")
def feedback_command(
    track_id: RequiredTrackIdOption,
    rating: FeedbackRatingOption,
    query: FeedbackQueryOption = None,
    note: FeedbackNoteOption = None,
    db: DbOption = None,
    json_output: JsonOption = False,
) -> None:
    """Store one non-interactive feedback judgment for UI integrations."""
    parsed_rating = _parse_feedback_rating(rating)
    if parsed_rating is None:
        console.print("[red]Error:[/red] --rating must be one of: good, bad, neutral, 1, 0, -1")
        raise typer.Exit(1)

    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        row = conn.execute("SELECT id FROM tracks WHERE id = ?", (track_id,)).fetchone()
        if row is None:
            console.print(f"[red]Error:[/red] unknown track ID: {track_id}")
            raise typer.Exit(1)
        event_id = None
        if query:
            event_id = save_feedback_event(conn, query=query, track_ids=[track_id])
        feedback_id = save_track_feedback(
            conn,
            search_event_id=event_id,
            track_id=track_id,
            rating=parsed_rating,
            note=note,
        )
        summary = feedback_summary(conn)
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "feedback_id": feedback_id,
        "search_event_id": event_id,
        "track_id": track_id,
        "rating": parsed_rating,
        "query": query,
        "note": note,
        "feedback_summary": summary,
    }
    if json_output:
        _print_json(payload)
        return
    console.print(
        f"[green]Saved feedback[/green] track_id={track_id} rating={parsed_rating}"
    )


@app.command("export")
def export_command(
    query: SearchQueryArg,
    db: DbOption = None,
    out: OutputPathOption = None,
    limit: OptionalLimitOption = None,
    semantic_model: SemanticModelOption = DEFAULT_EMBEDDING_MODEL,
    include_missing: IncludeMissingOption = False,
    output_format: ExportFormatOption = "m3u",
    use_llm: UseLlmOption = False,
    llm_provider: LlmProviderOption = "gemini",
    llm_model: LlmModelOption = None,
    llm_timeout: LlmTimeoutOption = 30.0,
    absolute_paths: AbsolutePathsOption = False,
    relative_paths: RelativePathsOption = False,
    json_output: JsonOption = False,
) -> None:
    """Export a search result set as M3U, JSON, or CSV."""
    if json_output:
        output_format = "json"
    if output_format not in {"m3u", "json", "csv"}:
        console.print("[red]Error:[/red] --format must be one of: m3u, json, csv")
        raise typer.Exit(1)
    if absolute_paths and relative_paths:
        console.print("[red]Error:[/red] choose only one of --absolute-paths or --relative-paths")
        raise typer.Exit(1)

    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        llm_hints, parser, llm_error = _maybe_parse_with_llm(
            conn,
            query,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
            timeout_sec=llm_timeout,
            include_missing=include_missing,
        )
        response = search_music(
            conn,
            query,
            limit=limit,
            include_missing=include_missing,
            semantic_model=semantic_model,
            explain=True,
            llm_hints=llm_hints,
            parser=parser,
            llm_error=llm_error,
        )
    finally:
        conn.close()

    path_mode = _export_path_mode(absolute_paths=absolute_paths, relative_paths=relative_paths)
    base_dir = out.parent if out else Path.cwd()
    content = _format_export(
        response,
        db_path=str(db_path),
        output_format=output_format,
        path_mode=path_mode,
        base_dir=base_dir,
    )
    _write_or_print(out, content)
    if out is not None:
        console.print(f"[green]Wrote {output_format.upper()} export:[/green] {out}")


@models_app.command("path")
def models_path_command(
    models_path: ModelsPathOption = None,
    json_output: JsonOption = False,
) -> None:
    """Show the local model directory path."""
    resolved_models_path = resolve_models_path(models_path)
    payload = {
        "models_path": str(resolved_models_path),
        "manifest_path": str(resolved_models_path / "manifest.json"),
    }
    if json_output:
        _print_json(payload)
        return
    console.print(f"[bold]Models path:[/bold] {payload['models_path']}")
    console.print(f"[bold]Manifest:[/bold] {payload['manifest_path']}")


@models_app.command("list")
def models_list_command(
    models_path: ModelsPathOption = None,
    json_output: JsonOption = False,
) -> None:
    """List local Essentia model specs from manifest.json."""
    resolved_models_path = resolve_models_path(models_path)
    status = model_manifest_status(resolved_models_path)
    payload = status.as_dict()
    if json_output:
        _print_json(payload)
        return

    console.print(f"[bold]Models path:[/bold] {status.models_path}")
    console.print(f"[bold]Manifest:[/bold] {status.manifest_path}")
    console.print(f"Essentia installed: {status.essentia_available}")
    if status.errors:
        for error in status.errors:
            console.print(f"[red]Error:[/red] {error}")
    if not status.manifest_exists:
        console.print("[yellow]No manifest.json found.[/yellow]")
        return
    if not status.models:
        console.print("[yellow]No models defined in manifest.json.[/yellow]")
        return

    table = Table(title="Local Essentia models")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Profile")
    table.add_column("Available")
    table.add_column("Labels", justify="right")
    table.add_column("Missing files")
    for model in status.models:
        table.add_row(
            str(model["name"]),
            str(model["kind"]),
            str(model["profile"]),
            "yes" if model["available"] else "no",
            str(model["labels"]),
            ", ".join(model["missing_files"]),
        )
    console.print(table)


def _parse_feedback_rating(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"good", "yes", "y", "+", "+1", "1", "positive"}:
        return 1
    if normalized in {"bad", "no", "n", "-", "-1", "negative"}:
        return -1
    if normalized in {"neutral", "skip", "s", "0"}:
        return 0
    return None


def _export_path_mode(*, absolute_paths: bool, relative_paths: bool) -> str:
    if absolute_paths:
        return "absolute"
    if relative_paths:
        return "relative"
    return "stored"


def _format_export(
    response: Any,
    *,
    db_path: str,
    output_format: str,
    path_mode: str,
    base_dir: Path,
) -> str:
    if output_format == "m3u":
        return _m3u_text(response.results, path_mode=path_mode, base_dir=base_dir)
    if output_format == "csv":
        return _csv_text(response.results, path_mode=path_mode, base_dir=base_dir)
    if output_format == "json":
        payload = _search_payload(response, db_path=db_path, concise=True)
        payload["export"] = {"format": "json", "path_mode": path_mode}
        for result in payload["results"]:
            original_path = result["path"]
            result["original_path"] = original_path
            result["path"] = _export_result_path(
                original_path,
                path_mode=path_mode,
                base_dir=base_dir,
            )
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
    raise ValueError(f"unsupported export format: {output_format}")


def _write_or_print(out: Path | None, content: str) -> None:
    if out is None:
        print(content, end="")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def _export_result_path(path: str, *, path_mode: str, base_dir: Path) -> str:
    track_path = Path(path).expanduser()
    if path_mode == "absolute":
        return str(track_path.resolve())
    if path_mode == "relative":
        try:
            return os.path.relpath(track_path.resolve(), start=base_dir.resolve())
        except ValueError:
            return str(track_path)
    return path


def _search_payload(response: Any, *, db_path: str, concise: bool) -> dict[str, Any]:
    if not concise:
        return {"db_path": db_path, **response.as_dict()}

    intent = response.intent
    return {
        "db_path": db_path,
        "query": response.query,
        "parser": intent.parser,
        "llm_error": intent.llm_error,
        "llm_hints": intent.llm_hints.as_dict() if intent.llm_hints else None,
        "intent": {
            "limit": intent.limit,
            "contexts": intent.contexts,
            "prefer_tags": intent.prefer_tags,
            "avoid_tags": intent.avoid_tags,
            "feature_ranges": {
                name: feature_range.as_dict()
                for name, feature_range in intent.feature_ranges.items()
            },
            "sort_by": [sort_spec.as_dict() for sort_spec in intent.sort_by],
            "semantic_model": intent.semantic_model,
            "use_semantic": intent.use_semantic,
        },
        "diagnostics": response.diagnostics,
        "results": [_concise_result(result) for result in response.results],
    }


def _concise_result(result: Any) -> dict[str, Any]:
    breakdown = result.breakdown
    return {
        "track_id": result.track_id,
        "path": result.path,
        "title": result.title,
        "artist": result.artist,
        "album": result.album,
        "genre": result.genre,
        "score": result.score,
        "raw_score": round(float(breakdown.get("raw_score", result.score)), 6),
        "why": result.explanation,
        "scores": {
            "semantic": round(float(breakdown.get("semantic_score", 0.0)), 6),
            "tags": round(float(breakdown.get("tag_score", 0.0)), 6),
            "features": round(float(breakdown.get("feature_score", 0.0)), 6),
            "text": round(float(breakdown.get("text_score", 0.0)), 6),
            "feedback": round(float(breakdown.get("feedback_score", 0.0)), 6),
        },
        "matched_tags": [
            {
                "tag": tag["tag"],
                "score": tag["score"],
                "source": tag["source"],
            }
            for tag in (breakdown.get("matched_tags") or [])[:5]
        ],
    }


def _maybe_parse_with_llm(
    conn: sqlite3.Connection,
    query: str,
    *,
    use_llm: bool,
    llm_provider: str,
    llm_model: str | None,
    timeout_sec: float,
    include_missing: bool,
) -> tuple[Any | None, str, str | None]:
    if not use_llm:
        return None, "dynamic", None
    try:
        profile = build_library_profile(conn, include_missing=include_missing)
        hints = parse_intent_llm(
            query,
            profile,
            provider=llm_provider,
            model=llm_model
            or (default_gemini_model() if llm_provider == "gemini" else None),
            timeout_sec=timeout_sec,
        )
    except LLMIntentError as exc:
        return None, "dynamic", str(exc)
    return hints, f"dynamic+{llm_provider}", None


def _print_m3u(results: list[Any]) -> None:
    print(_m3u_text(results, path_mode="stored", base_dir=Path.cwd()), end="")


def _m3u_text(results: list[Any], *, path_mode: str, base_dir: Path) -> str:
    lines = ["#EXTM3U"]
    for result in results:
        artist_title = " - ".join(
            part for part in [result.artist, result.title] if part
        ) or result.path
        lines.append(f"#EXTINF:-1,{artist_title}")
        lines.append(
            _export_result_path(result.path, path_mode=path_mode, base_dir=base_dir)
        )
    return "\n".join(lines) + "\n"


def _csv_text(results: list[Any], *, path_mode: str, base_dir: Path) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=["track_id", "path", "title", "artist", "album", "genre", "score"],
    )
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "track_id": result.track_id,
                "path": _export_result_path(
                    result.path,
                    path_mode=path_mode,
                    base_dir=base_dir,
                ),
                "title": result.title or "",
                "artist": result.artist or "",
                "album": result.album or "",
                "genre": result.genre or "",
                "score": result.score,
            }
        )
    return stream.getvalue()


def _sqlite_fts5_available() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE fts_check USING fts5(value)")
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()
    return True


def main() -> None:
    app()


if __name__ == "__main__":
    main()
