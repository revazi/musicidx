"""Typer command-line interface for MusicIdx."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from musicidx import __version__
from musicidx.analyzer.basic_features import is_librosa_available, process_basic_analysis
from musicidx.analyzer.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    is_sentence_transformers_available,
    process_embeddings,
    search_semantic,
)
from musicidx.analyzer.essentia_models import (
    DEFAULT_MIN_SCORE,
    is_essentia_available,
    list_track_tags,
    model_manifest_status,
    process_tags,
)
from musicidx.config import (
    DB_PATH_ENV_VAR,
    DEFAULT_DB_FILENAME,
    DEFAULT_MODELS_DIRNAME,
    MODELS_PATH_ENV_VAR,
    resolve_db_path,
    resolve_models_path,
)
from musicidx.db import CORE_TABLES, connect_db, db_info, init_db
from musicidx.fingerprint import find_duplicate_groups, is_fpcalc_available, process_fingerprints
from musicidx.metadata import is_ffprobe_available, process_metadata, search_text
from musicidx.scanner import scan_library

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
LimitOption = Annotated[
    int,
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
WorkersOption = Annotated[
    int,
    typer.Option("--workers", min=1, max=32, help="Number of analysis worker threads."),
]
MinScoreOption = Annotated[
    float,
    typer.Option("--min-score", min=0.0, max=1.0, help="Minimum tag score to store."),
]
EmbeddingModelOption = Annotated[
    str,
    typer.Option("--model", help="Sentence-transformers model name for profile embeddings."),
]
BatchSizeOption = Annotated[
    int,
    typer.Option("--batch-size", min=1, max=512, help="Embedding batch size."),
]
RefreshOption = Annotated[
    bool,
    typer.Option("--refresh", help="Recompute embeddings even when stored text is current."),
]


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _status(available: bool) -> str:
    return "available" if available else "missing"


@app.command("doctor")
def doctor_command(json_output: JsonOption = False) -> None:
    """Check local dependencies and runtime capabilities."""
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
            "status": _status(shutil.which("ffprobe") is not None),
            "detail": shutil.which("ffprobe") or "ffprobe not found on PATH",
        },
        {
            "name": "fpcalc",
            "status": _status(shutil.which("fpcalc") is not None),
            "detail": shutil.which("fpcalc") or "fpcalc not found on PATH",
        },
        {
            "name": "Ollama",
            "status": _status(shutil.which("ollama") is not None),
            "detail": shutil.which("ollama") or "ollama not found on PATH",
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
    ]

    payload = {"version": __version__, "checks": checks}
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

    payload = {"db_path": str(db_path), **summary.as_dict()}
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
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_metadata(conn, track_id=track_id, missing_only=missing_only)
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "ffprobe_available": is_ffprobe_available(),
        **summary.as_dict(),
    }
    if json_output:
        _print_json(payload)
        return

    if not payload["ffprobe_available"]:
        console.print(
            "[yellow]ffprobe is not available on PATH; metadata extraction failed.[/yellow]"
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
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_fingerprints(conn, track_id=track_id, missing_only=missing_only)
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "fpcalc_available": is_fpcalc_available(),
        **summary.as_dict(),
    }
    if json_output:
        _print_json(payload)
        return

    if not payload["fpcalc_available"]:
        console.print(
            "[yellow]fpcalc is not available on PATH; fingerprinting failed.[/yellow]"
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
    workers: WorkersOption = 1,
    json_output: JsonOption = False,
) -> None:
    """Analyze basic audio features for scanned tracks."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_basic_analysis(
            conn,
            track_id=track_id,
            quick=quick,
            workers=workers,
        )
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "librosa_available": is_librosa_available(),
        "quick": quick,
        "workers": workers,
        **summary.as_dict(),
    }
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
    console.print(table)


@app.command("analyze-tags")
def analyze_tags_command(
    db: DbOption = None,
    models_path: ModelsPathOption = None,
    track_id: TrackIdOption = None,
    missing_only: TagMissingOnlyOption = False,
    min_score: MinScoreOption = DEFAULT_MIN_SCORE,
    workers: WorkersOption = 1,
    json_output: JsonOption = False,
) -> None:
    """Analyze ML mood/genre tags with local Essentia models."""
    db_path = resolve_db_path(db)
    resolved_models_path = resolve_models_path(models_path)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_tags(
            conn,
            models_path=resolved_models_path,
            track_id=track_id,
            missing_only=missing_only,
            min_score=min_score,
            workers=workers,
        )
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "models_path": str(resolved_models_path),
        "essentia_available": is_essentia_available(),
        "min_score": min_score,
        "workers": workers,
        **summary.as_dict(),
    }
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
    for key in ["processed", "updated", "skipped", "errors", "model_count"]:
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
    batch_size: BatchSizeOption = 32,
    refresh: RefreshOption = False,
    json_output: JsonOption = False,
) -> None:
    """Embed enriched track profile text for semantic search."""
    db_path = resolve_db_path(db)
    conn = connect_db(db_path)
    try:
        init_db(conn)
        summary = process_embeddings(
            conn,
            track_id=track_id,
            model_name=model,
            batch_size=batch_size,
            refresh=refresh,
        )
    finally:
        conn.close()

    payload = {
        "db_path": str(db_path),
        "sentence_transformers_available": is_sentence_transformers_available(),
        "batch_size": batch_size,
        "refresh": refresh,
        **summary.as_dict(),
    }
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
