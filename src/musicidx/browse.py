"""SQLite-backed indexed-library browser and simple metadata search."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

BrowseSort = Literal["artist", "title", "album", "genre", "bpm", "duration", "path", "indexed_at"]
BrowseDirection = Literal["asc", "desc"]

_SORT_SQL: dict[BrowseSort, tuple[str, ...]] = {
    "artist": ("lower(COALESCE(t.artist, ''))", "lower(COALESCE(t.title, ''))", "t.path"),
    "title": ("lower(COALESCE(t.title, ''))", "lower(COALESCE(t.artist, ''))", "t.path"),
    "album": (
        "lower(COALESCE(t.album, ''))",
        "lower(COALESCE(t.artist, ''))",
        "lower(COALESCE(t.title, ''))",
        "t.path",
    ),
    "genre": (
        "lower(COALESCE(t.genre, ''))",
        "lower(COALESCE(t.artist, ''))",
        "lower(COALESCE(t.title, ''))",
        "t.path",
    ),
    "bpm": ("af.bpm",),
    "duration": ("t.duration_sec",),
    "path": ("lower(t.path)",),
    "indexed_at": ("t.indexed_at",),
}


def browse_library(
    conn: sqlite3.Connection,
    *,
    path: Path | str | None = None,
    query: str | None = None,
    sort: str = "artist",
    direction: str = "asc",
    include_missing: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    """Return indexed roots, child folders, and paginated tracks under ``path``.

    The browser intentionally uses SQLite as the source of truth instead of
    walking the filesystem. Folders are synthesized from indexed track paths.
    When ``query`` is provided, track rows become a recursive metadata/path
    search under ``path``; otherwise they are direct children of ``path``.
    """
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    safe_sort = _normalise_sort(sort)
    safe_direction = _normalise_direction(direction)
    search_query = (query or "").strip()

    roots = _library_roots(conn, include_missing=include_missing)
    selected_root = _select_root(roots, path)
    cwd = _normalise_path(path) if path is not None else selected_root
    warning: str | None = None

    if selected_root is not None and cwd is not None and not _is_relative_to(cwd, selected_root):
        warning = "Requested path is outside indexed library roots; showing nearest indexed root."
        cwd = selected_root
    elif selected_root is None and roots:
        cwd = roots[0]["path"]
        selected_root = cwd
        warning = "Requested path is outside indexed library roots; showing first indexed root."

    if cwd is None:
        cwd = _normalise_path(path) if path is not None else None

    folders: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    parent: str | None = None
    total_tracks = 0
    mode = "search" if search_query else "browse"

    if cwd is not None:
        selected_root = selected_root or _select_root(roots, cwd)
        if selected_root is not None and cwd != selected_root:
            parent_path = cwd.parent
            parent = str(parent_path) if _is_relative_to(parent_path, selected_root) else None
        folders = _folders_for_path(conn, cwd, include_missing=include_missing)
        if search_query:
            tracks, total_tracks = _search_tracks_for_path(
                conn,
                cwd,
                query=search_query,
                include_missing=include_missing,
                limit=safe_limit,
                offset=safe_offset,
                sort=safe_sort,
                direction=safe_direction,
            )
        else:
            tracks, total_tracks = _direct_tracks_for_path(
                conn,
                cwd,
                include_missing=include_missing,
                limit=safe_limit,
                offset=safe_offset,
                sort=safe_sort,
                direction=safe_direction,
            )

    return {
        "roots": roots,
        "cwd": str(cwd) if cwd is not None else None,
        "root": str(selected_root) if selected_root is not None else None,
        "parent": parent,
        "folders": folders,
        "tracks": tracks,
        "track_count": total_tracks,
        "limit": safe_limit,
        "offset": safe_offset,
        "has_more": safe_offset + len(tracks) < total_tracks,
        "mode": mode,
        "query": search_query,
        "sort": safe_sort,
        "sort_direction": safe_direction,
        "include_missing": include_missing,
        "warning": warning,
    }


def _library_roots(conn: sqlite3.Connection, *, include_missing: bool) -> list[dict[str, Any]]:
    missing_clause = "" if include_missing else "AND t.missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT r.path, COUNT(t.id) AS track_count
        FROM library_roots r
        LEFT JOIN tracks t ON t.root_id = r.id {missing_clause}
        GROUP BY r.id, r.path
        ORDER BY r.path
        """
    ).fetchall()
    return [
        {
            "path": row["path"],
            "name": Path(row["path"]).name or row["path"],
            "track_count": row["track_count"],
        }
        for row in rows
    ]


def _folders_for_path(
    conn: sqlite3.Connection,
    cwd: Path,
    *,
    include_missing: bool,
) -> list[dict[str, Any]]:
    prefix = _path_prefix(cwd)
    missing_clause = "" if include_missing else "AND missing_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT path
        FROM tracks
        WHERE path LIKE ? {missing_clause}
        """,
        (f"{prefix}%",),
    ).fetchall()
    folder_counts: dict[str, int] = {}
    for row in rows:
        try:
            relative = Path(row["path"]).relative_to(cwd)
        except ValueError:
            continue
        if len(relative.parts) > 1:
            folder_name = relative.parts[0]
            folder_counts[folder_name] = folder_counts.get(folder_name, 0) + 1
    return [
        {"name": name, "path": str(cwd / name), "track_count": count}
        for name, count in sorted(folder_counts.items(), key=lambda item: item[0].lower())
    ]


def _direct_tracks_for_path(
    conn: sqlite3.Connection,
    cwd: Path,
    *,
    include_missing: bool,
    limit: int,
    offset: int,
    sort: BrowseSort,
    direction: BrowseDirection,
) -> tuple[list[dict[str, Any]], int]:
    rows = _track_rows_for_path(conn, cwd, include_missing=include_missing)
    direct_rows = []
    for row in rows:
        try:
            relative = Path(row["path"]).relative_to(cwd)
        except ValueError:
            continue
        if len(relative.parts) == 1:
            direct_rows.append(row)
    _sort_direct_rows(direct_rows, sort=sort, direction=direction)
    total = len(direct_rows)
    return [_track_row(row) for row in direct_rows[offset : offset + limit]], total


def _search_tracks_for_path(
    conn: sqlite3.Connection,
    cwd: Path,
    *,
    query: str,
    include_missing: bool,
    limit: int,
    offset: int,
    sort: BrowseSort,
    direction: BrowseDirection,
) -> tuple[list[dict[str, Any]], int]:
    where, params = _base_track_where(cwd, include_missing=include_missing)
    for term in _query_terms(query):
        where.append(
            """
            lower(
                COALESCE(t.title, '') || ' ' ||
                COALESCE(t.artist, '') || ' ' ||
                COALESCE(t.album, '') || ' ' ||
                COALESCE(t.genre, '') || ' ' ||
                COALESCE(t.album_artist, '') || ' ' ||
                t.path
            ) LIKE ? ESCAPE '\\'
            """
        )
        params.append(f"%{_escape_like(term)}%")

    where_sql = " AND ".join(f"({clause})" for clause in where)
    total = conn.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM tracks t
        LEFT JOIN audio_features af ON af.track_id = t.id
        WHERE {where_sql}
        """,
        params,
    ).fetchone()["count"]
    rows = conn.execute(
        f"""
        SELECT {_track_select_columns()}
        FROM tracks t
        LEFT JOIN audio_features af ON af.track_id = t.id
        WHERE {where_sql}
        ORDER BY {_order_by_sql(sort, direction)}
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()
    return [_track_row(row) for row in rows], total


def _track_rows_for_path(
    conn: sqlite3.Connection,
    cwd: Path,
    *,
    include_missing: bool,
) -> list[sqlite3.Row]:
    where, params = _base_track_where(cwd, include_missing=include_missing)
    return conn.execute(
        f"""
        SELECT {_track_select_columns()}
        FROM tracks t
        LEFT JOIN audio_features af ON af.track_id = t.id
        WHERE {" AND ".join(f"({clause})" for clause in where)}
        """,
        params,
    ).fetchall()


def _base_track_where(cwd: Path, *, include_missing: bool) -> tuple[list[str], list[Any]]:
    prefix = _path_prefix(cwd)
    where = ["(t.path = ? OR t.path LIKE ?)"]
    params: list[Any] = [str(cwd), f"{prefix}%"]
    if not include_missing:
        where.append("t.missing_at IS NULL")
    return where, params


def _track_select_columns() -> str:
    return """
        t.id AS track_id,
        t.path,
        t.title,
        t.artist,
        t.album,
        t.genre,
        t.duration_sec,
        t.indexed_at,
        t.missing_at,
        af.bpm
    """


def _track_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "track_id": row["track_id"],
        "path": row["path"],
        "title": row["title"],
        "artist": row["artist"],
        "album": row["album"],
        "genre": row["genre"],
        "duration_sec": row["duration_sec"],
        "bpm": row["bpm"],
        "missing": row["missing_at"] is not None,
    }


def _select_root(roots: list[dict[str, Any]], path: Path | str | None) -> Path | None:
    root_paths = [_normalise_path(root["path"]) for root in roots]
    if path is None:
        return root_paths[0] if root_paths else None
    selected = _normalise_path(path)
    candidates = [root for root in root_paths if _is_relative_to(selected, root)]
    if candidates:
        return max(candidates, key=lambda item: len(str(item)))
    return None


def _normalise_sort(sort: str) -> BrowseSort:
    if sort in _SORT_SQL:
        return sort  # type: ignore[return-value]
    return "artist"


def _normalise_direction(direction: str) -> BrowseDirection:
    return "desc" if direction == "desc" else "asc"


def _order_by_sql(sort: BrowseSort, direction: BrowseDirection) -> str:
    direction_sql = "DESC" if direction == "desc" else "ASC"
    expressions = _SORT_SQL[sort]
    if sort in {"bpm", "duration", "indexed_at"}:
        expression = expressions[0]
        return (
            f"{expression} IS NULL, {expression} {direction_sql}, "
            "lower(COALESCE(t.title, '')) ASC, t.path ASC"
        )
    return ", ".join(f"{expression} {direction_sql}" for expression in expressions)


def _sort_direct_rows(
    rows: list[sqlite3.Row],
    *,
    sort: BrowseSort,
    direction: BrowseDirection,
) -> None:
    if direction == "asc" or sort not in {"bpm", "duration", "indexed_at"}:
        rows.sort(key=_python_sort_key(sort), reverse=direction == "desc")
        return
    rows.sort(key=_python_desc_numeric_sort_key(sort))


def _python_desc_numeric_sort_key(sort: BrowseSort):
    def key(row: sqlite3.Row) -> tuple[Any, ...]:
        if sort == "bpm":
            return (row["bpm"] is None, -(row["bpm"] or 0.0), _text(row["title"]), row["path"])
        if sort == "duration":
            return (
                row["duration_sec"] is None,
                -(row["duration_sec"] or 0.0),
                _text(row["title"]),
                row["path"],
            )
        return (row["indexed_at"] is None, _reverse_text(row["indexed_at"]), row["path"])

    return key


def _python_sort_key(sort: BrowseSort):
    def key(row: sqlite3.Row) -> tuple[Any, ...]:
        if sort == "bpm":
            return (row["bpm"] is None, row["bpm"] or 0.0, _text(row["title"]), row["path"])
        if sort == "duration":
            return (
                row["duration_sec"] is None,
                row["duration_sec"] or 0.0,
                _text(row["title"]),
                row["path"],
            )
        if sort == "indexed_at":
            return (row["indexed_at"] is None, row["indexed_at"] or "", row["path"])
        if sort == "title":
            return (_text(row["title"]), _text(row["artist"]), row["path"])
        if sort == "album":
            return (_text(row["album"]), _text(row["artist"]), _text(row["title"]), row["path"])
        if sort == "genre":
            return (_text(row["genre"]), _text(row["artist"]), _text(row["title"]), row["path"])
        if sort == "path":
            return (_text(row["path"]),)
        return (_text(row["artist"]), _text(row["title"]), row["path"])

    return key


def _text(value: Any) -> str:
    return str(value or "").lower()


def _reverse_text(value: Any) -> str:
    return "".join(chr(0x10FFFF - ord(char)) for char in str(value or ""))


def _query_terms(query: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[\w'-]+", query)[:12]]


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _normalise_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_prefix(path: Path) -> str:
    value = str(path)
    return value if value.endswith("/") else f"{value}/"
