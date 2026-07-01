from __future__ import annotations

from musicidx.browse import browse_library
from musicidx.db import connect_db, init_db


def test_browse_library_returns_roots_folders_and_direct_tracks(tmp_path):
    db_path = tmp_path / "index.sqlite"
    root = tmp_path / "Music"
    house = root / "House"
    techno = root / "Techno"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        conn.execute(
            """
            INSERT INTO library_roots (id, path, created_at, updated_at)
            VALUES (1, ?, 'now', 'now')
            """,
            (str(root),),
        )
        conn.executemany(
            """
            INSERT INTO tracks (
                id, root_id, path, path_hash, extension, file_size, file_mtime_ns,
                title, artist, album, genre, duration_sec, indexed_at
            ) VALUES (?, 1, ?, ?, '.mp3', 1, 1, ?, ?, ?, ?, ?, 'now')
            """,
            [
                (
                    "direct",
                    str(root / "Intro.mp3"),
                    "hash-direct",
                    "Intro",
                    "A",
                    "Album",
                    "ambient",
                    60.0,
                ),
                (
                    "house-1",
                    str(house / "Track.mp3"),
                    "hash-house",
                    "Track",
                    "B",
                    "Album",
                    "house",
                    180.0,
                ),
                (
                    "techno-1",
                    str(techno / "Peak.mp3"),
                    "hash-techno",
                    "Peak",
                    "C",
                    "Album",
                    "techno",
                    200.0,
                ),
            ],
        )
        conn.execute(
            "INSERT INTO audio_features (track_id, bpm, updated_at) VALUES ('direct', 95, 'now')"
        )
        conn.commit()

        payload = browse_library(conn, path=root)

        assert payload["cwd"] == str(root.resolve(strict=False))
        assert payload["parent"] is None
        assert payload["roots"] == [{"path": str(root), "name": "Music", "track_count": 3}]
        assert [folder["name"] for folder in payload["folders"]] == ["House", "Techno"]
        assert payload["tracks"][0]["track_id"] == "direct"
        assert payload["tracks"][0]["bpm"] == 95
        assert payload["mode"] == "browse"
        assert payload["offset"] == 0
    finally:
        conn.close()


def test_browse_library_child_folder_has_parent(tmp_path):
    db_path = tmp_path / "index.sqlite"
    root = tmp_path / "Music"
    child = root / "House"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        conn.execute(
            """
            INSERT INTO library_roots (id, path, created_at, updated_at)
            VALUES (1, ?, 'now', 'now')
            """,
            (str(root),),
        )
        conn.execute(
            """
            INSERT INTO tracks (
                id, root_id, path, path_hash, extension, file_size, file_mtime_ns,
                title, indexed_at
            ) VALUES ('house-1', 1, ?, 'hash-house', '.mp3', 1, 1, 'Track', 'now')
            """,
            (str(child / "Track.mp3"),),
        )
        conn.commit()

        payload = browse_library(conn, path=child)

        assert payload["cwd"] == str(child.resolve(strict=False))
        assert payload["parent"] == str(root.resolve(strict=False))
        assert payload["tracks"][0]["track_id"] == "house-1"
    finally:
        conn.close()


def test_browse_library_searches_recursively_by_metadata_and_path(tmp_path):
    db_path = tmp_path / "index.sqlite"
    root = tmp_path / "Music"
    child = root / "Soul"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        conn.execute(
            """
            INSERT INTO library_roots (id, path, created_at, updated_at)
            VALUES (1, ?, 'now', 'now')
            """,
            (str(root),),
        )
        conn.executemany(
            """
            INSERT INTO tracks (
                id, root_id, path, path_hash, extension, file_size, file_mtime_ns,
                title, artist, album, genre, duration_sec, indexed_at
            ) VALUES (?, 1, ?, ?, '.mp3', 1, 1, ?, ?, ?, ?, ?, 'now')
            """,
            [
                (
                    "soul-1",
                    str(child / "Bobby Womack - Across 110th Street.mp3"),
                    "hash-soul",
                    "Across 110th Street",
                    "Bobby Womack",
                    "Soundtrack",
                    "soul",
                    220.0,
                ),
                (
                    "techno-1",
                    str(root / "Peak.mp3"),
                    "hash-techno",
                    "Peak",
                    "DJ Example",
                    "Club",
                    "techno",
                    200.0,
                ),
            ],
        )
        conn.commit()

        payload = browse_library(conn, path=root, query="bobby street", sort="title")

        assert payload["mode"] == "search"
        assert payload["track_count"] == 1
        assert payload["tracks"][0]["track_id"] == "soul-1"
    finally:
        conn.close()


def test_browse_library_paginates_and_sorts_direct_tracks(tmp_path):
    db_path = tmp_path / "index.sqlite"
    root = tmp_path / "Music"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        conn.execute(
            """
            INSERT INTO library_roots (id, path, created_at, updated_at)
            VALUES (1, ?, 'now', 'now')
            """,
            (str(root),),
        )
        conn.executemany(
            """
            INSERT INTO tracks (
                id, root_id, path, path_hash, extension, file_size, file_mtime_ns,
                title, artist, indexed_at
            ) VALUES (?, 1, ?, ?, '.mp3', 1, 1, ?, ?, 'now')
            """,
            [
                ("track-a", str(root / "A.mp3"), "hash-a", "A Title", "Artist C"),
                ("track-b", str(root / "B.mp3"), "hash-b", "B Title", "Artist B"),
                ("track-c", str(root / "C.mp3"), "hash-c", "C Title", "Artist A"),
            ],
        )
        conn.commit()

        payload = browse_library(conn, path=root, sort="artist", limit=1, offset=1)

        assert payload["track_count"] == 3
        assert payload["has_more"] is True
        assert [track["track_id"] for track in payload["tracks"]] == ["track-b"]
    finally:
        conn.close()
