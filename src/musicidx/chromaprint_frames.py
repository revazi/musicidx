"""Helpers for storing and comparing decoded Chromaprint fingerprints."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import chromaprint


@dataclass(frozen=True, slots=True)
class DecodedChromaprint:
    frames: tuple[int, ...]
    algorithm: int

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def decode_chromaprint_text(value: str | None) -> DecodedChromaprint | None:
    """Decode fpcalc/chromaprint text into integer frames."""
    if not value:
        return None
    try:
        frames, algorithm = chromaprint.decode_fingerprint(str(value).encode("ascii"))
    except (chromaprint.FingerprintError, TypeError, ValueError, UnicodeEncodeError):
        return None
    decoded = tuple(int(frame) for frame in frames)
    if not decoded:
        return None
    return DecodedChromaprint(frames=decoded, algorithm=int(algorithm))


def frames_to_blob(frames: tuple[int, ...] | list[int]) -> bytes | None:
    """Pack uint32 Chromaprint frames into a stable little-endian SQLite blob."""
    if not frames:
        return None
    return struct.pack(f"<{len(frames)}I", *[int(frame) & 0xFFFFFFFF for frame in frames])


def blob_to_frames(
    blob: bytes | memoryview | None,
    frame_count: int | None = None,
) -> tuple[int, ...]:
    """Unpack stored uint32 Chromaprint frames from SQLite."""
    if not blob:
        return ()
    data = bytes(blob)
    if len(data) % 4 != 0:
        return ()
    count = len(data) // 4
    if frame_count is not None and int(frame_count) != count:
        return ()
    return tuple(int(value) for value in struct.unpack(f"<{count}I", data))


def decoded_storage_values(value: str | None) -> tuple[int | None, bytes | None, int | None]:
    """Return algorithm/blob/count values suitable for the tracks table."""
    decoded = decode_chromaprint_text(value)
    if decoded is None:
        return None, None, None
    return decoded.algorithm, frames_to_blob(decoded.frames), decoded.frame_count


def row_frames(row: Any) -> tuple[int, ...]:
    """Return decoded frames from a sqlite row, preferring stored decoded blobs."""
    keys = set(row.keys()) if hasattr(row, "keys") else set()
    if {"chromaprint_frames", "chromaprint_frame_count"}.issubset(keys):
        frames = blob_to_frames(row["chromaprint_frames"], row["chromaprint_frame_count"])
        if frames:
            return frames
    decoded = decode_chromaprint_text(row["chromaprint"] if "chromaprint" in keys else None)
    return decoded.frames if decoded else ()
