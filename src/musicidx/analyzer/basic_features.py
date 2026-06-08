"""Basic deterministic audio feature extraction with librosa."""

from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from musicidx.db import utc_now
from musicidx.profiles import rebuild_track_profile

ANALYSIS_VERSION = 1
DEFAULT_SAMPLE_RATE = 22050
QUICK_DURATION_SEC = 120.0
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class AudioAnalysisError(RuntimeError):
    """Raised when basic audio analysis cannot be completed."""


@dataclass(slots=True)
class AudioFeatures:
    """Basic audio features stored in the local index."""

    bpm: float | None = None
    key_name: str | None = None
    mode: str | None = None
    dynamic_range: float | None = None
    energy: float | None = None
    danceability: float | None = None
    aggression: float | None = None
    brightness: float | None = None
    spectral_centroid_mean: float | None = None
    spectral_centroid_std: float | None = None
    spectral_flatness_mean: float | None = None
    spectral_rolloff_mean: float | None = None
    zero_crossing_rate_mean: float | None = None
    mfcc_mean: list[float] = field(default_factory=list)
    mfcc_std: list[float] = field(default_factory=list)
    chroma_profile: list[float] = field(default_factory=list)
    raw_features: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BasicAnalysisSummary:
    """Summary counters for a basic audio analysis run."""

    processed: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    analysis_version: int = ANALYSIS_VERSION

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def is_librosa_available() -> bool:
    """Return True when librosa is importable."""
    return importlib.util.find_spec("librosa") is not None


def analyze_basic_features(
    path: Path,
    *,
    quick: bool = False,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AudioFeatures:
    """Analyze a track and return deterministic audio descriptors."""
    try:
        import librosa  # type: ignore[import-untyped]
        import numpy as np
    except ImportError as exc:
        raise AudioAnalysisError("librosa/numpy are not installed") from exc

    try:
        y, sr = librosa.load(
            str(path),
            sr=sample_rate,
            mono=True,
            duration=QUICK_DURATION_SEC if quick else None,
        )
    except Exception as exc:  # pragma: no cover - exact decoder exceptions vary by backend
        raise AudioAnalysisError(f"failed to decode audio: {exc}") from exc

    if y is None or len(y) == 0:
        raise AudioAnalysisError("decoded audio was empty")

    try:
        rms = librosa.feature.rms(y=y)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    except Exception as exc:  # pragma: no cover - librosa internals vary by version
        raise AudioAnalysisError(f"failed to compute audio features: {exc}") from exc

    rms_mean = _safe_mean(rms, np=np)
    rms_std = _safe_std(rms, np=np)
    centroid_mean = _safe_mean(spectral_centroid, np=np)
    centroid_std = _safe_std(spectral_centroid, np=np)
    flatness_mean = _safe_mean(spectral_flatness, np=np)
    rolloff_mean = _safe_mean(spectral_rolloff, np=np)
    zcr_mean = _safe_mean(zero_crossing_rate, np=np)
    onset_mean = _safe_mean(onset_env, np=np)
    dynamic_range = _safe_percentile(rms, 95, np=np) - _safe_percentile(rms, 5, np=np)
    bpm = _safe_float(np.asarray(tempo_raw).mean())

    mfcc_mean = [_round_float(value) for value in np.mean(mfcc, axis=1).tolist()]
    mfcc_std = [_round_float(value) for value in np.std(mfcc, axis=1).tolist()]
    chroma_profile = [_round_float(value) for value in np.mean(chroma, axis=1).tolist()]
    key_name, mode = estimate_key_mode(chroma_profile)

    energy = _clamp(rms_mean / 0.20)
    brightness = _clamp(centroid_mean / 5000.0)
    zcr_score = _clamp(zcr_mean / 0.18)
    flatness_score = _clamp(flatness_mean / 0.50)
    aggression = _clamp((0.45 * energy) + (0.35 * zcr_score) + (0.20 * flatness_score))
    tempo_score = _tempo_dance_score(bpm)
    onset_score = _clamp(onset_mean / 3.0)
    danceability = _clamp((0.45 * tempo_score) + (0.35 * onset_score) + (0.20 * energy))

    raw_features = {
        "analysis_version": ANALYSIS_VERSION,
        "quick": quick,
        "sample_rate": int(sr),
        "rms_mean": _round_float(rms_mean),
        "rms_std": _round_float(rms_std),
        "onset_strength_mean": _round_float(onset_mean),
        "chroma_profile": chroma_profile,
    }

    return AudioFeatures(
        bpm=_round_optional(bpm),
        key_name=key_name,
        mode=mode,
        dynamic_range=_round_float(dynamic_range),
        energy=_round_float(energy),
        danceability=_round_float(danceability),
        aggression=_round_float(aggression),
        brightness=_round_float(brightness),
        spectral_centroid_mean=_round_float(centroid_mean),
        spectral_centroid_std=_round_float(centroid_std),
        spectral_flatness_mean=_round_float(flatness_mean),
        spectral_rolloff_mean=_round_float(rolloff_mean),
        zero_crossing_rate_mean=_round_float(zcr_mean),
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        chroma_profile=chroma_profile,
        raw_features=raw_features,
    )


def estimate_key_mode(chroma_profile: list[float]) -> tuple[str | None, str | None]:
    """Estimate a rough key and mode from a 12-bin chroma profile."""
    if len(chroma_profile) != 12 or not any(chroma_profile):
        return None, None

    chroma_sum = sum(chroma_profile)
    if chroma_sum <= 0:
        return None, None
    chroma = [value / chroma_sum for value in chroma_profile]

    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    best_score = -math.inf
    best_key = 0
    best_mode = "major"

    for root in range(12):
        major_score = _dot(chroma, _rotate(major_profile, root))
        minor_score = _dot(chroma, _rotate(minor_profile, root))
        if major_score > best_score:
            best_score = major_score
            best_key = root
            best_mode = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = root
            best_mode = "minor"

    return KEY_NAMES[best_key], best_mode


def process_basic_analysis(
    conn: sqlite3.Connection,
    *,
    track_id: str | None = None,
    quick: bool = False,
    workers: int = 1,
) -> BasicAnalysisSummary:
    """Analyze selected tracks and persist feature rows."""
    summary = BasicAnalysisSummary()
    jobs = _select_tracks_for_analysis(conn, track_id=track_id)

    if workers <= 1:
        for job in jobs:
            _process_one_job(conn, job, summary, quick=quick)
        conn.commit()
        return summary

    pending_jobs: list[dict[str, Any]] = []
    for job in jobs:
        if _should_skip_job(conn, job):
            summary.skipped += 1
            continue
        path = Path(job["path"])
        if not path.exists():
            summary.skipped += 1
            _record_track_error(conn, job["id"], "file is missing on disk")
            continue
        pending_jobs.append(job)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(analyze_basic_features, Path(job["path"]), quick=quick): job
            for job in pending_jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            summary.processed += 1
            try:
                features = future.result()
                save_audio_features(conn, job["id"], features)
                summary.updated += 1
            except AudioAnalysisError as exc:
                summary.errors += 1
                _record_track_error(conn, job["id"], str(exc))
            except Exception as exc:  # pragma: no cover - defensive safety net
                summary.errors += 1
                _record_track_error(conn, job["id"], f"unexpected analysis error: {exc}")

    conn.commit()
    return summary


def save_audio_features(
    conn: sqlite3.Connection,
    track_id: str,
    features: AudioFeatures,
) -> None:
    """Persist audio features and refresh profile/FTS text."""
    now = utc_now()
    conn.execute(
        """
        INSERT INTO audio_features (
            track_id, bpm, key_name, mode, dynamic_range, energy, danceability,
            aggression, brightness, spectral_centroid_mean, spectral_centroid_std,
            spectral_flatness_mean, spectral_rolloff_mean, zero_crossing_rate_mean,
            mfcc_mean_json, mfcc_std_json, raw_features_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            bpm = excluded.bpm,
            key_name = excluded.key_name,
            mode = excluded.mode,
            dynamic_range = excluded.dynamic_range,
            energy = excluded.energy,
            danceability = excluded.danceability,
            aggression = excluded.aggression,
            brightness = excluded.brightness,
            spectral_centroid_mean = excluded.spectral_centroid_mean,
            spectral_centroid_std = excluded.spectral_centroid_std,
            spectral_flatness_mean = excluded.spectral_flatness_mean,
            spectral_rolloff_mean = excluded.spectral_rolloff_mean,
            zero_crossing_rate_mean = excluded.zero_crossing_rate_mean,
            mfcc_mean_json = excluded.mfcc_mean_json,
            mfcc_std_json = excluded.mfcc_std_json,
            raw_features_json = excluded.raw_features_json,
            updated_at = excluded.updated_at
        """,
        (
            track_id,
            features.bpm,
            features.key_name,
            features.mode,
            features.dynamic_range,
            features.energy,
            features.danceability,
            features.aggression,
            features.brightness,
            features.spectral_centroid_mean,
            features.spectral_centroid_std,
            features.spectral_flatness_mean,
            features.spectral_rolloff_mean,
            features.zero_crossing_rate_mean,
            json.dumps(features.mfcc_mean),
            json.dumps(features.mfcc_std),
            json.dumps(features.raw_features, sort_keys=True),
            now,
        ),
    )
    conn.execute(
        """
        UPDATE tracks
        SET analysis_version = ?, analyzed_at = ?, last_error = NULL
        WHERE id = ?
        """,
        (ANALYSIS_VERSION, now, track_id),
    )
    refresh_profile_with_audio_features(conn, track_id, features, updated_at=now)


def refresh_profile_with_audio_features(
    conn: sqlite3.Connection,
    track_id: str,
    features: AudioFeatures,
    *,
    updated_at: str | None = None,
) -> None:
    """Refresh track profile text/JSON and FTS after analysis."""
    _ = features
    rebuild_track_profile(conn, track_id, updated_at=updated_at)


def describe_audio_features(features: AudioFeatures) -> str:
    """Build deterministic human-readable audio descriptors."""
    parts: list[str] = []
    if features.energy is not None:
        parts.append(f"{_band(features.energy, 'energy')} energy")
    if features.bpm is not None:
        parts.append(f"tempo around {features.bpm:.0f} BPM")
    if features.brightness is not None:
        parts.append(f"{_band(features.brightness, 'brightness')} brightness")
    if features.danceability is not None:
        parts.append(f"{_band(features.danceability, 'danceability')} danceability")
    if features.aggression is not None:
        parts.append(f"{_band(features.aggression, 'aggression')} aggression")
    if features.key_name and features.mode:
        parts.append(f"rough key {features.key_name} {features.mode}")
    return ", ".join(parts) + "." if parts else "basic audio features analyzed."


def _process_one_job(
    conn: sqlite3.Connection,
    job: sqlite3.Row,
    summary: BasicAnalysisSummary,
    *,
    quick: bool,
) -> None:
    if _should_skip_job(conn, job):
        summary.skipped += 1
        return

    path = Path(job["path"])
    if not path.exists():
        summary.skipped += 1
        _record_track_error(conn, job["id"], "file is missing on disk")
        return

    summary.processed += 1
    try:
        features = analyze_basic_features(path, quick=quick)
        save_audio_features(conn, job["id"], features)
        summary.updated += 1
    except AudioAnalysisError as exc:
        summary.errors += 1
        _record_track_error(conn, job["id"], str(exc))
    except Exception as exc:  # pragma: no cover - defensive safety net
        summary.errors += 1
        _record_track_error(conn, job["id"], f"unexpected analysis error: {exc}")


def _select_tracks_for_analysis(
    conn: sqlite3.Connection,
    *,
    track_id: str | None,
) -> list[sqlite3.Row]:
    clauses = ["missing_at IS NULL"]
    params: list[Any] = []
    if track_id is not None:
        clauses.append("id = ?")
        params.append(track_id)

    return conn.execute(
        f"""
        SELECT id, path, analysis_version
        FROM tracks
        WHERE {' AND '.join(clauses)}
        ORDER BY path
        """,
        params,
    ).fetchall()


def _should_skip_job(conn: sqlite3.Connection, job: sqlite3.Row | dict[str, Any]) -> bool:
    if int(job["analysis_version"] or 0) < ANALYSIS_VERSION:
        return False
    row = conn.execute(
        "SELECT 1 FROM audio_features WHERE track_id = ?",
        (job["id"],),
    ).fetchone()
    return row is not None


def _record_track_error(conn: sqlite3.Connection, track_id: str, error: str) -> None:
    conn.execute("UPDATE tracks SET last_error = ? WHERE id = ?", (error, track_id))


def _safe_mean(values: Any, *, np: Any) -> float:
    if len(values) == 0:
        return 0.0
    value = float(np.mean(values))
    return value if math.isfinite(value) else 0.0


def _safe_std(values: Any, *, np: Any) -> float:
    if len(values) == 0:
        return 0.0
    value = float(np.std(values))
    return value if math.isfinite(value) else 0.0


def _safe_percentile(values: Any, percentile: float, *, np: Any) -> float:
    if len(values) == 0:
        return 0.0
    value = float(np.percentile(values, percentile))
    return value if math.isfinite(value) else 0.0


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _round_optional(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _tempo_dance_score(bpm: float | None) -> float:
    if bpm is None or bpm <= 0:
        return 0.0
    if 90 <= bpm <= 130:
        return 1.0
    if 60 <= bpm < 90:
        return (bpm - 60) / 30
    if 130 < bpm <= 170:
        return (170 - bpm) / 40
    return 0.0


def _band(value: float, name: str) -> str:
    if value < 0.33:
        return "low"
    if value < 0.66:
        return "medium"
    if name == "brightness":
        return "high"
    return "high"


def _rotate(values: list[float], amount: int) -> list[float]:
    return values[-amount:] + values[:-amount] if amount else values[:]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
