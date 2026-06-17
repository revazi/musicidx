"""Adaptive resource recommendations for MusicIdx indexing."""

from __future__ import annotations

import ctypes
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

VALID_RESOURCE_PROFILES = {"auto", "low", "balanced", "full"}


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """Elapsed-time and best-effort peak-memory diagnostics for one CLI step."""

    started_at: str
    finished_at: str
    duration_sec: float
    peak_memory_bytes: int | None
    peak_memory_mb: float | None
    peak_memory_source: str | None
    child_peak_memory_bytes: int | None = None
    child_peak_memory_mb: float | None = None
    child_peak_memory_source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_memory_source": self.peak_memory_source,
            "child_peak_memory_bytes": self.child_peak_memory_bytes,
            "child_peak_memory_mb": self.child_peak_memory_mb,
            "child_peak_memory_source": self.child_peak_memory_source,
        }


class RuntimeTimer:
    """Measure wall time and best-effort RSS diagnostics for a command."""

    def __init__(self) -> None:
        self.started_at = datetime.now(UTC)
        self.started_perf = time.perf_counter()

    def finish(self, *, include_child_peak: bool = False) -> RuntimeDiagnostics:
        finished_at = datetime.now(UTC)
        peak = peak_rss_bytes()
        child_peak = child_peak_rss_bytes() if include_child_peak else (None, None)
        return RuntimeDiagnostics(
            started_at=self.started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_sec=round(time.perf_counter() - self.started_perf, 3),
            peak_memory_bytes=peak[0],
            peak_memory_mb=_bytes_to_mb(peak[0]),
            peak_memory_source=peak[1],
            child_peak_memory_bytes=child_peak[0],
            child_peak_memory_mb=_bytes_to_mb(child_peak[0]),
            child_peak_memory_source=child_peak[1],
        )


@dataclass(frozen=True)
class SystemResources:
    """Small cross-platform snapshot of host resources."""

    cpu_count: int
    total_memory_bytes: int | None

    @property
    def total_memory_gb(self) -> float | None:
        if self.total_memory_bytes is None:
            return None
        return self.total_memory_bytes / 1024**3

    def as_dict(self) -> dict[str, Any]:
        memory_gb = self.total_memory_gb
        return {
            "cpu_count": self.cpu_count,
            "total_memory_bytes": self.total_memory_bytes,
            "total_memory_gb": round(memory_gb, 2) if memory_gb is not None else None,
        }


@dataclass(frozen=True)
class IndexingResourcePlan:
    """Recommended indexing settings for a resource profile."""

    requested_profile: str
    effective_profile: str
    resources: SystemResources
    basic_workers: int
    tag_workers: int
    embedding_batch_size: int
    tag_batch_size: int
    basic_chunk_sec: float
    quick_basic: bool
    missing_only: bool
    reason: str
    warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_profile": self.requested_profile,
            "effective_profile": self.effective_profile,
            "resources": self.resources.as_dict(),
            "basic_workers": self.basic_workers,
            "tag_workers": self.tag_workers,
            "embedding_batch_size": self.embedding_batch_size,
            "tag_batch_size": self.tag_batch_size,
            "basic_chunk_sec": self.basic_chunk_sec,
            "quick_basic": self.quick_basic,
            "missing_only": self.missing_only,
            "reason": self.reason,
            "warning": self.warning,
        }


def detect_system_resources() -> SystemResources:
    """Detect CPU count and physical RAM without requiring psutil."""
    cpu_count = max(1, os.cpu_count() or 1)
    return SystemResources(cpu_count=cpu_count, total_memory_bytes=_detect_total_memory_bytes())


def peak_rss_bytes() -> tuple[int | None, str | None]:
    """Return current process peak RSS bytes when the platform exposes it."""
    windows_peak = _peak_rss_windows()
    if windows_peak[0] is not None:
        return windows_peak
    return _peak_rss_resource(children=False)


def child_peak_rss_bytes() -> tuple[int | None, str | None]:
    """Return peak RSS bytes for waited child processes when available."""
    return _peak_rss_resource(children=True)


def with_runtime_diagnostics(
    payload: dict[str, Any],
    timer: RuntimeTimer,
    *,
    include_child_peak: bool = False,
) -> dict[str, Any]:
    """Attach timing/memory diagnostics to a JSON payload."""
    diagnostics = timer.finish(include_child_peak=include_child_peak)
    diagnostics_dict = diagnostics.as_dict()
    payload["duration_sec"] = diagnostics.duration_sec
    payload["peak_memory_mb"] = diagnostics.peak_memory_mb
    if diagnostics.child_peak_memory_mb is not None:
        payload["child_peak_memory_mb"] = diagnostics.child_peak_memory_mb
    payload["diagnostics"] = diagnostics_dict
    return payload


def recommend_indexing_plan(
    profile: str = "auto",
    *,
    resources: SystemResources | None = None,
) -> IndexingResourcePlan:
    """Return conservative indexing defaults for the current machine.

    The recommendations intentionally keep Essentia/TensorFlow tag workers at 1.
    That step is memory-sensitive and should stay serial until subprocess batching
    is implemented.
    """
    normalized_profile = normalize_resource_profile(profile)
    resources = resources or detect_system_resources()
    effective_profile, reason = _effective_profile(normalized_profile, resources)
    cpu_count = max(1, resources.cpu_count)

    if effective_profile == "low":
        basic_workers = 1
        embedding_batch_size = 8
        tag_batch_size = 3
        basic_chunk_sec = 30.0
    elif effective_profile == "balanced":
        basic_workers = min(2, _cpu_room(cpu_count))
        embedding_batch_size = 16
        tag_batch_size = 5
        basic_chunk_sec = 60.0
    else:
        basic_workers = min(4, max(1, cpu_count // 2))
        embedding_batch_size = 32
        tag_batch_size = 10
        basic_chunk_sec = 120.0

    warning = None
    memory_gb = resources.total_memory_gb
    if memory_gb is None:
        warning = "total memory could not be detected; using conservative defaults"
    elif memory_gb < 8:
        warning = "less than 8GB RAM detected; keep indexing serial and avoid full-quality runs"

    return IndexingResourcePlan(
        requested_profile=normalized_profile,
        effective_profile=effective_profile,
        resources=resources,
        basic_workers=basic_workers,
        tag_workers=1,
        embedding_batch_size=embedding_batch_size,
        tag_batch_size=tag_batch_size,
        basic_chunk_sec=basic_chunk_sec,
        quick_basic=False,
        missing_only=True,
        reason=reason,
        warning=warning,
    )


def normalize_resource_profile(profile: str | None) -> str:
    """Validate and normalize a resource profile name."""
    normalized = (profile or "auto").strip().lower()
    if normalized not in VALID_RESOURCE_PROFILES:
        choices = ", ".join(sorted(VALID_RESOURCE_PROFILES))
        raise ValueError(f"resource profile must be one of: {choices}")
    return normalized


def resolve_worker_count(
    value: str | int,
    *,
    kind: str,
    profile: str = "auto",
    resources: SystemResources | None = None,
    minimum: int = 1,
    maximum: int = 32,
) -> int:
    """Resolve an integer worker count or the adaptive `auto` value."""
    raw = str(value).strip().lower()
    plan = recommend_indexing_plan(profile, resources=resources)
    if raw in {"", "auto"}:
        if kind == "basic":
            return plan.basic_workers
        if kind == "tags":
            return plan.tag_workers
        raise ValueError(f"unsupported worker kind: {kind}")

    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError("workers must be an integer or 'auto'") from exc
    if workers < minimum or workers > maximum:
        raise ValueError(f"workers must be between {minimum} and {maximum}")
    return workers


def resolve_embedding_batch_size(
    value: str | int,
    *,
    profile: str = "auto",
    resources: SystemResources | None = None,
    minimum: int = 1,
    maximum: int = 512,
) -> int:
    """Resolve an integer embedding batch size or the adaptive `auto` value."""
    return _resolve_int_or_auto(
        value,
        auto_value=recommend_indexing_plan(profile, resources=resources).embedding_batch_size,
        name="batch size",
        minimum=minimum,
        maximum=maximum,
    )


def resolve_tag_batch_size(
    value: str | int,
    *,
    profile: str = "auto",
    resources: SystemResources | None = None,
    minimum: int = 1,
    maximum: int = 256,
) -> int:
    """Resolve an integer tag subprocess batch size or the adaptive `auto` value."""
    return _resolve_int_or_auto(
        value,
        auto_value=recommend_indexing_plan(profile, resources=resources).tag_batch_size,
        name="tag batch size",
        minimum=minimum,
        maximum=maximum,
    )


def resolve_basic_chunk_sec(
    value: str | float,
    *,
    profile: str = "auto",
    resources: SystemResources | None = None,
    minimum: float = 1.0,
    maximum: float = 600.0,
) -> float:
    """Resolve a basic-analysis chunk duration or the adaptive `auto` value."""
    raw = str(value).strip().lower()
    if raw in {"", "auto"}:
        return recommend_indexing_plan(profile, resources=resources).basic_chunk_sec
    try:
        resolved = float(raw)
    except ValueError as exc:
        raise ValueError("chunk duration must be a number of seconds or 'auto'") from exc
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"chunk duration must be between {minimum:g} and {maximum:g} seconds")
    return resolved


def _resolve_int_or_auto(
    value: str | int,
    *,
    auto_value: int,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    raw = str(value).strip().lower()
    if raw in {"", "auto"}:
        return auto_value
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer or 'auto'") from exc
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return resolved


def _effective_profile(profile: str, resources: SystemResources) -> tuple[str, str]:
    if profile != "auto":
        return profile, f"explicit {profile} profile requested"

    memory_gb = resources.total_memory_gb
    if memory_gb is None:
        return "low", "memory unknown; using low-impact defaults"
    if memory_gb < 16:
        return "low", f"{memory_gb:.1f}GB RAM detected"
    if memory_gb < 64:
        return "balanced", f"{memory_gb:.1f}GB RAM detected"
    return "full", f"{memory_gb:.1f}GB RAM detected"


def _cpu_room(cpu_count: int) -> int:
    return max(1, cpu_count - 1)


def _detect_total_memory_bytes() -> int | None:
    return _detect_total_memory_windows() or _detect_total_memory_sysconf()


def _detect_total_memory_sysconf() -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(page_size, int) or not isinstance(pages, int):
        return None
    if page_size <= 0 or pages <= 0:
        return None
    return page_size * pages


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / 1024**2, 2)


def _peak_rss_resource(*, children: bool) -> tuple[int | None, str | None]:
    try:
        import resource
    except ImportError:
        return None, None

    usage_target = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    try:
        raw_rss = int(resource.getrusage(usage_target).ru_maxrss)
    except (OSError, ValueError):
        return None, None
    if raw_rss <= 0:
        return None, None

    # Linux reports ru_maxrss in KiB. macOS reports bytes.
    system = platform.system().lower()
    bytes_value = raw_rss if system == "darwin" else raw_rss * 1024
    source = "resource.ru_maxrss.children" if children else "resource.ru_maxrss.self"
    return bytes_value, source


def _peak_rss_windows() -> tuple[int | None, str | None]:
    if os.name != "nt":
        return None, None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    try:
        process = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            process,
            ctypes.byref(counters),
            counters.cb,
        )
    except Exception:  # pragma: no cover - defensive Windows fallback
        return None, None
    if not ok or counters.PeakWorkingSetSize <= 0:
        return None, None
    return int(counters.PeakWorkingSetSize), "windows.PeakWorkingSetSize"


def _detect_total_memory_windows() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive Windows fallback
        return None
    if not ok:
        return None
    return int(status.ullTotalPhys)
