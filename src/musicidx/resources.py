"""Adaptive resource recommendations for MusicIdx indexing."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any

VALID_RESOURCE_PROFILES = {"auto", "low", "balanced", "full"}


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
            "quick_basic": self.quick_basic,
            "missing_only": self.missing_only,
            "reason": self.reason,
            "warning": self.warning,
        }


def detect_system_resources() -> SystemResources:
    """Detect CPU count and physical RAM without requiring psutil."""
    cpu_count = max(1, os.cpu_count() or 1)
    return SystemResources(cpu_count=cpu_count, total_memory_bytes=_detect_total_memory_bytes())


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
    elif effective_profile == "balanced":
        basic_workers = min(2, _cpu_room(cpu_count))
        embedding_batch_size = 16
        tag_batch_size = 5
    else:
        basic_workers = min(4, max(1, cpu_count // 2))
        embedding_batch_size = 32
        tag_batch_size = 10

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
        quick_basic=True,
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
