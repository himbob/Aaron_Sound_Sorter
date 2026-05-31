# SOURCE-NAME BLINDNESS INVARIANT:
# This module may use file paths only as I/O cache keys. It must never parse
# path text, filenames, folders, ZIP member names, or producer labels as
# classification evidence.
"""Per-run audio analysis cache for repeated voter/debug passes.

The sorter may inspect the same staged audio file through multiple lanes:
full fingerprint, direct/body fingerprint, wetness diagnostics, and optional
harmonic-core recall.  This cache keeps those expensive measurements stable and
reusable inside one process without changing routing evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np

from aaron_sound_sorter.domain.models import AudioPhysics


@dataclass(frozen=True)
class FileAnalysisKey:
    """Stable I/O cache key for one physical file state."""

    resolved_path: str
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True)
class AudioAnalysisPacket:
    """One per-file measured-evidence packet for a single sorter run.

    This packet intentionally stores measured analysis only.  It does not store
    final labels, winning folders, voter decisions, training labels, or manual
    corrections.
    """

    audio_physics: AudioPhysics
    wetness_profile: dict[str, Any]
    harmonic_physics: AudioPhysics | None
    cache_stats_before: dict[str, Any]
    cache_stats_after: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a source-blind debug summary for manifests/traces."""
        return {
            "cache_policy": "per_run_memory_only",
            "persistent_disk_cache": False,
            "stores_final_decisions": False,
            "stores_training_labels": False,
            "audio_status": str(self.audio_physics.read_status),
            "audio_duration_sec": float(self.audio_physics.duration_sec),
            "direct_body_status": str(self.audio_physics.direct_body_status),
            "direct_body_duration_sec": float(self.audio_physics.direct_body_duration_sec),
            "wetness_status": str(self.wetness_profile.get("status", "missing")),
            "wetness_score": _safe_float(self.wetness_profile.get("wetness_score", 0.0)),
            "harmonic_status": (
                str(self.harmonic_physics.read_status) if self.harmonic_physics is not None else "not_computed"
            ),
            "harmonic_duration_sec": (
                float(self.harmonic_physics.duration_sec) if self.harmonic_physics is not None else 0.0
            ),
            "cache_stats_before": dict(self.cache_stats_before),
            "cache_stats_after": dict(self.cache_stats_after),
        }


class AudioAnalysisCache:
    """Small bounded per-run cache for expensive audio analysis outputs."""

    def __init__(self, max_items: int = 256) -> None:
        self.max_items = max(1, int(max_items))
        self._lock = Lock()
        self._audio_physics: dict[FileAnalysisKey, AudioPhysics] = {}
        self._wetness_profiles: dict[FileAnalysisKey, dict[str, Any]] = {}
        self._harmonic_physics: dict[FileAnalysisKey, AudioPhysics] = {}
        self._order: list[tuple[str, FileAnalysisKey]] = []
        self.hits = 0
        self.misses = 0
        self.policy_name = "per_run_memory_only"
        self.persistent_disk_cache = False

    def audio_physics(
        self,
        path: Path,
        factory: Callable[[Path], AudioPhysics],
    ) -> AudioPhysics:
        """Return cached full/direct-body physics for ``path``."""
        key = self.file_key(path)
        cached = self._get(self._audio_physics, "audio", key)
        if cached is not None:
            return clone_audio_physics(cached)
        value = factory(path)
        self._put(self._audio_physics, "audio", key, clone_audio_physics(value))
        return clone_audio_physics(value)

    def wetness_profile(
        self,
        path: Path,
        factory: Callable[[Path], dict[str, Any]],
    ) -> dict[str, Any]:
        """Return cached wetness diagnostics for ``path``."""
        key = self.file_key(path)
        cached = self._get(self._wetness_profiles, "wetness", key)
        if cached is not None:
            return deepcopy(cached)
        value = dict(factory(path) or {})
        self._put(self._wetness_profiles, "wetness", key, deepcopy(value))
        return deepcopy(value)

    def harmonic_physics(
        self,
        path: Path,
        factory: Callable[[Path], AudioPhysics],
    ) -> AudioPhysics:
        """Return cached harmonic-core physics for ``path``."""
        key = self.file_key(path)
        cached = self._get(self._harmonic_physics, "harmonic", key)
        if cached is not None:
            return clone_audio_physics(cached)
        value = factory(path)
        self._put(self._harmonic_physics, "harmonic", key, clone_audio_physics(value))
        return clone_audio_physics(value)

    def stats(self) -> dict[str, Any]:
        """Return cache counters for debug/report use."""
        with self._lock:
            audio_items = len(self._audio_physics)
            wetness_items = len(self._wetness_profiles)
            harmonic_items = len(self._harmonic_physics)
            return {
                "policy": self.policy_name,
                "persistent_disk_cache": bool(self.persistent_disk_cache),
                "hits": int(self.hits),
                "misses": int(self.misses),
                "audio_items": audio_items,
                "wetness_items": wetness_items,
                "harmonic_items": harmonic_items,
                "total_items": audio_items + wetness_items + harmonic_items,
                "max_items": int(self.max_items),
            }

    def reset(self) -> None:
        """Clear all per-run analysis state.

        SortSamplesUseCase calls this at the beginning of a new user-facing run
        so measured evidence never leaks between runs in a long-lived process.
        """
        with self._lock:
            self._audio_physics.clear()
            self._wetness_profiles.clear()
            self._harmonic_physics.clear()
            self._order.clear()
            self.hits = 0
            self.misses = 0

    @staticmethod
    def file_key(path: Path) -> FileAnalysisKey:
        """Build a file-state key without using name text as evidence."""
        resolved = Path(path).expanduser().resolve()
        try:
            stat = resolved.stat()
            return FileAnalysisKey(
                resolved_path=str(resolved),
                size_bytes=int(stat.st_size),
                modified_ns=int(stat.st_mtime_ns),
            )
        except OSError:
            return FileAnalysisKey(resolved_path=str(resolved), size_bytes=-1, modified_ns=-1)

    def _get(
        self,
        table: dict[FileAnalysisKey, Any],
        namespace: str,
        key: FileAnalysisKey,
    ) -> Any | None:
        with self._lock:
            if key in table:
                self.hits += 1
                return table[key]
            self.misses += 1
            return None

    def _put(
        self,
        table: dict[FileAnalysisKey, Any],
        namespace: str,
        key: FileAnalysisKey,
        value: Any,
    ) -> None:
        with self._lock:
            table[key] = value
            self._order.append((namespace, key))
            self._evict_locked()

    def _evict_locked(self) -> None:
        total = len(self._audio_physics) + len(self._wetness_profiles) + len(self._harmonic_physics)
        while total > self.max_items and self._order:
            namespace, key = self._order.pop(0)
            if namespace == "audio":
                self._audio_physics.pop(key, None)
            elif namespace == "wetness":
                self._wetness_profiles.pop(key, None)
            elif namespace == "harmonic":
                self._harmonic_physics.pop(key, None)
            total = len(self._audio_physics) + len(self._wetness_profiles) + len(self._harmonic_physics)


def _safe_float(value: Any) -> float:
    """Coerce scalar debug values without raising."""
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def clone_audio_physics(physics: AudioPhysics) -> AudioPhysics:
    """Copy mutable arrays while preserving attached diagnostic attributes."""
    clone = AudioPhysics(
        source_path=Path(physics.source_path),
        fingerprint=np.array(physics.fingerprint, copy=True),
        duration_sec=float(physics.duration_sec),
        read_status=str(physics.read_status),
        direct_body_fingerprint=(
            None if physics.direct_body_fingerprint is None else np.array(physics.direct_body_fingerprint, copy=True)
        ),
        direct_body_duration_sec=float(physics.direct_body_duration_sec),
        direct_body_status=str(physics.direct_body_status),
    )
    for attr in ("direct_body_profile", "wetness_profile"):
        if hasattr(physics, attr):
            object.__setattr__(clone, attr, getattr(physics, attr))
    return clone
