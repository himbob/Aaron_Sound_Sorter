# SOURCE-NAME BLINDNESS INVARIANT:
# This module may use file paths only as I/O cache keys. It must never parse
# path text, filenames, folders, ZIP member names, or producer labels as
# classification evidence.
"""Audio analysis cache for repeated voter/debug passes.

The sorter may inspect the same staged audio file through multiple lanes:
full fingerprint, direct/body fingerprint, wetness diagnostics, and optional
harmonic-core recall.  This cache keeps those expensive measurements stable and
reusable inside one process, and can optionally persist measured analysis to
disk between runs, without changing routing evidence.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np

from aaron_sound_sorter.domain.models import AudioPhysics

ANALYSIS_CACHE_SCHEMA_VERSION = "audio_analysis_cache_v1_20260721"


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
        cache_policy = str(self.cache_stats_after.get("policy", "per_run_memory_only"))
        persistent_enabled = bool(self.cache_stats_after.get("persistent_disk_cache", False))
        return {
            "cache_policy": cache_policy,
            "persistent_disk_cache": persistent_enabled,
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
    """Small bounded cache for expensive measured audio analysis outputs."""

    def __init__(self, max_items: int = 256, persistent_dir: Path | None = None) -> None:
        self.max_items = max(1, int(max_items))
        self._lock = Lock()
        self._audio_physics: dict[FileAnalysisKey, AudioPhysics] = {}
        self._wetness_profiles: dict[FileAnalysisKey, dict[str, Any]] = {}
        self._harmonic_physics: dict[FileAnalysisKey, AudioPhysics] = {}
        self._order: list[tuple[str, FileAnalysisKey]] = []
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.disk_misses = 0
        self.disk_writes = 0
        self.disk_errors = 0
        self.policy_name = "memory_only"
        self.persistent_disk_cache = False
        self.persistent_dir: Path | None = None
        self.configure_persistent(persistent_dir)

    def configure_persistent(self, persistent_dir: Path | None) -> None:
        """Enable or disable persistent measured-analysis cache storage.

        Args:
            persistent_dir: Directory for source-blind cache files. ``None``
                disables disk persistence.

        Side Effects:
            Creates the cache directory when persistence is enabled.

        Important Constraints:
            The disk cache stores measured feature packets only. It never stores
            final folders, voter winners, training labels, or manual
            corrections.
        """
        with self._lock:
            if persistent_dir is None:
                self.persistent_dir = None
                self.persistent_disk_cache = False
                self.policy_name = "memory_only"
                return
            cache_dir = Path(persistent_dir).expanduser().resolve()
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.persistent_dir = None
                self.persistent_disk_cache = False
                self.policy_name = "memory_only"
                self.disk_errors += 1
                return
            self.persistent_dir = cache_dir
            self.persistent_disk_cache = True
            self.policy_name = "memory_plus_persistent_disk"

    def audio_physics(
        self,
        path: Path,
        factory: Callable[[Path], AudioPhysics],
    ) -> AudioPhysics:
        """Return cached full/direct-body physics for ``path``."""
        key = self.file_key(path)
        cached = self._get_from_memory(self._audio_physics, key)
        if cached is not None:
            return clone_audio_physics(cached)
        cached = self._load_audio_physics_from_disk("audio", key)
        if cached is not None:
            self._put(self._audio_physics, "audio", key, clone_audio_physics(cached), write_disk=False)
            return clone_audio_physics(cached)
        self._record_miss()
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
        cached = self._get_from_memory(self._wetness_profiles, key)
        if cached is not None:
            return deepcopy(cached)
        cached = self._load_dict_from_disk("wetness", key)
        if cached is not None:
            self._put(self._wetness_profiles, "wetness", key, deepcopy(cached), write_disk=False)
            return deepcopy(cached)
        self._record_miss()
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
        cached = self._get_from_memory(self._harmonic_physics, key)
        if cached is not None:
            return clone_audio_physics(cached)
        cached = self._load_audio_physics_from_disk("harmonic", key)
        if cached is not None:
            self._put(self._harmonic_physics, "harmonic", key, clone_audio_physics(cached), write_disk=False)
            return clone_audio_physics(cached)
        self._record_miss()
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
                "disk_hits": int(self.disk_hits),
                "disk_misses": int(self.disk_misses),
                "disk_writes": int(self.disk_writes),
                "disk_errors": int(self.disk_errors),
                "audio_items": audio_items,
                "wetness_items": wetness_items,
                "harmonic_items": harmonic_items,
                "total_items": audio_items + wetness_items + harmonic_items,
                "max_items": int(self.max_items),
                "persistent_dir": str(self.persistent_dir) if self.persistent_dir is not None else "",
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
            self.disk_hits = 0
            self.disk_misses = 0
            self.disk_writes = 0
            self.disk_errors = 0

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

    def _get_from_memory(self, table: dict[FileAnalysisKey, Any], key: FileAnalysisKey) -> Any | None:
        with self._lock:
            if key in table:
                self.hits += 1
                return table[key]
            return None

    def _put(
        self,
        table: dict[FileAnalysisKey, Any],
        namespace: str,
        key: FileAnalysisKey,
        value: Any,
        *,
        write_disk: bool = True,
    ) -> None:
        with self._lock:
            table[key] = value
            self._order.append((namespace, key))
            self._evict_locked()
        if write_disk:
            self._save_to_disk(namespace, key, value)

    def _record_miss(self) -> None:
        with self._lock:
            self.misses += 1

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

    def _cache_path(self, namespace: str, key: FileAnalysisKey) -> Path | None:
        cache_dir = self.persistent_dir
        if cache_dir is None:
            return None
        digest_input = json.dumps(
            {
                "version": ANALYSIS_CACHE_SCHEMA_VERSION,
                "namespace": namespace,
                "resolved_path": key.resolved_path,
                "size_bytes": key.size_bytes,
                "modified_ns": key.modified_ns,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        return cache_dir / namespace / f"{digest}.json"

    def _load_audio_physics_from_disk(self, namespace: str, key: FileAnalysisKey) -> AudioPhysics | None:
        record = self._read_persistent_record(namespace, key)
        if record is None:
            return None
        return audio_physics_from_cache_record(record, key)

    def _load_dict_from_disk(self, namespace: str, key: FileAnalysisKey) -> dict[str, Any] | None:
        record = self._read_persistent_record(namespace, key)
        payload = record.get("payload") if isinstance(record, dict) else None
        return payload if isinstance(payload, dict) else None

    def _read_persistent_record(self, namespace: str, key: FileAnalysisKey) -> dict[str, Any] | None:
        path = self._cache_path(namespace, key)
        if path is None:
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if not self._persistent_record_matches(record, namespace):
                self._record_disk_error()
                return None
        except FileNotFoundError:
            self._record_disk_miss()
            return None
        except Exception:
            self._record_disk_error()
            return None
        self._record_disk_hit()
        return record

    def _save_to_disk(self, namespace: str, key: FileAnalysisKey, value: Any) -> None:
        path = self._cache_path(namespace, key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = self._cache_record(namespace, value)
            temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            temp_path.replace(path)
        except Exception:
            self._record_disk_error()
            return
        with self._lock:
            self.disk_writes += 1

    @staticmethod
    def _cache_record(namespace: str, value: Any) -> dict[str, Any]:
        if isinstance(value, AudioPhysics):
            payload = audio_physics_to_cache_record(value)
        elif isinstance(value, dict):
            payload = json_ready(value)
        else:
            payload = {}
        return {
            "version": ANALYSIS_CACHE_SCHEMA_VERSION,
            "namespace": namespace,
            "stores_final_decisions": False,
            "stores_training_labels": False,
            "payload": payload,
        }

    @staticmethod
    def _persistent_record_matches(record: Any, namespace: str) -> bool:
        return bool(
            isinstance(record, dict)
            and record.get("version") == ANALYSIS_CACHE_SCHEMA_VERSION
            and record.get("namespace") == namespace
        )

    def _record_disk_hit(self) -> None:
        with self._lock:
            self.hits += 1
            self.disk_hits += 1

    def _record_disk_miss(self) -> None:
        with self._lock:
            self.disk_misses += 1

    def _record_disk_error(self) -> None:
        with self._lock:
            self.disk_errors += 1


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
        third_party_feature_profile=(
            None if physics.third_party_feature_profile is None else deepcopy(physics.third_party_feature_profile)
        ),
    )
    for attr in ("direct_body_profile", "wetness_profile", "third_party_feature_profile"):
        if hasattr(physics, attr):
            object.__setattr__(clone, attr, getattr(physics, attr))
    return clone


def audio_physics_to_cache_record(physics: AudioPhysics) -> dict[str, Any]:
    """Convert measured audio physics to JSON-safe persistent cache payload."""
    payload: dict[str, Any] = {
        "fingerprint": array_to_json_list(physics.fingerprint),
        "duration_sec": float(physics.duration_sec),
        "read_status": str(physics.read_status),
        "direct_body_fingerprint": (
            None if physics.direct_body_fingerprint is None else array_to_json_list(physics.direct_body_fingerprint)
        ),
        "direct_body_duration_sec": float(physics.direct_body_duration_sec),
        "direct_body_status": str(physics.direct_body_status),
        "third_party_feature_profile": json_ready(physics.third_party_feature_profile or {}),
    }
    for attr in ("direct_body_profile", "wetness_profile"):
        if hasattr(physics, attr):
            payload[attr] = json_ready(getattr(physics, attr))
    return payload


def audio_physics_from_cache_record(record: dict[str, Any], key: FileAnalysisKey) -> AudioPhysics | None:
    """Rebuild measured audio physics from a JSON cache record."""
    payload = record.get("payload") if isinstance(record, dict) else None
    if not isinstance(payload, dict):
        return None
    try:
        direct_body = payload.get("direct_body_fingerprint")
        physics = AudioPhysics(
            source_path=Path(key.resolved_path),
            fingerprint=array_from_json_list(payload.get("fingerprint")),
            duration_sec=float(payload.get("duration_sec", 0.0) or 0.0),
            read_status=str(payload.get("read_status", "cached")),
            direct_body_fingerprint=None if direct_body is None else array_from_json_list(direct_body),
            direct_body_duration_sec=float(payload.get("direct_body_duration_sec", 0.0) or 0.0),
            direct_body_status=str(payload.get("direct_body_status", "cached")),
            third_party_feature_profile=dict(payload.get("third_party_feature_profile") or {}),
        )
    except Exception:
        return None
    for attr in ("direct_body_profile", "wetness_profile"):
        value = payload.get(attr)
        if isinstance(value, dict):
            object.__setattr__(physics, attr, deepcopy(value))
    return physics


def array_to_json_list(value: Any) -> list[float]:
    """Return a compact JSON-safe float list for a measured vector."""
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(number) for number in np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)]


def array_from_json_list(value: Any) -> np.ndarray:
    """Return a float32 measured vector from a JSON cache payload."""
    if not isinstance(value, list):
        return np.zeros(0, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def json_ready(value: Any) -> Any:
    """Convert measured diagnostic values to plain JSON-safe containers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if number == number and number not in {float("inf"), float("-inf")} else 0.0
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return array_to_json_list(value)
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(child) for child in value]
    return str(value)
