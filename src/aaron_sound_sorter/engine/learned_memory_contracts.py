# SOURCE-NAME BLINDNESS INVARIANT:
# Learned-memory contracts may inspect only internal memory evidence, trained
# labels, and voter branch/role metadata. They must never inspect source
# filenames, source folders, ZIP member names, or sample-pack labels.
"""Shared contracts for trainable learned-memory evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LEARNED_MEMORY_EVIDENCE_KEYS = ("learned_voter_memory", "learned_physics_memory")

VOICE_PATH_FRAGMENTS = (
    "voice",
    "vocal",
    "vocals",
    "spoken",
    "choir",
    "breath",
    "mouth",
    "human and voice",
)


@dataclass(frozen=True)
class LearnedMemoryMatch:
    """Normalized learned-memory row from shared audio evidence.

    Args:
        evidence_key: Evidence dictionary key that supplied this match.
        matched: Whether the memory lane accepted the current fingerprint.
        label: Internal trained label or folder path owned by this memory.
        top_family: Internal top family for the label, when supplied.
        branch: Voter branch name, such as ``Voice`` or ``Synth``.
        role: Learned voter role, such as ``instrument_voice_loop``.
        confidence: Match confidence normalized to the memory lane's scale.
        effective_weight: Human-training support weight.
        nearest_distance: Distance to the nearest stored fingerprint.

    Returns:
        Immutable learned-memory match value object.

    Side Effects:
        None.

    Raises:
        No intentional exceptions. Malformed numeric fields become safe
        defaults.

    Important Constraints:
        ``label`` is internal taxonomy metadata. It is not the input audio file
        name or source path.
    """

    evidence_key: str
    matched: bool
    label: str
    top_family: str
    branch: str
    role: str
    confidence: float
    effective_weight: int
    nearest_distance: float

    @classmethod
    def from_mapping(cls, evidence_key: str, raw_memory: Mapping[str, Any]) -> LearnedMemoryMatch:
        """Build a normalized learned-memory row from a loose evidence dict.

        Args:
            evidence_key: Key that contained the memory row.
            raw_memory: JSON-like memory evidence dictionary.

        Returns:
            ``LearnedMemoryMatch`` with normalized strings and safe numbers.

        Side Effects:
            None.
        """
        label = normalize_internal_path(str(raw_memory.get("label") or raw_memory.get("folder_path") or ""))
        explicit_top = str(raw_memory.get("top_family") or "").strip()
        return cls(
            evidence_key=evidence_key,
            matched=bool(raw_memory.get("matched")),
            label=label,
            top_family=explicit_top or top_family_from_path(label),
            branch=str(raw_memory.get("branch") or raw_memory.get("physics_branch") or "").strip(),
            role=str(raw_memory.get("role") or raw_memory.get("source_role") or "").strip(),
            confidence=max(
                safe_float(raw_memory.get("confidence"), 0.0),
                safe_float(raw_memory.get("match_confidence"), 0.0),
            ),
            effective_weight=safe_int(raw_memory.get("effective_weight"), 0),
            nearest_distance=safe_float(raw_memory.get("nearest_distance"), 9999.0),
        )

    @property
    def origin(self) -> str:
        """Return a stable diagnostic origin name for this memory lane."""
        if self.evidence_key == "learned_physics_memory":
            return "physics_memory"
        if self.evidence_key == "learned_voter_memory":
            return "voter_memory"
        return self.evidence_key

    @property
    def is_voice_source(self) -> bool:
        """Return whether this memory row claims a Voice-like source."""
        return is_voice_memory_target(
            label=self.label,
            top_family=self.top_family,
            branch=self.branch,
            role=self.role,
        )

    @property
    def is_non_voice_source(self) -> bool:
        """Return whether this memory row claims a non-Voice source."""
        if self.is_voice_source:
            return False
        if self.label and not self.label.lower().startswith("_to_review"):
            return True
        if self.top_family.lower() in {"drums", "instruments", "fx"}:
            return True
        return bool(self.branch or self.role)

    def matches_path(self, folder_path: str) -> bool:
        """Return whether this memory owns the exact internal folder path."""
        return bool(self.label and self.label.lower() == normalize_internal_path(folder_path).lower())


def iter_learned_memory_matches(
    source: object,
    *,
    minimum_confidence: float = 0.0,
) -> list[LearnedMemoryMatch]:
    """Return matched learned-memory rows from facts or an evidence dict.

    Args:
        source: ``SharedAudioFacts``-like object or an evidence dictionary.
        minimum_confidence: Minimum confidence required for returned matches.

    Returns:
        List of normalized learned-memory rows that are matched and confident
        enough.

    Side Effects:
        None.
    """
    evidence = evidence_mapping(source)
    if not evidence:
        return []
    matches: list[LearnedMemoryMatch] = []
    for evidence_key in LEARNED_MEMORY_EVIDENCE_KEYS:
        raw_memory = evidence.get(evidence_key)
        if not isinstance(raw_memory, Mapping):
            continue
        match = LearnedMemoryMatch.from_mapping(evidence_key, raw_memory)
        if match.matched and match.confidence >= minimum_confidence:
            matches.append(match)
    return matches


def has_non_voice_memory_match(source: object, *, minimum_confidence: float = 0.86) -> bool:
    """Return whether learned memory strongly claims a non-Voice source.

    Args:
        source: ``SharedAudioFacts``-like object or an evidence dictionary.
        minimum_confidence: Confidence threshold for a blocking memory match.

    Returns:
        ``True`` when any matched memory row claims a non-Voice source.
    """
    return any(
        match.is_non_voice_source
        for match in iter_learned_memory_matches(source, minimum_confidence=minimum_confidence)
    )


def has_dominant_non_voice_memory(
    source: object,
    *,
    minimum_confidence: float = 0.86,
    margin_over_voice: float = 0.04,
) -> bool:
    """Return whether non-Voice memory dominates competing Voice memory.

    Args:
        source: ``SharedAudioFacts``-like object or an evidence dictionary.
        minimum_confidence: Required confidence for the winning non-Voice row.
        margin_over_voice: Required confidence margin over any Voice row.

    Returns:
        ``True`` when a non-Voice learned memory row should block Voice
        fallbacks.
    """
    best_voice = 0.0
    best_non_voice = 0.0
    for match in iter_learned_memory_matches(source):
        if match.is_voice_source:
            best_voice = max(best_voice, match.confidence)
        elif match.is_non_voice_source:
            best_non_voice = max(best_non_voice, match.confidence)
    return bool(best_non_voice >= minimum_confidence and best_non_voice >= best_voice + margin_over_voice)


def has_exact_dominant_memory_for_path(
    source: object,
    folder_path: str,
    *,
    minimum_confidence: float = 0.86,
) -> bool:
    """Return whether learned memory strongly owns an exact internal label.

    Args:
        source: ``SharedAudioFacts``-like object or an evidence dictionary.
        folder_path: Internal candidate folder path.
        minimum_confidence: Required confidence for exact ownership.

    Returns:
        ``True`` when matched memory owns ``folder_path``.
    """
    normalized_path = normalize_internal_path(folder_path)
    if not normalized_path or normalized_path.lower().startswith("_to_review"):
        return False
    return any(
        match.matches_path(normalized_path)
        for match in iter_learned_memory_matches(source, minimum_confidence=minimum_confidence)
    )


def memory_conflicts_with_candidate_top_family(
    source: object,
    candidate_folder_path: str,
    *,
    minimum_confidence: float = 0.86,
) -> bool:
    """Return whether learned memory owns a different top family.

    Args:
        source: ``SharedAudioFacts``-like object or an evidence dictionary.
        candidate_folder_path: Internal candidate folder being scored.
        minimum_confidence: Confidence threshold for a blocking memory match.

    Returns:
        ``True`` when a matched memory row conflicts with the candidate's top
        family. Exact label matches never conflict.
    """
    candidate_path = normalize_internal_path(candidate_folder_path)
    candidate_top = top_family_from_path(candidate_path)
    if not candidate_top:
        return False
    for match in iter_learned_memory_matches(source, minimum_confidence=minimum_confidence):
        if match.matches_path(candidate_path):
            return False
        match_top = match.top_family.lower()
        if match_top in {"drums", "instruments", "fx"} and match_top != candidate_top.lower():
            return True
    return False


def is_voice_category_path(path: str) -> bool:
    """Return whether an internal taxonomy path belongs to Voice."""
    normalized = normalize_internal_path(path).lower()
    if normalized.startswith("instruments/voice") or normalized.startswith("fx/human and voice fx"):
        return True
    return any(fragment in normalized for fragment in VOICE_PATH_FRAGMENTS)


def is_voice_memory_target(*, label: str, top_family: str = "", branch: str = "", role: str = "") -> bool:
    """Return whether learned-memory metadata targets Voice.

    Args:
        label: Internal learned-memory label.
        top_family: Internal top family from the memory row.
        branch: Voter branch from the memory row.
        role: Learned voter role from the memory row.

    Returns:
        ``True`` for Voice and Human/Voice FX source targets.
    """
    normalized_role = str(role or "").lower()
    normalized_branch = str(branch or "").lower()
    if is_voice_category_path(label):
        return True
    normalized_top_family = str(top_family or "").lower()
    if normalized_top_family == "instruments" and normalized_branch == "voice":
        return True
    if "voice" in normalized_role or "vocal" in normalized_role or "spoken" in normalized_role:
        return True
    return bool(
        normalized_top_family == "fx"
        and (
            normalized_branch in {"formantfx", "humancreaturefx"}
            or "human_voice" in normalized_role
            or "formant" in normalized_role
        )
    )


def evidence_mapping(source: object) -> Mapping[str, Any]:
    """Return an evidence mapping from facts-like objects or dicts."""
    if isinstance(source, Mapping):
        return source
    evidence = getattr(source, "evidence", {})
    return evidence if isinstance(evidence, Mapping) else {}


def normalize_internal_path(value: str) -> str:
    """Return a normalized internal category path string."""
    return str(value or "").replace("\\", "/").strip("/")


def top_family_from_path(folder_path: str) -> str:
    """Return the top family token from an internal category path."""
    return normalize_internal_path(folder_path).split("/", 1)[0]


def safe_float(value: object, default: float = 0.0) -> float:
    """Return a finite float from loose JSON values."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return number if math.isfinite(number) else default


def safe_int(value: object, default: int = 0) -> int:
    """Return an integer from loose JSON values."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except Exception:
        return default
