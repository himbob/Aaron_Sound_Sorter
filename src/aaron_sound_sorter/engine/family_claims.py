# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Typed claim objects used before final placement arbitration.

The sorter has several evidence producers: voters, measured parent roles,
shape checks, and conflict checks.  None of those producers should decide the
final folder directly.  They report claims.  The FamilyClaimArbiter is the only
normal engine component that converts those claims into a ConsensusDecision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FamilyClaim:
    """A broad, positive family claim used by specialized claim builders."""

    family: str
    strength: float
    can_override: bool
    reason: str


@dataclass(frozen=True)
class ConsensusClaim:
    """A typed evidence object emitted by pre-arbiter logic.

    Attributes:
        family: Broad top family such as ``Drums``, ``Instruments``, ``FX``,
            ``Textures``, or ``_TO_REVIEW``.
        sub_family: Stable placement key understood by PlacementResolver.
        strength: Normalized evidence strength in the ``0.0`` to ``1.0`` range.
        can_override: True when the claim is allowed to beat the raw candidate.
        reason: Human-readable diagnostic string.
        source: The subsystem that produced this claim.
        raw_candidate_score: Combined rank score of the real candidate that
            supports this claim.  ``None`` means the claim is inferred from
            measured facts rather than backed by a concrete voter candidate.
        label: Internal label to report if the claim wins.
        folder_path: Optional already-known target folder for real candidates.
            PlacementResolver may use this for raw/leaf candidates.
        brain_rank: Brain voter rank, when available.
        physics_rank: Physics voter rank, when available.
        shared_candidates: Shared candidate rows preserved for manifest output.
        is_review: True when the claim asks the arbiter to review instead of
            auto-place.
        is_real_candidate: True only when the claim came from an actual voter
            candidate row rather than a synthetic broad bucket.
    """

    family: str
    sub_family: str
    strength: float
    can_override: bool
    reason: str
    source: str
    raw_candidate_score: float | None = None
    label: str = ""
    folder_path: str = ""
    brain_rank: int | None = None
    physics_rank: int | None = None
    shared_winner: str = ""
    shared_candidates: list[dict[str, Any]] = field(default_factory=list)
    is_review: bool = False
    is_real_candidate: bool = False

    @property
    def final_label(self) -> str:
        """Return the label to write into the final decision."""
        return self.label or self.folder_path or self.sub_family or self.family

    @property
    def final_top(self) -> str:
        """Return the final top family for compatibility with older helpers."""
        return self.family

    @property
    def combined_rank_score(self) -> float | None:
        """Compatibility alias for candidate-based helpers."""
        return self.raw_candidate_score


def clamp_strength(value: float) -> float:
    """Clamp a raw strength value to the claim strength interval."""
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, numeric))


def top_family_from_path(folder_path: str) -> str:
    """Return the first folder component from an internal folder path."""
    clean_path = str(folder_path or "_TO_REVIEW/Measured Role Conflict").strip("/")
    return clean_path.split("/", 1)[0] if clean_path else "_TO_REVIEW"


def sub_family_from_path(folder_path: str) -> str:
    """Return a stable resolver key from an internal folder path."""
    clean_path = str(folder_path or "_TO_REVIEW/Measured Role Conflict").strip("/")
    normalized = clean_path.lower()
    if normalized.startswith("_to_review"):
        return "Measured Role Conflict"
    if "drum loops" in normalized:
        return "Drum Loops"
    if "generic kick" in normalized or "kick drums" in normalized:
        return "Kick One Shot"
    if "generic percussion" in normalized or "/percussion/" in normalized:
        return "Percussion One Shot"
    if "bass loops" in normalized:
        return "Bass Loops"
    if "instrument loops" in normalized or "mixed musical loops" in normalized:
        return "Instrument Loops"
    if (
        "brass and woodwinds" in normalized
        or "woodwinds" in normalized
        or "saxophone" in normalized
        or "/sax/" in normalized
    ):
        return "Brass Woodwinds"
    if "human and voice" in normalized or "voice" in normalized or "vocal" in normalized:
        return "Human and Voice FX" if normalized.startswith("fx/") else "Voice"
    if "risers" in normalized or "builds" in normalized:
        return "Risers and Builds"
    if "drops" in normalized or "downlifters" in normalized:
        return "Drops and Downlifters"
    if normalized.startswith("textures/"):
        return "Hybrid Textures"
    if normalized.startswith("fx/"):
        return "Hybrid Designed FX"
    return clean_path.split("/", 1)[1] if "/" in clean_path else clean_path


def claim_from_candidate_row(
    *,
    row: dict[str, Any],
    source: str,
    reason: str,
    shared: list[dict[str, Any]],
    can_override: bool,
    strength: float | None = None,
) -> ConsensusClaim:
    """Build a claim from a real shared candidate row."""
    score = _float_or_default(row.get("combined_rank_score"), 9999.0)
    inferred_strength = max(0.0, min(1.0, (16.0 - score) / 16.0))
    folder_path = str(row.get("folder_path") or row.get("label") or "").strip("/")
    label = str(row.get("label") or folder_path)
    family = str(row.get("top_family") or top_family_from_path(folder_path))
    return ConsensusClaim(
        family=family,
        sub_family=sub_family_from_path(folder_path),
        strength=clamp_strength(inferred_strength if strength is None else strength),
        can_override=can_override,
        reason=reason,
        source=source,
        raw_candidate_score=score,
        label=label,
        folder_path=folder_path,
        brain_rank=_int_or_none(row.get("brain_rank")),
        physics_rank=_int_or_none(row.get("physics_rank")),
        shared_winner=label,
        shared_candidates=shared,
        is_review=False,
        is_real_candidate=True,
    )


def claim_from_folder_path(
    *,
    folder_path: str,
    source: str,
    reason: str,
    shared: list[dict[str, Any]],
    raw_candidate_score: float | None,
    brain_rank: int | None,
    physics_rank: int | None,
    shared_winner: str,
    can_override: bool,
    strength: float,
    is_real_candidate: bool = False,
) -> ConsensusClaim:
    """Build a claim from a broad internal folder path."""
    clean_path = str(folder_path or "_TO_REVIEW/Measured Role Conflict").strip("/")
    family = top_family_from_path(clean_path)
    return ConsensusClaim(
        family=family,
        sub_family=sub_family_from_path(clean_path),
        strength=clamp_strength(strength),
        can_override=can_override,
        reason=reason,
        source=source,
        raw_candidate_score=raw_candidate_score,
        label=clean_path,
        folder_path=clean_path,
        brain_rank=brain_rank,
        physics_rank=physics_rank,
        shared_winner=shared_winner,
        shared_candidates=shared,
        is_review=family == "_TO_REVIEW",
        is_real_candidate=is_real_candidate,
    )


def claim_from_category_guess(
    *,
    guess: Any,
    folder_path: str,
    source: str,
    reason: str,
    shared: list[dict[str, Any]],
    raw_candidate_score: float | None,
    can_override: bool,
    strength: float,
) -> ConsensusClaim:
    """Build a claim from one real voter candidate.

    This is used when one voter has a clear profile leaf and another voter or
    measured-role layer supports the broad family, but the exact leaf was not a
    strict shared candidate.  The candidate label is internal model output, not
    input-file text.
    """
    clean_path = str(folder_path or getattr(guess, "folder_path", "") or getattr(guess, "label", "")).strip("/")
    family = str(getattr(guess, "top_family", "") or top_family_from_path(clean_path))
    label = str(getattr(guess, "label", "") or clean_path)
    return ConsensusClaim(
        family=family,
        sub_family=sub_family_from_path(clean_path),
        strength=clamp_strength(strength),
        can_override=can_override,
        reason=reason,
        source=source,
        raw_candidate_score=raw_candidate_score,
        label=label,
        folder_path=clean_path,
        brain_rank=_int_or_none(getattr(guess, "rank", None)),
        physics_rank=None,
        shared_winner=label,
        shared_candidates=shared,
        is_review=False,
        is_real_candidate=True,
    )


def review_claim(
    *,
    label: str,
    reason: str,
    source: str,
    shared: list[dict[str, Any]] | None = None,
    winner: dict[str, Any] | ConsensusClaim | None = None,
    strength: float = 1.0,
) -> ConsensusClaim:
    """Return a review claim without creating a final decision."""
    winner_label = ""
    brain_rank: int | None = None
    physics_rank: int | None = None
    combined_score: float | None = None
    if isinstance(winner, ConsensusClaim):
        winner_label = winner.shared_winner or winner.final_label
        brain_rank = winner.brain_rank
        physics_rank = winner.physics_rank
        combined_score = winner.raw_candidate_score
    elif isinstance(winner, dict):
        winner_label = str(winner.get("label", ""))
        brain_rank = _int_or_none(winner.get("brain_rank"))
        physics_rank = _int_or_none(winner.get("physics_rank"))
        combined_score = _float_or_none(winner.get("combined_rank_score"))
    return ConsensusClaim(
        family="_TO_REVIEW",
        sub_family="Measured Role Conflict",
        strength=clamp_strength(strength),
        can_override=True,
        reason=reason,
        source=source,
        raw_candidate_score=combined_score,
        label=label,
        folder_path=label,
        brain_rank=brain_rank,
        physics_rank=physics_rank,
        shared_winner=winner_label,
        shared_candidates=shared or [],
        is_review=True,
        is_real_candidate=False,
    )


def build_human_voice_claim(
    *,
    has_voice_candidate: bool,
    voice_candidate_score: float | None,
    raw_score: float,
    competitor_score: float | None,
    voice_role_strength: float,
    candidate_role_strength: float,
    stable_instrument_claim_strength: float = 0.0,
) -> FamilyClaim:
    """Return a positive Human/Voice claim only when evidence beats rivals."""
    if not has_voice_candidate or voice_candidate_score is None:
        return FamilyClaim("HumanVoice", 0.0, False, "no Human/Voice candidate")

    role_supported = candidate_role_strength >= 0.75 or voice_role_strength >= 0.82
    if stable_instrument_claim_strength >= 0.70 and not role_supported:
        return FamilyClaim(
            "HumanVoice",
            max(0.0, min(1.0, voice_role_strength)),
            False,
            "stable instrument claim blocked weak Human/Voice claim",
        )

    if role_supported:
        beats_competitor = competitor_score is None or voice_candidate_score <= competitor_score + 3.0
    else:
        beats_competitor = competitor_score is None or voice_candidate_score <= competitor_score - 4.0

    beats_raw = raw_score >= 999.0 or voice_candidate_score <= raw_score + 6.0
    clear_score_win = voice_candidate_score <= raw_score - 6.0
    can_override = bool(beats_competitor and beats_raw and (role_supported or clear_score_win))

    rank_strength = 0.0
    if raw_score < 999.0:
        rank_strength = max(0.0, min(1.0, (raw_score - voice_candidate_score + 8.0) / 16.0))
    measured_strength = max(0.0, min(1.0, max(voice_role_strength, candidate_role_strength)))
    strength = max(rank_strength, measured_strength if role_supported else measured_strength * 0.5)
    reason = "positive Human/Voice claim" if can_override else "Human/Voice claim did not beat rival family"
    return FamilyClaim("HumanVoice", strength, can_override, reason)


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None
