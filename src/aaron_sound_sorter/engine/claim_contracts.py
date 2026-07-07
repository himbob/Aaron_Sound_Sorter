# SOURCE-NAME BLINDNESS INVARIANT:
# Claim contracts must never use producer filenames, source folder names, ZIP
# member names, path tokens, or sample-pack labels as classification evidence.
# Internal category labels are allowed only as typed decision metadata.
"""Shared source-blind contracts for synthetic decision claims."""

from __future__ import annotations

from dataclasses import dataclass

from aaron_sound_sorter.engine.decision_helpers import _norm_path
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


@dataclass(frozen=True)
class ClaimContract:
    """Describe the authority boundary for one arbitration claim.

    Args:
        claim_id: Stable display identifier for debug output.
        claim_type: Broad authority kind such as role, structure, or source identity.
        scope: What the claim is allowed to affect.
        authority: Human-readable authority summary.
        family: Top-level family carried by the claim.
        role: Broad measured role represented by the claim.
        structure: Structural placement such as loop, one shot, or review.
        required_evidence: Evidence types expected before this claim is trusted.
        veto_evidence: Evidence types that should block this claim.
        allowed_transitions: Family or bucket transitions this contract may make.
        source_features_used: Source-blind feature groups used by this contract.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        The contract is diagnostic metadata only.  It must not inspect producer
        filenames or source folders.
    """

    claim_id: str
    claim_type: str
    scope: str
    authority: str
    family: str
    role: str
    structure: str
    required_evidence: tuple[str, ...]
    veto_evidence: tuple[str, ...]
    allowed_transitions: tuple[str, ...]
    source_features_used: tuple[str, ...]

    def debug_line(self) -> str:
        """Return a compact single-line representation for claim debug logs.

        Args:
            None.

        Returns:
            A source-blind debug string describing the claim contract.

        Side Effects:
            None.

        Raises:
            No intentional exceptions.

        Important Constraints:
            The line includes internal labels only as claim metadata.
        """
        required = ",".join(self.required_evidence) or "none"
        veto = ",".join(self.veto_evidence) or "none"
        transitions = ",".join(self.allowed_transitions) or "none"
        return (
            "CONTRACT "
            f"id={self.claim_id} "
            f"type={self.claim_type} "
            f"scope={self.scope} "
            f"authority={self.authority} "
            f"role={self.role} "
            f"structure={self.structure} "
            f"required={required} "
            f"veto={veto} "
            f"transitions={transitions}"
        )


def contract_from_claim(claim: ConsensusClaim, *, index: int) -> ClaimContract:
    """Build source-blind authority metadata for a claim.

    Args:
        claim: Arbitration claim to describe.
        index: Stable one-based order in the trace.

    Returns:
        A diagnostic contract describing the claim's intended authority.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This maps internal claim metadata to contract metadata.  It must not read
        audio filenames, source folders, or ZIP member names.
    """
    claim_id = f"{index:03d}:{claim.source}:{claim.folder_path or claim.label}"
    normalized_path = _norm_path(claim.folder_path or claim.label)
    source = str(claim.source or "").lower()
    family = str(claim.family or "")
    role = _role_from_claim(family, normalized_path)
    structure = _structure_from_path(normalized_path)

    if family == "_TO_REVIEW" or bool(claim.is_review):
        return ClaimContract(
            claim_id=claim_id,
            claim_type="safety_review",
            scope="safety_review_only",
            authority="may_downgrade_to_review",
            family=family,
            role="review",
            structure="review",
            required_evidence=("conflict_or_low_confidence",),
            veto_evidence=(),
            allowed_transitions=("review",),
            source_features_used=("claim_conflict",),
        )

    if "shape" in source:
        return ClaimContract(
            claim_id=claim_id,
            claim_type="structure",
            scope="structure_only",
            authority="may_block_or_broaden_structure",
            family=family,
            role=role,
            structure=structure,
            required_evidence=("shape_vote", "measured_role"),
            veto_evidence=("source_identity_conflict",),
            allowed_transitions=(f"same_family:{family}", f"broad_bucket:{family}"),
            source_features_used=("shape_vote", "measured_role"),
        )

    if _is_broad_role_claim(claim, normalized_path, source):
        return ClaimContract(
            claim_id=claim_id,
            claim_type="role",
            scope="role_and_broad_bucket",
            authority="may_choose_broad_bucket_when_identity_abstains",
            family=family,
            role=role,
            structure=structure,
            required_evidence=("measured_role", "shape_vote"),
            veto_evidence=("stronger_source_identity",),
            allowed_transitions=(f"same_family:{family}", f"broad_bucket:{family}"),
            source_features_used=("measured_role", "shape_vote"),
        )

    return ClaimContract(
        claim_id=claim_id,
        claim_type="source_identity",
        scope="source_identity_when_not_vetoed",
        authority="may_choose_leaf_if_physics_and_shape_allow",
        family=family,
        role=role,
        structure=structure,
        required_evidence=("brain_topk", "physics_topk", "no_shape_veto"),
        veto_evidence=("shape_structure_conflict",),
        allowed_transitions=(f"same_family:{family}",),
        source_features_used=("brain_topk", "physics_topk", "shape_vote"),
    )


def contracts_from_claims(claims: list[ConsensusClaim]) -> list[ClaimContract]:
    """Return ordered claim contracts for debug reporting.

    Args:
        claims: Claims in arbitration order.

    Returns:
        Claim contracts with stable one-based identifiers.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This is a diagnostic adapter and does not change routing behavior.
    """
    return [contract_from_claim(claim, index=index) for index, claim in enumerate(claims, start=1)]


def is_rank_one_concrete_non_sax_instrument_consensus(claim: ConsensusClaim) -> bool:
    """Return whether a concrete non-sax Instrument consensus should stand.

    Args:
        claim: Raw shared voter claim before synthetic measured claims are added.

    Returns:
        True when Brain and Physics already agree at rank one on a concrete
        non-sax Instrument label.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This checks only internal voter/category metadata.  It must never inspect
        producer filenames, source folders, ZIP member names, or sample-pack labels.
    """
    if _claim_is_rank_one_concrete_non_sax_instrument_consensus(claim):
        return True
    return any(_shared_row_is_rank_one_concrete_non_sax_instrument_consensus(row) for row in claim.shared_candidates)


def is_rank_one_concrete_non_sax_instrument_row(row: object) -> bool:
    """Return whether a shared candidate row is protected concrete Instrument consensus.

    Args:
        row: Shared candidate metadata from Brain/Physics overlap.

    Returns:
        True when the row is an internal concrete non-sax Instrument label ranked
        first by both Brain and Physics.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This checks only internal category and voter-rank metadata.  It must never
        inspect producer filenames, source folders, ZIP member names, or sample-pack
        labels.
    """
    return _shared_row_is_rank_one_concrete_non_sax_instrument_consensus(row)


def _claim_is_rank_one_concrete_non_sax_instrument_consensus(claim: ConsensusClaim) -> bool:
    """Return True when the claim itself is a concrete non-sax consensus."""
    if claim.family != "Instruments" or not claim.is_real_candidate:
        return False
    if claim.source != "strong_consensus":
        return False
    return _is_rank_one_concrete_non_sax_path(
        claim.folder_path or claim.label,
        brain_rank=claim.brain_rank,
        physics_rank=claim.physics_rank,
        raw_candidate_score=claim.raw_candidate_score,
        allow_teacher_support=False,
    )


def _shared_row_is_rank_one_concrete_non_sax_instrument_consensus(row: object) -> bool:
    """Return True when a shared row carries a protected concrete consensus."""
    if not isinstance(row, dict):
        return False
    folder_path = str(row.get("folder_path") or row.get("label") or "")
    family = str(row.get("top_family") or folder_path.split("/", 1)[0])
    if family != "Instruments":
        return False
    return _is_rank_one_concrete_non_sax_path(
        folder_path,
        brain_rank=_safe_int(row.get("brain_rank")),
        physics_rank=_safe_int(row.get("physics_rank")),
        raw_candidate_score=_safe_float(row.get("combined_rank_score")),
        allow_teacher_support=_shared_row_has_human_teacher_support(row),
    )


def _is_rank_one_concrete_non_sax_path(
    folder_path: str,
    *,
    brain_rank: int | None,
    physics_rank: int | None,
    raw_candidate_score: float | None,
    allow_teacher_support: bool,
) -> bool:
    """Return True for protected rank-one concrete non-sax Instrument paths."""
    if brain_rank != 1 or physics_rank != 1:
        return False
    if raw_candidate_score is None:
        return False
    if raw_candidate_score > 2.0 and not allow_teacher_support:
        return False
    path = _norm_path(folder_path)
    if not path or "sax" in path or "saxophone" in path:
        return False
    broad_parent_fragments = (
        "instrument loops",
        "mixed musical loops",
        "brass and woodwinds",
    )
    return not any(fragment in path for fragment in broad_parent_fragments)


def _shared_row_has_human_teacher_support(row: dict[str, object]) -> bool:
    """Return True when a row is backed by GUI correction recall evidence."""
    evidence = row.get("brain_evidence")
    if not isinstance(evidence, dict):
        return False
    return bool(
        evidence.get("human_override_matched")
        or evidence.get("human_override_exact_audio_match")
        or evidence.get("human_override_generalized_audio_match")
    )


def _safe_int(value: object) -> int | None:
    """Return an integer from loose shared-row metadata."""
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    """Return a finite float from loose shared-row metadata."""
    try:
        number = float(value)
    except Exception:
        return None
    return None if number != number else number


def _is_broad_role_claim(claim: ConsensusClaim, normalized_path: str, source: str) -> bool:
    """Return whether a claim is broad role evidence rather than leaf identity."""
    if "role" in source:
        return True
    sub_family = str(claim.sub_family or "").lower()
    broad_fragments = (
        "instrument loops",
        "mixed musical loops",
        "drum loops",
        "percussion",
    )
    return any(fragment in normalized_path or fragment in sub_family for fragment in broad_fragments)


def _role_from_claim(family: str, normalized_path: str) -> str:
    """Return a broad role name from internal claim metadata."""
    if family == "_TO_REVIEW":
        return "review"
    if "drum loop" in normalized_path:
        return "drum_loop"
    if "one shot" in normalized_path or "one shots" in normalized_path:
        return "one_shot"
    if "loop" in normalized_path:
        return "pitched_music_loop" if family == "Instruments" else "loop"
    if family == "Drums":
        return "percussive_source"
    if family == "FX":
        return "fx_role"
    return "source_identity"


def _structure_from_path(normalized_path: str) -> str:
    """Return a broad placement structure from an internal folder path."""
    if normalized_path.startswith("_to_review"):
        return "review"
    if "one shot" in normalized_path or "one shots" in normalized_path:
        return "one_shot"
    if "loop" in normalized_path:
        return "loop"
    if "long fx" in normalized_path:
        return "long_fx"
    return "unknown"
