# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Thin claim-contract adapter for existing arbiter claims.

This module is intentionally an adapter, not a new classifier.  It wraps the
current ``ConsensusClaim`` objects in explicit authority metadata so the next
architecture pass can reason about structure, top family, role, identity, and
review claims without guessing from scattered source strings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from aaron_sound_sorter.engine.family_claims import ConsensusClaim


@dataclass(frozen=True)
class ClaimContract:
    """Explicit authority contract derived from an existing ConsensusClaim.

    The fields are diagnostic and arbiter-facing.  They do not inspect source
    filenames or producer path text.  They only use claim metadata already
    produced by voters, measured-role code, and the current arbiter pipeline.
    """

    claim_id: str
    voter_name: str
    claim_type: str
    public_label: str
    family: str
    role: str
    structure: str
    score: float | None
    confidence: float
    scope: str
    authority: str
    required_evidence: tuple[str, ...] = field(default_factory=tuple)
    veto_evidence: tuple[str, ...] = field(default_factory=tuple)
    allowed_transitions: tuple[str, ...] = field(default_factory=tuple)
    source_features_used: tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/debug friendly dictionary."""
        return asdict(self)

    def debug_line(self) -> str:
        """Return a compact stable line for claim debug traces."""
        return (
            "CONTRACT "
            f"id={self.claim_id} "
            f"type={self.claim_type} "
            f"scope={self.scope} "
            f"authority={self.authority} "
            f"family={self.family} "
            f"role={self.role} "
            f"structure={self.structure} "
            f"confidence={self.confidence:.3f} "
            f"source={self.voter_name} "
            f"label={self.public_label}"
        )


def contract_from_claim(claim: ConsensusClaim, *, index: int = 0) -> ClaimContract:
    """Build a ClaimContract from the current ConsensusClaim shape.

    This is the safe first step toward the report's claim-contract design: it
    makes existing claim authority visible without rewriting BrainVoter,
    PhysicsVoter, ShapeVoter, or the consensus runner yet.
    """
    source = str(claim.source or "unknown")
    public_label = str(claim.folder_path or claim.label or claim.sub_family or claim.family)
    claim_type = infer_claim_type(claim)
    role = infer_role(public_label, claim.sub_family)
    structure = infer_structure(public_label, claim.sub_family, source)
    scope = infer_scope(claim_type)
    authority = infer_authority(claim, claim_type)
    return ClaimContract(
        claim_id=f"{index:03d}:{source}:{claim.family}:{claim.sub_family}",
        voter_name=source,
        claim_type=claim_type,
        public_label=public_label,
        family=str(claim.family or "unknown"),
        role=role,
        structure=structure,
        score=claim.raw_candidate_score,
        confidence=float(claim.strength),
        scope=scope,
        authority=authority,
        required_evidence=infer_required_evidence(claim, claim_type),
        veto_evidence=infer_veto_evidence(claim, claim_type),
        allowed_transitions=infer_allowed_transitions(claim, claim_type),
        source_features_used=infer_source_features_used(source, claim_type),
        explanation=str(claim.reason or ""),
    )


def contracts_from_claims(claims: list[ConsensusClaim]) -> list[ClaimContract]:
    """Return explicit contracts for a claim list in stable order."""
    return [contract_from_claim(claim, index=index) for index, claim in enumerate(claims, start=1)]


def infer_claim_type(claim: ConsensusClaim) -> str:
    """Infer the safest broad claim type from existing claim metadata."""
    source = str(claim.source or "").lower()
    path = str(claim.folder_path or claim.label or claim.sub_family or "").lower()
    if claim.is_review or claim.family == "_TO_REVIEW":
        return "safety_review"
    if "shape" in source:
        return "structure"
    if "top_family" in source or "top family" in source:
        return "top_family"
    if "role" in source or "true_bucket" in source or "measured" in source:
        return "role"
    if claim.sub_family in {"Instrument Loops", "Drum Loops", "Bass Loops"}:
        return "role"
    if is_broad_depth_path(path):
        return "leaf_depth"
    if claim.is_real_candidate:
        return "source_identity"
    return "leaf_depth"


def infer_scope(claim_type: str) -> str:
    """Map claim type to the authority scope used by the arbiter."""
    scopes = {
        "structure": "structure_only",
        "top_family": "top_family_only",
        "role": "role_and_broad_bucket",
        "source_identity": "source_identity_when_not_vetoed",
        "leaf_depth": "depth_selection_only",
        "safety_review": "safety_review_only",
    }
    return scopes.get(claim_type, "diagnostic_only")


def infer_authority(claim: ConsensusClaim, claim_type: str) -> str:
    """Describe whether this claim can compete under current arbiter rules."""
    if claim.is_review or claim.family == "_TO_REVIEW":
        return "may_review"
    if claim_type == "structure":
        return "may_block_or_broaden_structure"
    if claim_type == "top_family":
        return "may_choose_broad_top_family"
    if claim_type == "role":
        return "may_choose_broad_role_bucket"
    if claim_type == "source_identity":
        return "may_choose_leaf_if_physics_and_shape_allow"
    if claim_type == "leaf_depth":
        return "may_broaden_depth"
    return "diagnostic_only"


def infer_role(public_label: str, sub_family: str) -> str:
    """Infer broad musical role from internal label/folder metadata."""
    text = f"{public_label} {sub_family}".lower()
    if "drum loops" in text or "drum loop" in text:
        return "drum_loop"
    if "instrument loops" in text or "mixed musical" in text or "multi instrument" in text:
        return "pitched_music_loop"
    if "bass loops" in text or "bass loop" in text:
        return "bass_loop"
    if "one shot" in text or "one-shot" in text:
        return "one_shot"
    if "riser" in text or "build" in text or "drop" in text or "downlifter" in text:
        return "transition_fx"
    if "texture" in text or "drone" in text or "atmosphere" in text:
        return "texture_bed"
    if "voice" in text or "vocal" in text:
        return "voice_or_vocal"
    return "unknown"


def infer_structure(public_label: str, sub_family: str, source: str) -> str:
    """Infer structure claim from internal destination and claim source."""
    text = f"{public_label} {sub_family} {source}".lower()
    if "loop" in text:
        return "loop"
    if "one shot" in text or "one-shot" in text or "hit" in text:
        return "one_shot"
    if "riser" in text or "build" in text:
        return "rising_transition"
    if "drop" in text or "downlifter" in text:
        return "falling_transition"
    if "drone" in text or "atmosphere" in text or "pad" in text:
        return "sustained"
    if "phrase" in text:
        return "phrase"
    return "unknown"


def infer_required_evidence(claim: ConsensusClaim, claim_type: str) -> tuple[str, ...]:
    """Return the evidence class this contract should be backed by."""
    if claim_type == "structure":
        return ("shape_vote", "measured_events")
    if claim_type == "top_family":
        return ("dynamic_role_gate", "shape_or_role_support")
    if claim_type == "role":
        return ("measured_roles", "shared_candidate_support")
    if claim_type == "source_identity":
        return ("brain_candidate", "physics_candidate", "no_shape_veto")
    if claim_type == "safety_review":
        return ("integrity_or_conflict_evidence",)
    if claim_type == "leaf_depth":
        return ("parent_role_or_depth_policy",)
    return ()


def infer_veto_evidence(claim: ConsensusClaim, claim_type: str) -> tuple[str, ...]:
    """Return evidence that should be allowed to block this claim later."""
    if claim_type == "source_identity":
        return ("strong_cross_family_consensus", "shape_structure_conflict", "physics_role_conflict")
    if claim_type in {"role", "leaf_depth"}:
        return ("hard_integrity_conflict", "decisive_source_identity_consensus")
    if claim_type == "top_family":
        return ("hard_shape_contradiction", "concrete_measured_transition")
    return ()


def infer_allowed_transitions(claim: ConsensusClaim, claim_type: str) -> tuple[str, ...]:
    """Return the destination families this contract may normally affect."""
    family = str(claim.family or "unknown")
    if claim_type == "safety_review":
        return ("*_to_review",)
    if claim_type == "source_identity":
        return (f"same_family:{family}",)
    if claim_type in {"structure", "role", "leaf_depth"}:
        return (f"same_family:{family}", f"broad_bucket:{family}")
    if claim_type == "top_family":
        return (f"top_family:{family}",)
    return ()


def infer_source_features_used(source: str, claim_type: str) -> tuple[str, ...]:
    """Name the evidence panels implied by a current claim source."""
    clean_source = str(source or "").lower()
    features: list[str] = []
    if "shape" in clean_source or claim_type == "structure":
        features.append("shape_vote")
    if "role" in clean_source or "measured" in clean_source or claim_type == "role":
        features.append("measured_roles")
    if "family" in clean_source or claim_type == "top_family":
        features.append("dynamic_role_gate")
    if claim_type == "source_identity":
        features.extend(["brain_topk", "physics_topk"])
    if not features:
        features.append("claim_metadata")
    return tuple(dict.fromkeys(features))


def is_broad_depth_path(path: str) -> bool:
    """Return True when a folder is a broad bucket, not a terminal source leaf."""
    return any(
        fragment in path
        for fragment in (
            "instrument loops",
            "drum loops",
            "bass loops",
            "hybrid textures",
            "hybrid designed fx",
            "generic percussion",
            "generic kick",
            "brass and woodwinds/loops",
        )
    )
