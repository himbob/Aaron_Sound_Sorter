# SOURCE-NAME BLINDNESS INVARIANT:
# Profile candidate producers may inspect voter output and measured audio facts.
# They must never inspect source filenames or source folders.
"""FX profile candidate claims."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import VoterResult
from aaron_sound_sorter.engine.decision_helpers import _path_has_any
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_category_guess


class ProfileFxClaimMixin:
    """Produce transition-FX claims from shape and BrainVoter evidence."""

    def transition_fx_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        shape_name: str,
        shape_confidence: float,
        raw_path: str,
        raw_score: float,
    ) -> ConsensusClaim | None:
        """Return an FX transition claim from shape plus real voter candidate."""
        if shape_confidence < 0.72 or shape_name not in {"transition_riser", "transition_drop"}:
            return None
        if raw.final_top == "FX" and not _path_has_any(raw_path, ("instrument loops", "bass loops")):
            return None
        if shape_name == "transition_riser":
            fragments = ("riser", "build", "sweep", "whoosh", "reverse")
            folder_path = "FX/Structural and Transitional FX/Risers and Builds"
        else:
            fragments = ("drop", "downlifter", "downlift", "fall", "sweep")
            folder_path = "FX/Structural and Transitional FX/Drops and Downlifters"
        best_guess = self.best_brain_guess(brain_result, fragments=fragments, max_rank=10)
        if best_guess is None:
            return None
        return claim_from_category_guess(
            guess=best_guess,
            folder_path=folder_path,
            source="profile_candidate_transition_fx_claim",
            reason=(
                "BrainVoter had a nearby transition-FX candidate and ShapeVoter "
                f"supported {shape_name}:{shape_confidence:.2f}; raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=min(raw_score + 2.0, float(best_guess.rank) + 2.0),
            can_override=True,
            strength=max(0.86, min(0.95, shape_confidence)),
        )
