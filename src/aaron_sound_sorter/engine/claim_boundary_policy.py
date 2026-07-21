# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Claim boundary policy for source-name-blind arbitration.

The policy does not choose folders.  It only prevents a claim from competing
when the claim oversteps its evidence type: shape/role evidence may describe
motion, repetition, and broad loop behavior, while concrete source leaves need
their own measured source proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claims import ConsensusClaim
from aaron_sound_sorter.engine.measured_source_contracts import supports_measured_drum_loop_owner


class ClaimEvidenceKind(str, Enum):
    """High-level contract for what a claim is allowed to assert."""

    SHAPE = "shape"
    ROLE = "role"
    SOURCE_IDENTITY = "source_identity"
    FAMILY_PERMISSION = "family_permission"
    FAMILY_VETO = "family_veto"
    LEAF_SUGGESTION = "leaf_suggestion"
    REVIEW_REASON = "review_reason"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClaimBoundaryDecision:
    """Result of a boundary-policy check."""

    allowed: bool
    evidence_kind: ClaimEvidenceKind
    reason: str = ""


class ClaimBoundaryPolicy:
    """Reject claims that cross from role/shape into unsupported identity."""

    def evaluate(
        self,
        *,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ClaimBoundaryDecision:
        """Return whether ``claim`` stays inside its evidence lane."""
        kind = self.evidence_kind(claim)
        if claim.is_review:
            return ClaimBoundaryDecision(True, ClaimEvidenceKind.REVIEW_REASON)
        if self._drum_loop_over_clean_bass_loop(claim, facts):
            return ClaimBoundaryDecision(False, kind, "drum_loop_claim_conflicts_with_clean_bass_loop")
        if self._drum_loop_over_clean_synth_loop(claim, facts):
            return ClaimBoundaryDecision(False, kind, "drum_loop_claim_conflicts_with_clean_synth_loop")
        if self._woodwind_leaf_over_bright_percussive_loop(claim, facts):
            return ClaimBoundaryDecision(False, kind, "woodwind_leaf_conflicts_with_bright_percussive_loop")
        if self._woodwind_leaf_over_synth_or_voice_pressure(claim, facts):
            return ClaimBoundaryDecision(False, kind, "woodwind_leaf_lacks_source_margin")
        if self._broad_wind_loop_over_crowded_instrument_loop(claim, facts):
            return ClaimBoundaryDecision(False, kind, "broad_wind_loop_lacks_source_margin")
        if self._transition_fx_over_stable_music_loop(raw_claim, claim, facts):
            return ClaimBoundaryDecision(False, kind, "transition_fx_lacks_motion_authority")
        if self._human_voice_fx_over_true_voice_instrument(claim, facts):
            return ClaimBoundaryDecision(False, kind, "human_voice_fx_conflicts_with_true_voice_instrument")
        if self._human_voice_fx_over_clean_instrument_loop(claim, facts):
            return ClaimBoundaryDecision(False, kind, "human_voice_fx_conflicts_with_clean_instrument_loop")
        return ClaimBoundaryDecision(True, kind)

    @staticmethod
    def evidence_kind(claim: ConsensusClaim) -> ClaimEvidenceKind:
        """Classify a claim by source contract, not by audio source identity."""
        source = str(claim.source or "")
        if claim.is_review or claim.family == "_TO_REVIEW":
            return ClaimEvidenceKind.REVIEW_REASON
        if source.startswith("measured_") or source.startswith("final_measured_"):
            return ClaimEvidenceKind.SOURCE_IDENTITY if claim.is_real_candidate else ClaimEvidenceKind.ROLE
        if source.startswith("learned_owner_"):
            return ClaimEvidenceKind.SOURCE_IDENTITY
        if "parent_eligibility" in source:
            return ClaimEvidenceKind.FAMILY_PERMISSION
        if "shape" in source:
            return ClaimEvidenceKind.SHAPE
        if "role" in source:
            return ClaimEvidenceKind.ROLE
        if "profile_candidate" in source or "baby_recall" in source:
            return ClaimEvidenceKind.LEAF_SUGGESTION
        if claim.is_real_candidate:
            return ClaimEvidenceKind.LEAF_SUGGESTION
        return ClaimEvidenceKind.UNKNOWN

    def _drum_loop_over_clean_bass_loop(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        if claim.family != "Drums" or claim.sub_family != "Drum Loops":
            return False
        shape = self._shape(facts)
        older_clean_bass_guard = bool(
            shape in {"bass_phrase", "beat_loop", "pitched_repetition_phrase"}
            and self._shape_confidence(facts) >= 0.86
            and self._metric(facts, "low_event_ratio") >= 0.86
            and self._metric(facts, "pitched_event_ratio") >= 0.88
            and self._metric(facts, "sustained_tonal_frame_ratio") >= 0.82
            and self._metric(facts, "percussive_event_ratio") <= 0.10
            and self._metric(facts, "drumlike_frame_ratio") <= 0.10
            and self._score(facts, "bass_synth_score", "bass_sub_score", "low_end_source_score") >= 0.62
            and self._score(facts, "drum_loop_source_score", "rhythmic_break_loop_score") <= 0.46
        )
        clean_bass_phrase_guard = bool(
            shape == "bass_phrase"
            and self._shape_confidence(facts) >= 0.86
            and self._metric(facts, "low_event_ratio") >= 0.86
            and self._metric(facts, "pitched_event_ratio") >= 0.88
            and self._metric(facts, "pitch_confidence") >= 0.70
            and self._metric(facts, "non_event_tonal_ratio") >= 0.50
            and self._metric(facts, "percussive_event_ratio") <= 0.08
            and self._metric(facts, "drumlike_frame_ratio") <= 0.08
            and self._score(facts, "bass_synth_score", "bass_sub_score", "low_end_source_score") >= 0.62
            and self._score(
                facts,
                "drum_hit_score",
                "drum_kick_source_score",
                "drum_snare_source_score",
                "drum_tom_conga_source_score",
                "drum_closed_hat_source_score",
                "drum_cymbal_source_score",
                "drum_rim_stick_source_score",
                "drum_shaker_tambourine_source_score",
                "drum_metallic_percussion_source_score",
            )
            <= 0.52
        )
        return older_clean_bass_guard or clean_bass_phrase_guard

    def _drum_loop_over_clean_synth_loop(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        if claim.family != "Drums" or claim.sub_family != "Drum Loops":
            return False
        if supports_measured_drum_loop_owner(facts):
            return False
        shape = self._shape(facts)
        synth_identity = self._score(
            facts,
            "synth_tonal_source_score",
            "synth_pad_score",
            "synth_chord_score",
            "synth_lead_score",
        )
        return bool(
            shape in {"beat_loop", "pitched_repetition_phrase", "repeated_phrase_loop", "sustained_pad"}
            and self._shape_confidence(facts) >= 0.78
            and synth_identity >= 0.56
            and self._metric(facts, "pitched_event_ratio") >= 0.84
            and self._metric(facts, "sustained_tonal_frame_ratio") >= 0.70
            and self._metric(facts, "percussive_event_ratio") <= 0.12
            and self._metric(facts, "drumlike_frame_ratio") <= 0.12
            and self._score(facts, "drum_hit_score") <= 0.48
        )

    def _woodwind_leaf_over_bright_percussive_loop(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        path = self._claim_path(claim)
        if claim.family != "Instruments" or not self._is_woodwind_leaf(path):
            return False
        shape = self._shape(facts)
        return bool(
            shape in {"pitched_repetition_phrase", "repeated_phrase_loop", "top_loop", "beat_loop"}
            and self._shape_confidence(facts) >= 0.72
            and self._metric(facts, "high_event_ratio") >= 0.76
            and self._metric(facts, "low_event_ratio") <= 0.10
            and self._metric(facts, "onset_count") >= 8.0
            and self._score(
                facts,
                "drum_shaker_tambourine_source_score",
                "drum_cymbal_source_score",
                "drum_metallic_percussion_source_score",
            )
            >= 0.68
            and self._score(facts, "reed_wind_authority_score", "woodwind_sax_score") <= 0.58
        )

    def _woodwind_leaf_over_synth_or_voice_pressure(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        path = self._claim_path(claim)
        if claim.family != "Instruments" or not self._is_woodwind_leaf(path):
            return False
        if self._shape(facts) not in {
            "pitched_repetition_phrase",
            "repeated_phrase_loop",
            "pitched_phrase",
            "pitched_phrase_shape",
            "sustained_pad",
        }:
            return False
        woodwind = self._score(facts, "woodwind_sax_score", "woodwind_flute_score", "reed_wind_score")
        synth_like_counter = self._score(
            facts,
            "synth_tonal_source_score",
            "synth_lead_score",
            "synth_pad_score",
            "synth_chord_score",
        )
        synth_floor = 0.58 if claim.source == "final_measured_branch_loop_broad_bucket" else 0.66
        synth_margin = 0.02 if claim.source == "final_measured_branch_loop_broad_bucket" else 0.04
        return bool(
            self._shape_confidence(facts) >= 0.78
            and synth_like_counter >= synth_floor
            and woodwind < synth_like_counter - synth_margin
            and self._score(facts, "reed_wind_authority_score") < 0.56
        )

    def _broad_wind_loop_over_crowded_instrument_loop(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        path = self._claim_path(claim)
        if claim.family != "Instruments" or path != "instruments/brass and woodwinds/loops":
            return False
        if self._shape(facts) not in {"pitched_repetition_phrase", "repeated_phrase_loop"}:
            return False
        woodwind = self._score(facts, "woodwind_sax_score", "woodwind_flute_score", "reed_wind_score")
        non_wind_counter = self._score(
            facts,
            "voice_score",
            "synth_tonal_source_score",
            "plucked_string_score",
            "struck_keys_score",
            "bowed_string_score",
            "string_violin_score",
            "string_cello_score",
            "pitched_mallet_instrument_score",
        )
        crowded_loop = self._score(
            facts,
            "plucked_string_score",
            "bowed_string_score",
            "string_violin_score",
            "pitched_mallet_instrument_score",
        )
        weak_margin_crowded_loop = bool(
            crowded_loop >= 0.60 and woodwind < 0.66 and self._score(facts, "reed_wind_authority_score") < 0.42
        )
        return bool(
            self._shape_confidence(facts) >= 0.82
            and (non_wind_counter >= woodwind + 0.06 or weak_margin_crowded_loop)
            and crowded_loop >= 0.56
            and self._score(facts, "reed_wind_authority_score") < 0.50
        )

    def _transition_fx_over_stable_music_loop(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        path = self._claim_path(claim)
        raw_path = self._claim_path(raw_claim)
        if claim.family != "FX" or not self._is_transition_fx(path):
            return False
        if raw_claim.family == "FX" and self._is_transition_fx(raw_path):
            return False
        return bool(
            self._shape(facts)
            in {"pitched_repetition_phrase", "repeated_phrase_loop", "pitched_phrase", "pitched_phrase_shape"}
            and self._shape_confidence(facts) >= 0.78
            and self._metric(facts, "pitched_event_ratio") >= 0.76
            and self._metric(facts, "sustained_tonal_frame_ratio") >= 0.70
            and self._score(facts, "fx_transition_authority_score", "fx_motion_score", "fx_riser_build_score") < 0.46
        )

    def _human_voice_fx_over_clean_instrument_loop(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        path = self._claim_path(claim)
        if claim.family != "FX" or "human and voice" not in path:
            return False
        return bool(
            self._shape(facts)
            in {"pitched_repetition_phrase", "repeated_phrase_loop", "pitched_phrase_shape", "sustained_pad"}
            and self._shape_confidence(facts) >= 0.86
            and self._metric(facts, "pitched_event_ratio") >= 0.84
            and self._metric(facts, "sustained_tonal_frame_ratio") >= 0.74
            and self._metric(facts, "percussive_event_ratio") <= 0.14
            and self._metric(facts, "drumlike_frame_ratio") <= 0.14
            and self._score(facts, "synth_tonal_source_score", "struck_keys_score", "plucked_string_score") >= 0.50
            and self._score(facts, "voice_score", "human_spoken_voice_score") < 0.88
        )

    def _human_voice_fx_over_true_voice_instrument(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        path = self._claim_path(claim)
        if claim.family != "FX" or "human and voice" not in path:
            return False
        voice_role_strength = max(
            self._measured_role_strength(facts, "vocal_music_phrase"),
            self._measured_role_strength(facts, "vocal_phrase"),
            self._measured_role_strength(facts, "vocal_one_shot"),
            self._measured_role_strength(facts, "voiced_one_shot"),
        )
        transition_strength = self._score(
            facts,
            "fx_transition_authority_score",
            "fx_motion_score",
            "fx_riser_build_score",
            "fx_drop_downlifter_score",
            "fx_whoosh_sweep_score",
        )
        return bool(
            voice_role_strength >= 0.68
            and self._score(facts, "voice_score", "human_spoken_voice_score") >= 0.78
            and self._metric(facts, "pitched_event_ratio") >= 0.82
            and self._metric(facts, "f0_voiced_ratio") >= 0.68
            and self._metric(facts, "sustained_tonal_frame_ratio") >= 0.68
            and self._metric(facts, "percussive_event_ratio") <= 0.16
            and self._metric(facts, "drumlike_frame_ratio") <= 0.16
            and transition_strength <= 0.42
        )

    @staticmethod
    def _claim_path(claim: ConsensusClaim) -> str:
        return str(claim.folder_path or claim.label or "").strip("/").lower()

    @staticmethod
    def _is_woodwind_leaf(path: str) -> bool:
        if "brass and woodwinds/loops" in path:
            return False
        return any(token in path for token in ("woodwind", "saxophone", "flute", "clarinet", "bassoon"))

    @staticmethod
    def _is_transition_fx(path: str) -> bool:
        return any(token in path for token in ("risers", "builds", "drops", "downlifters", "whoosh", "sweep"))

    @staticmethod
    def _shape(facts: SharedAudioFacts | None) -> str:
        shape_vote = ClaimBoundaryPolicy._shape_vote(facts)
        return str(shape_vote.get("primary_shape") or "")

    @staticmethod
    def _shape_confidence(facts: SharedAudioFacts | None) -> float:
        return ClaimBoundaryPolicy._float(ClaimBoundaryPolicy._shape_vote(facts).get("confidence"))

    @staticmethod
    def _shape_vote(facts: SharedAudioFacts | None) -> dict:
        evidence = getattr(facts, "evidence", {}) if facts is not None else {}
        shape_vote = evidence.get("shape_vote") if isinstance(evidence, dict) else {}
        return shape_vote if isinstance(shape_vote, dict) else {}

    @staticmethod
    def _metric(facts: SharedAudioFacts | None, name: str) -> float:
        shape_vote = ClaimBoundaryPolicy._shape_vote(facts)
        value = ClaimBoundaryPolicy._float(shape_vote.get(name))
        feature_values = getattr(facts, "feature_values_by_name", {}) if facts is not None else {}
        if isinstance(feature_values, dict):
            value = max(value, ClaimBoundaryPolicy._float(feature_values.get(name)))
        return value

    @staticmethod
    def _score(facts: SharedAudioFacts | None, *names: str) -> float:
        evidence = getattr(facts, "evidence", {}) if facts is not None else {}
        flat = {}
        if isinstance(evidence, dict):
            subpanels = evidence.get("physics_subpanels")
            if isinstance(subpanels, dict) and isinstance(subpanels.get("flat"), dict):
                flat = subpanels["flat"]
        feature_values = getattr(facts, "feature_values_by_name", {}) if facts is not None else {}
        best = 0.0
        for name in names:
            if isinstance(evidence, dict):
                best = max(best, ClaimBoundaryPolicy._float(evidence.get(name)))
            if isinstance(flat, dict):
                best = max(best, ClaimBoundaryPolicy._float(flat.get(name)))
            if isinstance(feature_values, dict):
                best = max(best, ClaimBoundaryPolicy._float(feature_values.get(name)))
        return best

    @staticmethod
    def _measured_role_strength(facts: SharedAudioFacts | None, name: str) -> float:
        evidence = getattr(facts, "evidence", {}) if facts is not None else {}
        roles = evidence.get("measured_roles") if isinstance(evidence, dict) else {}
        if not isinstance(roles, dict):
            return 0.0
        value = roles.get(name)
        if isinstance(value, dict):
            value = value.get("strength", value.get("score"))
        return ClaimBoundaryPolicy._float(value)

    @staticmethod
    def _float(value: object) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0
