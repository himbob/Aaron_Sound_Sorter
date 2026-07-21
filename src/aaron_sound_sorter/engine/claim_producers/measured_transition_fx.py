# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured transition-FX claim producer.

The final arbiter used to repair obvious riser/drop/transition bodies after a
winner had already been picked.  This producer emits that same broad structure
claim earlier from ShapeVoter and PhysicsFXRoleLayer evidence.  It deliberately
routes only to broad transition/FX folders; it does not choose concrete source
identity such as a specific pack label or named object.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _feature_number_from_facts,
    _norm_path,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path
from aaron_sound_sorter.engine.measured_source_contracts import supports_clean_low_bass_phrase_owner
from aaron_sound_sorter.voters.scoring_tools import role_strength

MOTION_BRANCHES = {
    "RiserBuild",
    "DropDownlifter",
    "WhooshSweep",
    "ReverseSwell",
    "ImpactHit",
    "GlitchStutter",
    "BlipBeep",
    "SirenAlarm",
}

TRUSTED_FX_BRANCHES = MOTION_BRANCHES | {
    "TextureAmbience",
    "MachineMechanical",
    "FoleyMaterial",
    "SmallObjectCluster",
    "HumanCreatureFX",
    "FormantFX",
    "RadioElectrical",
    "DesignedNoiseHybrid",
}


DESIGNED_FX_SHAPES = {
    "designed_low_fx",
    "designed_motion_fx_loop",
    "designed_tonal_fx",
}

DESIGNED_FX_CANDIDATE_FRAGMENTS = (
    "hybrid designed",
    "designed noise",
    "structural and transitional",
    "sweep",
    "whoosh",
    "swoosh",
    "swish",
    "riser",
    "build",
    "drop",
    "downlifter",
    "reverse",
    "impact",
    "boom",
    "glitch",
    "stutter",
    "siren",
    "alarm",
    "radio",
    "electrical",
    "human and voice fx",
    "mouth sounds",
    "breath",
    "scream",
)


class MeasuredTransitionFxClaimProducer:
    """Emit broad transition-FX claims before final arbitration."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return a measured transition-FX claim, or no claim when support is weak."""
        claim = self._produce_claim(context)
        return [] if claim is None else [claim]

    def _produce_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        raw = context.raw
        facts = context.facts
        if self._facts_support_measured_sub_impact_hit(facts):
            return claim_from_folder_path(
                folder_path="FX/Impacts and Hits/Sub Hit/One Shots",
                source="final_measured_sub_impact_hit_invariant",
                reason=(
                    "measured FX sub-hit claim: short low impact body was established "
                    "before broad bass/instrument-loop fallbacks could win"
                ),
                shared=raw.shared_candidates,
                raw_candidate_score=raw.raw_candidate_score,
                brain_rank=raw.brain_rank,
                physics_rank=raw.physics_rank,
                shared_winner=raw.shared_winner or raw.folder_path,
                can_override=True,
                strength=max(0.90, raw.strength),
                is_real_candidate=False,
            )
        if not self._facts_support_measured_transition_fx(facts, raw):
            return None

        shape = _shape_vote_from_facts(facts)
        if (
            self._parent_eligibility_blocks_fx(context)
            and not self._facts_support_measured_transition_owner(facts)
            and not self._facts_support_shape_only_transition_rehome(facts, raw)
            and not self._facts_support_strong_designed_motion_fx(facts)
            and not self._facts_support_measured_impact_tail_body(facts, raw)
            and not self._facts_support_learned_or_brain_fx_role_owner(facts)
        ):
            return None
        designed_shape_body = self._facts_support_measured_designed_fx_body(facts, raw)
        branch = self._measured_physics_fx_role_branch(facts)
        use_branch_folder = bool(
            branch
            and (
                self._facts_support_learned_or_brain_fx_role_owner(facts)
                or not (shape in DESIGNED_FX_SHAPES and designed_shape_body)
            )
        )
        if (
            not use_branch_folder
            and self._parent_music_loop_release_should_block_transition(raw, facts)
            and not designed_shape_body
            and not self._facts_support_measured_transition_owner(facts)
            and not self._facts_support_measured_impact_tail_body(facts, raw)
        ):
            return None
        if use_branch_folder:
            folder_path = self._measured_physics_fx_role_folder(branch or "")
            reason_tail = f"PhysicsFXRoleLayer measured {branch}"
        else:
            if shape == "transition_riser":
                folder_path = "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
                reason_tail = "ShapeVoter measured rising transition motion"
            elif shape in {"transition_drop", "transition_downlifter"}:
                folder_path = (
                    "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX"
                )
                reason_tail = "ShapeVoter measured falling/down transition motion"
            elif shape == "designed_motion_fx_loop":
                folder_path = self._designed_motion_fx_loop_folder(facts)
                reason_tail = "ShapeVoter measured designed FX motion-loop decoy"
            elif shape == "designed_low_fx":
                folder_path = self._designed_low_fx_folder(facts)
                reason_tail = "ShapeVoter measured designed low/motion FX decoy"
            elif shape == "whoosh_sweep":
                folder_path = "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/Long FX"
                reason_tail = "ShapeVoter measured whoosh/sweep motion"
            elif shape == "reverse_swell":
                folder_path = "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/Long FX"
                reason_tail = "ShapeVoter measured reverse/swell motion"
            elif shape == "hybrid_fx_motion":
                folder_path = self._hybrid_motion_fx_folder(facts)
                reason_tail = "ShapeVoter measured high-confidence hybrid FX motion"
            elif shape == "impact_with_tail":
                folder_path = "FX/Impacts and Hits/Generic Impact/Long FX"
                reason_tail = "ShapeVoter measured impact-tail body with FX impact support"
            else:
                folder_path = "FX/Hybrid Designed FX"
                reason_tail = "ShapeVoter measured designed tonal/formant FX decoy"

        return claim_from_folder_path(
            folder_path=folder_path,
            source="final_measured_transition_fx_invariant",
            reason=(
                f"measured transition-FX claim: broad FX motion was established before final arbitration; {reason_tail}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(0.90, raw.strength),
            is_real_candidate=False,
        )

    def _parent_eligibility_blocks_fx(self, context: DecisionContext) -> bool:
        """Return True when parent-role evidence already excludes FX."""
        eligibility = context.eligibility
        if eligibility is None:
            return False
        allowed = {str(top) for top in getattr(eligibility, "allowed_top_families", ())}
        if not allowed or "FX" in allowed:
            return False
        confidence = self._safe_float(getattr(eligibility, "confidence", 0.0), 0.0)
        if confidence < 0.74 and not bool(getattr(eligibility, "decisive", False)):
            return False
        return bool(allowed & {"Instruments", "Drums"})

    def _facts_support_strong_designed_motion_fx(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for designed-FX shapes with real motion/noise authority.

        This guard lets obvious sweeps/uplifters survive a broad music-loop
        parent decision while keeping clean tonal, vocal, and mixed instrument
        loops from being pulled into generic Hybrid FX by formant-ish shape
        scores alone.
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in DESIGNED_FX_SHAPES:
            return False
        fx_motion = max(
            self._subpanel_score(facts, "fx_motion_score"),
            self._subpanel_score(facts, "fx_transition_authority_score"),
            self._subpanel_score(facts, "fx_whoosh_sweep_score"),
            self._subpanel_score(facts, "fx_drop_downlifter_score"),
            self._subpanel_score(facts, "fx_riser_build_score"),
            self._subpanel_score(facts, "fx_reverse_score"),
            self._subpanel_score(facts, "fx_glitch_stutter_score"),
            self._subpanel_score(facts, "fx_radio_electrical_score"),
        )
        shape_motion = max(
            self._shape_score(facts, "designed_motion_fx_loop"),
            self._shape_score(facts, "hybrid_fx_motion"),
            self._shape_score(facts, "whoosh_sweep"),
            self._shape_score(facts, "reverse_swell"),
            self._shape_score(facts, "transition_riser"),
            self._shape_score(facts, "transition_drop"),
        )
        noisy_or_rough = max(
            self._shape_number(facts, "spectral_flatness_mean"),
            self._shape_number(facts, "spectral_entropy_mean"),
        )
        clean_music_loop = bool(
            self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and max(
                self._shape_number(facts, "sustained_tonal_frame_ratio"),
                self._shape_number(facts, "non_event_tonal_ratio"),
            )
            >= 0.82
            and fx_motion < 0.62
        )
        strong_voice_loop = bool(
            max(
                self._subpanel_score(facts, "voice_score"),
                self._subpanel_score(facts, "human_spoken_voice_score"),
                _feature_number_from_facts(facts, "voice_score"),
                _feature_number_from_facts(facts, "human_spoken_voice_score"),
            )
            >= 0.78
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.60
        )
        return bool(
            _shape_confidence_from_facts(facts) >= 0.82
            and fx_motion >= 0.62
            and shape_motion >= 0.70
            and noisy_or_rough >= 0.50
            and not clean_music_loop
            and not strong_voice_loop
        )

    def _facts_support_measured_sub_impact_hit(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for short low sub/boom hits that are FX, not bass instruments."""
        if facts is None:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 0.95:
            return False
        event_count = max(
            self._shape_number(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        if event_count > 3.0:
            return False
        low_body = max(
            self._shape_number(facts, "low_event_ratio"),
            _feature_number_from_facts(facts, "low_total"),
        )
        if low_body < 0.78:
            return False
        sub_hit = self._subpanel_score(facts, "fx_sub_hit_score")
        impact = max(
            self._subpanel_score(facts, "fx_impact_score"),
            self._subpanel_score(facts, "fx_slam_score"),
            self._subpanel_score(facts, "fx_impacts_and_hits_sub_hit_one_shots_score"),
        )
        if max(sub_hit, impact) < 0.56:
            return False
        # Clean 808/bass one-shots usually have a sustained voiced pitch contour.
        # Sub impacts can have pitch confidence, but low voiced-frame coverage.
        if self._shape_number(facts, "f0_voiced_ratio") > 0.38:
            return False
        roles = facts.evidence.get("measured_roles", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
        if isinstance(roles, dict) and role_strength(roles, "bass_loop") >= 0.60:
            return False
        if self._subpanel_score(facts, "drum_kick_source_score") >= 0.58:
            return False
        return True

    def _facts_support_measured_impact_tail_body(
        self,
        facts: SharedAudioFacts | None,
        raw: ConsensusClaim,
    ) -> bool:
        """Return True for FX impact tails that are being mistaken for music loops.

        The audio body must look like an impact or boom with a tail, not a
        clean loop: low voiced-frame coverage, weak loop periodicity, and
        moderate impact/boom/slam/sub-hit physics.  A nearby internal FX-impact
        candidate is also required unless the raw candidate is already FX or
        review.  This lets learned/brain evidence authorize the claim without
        using source filenames.
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {"impact_with_tail", "hit_with_tail"}:
            return False
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape_confidence < 0.70:
            return False
        impact_support = max(
            self._subpanel_score(facts, "fx_impact_score"),
            self._subpanel_score(facts, "fx_boom_score"),
            self._subpanel_score(facts, "fx_slam_score"),
            self._subpanel_score(facts, "fx_sub_hit_score"),
            self._shape_score(facts, "impact_with_tail"),
        )
        if impact_support < 0.50:
            return False
        if self._facts_support_short_drum_hit_over_impact_fx(facts, raw):
            return False
        weak_loop_periodicity = max(
            self._shape_number(facts, "loop_onset_periodicity"),
            self._shape_number(facts, "loop_pulse_clarity"),
        )
        if weak_loop_periodicity > 0.42:
            return False
        f0_voiced = self._shape_number(facts, "f0_voiced_ratio")
        if f0_voiced > 0.35:
            return False
        if self._facts_support_true_voice_role(facts):
            return False
        if self._facts_support_clean_instrument_loop_over_designed_fx(facts, raw):
            return False
        if shape_confidence >= 0.74 and impact_support >= 0.56 and weak_loop_periodicity <= 0.36 and f0_voiced <= 0.25:
            return True
        has_fx_impact_candidate = self._shared_candidate_has_top_family(
            raw,
            ("impact", "boom", "slam", "sub hit"),
            top_family="FX",
            max_score=64.0,
            max_brain_rank=10,
            max_physics_rank=64,
        )
        return bool(raw.family in {"FX", "_TO_REVIEW"} or has_fx_impact_candidate)

    def _facts_support_short_drum_hit_over_impact_fx(
        self,
        facts: SharedAudioFacts | None,
        raw: ConsensusClaim,
    ) -> bool:
        """Return True when a short drum hit explains impact-like shape better."""
        if facts is None or raw.family != "Drums":
            return False
        drum_source = max(
            self._subpanel_score(facts, "drum_hit_score"),
            self._subpanel_score(facts, "drum_kick_source_score"),
            self._subpanel_score(facts, "drum_snare_source_score"),
            self._subpanel_score(facts, "drum_clap_source_score"),
            self._subpanel_score(facts, "drum_tom_conga_source_score"),
            self._subpanel_score(facts, "drum_cymbal_source_score"),
        )
        one_shot = max(
            self._subpanel_score(facts, "role_one_shot_score"),
            self._shape_score(facts, "single_hit"),
            self._shape_score(facts, "hit_with_tail"),
        )
        fx_motion = max(
            self._subpanel_score(facts, "fx_motion_score"),
            self._subpanel_score(facts, "fx_transition_authority_score"),
            self._subpanel_score(facts, "fx_riser_build_score"),
            self._subpanel_score(facts, "fx_drop_downlifter_score"),
            self._subpanel_score(facts, "fx_whoosh_sweep_score"),
            self._subpanel_score(facts, "fx_reverse_score"),
        )
        noisy_air = self._subpanel_score(facts, "physics_subpanel_noisy_air")
        clean_tone = self._subpanel_score(facts, "physics_subpanel_clean_tone")
        return bool(
            drum_source >= 0.66 and one_shot >= 0.70 and fx_motion <= 0.40 and (noisy_air <= 0.48 or clean_tone >= 0.45)
        )

    def _facts_support_measured_transition_fx(
        self,
        facts: SharedAudioFacts | None,
        raw: ConsensusClaim,
    ) -> bool:
        """Return True for measured broad transition-FX structure."""
        if facts is None:
            return False
        if self._facts_support_rhythmic_drum_loop_over_transition(facts):
            return False
        if self._facts_support_clean_bass_loop_owner_over_fx(facts):
            return False
        measured_transition_body = self._facts_support_measured_transition_body(facts)
        measured_designed_body = self._facts_support_measured_designed_fx_body(facts, raw)
        measured_fx_decoy_body = self._facts_support_measured_fx_decoy_shape_body(facts, raw)
        measured_impact_tail_body = self._facts_support_measured_impact_tail_body(facts, raw)
        if self._facts_support_protected_loop_owner_over_fx(facts) and not measured_impact_tail_body:
            return False
        if raw.family == "Drums" and not (
            measured_transition_body or measured_designed_body or measured_fx_decoy_body or measured_impact_tail_body
        ):
            return False

        branch = self._measured_physics_fx_role_branch(facts)
        shape = _shape_vote_from_facts(facts)
        if branch and not (shape in DESIGNED_FX_SHAPES and measured_designed_body):
            if self._facts_support_true_voice_role(facts):
                return False
            if raw.family == "FX" and branch not in MOTION_BRANCHES:
                return False
            if raw.family in {"FX", "_TO_REVIEW"}:
                return True
            if self._facts_support_learned_or_brain_fx_role_owner(facts):
                return True
            return self._shared_candidate_has_top_family(
                raw,
                self._measured_physics_fx_role_fragments(branch),
                top_family="FX",
                max_score=18.0,
                max_brain_rank=10,
                max_physics_rank=10,
            )

        shape_confidence = _shape_confidence_from_facts(facts)
        if shape in DESIGNED_FX_SHAPES:
            if shape_confidence < 0.62:
                return False
            if self._facts_support_true_voice_role(facts):
                return False
            if not measured_designed_body:
                return False
            if self._facts_support_clean_instrument_loop_over_designed_fx(facts, raw):
                return False
            # When ShapeVoter and the measured FX role layer both agree, do not
            # require the candidate ranker to have found a named FX leaf.  That
            # old requirement is exactly how short stutters/grooves got stolen
            # by snare/bass/instrument shortcuts even though the measured body
            # was already a designed-FX decoy.
            if branch:
                return True
            if raw.family in {"FX", "_TO_REVIEW"}:
                return True
            return self._shared_candidate_has_top_family(
                raw,
                DESIGNED_FX_CANDIDATE_FRAGMENTS,
                top_family="FX",
                max_score=22.0,
                max_brain_rank=14,
                max_physics_rank=14,
            )
        if measured_fx_decoy_body:
            if self._facts_support_true_voice_role(facts):
                return False
            if self._facts_support_clean_instrument_loop_over_designed_fx(facts, raw):
                return False
            if raw.family in {"FX", "_TO_REVIEW", "Drums", "Instruments"}:
                return True
        if measured_impact_tail_body:
            if self._facts_support_true_voice_role(facts):
                return False
            return True
        if shape not in {
            "transition_riser",
            "transition_downlifter",
            "transition_drop",
            "whoosh_sweep",
            "reverse_swell",
            "hybrid_fx_motion",
        }:
            return False
        if shape_confidence < 0.70:
            return False

        transition_candidate = self._shared_candidate_has_top_family(
            raw,
            (
                "riser",
                "build",
                "drop",
                "downlifter",
                "transition",
                "sweep",
                "whoosh",
                "reverse",
                "tail",
                "glitch",
                "stutter",
                "hybrid designed",
            ),
            top_family="FX",
            max_score=18.0,
            max_brain_rank=10,
            max_physics_rank=10,
        )
        if self._facts_support_measured_transition_owner(facts):
            return True
        if not measured_transition_body:
            shape_only_transition_rehome = bool(
                raw.family == "FX"
                and transition_candidate
                and shape_confidence >= 0.78
                and shape
                in {
                    "transition_riser",
                    "transition_downlifter",
                    "transition_drop",
                    "whoosh_sweep",
                    "reverse_swell",
                    "hybrid_fx_motion",
                }
            )
            if not shape_only_transition_rehome:
                return False

        if self._shape_number(facts, "drumlike_frame_ratio") >= 0.42 and not (
            measured_transition_body and transition_candidate
        ):
            return False
        if raw.family in {"FX", "_TO_REVIEW"}:
            return True

        strong_instrument_parent = self._shared_candidate_count_top_family(
            raw,
            (
                "instrument loops",
                "guitar",
                "keys",
                "piano",
                "strings",
                "woodwind",
                "sax",
                "brass",
                "voice",
                "vocal",
                "synth",
            ),
            top_family="Instruments",
            max_score=16.0,
            max_brain_rank=6,
            max_physics_rank=6,
        )
        return bool(
            transition_candidate and (strong_instrument_parent == 0 or raw.source == "profile_candidate_synth_claim")
        )

    def _facts_support_measured_transition_owner(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when shape and FX motion prove transition ownership."""
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        transition_motion_score = max(
            self._subpanel_score(facts, "fx_riser_build_score"),
            self._subpanel_score(facts, "fx_drop_downlifter_score"),
            self._subpanel_score(facts, "fx_whoosh_sweep_score"),
            self._subpanel_score(facts, "fx_reverse_score"),
            self._subpanel_score(facts, "fx_transition_authority_score"),
            self._subpanel_score(facts, "fx_motion_score"),
        )
        transition_slope = max(
            abs(self._shape_signed_number(facts, "centroid_slope_norm")),
            abs(self._shape_signed_number(facts, "librosa_spectral_centroid_slope_norm")),
        )
        clean_instrument_counter = bool(
            max(
                self._shape_number(facts, "pitched_event_ratio"),
                self._shape_number(facts, "pitch_confidence"),
            )
            >= 0.88
            and max(
                self._shape_number(facts, "sustained_tonal_frame_ratio"),
                self._shape_number(facts, "non_event_tonal_ratio"),
            )
            >= 0.82
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.16
            and max(
                self._subpanel_score(facts, "synth_tonal_source_score"),
                self._subpanel_score(facts, "synth_pad_score"),
                self._subpanel_score(facts, "synth_chord_score"),
                self._subpanel_score(facts, "struck_keys_score"),
                self._subpanel_score(facts, "keys_tonal_decay_score"),
                self._subpanel_score(facts, "bass_synth_score"),
                self._subpanel_score(facts, "woodwind_sax_score"),
                self._subpanel_score(facts, "reed_wind_score"),
            )
            >= 0.66
            and transition_motion_score < 0.64
        )
        return bool(
            shape
            in {
                "transition_riser",
                "transition_downlifter",
                "transition_drop",
                "whoosh_sweep",
                "reverse_swell",
                "hybrid_fx_motion",
            }
            and shape_confidence >= 0.90
            and transition_motion_score >= 0.64
            and transition_slope >= 0.18
            and not clean_instrument_counter
            and not self._facts_support_true_voice_role(facts)
        )

    def _facts_support_shape_only_transition_rehome(
        self,
        facts: SharedAudioFacts | None,
        raw: ConsensusClaim,
    ) -> bool:
        """Return True when a raw FX transition leaf should follow ShapeVoter direction.

        This does not let FX steal from a non-FX raw winner. It only replaces a
        concrete FX transition leaf, such as a riser, with the measured broad
        transition direction when ShapeVoter says the body behaves as a drop,
        downlifter, riser, whoosh, or reverse.
        """
        if facts is None or raw.family != "FX":
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {
            "transition_riser",
            "transition_downlifter",
            "transition_drop",
            "whoosh_sweep",
            "reverse_swell",
            "hybrid_fx_motion",
        }:
            return False
        if _shape_confidence_from_facts(facts) < 0.78:
            return False
        if self._facts_support_true_voice_role(facts):
            return False
        if self._shape_number(facts, "drumlike_frame_ratio") >= 0.42:
            return False
        return self._shared_candidate_has_top_family(
            raw,
            (
                "riser",
                "build",
                "drop",
                "downlifter",
                "transition",
                "sweep",
                "whoosh",
                "reverse",
                "tail",
                "hybrid designed",
            ),
            top_family="FX",
            max_score=18.0,
            max_brain_rank=10,
            max_physics_rank=10,
        )

    def _facts_support_protected_loop_owner_over_fx(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when a real loop/phrase owner should block broad FX rescue.

        This is the owner-arbitration seam: shape can rescue obvious designed
        FX, but a high-confidence repeated musical/drum body is a protected
        owner unless there is clear directional FX motion.
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {
            "beat_loop",
            "top_loop",
            "repeated_phrase_loop",
            "pitched_repetition_phrase",
            "designed_motion_fx_loop",
            "designed_tonal_fx",
            "designed_low_fx",
        }:
            return False
        if shape_confidence < 0.78:
            return False
        onset_count = max(
            self._shape_number(facts, "onset_count"), _feature_number_from_facts(facts, "event_count_estimate")
        )
        onset_span = self._shape_number(facts, "onset_span_ratio")
        true_repetition = self._shape_number(facts, "true_repetition_score")
        if onset_count < 8.0 or onset_span < 0.62 or true_repetition < 0.62:
            return False
        slope = max(
            abs(self._shape_signed_number(facts, "centroid_slope_norm")),
            abs(self._shape_signed_number(facts, "librosa_spectral_centroid_slope_norm")),
        )
        fx_motion = max(
            self._subpanel_score(facts, "fx_motion_score"),
            self._subpanel_score(facts, "fx_transition_authority_score"),
            self._subpanel_score(facts, "fx_whoosh_sweep_score"),
            self._subpanel_score(facts, "fx_drop_downlifter_score"),
            self._subpanel_score(facts, "fx_riser_build_score"),
            self._subpanel_score(facts, "fx_glitch_stutter_score"),
            self._subpanel_score(facts, "fx_radio_electrical_score"),
        )
        designed_fx_pressure = max(
            self._shape_score(facts, "designed_low_fx"),
            self._shape_score(facts, "designed_motion_fx_loop"),
            self._shape_score(facts, "designed_tonal_fx"),
            self._shape_score(facts, "hybrid_fx_motion"),
            self._shape_score(facts, "whoosh_sweep"),
            self._shape_score(facts, "reverse_swell"),
            self._shape_score(facts, "transition_riser"),
            self._shape_score(facts, "transition_drop"),
            self._subpanel_score(facts, "fx_formant_score"),
            self._subpanel_score(facts, "fx_siren_score"),
            self._subpanel_score(facts, "fx_alarm_score"),
            fx_motion,
        )
        real_drum_owner = bool(
            max(
                self._subpanel_score(facts, "drum_loop_source_score"),
                self._subpanel_score(facts, "drum_hit_score"),
                self._subpanel_score(facts, "drum_cymbal_source_score"),
                self._subpanel_score(facts, "drum_snare_source_score"),
                self._subpanel_score(facts, "drum_shaker_tambourine_source_score"),
            )
            >= 0.76
            and max(
                self._shape_number(facts, "percussive_event_ratio"),
                self._shape_number(facts, "drumlike_frame_ratio"),
                self._shape_number(facts, "pulse_regularity"),
            )
            >= 0.45
        )
        clean_musical_owner = bool(
            self._shape_number(facts, "pitched_event_ratio") >= 0.82
            and max(
                self._shape_number(facts, "sustained_tonal_frame_ratio"),
                self._shape_number(facts, "non_event_tonal_ratio"),
            )
            >= 0.78
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.18
            and fx_motion < 0.46
        )
        if (
            shape in {"designed_low_fx", "designed_motion_fx_loop", "designed_tonal_fx"}
            and designed_fx_pressure >= 0.76
            and fx_motion >= 0.54
            and not real_drum_owner
            and not clean_musical_owner
        ):
            return False
        # Strong directional sweeps/downlifters are the FX class we explicitly
        # still allow to beat loop-looking evidence.
        if slope >= 0.30 and fx_motion >= 0.60 and self._shape_number(facts, "pitched_event_ratio") <= 0.25:
            return False
        pitched_loop_owner = bool(
            self._shape_number(facts, "pitched_event_ratio") >= 0.72
            and max(
                self._shape_number(facts, "sustained_tonal_frame_ratio"),
                self._shape_number(facts, "non_event_tonal_ratio"),
            )
            >= 0.62
            and self._shape_number(facts, "pitch_confidence") >= 0.55
            and (slope < 0.45 or fx_motion < 0.60)
        )
        drum_groove_anchor = max(
            self._shape_number(facts, "pulse_regularity"),
            self._shape_number(facts, "low_event_ratio"),
            self._shape_number(facts, "mid_event_ratio") * 0.65,
        )
        drum_loop_owner = bool(
            drum_groove_anchor >= 0.16
            and max(
                self._shape_number(facts, "percussive_event_ratio"),
                self._shape_number(facts, "drumlike_frame_ratio"),
                self._subpanel_score(facts, "drum_loop_source_score"),
                self._subpanel_score(facts, "drum_hit_score"),
                role_strength(
                    facts.evidence.get("measured_roles", {})
                    if isinstance(getattr(facts, "evidence", None), dict)
                    else {},
                    "low_rhythmic_drum_loop",
                ),
                role_strength(
                    facts.evidence.get("measured_roles", {})
                    if isinstance(getattr(facts, "evidence", None), dict)
                    else {},
                    "percussive_drum_loop",
                ),
            )
            >= 0.18
            and (slope < 0.45 or fx_motion < 0.60)
        )
        return bool(pitched_loop_owner or drum_loop_owner)

    def _facts_support_measured_designed_fx_body(
        self,
        facts: SharedAudioFacts | None,
        raw: ConsensusClaim,
    ) -> bool:
        """Return True for shape-measured designed FX that mimics music/drums.

        This is intentionally stricter than the broad transition test: the shape
        must be one of the designed-FX decoy shapes, the physics subpanels must
        show compatible FX evidence, and real drum/instrument material gets a
        chance to block the rescue.
        """
        if facts is None:
            return False
        if self._facts_support_protected_loop_owner_over_fx(facts):
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in DESIGNED_FX_SHAPES:
            return False
        shape_confidence = _shape_confidence_from_facts(facts)
        fx_motion = max(
            self._subpanel_score(facts, "fx_motion_score"),
            self._subpanel_score(facts, "fx_transition_authority_score"),
            self._subpanel_score(facts, "fx_whoosh_sweep_score"),
            self._subpanel_score(facts, "fx_glitch_stutter_score"),
            self._subpanel_score(facts, "fx_reverse_score"),
            self._subpanel_score(facts, "fx_riser_build_score"),
            self._subpanel_score(facts, "fx_drop_downlifter_score"),
            self._subpanel_score(facts, "fx_radio_electrical_score"),
        )
        fx_low = max(
            self._subpanel_score(facts, "fx_sub_hit_score"),
            self._subpanel_score(facts, "fx_boom_score"),
            self._subpanel_score(facts, "fx_impact_score"),
            self._subpanel_score(facts, "fx_drop_downlifter_score"),
            self._subpanel_score(facts, "fx_siren_score"),
        )
        fx_tonal = max(
            self._subpanel_score(facts, "fx_formant_score"),
            self._subpanel_score(facts, "fx_siren_score"),
            self._subpanel_score(facts, "fx_alarm_score"),
            self._subpanel_score(facts, "fx_radio_electrical_score"),
            self._subpanel_score(facts, "fx_glitch_stutter_score"),
        )
        fx_support = max(fx_motion, fx_low, fx_tonal)
        if fx_support < 0.54:
            return False
        if shape_confidence < 0.62:
            return False
        if self._facts_support_real_drum_material_over_designed_fx(facts):
            return False
        if shape == "designed_low_fx":
            low_body = bool(
                fx_low >= 0.56
                and self._shape_number(facts, "low_event_ratio") >= 0.48
                and self._shape_number(facts, "tail_ratio") >= 0.24
            )
            motion_tail_body = bool(
                self._shape_number(facts, "tail_ratio") >= 0.70
                and self._shape_number(facts, "onset_span_ratio") >= 0.45
                and max(
                    self._shape_score(facts, "hybrid_fx_motion"),
                    self._shape_score(facts, "designed_motion_fx_loop"),
                    self._shape_score(facts, "transition_riser"),
                    self._shape_score(facts, "transition_drop"),
                    self._shape_score(facts, "reverse_swell"),
                    self._shape_score(facts, "whoosh_sweep"),
                )
                >= 0.68
                and max(
                    self._subpanel_score(facts, "fx_reverse_score"),
                    self._subpanel_score(facts, "fx_riser_build_score"),
                    self._subpanel_score(facts, "fx_drop_downlifter_score"),
                    self._subpanel_score(facts, "fx_whoosh_sweep_score"),
                    self._subpanel_score(facts, "fx_transition_authority_score"),
                    self._subpanel_score(facts, "fx_motion_score"),
                )
                >= 0.46
            )
            slow_swell_tail_body = bool(
                self._shape_number(facts, "tail_ratio") >= 0.70
                and self._shape_number(facts, "attack_rise_time_norm") >= 0.08
                and self._shape_number(facts, "temporal_centroid_ratio") >= 0.34
                and self._shape_number(facts, "onset_count") <= 3.0
                and max(
                    self._shape_score(facts, "designed_low_fx"),
                    self._shape_score(facts, "transition_riser"),
                    self._shape_score(facts, "transition_drop"),
                    self._shape_score(facts, "reverse_swell"),
                    self._shape_score(facts, "whoosh_sweep"),
                )
                >= 0.68
                and max(fx_motion, fx_low, fx_tonal) >= 0.54
            )
            short_directional_stutter = bool(
                self._shape_number(facts, "duration_sec") <= 1.25
                and abs(self._shape_signed_number(facts, "centroid_slope_norm")) >= 0.12
                and self._shape_number(facts, "spectral_flatness_mean") >= 0.45
                and max(fx_motion, fx_low, fx_tonal) >= 0.58
            )
            long_glitch_formant_loop = bool(
                self._shape_number(facts, "tail_ratio") >= 0.55
                and self._shape_number(facts, "onset_span_ratio") >= 0.75
                and max(
                    self._shape_score(facts, "designed_tonal_fx"),
                    self._shape_score(facts, "designed_motion_fx_loop"),
                    self._shape_score(facts, "hybrid_fx_motion"),
                    self._shape_score(facts, "glitch_stutter"),
                )
                >= 0.70
                and max(
                    self._subpanel_score(facts, "fx_formant_score"),
                    self._subpanel_score(facts, "fx_glitch_stutter_score"),
                    self._subpanel_score(facts, "fx_radio_electrical_score"),
                )
                >= 0.62
                and self._shape_number(facts, "spectral_flatness_mean") >= 0.32
            )
            return bool(
                low_body
                or motion_tail_body
                or slow_swell_tail_body
                or short_directional_stutter
                or long_glitch_formant_loop
            )
        if shape == "designed_motion_fx_loop":
            return bool(
                fx_motion >= 0.46
                and max(
                    self._shape_score(facts, "hybrid_fx_motion"),
                    self._shape_score(facts, "whoosh_sweep"),
                    self._shape_score(facts, "glitch_stutter"),
                    self._shape_number(facts, "true_repetition_score"),
                )
                >= 0.58
            )
        return bool(
            fx_tonal >= 0.54
            and max(
                self._shape_number(facts, "pitch_confidence"),
                self._shape_number(facts, "pitched_event_ratio"),
                self._shape_score(facts, "siren_alarm_tone"),
                self._shape_score(facts, "hybrid_fx_motion"),
            )
            >= 0.58
        )

    def _facts_support_measured_fx_decoy_shape_body(
        self,
        facts: SharedAudioFacts | None,
        raw: ConsensusClaim,
    ) -> bool:
        """Return True when a non-FX primary shape is clearly a designed-FX decoy.

        Some FX clips have a primary shape that honestly reads as a texture,
        hit, beat, or pitched phrase.  The repair should not rename every one
        of those as FX.  It only fires when the secondary shape stack and the
        FX physics subpanels agree on designed sound behavior, while clean
        instrument/drum anchors are absent or contradicted by stronger FX
        evidence.
        """
        if facts is None:
            return False
        if self._facts_support_protected_loop_owner_over_fx(facts):
            return False
        shape = _shape_vote_from_facts(facts)
        if shape in DESIGNED_FX_SHAPES or shape in {
            "transition_riser",
            "transition_downlifter",
            "transition_drop",
            "whoosh_sweep",
            "reverse_swell",
            "hybrid_fx_motion",
        }:
            return False
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape_confidence < 0.70:
            return False
        fx_shape_pressure = max(
            self._shape_score(facts, "designed_low_fx"),
            self._shape_score(facts, "designed_motion_fx_loop"),
            self._shape_score(facts, "designed_tonal_fx"),
            self._shape_score(facts, "hybrid_fx_motion"),
            self._shape_score(facts, "whoosh_sweep"),
            self._shape_score(facts, "reverse_swell"),
            self._shape_score(facts, "transition_riser"),
            self._shape_score(facts, "transition_drop"),
            self._shape_score(facts, "siren_alarm_tone"),
            self._shape_score(facts, "impact_with_tail"),
        )
        fx_physics_pressure = max(
            self._subpanel_score(facts, "fx_motion_score"),
            self._subpanel_score(facts, "fx_transition_authority_score"),
            self._subpanel_score(facts, "fx_whoosh_sweep_score"),
            self._subpanel_score(facts, "fx_glitch_stutter_score"),
            self._subpanel_score(facts, "fx_reverse_score"),
            self._subpanel_score(facts, "fx_riser_build_score"),
            self._subpanel_score(facts, "fx_drop_downlifter_score"),
            self._subpanel_score(facts, "fx_radio_electrical_score"),
            self._subpanel_score(facts, "fx_impact_score"),
            self._subpanel_score(facts, "fx_boom_score"),
            self._subpanel_score(facts, "fx_formant_score"),
            self._subpanel_score(facts, "fx_siren_score"),
            self._subpanel_score(facts, "fx_alarm_score"),
            self._subpanel_score(facts, "fx_sub_hit_score"),
        )
        noisy_or_rough = max(
            self._shape_number(facts, "spectral_flatness_mean"),
            self._shape_number(facts, "spectral_entropy_mean"),
        )
        # Single-event noisy percussion can look like a texture bed to the
        # waveform shape pass.  Do not let that alone become FX unless there
        # is clear measured motion/transition ownership; this protects real
        # short percussion hits from the broader designed-FX texture rule.
        one_hit_texture_decoy = bool(
            shape in {"texture_bed", "noise_texture", "static_bed"}
            and self._shape_number(facts, "onset_count") <= 2.0
            and self._shape_number(facts, "onset_span_ratio") <= 0.12
            and self._shape_number(facts, "tail_ratio") <= 0.08
            and max(
                self._shape_score(facts, "designed_motion_fx_loop"),
                self._shape_score(facts, "transition_riser"),
                self._shape_score(facts, "transition_drop"),
                self._shape_score(facts, "whoosh_sweep"),
            )
            < 0.70
        )
        if one_hit_texture_decoy:
            return False

        strong_texture_motion = bool(
            shape in {"texture_bed", "noise_texture", "static_bed", "top_loop", "beat_loop"}
            and fx_shape_pressure >= 0.74
            and fx_physics_pressure >= 0.64
            and noisy_or_rough >= 0.50
        )
        strong_tonal_fx_decoy = bool(
            shape in {"solo_phrase", "pitched_repetition_phrase", "hit_with_tail", "single_hit", "echo_tail_hit"}
            and fx_shape_pressure >= 0.68
            and fx_physics_pressure >= 0.66
            and max(
                self._subpanel_score(facts, "fx_formant_score"),
                self._subpanel_score(facts, "fx_siren_score"),
                self._subpanel_score(facts, "fx_alarm_score"),
                self._subpanel_score(facts, "fx_reverse_score"),
                self._subpanel_score(facts, "fx_glitch_stutter_score"),
            )
            >= 0.62
        )
        strong_loop_fx_decoy = bool(
            shape in {"beat_loop", "top_loop"}
            and fx_shape_pressure >= 0.72
            and max(
                self._subpanel_score(facts, "fx_alarm_score"),
                self._subpanel_score(facts, "fx_glitch_stutter_score"),
                self._subpanel_score(facts, "fx_boom_score"),
                self._subpanel_score(facts, "fx_sub_hit_score"),
                self._subpanel_score(facts, "fx_formant_score"),
            )
            >= 0.56
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "f0_voiced_ratio") <= 0.25
        )
        if not (strong_texture_motion or strong_tonal_fx_decoy or strong_loop_fx_decoy):
            return False
        # Clean stable instruments should keep ownership.  This is deliberately
        # narrower than "is pitched" because many designed FX are tonal.
        instrument_identity = max(
            self._subpanel_score(facts, "synth_tonal_source_score"),
            self._subpanel_score(facts, "struck_keys_score"),
            self._subpanel_score(facts, "keys_tonal_decay_score"),
            self._subpanel_score(facts, "woodwind_sax_score"),
            self._subpanel_score(facts, "reed_wind_score"),
            self._subpanel_score(facts, "bass_synth_score"),
            self._subpanel_score(facts, "bass_sub_score"),
            self._subpanel_score(facts, "bass_electric_score"),
        )
        clean_stable_instrument = bool(
            instrument_identity >= 0.70
            and self._shape_number(facts, "pitched_event_ratio") >= 0.82
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.78
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.22
            and fx_physics_pressure < 0.70
        )
        if clean_stable_instrument:
            return False
        if self._facts_support_real_drum_material_over_designed_fx(facts):
            # A real drum anchor can still be overruled only by very strong FX
            # pressure.  This catches metal-plate/radio/formant designed crashes
            # without stealing ordinary cymbal/snare material.
            if not (fx_shape_pressure >= 0.80 and fx_physics_pressure >= 0.72):
                return False
        return True

    def _facts_support_real_drum_material_over_designed_fx(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for real drum material strong enough to block designed-FX rescue."""
        if facts is None:
            return False
        drum_material = max(
            self._subpanel_score(facts, "drum_kick_source_score"),
            self._subpanel_score(facts, "drum_snare_source_score"),
            self._subpanel_score(facts, "drum_clap_source_score"),
            self._subpanel_score(facts, "drum_cymbal_source_score"),
            self._subpanel_score(facts, "drum_tom_conga_source_score"),
            self._subpanel_score(facts, "drum_metallic_percussion_source_score"),
            self._subpanel_score(facts, "drum_rim_stick_source_score"),
            self._subpanel_score(facts, "drum_shaker_tambourine_source_score"),
        )
        percussive = max(
            self._shape_number(facts, "percussive_event_ratio"),
            self._shape_number(facts, "drumlike_frame_ratio"),
            self._subpanel_score(facts, "drum_hit_score"),
        )
        fx_motion = max(
            self._subpanel_score(facts, "fx_motion_score"),
            self._subpanel_score(facts, "fx_transition_authority_score"),
            self._subpanel_score(facts, "fx_whoosh_sweep_score"),
            self._subpanel_score(facts, "fx_drop_downlifter_score"),
            self._subpanel_score(facts, "fx_riser_build_score"),
            self._subpanel_score(facts, "fx_glitch_stutter_score"),
            self._subpanel_score(facts, "fx_reverse_score"),
            self._subpanel_score(facts, "fx_radio_electrical_score"),
        )
        return bool(drum_material >= 0.62 and percussive >= 0.50 and fx_motion < 0.58)

    def _designed_low_fx_folder(self, facts: SharedAudioFacts | None) -> str:
        """Map a designed-low shape that is really motion/tail FX to a broad FX parent."""
        if facts is None:
            return "FX/Hybrid Designed FX"
        slope = self._shape_signed_number(facts, "centroid_slope_norm")
        reverse = max(self._subpanel_score(facts, "fx_reverse_score"), self._shape_score(facts, "reverse_swell"))
        riser = max(self._subpanel_score(facts, "fx_riser_build_score"), self._shape_score(facts, "transition_riser"))
        drop = max(self._subpanel_score(facts, "fx_drop_downlifter_score"), self._shape_score(facts, "transition_drop"))
        whoosh = max(self._subpanel_score(facts, "fx_whoosh_sweep_score"), self._shape_score(facts, "whoosh_sweep"))
        if reverse >= 0.72 and abs(slope) < 0.10:
            return "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/Long FX"
        slow_swell_riser = bool(
            riser >= 0.70
            and self._shape_number(facts, "attack_rise_time_norm") >= 0.08
            and self._shape_number(facts, "temporal_centroid_ratio") >= 0.34
            and slope >= -0.04
            and drop <= riser + 0.10
        )
        if slow_swell_riser:
            return "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
        if slope >= 0.04 and max(riser, whoosh) >= 0.58:
            return "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
        if slope <= -0.04 and max(drop, whoosh) >= 0.58:
            return "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX"
        return "FX/Hybrid Designed FX"

    def _designed_motion_fx_loop_folder(self, facts: SharedAudioFacts | None) -> str:
        """Map a measured designed-motion loop shape to a broad FX parent.

        Pulsed sweeps and lasers often look like beat loops, but the sign and
        magnitude of the spectral-centroid slope says whether the energy is
        rising, falling, or sweeping.  Use that measured direction only after
        shape/FX support has already passed the producer guard.
        """
        slope = self._shape_signed_number(facts, "centroid_slope_norm") if facts is not None else 0.0
        drop = self._subpanel_score(facts, "fx_drop_downlifter_score") if facts is not None else 0.0
        riser = self._subpanel_score(facts, "fx_riser_build_score") if facts is not None else 0.0
        whoosh = self._subpanel_score(facts, "fx_whoosh_sweep_score") if facts is not None else 0.0
        if slope <= -0.22 and max(drop, whoosh) >= 0.54:
            return "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX"
        if slope >= 0.22 and max(riser, whoosh) >= 0.54:
            return "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
        if abs(slope) >= 0.22 and whoosh >= 0.58:
            return "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/Long FX"
        return "FX/Hybrid Designed FX"

    def _hybrid_motion_fx_folder(self, facts: SharedAudioFacts | None) -> str:
        """Map high-confidence hybrid motion to the safest broad FX folder."""
        if facts is None:
            return "FX/Hybrid Designed FX"
        slope = self._shape_signed_number(facts, "centroid_slope_norm")
        reverse = self._subpanel_score(facts, "fx_reverse_score")
        whoosh = self._subpanel_score(facts, "fx_whoosh_sweep_score")
        riser = self._subpanel_score(facts, "fx_riser_build_score")
        drop = self._subpanel_score(facts, "fx_drop_downlifter_score")
        glitch = self._subpanel_score(facts, "fx_glitch_stutter_score")
        if reverse >= 0.58:
            return "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/Long FX"
        if slope >= 0.16 and max(riser, whoosh) >= 0.46:
            return "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
        if slope <= -0.16 and max(drop, whoosh) >= 0.46:
            return "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX"
        if whoosh >= 0.54:
            return "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/Long FX"
        if glitch >= 0.60:
            return "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX"
        return "FX/Hybrid Designed FX"

    def _facts_support_clean_instrument_loop_over_designed_fx(
        self,
        facts: SharedAudioFacts | None,
        raw: ConsensusClaim,
    ) -> bool:
        """Return True when a designed-FX shape is probably just a real instrument loop."""
        if facts is None:
            return False
        if self._facts_support_protected_loop_owner_over_fx(facts):
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in DESIGNED_FX_SHAPES:
            return False
        strong_fx_candidate = raw.family == "FX" or self._shared_candidate_has_top_family(
            raw,
            DESIGNED_FX_CANDIDATE_FRAGMENTS,
            top_family="FX",
            max_score=16.0,
            max_brain_rank=8,
            max_physics_rank=8,
        )
        if strong_fx_candidate:
            return False
        clean_tonal = bool(
            max(
                self._shape_number(facts, "pitched_event_ratio"),
                self._shape_number(facts, "pitch_confidence"),
            )
            >= 0.84
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.74
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.18
            and self._subpanel_score(facts, "fx_motion_score") < 0.42
            and self._subpanel_score(facts, "fx_transition_authority_score") < 0.42
        )
        instrument_identity = max(
            self._subpanel_score(facts, "synth_tonal_source_score"),
            self._subpanel_score(facts, "struck_keys_score"),
            self._subpanel_score(facts, "keys_tonal_decay_score"),
            self._subpanel_score(facts, "woodwind_sax_score"),
            self._subpanel_score(facts, "reed_wind_score"),
            self._subpanel_score(facts, "bass_synth_score"),
            self._subpanel_score(facts, "bass_sub_score"),
            self._subpanel_score(facts, "bass_electric_score"),
        )
        return bool(clean_tonal and instrument_identity >= 0.66)

    def _facts_support_rhythmic_drum_loop_over_transition(self, facts: SharedAudioFacts) -> bool:
        """Return True when transition slope is really a rhythmic drum loop."""
        return bool(
            _shape_vote_from_facts(facts) in {"transition_riser", "transition_downlifter", "transition_drop"}
            and _shape_confidence_from_facts(facts) >= 0.80
            and self._shape_number(facts, "onset_count") >= 8.0
            and max(
                self._shape_number(facts, "onset_span_ratio"),
                self._shape_number(facts, "librosa_loop_confidence"),
            )
            >= 0.80
            and max(
                self._shape_number(facts, "pulse_regularity"),
                self._shape_number(facts, "librosa_loop_confidence"),
            )
            >= 0.38
            and max(
                self._shape_number(facts, "percussive_event_ratio"),
                self._shape_number(facts, "librosa_percussive_confidence"),
            )
            >= 0.18
            and self._subpanel_score(facts, "drum_hit_score") >= 0.40
            and self._subpanel_score(facts, "fx_transition_authority_score") < 0.45
            and self._subpanel_score(facts, "fx_motion_score") < 0.42
        )

    def _facts_support_measured_transition_body(self, facts: SharedAudioFacts) -> bool:
        """Return True for clear transition-FX motion."""
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {
            "transition_riser",
            "transition_downlifter",
            "transition_drop",
            "reverse_swell",
            "whoosh_sweep",
            "hybrid_fx_motion",
        }:
            return False
        if shape_confidence < 0.84:
            return False
        transition_authority = self._subpanel_score(facts, "fx_transition_authority_score")
        riser_score = self._subpanel_score(facts, "fx_riser_build_score")
        whoosh_score = self._subpanel_score(facts, "fx_whoosh_sweep_score")
        reverse_score = self._subpanel_score(facts, "fx_reverse_score")
        impact_score = self._subpanel_score(facts, "fx_impact_score")
        motion_score = self._subpanel_score(facts, "fx_motion_score")
        slope = max(
            abs(self._shape_signed_number(facts, "centroid_slope_norm")),
            abs(self._shape_signed_number(facts, "librosa_spectral_centroid_slope_norm")),
        )
        onset_span = self._shape_number(facts, "onset_span_ratio")
        sustained_tonal = max(
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
            self._shape_number(facts, "non_event_tonal_ratio"),
        )
        flatness = self._shape_number(facts, "spectral_flatness_mean")
        entropy = self._shape_number(facts, "spectral_entropy_mean")
        shape_motion = max(
            self._shape_score(facts, "whoosh_sweep"),
            self._shape_score(facts, "hybrid_fx_motion"),
            self._shape_score(facts, "designed_tonal_fx"),
            self._shape_score(facts, "designed_motion_fx_loop"),
        )
        motion_support = max(transition_authority, riser_score, whoosh_score, reverse_score, impact_score, motion_score)
        has_transition_motion = bool(
            slope >= 0.075
            or onset_span >= 0.45
            or max(transition_authority, riser_score, whoosh_score, reverse_score, impact_score) >= 0.58
        )
        # Some real transition FX are pitched/tonal for much of their body
        # (string scrapes, lasers, synth sweeps).  High tonal-frame coverage
        # should only block the transition owner when the body is also clean,
        # stable, and low-noise like a normal instrument phrase.
        clean_tonal_instrument_counter = bool(
            sustained_tonal >= 0.78
            and flatness <= 0.22
            and entropy <= 0.46
            and shape_motion < 0.58
            and max(transition_authority, whoosh_score, motion_score) < 0.58
        )
        noisy_or_motion_tonal_transition = bool(
            sustained_tonal > 0.58
            and shape_confidence >= 0.88
            and max(shape_motion, flatness, entropy, whoosh_score, reverse_score) >= 0.58
            and motion_support >= 0.56
        )
        return bool(
            motion_support >= 0.58
            and has_transition_motion
            and not clean_tonal_instrument_counter
            and (sustained_tonal <= 0.58 or noisy_or_motion_tonal_transition)
        )

    def _measured_physics_fx_role_branch(self, facts: SharedAudioFacts | None) -> str:
        """Return a trusted FX branch selected by the physics layer."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return ""
        layer = facts.evidence.get("physics_layer_decision")
        if not isinstance(layer, dict):
            return ""
        branch = str(layer.get("physics_layer_branch") or layer.get("fx_branch_selected") or "")
        if branch not in TRUSTED_FX_BRANCHES:
            return ""
        strength = self._safe_float(layer.get("fx_role_strength"), 0.0)
        conflict = self._safe_float(layer.get("fx_role_conflict_strength"), 1.0)
        top_family = str(layer.get("physics_layer_top_family") or "")
        top_confidence = self._safe_float(layer.get("physics_layer_top_confidence"), 0.0)
        allows_fx = bool(layer.get("fx_role_allows_fx"))
        if not allows_fx or conflict >= 0.56:
            return ""
        if branch == "BlipBeep" and self._facts_support_clean_tonal_blip_as_instrument_or_bell(facts):
            return ""
        if strength >= 0.70:
            return branch
        if top_family == "FX" and top_confidence >= 0.60 and strength >= 0.60:
            return branch
        return ""

    def _facts_support_learned_or_brain_fx_role_owner(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when brain top-family and physics role both own FX.

        This is intentionally broader than exact-leaf overlap.  Designed FX
        labels are heterogeneous, so Brain may say Impact while Physics says
        Siren/Alarm.  When both lanes agree on top-family FX and the physics
        layer has a trusted synthetic/role branch, the arbiter may choose a
        broad FX owner instead of releasing a weak Instrument leaf.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        layer = facts.evidence.get("physics_layer_decision")
        if not isinstance(layer, dict):
            return False
        if str(layer.get("physics_layer_top_family") or "") != "FX":
            return False
        if self._safe_float(layer.get("physics_layer_top_confidence"), 0.0) < 0.60:
            return False
        branch = str(layer.get("physics_layer_branch") or "")
        if branch not in TRUSTED_FX_BRANCHES:
            return False
        trusted_body = bool(
            layer.get("fx_synthetic_alert_fx_body")
            or layer.get("fx_role_allows_fx")
            or self._safe_float(layer.get("fx_role_strength"), 0.0) >= 0.66
        )
        if not trusted_body:
            return False
        return self._brain_ensemble_top_family(facts) == "FX"

    def _facts_support_clean_bass_loop_owner_over_fx(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when measured bass-loop ownership should block FX."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        if supports_clean_low_bass_phrase_owner(facts):
            return True
        brain_label = self._brain_ensemble_top_label(facts).lower()
        if not brain_label.startswith("instruments/bass/"):
            return False
        roles = facts.evidence.get("measured_roles", {})
        role_owner = max(
            role_strength(roles, "bass_loop") if isinstance(roles, dict) else 0.0,
            role_strength(roles, "pitched_music_loop") if isinstance(roles, dict) else 0.0,
            role_strength(roles, "low_rhythmic_drum_loop") if isinstance(roles, dict) else 0.0,
        )
        low_body = max(
            self._shape_number(facts, "low_event_ratio"),
            _feature_number_from_facts(facts, "low_total"),
        )
        pitch_confidence = max(
            self._shape_number(facts, "pitch_confidence"),
            _feature_number_from_facts(facts, "pitch_confidence"),
        )
        return bool(role_owner >= 0.62 and low_body >= 0.70 and pitch_confidence >= 0.70)

    @staticmethod
    def _brain_ensemble_top_family(facts: SharedAudioFacts) -> str:
        """Return the Brain ensemble top family from diagnostic evidence."""
        folder_path = MeasuredTransitionFxClaimProducer._brain_ensemble_top_label(facts)
        return folder_path.split("/", 1)[0] if folder_path else ""

    @staticmethod
    def _brain_ensemble_top_label(facts: SharedAudioFacts) -> str:
        """Return the Brain ensemble top label from diagnostic evidence."""
        evidence = facts.evidence if isinstance(getattr(facts, "evidence", None), dict) else {}
        vote = evidence.get("brain_ensemble_vote_1")
        if isinstance(vote, dict):
            folder_path = str(vote.get("folder_path") or vote.get("label") or "")
            if folder_path:
                return folder_path
            top = str(vote.get("top_family") or "")
            return top
        return str(vote or "")

    def _facts_support_clean_tonal_blip_as_instrument_or_bell(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when a short tonal blip lacks real FX motion evidence."""
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {"ui_blip", "solo_phrase", "single_hit", "hit_with_tail", "echo_tail_hit"}:
            return False
        pitched_event = self._shape_number(facts, "pitched_event_ratio")
        sustained_tonal = max(
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
            self._shape_number(facts, "non_event_tonal_ratio"),
        )
        motion = self._subpanel_score(facts, "fx_motion_score")
        transition = self._subpanel_score(facts, "fx_transition_authority_score")
        tonal_body = max(
            self._subpanel_score(facts, "synth_tonal_source_score"),
            self._subpanel_score(facts, "compact_struck_tonal_percussion_score"),
            self._subpanel_score(facts, "pitched_metal_percussion_score"),
            self._subpanel_score(facts, "pitched_mallet_instrument_score"),
            self._subpanel_score(facts, "struck_keys_score"),
        )
        percussive_body = max(
            self._subpanel_score(facts, "drum_snare_source_score"),
            self._subpanel_score(facts, "drum_clap_source_score"),
            self._subpanel_score(facts, "drum_kick_source_score"),
        )
        return bool(
            pitched_event >= 0.85
            and sustained_tonal >= 0.74
            and max(motion, transition) < 0.35
            and tonal_body >= 0.56
            and percussive_body < 0.62
        )

    @staticmethod
    def _measured_physics_fx_role_folder(branch: str) -> str:
        """Map measured FX role branches to broad output folders."""
        return {
            "RiserBuild": "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
            "DropDownlifter": "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
            "WhooshSweep": "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/Long FX",
            "ReverseSwell": "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/Long FX",
            "ImpactHit": "FX/Impacts and Hits/Generic Impact/Long FX",
            "GlitchStutter": "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots",
            "BlipBeep": "FX/Designed Noise FX/Blip/One Shots",
            "SirenAlarm": "FX/Designed Noise FX/Alarm/Long FX",
            "TextureAmbience": "FX/Textures/Drones and Atmospheres/Atmosphere/Long FX",
            "MachineMechanical": "FX/Everyday Foley/Machines/Generic Machine/Long FX",
            "FoleyMaterial": "FX/Everyday Foley/Keys Coins and Small Objects/Coins/One Shots",
            "SmallObjectCluster": "FX/Everyday Foley/Keys Coins and Small Objects/Coins/One Shots",
            "HumanCreatureFX": "FX/Human and Voice FX/Mouth Sounds/Long FX",
            "FormantFX": "FX/Human and Voice FX/Mouth Sounds/Long FX",
            "RadioElectrical": "FX/Designed Noise FX/Radio and Electrical/Long FX",
            "DesignedNoiseHybrid": "FX/Hybrid Designed FX",
        }.get(branch, "FX/Designed Noise FX/Blip/One Shots")

    @staticmethod
    def _measured_physics_fx_role_fragments(branch: str) -> tuple[str, ...]:
        """Return FX candidate fragments compatible with a measured branch."""
        return {
            "RiserBuild": ("riser", "build", "uplifter", "transition"),
            "DropDownlifter": ("drop", "downlifter", "downlift", "fall", "transition"),
            "WhooshSweep": ("sweep", "whoosh", "swoosh", "swish"),
            "ReverseSwell": ("reverse", "swell", "tail"),
            "ImpactHit": ("impact", "boom", "slam", "hit", "crash"),
            "GlitchStutter": ("glitch", "stutter", "digital", "buffer"),
            "BlipBeep": ("blip", "beep", "chirp", "zap"),
            "SirenAlarm": ("siren", "alarm"),
            "TextureAmbience": (
                "texture",
                "ambience",
                "atmosphere",
                "drone",
                "rain",
                "water",
                "wind",
                "fire",
                "thunder",
                "hiss",
                "static",
                "noise",
            ),
            "MachineMechanical": ("machine", "motor", "engine", "mechanical"),
            "FoleyMaterial": ("foley", "door", "footstep", "splash", "object", "coins", "keys"),
            "SmallObjectCluster": ("coins", "keys", "carkeys", "small object"),
            "HumanCreatureFX": (
                "human and voice",
                "mouth",
                "breath",
                "scream",
                "crowd",
                "applause",
                "animals",
                "bird",
                "dog",
                "cat",
                "cricket",
            ),
            "FormantFX": ("formant", "vowel", "mouth", "voice fx", "human and voice"),
            "RadioElectrical": ("radio", "electrical", "static", "buzz", "hum"),
            "DesignedNoiseHybrid": ("designed", "hybrid", "noise"),
        }.get(branch, ("fx",))

    def _parent_music_loop_release_should_block_transition(
        self,
        raw: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when parent music-loop evidence should beat transition shape."""
        if facts is None or raw.family not in {"FX", "_TO_REVIEW", "Instruments"}:
            return False
        parent = facts.evidence.get("parent_eligibility_v2", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(parent, dict):
            return False
        broad_folder = str(parent.get("broad_folder_path") or "")
        role_name = str(parent.get("role_name") or "")
        confidence = self._safe_float(parent.get("confidence"), 0.0)
        allowed = parent.get("allowed_top_families", [])
        if (
            not broad_folder.startswith("Instruments/")
            or "Instruments" not in allowed
            or role_name
            not in {
                "mixed_music_loop",
                "pitched_music_loop",
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
                "bass_loop",
            }
            or confidence < 0.60
        ):
            return False
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.18:
            return False
        pitched_support = max(
            self._shape_number(facts, "pitched_event_ratio"),
            self._shape_number(facts, "librosa_tonal_confidence"),
        )
        sustained_support = max(
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
            self._shape_number(facts, "librosa_harmonic_energy_ratio"),
        )
        return not (pitched_support < 0.45 and sustained_support < 0.35)

    def _facts_support_true_voice_role(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when measured role and shape support actual voice."""
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        roles = facts.evidence.get("measured_roles", {})
        if not isinstance(roles, dict):
            return False
        voice_strength = max(
            role_strength(roles, "vocal_music_phrase"),
            role_strength(roles, "vocal_phrase"),
            role_strength(roles, "vocal_one_shot"),
            role_strength(roles, "voiced_one_shot"),
        )
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        voice_identity = _feature_number_from_facts(facts, "formant_light_voice_identity")
        voice_panel = max(
            _feature_number_from_facts(facts, "voice_score"),
            _feature_number_from_facts(facts, "human_spoken_voice_score"),
            _feature_number_from_facts(facts, "human_breath_mouth_score"),
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
        )
        drum_hit = max(
            _feature_number_from_facts(facts, "drum_hit_score"),
            self._subpanel_score(facts, "drum_hit_score"),
        )
        processed_voice_loop = bool(
            shape in {"designed_low_fx", "designed_tonal_fx", "pitched_repetition_phrase", "repeated_phrase_loop"}
            and shape_confidence >= 0.72
            and voice_strength >= 0.68
            and voice_panel >= 0.74
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.58
            and self._shape_number(facts, "pitched_event_ratio") >= 0.55
            and self._shape_number(facts, "percussive_event_ratio") <= 0.18
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.18
            and drum_hit < 0.66
        )
        if processed_voice_loop:
            return True
        return bool(
            max(voice_strength, voice_identity, voice_panel) >= 0.46
            and shape in {"vocal_phrase", "hit_with_tail", "single_hit", "pitched_phrase"}
            and shape_confidence >= 0.70
            and drum_hit < 0.66
        )

    @staticmethod
    def _shared_candidate_has_top_family(
        raw: ConsensusClaim,
        fragments: tuple[str, ...],
        *,
        top_family: str,
        max_score: float = 32.0,
        max_brain_rank: int = 24,
        max_physics_rank: int = 24,
    ) -> bool:
        """Return True for a matching candidate row from a requested top family."""
        wanted = str(top_family or "").lower()
        for row in raw.shared_candidates or []:
            row_top = str(row.get("top_family") or "").lower()
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            if row_top and row_top != wanted:
                continue
            if not row_top and wanted and not path.startswith(f"{wanted}/"):
                continue
            if not any(fragment in path for fragment in fragments):
                continue
            score = MeasuredTransitionFxClaimProducer._safe_float(row.get("combined_rank_score"), 9999.0)
            brain_rank = MeasuredTransitionFxClaimProducer._safe_int(row.get("brain_rank"), 9999)
            physics_rank = MeasuredTransitionFxClaimProducer._safe_int(row.get("physics_rank"), 9999)
            if score <= max_score or brain_rank <= max_brain_rank or physics_rank <= max_physics_rank:
                return True
        return False

    @staticmethod
    def _shared_candidate_count_top_family(
        raw: ConsensusClaim,
        fragments: tuple[str, ...],
        *,
        top_family: str,
        max_score: float = 32.0,
        max_brain_rank: int = 24,
        max_physics_rank: int = 24,
    ) -> int:
        """Count matching candidate rows from a requested top family."""
        wanted = str(top_family or "").lower()
        count = 0
        for row in raw.shared_candidates or []:
            row_top = str(row.get("top_family") or "").lower()
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            if row_top and row_top != wanted:
                continue
            if not row_top and wanted and not path.startswith(f"{wanted}/"):
                continue
            if not any(fragment in path for fragment in fragments):
                continue
            score = MeasuredTransitionFxClaimProducer._safe_float(row.get("combined_rank_score"), 9999.0)
            brain_rank = MeasuredTransitionFxClaimProducer._safe_int(row.get("brain_rank"), 9999)
            physics_rank = MeasuredTransitionFxClaimProducer._safe_int(row.get("physics_rank"), 9999)
            if score <= max_score or brain_rank <= max_brain_rank or physics_rank <= max_physics_rank:
                count += 1
        return count

    @staticmethod
    def _subpanel_score(facts: SharedAudioFacts | None, name: str) -> float:
        """Return one physics-subpanel score from the flat evidence map."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return 0.0
        subpanels = facts.evidence.get("physics_subpanels")
        if isinstance(subpanels, dict):
            flat = subpanels.get("flat")
            if isinstance(flat, dict):
                return max(0.0, min(1.0, MeasuredTransitionFxClaimProducer._safe_float(flat.get(name), 0.0)))
        return 0.0

    @staticmethod
    def _shape_score(facts: SharedAudioFacts | None, shape_name: str) -> float:
        """Return one named score from ShapeVoter's ranked shape score list."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return 0.0
        container = facts.evidence.get("shape_vote")
        scores = container.get("shape_scores") if isinstance(container, dict) else None
        if not isinstance(scores, (list, tuple)):
            return 0.0
        wanted = str(shape_name)
        for item in scores:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == wanted:
                return max(0.0, min(1.0, MeasuredTransitionFxClaimProducer._safe_float(item[1], 0.0)))
        return 0.0

    @staticmethod
    def _shape_signed_number(facts: SharedAudioFacts | None, metric_name: str) -> float:
        """Return a signed ShapeVoter/feature metric.

        Most measured shape metrics are non-negative ratios, but directional
        motion metrics such as spectral-centroid slope are signed.  The generic
        _shape_number helper intentionally favors non-negative evidence, so it
        loses falling sweeps.  This helper preserves that sign for direction
        decisions only.
        """
        if facts is None:
            return 0.0
        shape_value = _shape_metric_from_facts(facts, metric_name)
        if shape_value != 0.0:
            return shape_value
        return _feature_number_from_facts(facts, metric_name)

    @staticmethod
    def _shape_number(facts: SharedAudioFacts, metric_name: str) -> float:
        """Return a ShapeVoter metric, falling back to named feature evidence."""
        return max(_shape_metric_from_facts(facts, metric_name), _feature_number_from_facts(facts, metric_name))

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value if value is not None else default)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(float(value if value is not None else default))
        except Exception:
            return int(default)
