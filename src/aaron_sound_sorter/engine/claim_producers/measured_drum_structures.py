# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured drum/percussion structure claim producer.

This producer moves compact drum/percussion one-shot and high-band hat-loop
repairs out of the final arbiter chain.  The claims stay structure-first: they
use measured ShapeVoter, role, physics-subpanel, and candidate evidence before
final arbitration, not source names or late post-winner rescue.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _feature_number_from_facts,
    _measured_role_from_facts,
    _norm_path,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path
from aaron_sound_sorter.voters.scoring_tools import role_strength

DRUM_CANDIDATE_FRAGMENTS = (
    "drum",
    "kick",
    "snare",
    "tom",
    "percussion",
    "clap",
    "hat",
    "cymbal",
    "rim",
    "stick",
    "conga",
    "bongo",
    "tabla",
    "bell",
    "metallic",
    "world",
)


class MeasuredDrumStructureClaimProducer:
    """Emit measured drum-structure claims before final arbitration."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return conservative measured drum claims in specificity order."""
        claims: list[ConsensusClaim] = []
        for builder in (
            self._hi_hat_loop_claim,
            self._kick_one_shot_claim,
            self._decisive_struck_percussion_parent_claim,
            self._strong_drum_one_shot_claim,
        ):
            claim = builder(context)
            if claim is not None:
                claims.append(claim)
        return claims

    def _hi_hat_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Preserve measured high-band closed-hat loops before broad loop routing."""
        raw = context.raw
        facts = context.facts
        raw_path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Drums" and ("hi hat" in raw_path or "hi-hat" in raw_path):
            return None
        shape = _shape_vote_from_facts(facts)
        measured_drum_loop = max(
            self._role_value(facts, "bright_drum_loop"),
            self._role_value(facts, "drum_loop"),
            self._role_value(facts, "percussive_drum_loop"),
            self._subpanel_score(facts, "drum_loop_source_score"),
        )
        clean_top_hat_loop = bool(
            shape == "top_loop"
            and _shape_confidence_from_facts(facts) >= 0.78
            and self._shape_number(facts, "high_event_ratio") >= 0.80
            and self._shape_number(facts, "onset_count") >= 12.0
            and self._subpanel_score(facts, "drum_closed_hat_source_score") >= 0.70
        )
        high_repetition_drum_loop = bool(
            measured_drum_loop >= 0.70
            and self._shape_number(facts, "true_repetition_score") >= 0.70
            and self._shape_number(facts, "onset_count") >= 8.0
            and self._subpanel_score(facts, "drum_closed_hat_source_score") >= 0.68
        )
        if not (clean_top_hat_loop or high_repetition_drum_loop):
            return None
        target = "Drums/Hi Hats/Closed Hat/Loops" if clean_top_hat_loop else "Drums/Drum Loops/Loops"
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_hi_hat_loop_invariant",
            reason=(
                "measured hi-hat loop claim: top-loop/high-band repetition and "
                "closed-hat panel evidence established drum loop structure before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(0.93, raw.strength),
            is_real_candidate=False,
        )

    def _kick_one_shot_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Release measured short sub/kick one-shots before review/FX can win."""
        raw = context.raw
        facts = context.facts
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 1.35:
            return None
        if _shape_vote_from_facts(facts) not in {"solo_phrase", "single_hit", "hit_with_tail", "echo_tail_hit"}:
            return None
        if self._shape_number(facts, "onset_count") > 3.0:
            return None
        kick_score = self._subpanel_score(facts, "drum_kick_source_score")
        sub_hit = self._subpanel_score(facts, "fx_sub_hit_score")
        one_shot = max(
            _feature_number_from_facts(facts, "role_one_shot_score"),
            self._subpanel_score(facts, "role_one_shot_score"),
            self._shape_number(facts, "one_shot_score"),
        )
        kick_candidate = self._facts_have_internal_candidate(
            facts,
            ("kick",),
            top_family="Drums",
            max_rank=6,
            max_score=1.5,
        ) or self._has_candidate_support(
            raw,
            ("kick",),
            top_family="Drums",
            max_score=14.0,
            max_brain_rank=6,
            max_physics_rank=6,
        )
        if not (kick_score >= 0.56 and sub_hit >= 0.55 and one_shot >= 0.58 and kick_candidate):
            return None
        target = self._best_physics_drum_one_shot_folder(facts) or "Drums/Kick Drums/Generic Kick/One Shots"
        if "kick" not in target.lower():
            target = "Drums/Kick Drums/Generic Kick/One Shots"
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_kick_one_shot_invariant",
            reason=(
                "measured kick one-shot claim: sub/kick transient evidence and drum candidate support "
                "were established before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(0.94, raw.strength),
            is_real_candidate=True,
        )

    def _decisive_struck_percussion_parent_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Keep decisive compact struck bodies under a measured Drums parent."""
        raw = context.raw
        facts = context.facts
        if raw.family == "Drums" or facts is None:
            return None
        if not self._facts_support_decisive_struck_percussion_parent(facts):
            return None
        return claim_from_folder_path(
            folder_path=self._measured_struck_percussion_parent_target(facts),
            source="final_decisive_struck_percussion_parent_invariant",
            reason=(
                "measured struck-percussion parent claim: compact membrane/wood/metal/scrape evidence "
                "kept the sound under Drums before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(0.91, raw.strength),
            is_real_candidate=True,
        )

    def _strong_drum_one_shot_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Protect strong measured drum one-shots from non-drum fallback claims."""
        raw = context.raw
        facts = context.facts
        if not self._facts_support_strong_drum_one_shot(facts):
            return None
        if not (
            self._has_candidate_support(raw, DRUM_CANDIDATE_FRAGMENTS, top_family="Drums")
            or self._facts_have_internal_candidate(
                facts, DRUM_CANDIDATE_FRAGMENTS, top_family="Drums", max_rank=6, max_score=8.0
            )
        ):
            return None
        target = self._best_physics_drum_one_shot_folder(facts) or self._measured_struck_percussion_parent_target(facts)
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_drum_one_shot_invariant",
            reason=(
                "measured drum one-shot claim: strong drum-anchor evidence was emitted before "
                "older voice/bass/pitched-hit fallbacks could win"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(0.92, raw.strength),
            is_real_candidate=True,
        )

    def _facts_support_decisive_struck_percussion_parent(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 2.75:
            return False
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"), _feature_number_from_facts(facts, "event_count_estimate")
        )
        if facts.is_loop_like and duration > 1.25 and event_count > 5.0:
            return False
        if event_count > 6.0 and duration > 0.70:
            return False
        if self._facts_support_short_true_voice_one_shot(facts):
            return False
        if self._facts_support_clean_low_tonal_instrument_hit(facts):
            return False
        compact_struck = self._subpanel_score(facts, "compact_struck_tonal_percussion_score")
        hand_drum = self._subpanel_score(facts, "hand_drum_membrane_score")
        pitched_metal = self._subpanel_score(facts, "pitched_metal_percussion_score")
        struck_wood = self._subpanel_score(facts, "struck_wood_score")
        struck_material = max(hand_drum, pitched_metal, struck_wood)
        drum_branch = max(
            self._subpanel_score(facts, "drum_kick_source_score"),
            self._subpanel_score(facts, "drum_snare_source_score"),
            self._subpanel_score(facts, "drum_clap_source_score"),
            self._subpanel_score(facts, "drum_tom_conga_source_score"),
            self._subpanel_score(facts, "drum_rim_stick_source_score"),
            self._subpanel_score(facts, "drum_cymbal_source_score"),
            self._subpanel_score(facts, "drum_guiro_scrape_source_score"),
            self._subpanel_score(facts, "drum_metallic_percussion_source_score"),
            self._subpanel_score(facts, "drum_shaker_tambourine_source_score"),
        )
        onset_percussive = self._subpanel_score(facts, "onset_percussive_onset_score")
        drum_hit = self._subpanel_score(facts, "drum_hit_score")
        one_shot_role = self._subpanel_score(facts, "role_one_shot_score")
        shape = _shape_vote_from_facts(facts)
        pitched_music_stab_decoy = bool(
            _measured_role_from_facts(facts) == "pitched_music_phrase"
            and self._role_value(facts, "pitched_music_phrase") >= 0.70
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and max(
                self._shape_number(facts, "sustained_tonal_frame_ratio"),
                self._shape_number(facts, "non_event_tonal_ratio"),
            )
            >= 0.86
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and pitched_metal < 0.60
            and max(
                self._subpanel_score(facts, "drum_kick_source_score"),
                self._subpanel_score(facts, "drum_snare_source_score"),
                self._subpanel_score(facts, "drum_clap_source_score"),
            )
            < 0.66
        )
        if pitched_music_stab_decoy:
            return False
        voice_body = max(
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
        )
        if voice_body >= 0.64 and voice_body >= max(struck_material, drum_branch) + 0.08:
            return False
        compact_material_hit = bool(
            one_shot_role >= 0.58
            and compact_struck >= 0.74
            and struck_material >= 0.56
            and drum_branch >= 0.50
            and max(onset_percussive, drum_hit, drum_branch) >= 0.52
        )
        tonal_bell_hit = bool(
            one_shot_role >= 0.56
            and compact_struck >= 0.70
            and pitched_metal >= 0.60
            and self._subpanel_score(facts, "pitched_mallet_instrument_score") >= 0.70
            and self._subpanel_score(facts, "fx_motion_score") < 0.35
            and self._subpanel_score(facts, "fx_transition_authority_score") < 0.35
            and self._subpanel_score(facts, "drum_kick_source_score") < 0.56
        )
        membrane_hit = bool(one_shot_role >= 0.56 and hand_drum >= 0.72 and drum_branch >= 0.58)
        metallic_hit = bool(
            one_shot_role >= 0.56
            and pitched_metal >= 0.62
            and max(
                self._subpanel_score(facts, "drum_cymbal_source_score"),
                self._subpanel_score(facts, "drum_metallic_percussion_source_score"),
            )
            >= 0.62
        )
        scrape_hit = bool(
            one_shot_role >= 0.52
            and self._subpanel_score(facts, "drum_guiro_scrape_source_score") >= 0.72
            and compact_struck >= 0.60
        )
        shape_hit = shape in {
            "single_hit",
            "hit_with_tail",
            "echo_tail_hit",
            "solo_phrase",
            "foley_action",
            "texture_bed",
        }
        return bool(
            shape_hit and (compact_material_hit or tonal_bell_hit or membrane_hit or metallic_hit or scrape_hit)
        )

    def _facts_support_strong_drum_one_shot(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 2.75:
            return False
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"), _feature_number_from_facts(facts, "event_count_estimate")
        )
        if facts.is_loop_like and duration > 1.25 and event_count > 4.0:
            return False
        decisive_drum_panel = bool(
            self._subpanel_score(facts, "drum_hit_score") >= 0.70
            and max(
                self._subpanel_score(facts, "drum_snare_source_score"),
                self._subpanel_score(facts, "drum_clap_source_score"),
                self._subpanel_score(facts, "drum_tom_conga_source_score"),
                self._subpanel_score(facts, "drum_rim_stick_source_score"),
                self._subpanel_score(facts, "drum_cymbal_source_score"),
                self._subpanel_score(facts, "drum_closed_hat_source_score"),
            )
            >= 0.72
            and _shape_vote_from_facts(facts) in {"foley_action", "hit_with_tail", "single_hit"}
            and event_count <= 3.0
            and not (
                self._facts_support_short_true_voice_one_shot(facts)
                and self._subpanel_score(facts, "drum_hit_score") < 0.70
            )
            and not self._facts_support_measured_transition_body(facts)
        )
        if decisive_drum_panel:
            return True
        if self._facts_support_clean_low_tonal_instrument_hit(facts) or self._facts_support_non_drum_voiced_phrase_hit(
            facts
        ):
            return False
        if self._facts_support_measured_transition_body(facts):
            return False
        if self._facts_support_material_struck_drum_one_shot(facts):
            return True
        direct_panel_drum_one_shot = bool(
            self._subpanel_score(facts, "drum_hit_score") >= 0.70
            and self._shape_number(facts, "drumlike_frame_ratio") >= 0.70
            and self._shape_number(facts, "percussive_event_ratio") >= 0.70
            and self._shape_number(facts, "onset_count") <= 3.0
            and _shape_vote_from_facts(facts) in {"foley_action", "hit_with_tail", "single_hit"}
            and not self._facts_support_measured_transition_body(facts)
        )
        if direct_panel_drum_one_shot:
            return True
        guess = self._top_physics_drum_guess(facts)
        if not guess:
            return False
        evidence = guess.get("evidence") if isinstance(guess, dict) else {}
        if not isinstance(evidence, dict):
            evidence = {}
        anchor = self._safe_float(evidence.get("drum_anchor_strength"))
        branch = self._safe_float(evidence.get("drum_branch_selected_confidence"))
        role = str(evidence.get("detected_parent_role") or "")
        percussive_role = role in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}
        physics_confidence = self._safe_float(guess.get("confidence"))
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        allowed = parent.get("allowed_top_families", []) if isinstance(parent, dict) else []
        parent_role_name = str(parent.get("role_name") or "") if isinstance(parent, dict) else ""
        if (
            parent_role_name in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}
            and "Drums" in allowed
            and "Instruments" not in allowed
            and anchor >= 0.50
            and branch >= 0.46
        ):
            return True
        if anchor >= 0.66 and (branch >= 0.50 or percussive_role):
            return True
        if anchor >= 0.58 and branch >= 0.58 and _shape_vote_from_facts(facts) in {"single_hit", "hit_with_tail"}:
            return True
        if physics_confidence >= 0.65 and anchor >= 0.55 and branch >= 0.45 and facts.is_short_hit_like:
            return True
        return bool(
            percussive_role
            and anchor >= 0.54
            and _shape_confidence_from_facts(facts) >= 0.64
            and facts.is_short_hit_like
        )

    def _facts_support_material_struck_drum_one_shot(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        voice_panel = max(
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
        )
        if shape not in {"single_hit", "hit_with_tail", "echo_tail_hit", "solo_phrase", "foley_action"}:
            return False
        struck_material = max(
            self._subpanel_score(facts, "hand_drum_membrane_score"),
            self._subpanel_score(facts, "pitched_metal_percussion_score"),
            self._subpanel_score(facts, "struck_wood_score"),
        )
        material_flag = bool(
            self._subpanel_bool(facts, "hand_drum_material_evidence")
            or self._subpanel_bool(facts, "pitched_metal_material_evidence")
            or self._subpanel_bool(facts, "struck_wood_material_evidence")
            or self._subpanel_bool(facts, "struck_percussion_guard_exception")
        )
        drum_source = max(
            self._subpanel_score(facts, "drum_hit_score"),
            self._subpanel_score(facts, "drum_tom_conga_source_score"),
            self._subpanel_score(facts, "drum_rim_stick_source_score"),
            self._subpanel_score(facts, "drum_metallic_percussion_source_score"),
            self._subpanel_score(facts, "drum_cymbal_source_score"),
        )
        if not (voice_panel < 0.58 and material_flag and struck_material >= 0.56 and drum_source >= 0.48):
            return False
        if self._facts_have_physics_drum_candidate(facts, max_rank=3, max_score=8.0):
            return True
        return self._facts_have_internal_candidate(
            facts, DRUM_CANDIDATE_FRAGMENTS, top_family="Drums", max_rank=6, max_score=8.0
        )

    def _facts_support_clean_low_tonal_instrument_hit(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        top_path = self._top_physics_guess_path(facts)
        if not (top_path.startswith("instruments/bass/") or top_path.startswith("instruments/synths/")):
            return False
        if _shape_vote_from_facts(facts) not in {
            "bass_phrase",
            "pitched_phrase",
            "solo_phrase",
            "hit_with_tail",
            "single_hit",
        }:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 1.75:
            return False
        low_tonal_source = max(
            self._subpanel_score(facts, "bass_sub_score"),
            self._subpanel_score(facts, "bass_synth_score"),
            self._subpanel_score(facts, "low_end_source_score"),
            self._subpanel_score(facts, "synth_tonal_source_score"),
            self._subpanel_score(facts, "physics_subpanel_clean_tone"),
        )
        drum_body = max(
            self._subpanel_score(facts, "drum_hit_score"),
            self._subpanel_score(facts, "drum_kick_source_score"),
            self._subpanel_score(facts, "drum_snare_source_score"),
            self._subpanel_score(facts, "drum_clap_source_score"),
            self._subpanel_score(facts, "drum_tom_conga_source_score"),
            self._subpanel_score(facts, "drum_rim_stick_source_score"),
            self._subpanel_score(facts, "drum_cymbal_source_score"),
        )
        material_flags = bool(
            self._subpanel_bool(facts, "pitched_metal_material_evidence")
            or self._subpanel_bool(facts, "hand_drum_material_evidence")
            or self._subpanel_bool(facts, "struck_wood_material_evidence")
            or self._subpanel_bool(facts, "struck_percussion_guard_exception")
        )
        return bool(
            low_tonal_source >= 0.60
            and drum_body <= 0.52
            and not material_flags
            and self._shape_number(facts, "pitched_event_ratio") >= 0.85
            and self._shape_number(facts, "pitch_confidence") >= 0.70
            and self._shape_number(facts, "low_event_ratio") >= 0.70
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and self._subpanel_score(facts, "physics_subpanel_noisy_air") <= 0.30
        )

    def _facts_support_non_drum_voiced_phrase_hit(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        voice = max(
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
        )
        return bool(voice >= 0.68 and voice >= self._subpanel_score(facts, "drum_hit_score") + 0.08)

    def _facts_support_short_true_voice_one_shot(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        voice = max(
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
        )
        return bool(
            _feature_number_from_facts(facts, "duration_sec") <= 1.6
            and voice >= 0.66
            and self._role_value(facts, "voiced_one_shot") >= 0.55
        )

    def _facts_support_measured_transition_body(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        return bool(
            _shape_vote_from_facts(facts)
            in {
                "transition_riser",
                "transition_downlifter",
                "transition_drop",
                "reverse_swell",
                "whoosh_sweep",
                "hybrid_fx_motion",
            }
            and _shape_confidence_from_facts(facts) >= 0.84
            and max(
                self._subpanel_score(facts, "fx_transition_authority_score"),
                self._subpanel_score(facts, "fx_riser_build_score"),
                self._subpanel_score(facts, "fx_whoosh_sweep_score"),
                self._subpanel_score(facts, "fx_reverse_score"),
                self._subpanel_score(facts, "fx_impact_score"),
                self._subpanel_score(facts, "fx_motion_score"),
            )
            >= 0.58
        )

    def _measured_struck_percussion_parent_target(self, facts: SharedAudioFacts) -> str:
        kick = self._subpanel_score(facts, "drum_kick_source_score")
        snare = self._subpanel_score(facts, "drum_snare_source_score")
        clap = self._subpanel_score(facts, "drum_clap_source_score")
        tom = self._subpanel_score(facts, "drum_tom_conga_source_score")
        rim = self._subpanel_score(facts, "drum_rim_stick_source_score")
        cymbal = self._subpanel_score(facts, "drum_cymbal_source_score")
        guiro = self._subpanel_score(facts, "drum_guiro_scrape_source_score")
        metallic = self._subpanel_score(facts, "drum_metallic_percussion_source_score")
        shaker = self._subpanel_score(facts, "drum_shaker_tambourine_source_score")
        measured_short_low_kick = bool(
            _feature_number_from_facts(facts, "duration_sec") <= 0.75
            and self._shape_number(facts, "onset_count") <= 4.0
            and self._shape_number(facts, "low_event_ratio") >= 0.88
            and self._shape_number(facts, "high_event_ratio") <= 0.08
            and self._shape_number(facts, "f0_voiced_ratio") <= 0.12
            and kick >= 0.52
        )
        if measured_short_low_kick or (kick >= 0.66 and kick >= max(tom, cymbal, guiro, metallic) + 0.06):
            return "Drums/Kick Drums/Generic Kick/One Shots"
        if (
            self._subpanel_score(facts, "pitched_metal_percussion_score") >= 0.60
            and self._subpanel_score(facts, "pitched_mallet_instrument_score") >= 0.70
            and self._subpanel_score(facts, "compact_struck_tonal_percussion_score") >= 0.70
            and kick < 0.56
        ):
            return "Drums/Percussion/Bells and Metallic Percussion/One Shots"
        if snare >= 0.76 and clap >= 0.70 and max(snare, clap) >= max(rim, tom, metallic) - 0.08:
            return (
                "Drums/Claps Snaps Slaps/Generic Clap/One Shots"
                if clap > snare + 0.04
                else "Drums/Snares/Generic Snare/One Shots"
            )
        if snare >= 0.70 and snare >= max(clap, rim, tom) + 0.04:
            return "Drums/Snares/Generic Snare/One Shots"
        if clap >= 0.70 and clap >= max(snare, rim) + 0.04:
            return "Drums/Claps Snaps Slaps/Generic Clap/One Shots"
        if guiro >= 0.72:
            return "Drums/Percussion/Guiros Scrapes and Rasps/One Shots"
        if rim >= 0.68 or self._subpanel_score(facts, "struck_wood_score") >= 0.82:
            return "Drums/Rims and Sticks/Generic Rim or Stick/One Shots"
        if cymbal >= 0.70 or metallic >= 0.70:
            return "Drums/Cymbals/Generic Cymbal/One Shots"
        if tom >= 0.66 or self._subpanel_score(facts, "hand_drum_membrane_score") >= 0.76:
            return "Drums/Toms/Generic Tom/One Shots"
        if shaker >= 0.70:
            return "Drums/Percussion/Shakers and Tambourines/One Shots"
        return "Drums/Percussion/Generic Percussion/One Shots"

    def _best_physics_drum_one_shot_folder(self, facts: SharedAudioFacts | None) -> str:
        guess = self._top_physics_drum_guess(facts)
        if not guess:
            return ""
        path = str(guess.get("folder_path") or guess.get("label") or "").strip("/")
        if not path.lower().startswith("drums/"):
            return ""
        if "drum loops" in path.lower():
            return "Drums/Percussion/Generic Percussion/One Shots"
        return path

    def _top_physics_guess_path(self, facts: SharedAudioFacts | None) -> str:
        result = (
            facts.evidence.get("physics_vote_result")
            if facts is not None and isinstance(getattr(facts, "evidence", None), dict)
            else None
        )
        if isinstance(result, dict):
            guesses = result.get("top_guesses")
            if isinstance(guesses, list) and guesses:
                first = guesses[0]
                if isinstance(first, dict):
                    return _norm_path(str(first.get("folder_path") or first.get("label") or ""))
        return ""

    def _top_physics_drum_guess(self, facts: SharedAudioFacts | None) -> dict | None:
        result = (
            facts.evidence.get("physics_vote_result")
            if facts is not None and isinstance(getattr(facts, "evidence", None), dict)
            else None
        )
        guesses = result.get("top_guesses") if isinstance(result, dict) else None
        if not isinstance(guesses, list):
            return None
        for fallback_rank, guess in enumerate(guesses[:6], start=1):
            if not isinstance(guess, dict):
                continue
            path = _norm_path(str(guess.get("folder_path") or guess.get("label") or ""))
            top = str(guess.get("top_family") or "").lower()
            rank = self._safe_int(guess.get("rank"), fallback_rank)
            score = self._safe_float(guess.get("score"), float(rank + 1))
            if rank <= 6 and score <= 8.0 and (top == "drums" or path.startswith("drums/")):
                return guess
        return None

    def _facts_have_physics_drum_candidate(
        self, facts: SharedAudioFacts | None, *, max_rank: int, max_score: float
    ) -> bool:
        result = (
            facts.evidence.get("physics_vote_result")
            if facts is not None and isinstance(getattr(facts, "evidence", None), dict)
            else None
        )
        guesses = result.get("top_guesses") if isinstance(result, dict) else None
        if not isinstance(guesses, list):
            return False
        for fallback_rank, guess in enumerate(guesses[: max(1, max_rank)], start=1):
            if not isinstance(guess, dict):
                continue
            path = _norm_path(str(guess.get("folder_path") or guess.get("label") or ""))
            top = str(guess.get("top_family") or "").lower()
            if top != "drums" and not path.startswith("drums/"):
                continue
            rank = self._safe_int(guess.get("rank"), fallback_rank)
            score = self._safe_float(guess.get("score"), float(rank + 1))
            if rank <= max_rank and score <= max_score:
                return True
        return False

    def _facts_have_internal_candidate(
        self,
        facts: SharedAudioFacts | None,
        fragments: tuple[str, ...],
        *,
        top_family: str,
        max_rank: int,
        max_score: float,
    ) -> bool:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        for key in ("physics_vote_result", "brain_vote_result", "candidate_vote_result"):
            result = facts.evidence.get(key)
            guesses = result.get("top_guesses") if isinstance(result, dict) else None
            if not isinstance(guesses, list):
                continue
            for fallback_rank, guess in enumerate(guesses[: max(1, max_rank)], start=1):
                if not isinstance(guess, dict):
                    continue
                path = _norm_path(str(guess.get("folder_path") or guess.get("label") or ""))
                family = str(guess.get("top_family") or "") or (path.split("/", 1)[0].title() if path else "")
                if family != top_family and not path.startswith(top_family.lower() + "/"):
                    continue
                if not any(fragment in path for fragment in fragments):
                    continue
                rank = self._safe_int(guess.get("rank"), fallback_rank)
                score = self._safe_float(guess.get("score"), float(rank + 1))
                if rank <= max_rank and score <= max_score:
                    return True
        return False

    def _has_candidate_support(
        self,
        raw: ConsensusClaim,
        fragments: tuple[str, ...],
        *,
        top_family: str,
        max_score: float = 36.0,
        max_brain_rank: int = 8,
        max_physics_rank: int = 12,
    ) -> bool:
        for row in raw.shared_candidates or []:
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            family = str(row.get("top_family") or "") or (path.split("/", 1)[0].title() if path else "")
            if family != top_family and not path.startswith(top_family.lower() + "/"):
                continue
            if not any(fragment in path for fragment in fragments):
                continue
            combined = self._safe_float(row.get("combined_rank_score"), self._safe_float(row.get("score"), 9999.0))
            brain_rank = self._safe_int(row.get("brain_rank"), 999)
            physics_rank = self._safe_int(row.get("physics_rank"), 999)
            if combined <= max_score and brain_rank <= max_brain_rank and physics_rank <= max_physics_rank:
                return True
        return False

    def _subpanel_score(self, facts: SharedAudioFacts | None, name: str) -> float:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return 0.0
        panels = facts.evidence.get("physics_subpanels", {})
        flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
        value = flat.get(name) if isinstance(flat, dict) else None
        return self._safe_float(value)

    def _subpanel_bool(self, facts: SharedAudioFacts | None, name: str) -> bool:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        panels = facts.evidence.get("physics_subpanels", {})
        flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
        return bool(flat.get(name)) if isinstance(flat, dict) else False

    def _shape_number(self, facts: SharedAudioFacts | None, metric_name: str) -> float:
        return max(_shape_metric_from_facts(facts, metric_name), _feature_number_from_facts(facts, metric_name))

    def _role_value(self, facts: SharedAudioFacts | None, role_name: str) -> float:
        roles = (
            facts.evidence.get("measured_roles", {})
            if facts is not None and isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        return role_strength(roles if isinstance(roles, dict) else {}, role_name)

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            number = float(value if value is not None else default)
        except Exception:
            return float(default)
        return 0.0 if number != number else number

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(float(value if value is not None else default))
        except Exception:
            return int(default)
