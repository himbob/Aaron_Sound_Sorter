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
            self._protected_percussive_parent_claim,
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
            self._shape_number(facts, "librosa_loop_confidence"),
            self._shape_number(facts, "librosa_percussive_confidence"),
        )
        clean_top_hat_loop = bool(
            shape == "top_loop"
            and _shape_confidence_from_facts(facts) >= 0.78
            and self._shape_number(facts, "high_event_ratio") >= 0.80
            and max(
                self._shape_number(facts, "onset_count"),
                self._shape_number(facts, "librosa_onset_event_count"),
            )
            >= 12.0
            and self._subpanel_score(facts, "drum_closed_hat_source_score") >= 0.70
        )
        high_repetition_drum_loop = bool(
            measured_drum_loop >= 0.70
            and max(
                self._shape_number(facts, "true_repetition_score"),
                self._shape_number(facts, "librosa_loop_confidence"),
            )
            >= 0.70
            and max(
                self._shape_number(facts, "onset_count"),
                self._shape_number(facts, "librosa_onset_event_count"),
            )
            >= 8.0
            and self._subpanel_score(facts, "drum_closed_hat_source_score") >= 0.68
        )
        if not (clean_top_hat_loop or high_repetition_drum_loop):
            return None
        if self._designed_motion_shape_blocks_hat_loop(facts):
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
        shape = _shape_vote_from_facts(facts)
        if shape not in {"solo_phrase", "single_hit", "hit_with_tail", "echo_tail_hit", "bass_phrase"}:
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
        low_kick_like_hit = bool(
            shape == "bass_phrase"
            and duration <= 1.25
            and kick_score >= 0.66
            and sub_hit >= 0.58
            and one_shot >= 0.53
            and self._shape_number(facts, "low_event_ratio") >= 0.65
            and self._shape_number(facts, "onset_count") <= 3.0
            and max(
                self._subpanel_score(facts, "compact_struck_tonal_percussion_score"),
                self._subpanel_score(facts, "hand_drum_membrane_score"),
                self._subpanel_score(facts, "drum_hit_score"),
            )
            >= 0.50
        )
        sub_kick_hit = bool(
            duration <= 0.95
            and self._shape_number(facts, "onset_count") <= 3.0
            and self._shape_number(facts, "low_event_ratio") >= 0.78
            and self._shape_number(facts, "f0_voiced_ratio") <= 0.38
            and kick_score >= 0.58
            and sub_hit >= 0.58
        )
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        parent_low_kick_hit = bool(
            isinstance(parent, dict)
            and parent.get("role_name") == "low_kick_like_hit"
            and "Drums" in (parent.get("allowed_top_families") or [])
            and duration <= 1.35
            and self._shape_number(facts, "low_event_ratio") >= 0.72
            and max(
                self._role_value(facts, "percussive_one_shot"),
                one_shot,
            )
            >= 0.68
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
        if not (
            (kick_score >= 0.56 and sub_hit >= 0.55 and one_shot >= 0.58 and kick_candidate)
            or low_kick_like_hit
            or sub_kick_hit
            or parent_low_kick_hit
        ):
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

    def _protected_percussive_parent_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Emit parent-protected percussion before blocked-role review can fire.

        This is the lower-level counterpart to the late arbiter safety valve.
        Parent eligibility has already decided that the measured body belongs in
        the Drums lane and that Instruments/FX leaves are blocked.  When the
        body is a very short struck/noisy percussion hit, emit an explicit
        Drums claim here instead of asking the arbiter to rescue a review later.
        """
        raw = context.raw
        facts = context.facts
        if facts is None:
            return None
        if self._facts_support_decisive_struck_percussion_parent(facts):
            return None
        if not self._facts_support_parent_protected_percussive_hit(facts):
            return None
        target = self._measured_struck_percussion_parent_target(facts)
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_protected_percussive_parent_claim",
            reason=(
                "measured protected-percussion parent claim: parent eligibility, "
                "short one-shot structure, and drum material panels established "
                "a Drums lane before blocked role/shape review"
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

    def _facts_support_parent_protected_percussive_hit(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for measured parent-protected short percussion hits.

        This stays source-name blind and intentionally depends on lower-level
        measured evidence: parent eligibility, shape timing, and drum material
        subpanels.  It exists because some very short cymbal/guiro/metal hits
        are shaped as ``texture_bed`` by the shape voter due to noisy tails,
        even though the parent role and material panels correctly identify a
        protected percussion one-shot.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        parent = facts.evidence.get("parent_eligibility_v2", {})
        if not isinstance(parent, dict):
            return False
        if str(parent.get("role_name") or "") != "protected_percussive_one_shot":
            return False
        allowed = parent.get("allowed_top_families", [])
        blocked = parent.get("blocked_path_fragments", [])
        if "Drums" not in allowed or "Instruments" in allowed:
            return False
        if "FX" not in blocked and "Instruments" not in blocked:
            return False
        if facts.is_broken_or_tiny or facts.is_loop_like:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 0.95:
            return False
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        if event_count <= 0.0 or event_count > 3.0:
            return False
        if self._facts_support_short_true_voice_one_shot(facts) and not (
            self._facts_support_parent_protected_compact_struck_voice_decoy(facts)
            or self._facts_support_parent_protected_bright_metallic_voice_decoy(facts)
        ):
            return False
        if self._facts_support_clean_low_tonal_instrument_hit(facts):
            return False
        if self._facts_support_measured_transition_body(facts):
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {
            "single_hit",
            "hit_with_tail",
            "echo_tail_hit",
            "solo_phrase",
            "foley_action",
            "texture_bed",
            "noise_texture",
            "ui_blip",
        }:
            return False
        compact = self._subpanel_score(facts, "compact_struck_tonal_percussion_score")
        drum_branch = max(
            self._subpanel_score(facts, "drum_hit_score"),
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
        material = max(
            compact,
            self._subpanel_score(facts, "hand_drum_membrane_score"),
            self._subpanel_score(facts, "pitched_metal_percussion_score"),
            self._subpanel_score(facts, "struck_wood_score"),
        )
        fast_enough = self._shape_number(facts, "attack_rise_time_norm") <= 0.14
        early_enough = self._shape_number(facts, "temporal_centroid_ratio") <= 0.30
        noisy_short_hit = bool(
            shape in {"texture_bed", "noise_texture"}
            and duration <= 0.35
            and compact >= 0.64
            and drum_branch >= 0.62
            and fast_enough
            and early_enough
        )
        bright_noisy_tail_parent_hit = bool(
            shape in {"texture_bed", "noise_texture"}
            and duration <= 0.42
            and event_count <= 1.0
            and self._shape_number(facts, "drumlike_frame_ratio") >= 0.90
            and self._shape_number(facts, "percussive_event_ratio") >= 0.85
            and self._shape_number(facts, "pitched_event_ratio") <= 0.15
            and self._shape_number(facts, "high_event_ratio") >= 0.62
            and self._shape_number(facts, "tail_ratio") <= 0.58
            and compact >= 0.54
            and drum_branch >= 0.72
            and max(
                self._subpanel_score(facts, "drum_cymbal_source_score"),
                self._subpanel_score(facts, "drum_guiro_scrape_source_score"),
                self._subpanel_score(facts, "drum_metallic_percussion_source_score"),
            )
            >= 0.73
            and max(
                self._subpanel_score(facts, "woodwind_sax_score"),
                self._subpanel_score(facts, "reed_wind_score"),
            )
            < 0.62
        )
        bright_metallic_voice_decoy_parent_hit = bool(
            shape in {"texture_bed", "noise_texture", "hit_with_tail", "echo_tail_hit", "solo_phrase"}
            and duration <= 0.95
            and event_count <= 1.0
            and self._subpanel_score(facts, "role_one_shot_score") >= 0.90
            and self._shape_number(facts, "attack_rise_time_norm") <= 0.08
            and self._shape_number(facts, "temporal_centroid_ratio") <= 0.28
            and self._shape_number(facts, "tail_ratio") <= 0.35
            and compact >= 0.74
            and self._subpanel_score(facts, "onset_percussive_onset_score") >= 0.80
            and max(
                self._subpanel_score(facts, "drum_cymbal_source_score"),
                self._subpanel_score(facts, "drum_guiro_scrape_source_score"),
                self._subpanel_score(facts, "drum_metallic_percussion_source_score"),
            )
            >= 0.79
            and max(
                self._subpanel_score(facts, "pitched_metal_percussion_score"),
                self._subpanel_score(facts, "struck_wood_score"),
                self._subpanel_score(facts, "hand_drum_membrane_score"),
            )
            >= 0.62
        )
        normal_short_hit = bool(
            shape not in {"texture_bed", "noise_texture"}
            and material >= 0.66
            and drum_branch >= 0.50
            and fast_enough
            and early_enough
        )
        low_membrane_parent_hit = bool(
            shape in {"solo_phrase", "hit_with_tail", "echo_tail_hit", "single_hit"}
            and duration <= 0.45
            and event_count <= 2.0
            and self._subpanel_score(facts, "hand_drum_membrane_score") >= 0.76
            and compact >= 0.50
            and drum_branch >= 0.40
            and self._shape_number(facts, "low_event_ratio") >= 0.72
            and self._shape_number(facts, "attack_rise_time_norm") <= 0.20
            and self._shape_number(facts, "temporal_centroid_ratio") <= 0.35
            and max(
                self._subpanel_score(facts, "human_spoken_voice_score"),
                self._subpanel_score(facts, "human_breath_mouth_score"),
                self._subpanel_score(facts, "voice_score"),
            )
            <= 0.68
        )
        compact_struck_parent_hit = bool(
            shape in {"hit_with_tail", "echo_tail_hit", "single_hit"}
            and duration <= 0.30
            and event_count <= 1.0
            and compact >= 0.78
            and max(
                self._subpanel_score(facts, "struck_wood_score"),
                self._subpanel_score(facts, "hand_drum_membrane_score"),
                self._subpanel_score(facts, "pitched_metal_percussion_score"),
            )
            >= 0.62
            and self._subpanel_score(facts, "onset_percussive_onset_score") >= 0.72
            and max(
                _feature_number_from_facts(facts, "role_one_shot_score"),
                self._subpanel_score(facts, "role_one_shot_score"),
            )
            >= 0.90
            and self._role_value(facts, "percussive_one_shot") >= 0.80
            and self._role_value(facts, "voiced_one_shot") < 0.68
            and self._role_value(facts, "vocal_music_phrase") <= 0.05
            and self._shape_number(facts, "attack_rise_time_norm") <= 0.06
            and self._shape_number(facts, "temporal_centroid_ratio") <= 0.24
        )
        return (
            noisy_short_hit
            or bright_noisy_tail_parent_hit
            or bright_metallic_voice_decoy_parent_hit
            or normal_short_hit
            or low_membrane_parent_hit
            or compact_struck_parent_hit
        )

    def _facts_support_parent_protected_bright_metallic_voice_decoy(self, facts: SharedAudioFacts | None) -> bool:
        """True when voice/formant panels are decoys on a bright metallic percussion hit."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        parent = facts.evidence.get("parent_eligibility_v2", {})
        if not isinstance(parent, dict) or str(parent.get("role_name") or "") != "protected_percussive_one_shot":
            return False
        shape = _shape_vote_from_facts(facts)
        duration = _feature_number_from_facts(facts, "duration_sec")
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        return bool(
            shape in {"texture_bed", "noise_texture", "hit_with_tail", "echo_tail_hit", "solo_phrase"}
            and 0.30 <= duration <= 0.95
            and event_count <= 1.0
            and self._subpanel_score(facts, "role_one_shot_score") >= 0.90
            and self._shape_number(facts, "attack_rise_time_norm") <= 0.08
            and self._shape_number(facts, "temporal_centroid_ratio") <= 0.28
            and self._shape_number(facts, "tail_ratio") <= 0.35
            and self._subpanel_score(facts, "compact_struck_tonal_percussion_score") >= 0.74
            and self._subpanel_score(facts, "onset_percussive_onset_score") >= 0.80
            and max(
                self._subpanel_score(facts, "drum_cymbal_source_score"),
                self._subpanel_score(facts, "drum_guiro_scrape_source_score"),
                self._subpanel_score(facts, "drum_metallic_percussion_source_score"),
            )
            >= 0.79
            and max(
                self._subpanel_score(facts, "pitched_metal_percussion_score"),
                self._subpanel_score(facts, "struck_wood_score"),
                self._subpanel_score(facts, "hand_drum_membrane_score"),
            )
            >= 0.62
        )

    def _facts_support_parent_protected_compact_struck_voice_decoy(self, facts: SharedAudioFacts | None) -> bool:
        """True when voice-like panels are decoys on a compact struck parent hit."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        parent = facts.evidence.get("parent_eligibility_v2", {})
        if not isinstance(parent, dict) or str(parent.get("role_name") or "") != "protected_percussive_one_shot":
            return False
        if "Drums" not in parent.get("allowed_top_families", []):
            return False
        if "Voice" not in parent.get("blocked_path_fragments", []) and "Human" not in parent.get(
            "blocked_path_fragments", []
        ):
            return False
        return bool(
            _feature_number_from_facts(facts, "duration_sec") <= 0.30
            and max(
                _shape_metric_from_facts(facts, "onset_count"),
                _feature_number_from_facts(facts, "event_count_estimate"),
            )
            <= 1.0
            and _shape_vote_from_facts(facts) in {"hit_with_tail", "echo_tail_hit", "single_hit"}
            and self._subpanel_score(facts, "compact_struck_tonal_percussion_score") >= 0.78
            and max(
                self._subpanel_score(facts, "struck_wood_score"),
                self._subpanel_score(facts, "hand_drum_membrane_score"),
                self._subpanel_score(facts, "pitched_metal_percussion_score"),
            )
            >= 0.62
            and self._subpanel_score(facts, "onset_percussive_onset_score") >= 0.72
            and self._role_value(facts, "percussive_one_shot") >= 0.80
            and self._role_value(facts, "voiced_one_shot") < 0.68
            and self._role_value(facts, "vocal_music_phrase") <= 0.05
            and self._shape_number(facts, "attack_rise_time_norm") <= 0.06
            and self._shape_number(facts, "temporal_centroid_ratio") <= 0.24
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
            strength=max(0.94, raw.strength),
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
        struck_percussion_body = bool(
            one_shot_role >= 0.54
            and compact_struck >= 0.72
            and struck_material >= 0.78
            and max(onset_percussive, drum_hit, drum_branch) >= 0.50
        )
        pitched_music_stab_decoy = bool(
            not struck_percussion_body
            and _measured_role_from_facts(facts) == "pitched_music_phrase"
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
        # Freesound/percussion-pack one-shots often have pitched glass, wood,
        # metal, or membrane resonance.  Those can look like guitar/sax/synth
        # to clean-tonal APIs even though the physical event is still a struck
        # percussion hit.  Keep this source-name blind: require compact struck
        # material plus at least moderate drum/onset branch evidence.
        compact_material_hit = bool(
            one_shot_role >= 0.54
            and compact_struck >= 0.72
            and struck_material >= 0.56
            and (drum_branch >= 0.48 or struck_material >= 0.78)
            and max(onset_percussive, drum_hit, drum_branch) >= 0.50
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
        short_low_material_hit = bool(
            duration <= 0.40
            and event_count <= 3.0
            and compact_struck >= 0.82
            and max(struck_wood, hand_drum, pitched_metal) >= 0.72
            and self._shape_number(facts, "low_event_ratio") >= 0.65
            and self._shape_number(facts, "attack_rise_time_norm") <= 0.025
            and self._shape_number(facts, "temporal_centroid_ratio") <= 0.12
        )
        short_rhythmic_struck_phrase = bool(
            duration <= 1.25
            and event_count <= 8.0
            and shape
            in {"pitched_repetition_phrase", "repeated_phrase_loop", "beat_loop", "bass_phrase", "solo_phrase"}
            and onset_percussive >= 0.70
            and max(
                self._subpanel_score(facts, "rhythmic_break_loop_score"),
                self._shape_number(facts, "true_repetition_score"),
            )
            >= 0.42
            and max(struck_wood, compact_struck) >= 0.62
            and self._shape_number(facts, "low_event_ratio") >= 0.68
            and self._shape_number(facts, "attack_rise_time_norm") <= 0.03
            and voice_body <= 0.68
        )
        shape_hit = bool(
            shape
            in {
                "single_hit",
                "hit_with_tail",
                "echo_tail_hit",
                "solo_phrase",
                "pitched_repetition_phrase",
                "foley_action",
                "texture_bed",
            }
            or (shape == "pitched_phrase" and struck_percussion_body and event_count <= 6.0)
            or (shape == "bass_phrase" and compact_material_hit and duration <= 1.50 and event_count <= 8.0)
            or (shape == "ui_blip" and compact_struck >= 0.82 and struck_material >= 0.80 and drum_branch >= 0.54)
        )
        return bool(
            shape_hit
            and (
                compact_material_hit
                or tonal_bell_hit
                or membrane_hit
                or metallic_hit
                or scrape_hit
                or short_low_material_hit
                or short_rhythmic_struck_phrase
            )
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
        compact_struck = self._subpanel_score(facts, "compact_struck_tonal_percussion_score")
        struck_material = max(
            self._subpanel_score(facts, "pitched_metal_percussion_score"),
            self._subpanel_score(facts, "struck_wood_score"),
            self._subpanel_score(facts, "hand_drum_membrane_score"),
        )
        short_low_struck_material = bool(
            compact_struck >= 0.82
            and struck_material >= 0.72
            and self._shape_number(facts, "low_event_ratio") >= 0.65
            and self._shape_number(facts, "attack_rise_time_norm") <= 0.025
            and self._shape_number(facts, "temporal_centroid_ratio") <= 0.12
        )
        if short_low_struck_material:
            return False
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

    def _shape_score(self, facts: SharedAudioFacts | None, name: str) -> float:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return 0.0
        shape_vote = facts.evidence.get("shape_vote")
        if not isinstance(shape_vote, dict):
            return 0.0
        scores = shape_vote.get("shape_scores")
        if isinstance(scores, (list, tuple)):
            for item in scores:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == name:
                    return self._safe_float(item[1])
        return 0.0

    def _designed_motion_shape_blocks_hat_loop(self, facts: SharedAudioFacts | None) -> bool:
        """Block hat-loop claims for rhythmic sweeps/lasers with strong motion.

        Some designed FX loops are full of bright onsets, so the closed-hat
        panel fires.  Do not let that claim own the parent when the shape stack
        shows a broad sweep/stutter body with a large spectral-motion slope.
        """
        motion_shape = max(
            self._shape_score(facts, "whoosh_sweep"),
            self._shape_score(facts, "designed_motion_fx_loop"),
            self._shape_score(facts, "hybrid_fx_motion"),
            self._shape_score(facts, "transition_riser"),
            self._shape_score(facts, "transition_drop"),
            self._shape_score(facts, "reverse_swell"),
            self._shape_score(facts, "glitch_stutter"),
        )
        fx_panel = max(
            self._subpanel_score(facts, "fx_glitch_stutter_score"),
            self._subpanel_score(facts, "fx_radio_electrical_score"),
            self._subpanel_score(facts, "fx_whoosh_sweep_score"),
            self._subpanel_score(facts, "fx_motion_score"),
            self._subpanel_score(facts, "fx_transition_authority_score"),
            self._subpanel_score(facts, "fx_reverse_score"),
            self._subpanel_score(facts, "fx_siren_score"),
            self._subpanel_score(facts, "fx_alarm_score"),
        )
        slope = abs(self._shape_number(facts, "centroid_slope_norm"))
        pulse = self._shape_number(facts, "pulse_regularity")
        return bool(motion_shape >= 0.78 and fx_panel >= 0.70 and slope >= 0.35 and pulse <= 0.48)

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
