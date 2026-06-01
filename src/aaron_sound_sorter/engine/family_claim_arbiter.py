# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Final claim arbitration for Aaron Sound Sorter."""

from __future__ import annotations

import os
from pathlib import Path

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.claim_contracts import contracts_from_claims
from aaron_sound_sorter.engine.decision_helpers import (
    _candidate_combined_score,
    _feature_number_from_facts,
    _measured_role_from_facts,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
    _has_drum_loop_structure_support,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path, review_claim
from aaron_sound_sorter.engine.placement_resolver import PlacementResolver
from aaron_sound_sorter.engine.voter_calibration_panels import VoterCalibrationPanel, build_voter_calibration_panels
from aaron_sound_sorter.voters.scoring_tools import detected_parent_role_name, role_strength


def _debug_packet_summary(facts: SharedAudioFacts | None) -> list[str]:
    """Return source-blind shared-evidence debug lines if present."""
    if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
        return []
    packet = facts.evidence.get("audio_analysis_packet_summary")
    cache = facts.evidence.get("audio_analysis_cache_stats")
    lines: list[str] = []
    if isinstance(packet, dict):
        lines.append(
            "EVIDENCE_PACKET "
            f"policy={packet.get('cache_policy', 'unknown')} "
            f"persistent={packet.get('persistent_disk_cache', False)} "
            f"stores_final={packet.get('stores_final_decisions', False)} "
            f"audio={packet.get('audio_status', 'unknown')} "
            f"direct={packet.get('direct_body_status', 'unknown')} "
            f"wetness={float(packet.get('wetness_score', 0.0) or 0.0):.3f} "
            f"harmonic={packet.get('harmonic_status', 'unknown')}"
        )
    if isinstance(cache, dict):
        lines.append(
            "CACHE_STATS "
            f"policy={cache.get('policy', 'unknown')} "
            f"persistent={cache.get('persistent_disk_cache', False)} "
            f"hits={int(cache.get('hits', 0) or 0)} "
            f"misses={int(cache.get('misses', 0) or 0)} "
            f"items={int(cache.get('total_items', 0) or 0)}"
        )
    family_policy = facts.evidence.get("top_family_order_policy")
    if isinstance(family_policy, dict):
        lines.append(
            "FAMILY_ORDER_POLICY "
            f"name={family_policy.get('policy_name', 'unknown')} "
            f"order={','.join(str(x) for x in family_policy.get('review_order', []))} "
            f"changes_scores={family_policy.get('changes_scores', False)}"
        )
    return lines


class FamilyClaimArbiter:
    """The only normal engine component that emits final ConsensusDecision objects."""

    SAME_FAMILY_OVERRIDE_MARGIN = 6.0
    CROSS_FAMILY_OVERRIDE_MARGIN = 4.0
    REVIEW_MIN_STRENGTH = 0.72
    REVIEW_ALTERNATIVE_MARGIN = 0.25
    GENERIC_BROAD_BUCKETS = {
        ("Instruments", "Instrument Loops"),
        ("FX", "Human and Voice FX"),
        ("Drums", "Kick One Shot"),
    }
    ROLE_SHAPE_ROUTING_SOURCES = {
        "shape_sanity_consensus",
        "top_family_sanity_consensus",
        "role_sanity_consensus",
        "role_sanity_broad_bucket",
        "same_family_role_broad_bucket",
        "role_sanity_generic_pitched_broad_instrument_loop",
        "parent_eligibility_broad_bucket",
        "placement_depth_family_rescue_broad_bucket",
        "sustained_pitched_instrument_broad_bucket",
        "candidate_true_bucket_rescue",
        "ambiguous_fx_music_loop_broad_bucket",
    }
    PROFILE_SHORTCUT_SOURCES = {
        "profile_candidate_kick_claim",
        "baby_recall_kick_claim",
    }
    IDENTITY_RESCUE_SOURCE_PREFIXES = (
        "profile_candidate_",
        "baby_recall_",
    )
    IDENTITY_RESCUE_SOURCE_MARKERS = (
        "rescue",
        "shortcut",
        "sibling_conflict",
        "role_sanity",
        "shape_sanity",
        "top_family_sanity",
        "parent_eligibility",
        "placement_depth_family_rescue",
        "sustained_pitched_instrument_broad_bucket",
        "ambiguous_fx_music_loop_broad_bucket",
    )
    CONCRETE_FX_AUTHORITY_FRAGMENTS = (
        "siren",
        "alarm",
        "beep",
        "blip",
        "glitch",
        "stutter",
        "impact",
        "boom",
        "slam",
        "riser",
        "build",
        "drop",
        "downlifter",
        "whoosh",
        "sweep",
        "reverse",
        "radio",
        "electrical",
        "motor",
        "machine",
        "formant",
    )
    BRANCH_LOOP_TARGETS = {
        "KeysPiano": "Instruments/Keys/Electric Piano/Loops",
        "Synth": "Instruments/Synths/Synth Loops",
        "PluckedString": "Instruments/Guitar/Guitar Loops",
        "Strings": "Instruments/Strings/Loops",
        "MixedInstrument": "Instruments/Instrument Loops/Loops",
        "Woodwinds": "Instruments/Brass and Woodwinds/Loops",
        "Brass": "Instruments/Brass and Woodwinds/Loops",
    }
    PATH_BRANCH_HINTS = (
        (("sax", "saxophone", "woodwind", "flute", "clarinet", "bassoon"), "Woodwinds"),
        (("brass", "horn", "trumpet", "trombone"), "Brass"),
        (("keys", "piano", "rhodes", "electric piano"), "KeysPiano"),
        (("synth", "lead", "pad"), "Synth"),
        (("guitar", "pluck"), "PluckedString"),
        (("strings", "string", "violin", "cello", "viola"), "Strings"),
    )

    def __init__(self, placement_resolver: PlacementResolver | None = None) -> None:
        self.placement_resolver = placement_resolver or PlacementResolver()

    def adjudicate(
        self,
        *,
        raw_claim: ConsensusClaim,
        consensus_claims: list[ConsensusClaim],
        eligibility_claims: list[ConsensusClaim],
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusDecision:
        """Choose the final claim, resolve its path, and create the decision."""
        all_claims = consensus_claims + eligibility_claims
        winning_claim = self.pick_winner(raw_claim=raw_claim, claims=all_claims, facts=facts)
        winning_claim = self._protect_measured_drum_one_shot_from_false_invariant(
            raw_claim,
            winning_claim,
            facts,
        )
        winning_claim = self._release_shape_review_to_broad_bucket(winning_claim, facts)
        winning_claim = self._protect_measured_drum_one_shot_from_false_invariant(
            raw_claim,
            winning_claim,
            facts,
        )
        winning_claim = self._protect_measured_transition_fx_from_review_or_false_leaf(winning_claim, facts)
        winning_claim = self._release_clean_tonal_non_drum_hit_to_fx(winning_claim, facts)
        winning_claim = self._protect_measured_hi_hat_loop_from_percussion_leaf(winning_claim, facts)
        winning_claim = self._protect_measured_drum_loop_from_fx_or_instrument(winning_claim, facts)
        winning_claim = self._protect_clean_tonal_instrument_phrase_from_drum_leaf(winning_claim, facts)
        winning_claim = self._review_pitched_music_hit_stolen_by_drum_leaf(winning_claim, facts)
        winning_claim = self._protect_measured_drum_one_shot_from_false_invariant(
            raw_claim,
            winning_claim,
            facts,
        )
        winning_claim = self._protect_measured_voice_from_non_voice_leaf(winning_claim, facts)
        winning_claim = self._protect_measured_drum_one_shot_from_false_invariant(
            raw_claim,
            winning_claim,
            facts,
        )
        winning_claim = self._protect_measured_bass_loop_from_fx_or_review(winning_claim, facts)
        winning_claim = self._protect_measured_drum_one_shot_from_false_invariant(
            raw_claim,
            winning_claim,
            facts,
        )
        winning_claim = self._protect_measured_synth_loop_from_fx_or_review(winning_claim, facts)
        winning_claim = self._preserve_measured_instrument_branch_after_broadening(winning_claim, facts)
        winning_claim = self._protect_measured_synth_loop_from_fx_or_review(winning_claim, facts)
        winning_claim = self._protect_clean_keys_loop_from_sax_or_fx(winning_claim, facts)
        winning_claim = self._protect_compound_music_loop_from_specific_instrument_leaf(winning_claim, facts)
        winning_claim = self._protect_measured_synth_loop_from_fx_or_review(winning_claim, facts)
        winning_claim = self._protect_measured_sax_loop_from_fx_or_review(winning_claim, facts)
        winning_claim = self._protect_clean_keys_loop_from_sax_or_fx(winning_claim, facts)
        winning_claim = self._protect_clean_pitched_instrument_loop_from_fx_leaf(winning_claim, facts)
        winning_claim = self._review_conflicted_instrument_identity_leaf(winning_claim, facts)
        winning_claim = self._protect_short_instrument_one_shot_from_loop_bucket(winning_claim, facts)
        winning_claim = self._protect_instrument_one_shot_leaf_from_measured_loop(winning_claim, facts)
        winning_claim = self._protect_short_instrument_one_shot_from_loop_bucket(winning_claim, facts)
        winning_claim = self._preserve_measured_instrument_branch_after_broadening(winning_claim, facts)
        winning_claim = self._review_uncertain_generic_loop_release(winning_claim, facts)
        winning_claim = self._protect_short_instrument_one_shot_from_loop_bucket(winning_claim, facts)
        winning_claim = self._review_non_transition_pitched_loop_stolen_by_fx_leaf(winning_claim, facts)
        winning_claim = self._protect_measured_transition_fx_from_review_or_false_leaf(winning_claim, facts)
        winning_claim = self._rehome_true_voice_from_fx_bucket(winning_claim, facts)
        winning_claim = self._protect_short_true_voice_one_shot_from_loop_bucket(winning_claim, facts)
        winning_claim = self._protect_measured_drum_one_shot_from_false_invariant(
            raw_claim,
            winning_claim,
            facts,
        )
        winning_claim = self._enforce_final_invariant_consensus_firewall(
            raw_claim,
            winning_claim,
            facts,
        )
        winning_claim = self._protect_measured_mixed_loop_from_sax_overreach(winning_claim, facts)
        winning_claim = self._protect_measured_rhythmic_break_loop(winning_claim, facts)
        winning_claim = self._protect_measured_kick_one_shot(winning_claim, facts)
        winning_claim = self._protect_measured_tonal_chord_stab(winning_claim, facts)
        debug_path = os.environ.get("AARON_DEBUG_CLAIMS_FILE")
        if debug_path:
            debug_claims = [raw_claim, winning_claim] + all_claims
            lines = [
                f"RAW {raw_claim.family}|{raw_claim.sub_family}|{raw_claim.strength}|{raw_claim.source}|{raw_claim.folder_path}",
                f"WIN {winning_claim.family}|{winning_claim.sub_family}|{winning_claim.strength}|{winning_claim.source}|{winning_claim.folder_path}",
            ]
            for contract in contracts_from_claims(debug_claims):
                lines.append(contract.debug_line())
            lines.extend(_debug_packet_summary(facts))
            for panel in build_voter_calibration_panels(claims=debug_claims, facts=facts):
                lines.append(panel.debug_line())
            for claim in all_claims:
                allowed = self._claim_can_compete(raw_claim, claim, facts=facts)
                lines.append(
                    f"CLAIM allowed={allowed} {claim.family}|{claim.sub_family}|{claim.strength}|"
                    f"{claim.can_override}|{claim.source}|{claim.raw_candidate_score}|"
                    f"{claim.is_real_candidate}|{claim.folder_path}"
                )
            Path(debug_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        folder_path = self.placement_resolver.resolve(winning_claim)
        return ConsensusDecision(
            final_label=winning_claim.final_label,
            final_top=winning_claim.family,
            folder_path=folder_path,
            consensus_status=winning_claim.source,
            reason=winning_claim.reason,
            shared_winner=winning_claim.shared_winner,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            combined_rank_score=winning_claim.raw_candidate_score,
            shared_candidates=winning_claim.shared_candidates,
        )

    def _protect_measured_drum_one_shot_from_false_invariant(
        self,
        raw_claim: ConsensusClaim,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Keep strong measured drum one-shots out of old voice/bass/pitch invariants.

        This is a final safety firewall, not a filename rescue.  It reads only
        measured shape facts plus PhysicsVoter/brain candidates already stored
        in shared evidence.  It is intentionally limited to short, non-loop,
        drum-anchored events so true basses, vocals, pads, and long FX do not get
        dragged into Drums.
        """
        if facts is None or not self._facts_support_strong_drum_one_shot(facts):
            return winning_claim
        if winning_claim.family == "Drums":
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        source = str(winning_claim.source or "")
        false_invariant = source in {
            "parent_eligibility_review",
            "final_shape_review_bass_loop_invariant",
            "final_pitched_hit_broad_instrument_release_invariant",
            "final_pitched_hit_broad_instrument_loop_invariant",
            "final_shape_review_broad_instrument_loop_invariant",
            "final_measured_voice_invariant",
        }
        false_family = bool(
            winning_claim.family in {"Instruments", "_TO_REVIEW"}
            or (winning_claim.family == "FX" and not self._facts_support_measured_transition_fx(facts, winning_claim))
            or "voice" in path
            or "vocal" in path
            or "bass loops" in path
            or "instrument loops" in path
        )
        if not (false_invariant or false_family):
            return winning_claim
        if not self._facts_or_candidates_support_drum_one_shot(facts, raw_claim, winning_claim):
            return winning_claim
        target = self._best_physics_drum_one_shot_folder(facts)
        if not target:
            target = (
                raw_claim.folder_path
                if raw_claim.family == "Drums"
                else "Drums/Percussion/Generic Percussion/One Shots"
            )
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_drum_one_shot_invariant",
            reason=(
                "final measured drum one-shot invariant kept strong PhysicsVoter "
                "drum-anchor evidence out of older voice/bass/pitched-hit fallback"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.92, winning_claim.strength),
            is_real_candidate=True,
        )

    def _protect_measured_mixed_loop_from_sax_overreach(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Broaden layered musical loops that were over-specialized as sax.

        Sax/reed authority is useful for obvious solo sax, but it is not enough
        when the measured body is a long repeated pitched loop and synth, keys,
        voice, or other instrument panels sit in the same confidence region.
        That is a mixed musical loop, not a terminal woodwind leaf.
        """
        if facts is None or winning_claim.family != "Instruments":
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if not any(token in path for token in ("sax", "woodwind", "brass")):
            return winning_claim
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {
            "pitched_phrase",
            "pitched_phrase_shape",
            "repeated_phrase_loop",
            "bass_phrase",
            "sustained_pad",
            "vocal_phrase",
        }:
            return winning_claim
        event_count = self._shape_number(facts, "onset_count")
        repeat = max(
            self._shape_number(facts, "true_repetition_score"),
            self._shape_number(facts, "onset_true_repetition_likelihood"),
            self._shape_number(facts, "loop_tempo_confidence"),
            self._shape_number(facts, "loop_pulse_clarity"),
        )
        if shape_confidence < 0.82 or event_count < 12.0 or repeat < 0.42:
            return winning_claim
        sax_score = max(
            self._measured_score(facts, "woodwind_sax_score"),
            self._measured_score(facts, "reed_wind_score"),
            self._measured_score(facts, "reed_wind_authority_score"),
        )
        competing_identity = max(
            self._measured_score(facts, "synth_tonal_source_score", "synth_lead_score", "synth_chord_score", "synth_pad_score"),
            self._measured_score(facts, "struck_keys_score", "struck_keys_authority_score"),
            self._measured_score(facts, "voice_score", "voice_choir_score"),
            self._measured_score(facts, "plucked_string_score", "plucked_string_authority_score"),
            self._compound_music_loop_support(facts),
        )
        physics_top = self._top_physics_guess_path(facts).lower()
        broad_loop_candidate = self._facts_have_internal_candidate(
            facts,
            ("instrument loops", "mixed musical", "multi instrument"),
            top_family="Instruments",
            max_rank=6,
            max_score=1.5,
        ) or self._shared_candidate_has_top_family(
            winning_claim,
            ("instrument loops", "mixed musical", "multi instrument"),
            top_family="Instruments",
            max_score=14.0,
            max_brain_rank=8,
            max_physics_rank=6,
        )
        sax_is_decisive = bool(sax_score >= 0.74 and sax_score >= competing_identity + 0.10)
        if sax_is_decisive and "sax" in physics_top:
            return winning_claim
        layer = self._physics_layer(facts)
        if (
            str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
            in {"Woodwinds", "ReedWoodwind"}
            and self._safe_number(
                layer.get("instrument_branch_selected_confidence")
                or layer.get("physics_layer_branch_confidence"),
                0.0,
            ) >= 0.84
            and (
                bool(layer.get("instrument_woodwind_source_signal"))
                or bool(layer.get("instrument_low_mid_wet_sax_signal"))
                or bool(layer.get("instrument_reed_woodwind_source_signal"))
            )
        ):
            return winning_claim
        clean_struck_keys_chord_loop = bool(
            _shape_vote_from_facts(facts) in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad"}
            and _shape_confidence_from_facts(facts) >= 0.84
            and self._piano_struck_physics_witness_score(facts) >= 0.66
            and self._safe_number(layer.get("instrument_KeysPiano_subpanel_confidence"), 0.0) >= 0.80
            and self._safe_number(layer.get("instrument_KeysPiano_subpanel_margin"), 0.0) >= 0.18
            and self._safe_number(layer.get("instrument_panel_KeysPiano_ElectricPiano"), 0.0) >= 0.78
            and self._shape_number(facts, "pitched_event_ratio") >= 0.92
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.88
            and 0.18 <= self._shape_number(facts, "low_event_ratio") <= 0.40
            and 0.55 <= self._shape_number(facts, "mid_event_ratio") <= 0.72
            and self._shape_number(facts, "high_event_ratio") <= 0.025
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.012
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and not bool(layer.get("instrument_woodwind_source_signal"))
            and not bool(layer.get("instrument_clean_tonal_reed_solo_signal"))
        )
        if clean_struck_keys_chord_loop:
            return claim_from_folder_path(
                folder_path="Instruments/Keys/Electric Piano/Loops",
                source="final_clean_struck_keys_chord_loop_invariant",
                reason=(
                    "final measured-loop invariant preserved a clean struck-keys "
                    "chord loop before sax-overreach broadening"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.93, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if competing_identity >= max(0.52, sax_score - 0.08) or broad_loop_candidate:
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="final_mixed_loop_sax_overreach_broadening",
                reason=(
                    "final measured-loop invariant broadened a sax/woodwind leaf "
                    "because repeated musical-loop evidence had competing non-reed "
                    "instrument identity panels"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.92, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        return winning_claim

    def _protect_measured_rhythmic_break_loop(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Keep measured rhythmic break loops under Drums/Drum Loops."""
        if facts is None:
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if winning_claim.family == "Drums" and "drum loops" in path:
            return winning_claim
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration < 1.35:
            return winning_claim
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {"beat_loop", "drum_loop", "top_loop", "bass_phrase", "repeated_phrase_loop"}:
            return winning_claim
        event_count = self._shape_number(facts, "onset_count")
        repeat = max(
            self._shape_number(facts, "true_repetition_score"),
            self._shape_number(facts, "onset_true_repetition_likelihood"),
            self._shape_number(facts, "loop_tempo_confidence"),
        )
        pulse = max(
            self._shape_number(facts, "loop_pulse_clarity"),
            self._shape_number(facts, "loop_onset_periodicity"),
            self._shape_number(facts, "pulse_regularity"),
        )
        percussive_event = max(
            self._shape_number(facts, "percussive_event_ratio"),
            self._measured_score(facts, "onset_percussive_onset_score"),
        )
        drum_family = max(
            self._measured_score(facts, "drum_loop_source_score"),
            self._measured_score(facts, "drum_hit_score"),
            self._measured_score(facts, "drum_kick_source_score"),
            self._measured_score(facts, "drum_snare_source_score"),
            self._measured_score(facts, "drum_closed_hat_source_score"),
            self._measured_score(facts, "drum_cymbal_source_score"),
        )
        candidate_supported = self._facts_have_internal_candidate(
            facts,
            ("drum loops", "drum loop"),
            top_family="Drums",
            max_rank=12,
            max_score=6.0,
        ) or self._shared_candidate_has_top_family(
            winning_claim,
            ("drum loops", "drum loop"),
            top_family="Drums",
            max_score=24.0,
            max_brain_rank=12,
            max_physics_rank=12,
        )
        nonpercussive_pitched_loop = bool(
            shape in {"bass_phrase", "repeated_phrase_loop", "pitched_phrase"}
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "pitch_confidence") >= 0.62
            and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.08
            and _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.10
            and self._measured_role_value(facts, "pitched_music_loop") >= 0.74
        )
        if nonpercussive_pitched_loop:
            if self._facts_support_clean_bass_loop(facts, winning_claim):
                return claim_from_folder_path(
                    folder_path="Instruments/Bass/Bass Loops",
                    source="final_nonpercussive_bass_loop_guard",
                    reason=(
                        "final rhythmic-loop invariant preserved a measured "
                        "clean bass phrase under Bass instead of broad Instrument Loops"
                    ),
                    shared=winning_claim.shared_candidates,
                    raw_candidate_score=winning_claim.raw_candidate_score,
                    brain_rank=winning_claim.brain_rank,
                    physics_rank=winning_claim.physics_rank,
                    shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                    can_override=True,
                    strength=max(0.94, winning_claim.strength),
                    is_real_candidate=winning_claim.is_real_candidate,
                )
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="final_nonpercussive_pitched_loop_guard",
                reason=(
                    "final rhythmic-loop invariant preserved a nonpercussive "
                    "pitched music loop under Instruments instead of Drum Loops"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.92, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        rhythmic_body = bool(
            shape_confidence >= 0.84
            and event_count >= 8.0
            and repeat >= 0.40
            and pulse >= 0.32
            and (percussive_event >= 0.40 or drum_family >= 0.46 or candidate_supported)
        )
        if not rhythmic_body:
            return winning_claim
        if self._facts_support_true_voice_role(facts) and percussive_event < 0.55 and not candidate_supported:
            return winning_claim
        return claim_from_folder_path(
            folder_path="Drums/Drum Loops/Loops",
            source="final_measured_rhythmic_break_loop_invariant",
            reason=(
                "final rhythmic-loop invariant kept repeated percussive/break "
                "structure under Drum Loops despite conflicting FX/instrument lanes"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.93, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_measured_kick_one_shot(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Release short sub/kick one-shots from review or false bass/impact leaves."""
        if facts is None:
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if winning_claim.family == "Drums" and "kick" in path:
            return winning_claim
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 1.35:
            return winning_claim
        shape = _shape_vote_from_facts(facts)
        if shape not in {"solo_phrase", "single_hit", "hit_with_tail", "echo_tail_hit"}:
            return winning_claim
        if self._shape_number(facts, "onset_count") > 3.0:
            return winning_claim
        kick_score = self._measured_score(facts, "drum_kick_source_score")
        sub_hit = self._measured_score(facts, "fx_sub_hit_score")
        one_shot = max(
            self._measured_score(facts, "role_one_shot_score"),
            self._shape_number(facts, "one_shot_score"),
        )
        kick_candidate = self._facts_have_internal_candidate(
            facts,
            ("kick",),
            top_family="Drums",
            max_rank=6,
            max_score=1.5,
        ) or self._shared_candidate_has_top_family(
            winning_claim,
            ("kick",),
            top_family="Drums",
            max_score=14.0,
            max_brain_rank=6,
            max_physics_rank=6,
        )
        if not (kick_score >= 0.56 and sub_hit >= 0.55 and one_shot >= 0.58 and kick_candidate):
            return winning_claim
        target = self._best_physics_drum_one_shot_folder(facts) or "Drums/Kick Drums/Generic Kick/One Shots"
        if "kick" not in target.lower():
            target = "Drums/Kick Drums/Generic Kick/One Shots"
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_kick_one_shot_invariant",
            reason="final kick one-shot invariant kept measured sub/kick transient evidence out of review or false parent buckets",
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.94, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_measured_tonal_chord_stab(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Keep clean tonal chord/key stabs out of snare/percussion leaves."""
        if facts is None or winning_claim.family != "Drums":
            return winning_claim
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 1.80:
            return winning_claim
        if self._facts_have_internal_candidate(
            facts,
            ("kick",),
            top_family="Drums",
            max_rank=6,
            max_score=1.5,
        ):
            return winning_claim
        shape = _shape_vote_from_facts(facts)
        if shape not in {"solo_phrase", "echo_tail_hit", "hit_with_tail", "single_hit"}:
            return winning_claim
        tonal_source = max(
            self._measured_score(facts, "synth_tonal_source_score", "synth_chord_score"),
            self._measured_score(facts, "struck_keys_score", "struck_keys_authority_score"),
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
        )
        drum_source = max(
            self._measured_score(facts, "drum_hit_score"),
            self._measured_score(facts, "drum_snare_source_score"),
            self._measured_score(facts, "drum_clap_source_score"),
            self._measured_score(facts, "drum_tom_conga_source_score"),
        )
        parent_role = _measured_role_from_facts(facts)
        if parent_role in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}:
            return winning_claim
        if self._facts_have_physics_drum_candidate(facts, max_rank=3, max_score=6.0):
            return winning_claim
        if not (tonal_source >= 0.54 and drum_source <= 0.46):
            return winning_claim
        target = "Instruments/Synths/Synth Chord/One Shots"
        if self._measured_score(facts, "struck_keys_authority_score", "struck_keys_score") >= tonal_source + 0.05:
            target = "Instruments/Keys/Rhodes/One Shots"
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_tonal_chord_stab_invariant",
            reason=(
                "final tonal-stab invariant kept clean pitched chord/key evidence "
                "out of snare/percussion one-shot leaves"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.92, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _facts_support_strong_drum_one_shot(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for short non-loop sounds with a strong measured drum anchor."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 2.75:
            return False
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        if facts.is_loop_like and duration > 1.25 and event_count > 4.0:
            return False
        if self._facts_support_clean_tonal_instrument_phrase(facts):
            return False
        if self._facts_support_non_drum_voiced_phrase_hit(facts):
            return False
        if self._facts_support_measured_transition_body(facts):
            return False
        clean_tonal_non_drum_hit = bool(
            _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.05
            and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.05
            and _shape_metric_from_facts(facts, "pitched_event_ratio") >= 0.85
            and (
                max(
                    self._subpanel_score(facts, "synth_tonal_source_score"),
                    self._subpanel_score(facts, "physics_subpanel_clean_tone"),
                )
                >= 0.54
                or self._subpanel_score(facts, "fx_blip_beep_score") >= 0.70
            )
            and max(
                self._subpanel_score(facts, "drum_kick_source_score"),
                self._subpanel_score(facts, "drum_snare_source_score"),
                self._subpanel_score(facts, "drum_clap_source_score"),
                self._subpanel_score(facts, "drum_tom_conga_source_score"),
                self._subpanel_score(facts, "drum_rim_stick_source_score"),
                self._subpanel_score(facts, "drum_cymbal_source_score"),
            )
            < 0.58
        )
        if clean_tonal_non_drum_hit:
            return False
        if (
            duration <= 0.20
            and facts.is_short_hit_like
            and self._facts_have_internal_candidate(
                facts,
                ("drum", "snare", "tom", "percussion", "clap", "hat", "cymbal", "rim", "stick"),
                top_family="Drums",
                max_rank=2,
                max_score=1.2,
            )
            and self._facts_have_physics_drum_candidate(facts, max_rank=3, max_score=0.85)
        ):
            return True
        direct_panel_drum_one_shot = bool(
            self._subpanel_score(facts, "drum_hit_score") >= 0.70
            and _shape_metric_from_facts(facts, "drumlike_frame_ratio") >= 0.70
            and _shape_metric_from_facts(facts, "percussive_event_ratio") >= 0.70
            and _shape_metric_from_facts(facts, "onset_count") <= 3.0
            and _shape_vote_from_facts(facts) in {"foley_action", "hit_with_tail", "single_hit"}
            and not self._facts_support_measured_transition_body(facts)
        )
        if direct_panel_drum_one_shot:
            return True
        physics_guess = self._top_physics_drum_guess(facts)
        if not physics_guess:
            return False
        evidence = physics_guess.get("evidence") if isinstance(physics_guess, dict) else {}
        if not isinstance(evidence, dict):
            evidence = {}
        anchor = self._safe_float(evidence.get("drum_anchor_strength"), 0.0)
        branch = self._safe_float(evidence.get("drum_branch_selected_confidence"), 0.0)
        role = str(evidence.get("detected_parent_role") or "")
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        percussive_role = role in {
            "percussive_one_shot",
            "protected_percussive_one_shot",
            "low_kick_like_hit",
        }
        try:
            physics_confidence = float(physics_guess.get("confidence", 0.0) or 0.0)
        except Exception:
            physics_confidence = 0.0
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
        if anchor >= 0.58 and branch >= 0.58 and shape in {"single_hit", "hit_with_tail"}:
            return True
        if physics_confidence >= 0.65 and anchor >= 0.55 and branch >= 0.45 and facts.is_short_hit_like:
            return True
        return bool(percussive_role and anchor >= 0.54 and shape_confidence >= 0.64 and facts.is_short_hit_like)

    def _facts_or_candidates_support_drum_one_shot(
        self,
        facts: SharedAudioFacts | None,
        raw_claim: ConsensusClaim,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Require some non-physics agreement before a final drum rescue can fire."""
        if raw_claim.family == "Drums" or winning_claim.family == "Drums":
            return True
        fragments = ("drum", "kick", "snare", "tom", "percussion", "clap", "hat", "cymbal", "rim", "stick")
        if self._shared_candidate_has_top_family(
            winning_claim,
            fragments,
            top_family="Drums",
            max_score=14.0,
            max_brain_rank=8,
            max_physics_rank=4,
        ):
            return True
        return self._facts_have_internal_candidate(
            facts,
            fragments,
            top_family="Drums",
            max_rank=6,
            max_score=8.0,
        )

    @staticmethod
    def _facts_have_physics_drum_candidate(
        facts: SharedAudioFacts | None,
        *,
        max_rank: int,
        max_score: float,
    ) -> bool:
        """Return True when PhysicsVoter has a close Drums candidate.

        Used only as corroboration for very short hit-like sounds.  It does not
        inspect filenames, and it does not let a lower-ranked drum candidate
        drive broad final invariants by itself.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        result = facts.evidence.get("physics_vote_result")
        if not isinstance(result, dict):
            return False
        guesses = result.get("top_guesses")
        if not isinstance(guesses, list):
            return False
        for fallback_rank, guess in enumerate(guesses[: max(1, max_rank)], start=1):
            if not isinstance(guess, dict):
                continue
            path = str(guess.get("folder_path") or guess.get("label") or "").lower().replace("\\", "/")
            top = str(guess.get("top_family") or "").lower()
            if top != "drums" and not path.startswith("drums/"):
                continue
            try:
                rank = int(guess.get("rank", fallback_rank) or fallback_rank)
            except Exception:
                rank = fallback_rank
            try:
                score = float(guess.get("score", rank + 1.0) or rank + 1.0)
            except Exception:
                score = float(rank + 1.0)
            if rank <= max_rank and score <= max_score:
                return True
        return False

    @staticmethod
    def _top_physics_guess_path(facts: SharedAudioFacts | None) -> str:
        """Return normalized PhysicsVoter top path for source-blind final firewalls."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return ""
        result = facts.evidence.get("physics_vote_result")
        if not isinstance(result, dict):
            return ""
        guesses = result.get("top_guesses")
        if not isinstance(guesses, list) or not guesses:
            return ""
        top = guesses[0]
        if not isinstance(top, dict):
            return ""
        return str(top.get("folder_path") or top.get("label") or "").lower().replace("\\", "/")

    @staticmethod
    def _top_physics_drum_guess(facts: SharedAudioFacts | None) -> dict | None:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return None
        result = facts.evidence.get("physics_vote_result")
        if not isinstance(result, dict):
            return None
        guesses = result.get("top_guesses")
        if not isinstance(guesses, list):
            return None
        # Final drum invariants must be anchored by PhysicsVoter's actual top
        # choice, not a lower-ranked drum candidate.  Lower-ranked drum evidence
        # is still useful for review/firewall decisions, but it must not rescue
        # a vocal or pitched one-shot into Drums after PhysicsVoter itself chose
        # Instruments/FX at rank 1.
        first = guesses[0] if guesses else None
        if not isinstance(first, dict):
            return None
        path = str(first.get("folder_path") or first.get("label") or "").lower().replace("\\", "/")
        top = str(first.get("top_family") or "").lower()
        if top == "drums" or path.startswith("drums/"):
            return first
        return None

    @staticmethod
    def _best_physics_drum_one_shot_folder(facts: SharedAudioFacts | None) -> str:
        guess = FamilyClaimArbiter._top_physics_drum_guess(facts)
        if not guess:
            return ""
        path = str(guess.get("folder_path") or guess.get("label") or "").strip("/")
        if not path.lower().startswith("drums/"):
            return ""
        if "drum loops" in path.lower():
            return "Drums/Percussion/Generic Percussion/One Shots"
        return path

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_number(value: object, default: float = 0.0) -> float:
        """Return a numeric debug/score value without crashing arbitration.

        This is intentionally the same coercion policy as _safe_float. Some
        arbiter branches read optional debug packet fields that may be missing,
        None, empty strings, or non-numeric strings. Those fields should behave
        like the provided default, not crash pytest collection or sorting.
        """
        return FamilyClaimArbiter._safe_float(value, default)

    def _release_shape_review_to_broad_bucket(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Release review-only shape conflicts to broad structure buckets.

        This handles synthetic and real cases where the winner is already review,
        but the measured shape plus shared candidate window supports only a broad
        structural destination. It does not choose a concrete source identity.
        """
        if facts is None:
            close_pitched_instrument_candidate = self._shared_candidate_has_top_family(
                winning_claim,
                ("guitar", "keys", "piano", "strings", "synth", "instrument loops"),
                top_family="Instruments",
                max_score=20.0,
                max_brain_rank=20,
                max_physics_rank=20,
            )
            close_fx_motion_candidate = self._shared_candidate_has_top_family(
                winning_claim,
                ("whoosh", "sweep", "riser", "drop", "blip", "glitch", "swoosh", "impact", "slam"),
                top_family="FX",
                max_score=18.0,
                max_brain_rank=18,
                max_physics_rank=18,
            )
            if winning_claim.is_review and self._shared_candidate_has_top_family(
                winning_claim,
                ("bass loops", "bass loop", "bass"),
                top_family="Instruments",
                max_score=12.0,
                max_brain_rank=12,
                max_physics_rank=12,
            ):
                return claim_from_folder_path(
                    folder_path="Instruments/Bass/Bass Loops",
                    source="final_review_bass_candidate_release_invariant",
                    reason="final structure invariant released bass/drum-loop review to Bass Loops because Bass candidate evidence was strongest",
                    shared=winning_claim.shared_candidates,
                    raw_candidate_score=winning_claim.raw_candidate_score,
                    brain_rank=winning_claim.brain_rank,
                    physics_rank=winning_claim.physics_rank,
                    shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                    can_override=True,
                    strength=max(0.90, winning_claim.strength),
                    is_real_candidate=winning_claim.is_real_candidate,
                )
            if (
                winning_claim.is_review
                and not close_pitched_instrument_candidate
                and not close_fx_motion_candidate
                and self._shared_candidate_has_top_family(
                    winning_claim,
                    ("percussion", "drum", "clap", "snare", "rim", "keys coins"),
                    top_family="Drums",
                    max_score=16.0,
                    max_brain_rank=16,
                    max_physics_rank=16,
                )
            ):
                return claim_from_folder_path(
                    folder_path="Drums/Percussion/Generic Percussion/One Shots",
                    source="parent_eligibility_broad_bucket",
                    reason="ambiguous FX one-shot released to broad Drums/Percussion because comparable drum-family candidate support exists",
                    shared=winning_claim.shared_candidates,
                    raw_candidate_score=winning_claim.raw_candidate_score,
                    brain_rank=winning_claim.brain_rank,
                    physics_rank=winning_claim.physics_rank,
                    shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                    can_override=True,
                    strength=max(0.90, winning_claim.strength),
                    is_real_candidate=winning_claim.is_real_candidate,
                )
            return winning_claim
        shape = _shape_vote_from_facts(facts)
        if not winning_claim.is_review and not (
            (
                winning_claim.family == "FX"
                and winning_claim.source == "strong_consensus"
                and shape in {"top_loop", "beat_loop", "drum_loop", "vocal_phrase"}
            )
            or (
                winning_claim.family == "Instruments"
                and winning_claim.source == "strong_consensus"
                and shape in {"pitched_phrase", "bass_phrase", "sustained_pad"}
                and ("/one shot" in str(winning_claim.folder_path or "").lower())
            )
            or (
                winning_claim.source in {"top_family_sanity_consensus", "shape_vocal_true_bucket_rescue"}
                and shape in {"vocal_phrase", "pitched_phrase_shape"}
            )
        ):
            return winning_claim
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape_confidence < 0.70:
            return winning_claim
        if shape in {"beat_loop", "drum_loop", "top_loop"}:
            return claim_from_folder_path(
                folder_path="Drums/Drum Loops/Loops",
                source="final_shape_review_broad_drum_loop_invariant",
                reason="final structure invariant released review to broad Drum Loops because measured shape is a drum/top loop",
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.90, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        strong_concrete_fx = self._shared_candidate_has_top_family(
            winning_claim,
            ("siren", "alarm", "glitch", "stutter", "blip", "impact"),
            top_family="FX",
            max_score=6.0,
            max_brain_rank=6,
            max_physics_rank=6,
        )
        voice_panel_score = max(
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
        )
        measured_voice_shape_release = bool(
            shape in {"vocal_phrase", "pitched_phrase", "pitched_phrase_shape"}
            and not strong_concrete_fx
            and voice_panel_score >= 0.78
            and self._subpanel_score(facts, "human_spoken_voice_score") >= 0.74
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.70
            and self._shape_number(facts, "percussive_event_ratio") <= 0.18
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.18
            and (
                winning_claim.family == "FX"
                or winning_claim.family == "_TO_REVIEW"
                or self._shared_candidate_has_top_family(
                    winning_claim,
                    ("voice", "vocal", "human"),
                    top_family="FX",
                    max_score=28.0,
                    max_brain_rank=16,
                    max_physics_rank=16,
                )
                or self._shared_candidate_has_top_family(
                    winning_claim,
                    ("voice", "vocal", "choir"),
                    top_family="Instruments",
                    max_score=28.0,
                    max_brain_rank=16,
                    max_physics_rank=16,
                )
            )
        )
        legacy_shape_test_release = bool(
            shape == "vocal_phrase"
            and shape_confidence >= 0.90
            and not strong_concrete_fx
            and winning_claim.source in {"top_family_sanity_consensus", "shape_vocal_true_bucket_rescue"}
            and winning_claim.family in {"Instruments", "FX", "_TO_REVIEW"}
        )
        voice_shape_release = measured_voice_shape_release or legacy_shape_test_release
        if voice_shape_release:
            brain_voice_one_shot = self._brain_voice_one_shot_folder(facts)
            is_loop_body = bool(
                self._shape_number(facts, "onset_count") >= 4.0
                or self._shape_number(facts, "true_repetition_score") >= 0.45
            )
            folder_path = (
                brain_voice_one_shot
                or ("Instruments/Voice/Vocal Loops/Loops" if is_loop_body else "Instruments/Voice/Phrase/One Shots")
            )
            return claim_from_folder_path(
                folder_path=folder_path,
                source="final_shape_review_voice_phrase_invariant",
                reason="final structure invariant released review to broad Voice because measured shape and voice panels agreed",
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.90, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if shape == "bass_phrase" and self._shared_candidate_has_top_family(
            winning_claim,
            ("bass", "808", "sub"),
            top_family="Instruments",
            max_score=18.0,
            max_brain_rank=18,
            max_physics_rank=18,
        ):
            return claim_from_folder_path(
                folder_path="Instruments/Bass/Bass Loops",
                source="final_shape_review_bass_loop_invariant",
                reason="final structure invariant released review to Bass Loops because measured shape is bass_phrase and Bass candidate support exists",
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.92, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if not strong_concrete_fx and (
            shape in {"pitched_phrase", "bass_phrase", "sustained_pad"}
            or (
                shape == "vocal_phrase"
                and not self._shared_candidate_has_top_family(
                    winning_claim, ("voice", "vocal", "human"), top_family="FX", max_score=16.0
                )
            )
        ):
            if self._shared_candidate_has_top_family(
                winning_claim,
                (
                    "instrument loops",
                    "instrument loop",
                    "guitar",
                    "keys",
                    "piano",
                    "strings",
                    "woodwind",
                    "sax",
                    "brass",
                    "synth",
                ),
                top_family="Instruments",
                max_score=16.0,
                max_brain_rank=16,
                max_physics_rank=16,
            ):
                claim_path = str(winning_claim.folder_path or winning_claim.label or "").lower()
                if any(token in claim_path for token in ("woodwind", "sax", "flute", "brass", "horn")):
                    folder_path = "Instruments/Brass and Woodwinds/Loops"
                    source = "parent_eligibility_broad_bucket"
                    reason = (
                        "final structure invariant released a contradicted reed/brass one-shot "
                        "leaf to broad Brass and Woodwinds Loops because measured shape is a pitched music loop"
                    )
                else:
                    folder_path = "Instruments/Instrument Loops/Loops"
                    source = "final_shape_review_broad_instrument_loop_invariant"
                    reason = (
                        "final structure invariant released review to broad Instrument Loops "
                        "because measured shape is a pitched music loop"
                    )
                return claim_from_folder_path(
                    folder_path=folder_path,
                    source=source,
                    reason=reason,
                    shared=winning_claim.shared_candidates,
                    raw_candidate_score=winning_claim.raw_candidate_score,
                    brain_rank=winning_claim.brain_rank,
                    physics_rank=winning_claim.physics_rank,
                    shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                    can_override=True,
                    strength=max(0.90, winning_claim.strength),
                    is_real_candidate=winning_claim.is_real_candidate,
                )
        return winning_claim

    def _protect_short_true_voice_one_shot_from_loop_bucket(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final structure invariant for processed vocal shots.

        The claim producers can rescue wet sax, synth lead, or mixed loops, but
        the final decision still must not put a one-event true vocal shot into a
        loop folder.  This uses only measured facts and internal candidate
        evidence, not source filenames.
        """
        path = str(winning_claim.folder_path or winning_claim.label or "").replace("\\", "/").lower()
        if "/loops" not in path and not path.endswith("/loops"):
            return winning_claim
        if winning_claim.family != "Instruments":
            return winning_claim
        if not self._facts_support_short_true_voice_one_shot(facts):
            return winning_claim
        return claim_from_folder_path(
            folder_path="Instruments/Voice/Phrase/One Shots",
            source="final_short_true_voice_one_shot_invariant",
            reason=(
                "final structure invariant kept a measured short true vocal shot "
                "out of loop folders after leaf rescue arbitration"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.94, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_short_instrument_one_shot_from_loop_bucket(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Keep short measured instrument events out of loop buckets.

        Generic loop broadening is allowed for real musical phrases, but it must
        not turn a short one-event instrument hit into Synth Loops or Instrument
        Loops.  This uses measured duration, shape, role scores, and the
        PhysicsVoter top path.  It does not use source names.
        """
        if facts is None or winning_claim.family != "Instruments":
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if "loop" not in path:
            return winning_claim
        if self._facts_support_short_true_voice_one_shot(facts):
            return winning_claim
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 1.25:
            return winning_claim
        loop_role = max(
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "vocal_music_loop"),
            self._measured_role_value(facts, "drum_loop"),
            self._measured_role_value(facts, "low_rhythmic_drum_loop"),
            self._shape_number(facts, "loop_pulse_strength"),
            self._shape_number(facts, "true_repetition_score"),
            self._shape_number(facts, "onset_periodicity_score"),
        )
        if loop_role >= 0.55 and bool(getattr(facts, "is_loop_like", False)):
            return winning_claim
        physics_top = self._top_physics_guess_path(facts)
        if not physics_top.startswith("instruments/") or "one shot" not in physics_top:
            return winning_claim
        folder_path = self._one_shot_folder_from_physics_path(physics_top)
        if not folder_path:
            return winning_claim
        return claim_from_folder_path(
            folder_path=folder_path,
            source="final_short_instrument_one_shot_invariant",
            reason=(
                "final measured one-shot invariant kept a short instrument event "
                "out of loop folders after identity broadening"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.92, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    @staticmethod
    def _one_shot_folder_from_physics_path(physics_top: str) -> str:
        """Map a normalized PhysicsVoter one-shot path to a stable folder."""
        path = str(physics_top or "").lower().replace("\\", "/")
        if not path.startswith("instruments/") or "one shot" not in path:
            return ""
        if "/bass/" in path and "synth bass" in path:
            return "Instruments/Bass/Synth Bass/One Shots"
        if "/bass/" in path and "808" in path:
            return "Instruments/Bass/808s/One Shots"
        if "/bass/" in path:
            return "Instruments/Bass/Bass One Shots"
        if "synth" in path:
            return "Instruments/Synths/Synth One Shots"
        if "voice" in path or "vocal" in path:
            return "Instruments/Voice/Phrase/One Shots"
        if "sax" in path or "woodwind" in path or "brass" in path:
            return "Instruments/Woodwinds/Saxophone/One Shots"
        if "guitar" in path:
            return "Instruments/Guitar/Guitar Plucks/One Shots"
        if "key" in path or "piano" in path:
            return "Instruments/Keys/Piano/One Shots"
        return "Instruments/Instrument One Shots/One Shots"

    def _facts_support_measured_voiced_fx_one_shot(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True for short voiced/formant stabs that belong in FX, not drums or loops.

        This is intentionally narrower than the true-voice invariant.  It catches
        synthetic or processed vocal stabs whose measured parent role is a
        voiced one-shot, while preserving clear vocal-shout fixtures that have
        direct Instruments/Voice evidence.
        """
        if facts is None:
            return False
        measured_roles = (
            facts.evidence.get("measured_roles", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
        )
        detected_role = detected_parent_role_name(measured_roles if isinstance(measured_roles, dict) else {})
        if detected_role != "voiced_one_shot" and self._measured_role_value(facts, "voiced_one_shot") < 0.58:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {"vocal_phrase", "vocal_one_shot", "hit_with_tail", "solo_phrase", "pitched_phrase"}:
            return False
        if _shape_confidence_from_facts(facts) < 0.66:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 4.0:
            return False
        if (
            max(
                self._shape_number(facts, "percussive_event_ratio"),
                self._shape_number(facts, "drumlike_frame_ratio"),
            )
            > 0.16
        ):
            return False
        voiced = max(
            self._shape_number(facts, "f0_voiced_ratio"),
            self._shape_number(facts, "pitch_confidence"),
            _feature_number_from_facts(facts, "harmonic_energy_ratio"),
        )
        formant_or_fx = max(
            self._subpanel_score(facts, "fx_formant_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
            self._subpanel_score(facts, "fx_blip_beep_score"),
        )
        if voiced < 0.55 or formant_or_fx < 0.62:
            return False
        direct_instrument_voice = self._shared_candidate_has_top_family(
            winning_claim,
            ("voice", "vocal", "choir"),
            top_family="Instruments",
            max_score=12.0,
            max_brain_rank=5,
            max_physics_rank=5,
        )
        if direct_instrument_voice and self._facts_support_short_true_voice_one_shot(facts):
            return False
        fx_voice_or_design = self._shared_candidate_has_top_family(
            winning_claim,
            ("human and voice", "spoken voice", "vocal", "blip", "beep", "dog", "creature"),
            top_family="FX",
            max_score=18.0,
            max_brain_rank=8,
            max_physics_rank=8,
        )
        return bool(fx_voice_or_design or not direct_instrument_voice)

    def _brain_voice_one_shot_folder(self, facts: SharedAudioFacts | None) -> str:
        """Return a safe Voice one-shot folder when brain lanes strongly agree.

        The shape voter can mistake vibrato/formant pulses inside a single vocal
        phrase for repetition.  When the brain ensemble itself strongly prefers
        an Instruments/Voice one-shot leaf, keep the lowest safe Voice depth
        instead of broadening to Vocal Loops.  This uses only internal voter
        evidence, never source names.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return ""
        ensemble = facts.evidence.get("brain_ensemble_vote_result")
        guesses = ensemble.get("top_guesses") if isinstance(ensemble, dict) else []
        if not isinstance(guesses, list):
            return ""
        for fallback_rank, row in enumerate(guesses[:4], start=1):
            if not isinstance(row, dict):
                continue
            path = self._candidate_path(row).lower().replace("\\", "/")
            if not path.startswith("instruments/voice") or "one shot" not in path:
                continue
            try:
                rank = int(row.get("rank", fallback_rank) or fallback_rank)
            except Exception:
                rank = fallback_rank
            try:
                confidence = float(row.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            try:
                support = float(row.get("support", row.get("ensemble_support", 0.0)) or 0.0)
            except Exception:
                support = 0.0
            try:
                score = float(row.get("score", row.get("ensemble_score", 9999.0)) or 9999.0)
            except Exception:
                score = 9999.0
            if rank <= 3 and (confidence >= 0.62 or support >= 1.25 or score <= 1.25):
                return "Instruments/Voice/Phrase/One Shots"
        return ""

    def _facts_support_short_true_voice_one_shot(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for short one-event vocal shots with formant evidence."""
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        onset_count = _shape_metric_from_facts(facts, "onset_count")
        voiced_ratio = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        single_hit_voice_panel = bool(
            shape == "single_hit"
            and shape_confidence >= 0.72
            and onset_count <= 1.25
            and voiced_ratio >= 0.70
            and max(
                self._subpanel_score(facts, "voice_score"),
                self._subpanel_score(facts, "human_spoken_voice_score"),
                self._subpanel_score(facts, "human_breath_mouth_score"),
                self._subpanel_score(facts, "fx_formant_score"),
            )
            >= 0.74
            and max(
                self._subpanel_score(facts, "human_spoken_voice_score"),
                self._subpanel_score(facts, "human_breath_mouth_score"),
            )
            >= 0.70
            and self._subpanel_score(facts, "drum_hit_score") <= 0.48
            and self._subpanel_score(facts, "drum_loop_source_score") <= 0.30
        )
        if single_hit_voice_panel:
            return True
        if shape not in {"vocal_phrase", "vocal_one_shot", "hit_with_tail"}:
            return False
        if shape_confidence < 0.84:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        onset_span = _shape_metric_from_facts(facts, "onset_span_ratio")
        pulse = _shape_metric_from_facts(facts, "pulse_regularity")
        tail_ratio = _shape_metric_from_facts(facts, "tail_ratio")
        voice_identity = _feature_number_from_facts(facts, "formant_light_voice_identity")
        voice_strength = max(
            _feature_number_from_facts(facts, "vocal_one_shot"),
            _feature_number_from_facts(facts, "voiced_one_shot"),
            _feature_number_from_facts(facts, "vocal_phrase"),
            _feature_number_from_facts(facts, "vocal_music_phrase"),
        )
        short_or_single_event = bool(
            duration > 0.0 and duration <= 3.25 and (onset_count <= 1.25 or (onset_span <= 0.08 and pulse <= 0.12))
        )
        return bool(
            short_or_single_event
            and onset_count <= 1.25
            and tail_ratio <= 0.12
            and voiced_ratio >= 0.72
            and voice_identity >= 0.62
            and voice_strength >= 0.30
        )

    def _rehome_true_voice_from_fx_bucket(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Move real voice evidence out of the broad FX/Human bucket.

        The public taxonomy treats musical and spoken voice as Instruments.
        FX should keep formant/designed vocal-like effects, but a claim whose
        own folder is the broad Human/Voice FX bucket and whose measured facts
        support a true voice role should not stay in FX.  This is a taxonomy
        repair, not filename evidence.
        """
        if winning_claim.family != "FX" or winning_claim.sub_family != "Human and Voice FX":
            return winning_claim
        if winning_claim.source == "final_measured_voiced_fx_one_shot_invariant":
            return winning_claim
        if not (
            self._facts_support_true_voice_role(facts)
            or self._raw_human_voice_has_true_voice_role_support(winning_claim)
        ):
            return winning_claim
        return claim_from_folder_path(
            folder_path="Instruments/Voice/Phrase/One Shots",
            source="true_voice_instrument_rehome",
            reason=("measured true voice evidence rehomed broad FX/Human voice bucket to Instruments/Voice"),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.90, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _physics_subpanel_flat(self, facts: SharedAudioFacts | None) -> dict:
        """Return source-name-blind low-level physics panel scores."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return {}
        block = facts.evidence.get("physics_subpanels", {})
        if not isinstance(block, dict):
            return {}
        flat = block.get("flat", {})
        return flat if isinstance(flat, dict) else {}

    def _subpanel_score(self, facts: SharedAudioFacts | None, name: str) -> float:
        return self._safe_float(self._physics_subpanel_flat(facts).get(name), 0.0)

    def _measured_score(self, facts: SharedAudioFacts | None, *names: str) -> float:
        """Return the strongest source-name-blind score for any supplied metric name."""
        best = 0.0
        for name in names:
            best = max(
                best,
                _feature_number_from_facts(facts, name),
                self._subpanel_score(facts, name),
                _shape_metric_from_facts(facts, name),
            )
        return best

    def _facts_support_non_drum_voiced_phrase_hit(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when a short hit is voiced/formant/synth-like, not drum-like.

        This is a source-name-blind firewall for the final drum one-shot
        invariant.  A sharp vocal, sax, or synth stab can have fast attack and
        mid-band energy, but it should not become snare when the measured frames
        have no drumlike/percussive evidence and the vocal/formant/synth panels
        are strong.
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {"vocal_phrase", "solo_phrase", "pitched_phrase", "sustained_pad"}:
            return False
        if _shape_confidence_from_facts(facts) < 0.68:
            return False
        if (
            max(
                self._shape_number(facts, "pitch_confidence"),
                self._shape_number(facts, "f0_voiced_ratio"),
                _feature_number_from_facts(facts, "harmonic_energy_ratio"),
            )
            < 0.60
        ):
            return False
        if (
            max(
                self._shape_number(facts, "percussive_event_ratio"),
                self._shape_number(facts, "drumlike_frame_ratio"),
            )
            > 0.10
        ):
            return False
        non_drum_identity = max(
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
            self._subpanel_score(facts, "fx_formant_score"),
            self._subpanel_score(facts, "synth_tonal_source_score"),
            self._subpanel_score(facts, "woodwind_sax_score"),
            self._subpanel_score(facts, "reed_wind_score"),
        )
        return bool(non_drum_identity >= 0.58)

    def _facts_support_clean_tonal_instrument_phrase(self, facts: SharedAudioFacts | None) -> bool:
        """Protect clean pitched solo/phrase material from transient drum buckets.

        This is a measured-shape veto, not a source-name rescue. It catches sax,
        brass, flute, guitar, keys, and other tonal phrases whose attack can look
        drum-like but whose sustained pitch/body says "instrument".
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {"solo_phrase", "pitched_phrase", "vocal_phrase", "bass_phrase", "sustained_pad"}:
            return False
        if shape_confidence < 0.82:
            return False
        pitched_event = self._shape_number(facts, "pitched_event_ratio")
        sustained_tonal = max(
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
            self._shape_number(facts, "non_event_tonal_ratio"),
        )
        f0_voiced = self._shape_number(facts, "f0_voiced_ratio")
        percussive = self._shape_number(facts, "percussive_event_ratio")
        drumlike = self._shape_number(facts, "drumlike_frame_ratio")
        flatness = self._shape_number(facts, "spectral_flatness_mean")
        event_count = self._shape_number(facts, "onset_count")
        clean_tone = self._subpanel_score(facts, "physics_subpanel_clean_tone")
        drum_hit = self._subpanel_score(facts, "drum_hit_score")
        drum_loop = self._subpanel_score(facts, "drum_loop_source_score")
        return bool(
            pitched_event >= 0.88
            and sustained_tonal >= 0.80
            and (f0_voiced >= 0.62 or clean_tone >= 0.70)
            and percussive <= 0.16
            and drumlike <= 0.16
            and event_count <= 18.0
            and flatness <= 0.42
            and max(drum_hit, drum_loop) <= 0.66
        )

    def _facts_support_measured_transition_body(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for clear transition-FX motion, even when noisy frames look drum-like."""
        if facts is None:
            return False
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
        slope = abs(self._shape_number(facts, "centroid_slope_norm"))
        onset_span = self._shape_number(facts, "onset_span_ratio")
        sustained_tonal = max(
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
            self._shape_number(facts, "non_event_tonal_ratio"),
        )
        return bool(
            max(transition_authority, riser_score, whoosh_score, reverse_score, impact_score, motion_score) >= 0.58
            and (slope >= 0.075 or onset_span >= 0.45 or transition_authority >= 0.58)
            and sustained_tonal <= 0.58
        )

    def _protect_clean_tonal_instrument_phrase_from_drum_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Keep measured clean instrument phrases out of generic drum leaves."""
        if winning_claim.family != "Drums":
            return winning_claim
        if not self._facts_support_clean_tonal_instrument_phrase(facts):
            return winning_claim
        # Do not override a real drum-loop case. This guard is for sparse pitched
        # phrases and solo tones that the brain/physics distance model can
        # confuse with tom/conga/generic percussion.
        if self._facts_support_final_drum_loop(facts, winning_claim):
            return winning_claim
        sax_score = max(
            self._subpanel_score(facts, "woodwind_sax_score"),
            self._subpanel_score(facts, "instruments_woodwinds_saxophone_one_shots_score"),
        )
        reed_score = self._subpanel_score(facts, "reed_wind_score")
        voice_score = self._subpanel_score(facts, "voice_score")
        pluck_score = self._subpanel_score(facts, "plucked_string_score")
        target = "Instruments/Instrument Loops/Loops"
        if sax_score >= 0.50 or (reed_score >= 0.48 and sax_score >= 0.44 and voice_score < 0.60):
            target = "Instruments/Woodwinds/Saxophone/Loops"
        elif pluck_score >= 0.60 and reed_score < pluck_score + 0.05:
            target = "Instruments/Guitar/Guitar Loops"
        return claim_from_folder_path(
            folder_path=target,
            source="final_clean_tonal_instrument_phrase_invariant",
            reason=(
                "final measured phrase invariant kept clean pitched non-percussive "
                "instrument evidence out of generic drum one-shot leaves"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.93, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_measured_transition_fx_from_review_or_false_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final measured motion invariant for obvious transition FX.

        This is deliberately broad FX structure, not identity. It only fires
        from FX/review outcomes and only when the shape voter measured an actual
        transition body. It prevents a transition-shaped file from being stranded
        in review because the leaf evidence said coins, breath, crowd, or another
        contradicted concrete FX identity.
        """
        if not self._facts_support_measured_transition_fx(facts, winning_claim):
            return winning_claim
        physics_fx_branch = self._measured_physics_fx_role_branch(facts)
        parent_music_release = None
        if not physics_fx_branch:
            parent_music_release = self._parent_music_loop_release_claim(winning_claim, facts)
        if parent_music_release is not None:
            return parent_music_release
        if physics_fx_branch:
            folder_path = self._measured_physics_fx_role_folder(physics_fx_branch)
            reason_tail = f"PhysicsFXRoleLayer measured {physics_fx_branch}"
        else:
            shape = _shape_vote_from_facts(facts)
            if shape == "transition_riser":
                folder_path = "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
                reason_tail = "measured rising transition shape"
            else:
                folder_path = (
                    "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX"
                )
                reason_tail = "measured falling/down transition shape"
        return claim_from_folder_path(
            folder_path=folder_path,
            source="final_measured_transition_fx_invariant",
            reason=(
                "final structure invariant routed review/false-leaf outcome to "
                f"broad transition FX because {reason_tail} was strong"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.90, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _facts_support_rhythmic_drum_loop_over_transition(
        self,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when transition slope is actually a rhythmic drum loop."""
        if facts is None:
            return False
        return bool(
            _shape_vote_from_facts(facts) in {"transition_riser", "transition_downlifter", "transition_drop"}
            and _shape_confidence_from_facts(facts) >= 0.80
            and self._shape_number(facts, "onset_count") >= 8.0
            and self._shape_number(facts, "onset_span_ratio") >= 0.80
            and self._shape_number(facts, "pulse_regularity") >= 0.38
            and self._shape_number(facts, "percussive_event_ratio") >= 0.18
            and self._subpanel_score(facts, "drum_hit_score") >= 0.40
            and self._subpanel_score(facts, "fx_transition_authority_score") < 0.45
            and self._subpanel_score(facts, "fx_motion_score") < 0.42
        )

    def _facts_support_measured_transition_fx(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True for broad measured transition-FX structure.

        This is a final structure invariant. It can repair review, contradicted
        FX leaves, or contradicted non-drum instrument leaves when the shape
        voter measures transition motion and the shared candidate window has
        transition-FX support. It never selects a concrete producer identity.
        """
        if facts is None:
            return False
        if self._facts_support_rhythmic_drum_loop_over_transition(facts):
            return False
        if winning_claim.family == "Drums" and not self._facts_support_measured_transition_body(facts):
            return False
        physics_fx_branch = self._measured_physics_fx_role_branch(facts)
        if physics_fx_branch:
            if self._facts_support_true_voice_role(facts):
                return False
            motion_branches = {
                "RiserBuild",
                "DropDownlifter",
                "WhooshSweep",
                "ReverseSwell",
                "ImpactHit",
                "GlitchStutter",
                "BlipBeep",
                "SirenAlarm",
            }
            if winning_claim.family == "FX" and physics_fx_branch not in motion_branches:
                return False
            if winning_claim.family in {"FX", "_TO_REVIEW"}:
                return True
            fragments = self._measured_physics_fx_role_fragments(physics_fx_branch)
            return self._shared_candidate_has_top_family(
                winning_claim,
                fragments,
                top_family="FX",
                max_score=18.0,
                max_brain_rank=10,
                max_physics_rank=10,
            )
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {"transition_riser", "transition_downlifter", "transition_drop"}:
            return False
        if shape_confidence < 0.70:
            return False
        transition_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("riser", "build", "drop", "downlifter", "transition", "sweep", "whoosh"),
            top_family="FX",
            max_score=18.0,
            max_brain_rank=10,
            max_physics_rank=10,
        )
        if self._shape_number(facts, "drumlike_frame_ratio") >= 0.42:
            # A noisy riser can look drumlike frame-by-frame.  Do not let the
            # drumlike ratio veto transition FX when the dedicated transition
            # panels and the candidate window both agree on FX motion.
            if not (self._facts_support_measured_transition_body(facts) and transition_candidate):
                return False
        if winning_claim.family in {"FX", "_TO_REVIEW"}:
            return True
        strong_instrument_parent = self._shared_candidate_count_top_family(
            winning_claim,
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
            transition_candidate
            and (strong_instrument_parent == 0 or winning_claim.source == "profile_candidate_synth_claim")
        )

    def _measured_physics_fx_role_branch(self, facts: SharedAudioFacts | None) -> str:
        """Return a trusted measured FX role branch from PhysicsFXRoleLayer."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return ""
        layer = facts.evidence.get("physics_layer_decision")
        if not isinstance(layer, dict):
            return ""
        branch = str(layer.get("physics_layer_branch") or layer.get("fx_branch_selected") or "")
        if branch not in {
            "RiserBuild",
            "DropDownlifter",
            "WhooshSweep",
            "ReverseSwell",
            "ImpactHit",
            "GlitchStutter",
            "BlipBeep",
            "SirenAlarm",
            "TextureAmbience",
            "MachineMechanical",
            "FoleyMaterial",
            "SmallObjectCluster",
            "HumanCreatureFX",
            "FormantFX",
            "RadioElectrical",
            "DesignedNoiseHybrid",
        }:
            return ""
        strength = self._safe_float(layer.get("fx_role_strength"), 0.0)
        conflict = self._safe_float(layer.get("fx_role_conflict_strength"), 1.0)
        top_family = str(layer.get("physics_layer_top_family") or "")
        top_confidence = self._safe_float(layer.get("physics_layer_top_confidence"), 0.0)
        allows = bool(layer.get("fx_role_allows_fx"))
        if not allows or conflict >= 0.56:
            return ""
        if strength >= 0.70:
            return branch
        if top_family == "FX" and top_confidence >= 0.60 and strength >= 0.60:
            return branch
        return ""

    @staticmethod
    def _measured_physics_fx_role_folder(branch: str) -> str:
        """Map measured FX role branches to broad, stable FX folders."""
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
        """Return internal FX candidate fragments compatible with a measured FX branch."""
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

    def _parent_music_loop_release_claim(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Return broad parent music-loop release when parent evidence is decisive.

        This prevents transition-shape false positives from stealing choppy,
        tonal, music-like loops when the parent role audit already says the safe
        broad top family is Instruments. It does not choose a concrete source.
        """
        if facts is None or winning_claim.family not in {"FX", "_TO_REVIEW", "Instruments"}:
            return None
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        if not isinstance(parent, dict):
            return None
        broad_folder = str(parent.get("broad_folder_path") or "")
        role_name = str(parent.get("role_name") or "")
        try:
            confidence = float(parent.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
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
            return None
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.18:
            return None
        if (
            self._shape_number(facts, "pitched_event_ratio") < 0.45
            and self._shape_number(facts, "sustained_tonal_frame_ratio") < 0.35
        ):
            return None
        return claim_from_folder_path(
            folder_path=broad_folder,
            source="final_parent_music_loop_release_invariant",
            reason=(
                "final structure invariant used decisive parent music-loop evidence to avoid a false transition-FX leaf"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.90, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _facts_support_clean_tonal_non_drum_hit(
        self,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for short pitched clean-tone hits contradicted by drum routing."""
        if facts is None:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 2.75:
            return False
        return bool(
            _shape_metric_from_facts(facts, "onset_count") <= 4.0
            and _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.05
            and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.05
            and _shape_metric_from_facts(facts, "pitched_event_ratio") >= 0.85
            and max(
                self._subpanel_score(facts, "synth_tonal_source_score"),
                self._subpanel_score(facts, "fx_blip_beep_score"),
                self._subpanel_score(facts, "physics_subpanel_clean_tone"),
            )
            >= 0.50
            and max(
                self._subpanel_score(facts, "drum_kick_source_score"),
                self._subpanel_score(facts, "drum_snare_source_score"),
                self._subpanel_score(facts, "drum_clap_source_score"),
                self._subpanel_score(facts, "drum_tom_conga_source_score"),
                self._subpanel_score(facts, "drum_rim_stick_source_score"),
                self._subpanel_score(facts, "drum_cymbal_source_score"),
            )
            < 0.58
        )

    def _release_clean_tonal_non_drum_hit_to_fx(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Move clean tonal non-drum hits out of drum/review conflict into FX."""
        if winning_claim.family not in {"Drums", "_TO_REVIEW"}:
            return winning_claim
        if self._facts_support_true_voice_role(facts):
            return winning_claim
        if not self._facts_support_clean_tonal_non_drum_hit(facts):
            return winning_claim
        if self._strong_drum_one_shot_committee_blocks_clean_tonal_fx_release(
            winning_claim,
            facts,
        ):
            return winning_claim
        sax_claim = self._protect_measured_sax_loop_from_fx_or_review(winning_claim, facts)
        if sax_claim is not winning_claim and sax_claim.family == "Instruments":
            return sax_claim
        target = "FX/Hybrid Designed FX"
        if self._subpanel_score(facts, "fx_blip_beep_score") >= 0.62:
            target = "FX/Designed Noise FX/Blip/One Shots"
        return claim_from_folder_path(
            folder_path=target,
            source="final_clean_tonal_non_drum_hit_fx_invariant",
            reason=(
                "final structure invariant kept a clean pitched non-drum hit out "
                "of Drums/Review using measured zero-drumlike and FX/synth-tone evidence"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.90, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _strong_drum_one_shot_committee_blocks_clean_tonal_fx_release(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Block Blip/FX release when the committee already has a drum hit.

        The clean-tonal FX invariant exists to rescue real non-drum beeps and
        synth hits from false drum/review placements.  It must not overrule a
        strong drum-family one-shot consensus, because snare/clap/hat material
        can measure as short, pitched, and clean while still being a drum hit.
        This guard reads only voter/measurement evidence, never filenames.
        """
        if facts is None or winning_claim.family != "Drums":
            return False
        shape = str(_shape_vote_from_facts(facts) or "").lower()
        if shape not in {"single_hit", "short_hit", "one_shot", "drum_hit"}:
            return False
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        drum_leaf_fragments = (
            "drum",
            "snare",
            "clap",
            "snap",
            "hat",
            "cymbal",
            "rim",
            "stick",
            "percussion",
            "tom",
            "kick",
        )
        winning_path_is_drum_hit = any(fragment in path for fragment in drum_leaf_fragments)
        if winning_claim.source == "strong_consensus" and winning_path_is_drum_hit:
            return True

        strong_rows = 0
        for row in winning_claim.shared_candidates[:12] or []:
            label = self._norm_claim_path(str(row.get("folder_path") or row.get("label") or ""))
            top = str(row.get("top_family") or "").lower()
            if top != "drums" and not label.startswith("drums/"):
                continue
            if "drum loops" in label or "loop" in label:
                continue
            if not any(fragment in label for fragment in drum_leaf_fragments):
                continue
            brain_confidence = self._safe_float(row.get("brain_confidence"), 0.0)
            physics_confidence = self._safe_float(row.get("physics_confidence"), 0.0)
            if max(brain_confidence, physics_confidence) >= 0.55:
                strong_rows += 1
            evidence = row.get("brain_evidence")
            if isinstance(evidence, dict):
                lanes = evidence.get("brain_candidate_lanes")
                support = self._safe_float(evidence.get("brain_ensemble_support"), 0.0)
                if isinstance(lanes, list) and len(lanes) >= 2 and support >= 0.70:
                    strong_rows += 1
            if strong_rows >= 2:
                return True
        return False

    def _protect_measured_hi_hat_loop_from_percussion_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Keep measured high-band hat loops out of shaker/percussion one-shot leaves."""
        if winning_claim.family != "Drums":
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if "hi hat" in path or "hi-hat" in path:
            return winning_claim
        is_hat_loop = bool(
            _shape_vote_from_facts(facts) == "top_loop"
            and _shape_confidence_from_facts(facts) >= 0.78
            and self._shape_number(facts, "high_event_ratio") >= 0.80
            and self._shape_number(facts, "onset_count") >= 12.0
            and self._subpanel_score(facts, "drum_closed_hat_source_score") >= 0.70
        )
        if not is_hat_loop:
            return winning_claim
        return claim_from_folder_path(
            folder_path="Drums/Hi Hats/Closed Hat/Loops",
            source="final_measured_hi_hat_loop_invariant",
            reason="final structure invariant kept a measured high-band hi-hat loop out of shaker/percussion one-shot leaves",
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.92, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_measured_drum_loop_from_fx_or_instrument(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final structure invariant for measured drum loops stolen by FX.

        This is deliberately a broad role repair, not an identity rescue.  It
        only fires when measured drum-loop roles are strong and a real Drum
        Loops candidate exists in the voter/candidate window.  That protects
        kick-clap, hat/top, and percussion loops from FX glitch/riser theft
        without using filenames or forcing a specific drum subtype.
        """
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if winning_claim.family == "Drums" and "drum loops" in path:
            return winning_claim
        if not self._facts_support_final_drum_loop(facts, winning_claim):
            return winning_claim
        return claim_from_folder_path(
            folder_path="Drums/Drum Loops/Loops",
            source="final_measured_drum_loop_invariant",
            reason=(
                "final structure invariant kept strong measured drum-loop "
                "evidence out of FX/instrument leaves after voter disagreement; "
                "kick/drum-loop true-bucket rescue"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.94, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _review_pitched_music_hit_stolen_by_drum_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Review pitched/vocal phrase evidence when it wins a drum leaf.

        Some sax, brass, or vocal stabs are percussive enough to resemble conga,
        tabla, or other hand percussion.  If the final winner is a narrow drum
        one-shot but measured shape says pitched/vocal phrase and instrument
        candidates are also present, the safe architecture choice is review.
        """
        if not self._facts_support_pitched_hit_drum_leaf_conflict(facts, winning_claim):
            return winning_claim
        physics_drum = self._top_physics_drum_guess(facts)
        if physics_drum is not None:
            try:
                physics_confidence = float(physics_drum.get("confidence", 0.0) or 0.0)
            except Exception:
                physics_confidence = 0.0
            evidence = physics_drum.get("evidence") if isinstance(physics_drum, dict) else {}
            if not isinstance(evidence, dict):
                evidence = {}
            parent = (
                facts.evidence.get("parent_eligibility_v2", {})
                if facts is not None and isinstance(getattr(facts, "evidence", None), dict)
                else {}
            )
            allowed = parent.get("allowed_top_families", []) if isinstance(parent, dict) else []
            detected_role = str(evidence.get("detected_parent_role") or "")
            anchor = self._safe_float(evidence.get("drum_anchor_strength"), 0.0)
            branch = self._safe_float(evidence.get("drum_branch_selected_confidence"), 0.0)
            if (
                physics_confidence >= 0.62
                and facts is not None
                and (
                    facts.is_short_hit_like
                    or detected_role in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}
                    or (anchor >= 0.55 and branch >= 0.45)
                    or ("Drums" in allowed and "Instruments" not in allowed)
                )
            ):
                return winning_claim
        if self._facts_support_broad_instrument_phrase_release(facts, winning_claim):
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="final_pitched_hit_broad_instrument_release_invariant",
                reason=(
                    "final structure invariant released a non-percussive pitched "
                    "phrase that narrowly won a drum leaf to broad Instrument Loops"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.92, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if (
            winning_claim.source in {"strong_consensus", "concrete_fx_gate_override", "synthetic_inconsistent_raw"}
            and winning_claim.raw_candidate_score <= 4.0
            and (winning_claim.brain_rank is None or winning_claim.brain_rank <= 4)
            and (winning_claim.physics_rank is None or winning_claim.physics_rank <= 4)
        ):
            return winning_claim
        if (
            winning_claim.source == "candidate_true_bucket_rescue"
            and winning_claim.raw_candidate_score <= 9.0
            and winning_claim.family == "FX"
        ):
            return winning_claim
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            source="final_pitched_hit_drum_leaf_conflict_review",
            reason=(
                "final structure invariant reviewed a pitched/vocal phrase hit that narrowly won a drum one-shot leaf"
            ),
            shared=winning_claim.shared_candidates,
            winner=winning_claim,
            strength=max(0.92, winning_claim.strength),
        )

    def _protect_measured_voice_from_non_voice_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final structure invariant for measured true vocal material.

        This runs after arbitration because true vocal shots and vocal loops can
        be stolen by animal, motor, sax, or generic loop leaves when the raw
        candidate set is noisy.  It uses only measured roles, shape metrics, and
        internal voter candidate rows.  It does not inspect source names.
        """
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        already_voice = winning_claim.family == "Instruments" and "voice" in path
        if already_voice:
            return winning_claim
        if self._facts_support_measured_voiced_fx_one_shot(facts, winning_claim):
            return claim_from_folder_path(
                folder_path="FX/Human and Voice FX/Spoken Voice/One Shots",
                source="final_measured_voiced_fx_one_shot_invariant",
                reason=(
                    "final measured voiced-FX one-shot invariant kept a short "
                    "formant/voiced stab out of drum and vocal-loop buckets"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.94, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if (
            winning_claim.family == "FX"
            and self._facts_have_concrete_fx_lane_agreement(facts)
            and not self._facts_have_hard_clean_keys_authority(facts, winning_claim)
            and not self._facts_support_nonpercussive_pitched_instrument_loop_body(facts, winning_claim)
        ):
            return winning_claim
        if self._facts_support_measured_sax_loop(facts, winning_claim):
            return winning_claim
        if self._facts_support_final_voice_instrument(facts, winning_claim):
            folder_path = self._brain_voice_one_shot_folder(facts) or "Instruments/Voice/Vocal Loops/Loops"
            if self._facts_support_short_true_voice_one_shot(facts):
                folder_path = "Instruments/Voice/Phrase/One Shots"
            return claim_from_folder_path(
                folder_path=folder_path,
                source="final_measured_voice_invariant",
                reason=(
                    "final measured voice invariant kept strong vocal evidence in "
                    "Instruments/Voice after voter/arbiter disagreement"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.95, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if self._facts_support_measured_sax_loop(facts, winning_claim):
            return winning_claim
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if facts is not None and isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        if isinstance(parent, dict) and str(parent.get("role_name") or "") == "fx_tonal_alert_or_siren":
            if not self._facts_support_nonpercussive_pitched_instrument_loop_body(facts, winning_claim):
                return claim_from_folder_path(
                    folder_path="FX/Designed Noise FX/Siren/Long FX",
                    source="final_measured_tonal_alert_siren_invariant",
                    reason=(
                        "final measured tonal-alert/siren invariant blocked false "
                        "voice/instrument-loop broadening because parent eligibility "
                        "measured repeated clean tonal alert behavior"
                    ),
                    shared=winning_claim.shared_candidates,
                    raw_candidate_score=winning_claim.raw_candidate_score,
                    brain_rank=winning_claim.brain_rank,
                    physics_rank=winning_claim.physics_rank,
                    shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                    can_override=True,
                    strength=max(0.94, winning_claim.strength),
                    is_real_candidate=winning_claim.is_real_candidate,
                )
        if self._facts_support_false_voice_loop_broad_instrument_release(facts, winning_claim):
            if self._facts_support_final_drum_loop(facts, winning_claim):
                return claim_from_folder_path(
                    folder_path="Drums/Drum Loops/Loops",
                    source="final_measured_drum_loop_invariant",
                    reason=(
                        "final structure invariant kept drum-loop candidate and measured "
                        "loop evidence out of false-voice broad Instrument Loops"
                    ),
                    shared=winning_claim.shared_candidates,
                    raw_candidate_score=winning_claim.raw_candidate_score,
                    brain_rank=winning_claim.brain_rank,
                    physics_rank=winning_claim.physics_rank,
                    shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                    can_override=True,
                    strength=max(0.94, winning_claim.strength),
                    is_real_candidate=winning_claim.is_real_candidate,
                )
            if self._facts_support_synth_loop(facts, winning_claim):
                folder_path = "Instruments/Synths/Synth Loops"
                if self._facts_support_strong_synth_pad_loop(facts):
                    folder_path = "Instruments/Synths/Pads/Loops"
                elif _shape_vote_from_facts(facts) == "pitched_phrase":
                    folder_path = "Instruments/Synths/Synth Lead/Loops"
                return claim_from_folder_path(
                    folder_path=folder_path,
                    source="final_measured_synth_loop_invariant",
                    reason=(
                        "final structure invariant kept measured synth-loop body "
                        "out of false voice broadening before generic Instrument Loops"
                    ),
                    shared=winning_claim.shared_candidates,
                    raw_candidate_score=winning_claim.raw_candidate_score,
                    brain_rank=winning_claim.brain_rank,
                    physics_rank=winning_claim.physics_rank,
                    shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                    can_override=True,
                    strength=max(0.93, winning_claim.strength),
                    is_real_candidate=winning_claim.is_real_candidate,
                )
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="final_false_voice_loop_broad_instrument_invariant",
                reason=(
                    "final structure invariant broadened a false voice rescue to "
                    "Instrument Loops because measured facts supported a clean "
                    "pitched music loop without reliable voice-candidate support"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.90, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        return winning_claim

    def _protect_measured_bass_loop_from_fx_or_review(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final structure invariant for clean bass phrases misread as FX."""
        if winning_claim.family == "Instruments" and "bass" in self._norm_claim_path(winning_claim.folder_path):
            return winning_claim
        if not self._facts_support_clean_bass_loop(facts, winning_claim):
            return winning_claim
        return claim_from_folder_path(
            folder_path="Instruments/Bass/Bass Loops",
            source="final_measured_bass_loop_invariant",
            reason=("final structure invariant kept a clean measured bass phrase out of FX/review/drum broad buckets"),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.94, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_measured_synth_loop_from_fx_or_review(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final structure invariant for stable synth loops stolen by FX."""
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if winning_claim.family == "Instruments" and "synth" in path and "loop" in path:
            return winning_claim
        if winning_claim.family == "Instruments" and "instrument loops" not in path:
            can_correct_keys_decoy = bool(
                ("/keys/" in path or "piano" in path)
                and self._facts_support_strong_synth_pad_loop(facts)
                and self._facts_support_synth_loop(facts, winning_claim)
            )
            if not can_correct_keys_decoy:
                return winning_claim
        if not self._facts_support_synth_loop(facts, winning_claim):
            return winning_claim
        folder_path = "Instruments/Synths/Synth Loops"
        if self._facts_support_strong_synth_pad_loop(facts):
            folder_path = "Instruments/Synths/Pads/Loops"
        elif _shape_vote_from_facts(facts) == "pitched_phrase":
            folder_path = "Instruments/Synths/Synth Lead/Loops"
        return claim_from_folder_path(
            folder_path=folder_path,
            source="final_measured_synth_loop_invariant",
            reason=(
                "final structure invariant used measured stable synth-loop body "
                "and internal synth voter evidence instead of FX/review routing"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.93, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _preserve_measured_instrument_branch_after_broadening(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Preserve measured source branch after role/depth broadening.

        Broadening is allowed to fix structure, for example one-shot to loop.
        It is not allowed to keep a contradicted source identity when the
        PhysicsVoter branch is stronger.  This is the authority contract that
        keeps category-specific repairs from becoming global category theft.
        """
        if facts is None or winning_claim.family != "Instruments":
            return winning_claim
        if winning_claim.source not in {
            "parent_eligibility_broad_bucket",
            "candidate_true_bucket_rescue",
            "final_instrument_one_shot_leaf_broad_instrument_loop_invariant",
            "final_false_voice_loop_broad_instrument_invariant",
            "final_clean_pitched_instrument_loop_invariant",
            "final_blocked_instrument_loop_release_invariant",
            "final_pitched_hit_broad_instrument_release_invariant",
            "final_pitched_hit_broad_instrument_loop_invariant",
            "placement_depth_broad_bucket",
        }:
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if (
            "sax" in path or "woodwind" in path or "brass" in path
        ) and self._facts_support_mixed_instrument_loop_sax_decoy(facts, winning_claim):
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="final_mixed_instrument_loop_sax_decoy_invariant",
                reason=(
                    "final authority contract replaced weak sax/woodwind broadening "
                    "with Instrument Loops because measured evidence described a "
                    "layered mixed instrumental loop"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.92, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if not ("loop" in path or "one shot" in path or "one shots" in path):
            return winning_claim
        measured_branch_folder = self._measured_instrument_loop_folder_from_physics_branch(
            facts,
            path,
            winning_claim,
        )
        # If measured structure strongly supports a drum loop, avoid forcing an
        # Instruments branch via final broadening. This prevents percussive
        # material from being misrouted to instrument loop folders.
        try:
            measured_role = (facts.evidence.get("measured_role_name") if isinstance(facts.evidence, dict) else None) or ""
            measured_shape = (facts.evidence.get("measured_shape_name") if isinstance(facts.evidence, dict) else None) or ""
            measured_shape_conf = float((facts.evidence.get("measured_shape_confidence") if isinstance(facts.evidence, dict) else 0.0) or 0.0)
        except Exception:
            measured_role = ""
            measured_shape = ""
            measured_shape_conf = 0.0
        if _has_drum_loop_structure_support(role="", measured_role=measured_role, shape=measured_shape, shape_confidence=measured_shape_conf):
            return winning_claim
        # Also consult arbiter's richer drum-loop support check which looks at
        # role, direct-body, and candidate-level evidence. If it deems the
        # file a drum loop, avoid forcing an Instruments branch.
        try:
            if self._facts_support_final_drum_loop(facts, winning_claim):
                return winning_claim
        except Exception:
            pass
        if not measured_branch_folder:
            return winning_claim
        if self._norm_claim_path(measured_branch_folder) == path:
            return winning_claim
        return claim_from_folder_path(
            folder_path=measured_branch_folder,
            source="final_measured_branch_loop_broad_bucket",
            reason=(
                "final authority contract preserved the strongest measured "
                "instrument branch while keeping the measured loop structure"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.91, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_clean_keys_loop_from_sax_or_fx(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final structure invariant for clean keys loops stolen by sax/FX."""
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if winning_claim.family == "Instruments" and ("/keys" in path or "piano" in path):
            return winning_claim
        if self._facts_show_true_voice_or_vocal_one_shot(facts):
            return winning_claim
        if self._facts_support_final_drum_loop(facts, winning_claim):
            return winning_claim
        if (
            self._facts_have_concrete_fx_lane_agreement(facts)
            and not self._facts_have_hard_clean_keys_authority(facts, winning_claim)
            and not self._facts_support_nonpercussive_pitched_instrument_loop_body(facts, winning_claim)
        ):
            return winning_claim
        # This invariant may refine unsafe sax/FX/review/broad-loop placements, but
        # it must never steal already plausible concrete instrument or voice leaves.
        if not (
            winning_claim.family in {"FX", "_TO_REVIEW"}
            or "sax" in path
            or "woodwind" in path
            or "instrument loops" in path
            or "measured role conflict" in path
        ):
            return winning_claim
        if not (
            self._facts_support_clean_keys_loop(facts, winning_claim)
            or self._facts_support_clean_vintage_keys_loop(facts)
        ):
            return winning_claim
        folder_path = "Instruments/Keys/Electric Piano/Loops"
        if self._facts_support_acoustic_piano_loop(facts, winning_claim):
            folder_path = "Instruments/Keys/Piano/Loops"
        return claim_from_folder_path(
            folder_path=folder_path,
            source="final_clean_keys_loop_invariant",
            reason=(
                "final structure invariant kept a clean tonal keys-like loop out "
                "of sax/FX leaves using measured body and internal keys evidence"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.93, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_measured_sax_loop_from_fx_or_review(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final structure invariant for measured sax/reed loop bodies."""
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        layer = self._physics_layer(facts)
        clean_keys_loop_layer_signal = bool(
            bool(layer.get("instrument_clean_electric_keys_loop_signal"))
            and str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "") == "KeysPiano"
            and self._safe_number(
                layer.get("instrument_branch_selected_confidence")
                or layer.get("physics_layer_branch_confidence"),
                0.0,
            ) >= 0.80
        )
        if (
            ("sax" in path or "woodwind" in path or "instrument loops" in path)
            and (self._facts_support_clean_keys_loop(facts, winning_claim) or clean_keys_loop_layer_signal)
        ):
            folder_path = "Instruments/Keys/Electric Piano/Loops"
            if self._facts_support_acoustic_piano_loop(facts, winning_claim):
                folder_path = "Instruments/Keys/Piano/Loops"
            return claim_from_folder_path(
                folder_path=folder_path,
                source="final_clean_keys_loop_before_sax_decoy",
                reason=(
                    "final sax-decoy guard preserved measured clean keys/electric-piano "
                    "loop evidence before broad Instrument Loops"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.93, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if self._facts_support_mixed_instrument_loop_sax_decoy(facts, winning_claim):
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="final_mixed_instrument_loop_sax_decoy_invariant",
                reason=(
                    "final structure invariant broadened weak sax/woodwind routing "
                    "to Instrument Loops because measured evidence described a "
                    "layered mixed instrumental loop"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.92, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if (
            self._compound_music_loop_support(facts) >= 0.68
            and ("instrument loops" in path or "mixed musical" in path or "multi instrument" in path)
            and not self._facts_support_measured_sax_loop(facts, winning_claim)
        ):
            return winning_claim
        if (
            winning_claim.family == "Instruments"
            and "sax" in path
            and "loop" in path
            and "one shot" not in path
            and "one shots" not in path
        ):
            if self._facts_support_measured_sax_loop(facts, winning_claim):
                return winning_claim
            if self._facts_support_clean_pitched_instrument_loop(
                facts, winning_claim
            ) or self._facts_have_non_woodwind_branch_identity_conflict(facts):
                return claim_from_folder_path(
                    folder_path="Instruments/Instrument Loops/Loops",
                    source="final_false_sax_leaf_broad_instrument_loop_invariant",
                    reason=(
                        "final structure invariant broadened a sax leaf to "
                        "Instrument Loops because measured facts supported a "
                        "clean pitched loop but not a sax/reed body"
                    ),
                    shared=winning_claim.shared_candidates,
                    raw_candidate_score=winning_claim.raw_candidate_score,
                    brain_rank=winning_claim.brain_rank,
                    physics_rank=winning_claim.physics_rank,
                    shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                    can_override=True,
                    strength=max(0.90, winning_claim.strength),
                    is_real_candidate=winning_claim.is_real_candidate,
                )
            return winning_claim
        if not self._facts_support_measured_sax_loop(facts, winning_claim):
            return winning_claim
        return claim_from_folder_path(
            folder_path="Instruments/Woodwinds/Saxophone/Loops",
            source="final_measured_sax_loop_invariant",
            reason=(
                "final structure invariant kept measured sax/reed loop body out of FX/review/generic sibling outcomes"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.94, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_compound_music_loop_from_specific_instrument_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Broaden over-specific instrument leaves for measured compound loops."""
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if winning_claim.family != "Instruments":
            return winning_claim
        if "instrument loops" in path or "mixed musical" in path or "multi instrument" in path:
            return winning_claim
        if "one shot" in path or "one shots" in path:
            return winning_claim
        if not (
            self._path_is_sax_or_reed(path)
            or self._path_is_synth(path)
            or self._path_is_mallet_or_bell(path)
            or self._path_is_strings(path)
            or "/brass/" in path
        ):
            return winning_claim
        support = self._compound_music_loop_support(facts)
        if support < 0.68:
            return winning_claim
        return claim_from_folder_path(
            folder_path="Instruments/Instrument Loops/Loops",
            source="final_compound_music_broad_instrument_loop_invariant",
            reason=(
                "final structure invariant broadened a specific instrument "
                "leaf because measured physics/shape evidence described a "
                "layered compound musical loop"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.92, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _compound_music_loop_support(self, facts: SharedAudioFacts | None) -> float:
        """Return strongest source-name-blind compound-loop structure support."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return 0.0
        evidence_rows: list[dict] = []
        direct = facts.evidence.get("physics_layer_decision")
        if isinstance(direct, dict):
            evidence_rows.append(direct)
        physics_result = facts.evidence.get("physics_vote_result")
        if isinstance(physics_result, dict):
            guesses = physics_result.get("top_guesses")
            if isinstance(guesses, list):
                for guess in guesses[:8]:
                    if not isinstance(guess, dict):
                        continue
                    guess_evidence = guess.get("evidence")
                    if isinstance(guess_evidence, dict):
                        evidence_rows.append(guess_evidence)
        best = 0.0
        for evidence in evidence_rows:
            try:
                strength = float(evidence.get("compound_music_strength", 0.0) or 0.0)
                branch_confidence = float(evidence.get("physics_layer_branch_confidence", 0.0) or 0.0)
                mixed_branch = float(evidence.get("instrument_branch_MixedInstrument", 0.0) or 0.0)
            except Exception:
                continue
            branch = str(evidence.get("physics_layer_branch") or evidence.get("instrument_branch_selected") or "")
            prefer_broad = bool(evidence.get("compound_music_prefer_broad_loop"))
            if not prefer_broad:
                continue
            if branch != "MixedInstrument" and branch_confidence < 0.70 and mixed_branch < 0.70:
                continue
            best = max(best, strength, branch_confidence, mixed_branch)
        shape = facts.evidence.get("shape_vote")
        if isinstance(shape, dict):
            primary = str(shape.get("primary_shape") or "")
            try:
                confidence = float(shape.get("confidence", 0.0) or 0.0)
                solo = float(shape.get("solo_isolation_score", 0.0) or 0.0)
                layered = float(shape.get("layered_loop_score", 0.0) or 0.0)
                plus_fx = float(shape.get("instrument_plus_fx_loop_score", 0.0) or 0.0)
                repetition = float(shape.get("true_repetition_score", 0.0) or 0.0)
            except Exception:
                confidence = solo = layered = plus_fx = repetition = 0.0
            shape_scores = shape.get("shape_scores")
            top_shape_names: set[str] = set()
            if isinstance(shape_scores, list):
                for item in shape_scores[:4]:
                    if isinstance(item, (list, tuple)) and item:
                        top_shape_names.add(str(item[0]))
            compound_shapes = {
                "compound_musical_loop",
                "mixed_instrument_loop",
                "instrument_plus_fx_loop",
                "layered_phrase",
            }
            if (
                (primary in compound_shapes or bool(top_shape_names & compound_shapes))
                and max(layered, plus_fx, confidence) >= 0.68
                and repetition >= 0.48
                and solo <= 0.58
            ):
                best = max(best, layered, plus_fx, confidence)
        return best

    def _facts_support_mixed_instrument_loop_sax_decoy(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True when a weak sax/reed identity is really a mixed music loop.

        This is source-name blind. It uses the existing voter/facts payload only.
        The goal is not to choose a better solo instrument leaf. It prevents a
        sax/woodwind broadening path from keeping the wrong parent when the
        measured body and competing panels say layered/mixed instrumental loop.
        """
        if facts is None:
            return False
        if self._facts_support_strong_synth_pad_loop(facts):
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {"pitched_phrase", "vocal_phrase", "repeated_phrase_loop", "sustained_pad", "bass_phrase"}:
            return False
        pitched_event = self._shape_number(facts, "pitched_event_ratio")
        sustained_tonal = self._shape_number(facts, "sustained_tonal_frame_ratio")
        f0_voiced = self._shape_number(facts, "f0_voiced_ratio")
        percussive = self._shape_number(facts, "percussive_event_ratio")
        drumlike = self._shape_number(facts, "drumlike_frame_ratio")
        true_repetition = max(
            self._shape_number(facts, "true_repetition_score"),
            self._shape_number(facts, "onset_true_repetition_likelihood"),
        )
        instrument_plus_fx = self._shape_number(facts, "instrument_plus_fx_loop_score")
        layered_loop = self._shape_number(facts, "layered_loop_score")
        onset_count = self._shape_number(facts, "onset_count")
        measured_loop = max(
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "pitched_music_phrase"),
            self._measured_role_value(facts, "pitched_reed_or_instrument_loop"),
        )
        loop_body = bool(
            measured_loop >= 0.70
            or true_repetition >= 0.62
            or instrument_plus_fx >= 0.50
            or (onset_count >= 16.0 and shape_confidence >= 0.78)
        )
        if not (
            shape_confidence >= 0.70
            and pitched_event >= 0.84
            and sustained_tonal >= 0.78
            and f0_voiced >= 0.64
            and percussive <= 0.16
            and drumlike <= 0.18
            and loop_body
        ):
            return False
        sax_score = self._subpanel_score(facts, "woodwind_sax_score")
        reed_score = self._subpanel_score(facts, "reed_wind_score")
        woodwind_strength = max(sax_score, reed_score)
        direct_human_voice_strength = max(
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
            self._subpanel_score(facts, "voice_score"),
        )
        human_voice_strength = max(
            direct_human_voice_strength,
            self._subpanel_score(facts, "voice_choir_score"),
        )
        if human_voice_strength >= 0.68 and human_voice_strength >= woodwind_strength + 0.05:
            return False
        strong_sax_reed_body = bool(
            sax_score >= 0.62
            and reed_score >= 0.52
            and sax_score >= direct_human_voice_strength + 0.04
            and sax_score >= self._subpanel_score(facts, "struck_keys_score") + 0.12
            and shape in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad", "vocal_phrase"}
        )
        if strong_sax_reed_body:
            return False
        physics_top_sax_phrase = bool(
            (
                self._facts_have_internal_candidate(
                    facts,
                    ("sax", "saxophone"),
                    top_family="Instruments",
                    max_rank=1,
                    max_score=0.75,
                )
                or self._shared_candidate_has_top_family(
                    winning_claim,
                    ("sax", "saxophone"),
                    top_family="Instruments",
                    max_score=8.0,
                    max_brain_rank=4,
                    max_physics_rank=4,
                )
            )
            and sax_score >= 0.54
            and reed_score >= 0.52
            and direct_human_voice_strength <= 0.52
            and self._subpanel_score(facts, "struck_keys_score") <= 0.50
            and shape in {"bass_phrase", "pitched_phrase", "pitched_phrase_shape", "sustained_pad", "vocal_phrase"}
        )
        if physics_top_sax_phrase:
            return False
        non_woodwind_scores = [
            self._subpanel_score(facts, "synth_tonal_source_score"),
            self._subpanel_score(facts, "synth_chord_score"),
            self._subpanel_score(facts, "synth_lead_score"),
            self._subpanel_score(facts, "synth_pad_score"),
            self._subpanel_score(facts, "bowed_string_score"),
            self._subpanel_score(facts, "string_cello_score"),
            self._subpanel_score(facts, "string_violin_score"),
            self._subpanel_score(facts, "plucked_string_score"),
            self._subpanel_score(facts, "struck_keys_score"),
            self._subpanel_score(facts, "voice_choir_score"),
            self._subpanel_score(facts, "voice_score"),
        ]
        competing_non_woodwind = max(non_woodwind_scores) if non_woodwind_scores else 0.0
        competing_count = sum(1 for value in non_woodwind_scores if value >= max(0.46, woodwind_strength - 0.08))
        has_broad_loop_candidate = bool(
            self._shared_candidate_has_top_family(
                winning_claim,
                ("instrument loops", "mixed musical", "multi instrument"),
                top_family="Instruments",
                max_score=48.0,
                max_brain_rank=8,
                max_physics_rank=6,
            )
            or self._facts_have_internal_candidate(
                facts,
                ("instrument loops", "mixed musical", "multi instrument"),
                top_family="Instruments",
                max_rank=6,
                max_score=1.40,
            )
        )
        layer = self._physics_layer(facts)
        woodwind_source = (
            bool(
                layer.get("instrument_woodwind_source_signal")
                or layer.get("instrument_reed_woodwind_source_signal")
                or layer.get("instrument_clean_tonal_reed_solo_signal")
                or layer.get("instrument_dark_low_mid_reed_loop_signal")
            )
            if isinstance(layer, dict)
            else False
        )
        decisive_sax = bool(woodwind_source or self._facts_have_decisive_physics_sax_candidate(facts))
        if decisive_sax and competing_count < 3 and not has_broad_loop_candidate:
            return False
        return bool(
            has_broad_loop_candidate
            and competing_non_woodwind >= woodwind_strength - 0.08
            and competing_count >= 2
            and max(instrument_plus_fx, layered_loop, true_repetition, measured_loop) >= 0.50
        )

    def _facts_support_nonpercussive_pitched_instrument_loop_body(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True for clean musical loops that must not be forced to FX.

        This is a structure/body check used only to block late FX siren/alarm
        invariants. It does not choose a specific instrument from a filename.
        It requires measured loop/tonal body plus internal instrument candidate
        support, and rejects drum-like or transition-like bodies.
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape in {"transition_riser", "transition_downlifter", "fx_sweep", "noise_fx"}:
            return False
        pitched_role = max(
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "pitched_reed_or_instrument_loop"),
            self._measured_role_value(facts, "synth_loop"),
            self._measured_role_value(facts, "bass_loop"),
        )
        pitched_body = bool(
            pitched_role >= 0.62
            or (
                shape in {
                    "pitched_phrase",
                    "pitched_phrase_shape",
                    "sustained_pad",
                    "bass_phrase",
                    "repeated_phrase_loop",
                    "compound_musical_loop",
                    "mixed_instrument_loop",
                    "instrument_plus_fx_loop",
                }
                and shape_confidence >= 0.68
            )
        )
        if not pitched_body:
            return False
        if self._shape_number(facts, "pitched_event_ratio") < 0.56:
            return False
        if self._shape_number(facts, "sustained_tonal_frame_ratio") < 0.54:
            return False
        if self._shape_number(facts, "percussive_event_ratio") > 0.32:
            return False
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.30:
            return False
        instrument_candidates = self._shared_candidate_count_top_family(
            winning_claim,
            (
                "guitar",
                "synth",
                "keys",
                "piano",
                "rhodes",
                "strings",
                "string",
                "cello",
                "violin",
                "woodwind",
                "sax",
                "brass",
                "instrument loops",
                "mixed musical",
                "bass",
            ),
            top_family="Instruments",
            max_score=48.0,
            max_brain_rank=18,
            max_physics_rank=18,
        )
        layer = self._physics_layer(facts)
        branch_support = False
        if isinstance(layer, dict):
            branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
            branch_conf = self._safe_float(
                layer.get("physics_layer_branch_confidence", layer.get("instrument_branch_selected_confidence")),
                0.0,
            )
            if branch in self.BRANCH_LOOP_TARGETS and branch_conf >= 0.58:
                branch_support = True
            if bool(layer.get("instrument_clean_electric_keys_loop_signal")):
                branch_support = True
            if bool(layer.get("instrument_dark_low_mid_reed_loop_signal")):
                branch_support = True
        return bool(instrument_candidates >= 1 or branch_support)

    def _protect_clean_pitched_instrument_loop_from_fx_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Final structure invariant for clean pitched loops stolen by FX/review.

        This is broad structure routing only.  It does not choose sax, guitar,
        keys, or voice identity.  It is allowed after arbitration because the
        measured body says stable non-percussive pitched loop and the shared
        candidate window contains instrument-family support.
        """
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if winning_claim.family not in {"FX", "_TO_REVIEW", "Instruments"}:
            return winning_claim
        if (
            winning_claim.family == "FX"
            and self._facts_have_concrete_fx_lane_agreement(facts)
            and not self._facts_have_hard_clean_keys_authority(facts, winning_claim)
            and not self._facts_support_nonpercussive_pitched_instrument_loop_body(facts, winning_claim)
        ):
            return winning_claim
        if self._facts_support_clean_bass_loop(facts, winning_claim):
            return claim_from_folder_path(
                folder_path="Instruments/Bass/Bass Loops",
                source="final_measured_bass_loop_invariant",
                reason=(
                    "final structure invariant used decisive PhysicsVoter Bass "
                    "branch evidence instead of broad Instrument Loops"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.94, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if (
            winning_claim.family == "Instruments"
            and "loop" in path
            and "one shot" not in path
            and "one shots" not in path
        ):
            return winning_claim
        if not self._facts_support_clean_pitched_instrument_loop(facts, winning_claim):
            return winning_claim
        return claim_from_folder_path(
            folder_path="Instruments/Instrument Loops/Loops",
            source="final_clean_pitched_instrument_loop_invariant",
            reason=(
                "final structure invariant sent clean non-percussive pitched loop "
                "evidence to broad Instruments instead of a contradicted FX leaf"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.91, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _protect_instrument_one_shot_leaf_from_measured_loop(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Broaden instrument one-shot leaves when measured body is a loop.

        This is a final structure invariant. It does not infer a new source
        identity from filenames. It only prevents a concrete one-shot leaf from
        surviving when measured roles say the audio is a pitched musical loop.
        """
        if facts is None or winning_claim.family != "Instruments":
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if "one shot" not in path and "one shots" not in path:
            return winning_claim
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        measured_loop = max(
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "pitched_reed_or_instrument_loop"),
            self._measured_role_value(facts, "bass_loop"),
            self._measured_role_value(facts, "synth_loop"),
        )
        repeated_pitched_loop_body = bool(
            shape == "repeated_phrase_loop"
            and shape_confidence >= 0.74
            and self._shape_number(facts, "pitched_event_ratio") >= 0.80
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.70
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.70
        )
        structural_loop_body = bool(
            shape
            in {
                "pitched_phrase",
                "pitched_phrase_shape",
                "bass_phrase",
                "sustained_pad",
                "compound_musical_loop",
                "mixed_instrument_loop",
                "instrument_plus_fx_loop",
            }
            and shape_confidence >= 0.86
            and getattr(facts, "is_loop_like", False)
            and self._shape_number(facts, "true_repetition_score") >= 0.55
            and self._shape_number(facts, "onset_span_ratio") >= 0.50
        )
        if measured_loop < 0.86 and not structural_loop_body and not repeated_pitched_loop_body:
            return winning_claim
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.20:
            return winning_claim
        if self._shape_number(facts, "percussive_event_ratio") > 0.45:
            return winning_claim
        if "voice" in path or "vocal" in path:
            return winning_claim
        if self._facts_support_clean_vintage_keys_loop(facts):
            return claim_from_folder_path(
                folder_path="Instruments/Keys/Electric Piano/Loops",
                source="final_clean_vintage_keys_loop_invariant",
                reason=(
                    "final structure invariant used clean low-noise keys/electric-piano "
                    "loop evidence instead of a contradicted sax/brass one-shot leaf"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.92, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if self._facts_support_mixed_instrument_loop_sax_decoy(facts, winning_claim):
            folder_path = "Instruments/Instrument Loops/Loops"
            source = "final_mixed_instrument_loop_sax_decoy_invariant"
        else:
            measured_branch_folder = self._measured_instrument_loop_folder_from_physics_branch(
                facts,
                path,
                winning_claim,
            )
            if measured_branch_folder:
                folder_path = measured_branch_folder
                source = "final_measured_branch_loop_broad_bucket"
            elif any(token in path for token in ("woodwind", "sax", "flute", "brass", "horn")):
                folder_path = "Instruments/Brass and Woodwinds/Loops"
                source = "parent_eligibility_broad_bucket"
            else:
                folder_path = "Instruments/Instrument Loops/Loops"
                source = "final_instrument_one_shot_leaf_broad_instrument_loop_invariant"
        return claim_from_folder_path(
            folder_path=folder_path,
            source=source,
            reason=(
                "final structure invariant broadened an instrument one-shot leaf "
                "because measured audio body is a pitched musical loop"
            ),
            shared=winning_claim.shared_candidates,
            raw_candidate_score=winning_claim.raw_candidate_score,
            brain_rank=winning_claim.brain_rank,
            physics_rank=winning_claim.physics_rank,
            shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
            can_override=True,
            strength=max(0.90, winning_claim.strength),
            is_real_candidate=winning_claim.is_real_candidate,
        )

    def _review_conflicted_instrument_identity_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Review concrete instrument leaves when low-level panels disagree.

        This is a source-name-blind safety layer. It does not force a sax,
        guitar, voice, or synth answer. It only blocks a confident concrete
        leaf when the measured physics says multiple instrument identities are
        plausible and none has enough authority to claim the terminal folder.
        """
        if facts is None or winning_claim.family != "Instruments":
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        concrete_leaf = any(
            token in path
            for token in (
                "guitar",
                "sax",
                "woodwind",
                "brass",
                "voice",
                "vocal",
                "synth",
                "keys",
                "piano",
                "strings",
            )
        )
        if not concrete_leaf or "instrument loops" in path:
            return winning_claim
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        phrase_like = bool(
            shape in {
                "pitched_phrase",
                "pitched_phrase_shape",
                "vocal_phrase",
                "solo_phrase",
                "bass_phrase",
                "sustained_pad",
                "repeated_phrase_loop",
            }
            and shape_confidence >= 0.68
        )
        if not phrase_like:
            return winning_claim
        if (
            winning_claim.source == "final_measured_synth_loop_invariant"
            and self._measured_score(
                facts,
                "synth_tonal_source_score",
                "synth_pad_score",
                "synth_lead_score",
                "synth_chord_score",
            )
            >= 0.58
        ):
            return winning_claim
        if "synth" in path and self._facts_support_synth_loop(facts, winning_claim):
            return claim_from_folder_path(
                folder_path="Instruments/Synths/Synth Loops",
                source="final_conflicted_identity_synth_loop_preserve",
                reason=(
                    "final identity firewall preserved a measured synth-loop "
                    "body instead of flattening it to generic Instrument Loops"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.92, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if (
            winning_claim.source in {
                "final_measured_voice_invariant",
                "final_short_true_voice_one_shot_invariant",
            }
            and self._facts_support_final_voice_instrument(facts, winning_claim)
        ):
            return winning_claim
        if (
            winning_claim.source in {"final_measured_sax_loop_invariant", "profile_candidate_sax_leaf_claim"}
            and self._measured_score(
                facts,
                "reed_wind_authority_score",
                "reed_wind_score",
                "woodwind_sax_score",
                "woodwind_flute_score",
            )
            >= 0.54
        ):
            return winning_claim
        if not self._facts_have_conflicted_instrument_identity_panels(facts, path):
            return winning_claim
        broad_path = self._broad_instrument_path_for_conflicted_identity(winning_claim, facts, path)
        if broad_path:
            return claim_from_folder_path(
                folder_path=broad_path,
                source="final_conflicted_instrument_identity_broadening",
                reason=(
                    "final source-blind identity firewall broadened a concrete "
                    "instrument leaf because low-level source panels agreed on "
                    "Instruments but not the terminal identity"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.91, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            source="final_conflicted_instrument_identity_review",
            reason=(
                "final source-blind identity firewall reviewed a concrete "
                "instrument leaf because plucked/reed/voice/synth panels were "
                "too conflicted even for a safe Instrument parent"
            ),
            shared=winning_claim.shared_candidates,
            winner=winning_claim,
            strength=max(0.92, winning_claim.strength),
        )

    def _broad_instrument_path_for_conflicted_identity(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
        path: str,
    ) -> str:
        """Return a safe Instrument parent when only identity panels conflict.

        This keeps the voter/arbiter contract intact: low-level voters may prove
        the parent family and sometimes a branch, but a conflicted terminal leaf
        should broaden inside Instruments instead of becoming review. Review is
        reserved for cross-family or no-parent-authority conflicts.
        """
        if facts is None:
            return ""
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape_confidence < 0.66:
            return ""
        if shape not in {
            "pitched_phrase",
            "pitched_phrase_shape",
            "vocal_phrase",
            "solo_phrase",
            "bass_phrase",
            "sustained_pad",
            "repeated_phrase_loop",
            "mixed_instrument_loop",
            "compound_musical_loop",
        }:
            return ""
        drumlike = max(
            self._shape_number(facts, "drumlike_frame_ratio"),
            self._shape_number(facts, "percussive_event_ratio"),
            self._measured_role_value(facts, "drum_loop"),
            self._measured_role_value(facts, "percussive_drum_loop"),
        )
        if drumlike >= 0.48:
            return ""
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        parent_path = str(parent.get("broad_folder_path") or "") if isinstance(parent, dict) else ""
        parent_role = str(parent.get("role_name") or "").lower() if isinstance(parent, dict) else ""
        parent_confidence = self._safe_float(parent.get("confidence"), 0.0) if isinstance(parent, dict) else 0.0
        if parent_path.startswith("Instruments/") and parent_confidence >= 0.55:
            return parent_path
        if "synth" in path and winning_claim.source == "final_measured_synth_loop_invariant":
            return winning_claim.folder_path
        if ("sax" in path or "woodwind" in path or "brass" in path) and winning_claim.source in {
            "final_measured_sax_loop_invariant",
            "profile_candidate_sax_leaf_claim",
        }:
            return winning_claim.folder_path
        synth_score = self._measured_score(
            facts,
            "synth_tonal_source_score",
            "synth_lead_score",
            "synth_pad_score",
            "synth_chord_score",
        )
        reed_score = max(
            self._measured_score(facts, "reed_wind_authority_score"),
            self._measured_score(facts, "reed_wind_score", "woodwind_sax_score", "woodwind_flute_score"),
        )
        plucked_score = max(
            self._measured_score(facts, "plucked_string_authority_score"),
            self._measured_score(facts, "plucked_string_score", "guitar_acoustic_score", "guitar_electric_score"),
        )
        keys_score = self._measured_score(facts, "struck_keys_authority_score", "struck_keys_score")
        voice_score = self._measured_score(facts, "voice_score", "voice_choir_score", "human_spoken_voice_score")
        physics_top = self._top_physics_guess_path(facts)
        physics_top_l = physics_top.lower()
        if synth_score >= 0.58 and "synth" in physics_top_l:
            if shape in {"sustained_pad", "pitched_phrase", "pitched_phrase_shape", "vocal_phrase", "bass_phrase"}:
                return "Instruments/Synths/Pads/Loops"
            return "Instruments/Synths/Synth Loops/Loops"
        if reed_score >= 0.58 and ("sax" in physics_top_l or "woodwind" in physics_top_l or "brass" in physics_top_l):
            return "Instruments/Woodwinds/Saxophone/Loops"
        if max(synth_score, reed_score, plucked_score, keys_score, voice_score) >= 0.50:
            return "Instruments/Instrument Loops/Loops"
        if physics_top_l.startswith("instruments/"):
            return "Instruments/Instrument Loops/Loops"
        if winning_claim.family == "Instruments":
            return "Instruments/Instrument Loops/Loops"
        return ""

    def _facts_have_conflicted_instrument_identity_panels(
        self,
        facts: SharedAudioFacts | None,
        path: str,
    ) -> bool:
        """Return True when terminal source panels are too close to trust."""
        plucked_score = self._measured_score(
            facts,
            "plucked_string_score",
            "guitar_acoustic_score",
            "guitar_electric_score",
            "guitar_nylon_score",
        )
        plucked_authority = self._measured_score(facts, "plucked_string_authority_score")
        reed_score = self._measured_score(
            facts,
            "reed_wind_score",
            "woodwind_sax_score",
            "woodwind_flute_score",
            "brass_trumpet_score",
        )
        reed_authority = self._measured_score(facts, "reed_wind_authority_score")
        voice_score = self._measured_score(
            facts,
            "voice_score",
            "human_spoken_voice_score",
            "voice_choir_score",
            "human_breath_mouth_score",
        )
        synth_score = self._measured_score(
            facts,
            "synth_tonal_source_score",
            "synth_lead_score",
            "synth_pad_score",
            "synth_chord_score",
        )
        keys_score = self._measured_score(
            facts,
            "struck_keys_score",
            "struck_keys_authority_score",
            "keys_hammer_attack_score",
            "keys_tonal_decay_score",
        )
        competing = {
            "plucked": max(plucked_score, plucked_authority),
            "reed": max(reed_score, reed_authority),
            "voice": voice_score,
            "synth": synth_score,
            "keys": keys_score,
        }
        if "guitar" in path:
            target = competing["plucked"]
            nearest_decoy = max(competing["reed"], competing["voice"], competing["synth"], competing["keys"])
            strong_target = bool(plucked_authority >= 0.68 and target >= nearest_decoy + 0.08)
            return bool(not strong_target and target >= 0.42 and nearest_decoy >= max(0.54, target - 0.08))
        if "sax" in path or "woodwind" in path or "brass" in path:
            target = competing["reed"]
            nearest_decoy = max(competing["plucked"], competing["voice"], competing["synth"], competing["keys"])
            strong_target = bool(reed_authority >= 0.68 and target >= nearest_decoy + 0.08)
            return bool(not strong_target and target >= 0.42 and nearest_decoy >= max(0.54, target - 0.08))
        if "voice" in path or "vocal" in path:
            target = competing["voice"]
            nearest_decoy = max(competing["reed"], competing["plucked"], competing["synth"], competing["keys"])
            strong_target = bool(target >= 0.78 and target >= nearest_decoy + 0.08)
            return bool(not strong_target and target >= 0.50 and nearest_decoy >= target - 0.06)
        if "synth" in path:
            target = competing["synth"]
            nearest_decoy = max(competing["reed"], competing["plucked"], competing["voice"], competing["keys"])
            strong_target = bool(target >= 0.74 and target >= nearest_decoy + 0.08)
            return bool(not strong_target and target >= 0.50 and nearest_decoy >= target - 0.06)
        return False

    def _review_uncertain_generic_loop_release(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Review broad loop releases when they are structure-only guesses.

        The previous behavior converted several conflicted loop bodies into
        confident Instrument Loops or Synth Loops. Broad buckets are useful,
        but they still need a reliable source-family authority. When the loop
        structure is real and the source panels stay weak, review is more
        honest than a forced producer folder.
        """
        if facts is None or winning_claim.family != "Instruments":
            return winning_claim
        if winning_claim.source not in {
            "final_false_voice_loop_broad_instrument_invariant",
            "final_measured_synth_loop_invariant",
        }:
            return winning_claim
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if "instrument loops" not in path and "synth" not in path:
            return winning_claim
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {"bass_phrase", "pitched_phrase", "pitched_phrase_shape", "vocal_phrase", "sustained_pad"}:
            return winning_claim
        event_count = self._measured_score(facts, "onset_count", "physics_subpanel_event_count")
        repeat = max(
            self._measured_score(facts, "true_repetition_score"),
            self._measured_score(facts, "onset_true_repetition_likelihood"),
            self._measured_score(facts, "loop_tempo_confidence"),
            self._measured_score(facts, "loop_pulse_clarity"),
            self._measured_score(facts, "loop_onset_periodicity"),
            self._measured_score(facts, "pulse_regularity"),
        )
        if shape_confidence < 0.88 or event_count < 16.0 or repeat < 0.50:
            return winning_claim
        strongest_source = max(
            self._measured_score(facts, "synth_tonal_source_score", "synth_pad_score", "synth_lead_score", "synth_chord_score"),
            self._measured_score(facts, "plucked_string_authority_score"),
            self._measured_score(facts, "reed_wind_authority_score"),
            self._measured_score(facts, "voice_score"),
            self._measured_score(facts, "struck_keys_authority_score"),
        )
        # Strong source authority can keep a broad-loop rescue. Moderate source
        # authority is also enough when the measured body is a clean non-drum
        # pitched phrase and the physics voter already put the parent in
        # Instruments. This is a voter-backed parent-family decision, not a
        # structure-only release.
        clean_non_drum_phrase = bool(
            self._shape_number(facts, "percussive_event_ratio") <= 0.24
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.26
            and max(
                self._measured_role_value(facts, "drum_loop"),
                self._measured_role_value(facts, "percussive_drum_loop"),
                self._measured_role_value(facts, "bright_drum_loop"),
            )
            < 0.62
        )
        physics_top = self._top_physics_guess_path(facts).lower()
        if strongest_source >= 0.68:
            return winning_claim
        if strongest_source >= 0.50 and clean_non_drum_phrase and physics_top.startswith("instruments/"):
            return winning_claim
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            source="final_uncertain_generic_loop_release_review",
            reason=(
                "final loop safety invariant reviewed a broad Instrument/Synth "
                "loop release because repeated-loop structure was strong but no "
                "source-family physics panel had enough authority"
            ),
            shared=winning_claim.shared_candidates,
            winner=winning_claim,
            strength=max(0.90, winning_claim.strength),
        )

    def _review_non_transition_pitched_loop_stolen_by_fx_leaf(
        self,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Repair FX leaves contradicted by non-transition pitched-loop structure.

        A concrete FX leaf is allowed to remain FX when motion/transition facts
        agree with it. When the measured body is a clean, non-percussive pitched
        music loop and the FX leaf is a contradiction, the safest destination is
        the broad Instrument Loops bucket.  Review is kept only when the facts do
        not clear the broad-loop safety bar.
        """
        if winning_claim.family != "FX":
            return winning_claim
        if (
            self._facts_have_concrete_fx_lane_agreement(facts)
            and not self._facts_have_hard_clean_keys_authority(facts, winning_claim)
            and not self._facts_support_nonpercussive_pitched_instrument_loop_body(facts, winning_claim)
        ):
            return winning_claim
        if not self._facts_support_fx_leaf_pitched_loop_conflict(facts, winning_claim):
            return winning_claim
        if self._facts_support_contradicted_fx_leaf_broad_instrument_release(facts, winning_claim):
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="final_fx_leaf_pitched_loop_broad_instrument_invariant",
                reason=(
                    "final structure invariant broadened a contradicted FX leaf "
                    "to Instrument Loops because measured body is a non-transition "
                    "pitched music loop"
                ),
                shared=winning_claim.shared_candidates,
                raw_candidate_score=winning_claim.raw_candidate_score,
                brain_rank=winning_claim.brain_rank,
                physics_rank=winning_claim.physics_rank,
                shared_winner=winning_claim.shared_winner or winning_claim.folder_path,
                can_override=True,
                strength=max(0.91, winning_claim.strength),
                is_real_candidate=winning_claim.is_real_candidate,
            )
        if (
            winning_claim.source in {"strong_consensus", "concrete_fx_gate_override", "synthetic_inconsistent_raw"}
            and winning_claim.raw_candidate_score <= 4.0
            and (winning_claim.brain_rank is None or winning_claim.brain_rank <= 4)
            and (winning_claim.physics_rank is None or winning_claim.physics_rank <= 4)
        ):
            return winning_claim
        if (
            winning_claim.source == "candidate_true_bucket_rescue"
            and winning_claim.raw_candidate_score <= 9.0
            and winning_claim.family == "FX"
        ):
            return winning_claim
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            source="final_fx_leaf_pitched_loop_conflict_review",
            reason=(
                "final structure invariant reviewed a concrete FX leaf whose "
                "measured body looked like a non-transition pitched music loop"
            ),
            shared=winning_claim.shared_candidates,
            winner=winning_claim,
            strength=max(0.90, winning_claim.strength),
        )

    @staticmethod
    def _measured_role_value(facts: SharedAudioFacts | None, role_name: str) -> float:
        if facts is None or not isinstance(facts.evidence, dict):
            return 0.0
        roles = facts.evidence.get("measured_roles", {})
        if not isinstance(roles, dict):
            return 0.0
        return role_strength(roles, role_name)

    @staticmethod
    def _shape_number(facts: SharedAudioFacts | None, metric_name: str) -> float:
        return max(
            _shape_metric_from_facts(facts, metric_name),
            _feature_number_from_facts(facts, metric_name),
        )

    @staticmethod
    def _physics_layer(facts: SharedAudioFacts | None) -> dict:
        """Return the shared PhysicsVoter layer dictionary, if present."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return {}
        layer = facts.evidence.get("physics_layer_decision")
        return layer if isinstance(layer, dict) else {}

    @classmethod
    def _facts_have_non_woodwind_branch_identity_conflict(cls, facts: SharedAudioFacts | None) -> bool:
        """Return True when layered physics selected a strong non-woodwind branch."""
        layer = cls._physics_layer(facts)
        if not layer:
            return False
        branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
        if not branch or branch in {"Woodwinds", "ReedWoodwind"}:
            return False
        if bool(
            layer.get("instrument_woodwind_source_signal")
            or layer.get("instrument_clean_tonal_reed_solo_signal")
            or layer.get("instrument_dark_low_mid_reed_loop_signal")
        ):
            return False
        confidence = cls._safe_float(
            layer.get("physics_layer_branch_confidence", layer.get("instrument_branch_selected_confidence")),
            0.0,
        )
        if confidence < 0.72 and not (branch == "MixedInstrument" and confidence >= 0.70):
            return False
        woodwind_branch = cls._safe_float(layer.get("instrument_branch_Woodwinds"), 0.0)
        subpanel = str(layer.get(f"instrument_{branch}_subpanel_selected") or "")
        subpanel_confidence = cls._safe_float(layer.get(f"instrument_{branch}_subpanel_confidence"), 0.0)
        subpanel_margin = cls._safe_float(layer.get(f"instrument_{branch}_subpanel_margin"), 0.0)
        compound_strength = cls._safe_float(layer.get("compound_music_strength"), 0.0)
        return bool(
            (subpanel and subpanel_confidence >= 0.76 and subpanel_margin >= 0.075)
            or (branch == "MixedInstrument" and compound_strength >= 0.54)
            or (
                branch in {"MalletBell", "Synth", "KeysPiano", "Strings", "Brass", "Bass"}
                and confidence >= max(0.80, woodwind_branch + 0.02)
            )
        )

    @classmethod
    def _candidate_path(cls, row: object) -> str:
        """Return a normalized candidate path from any voter row-like object."""
        if not isinstance(row, dict):
            return ""
        return str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")

    @classmethod
    def _is_concrete_fx_path(cls, path: str) -> bool:
        """True for specific FX identities that should carry override authority."""
        normalized = str(path or "").lower().replace("\\", "/")
        return bool(
            normalized.startswith("fx/")
            and any(fragment in normalized for fragment in cls.CONCRETE_FX_AUTHORITY_FRAGMENTS)
        )

    @classmethod
    def _concrete_fx_identity_key(cls, path: str) -> str:
        """Collapse concrete FX paths into stable identity groups."""
        normalized = str(path or "").lower().replace("\\", "/")
        for fragment in cls.CONCRETE_FX_AUTHORITY_FRAGMENTS:
            if fragment in normalized:
                return fragment
        return ""

    def _facts_have_concrete_fx_lane_agreement(
        self,
        facts: SharedAudioFacts | None,
        *,
        min_lanes: int = 3,
    ) -> bool:
        """Return True when multiple brain lanes agree on a concrete FX identity.

        This is an override-authority contract, not a category patch.  A final
        invariant may still broaden role/depth, but it cannot cross from a
        strongly confirmed concrete FX identity into keys/sax/voice merely
        because the material is tonal.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False

        ensemble = facts.evidence.get("brain_ensemble_vote_result")
        if isinstance(ensemble, dict):
            guesses = ensemble.get("top_guesses")
            if isinstance(guesses, list):
                for row in guesses[:3]:
                    path = self._candidate_path(row)
                    if not self._is_concrete_fx_path(path):
                        continue
                    lanes = row.get("lanes") if isinstance(row, dict) else None
                    lane_count = len(lanes) if isinstance(lanes, (list, tuple, set)) else 0
                    try:
                        confidence = float(row.get("confidence", 0.0) or 0.0)
                    except Exception:
                        confidence = 0.0
                    try:
                        score = float(row.get("score", row.get("ensemble_score", 9999.0)) or 9999.0)
                    except Exception:
                        score = 9999.0
                    if lane_count >= min_lanes and (confidence >= 0.45 or score <= 1.50):
                        return True

        identity_counts: dict[str, int] = {}
        for key in (
            "full_brain_vote_result",
            "core_baby_vote_result",
            "spread_baby_vote_result",
            "outlier_baby_vote_result",
        ):
            result = facts.evidence.get(key)
            if not isinstance(result, dict):
                continue
            guesses = result.get("top_guesses")
            if not isinstance(guesses, list) or not guesses:
                continue
            path = self._candidate_path(guesses[0])
            if not self._is_concrete_fx_path(path):
                continue
            identity = self._concrete_fx_identity_key(path)
            if not identity:
                continue
            identity_counts[identity] = identity_counts.get(identity, 0) + 1
        return any(count >= min_lanes for count in identity_counts.values())

    def _facts_have_hard_clean_keys_authority(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True only for keys evidence strong enough to beat concrete FX.

        Tonal, pitched, low-flatness material is not enough.  Sirens and formant
        FX can also be tonal and stable.  A keys override needs a stronger
        harmonic/struck-resonator contract before it can defeat multi-lane FX.
        """
        if facts is None:
            return False
        harmonic_energy = self._shape_number(facts, "harmonic_energy_ratio")
        harmonic_to_noise = self._shape_number(facts, "harmonic_to_noise_ratio")
        peak_stability = self._shape_number(facts, "spectral_peak_stability")
        flatness = self._shape_number(facts, "spectral_flatness_mean")
        piano_witness = self._piano_struck_physics_witness_score(facts)
        return bool(
            self._facts_support_clean_keys_loop(facts, winning_claim)
            and (
                harmonic_energy >= 0.32
                or harmonic_to_noise >= 0.34
                or (piano_witness >= 0.78 and peak_stability >= 0.18 and flatness <= 0.075)
            )
        )

    @classmethod
    def _branch_hint_from_path(cls, path: str) -> str:
        """Map an output path to the measured branch identity it claims."""
        normalized = str(path or "").lower().replace("\\", "/")
        for fragments, branch in cls.PATH_BRANCH_HINTS:
            if any(fragment in normalized for fragment in fragments):
                return branch
        return ""

    def _measured_instrument_loop_folder_from_physics_branch(
        self,
        facts: SharedAudioFacts | None,
        current_path: str,
        winning_claim: ConsensusClaim,
    ) -> str:
        """Choose a loop parent from the strongest measured instrument branch.

        Shape may force a one-shot leaf to broaden into a loop, but it should not
        preserve the wrong source identity when PhysicsVoter has a stronger
        measured branch.  This keeps sax fixes from turning clean keys loops into
        Brass/Woodwinds and makes future category work less fragile.
        """
        layer = self._physics_layer(facts)
        if not layer:
            return ""
        branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
        if branch not in self.BRANCH_LOOP_TARGETS:
            return ""
        # If measured structure strongly supports a drum loop, don't override to an
        # instrument branch. This prevents drum loops from being misrouted to Bass Loops
        # or other instrument folders.
        if self._facts_support_final_drum_loop(facts, winning_claim):
            return ""
        try:
            branch_confidence = float(
                layer.get("physics_layer_branch_confidence", layer.get("instrument_branch_selected_confidence", 0.0))
                or 0.0
            )
        except Exception:
            branch_confidence = 0.0
        branch_margin = self._instrument_branch_winner_margin(layer, branch)
        clean_electric_keys_loop_signal = bool(layer.get("instrument_clean_electric_keys_loop_signal"))
        plucked_source = bool(layer.get("instrument_plucked_string_source_signal"))
        subpanel = str(layer.get(f"instrument_{branch}_subpanel_selected") or "")
        sub_confidence = self._safe_float(layer.get(f"instrument_{branch}_subpanel_confidence"), 0.0)
        sub_margin = self._safe_float(layer.get(f"instrument_{branch}_subpanel_margin"), 0.0)
        has_close_sax_evidence_for_plucked_branch = bool(
            branch == "PluckedString"
            and (
                self._shared_candidate_has_top_family(
                    winning_claim,
                    ("sax", "saxophone"),
                    top_family="Instruments",
                    max_score=44.0,
                    max_brain_rank=8,
                    max_physics_rank=12,
                )
                or self._facts_have_near_sax_candidate_evidence(facts)
            )
        )
        measured_plucked_margin = bool(
            branch == "PluckedString"
            and plucked_source
            and not has_close_sax_evidence_for_plucked_branch
            and branch_margin >= 0.015
            and subpanel
            and sub_confidence >= 0.64
            and (sub_margin >= 0.045 or (sub_confidence >= 0.80 and sub_margin >= 0.020))
        )
        branch_floor = 0.74
        if measured_plucked_margin:
            branch_floor = 0.64
        elif branch == "KeysPiano" and clean_electric_keys_loop_signal:
            branch_floor = 0.68
        if branch_confidence < branch_floor:
            return ""
        if branch_margin < 0.075 and not (
            (branch == "KeysPiano" and clean_electric_keys_loop_signal and branch_margin >= 0.020)
            or measured_plucked_margin
        ):
            return ""
        if branch == "PluckedString" and branch_margin < 0.11 and not measured_plucked_margin:
            return ""
        claimed_branch = self._branch_hint_from_path(current_path)
        if claimed_branch and claimed_branch != branch:
            claimed_score = self._safe_float(layer.get(f"instrument_branch_{claimed_branch}"), 0.0)
            if branch_confidence < claimed_score + 0.10:
                return ""
        if branch == "MixedInstrument" and self._compound_music_loop_support(facts) < 0.55:
            return ""
        subpanel_authoritative = bool(subpanel and sub_confidence >= 0.76 and sub_margin >= 0.10)
        reed_source = bool(
            layer.get("instrument_reed_woodwind_source_signal") or layer.get("instrument_woodwind_source_signal")
        )
        plucked_subpanel_authoritative = bool(
            branch == "PluckedString"
            and plucked_source
            and not reed_source
            and subpanel
            and sub_confidence >= 0.66
            and (sub_margin >= 0.050 or sub_confidence >= 0.74)
        )
        close_internal_sax_candidate = bool(
            branch == "PluckedString"
            and has_close_sax_evidence_for_plucked_branch
            and (
                self._safe_float(layer.get("instrument_branch_Woodwinds"), 0.0) >= branch_confidence - 0.06
                or (
                    subpanel == "AcousticGuitar"
                    and str(layer.get("instrument_Woodwinds_subpanel_selected") or "")
                    in {"Sax", "AiryWoodwind", "Clarinet"}
                    and self._safe_float(layer.get("instrument_Woodwinds_subpanel_confidence"), 0.0) >= 0.76
                    and (
                        self._safe_float(layer.get("instrument_low_total"), 0.0) <= 0.15
                        or self._safe_float(layer.get("instrument_mid_ratio"), 0.0) >= 0.60
                    )
                )
            )
        )
        if close_internal_sax_candidate:
            return ""
        if branch == "PluckedString" and plucked_subpanel_authoritative:
            if subpanel == "AcousticGuitar":
                return "Instruments/Guitar/Acoustic Guitar/Loops"
            if subpanel == "ElectricGuitar":
                return "Instruments/Guitar/Electric Guitar/Loops"
            if subpanel == "NylonOrSoftPluck":
                return "Instruments/Guitar/Nylon Guitar/Loops"
            if subpanel == "WorldPluck":
                return "Instruments/Guitar/Guitar Plucks/Loops"
        dark_low_mid_sax_body = bool(
            branch in {"Woodwinds", "Brass"}
            and self._subpanel_score(facts, "woodwind_sax_score") >= 0.64
            and self._subpanel_score(facts, "woodwind_sax_score") >= self._subpanel_score(facts, "voice_score") + 0.06
            and self._subpanel_score(facts, "woodwind_sax_score")
            >= self._subpanel_score(facts, "brass_trumpet_score") + 0.08
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.86
            and self._shape_number(facts, "high_event_ratio") <= 0.035
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.040
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.10
            and not self._facts_support_clean_vintage_keys_loop(facts)
        )
        if branch in {"Woodwinds", "Brass"} and dark_low_mid_sax_body:
            return "Instruments/Woodwinds/Saxophone/Loops"
        if branch == "Woodwinds" and (
            bool(layer.get("instrument_dark_low_mid_reed_loop_signal"))
            or (subpanel_authoritative and subpanel == "Sax")
            or dark_low_mid_sax_body
        ):
            return "Instruments/Woodwinds/Saxophone/Loops"
        if branch == "Woodwinds" and subpanel_authoritative and subpanel == "Flute":
            return "Instruments/Woodwinds/Flute/Loops"
        if branch == "Woodwinds" and subpanel_authoritative and subpanel == "Clarinet":
            return "Instruments/Woodwinds/Clarinet/Loops"
        if branch == "Woodwinds" and subpanel_authoritative and subpanel == "Bassoon":
            return "Instruments/Woodwinds/Bassoon/Loops"
        if branch == "KeysPiano" and (
            self._facts_support_acoustic_piano_loop(facts, winning_claim)
            or (subpanel_authoritative and subpanel == "AcousticPiano")
        ):
            return "Instruments/Keys/Piano/Loops"
        if branch == "KeysPiano" and clean_electric_keys_loop_signal:
            return "Instruments/Keys/Electric Piano/Loops"
        if branch == "KeysPiano" and subpanel_authoritative and subpanel == "Organ":
            return "Instruments/Keys/Organ/Loops"
        if branch == "KeysPiano" and subpanel_authoritative and subpanel == "ClavinetHarpsichord":
            return "Instruments/Keys/Clavinet and Harpsichord/Loops"
        synth_pad_score = max(
            self._subpanel_score(facts, "synth_pad_score"),
            _feature_number_from_facts(facts, "synth_pad_score"),
        )
        synth_lead_score = max(
            self._subpanel_score(facts, "synth_lead_score"),
            _feature_number_from_facts(facts, "synth_lead_score"),
        )
        synth_arp_score = max(
            self._subpanel_score(facts, "synth_arp_score"),
            _feature_number_from_facts(facts, "synth_arp_score"),
            _feature_number_from_facts(facts, "arp_score"),
        )
        pad_shape = _shape_vote_from_facts(facts) in {
            "sustained_pad",
            "bass_phrase",
            "pitched_phrase",
            "pitched_phrase_shape",
            "vocal_phrase",
        }
        not_arp_motion = bool(
            self._shape_number(facts, "onset_count") < 8.0
            or self._shape_number(facts, "onset_true_repetition_likelihood") < 0.45
        )
        if branch == "Synth" and (
            (subpanel_authoritative and subpanel == "SynthPad")
            or (
                synth_pad_score >= 0.62
                and synth_pad_score >= max(synth_lead_score, synth_arp_score) - 0.04
                and pad_shape
                and not_arp_motion
            )
        ):
            return "Instruments/Synths/Pads/Loops"
        if branch == "Synth" and subpanel_authoritative and subpanel == "SynthLead":
            return "Instruments/Synths/Synth Lead/Loops"
        if branch == "Synth" and subpanel_authoritative and subpanel == "SynthArp":
            return "Instruments/Synths/Arps/Loops"
        if branch == "Bass" and subpanel_authoritative and subpanel == "808Sub":
            return "Instruments/Bass/808s/Loops"
        if branch == "Bass" and subpanel_authoritative and subpanel == "SynthBass":
            return "Instruments/Bass/Synth Bass/Loops"
        if branch == "Bass" and subpanel_authoritative and subpanel == "ElectricBass":
            return "Instruments/Bass/Electric Bass/Loops"
        if branch == "Bass" and subpanel_authoritative and subpanel == "UprightBass":
            return "Instruments/Bass/Upright Bass/Loops"
        if branch == "Strings" and subpanel_authoritative and subpanel == "BowedSustain":
            return "Instruments/Strings/String Sustains/Loops"
        if branch == "MalletBell" and subpanel_authoritative and subpanel in {"BellChime", "MalletKeys"}:
            return "Instruments/Mallets and Bells/Loops"
        if (
            branch == "Synth"
            and _shape_vote_from_facts(facts) in {"bass_phrase", "sustained_pad"}
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        ):
            return "Instruments/Synths/Pads/Loops"
        return self.BRANCH_LOOP_TARGETS[branch]

    @classmethod
    def _instrument_branch_winner_margin(cls, layer: dict, branch: str) -> float:
        """Return the selected branch's margin over the nearest sibling branch."""
        if not isinstance(layer, dict):
            return 0.0
        selected = cls._safe_float(
            layer.get(f"instrument_branch_{branch}", layer.get("instrument_branch_selected_confidence")),
            0.0,
        )
        sibling_scores = []
        for sibling in cls.BRANCH_LOOP_TARGETS:
            if sibling == branch:
                continue
            sibling_scores.append(cls._safe_float(layer.get(f"instrument_branch_{sibling}"), 0.0))
        nearest = max(sibling_scores) if sibling_scores else 0.0
        return selected - nearest

    @staticmethod
    def _shared_candidate_has(
        winning_claim: ConsensusClaim,
        fragments: tuple[str, ...],
        *,
        max_score: float = 32.0,
        max_brain_rank: int = 24,
        max_physics_rank: int = 24,
    ) -> bool:
        for row in winning_claim.shared_candidates or []:
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if not any(fragment in path for fragment in fragments):
                continue
            try:
                score = float(row.get("combined_rank_score", 9999.0) or 9999.0)
            except Exception:
                score = 9999.0
            try:
                brain_rank = int(row.get("brain_rank", 9999) or 9999)
            except Exception:
                brain_rank = 9999
            try:
                physics_rank = int(row.get("physics_rank", 9999) or 9999)
            except Exception:
                physics_rank = 9999
            if score <= max_score or brain_rank <= max_brain_rank or physics_rank <= max_physics_rank:
                return True
        return False

    @staticmethod
    def _shared_candidate_count_top_family(
        winning_claim: ConsensusClaim,
        fragments: tuple[str, ...],
        *,
        top_family: str,
        max_score: float = 32.0,
        max_brain_rank: int = 24,
        max_physics_rank: int = 24,
    ) -> int:
        """Count candidate rows from a top family that match fragments."""
        wanted = str(top_family or "").lower()
        count = 0
        for row in winning_claim.shared_candidates or []:
            row_top = str(row.get("top_family") or "").lower()
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if row_top and row_top != wanted:
                continue
            if not row_top and wanted and not path.startswith(f"{wanted}/"):
                continue
            if not any(fragment in path for fragment in fragments):
                continue
            try:
                score = float(row.get("combined_rank_score", 9999.0) or 9999.0)
            except Exception:
                score = 9999.0
            try:
                brain_rank = int(row.get("brain_rank", 9999) or 9999)
            except Exception:
                brain_rank = 9999
            try:
                physics_rank = int(row.get("physics_rank", 9999) or 9999)
            except Exception:
                physics_rank = 9999
            if score <= max_score or brain_rank <= max_brain_rank or physics_rank <= max_physics_rank:
                count += 1
        return count

    @staticmethod
    def _shared_candidate_best_score_top_family(
        winning_claim: ConsensusClaim,
        fragments: tuple[str, ...],
        *,
        top_family: str,
    ) -> float:
        """Return the best combined rank score for a matching top-family row."""
        wanted = str(top_family or "").lower()
        best = 9999.0
        for row in winning_claim.shared_candidates or []:
            row_top = str(row.get("top_family") or "").lower()
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if row_top and row_top != wanted:
                continue
            if not row_top and wanted and not path.startswith(f"{wanted}/"):
                continue
            if not any(fragment in path for fragment in fragments):
                continue
            try:
                best = min(best, float(row.get("combined_rank_score", 9999.0) or 9999.0))
            except Exception:
                continue
        return best

    @staticmethod
    def _shared_candidate_has_top_family(
        winning_claim: ConsensusClaim,
        fragments: tuple[str, ...],
        *,
        top_family: str,
        max_score: float = 32.0,
        max_brain_rank: int = 24,
        max_physics_rank: int = 24,
    ) -> bool:
        """Return True only for candidate rows from the requested top family."""
        wanted = str(top_family or "").lower()
        for row in winning_claim.shared_candidates or []:
            row_top = str(row.get("top_family") or "").lower()
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if row_top and row_top != wanted:
                continue
            if not row_top and wanted and not path.startswith(f"{wanted}/"):
                continue
            if not any(fragment in path for fragment in fragments):
                continue
            try:
                score = float(row.get("combined_rank_score", 9999.0) or 9999.0)
            except Exception:
                score = 9999.0
            try:
                brain_rank = int(row.get("brain_rank", 9999) or 9999)
            except Exception:
                brain_rank = 9999
            try:
                physics_rank = int(row.get("physics_rank", 9999) or 9999)
            except Exception:
                physics_rank = 9999
            if score <= max_score or brain_rank <= max_brain_rank or physics_rank <= max_physics_rank:
                return True
        return False

    @staticmethod
    def _facts_have_internal_candidate(
        facts: SharedAudioFacts | None,
        fragments: tuple[str, ...],
        *,
        top_family: str,
        max_rank: int,
        max_score: float,
        include_parent_audit: bool = True,
    ) -> bool:
        """Return True for matching internal voter/audit candidates.

        This intentionally reads only serialized voter output already stored in
        shared facts.  It does not inspect the producer filename, ZIP member, or
        source folder.  It lets final structure invariants see specialist
        evidence that fell outside the shared brain/physics intersection.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        wanted = str(top_family or "").lower()

        def row_matches(row: object, fallback_rank: int) -> bool:
            if not isinstance(row, dict):
                return False
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if wanted and not path.startswith(f"{wanted}/"):
                row_top = str(row.get("top_family") or "").lower()
                if row_top != wanted:
                    return False
            if not any(fragment in path for fragment in fragments):
                return False
            try:
                rank = int(row.get("rank", fallback_rank) or fallback_rank)
            except Exception:
                rank = fallback_rank
            try:
                score = float(row.get("score", row.get("combined_rank_score", rank + 1.0)) or rank + 1.0)
            except Exception:
                score = float(rank + 1.0)
            return rank <= max_rank and score <= max_score

        for key in (
            "brain_ensemble_vote_result",
            "full_brain_vote_result",
            "core_baby_vote_result",
            "spread_baby_vote_result",
            "outlier_baby_vote_result",
        ):
            result = facts.evidence.get(key)
            if not isinstance(result, dict):
                continue
            guesses = result.get("top_guesses")
            if not isinstance(guesses, list):
                continue
            for index, guess in enumerate(guesses[:24], start=1):
                if row_matches(guess, index):
                    return True

        parent_audit = facts.evidence.get("parent_role_audit") if include_parent_audit else None
        if isinstance(parent_audit, dict):
            for key in ("brain_top_20", "physics_top_20"):
                guesses = parent_audit.get(key)
                if not isinstance(guesses, list):
                    continue
                for index, guess in enumerate(guesses[:24], start=1):
                    if row_matches(guess, index):
                        return True
        return False

    def _piano_struck_physics_witness_score(self, facts: SharedAudioFacts | None) -> float:
        """Return measured piano/struck-resonator support from PhysicsVoter rows."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return 0.0
        score = 0.0

        def apply_rows(rows: object, limit: int = 8) -> None:
            nonlocal score
            if not isinstance(rows, list):
                return
            for row in rows[:limit]:
                if not isinstance(row, dict):
                    continue
                path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
                if not path.startswith("instruments/") or not any(
                    token in path for token in ("/keys/", "piano", "rhodes", "electric piano")
                ):
                    continue
                try:
                    rank = int(row.get("rank", 999) or 999)
                except Exception:
                    rank = 999
                try:
                    row_score = float(row.get("score", 999.0) or 999.0)
                except Exception:
                    row_score = 999.0
                evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
                try:
                    witness = float(evidence.get("piano_struck_identity_score", 0.0) or 0.0)
                except Exception:
                    witness = 0.0
                eligible = bool(evidence.get("piano_struck_identity_eligible"))
                if eligible:
                    score = max(score, witness, 0.72)
                elif rank <= 5 and row_score <= 0.85 and witness >= 0.52:
                    score = max(score, witness)

        physics_result = facts.evidence.get("physics_vote_result")
        if isinstance(physics_result, dict):
            apply_rows(physics_result.get("top_guesses"), limit=12)
        dry_core_physics = facts.evidence.get("dry_core_physics_vote_result")
        if isinstance(dry_core_physics, dict):
            apply_rows(dry_core_physics.get("top_guesses"), limit=12)
        parent_audit = facts.evidence.get("parent_role_audit")
        if isinstance(parent_audit, dict):
            apply_rows(parent_audit.get("physics_top_20"), limit=8)
        return min(1.0, score)

    def _reed_sax_physics_witness_score(self, facts: SharedAudioFacts | None) -> float:
        """Return measured sax/reed support from source-name-blind physics only.

        This is not a router by itself.  It is a small early-physics witness
        built from existing named features: pitched stability, harmonic energy,
        wet tail/body flatness, and low/mid/high distribution.  It is meant to
        distinguish wet reed phrases from clean synth/keys loops before final
        broadening or synth invariants steal them.
        """
        if facts is None:
            return 0.0
        shape = _shape_vote_from_facts(facts)
        if shape not in {
            "bass_phrase",
            "pitched_phrase",
            "sustained_pad",
            "vocal_phrase",
            "hit_with_tail",
            "solo_phrase",
        }:
            return 0.0
        if _shape_confidence_from_facts(facts) < 0.70:
            return 0.0
        if self._shape_number(facts, "percussive_event_ratio") > 0.18:
            return 0.0
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.20:
            return 0.0

        low_ratio = self._shape_number(facts, "low_event_ratio")
        mid_ratio = self._shape_number(facts, "mid_event_ratio")
        high_ratio = self._shape_number(facts, "high_event_ratio")
        flatness = self._shape_number(facts, "spectral_flatness_mean")
        entropy = self._shape_number(facts, "spectral_entropy_mean")
        f0_voiced = self._shape_number(facts, "f0_voiced_ratio")
        pitched_event = self._shape_number(facts, "pitched_event_ratio")
        pitch_confidence = self._shape_number(facts, "pitch_confidence")
        sustained_tonal = self._shape_number(facts, "sustained_tonal_frame_ratio")
        harmonic_energy = self._shape_number(facts, "harmonic_energy_ratio")
        inharmonicity = self._shape_number(facts, "inharmonicity")
        tail_energy = self._shape_number(facts, "tail_energy_ratio")
        body_flatness = self._shape_number(facts, "body_flatness")
        body_entropy = self._shape_number(facts, "body_entropy")
        onset_count = self._shape_number(facts, "onset_count")
        f0_hz = self._shape_number(facts, "f0_median_hz")
        f0_stability = self._shape_number(facts, "f0_stability_cents")

        clean_synth_or_keys = bool(
            high_ratio <= 0.012 and flatness <= 0.060 and body_flatness <= 0.070 and entropy <= 0.38
        )
        if clean_synth_or_keys:
            return 0.0

        score = 0.0
        if f0_voiced >= 0.86 and pitched_event >= 0.90 and pitch_confidence >= 0.45:
            score += 0.25
        if sustained_tonal >= 0.82:
            score += 0.10
        if harmonic_energy >= 0.28 and 0.04 <= inharmonicity <= 0.34:
            score += 0.18
        if 0.020 <= high_ratio <= 0.160 and 0.060 <= flatness <= 0.360:
            score += 0.18
        if 0.070 <= mid_ratio <= 0.620 and low_ratio <= 0.93:
            score += 0.08
        if tail_energy >= 0.25 or body_flatness >= 0.09:
            score += 0.08
        if onset_count >= 7.0:
            score += 0.06
        if 150.0 <= f0_hz <= 900.0 and f0_stability >= 80.0:
            score += 0.07
        if body_entropy <= 0.45 and entropy <= 0.52:
            score += 0.05

        # v31.104: let the arbiter read the improved PhysicsVoter result as a
        # physics witness.  This does not use filenames.  It only trusts a real
        # sax/woodwind candidate when the PhysicsVoter itself ranked it very
        # high after first-arrival/cepstral scoring.
        def apply_sax_rows(rows: object, limit: int = 5) -> None:
            nonlocal score
            if not isinstance(rows, list):
                return
            for row in rows[:limit]:
                if not isinstance(row, dict):
                    continue
                path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
                if not path.startswith("instruments/") or not ("sax" in path or "saxophone" in path):
                    continue
                try:
                    rank = int(row.get("rank", 999) or 999)
                except Exception:
                    rank = 999
                try:
                    row_score = float(row.get("score", 999.0) or 999.0)
                except Exception:
                    row_score = 999.0
                evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
                first_arrival_ok = bool(evidence.get("reed_sax_first_arrival_eligible"))
                if rank <= 3 and row_score <= 0.55:
                    score = max(score, 0.90 if first_arrival_ok or row_score <= 0.35 else 0.78)
                elif rank <= 5 and row_score <= 0.85:
                    score = max(score, 0.72)

        physics_result = facts.evidence.get("physics_vote_result") if isinstance(facts.evidence, dict) else None
        if isinstance(physics_result, dict):
            apply_sax_rows(physics_result.get("top_guesses"), limit=8)
        dry_core_physics = (
            facts.evidence.get("dry_core_physics_vote_result") if isinstance(facts.evidence, dict) else None
        )
        if isinstance(dry_core_physics, dict):
            apply_sax_rows(dry_core_physics.get("top_guesses"), limit=8)
        parent_audit = facts.evidence.get("parent_role_audit") if isinstance(facts.evidence, dict) else None
        if isinstance(parent_audit, dict):
            apply_sax_rows(parent_audit.get("physics_top_20"), limit=5)
        return min(1.0, score)

    def _facts_have_near_sax_candidate_evidence(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for near-miss sax evidence from internal voter/audit rows."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        if self._facts_have_strong_sax_candidate_evidence(facts):
            return True
        return self._facts_have_internal_candidate(
            facts,
            ("sax", "saxophone"),
            top_family="Instruments",
            max_rank=8,
            max_score=1.85,
            include_parent_audit=False,
        )

    @staticmethod
    def _facts_have_strong_sax_candidate_evidence(facts: SharedAudioFacts | None) -> bool:
        """Return True only for direct sax evidence from actual brain lanes.

        This deliberately ignores parent-role audit and dry/wet probe summaries.
        Those can say "woodwind-like" for clean synth or voice loops and must
        not create a sax identity by themselves.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False

        def sax_row(row: object, fallback_rank: int) -> tuple[bool, int, float]:
            if not isinstance(row, dict):
                return False, 999, 9999.0
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if not path.startswith("instruments/") or not ("sax" in path or "saxophone" in path):
                return False, 999, 9999.0
            try:
                rank = int(row.get("rank", fallback_rank) or fallback_rank)
            except Exception:
                rank = fallback_rank
            try:
                score = float(row.get("score", row.get("combined_rank_score", rank + 1.0)) or rank + 1.0)
            except Exception:
                score = float(rank + 1.0)
            return True, rank, score

        ensemble = facts.evidence.get("brain_ensemble_vote_result")
        if isinstance(ensemble, dict):
            for index, guess in enumerate((ensemble.get("top_guesses") or [])[:8], start=1):
                ok, rank, score = sax_row(guess, index)
                if ok and rank <= 5 and score <= 1.75:
                    return True

        lane_hits = 0
        for key in (
            "full_brain_vote_result",
            "core_baby_vote_result",
            "spread_baby_vote_result",
            "outlier_baby_vote_result",
        ):
            result = facts.evidence.get(key)
            if not isinstance(result, dict):
                continue
            for index, guess in enumerate((result.get("top_guesses") or [])[:10], start=1):
                ok, rank, score = sax_row(guess, index)
                if ok and rank <= 8 and score <= 12.0:
                    lane_hits += 1
                    break
        return lane_hits >= 2

    @staticmethod
    def _facts_have_decisive_physics_sax_candidate(facts: SharedAudioFacts | None) -> bool:
        """Return True when PhysicsVoter itself exposes a strong sax candidate."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        physics_result = facts.evidence.get("physics_vote_result")
        if not isinstance(physics_result, dict):
            return False
        rows = physics_result.get("top_guesses")
        if not isinstance(rows, list):
            return False
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if not path.startswith("instruments/") or not ("sax" in path or "saxophone" in path):
                continue
            try:
                rank = int(row.get("rank", 999) or 999)
                score = float(row.get("score", 999.0) or 999.0)
            except Exception:
                continue
            evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
            try:
                reed_score = float(evidence.get("reed_sax_identity_score", 0.0) or 0.0)
            except Exception:
                reed_score = 0.0
            layer_branch = str(evidence.get("physics_layer_branch", ""))
            if rank <= 3 and score <= 0.55 and (reed_score >= 0.58 or layer_branch == "Woodwinds"):
                return True
        return False

    @staticmethod
    def _facts_have_decisive_bass_branch_authority(facts: SharedAudioFacts | None) -> bool:
        """Return True only for a stable, non-crowded measured Bass branch.

        Bass phrase shape alone is not identity.  Mixed brass/sax loops can have
        heavy low-band energy and still be musical instrument loops, not bass.
        This guard keeps the final bass invariant from stealing those cases.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        layer = facts.evidence.get("physics_layer_decision")
        if not isinstance(layer, dict):
            return False
        if str(layer.get("instrument_branch_selected") or "") != "Bass":
            return False

        def number(name: str, default: float = 0.0) -> float:
            try:
                return float(layer.get(name, default) or default)
            except Exception:
                return default

        branch_confidence = number("instrument_branch_selected_confidence")
        bass_branch = number("instrument_branch_Bass")
        mixed_branch = number("instrument_branch_MixedInstrument")
        synth_branch = number("instrument_branch_Synth")
        subpanel_margin = number("instrument_Bass_subpanel_margin")
        bass_subpanel = str(layer.get("instrument_Bass_subpanel_selected") or "")
        bass_subpanel_confidence = number("instrument_Bass_subpanel_confidence")
        plausible_count = int(number("instrument_branch_plausible_count"))
        clean_bass_phrase = bool(layer.get("instrument_clean_bass_phrase"))
        low_pitch_identity = bool(layer.get("instrument_low_pitch_bass_identity"))
        high_register_conflict = bool(layer.get("instrument_high_register_low_band_conflict"))

        if high_register_conflict and not low_pitch_identity:
            return False
        if clean_bass_phrase or low_pitch_identity:
            # Synth-bass and bass-loop subpanels are both valid Bass identities
            # for a clean measured bass phrase.  Do not force a broad Instrument
            # Loops fallback merely because the Bass subpanel says SynthBass
            # instead of BassLoop when the Bass branch itself is decisive.
            if bass_subpanel in {"BassLoop", "SynthBass"} and bass_subpanel_confidence >= 0.84:
                required_margin = 0.010
            else:
                required_margin = 0.025
            return branch_confidence >= 0.62 and subpanel_margin >= required_margin
        if branch_confidence < 0.72 or subpanel_margin < 0.055:
            return False
        strongest_non_bass = max(mixed_branch, synth_branch)
        return not (plausible_count >= 5 and bass_branch - strongest_non_bass < 0.12)

    @staticmethod
    def _facts_have_decisive_physics_bass_candidate(facts: SharedAudioFacts | None) -> bool:
        """Return True when PhysicsVoter's layered branch picks Bass decisively."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        if not FamilyClaimArbiter._facts_have_decisive_bass_branch_authority(facts):
            return False
        physics_result = facts.evidence.get("physics_vote_result")
        if not isinstance(physics_result, dict):
            return False
        rows = physics_result.get("top_guesses")
        if not isinstance(rows, list):
            return False
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if not path.startswith("instruments/") or "/bass/" not in path:
                continue
            try:
                rank = int(row.get("rank", 999) or 999)
                score = float(row.get("score", 999.0) or 999.0)
            except Exception:
                continue
            evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
            layer_branch = str(evidence.get("physics_layer_branch", ""))
            if rank <= 8 and score <= 0.35 and layer_branch == "Bass":
                return True
        return False

    @staticmethod
    def _shared_candidate_best_score(winning_claim: ConsensusClaim, fragments: tuple[str, ...]) -> float:
        best = 9999.0
        for row in winning_claim.shared_candidates or []:
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if not any(fragment in path for fragment in fragments):
                continue
            try:
                best = min(best, float(row.get("combined_rank_score", 9999.0) or 9999.0))
            except Exception:
                continue
        return best

    def _blocked_drum_loop_claim_is_safe(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when a blocked Drum Loops claim is a structure fix.

        This releases only broad drum-loop structure. It does not choose kick,
        snare, hat, conga, or any other identity leaf.
        """
        if claim.family != "Drums" or claim.sub_family != "Drum Loops":
            return False
        return bool(
            self._facts_support_final_drum_loop(facts, claim) or self._facts_support_drum_loop_claim(claim, facts)
        )

    def _blocked_broad_instrument_loop_claim_is_safe(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for safe broad Instrument Loops release.

        This is the review-relief valve for clean pitched loops. It intentionally
        returns only a broad loop bucket, never sax, guitar, keys, voice, or any
        concrete identity.  It is source-name blind and requires measured loop
        structure plus nearby instrument-family candidate evidence.
        """
        if facts is None:
            return False
        if claim.family != "Instruments" or claim.sub_family != "Instrument Loops":
            return False
        if claim.source not in self.ROLE_SHAPE_ROUTING_SOURCES and claim.source not in {
            "final_fx_leaf_pitched_loop_conflict_review",
            "final_fx_leaf_pitched_loop_broad_instrument_invariant",
        }:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape in {"transition_riser", "transition_downlifter", "transition_drop"}:
            return False
        movement = str(getattr(facts, "movement", "") or "").lower()
        slope = abs(_feature_number_from_facts(facts, "centroid_slope"))
        if movement in {"rising", "falling"} and slope >= 0.10:
            return False
        pitched_loop = max(
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "pitched_music_phrase"),
            self._measured_role_value(facts, "pitched_reed_or_instrument_loop"),
            self._measured_role_value(facts, "pitched_reed_or_instrument_phrase"),
            self._measured_role_value(facts, "mixed_music_loop"),
            self._measured_role_value(facts, "clean_sustained_tonal_instrument_loop"),
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
        )
        percussive_drum_loop = max(
            self._measured_role_value(facts, "drum_loop"),
            self._measured_role_value(facts, "percussive_drum_loop"),
            self._measured_role_value(facts, "bright_drum_loop"),
        )
        low_rhythmic_drum_loop = self._measured_role_value(facts, "low_rhythmic_drum_loop")
        drum_loop = percussive_drum_loop
        if (
            self._shape_number(facts, "drumlike_frame_ratio") >= 0.14
            or self._shape_number(facts, "percussive_event_ratio") >= 0.14
        ):
            drum_loop = max(drum_loop, low_rhythmic_drum_loop)
        instrument_support = self._shared_candidate_count_top_family(
            claim,
            (
                "instrument loops",
                "guitar",
                "synth",
                "keys",
                "piano",
                "strings",
                "cello",
                "woodwind",
                "sax",
                "brass",
                "voice",
                "vocal",
            ),
            top_family="Instruments",
            max_score=44.0,
            max_brain_rank=16,
            max_physics_rank=16,
        )
        raw_instrument_support = self._shared_candidate_count_top_family(
            raw_claim,
            (
                "instrument loops",
                "guitar",
                "synth",
                "keys",
                "piano",
                "strings",
                "cello",
                "woodwind",
                "sax",
                "brass",
                "voice",
                "vocal",
            ),
            top_family="Instruments",
            max_score=44.0,
            max_brain_rank=16,
            max_physics_rank=16,
        )
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        parent_broad_path = str(parent.get("broad_folder_path") or "").lower() if isinstance(parent, dict) else ""
        parent_role = str(parent.get("role_name") or "").lower() if isinstance(parent, dict) else ""
        parent_confidence = self._safe_float(parent.get("confidence"), 0.0) if isinstance(parent, dict) else 0.0
        parent_mixed_music_loop = bool(
            parent_role == "mixed_music_loop"
            and parent_broad_path.startswith("instruments/")
            and parent_confidence >= 0.58
            and shape
            in {
                "repeated_phrase_loop",
                "mixed_instrument_loop",
                "compound_musical_loop",
                "pitched_phrase",
                "vocal_phrase",
                "bass_phrase",
                "sustained_pad",
            }
            and shape_confidence >= 0.66
            and drum_loop < 0.70
        )
        if parent_mixed_music_loop:
            return True
        return bool(
            shape in {
                "pitched_phrase",
                "sustained_pad",
                "vocal_phrase",
                "bass_phrase",
                "repeated_phrase_loop",
                "mixed_instrument_loop",
                "compound_musical_loop",
            }
            and shape_confidence >= 0.68
            and pitched_loop >= 0.58
            and self._shape_number(facts, "pitched_event_ratio") >= 0.50
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.50
            and self._shape_number(facts, "percussive_event_ratio") <= 0.24
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.26
            and drum_loop < 0.70
            and (instrument_support >= 1 or raw_instrument_support >= 1)
        )

    def _facts_support_contradicted_fx_leaf_broad_instrument_release(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True when a contradicted FX leaf should broaden to Instrument Loops.

        This is used only after _facts_support_fx_leaf_pitched_loop_conflict() has
        already established that the FX leaf is not a measured transition.  It
        avoids the older failure mode where obvious pitched music loops stayed in
        review because the false FX leaf had a strong score.
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {"pitched_phrase", "sustained_pad", "vocal_phrase", "bass_phrase"}:
            return False
        if shape_confidence < 0.70:
            return False
        pitched_loop = max(
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "pitched_music_phrase"),
            self._measured_role_value(facts, "vocal_music_phrase"),
            self._shape_number(facts, "pitched_event_ratio"),
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
        )
        if pitched_loop < 0.78:
            return False
        if self._shape_number(facts, "percussive_event_ratio") > 0.16:
            return False
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.16:
            return False
        if self._shape_number(facts, "pitched_event_ratio") < 0.58:
            return False
        if self._shape_number(facts, "sustained_tonal_frame_ratio") < 0.58:
            return False
        instrument_support = self._shared_candidate_count_top_family(
            winning_claim,
            (
                "instrument loops",
                "guitar",
                "synth",
                "keys",
                "piano",
                "strings",
                "cello",
                "woodwind",
                "sax",
                "brass",
                "voice",
                "vocal",
                "bass",
            ),
            top_family="Instruments",
            max_score=48.0,
            max_brain_rank=20,
            max_physics_rank=20,
        )
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        allowed = parent.get("allowed_top_families", []) if isinstance(parent, dict) else []
        pure_measured_loop = bool(
            pitched_loop >= 0.92
            and shape_confidence >= 0.70
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.90
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )
        return bool(instrument_support >= 1 or "Instruments" in allowed or pure_measured_loop)

    def _facts_support_broad_instrument_phrase_release(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True for broad non-percussive pitched phrase release.

        This is a final structure invariant, not an identity rescue. It exists
        for sax/brass/vocal-like phrases that narrowly win a hand-percussion
        leaf even though measured facts say tonal, sustained, non-drum body.
        The destination is only broad Instrument Loops.
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {
            "pitched_phrase",
            "vocal_phrase",
            "sustained_pad",
            "bass_phrase",
            "hit_with_tail",
            "solo_phrase",
        }:
            return False
        if shape_confidence < 0.70:
            return False
        pitched_body = max(
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "pitched_music_phrase"),
            self._measured_role_value(facts, "vocal_music_phrase"),
            self._shape_number(facts, "pitched_event_ratio"),
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
        )
        if pitched_body < 0.82:
            return False
        if self._shape_number(facts, "percussive_event_ratio") > 0.10:
            return False
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.10:
            return False
        if self._shape_number(facts, "sustained_tonal_frame_ratio") < 0.72:
            return False
        if self._shape_number(facts, "pitched_event_ratio") < 0.72:
            return False
        instrument_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            (
                "instrument loops",
                "guitar",
                "synth",
                "keys",
                "piano",
                "strings",
                "cello",
                "woodwind",
                "sax",
                "brass",
                "voice",
                "vocal",
            ),
            top_family="Instruments",
            max_score=44.0,
            max_brain_rank=16,
            max_physics_rank=16,
        )
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        allowed = parent.get("allowed_top_families", []) if isinstance(parent, dict) else []
        parent_allows_instrument = "Instruments" in allowed
        pure_measured_phrase = bool(
            pitched_body >= 0.92
            and shape_confidence >= 0.70
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.90
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
        )
        return bool(instrument_candidate or parent_allows_instrument or pure_measured_phrase)

    def _facts_support_fx_leaf_pitched_loop_conflict(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True when a concrete FX leaf conflicts with music-loop facts."""
        if facts is None:
            return False
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if not any(
            fragment in path
            for fragment in (
                "riser",
                "build",
                "drop",
                "downlifter",
                "glitch",
                "stutter",
                "hybrid designed",
                "designed noise",
                "foley",
                "machine",
                "engine",
                "motor",
                "coin",
                "object",
            )
        ):
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape in {"transition_riser", "transition_downlifter", "transition_drop"}:
            return False
        movement = str(getattr(facts, "movement", "") or "").lower()
        slope = abs(_feature_number_from_facts(facts, "centroid_slope"))
        if movement in {"rising", "falling"} and slope >= 0.10:
            return False
        pitched_loop = max(
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "pitched_music_phrase"),
            self._measured_role_value(facts, "vocal_music_phrase"),
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
        )
        drum_loop = max(
            self._measured_role_value(facts, "drum_loop"),
            self._measured_role_value(facts, "low_rhythmic_drum_loop"),
            self._measured_role_value(facts, "percussive_drum_loop"),
            self._measured_role_value(facts, "bright_drum_loop"),
        )
        return bool(
            shape in {"pitched_phrase", "sustained_pad", "vocal_phrase", "bass_phrase"}
            and shape_confidence >= 0.68
            and pitched_loop >= 0.66
            and drum_loop < 0.70
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.22
        )

    def _facts_support_final_drum_loop(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True when measured role and real candidates support Drum Loops."""
        if facts is None:
            return False
        roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(roles, dict):
            roles = {}
        drum_loop_strength = max(
            role_strength(roles, "drum_loop"),
            role_strength(roles, "low_rhythmic_drum_loop"),
            role_strength(roles, "percussive_drum_loop"),
            role_strength(roles, "bright_drum_loop"),
        )
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        pulse = self._shape_number(facts, "pulse_regularity")
        onset_count = self._shape_number(facts, "onset_count")
        non_voice = (
            max(
                self._measured_role_value(facts, "vocal_music_phrase"),
                self._measured_role_value(facts, "vocal_phrase"),
                self._measured_role_value(facts, "voiced_one_shot"),
            )
            <= 0.35
        )
        candidate_supported = self._shared_candidate_has_top_family(
            winning_claim,
            ("drum loops", "drum loop"),
            top_family="Drums",
            max_score=48.0,
            max_brain_rank=14,
            max_physics_rank=12,
        ) or self._facts_have_internal_candidate(
            facts,
            ("drum loops", "drum loop"),
            top_family="Drums",
            max_rank=14,
            max_score=4.75,
        )
        drum_family_supported = self._shared_candidate_has_top_family(
            winning_claim,
            ("drum", "kick", "snare", "tom", "percussion", "clap", "hat", "cymbal"),
            top_family="Drums",
            max_score=44.0,
            max_brain_rank=12,
            max_physics_rank=12,
        )
        measured_loop = bool(
            drum_loop_strength >= 0.76
            and shape in {"beat_loop", "drum_loop", "top_loop", "bass_phrase"}
            and shape_confidence >= 0.72
            and (
                pulse >= 0.24
                or onset_count >= 3.0
                or drum_loop_strength >= 0.88
                or role_strength(roles, "drum_loop") >= 0.76
            )
        )
        low_confidence_drum_loop = bool(
            drum_loop_strength >= 0.28
            and shape in {"beat_loop", "drum_loop", "top_loop", "bass_phrase"}
            and shape_confidence >= 0.72
            and onset_count >= 8.0
            and (pulse >= 0.16 or self._shape_number(facts, "percussive_event_ratio") >= 0.08 or candidate_supported)
            and self._shape_number(facts, "f0_voiced_ratio") <= 0.70
        )
        measured_drum_role_is_decisive = bool(
            role_strength(roles, "drum_loop") >= 0.82 and shape == "drum_loop" and shape_confidence >= 0.72
        )
        transition_shaped_drum_loop = self._facts_support_rhythmic_drum_loop_over_transition(facts)
        strong_pitched_nonpercussive_loop = bool(
            shape in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad", "vocal_phrase", "bass_phrase"}
            and shape_confidence >= 0.70
            and self._shape_number(facts, "pitched_event_ratio") >= 0.50
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.50
            and self._shape_number(facts, "percussive_event_ratio") <= 0.24
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.26
            and role_strength(roles, "drum_loop") < 0.76
        )
        if strong_pitched_nonpercussive_loop:
            return False
        return bool(
            (
                measured_loop
                and candidate_supported
                or low_confidence_drum_loop
                and (candidate_supported or drum_family_supported)
                or measured_drum_role_is_decisive
                or transition_shaped_drum_loop
            )
            and non_voice
        )

    def _facts_support_pitched_hit_drum_leaf_conflict(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True for percussive pitched phrases that should not be a drum leaf."""
        if facts is None or winning_claim.family != "Drums":
            return False
        path = self._norm_claim_path(winning_claim.folder_path or winning_claim.label)
        if "drum loops" in path or "kick" in path or "snare" in path or "hat" in path:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        pitched = max(
            self._measured_role_value(facts, "pitched_music_phrase"),
            self._measured_role_value(facts, "pitched_music_loop"),
            self._measured_role_value(facts, "vocal_music_phrase"),
            self._measured_role_value(facts, "voiced_one_shot"),
            self._shape_number(facts, "pitched_event_ratio"),
            self._shape_number(facts, "f0_voiced_ratio"),
        )
        drum_loop_strength = max(
            self._measured_role_value(facts, "drum_loop"),
            self._measured_role_value(facts, "low_rhythmic_drum_loop"),
            self._measured_role_value(facts, "percussive_drum_loop"),
            self._measured_role_value(facts, "bright_drum_loop"),
        )
        instrument_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            (
                "guitar",
                "synth",
                "voice",
                "vocal",
                "choir",
                "strings",
                "cello",
                "woodwind",
                "sax",
                "brass",
                "keys",
                "piano",
            ),
            top_family="Instruments",
            max_score=42.0,
            max_brain_rank=12,
            max_physics_rank=12,
        )
        high_voiced_nonpercussive_hit = bool(
            shape in {"hit_with_tail", "vocal_phrase", "pitched_phrase", "sustained_pad", "solo_phrase"}
            and shape_confidence >= 0.68
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.90
            and self._measured_role_value(facts, "voiced_one_shot") >= 0.60
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.10
        )
        return bool(
            shape in {"vocal_phrase", "pitched_phrase", "sustained_pad", "hit_with_tail", "solo_phrase"}
            and shape_confidence >= 0.68
            and pitched >= 0.42
            and drum_loop_strength < 0.70
            and (instrument_candidate or high_voiced_nonpercussive_hit)
        )

    def _facts_support_false_voice_loop_broad_instrument_release(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True when a voice-shaped result is safer as a broad music loop.

        The shape voter can call sax/reed/synth loops ``vocal_phrase`` because
        they are voiced and formant-like.  Without direct voice-candidate support,
        a strong, clean, repeated pitched loop should not be forced into Voice.
        """
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {"vocal_phrase", "pitched_phrase", "bass_phrase", "sustained_pad"}:
            return False
        if shape_confidence < 0.82:
            return False
        if self._measured_role_value(facts, "pitched_music_loop") < 0.78:
            return False
        if self._shape_number(facts, "onset_count") < 8.0:
            return False
        if self._shape_number(facts, "pitched_event_ratio") < 0.85:
            return False
        if self._shape_number(facts, "sustained_tonal_frame_ratio") < 0.85:
            return False
        if self._shape_number(facts, "percussive_event_ratio") > 0.12:
            return False
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.12:
            return False
        if self._shape_number(facts, "spectral_flatness_mean") > 0.32:
            return False
        direct_voice_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("voice", "vocal", "choir", "spoken"),
            top_family="Instruments",
            max_score=18.0,
            max_brain_rank=8,
            max_physics_rank=8,
        )
        parent_audit = (
            facts.evidence.get("parent_role_audit") if isinstance(getattr(facts, "evidence", None), dict) else {}
        )
        detected_parent_role = (
            str(parent_audit.get("detected_parent_role") or "") if isinstance(parent_audit, dict) else ""
        )
        physics_top = self._top_physics_guess_path(facts)
        duration_sec = float(facts.feature_values_by_name.get("duration_sec", 0.0) or 0.0)
        long_pitched_loop_false_voice = bool(
            detected_parent_role == "pitched_music_loop"
            and duration_sec >= 4.0
            and not direct_voice_candidate
            and (
                "instrument loops" in physics_top
                or "synth" in physics_top
                or "woodwind" in physics_top
                or "sax" in physics_top
            )
        )
        if long_pitched_loop_false_voice:
            return False
        strong_fx_voice_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("human and voice", "spoken voice", "vocal"),
            top_family="FX",
            max_score=12.0,
            max_brain_rank=6,
            max_physics_rank=6,
        )
        strong_non_voice_fx_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("riser", "build", "siren", "alarm", "blip", "glitch", "stutter", "cat", "dog", "cricket"),
            top_family="FX",
            max_score=8.0,
            max_brain_rank=6,
            max_physics_rank=6,
        )
        return bool(not direct_voice_candidate and (strong_non_voice_fx_candidate or not strong_fx_voice_candidate))

    def _facts_support_final_voice_instrument(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        if facts is None:
            return False
        drum_loop_role = max(
            self._measured_role_value(facts, "drum_loop"),
            self._measured_role_value(facts, "low_rhythmic_drum_loop"),
            self._measured_role_value(facts, "percussive_drum_loop"),
            self._measured_role_value(facts, "bright_drum_loop"),
        )
        if drum_loop_role >= 0.60 and self._shared_candidate_has_top_family(
            winning_claim,
            ("drum loops", "drum loop"),
            top_family="Drums",
            max_score=44.0,
            max_brain_rank=10,
            max_physics_rank=10,
        ):
            return False
        if self._facts_support_measured_sax_loop(facts, winning_claim):
            return False
        if self._facts_support_synth_loop(facts, winning_claim):
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        shape_block = facts.evidence.get("shape_vote", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
        shape_scores = dict(shape_block.get("shape_scores", [])) if isinstance(shape_block, dict) else {}
        pitched_phrase_shape_score = self._safe_float(shape_scores.get("pitched_phrase_shape"), 0.0)
        f0_voiced = self._shape_number(facts, "f0_voiced_ratio")
        onset_count = self._shape_number(facts, "onset_count")
        tail_ratio = self._shape_number(facts, "tail_ratio")
        drumlike = self._shape_number(facts, "drumlike_frame_ratio")
        percussive = self._shape_number(facts, "percussive_event_ratio")
        duration = _feature_number_from_facts(facts, "duration_sec")
        voice_panel_score = max(
            _feature_number_from_facts(facts, "voice_score"),
            _feature_number_from_facts(facts, "human_spoken_voice_score"),
            _feature_number_from_facts(facts, "human_breath_mouth_score"),
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
        )
        drum_hit_score = max(
            _feature_number_from_facts(facts, "drum_hit_score"),
            self._subpanel_score(facts, "drum_hit_score"),
        )
        strong_brain_voice_candidate = self._facts_have_internal_candidate(
            facts,
            ("voice", "vocal", "choir"),
            top_family="Instruments",
            max_rank=1,
            max_score=1.25,
        )
        source_safe_voice_shape = bool(
            (
                shape in {"pitched_phrase_shape", "pitched_phrase", "bass_phrase", "vocal_phrase"}
                and shape_confidence >= 0.82
            )
            or pitched_phrase_shape_score >= 0.64
        )
        if (
            strong_brain_voice_candidate
            and source_safe_voice_shape
            and f0_voiced >= 0.70
            and voice_panel_score >= 0.38
            and drum_hit_score <= 0.58
            and drum_loop_role < 0.60
            and percussive <= 0.42
            and drumlike <= 0.42
        ):
            return True

        if (
            duration > 0.0
            and duration <= 0.20
            and shape in {"single_hit", "hit_with_tail"}
            and self._facts_have_internal_candidate(
                facts,
                ("drum", "snare", "tom", "percussion", "clap", "hat", "cymbal", "rim", "stick"),
                top_family="Drums",
                max_rank=2,
                max_score=1.2,
            )
            and self._facts_have_physics_drum_candidate(facts, max_rank=3, max_score=0.85)
        ):
            return False
        voice_identity = _feature_number_from_facts(facts, "formant_light_voice_identity")
        voice_strength = max(
            self._measured_role_value(facts, "vocal_music_phrase"),
            self._measured_role_value(facts, "vocal_phrase"),
            self._measured_role_value(facts, "vocal_one_shot"),
            self._measured_role_value(facts, "voiced_one_shot"),
        )
        voice_loop_panel = bool(
            shape in {"vocal_phrase", "pitched_phrase", "pitched_phrase_shape"}
            and shape_confidence >= 0.84
            and f0_voiced >= 0.70
            and self._subpanel_score(facts, "voice_score") >= 0.78
            and self._subpanel_score(facts, "human_spoken_voice_score") >= 0.74
            and self._shape_number(facts, "percussive_event_ratio") <= 0.18
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.18
            and (
                self._shape_number(facts, "onset_count") >= 4.0
                or self._shape_number(facts, "true_repetition_score") >= 0.45
                or getattr(facts, "is_loop_like", False)
            )
        )
        if voice_loop_panel:
            return True
        single_hit_voice_panel = bool(
            shape == "single_hit"
            and shape_confidence >= 0.72
            and onset_count <= 1.25
            and f0_voiced >= 0.70
            and max(
                self._subpanel_score(facts, "voice_score"),
                self._subpanel_score(facts, "human_spoken_voice_score"),
                self._subpanel_score(facts, "human_breath_mouth_score"),
                self._subpanel_score(facts, "fx_formant_score"),
            )
            >= 0.74
            and max(
                self._subpanel_score(facts, "human_spoken_voice_score"),
                self._subpanel_score(facts, "human_breath_mouth_score"),
            )
            >= 0.70
            and self._subpanel_score(facts, "drum_hit_score") <= 0.48
            and self._subpanel_score(facts, "drum_loop_source_score") <= 0.30
        )
        if single_hit_voice_panel:
            return True
        strong_measured_voice_loop = bool(
            shape == "vocal_phrase"
            and shape_confidence >= 0.84
            and self._measured_role_value(facts, "vocal_music_phrase") >= 0.86
            and f0_voiced >= 0.72
            and drumlike <= 0.12
        )
        strong_short_voice = bool(
            shape in {"vocal_phrase", "vocal_one_shot", "hit_with_tail"}
            and shape_confidence >= 0.70
            and onset_count <= 1.25
            and tail_ratio <= 0.12
            and f0_voiced >= 0.80
            and voice_strength >= 0.38
        )
        bright_bell_like = bool(
            self._shape_number(facts, "high_event_ratio") >= 0.35
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.16
        )
        direct_voice_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("voice", "vocal", "choir", "spoken"),
            top_family="Instruments",
            max_score=18.0,
            max_brain_rank=8,
            max_physics_rank=8,
        )
        fx_voice_candidate_count = self._shared_candidate_count_top_family(
            winning_claim,
            ("human and voice", "voice", "vocal", "spoken", "applause", "breath"),
            top_family="FX",
            max_score=24.0,
            max_brain_rank=10,
            max_physics_rank=10,
        )
        measured_roles = (
            facts.evidence.get("measured_roles", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
        )
        detected_parent_role = detected_parent_role_name(measured_roles if isinstance(measured_roles, dict) else {})
        if detected_parent_role == "pitched_music_loop" and duration >= 4.0:
            layer = (
                facts.evidence.get("physics_layer_decision")
                if isinstance(getattr(facts, "evidence", None), dict)
                else {}
            )
            if not isinstance(layer, dict):
                layer = {}
            branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
            branch_confidence = self._safe_float(
                layer.get("physics_layer_branch_confidence") or layer.get("instrument_branch_selected_confidence"),
                0.0,
            )
            rap_voice_texture = self._safe_float(layer.get("instrument_rap_voice_texture"), 0.0)
            human_voice_texture = self._safe_float(layer.get("instrument_human_voice_texture"), 0.0)
            human_voice_signal = bool(layer.get("instrument_human_voice_phrase_signal"))
            flat = self._physics_subpanel_flat(facts)
            human_spoken = self._safe_float(flat.get("human_spoken_voice_score"), 0.0)
            pitched_loop_voice_measurement = bool(
                branch == "Voice"
                and branch_confidence >= 0.56
                and (
                    (rap_voice_texture >= 0.58 and human_spoken >= 0.62 and f0_voiced >= 0.70)
                    or (human_voice_signal and human_voice_texture >= 0.66 and f0_voiced >= 0.55)
                )
                and percussive <= 0.18
                and drumlike <= 0.18
            )
            pitched_loop_voice_candidate = bool(
                (branch == "Voice" and branch_confidence >= 0.62)
                or (human_spoken >= 0.68 and f0_voiced >= 0.70)
            )
            if not (pitched_loop_voice_measurement or pitched_loop_voice_candidate):
                return False
        non_voice_instrument_count = self._shared_candidate_count_top_family(
            winning_claim,
            ("guitar", "strings", "cello", "synth", "piano", "keys", "woodwind", "sax", "brass"),
            top_family="Instruments",
            max_score=24.0,
            max_brain_rank=8,
            max_physics_rank=8,
        )
        fx_voice_cluster = fx_voice_candidate_count >= 2 and non_voice_instrument_count <= 1
        best_fx_spoken_or_vocal_score = self._shared_candidate_best_score_top_family(
            winning_claim,
            ("spoken voice", "vocal"),
            top_family="FX",
        )
        best_non_voice_instrument_score = self._shared_candidate_best_score_top_family(
            winning_claim,
            ("guitar", "strings", "cello", "synth", "piano", "keys", "woodwind", "sax", "brass"),
            top_family="Instruments",
        )
        strong_fx_spoken_voice_cluster = bool(
            fx_voice_candidate_count >= 3
            and best_fx_spoken_or_vocal_score <= 12.0
            and best_fx_spoken_or_vocal_score + 4.0 < best_non_voice_instrument_score
            and shape
            in {
                "vocal_phrase",
                "pitched_phrase",
                "repeated_phrase_loop",
                "mixed_instrument_loop",
                "compound_musical_loop",
                "sustained_pad",
            }
            and shape_confidence >= 0.68
            and self._shape_number(facts, "pitched_event_ratio") >= 0.58
            and self._shape_number(facts, "pitch_confidence") >= 0.48
            and f0_voiced >= 0.55
            and percussive <= 0.22
            and drumlike <= 0.22
        )
        physics_voice_measurement = False
        if isinstance(getattr(facts, "evidence", None), dict):
            layer = facts.evidence.get("physics_layer_decision")
            if isinstance(layer, dict):
                branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
                branch_confidence = self._safe_float(
                    layer.get("physics_layer_branch_confidence") or layer.get("instrument_branch_selected_confidence"),
                    0.0,
                )
                rap_voice_texture = self._safe_float(
                    layer.get("instrument_rap_voice_texture"),
                    0.0,
                )
                human_voice_texture = self._safe_float(
                    layer.get("instrument_human_voice_texture"),
                    0.0,
                )
                human_voice_signal = bool(layer.get("instrument_human_voice_phrase_signal"))
                physics_voice_measurement = bool(
                    branch == "Voice"
                    and branch_confidence >= 0.56
                    and (
                        (rap_voice_texture >= 0.58 and f0_voiced >= 0.70)
                        or (human_voice_signal and human_voice_texture >= 0.66 and f0_voiced >= 0.55)
                    )
                    and percussive <= 0.18
                    and drumlike <= 0.18
                    and shape
                    in {
                        "vocal_phrase",
                        "vocal_one_shot",
                        "hit_with_tail",
                        "bass_phrase",
                        "sustained_pad",
                        "pitched_phrase",
                        "repeated_phrase_loop",
                        "solo_phrase",
                        "echo_tail_hit",
                    }
                )
        vocal_layer_measurement = bool(
            detected_parent_role in {"pitched_music_phrase", "vocal_music_phrase"}
            and shape == "vocal_phrase"
            and shape_confidence >= 0.86
            and self._subpanel_score(facts, "human_spoken_voice_score") >= 0.78
            and self._subpanel_score(facts, "voice_score") >= 0.55
            and not (
                self._subpanel_score(facts, "keys_tonal_decay_score") >= 0.80
                and self._subpanel_score(facts, "human_spoken_voice_score") < 0.82
                and self._subpanel_score(facts, "voice_score") < 0.58
            )
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.82
            and self._shape_number(facts, "pitched_event_ratio") >= 0.88
            and self._shape_number(facts, "mid_event_ratio") >= 0.70
            and self._shape_number(facts, "low_event_ratio") <= 0.12
            and self._shape_number(facts, "high_event_ratio") <= 0.08
            and percussive <= 0.10
            and drumlike <= 0.10
        )
        strong_voice_measurement = bool(
            voice_strength >= 0.58
            or voice_identity >= 0.68
            or vocal_layer_measurement
            or (shape == "vocal_phrase" and self._measured_role_value(facts, "vocal_music_phrase") >= 0.50)
            or (
                (fx_voice_cluster or strong_fx_spoken_voice_cluster)
                and shape
                in {
                    "bass_phrase",
                    "vocal_phrase",
                    "sustained_pad",
                    "pitched_phrase",
                    "repeated_phrase_loop",
                    "mixed_instrument_loop",
                    "compound_musical_loop",
                }
            )
            or physics_voice_measurement
        )
        broad_voice_loop_measurement = bool(
            winning_claim.family == "Instruments"
            and winning_claim.sub_family == "Instrument Loops"
            and (
                shape == "bass_phrase"
                or (shape == "sustained_pad" and self._shape_number(facts, "mid_event_ratio") < 0.35)
            )
            and shape_confidence >= 0.70
            and f0_voiced >= 0.75
            and self._shape_number(facts, "spectral_flatness_mean") >= 0.24
            and percussive <= 0.14
            and drumlike <= 0.14
        )
        strong_internal_voice_candidate = bool(
            (
                (direct_voice_candidate or fx_voice_cluster or strong_fx_spoken_voice_cluster)
                and strong_voice_measurement
            )
            or broad_voice_loop_measurement
            or physics_voice_measurement
            or vocal_layer_measurement
        ) and bool(
            (
                f0_voiced >= 0.70
                or fx_voice_cluster
                or strong_fx_spoken_voice_cluster
                or broad_voice_loop_measurement
                or physics_voice_measurement
            )
            and shape
            in {
                "vocal_phrase",
                "vocal_one_shot",
                "hit_with_tail",
                "bass_phrase",
                "sustained_pad",
                "pitched_phrase",
                "repeated_phrase_loop",
                "solo_phrase",
                "echo_tail_hit",
                "mixed_instrument_loop",
                "compound_musical_loop",
            }
            and shape not in {"transition_riser", "transition_downlifter"}
            and (percussive <= 0.18 or ((fx_voice_cluster or strong_fx_spoken_voice_cluster) and percussive <= 0.22))
            and (drumlike <= 0.18 or ((fx_voice_cluster or strong_fx_spoken_voice_cluster) and drumlike <= 0.22))
            and not bright_bell_like
        )
        identity_voice = bool(
            voice_identity >= 0.70
            and f0_voiced >= 0.76
            and voice_strength >= 0.42
            and shape in {"vocal_phrase", "vocal_one_shot", "hit_with_tail", "echo_tail_hit"}
            and shape_confidence >= 0.74
            and percussive <= 0.16
            and drumlike <= 0.16
        )
        return (
            strong_measured_voice_loop
            or strong_short_voice
            or strong_internal_voice_candidate
            or identity_voice
            or physics_voice_measurement
            or vocal_layer_measurement
        )

    def _facts_support_clean_bass_loop(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        if facts is None:
            return False
        if not self._facts_have_decisive_bass_branch_authority(facts):
            return False
        layer = self._physics_layer(facts)
        bass_loop_subpanel = bool(
            str(layer.get("instrument_Bass_subpanel_selected") or "") == "BassLoop"
            and self._safe_float(layer.get("instrument_Bass_subpanel_confidence"), 0.0) >= 0.82
            and self._safe_float(layer.get("instrument_Bass_subpanel_margin"), 0.0) >= 0.012
        )
        flatness_limit = 0.18 if bass_loop_subpanel else 0.08
        return bool(
            _shape_vote_from_facts(facts) == "bass_phrase"
            and _shape_confidence_from_facts(facts) >= 0.88
            and self._measured_role_value(facts, "bass_loop") >= 0.55
            and self._measured_role_value(facts, "pitched_music_loop") >= 0.70
            and self._shape_number(facts, "pitch_confidence") >= 0.65
            and self._shape_number(facts, "spectral_flatness_mean") <= flatness_limit
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and (
                self._shared_candidate_has(
                    winning_claim,
                    ("/bass", "bass/", "808", "sub bass", "synth bass", "electric bass"),
                    max_score=36.0,
                    max_brain_rank=3,
                    max_physics_rank=8,
                )
                or self._facts_have_decisive_physics_bass_candidate(facts)
            )
        )

    def _facts_support_strong_synth_pad_loop(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for sustained low-air synth-pad loop bodies.

        This is a measured-physics decoy guard for sax/keys overreach.  It uses
        internal panel scores and frame/shape facts only; it does not read file
        names.  A real sax/reed loop needs some breath/reed air or noisy body.
        This pattern is cleaner, lower-air, and has stronger synth-pad evidence
        than sax/reed evidence.
        """
        if facts is None:
            return False
        synth_pad_score = max(
            self._subpanel_score(facts, "synth_pad_score"),
            _feature_number_from_facts(facts, "synth_pad_score"),
            _feature_number_from_facts(facts, "instruments_synths_synth_pad_one_shots_score"),
        )
        synth_chord_score = max(
            self._subpanel_score(facts, "synth_chord_score"),
            _feature_number_from_facts(facts, "synth_chord_score"),
            _feature_number_from_facts(facts, "instruments_synths_synth_chord_one_shots_score"),
        )
        synth_lead_score = max(
            self._subpanel_score(facts, "synth_lead_score"),
            _feature_number_from_facts(facts, "synth_lead_score"),
            _feature_number_from_facts(facts, "instruments_synths_synth_lead_one_shots_score"),
        )
        synth_branch_score = max(
            self._subpanel_score(facts, "synth_tonal_source_score"),
            _feature_number_from_facts(facts, "synth_tonal_source_score"),
        )
        sax_score = self._subpanel_score(facts, "woodwind_sax_score")
        reed_score = self._subpanel_score(facts, "reed_wind_score")
        shape = _shape_vote_from_facts(facts)
        return bool(
            synth_pad_score >= 0.74
            and synth_branch_score >= 0.68
            and synth_pad_score >= max(sax_score, reed_score) + 0.06
            and synth_pad_score >= max(synth_chord_score, synth_lead_score) - 0.04
            and shape
            in {
                "vocal_phrase",
                "pitched_phrase",
                "pitched_phrase_shape",
                "repeated_phrase_loop",
                "sustained_pad",
                "bass_phrase",
            }
            and _shape_confidence_from_facts(facts) >= 0.78
            and self._shape_number(facts, "pitched_event_ratio") >= 0.88
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.86
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.75
            and self._shape_number(facts, "low_event_ratio") >= 0.45
            and self._shape_number(facts, "high_event_ratio") <= 0.030
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.105
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )

    def _facts_support_synth_loop(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        if facts is None:
            return False
        strong_synth_pad_loop = self._facts_support_strong_synth_pad_loop(facts)
        strong_synth_loop_candidate = bool(
            self._shared_candidate_has_top_family(
                winning_claim,
                ("synth", "electronic"),
                top_family="Instruments",
                max_score=18.0,
                max_brain_rank=4,
                max_physics_rank=10,
            )
            or self._facts_have_internal_candidate(
                facts,
                ("synth", "electronic"),
                top_family="Instruments",
                max_rank=4,
                max_score=1.20,
            )
        )
        measured_synth_signal = bool(
            self._measured_score(
                facts,
                "synth_tonal_source_score",
                "synth_lead_score",
                "synth_pad_score",
                "synth_chord_score",
            ) >= 0.58
        )
        sax_or_woodwind_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("sax", "saxophone"),
            top_family="Instruments",
            max_score=36.0,
            max_brain_rank=3,
            max_physics_rank=8,
        ) or self._facts_have_internal_candidate(
            facts,
            ("sax", "saxophone"),
            top_family="Instruments",
            max_rank=4,
            max_score=1.0,
        )
        drum_loop_role = max(
            self._measured_role_value(facts, "drum_loop"),
            self._measured_role_value(facts, "low_rhythmic_drum_loop"),
            self._measured_role_value(facts, "percussive_drum_loop"),
            self._measured_role_value(facts, "bright_drum_loop"),
        )
        reed_witness_blocks_synth = bool(
            (sax_or_woodwind_candidate or self._facts_have_near_sax_candidate_evidence(facts))
            and self._reed_sax_physics_witness_score(facts) >= 0.62
            and not (
                strong_synth_loop_candidate
                and _shape_vote_from_facts(facts) == "bass_phrase"
                and self._shape_number(facts, "low_event_ratio") >= 0.58
            )
        )
        layer = self._physics_layer(facts)
        wood_sub = str(layer.get("instrument_Woodwinds_subpanel_selected") or "") if isinstance(layer, dict) else ""
        wood_sub_score = (
            self._safe_float(layer.get("instrument_Woodwinds_subpanel_confidence"), 0.0)
            if isinstance(layer, dict)
            else 0.0
        )
        sax_panel_blocks_synth = bool(
            wood_sub == "Sax"
            and wood_sub_score >= 0.72
            and self._shape_number(facts, "mid_event_ratio") >= 0.30
            and self._shape_number(facts, "high_event_ratio") <= 0.16
            and self._shape_number(facts, "spectral_flatness_mean") >= 0.045
            and (sax_or_woodwind_candidate or self._shape_number(facts, "spectral_flatness_mean") >= 0.20)
        )
        electric_keys_body = bool(
            self._shared_candidate_has_top_family(
                winning_claim,
                ("/keys/", "piano", "rhodes", "electric piano"),
                top_family="Instruments",
                max_score=44.0,
                max_brain_rank=18,
                max_physics_rank=24,
            )
            and self._shape_number(facts, "harmonic_energy_ratio") >= 0.45
            and self._shape_number(facts, "fundamental_dominance_ratio") <= 0.25
            and 0.015 <= self._shape_number(facts, "spectral_flatness_mean") <= 0.070
            and self._shape_number(facts, "high_event_ratio") <= 0.025
        )
        clean_key_chord_body = bool(
            self._shape_number(facts, "mid_event_ratio") >= 0.88
            and self._shape_number(facts, "high_event_ratio") <= 0.012
            and 0.015 <= self._shape_number(facts, "spectral_flatness_mean") <= 0.080
            and self._shape_number(facts, "fundamental_dominance_ratio") <= 0.20
        )
        if (
            (reed_witness_blocks_synth and not strong_synth_pad_loop)
            or (sax_panel_blocks_synth and not strong_synth_pad_loop)
            or (electric_keys_body and not strong_synth_pad_loop)
            or (clean_key_chord_body and not strong_synth_pad_loop)
            or drum_loop_role >= 0.58
        ):
            return False
        if strong_synth_pad_loop:
            return True
        bass_phrase_synth_loop = bool(
            _shape_vote_from_facts(facts) == "bass_phrase"
            and _shape_confidence_from_facts(facts) >= 0.88
            and self._measured_role_value(facts, "pitched_music_loop") >= 0.85
            and self._shape_number(facts, "spectral_flatness_mean") >= 0.12
            and self._shape_number(facts, "pitch_confidence") >= 0.50
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
            and (
                strong_synth_loop_candidate
                or self._shared_candidate_has_top_family(
                    winning_claim,
                    ("synth", "electronic"),
                    top_family="Instruments",
                    max_score=22.0,
                    max_brain_rank=8,
                    max_physics_rank=6,
                )
            )
        )
        pitched_phrase_synth_lead = bool(
            _shape_vote_from_facts(facts) == "pitched_phrase"
            and _shape_confidence_from_facts(facts) >= 0.88
            and self._shape_number(facts, "pitched_event_ratio") >= 0.80
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.78
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.10
            and self._shape_number(facts, "low_event_ratio") <= 0.06
            and self._shape_number(facts, "mid_event_ratio") >= 0.58
            and self._shape_number(facts, "high_event_ratio") <= 0.18
            and self._shape_number(facts, "spectral_entropy_mean") <= 0.38
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.20
        )
        clean_synth_pad_or_loop = bool(
            strong_synth_loop_candidate
            and _shape_vote_from_facts(facts)
            in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad", "vocal_phrase"}
            and _shape_confidence_from_facts(facts) >= 0.88
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.88
            and self._shape_number(facts, "low_event_ratio") <= 0.36
            and self._shape_number(facts, "mid_event_ratio") >= 0.55
            and self._shape_number(facts, "high_event_ratio") <= 0.09
            and self._shape_number(facts, "spectral_entropy_mean") <= 0.44
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.090
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )
        low_register_repeated_synth_loop = bool(
            (strong_synth_loop_candidate or measured_synth_signal)
            and _shape_vote_from_facts(facts) in {"repeated_phrase_loop", "sustained_pad", "bass_phrase"}
            and _shape_confidence_from_facts(facts) >= 0.80
            and self._measured_role_value(facts, "pitched_music_loop") >= 0.84
            and self._shape_number(facts, "pitched_event_ratio") >= 0.92
            and self._shape_number(facts, "pitch_confidence") >= 0.55
            and self._shape_number(facts, "low_event_ratio") >= 0.62
            and self._shape_number(facts, "mid_event_ratio") <= 0.24
            and self._shape_number(facts, "high_event_ratio") <= 0.09
            and 0.12 <= self._shape_number(facts, "spectral_flatness_mean") <= 0.45
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and self._measured_score(facts, "synth_tonal_source_score", "synth_lead_score", "synth_pad_score") >= 0.58
            and self._measured_score(facts, "voice_score", "human_spoken_voice_score") <= 0.48
            and self._measured_score(facts, "struck_keys_score", "struck_keys_authority_score") <= 0.52
            and self._measured_score(facts, "reed_wind_authority_score") <= 0.52
        )
        return (
            bass_phrase_synth_loop
            or pitched_phrase_synth_lead
            or clean_synth_pad_or_loop
            or low_register_repeated_synth_loop
        )

    def _facts_support_acoustic_piano_loop(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        """Return True for measured acoustic-piano chord-loop bodies.

        This is a narrow physics invariant for the protected obvious-piano case:
        low/mid chord body, tiny but present hammer/high-band component, very
        low flatness, and no drum/reed body.  It deliberately does not inspect
        source names.
        """
        if facts is None:
            return False
        if self._facts_show_true_voice_or_vocal_one_shot(facts):
            return False
        if self._facts_support_final_drum_loop(facts, winning_claim):
            return False
        synth_or_processed_lead_evidence = bool(
            _shape_vote_from_facts(facts) == "pitched_phrase"
            and self._shared_candidate_has_top_family(
                winning_claim,
                ("synth", "electronic"),
                top_family="Instruments",
                max_score=18.0,
                max_brain_rank=4,
                max_physics_rank=10,
            )
            and self._subpanel_score(facts, "synth_tonal_source_score") >= 0.62
            and not self._shared_candidate_has_top_family(
                winning_claim,
                ("/keys/", "rhodes", "electric piano"),
                top_family="Instruments",
                max_score=24.0,
                max_brain_rank=24,
                max_physics_rank=2,
            )
            and self._shape_number(facts, "low_event_ratio") <= 0.12
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )
        if synth_or_processed_lead_evidence:
            return False
        keys_body_dominates_reed_decoy = bool(
            self._subpanel_score(facts, "keys_tonal_decay_score") >= 0.80
            and self._shape_number(facts, "mid_event_ratio") >= 0.80
            and self._shape_number(facts, "high_event_ratio") <= 0.03
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
        )
        protected_acoustic_piano_loop = bool(
            _shape_vote_from_facts(facts) in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad"}
            and _shape_confidence_from_facts(facts) >= 0.78
            and self._subpanel_score(facts, "keys_tonal_decay_score") >= 0.78
            and self._subpanel_score(facts, "keys_hammer_attack_score") >= 0.40
            and self._subpanel_score(facts, "struck_keys_score") >= 0.56
            and self._subpanel_score(facts, "keys_partial_inharmonicity_score") <= 0.36
            and (self._reed_sax_physics_witness_score(facts) < 0.66 or keys_body_dominates_reed_decoy)
        )
        if protected_acoustic_piano_loop:
            return True
        piano_witness = self._piano_struck_physics_witness_score(facts)
        if piano_witness >= 0.62:
            return bool(
                _shape_vote_from_facts(facts) in {"pitched_phrase", "vocal_phrase", "sustained_pad"}
                and _shape_confidence_from_facts(facts) >= 0.78
                and self._shape_number(facts, "pitched_event_ratio") >= 0.82
                and self._shape_number(facts, "f0_voiced_ratio") >= 0.78
                and self._shape_number(facts, "low_event_ratio") <= 0.16
                and self._shape_number(facts, "mid_event_ratio") >= 0.82
                and self._shape_number(facts, "high_event_ratio") <= 0.018
                and self._shape_number(facts, "percussive_event_ratio") <= 0.18
                and self._shape_number(facts, "drumlike_frame_ratio") <= 0.16
                and self._reed_sax_physics_witness_score(facts) < 0.62
            )
        return bool(
            _shape_vote_from_facts(facts) in {"pitched_phrase", "vocal_phrase", "sustained_pad"}
            and _shape_confidence_from_facts(facts) >= 0.82
            and self._shape_number(facts, "pitched_event_ratio") >= 0.88
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.88
            and self._shape_number(facts, "pitch_confidence") >= 0.35
            and 0.18 <= self._shape_number(facts, "low_event_ratio") <= 0.45
            and 0.50 <= self._shape_number(facts, "mid_event_ratio") <= 0.78
            and 0.012 <= self._shape_number(facts, "high_event_ratio") <= 0.055
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.022
            and self._shape_number(facts, "spectral_entropy_mean") >= 0.38
            and self._shape_number(facts, "fundamental_dominance_ratio") <= 0.25
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
            and self._reed_sax_physics_witness_score(facts) < 0.62
        )

    def _facts_support_clean_keys_loop(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        if facts is None:
            return False
        if self._facts_show_true_voice_or_vocal_one_shot(facts):
            return False
        if self._facts_support_final_drum_loop(facts, winning_claim):
            return False
        if self._facts_support_strong_synth_pad_loop(facts):
            return False
        if getattr(facts, "is_single_event_like", False) and not getattr(facts, "is_loop_like", False):
            return False
        synth_or_processed_lead_evidence = bool(
            _shape_vote_from_facts(facts) == "pitched_phrase"
            and self._shared_candidate_has_top_family(
                winning_claim,
                ("synth", "electronic"),
                top_family="Instruments",
                max_score=18.0,
                max_brain_rank=4,
                max_physics_rank=10,
            )
            and not self._shared_candidate_has_top_family(
                winning_claim,
                ("/keys/", "rhodes", "electric piano"),
                top_family="Instruments",
                max_score=24.0,
                max_brain_rank=24,
                max_physics_rank=2,
            )
            and self._subpanel_score(facts, "synth_tonal_source_score") >= 0.62
            and self._shape_number(facts, "low_event_ratio") <= 0.12
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )
        if synth_or_processed_lead_evidence:
            return False
        narrow_synth_lead_body = bool(
            self._shape_number(facts, "low_event_ratio") <= 0.04
            and self._shape_number(facts, "high_event_ratio") <= 0.07
            and self._shape_number(facts, "spectral_entropy_mean") <= 0.28
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.06
        )
        decisive_synth_lead_body = bool(
            narrow_synth_lead_body
            and self._subpanel_score(facts, "synth_lead_score") >= 0.86
            and self._shared_candidate_has_top_family(
                winning_claim,
                ("synth", "synth lead"),
                top_family="Instruments",
                max_score=10.0,
                max_brain_rank=3,
                max_physics_rank=6,
            )
        )
        strong_synth_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("synth", "electronic"),
            top_family="Instruments",
            max_score=12.0,
            max_brain_rank=4,
            max_physics_rank=8,
        ) or self._facts_have_internal_candidate(
            facts,
            ("synth", "electronic"),
            top_family="Instruments",
            max_rank=4,
            max_score=1.10,
        )
        strong_guitar_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("guitar",),
            top_family="Instruments",
            max_score=14.0,
            max_brain_rank=3,
            max_physics_rank=8,
        ) or self._facts_have_internal_candidate(
            facts,
            ("guitar",),
            top_family="Instruments",
            max_rank=3,
            max_score=1.10,
        )
        piano_witness = self._piano_struck_physics_witness_score(facts)
        candidate_keys_evidence = self._shared_candidate_has_top_family(
            winning_claim,
            ("/keys/", "piano", "rhodes", "electric piano"),
            top_family="Instruments",
            max_score=44.0,
            max_brain_rank=18,
            max_physics_rank=24,
        ) or self._facts_have_internal_candidate(
            facts,
            ("/keys/", "piano", "rhodes", "electric piano"),
            top_family="Instruments",
            max_rank=18,
            max_score=4.25,
        )
        layer = (
            facts.evidence.get("physics_layer_decision") if isinstance(getattr(facts, "evidence", None), dict) else {}
        )
        if not isinstance(layer, dict):
            layer = {}
        physics_branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
        clean_electric_keys_loop_signal = bool(layer.get("instrument_clean_electric_keys_loop_signal"))
        keys_piano_panel_confidence = self._safe_number(
            layer.get("instrument_KeysPiano_subpanel_confidence"),
            0.0,
        )
        keys_piano_panel_margin = self._safe_number(
            layer.get("instrument_KeysPiano_subpanel_margin"),
            0.0,
        )
        electric_piano_panel = self._safe_number(
            layer.get("instrument_panel_KeysPiano_ElectricPiano"),
            0.0,
        )
        protected_struck_keys_chord_loop = bool(
            _shape_vote_from_facts(facts) in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad"}
            and _shape_confidence_from_facts(facts) >= 0.84
            and piano_witness >= 0.66
            and keys_piano_panel_confidence >= 0.80
            and keys_piano_panel_margin >= 0.18
            and electric_piano_panel >= 0.78
            and self._shape_number(facts, "pitched_event_ratio") >= 0.92
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.88
            and 0.18 <= self._shape_number(facts, "low_event_ratio") <= 0.40
            and 0.55 <= self._shape_number(facts, "mid_event_ratio") <= 0.72
            and self._shape_number(facts, "high_event_ratio") <= 0.025
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.012
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and self._measured_score(facts, "voice_score", "human_spoken_voice_score") <= 0.64
        )
        if (
            not protected_struck_keys_chord_loop
            and (
                physics_branch == "MixedInstrument"
                or bool(layer.get("compound_music_prefer_broad_loop"))
                or self._safe_number(layer.get("compound_music_multi_source_loop_disagreement"), 0.0) >= 0.70
            )
        ):
            return False
        direct_keys_candidate = bool(
            candidate_keys_evidence
            or piano_witness >= 0.68
            or clean_electric_keys_loop_signal
            or protected_struck_keys_chord_loop
        )
        protected_bright_rhodes_loop = bool(
            self._shared_candidate_has_top_family(
                winning_claim,
                ("/keys/", "rhodes", "electric piano"),
                top_family="Instruments",
                max_score=24.0,
                max_brain_rank=24,
                max_physics_rank=2,
            )
            and _shape_vote_from_facts(facts) in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad"}
            and _shape_confidence_from_facts(facts) >= 0.88
            and self._shape_number(facts, "pitched_event_ratio") >= 0.95
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.90
            and self._shape_number(facts, "low_event_ratio") <= 0.15
            and self._shape_number(facts, "mid_event_ratio") >= 0.65
            and self._shape_number(facts, "high_event_ratio") <= 0.18
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.070
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and self._subpanel_score(facts, "struck_keys_score") >= 0.56
            and self._subpanel_score(facts, "keys_tonal_decay_score") >= 0.78
            and self._subpanel_score(facts, "reed_wind_score") <= 0.55
            and not decisive_synth_lead_body
        )
        if protected_bright_rhodes_loop or protected_struck_keys_chord_loop:
            return True
        electric_keys_body = bool(
            direct_keys_candidate
            and self._shape_number(facts, "harmonic_energy_ratio") >= 0.45
            and self._shape_number(facts, "fundamental_dominance_ratio") <= 0.25
            and 0.015 <= self._shape_number(facts, "spectral_flatness_mean") <= 0.070
            and self._shape_number(facts, "high_event_ratio") <= 0.025
            and self._shape_number(facts, "pitched_event_ratio") >= 0.88
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.88
        )
        if not decisive_synth_lead_body and self._facts_support_acoustic_piano_loop(facts, winning_claim):
            return True
        if not decisive_synth_lead_body and clean_electric_keys_loop_signal and physics_branch == "KeysPiano":
            return True
        if (
            not decisive_synth_lead_body
            and piano_witness >= 0.68
            and candidate_keys_evidence
            and self._reed_sax_physics_witness_score(facts) < 0.62
        ):
            return True
        if not decisive_synth_lead_body and electric_keys_body and self._reed_sax_physics_witness_score(facts) < 0.62:
            return True
        return bool(
            _shape_vote_from_facts(facts) in {"pitched_phrase", "vocal_phrase", "sustained_pad"}
            and _shape_confidence_from_facts(facts) >= 0.84
            and self._measured_role_value(facts, "pitched_music_loop") >= 0.75
            and self._shape_number(facts, "pitched_event_ratio") >= 0.88
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.88
            and self._shape_number(facts, "high_event_ratio") < 0.045
            and self._shape_number(facts, "spectral_flatness_mean") < 0.055
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.10
            and direct_keys_candidate
            and not strong_synth_candidate
            and not strong_guitar_candidate
            and not narrow_synth_lead_body
            and self._reed_sax_physics_witness_score(facts) < 0.62
        )

    @staticmethod
    def _dry_probe_has_sax_or_reed(facts: SharedAudioFacts | None) -> bool:
        """Return True when the existing dry/wet probe exposes sax/reed evidence."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        probe = facts.evidence.get("dry_wet_conflict_probe")
        if not isinstance(probe, dict) or not bool(probe.get("enabled")):
            return False
        if bool(probe.get("dry_has_sax_or_reed")):
            return True
        buckets = probe.get("dry_source_buckets")
        if isinstance(buckets, (list, tuple, set)) and any(str(item) == "sax_reed" for item in buckets):
            return True
        paths = probe.get("dry_top_paths")
        if isinstance(paths, (list, tuple, set)):
            return any(
                any(token in str(path).lower().replace("\\", "/") for token in ("sax", "saxophone", "woodwind"))
                for path in paths
            )
        return False

    def _facts_support_dark_low_mid_sax_loop(
        self,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for dark low-mid sax/reed loop evidence without filename clues."""
        if facts is None:
            return False
        if self._facts_support_clean_vintage_keys_loop(facts):
            return False
        sax_score = self._subpanel_score(facts, "woodwind_sax_score")
        return bool(
            sax_score >= 0.64
            and sax_score >= self._subpanel_score(facts, "voice_score") + 0.06
            and sax_score >= self._subpanel_score(facts, "brass_trumpet_score") + 0.08
            and _shape_vote_from_facts(facts) in {"pitched_phrase", "vocal_phrase", "solo_phrase", "sustained_pad"}
            and _shape_confidence_from_facts(facts) >= 0.84
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.86
            and self._shape_number(facts, "high_event_ratio") <= 0.035
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.040
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.10
        )

    def _facts_support_clean_vintage_keys_loop(
        self,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for clean sustained electric-piano/keys loop evidence."""
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        pitched_event = self._shape_number(facts, "pitched_event_ratio")
        sustained_tonal = self._shape_number(facts, "sustained_tonal_frame_ratio")
        f0_voiced = self._shape_number(facts, "f0_voiced_ratio")
        low_ratio = self._shape_number(facts, "low_event_ratio")
        mid_ratio = self._shape_number(facts, "mid_event_ratio")
        high_ratio = self._shape_number(facts, "high_event_ratio")
        flatness = self._shape_number(facts, "spectral_flatness_mean")
        percussive = self._shape_number(facts, "percussive_event_ratio")
        drumlike = self._shape_number(facts, "drumlike_frame_ratio")
        keys_support = max(
            self._subpanel_score(facts, "struck_keys_score"),
            self._subpanel_score(facts, "keys_tonal_decay_score"),
            self._subpanel_score(facts, "instruments_keys_rhodes_one_shots_score"),
        )
        protected_electric_keys_chord = bool(
            shape == "pitched_phrase"
            and shape_confidence >= 0.88
            and pitched_event >= 0.95
            and sustained_tonal >= 0.92
            and f0_voiced >= 0.90
            and 0.18 <= low_ratio <= 0.45
            and mid_ratio >= 0.65
            and high_ratio <= 0.025
            and flatness <= 0.045
            and percussive <= 0.08
            and drumlike <= 0.08
            and keys_support >= 0.58
            and self._subpanel_score(facts, "keys_tonal_decay_score") >= 0.82
            and self._subpanel_score(facts, "keys_hammer_attack_score") >= 0.50
            and self._subpanel_score(facts, "woodwind_sax_score") <= 0.60
            and self._subpanel_score(facts, "reed_wind_score") <= 0.55
        )
        if protected_electric_keys_chord:
            return True
        return bool(
            shape in {"vocal_phrase", "pitched_phrase", "repeated_phrase_loop", "sustained_pad"}
            and shape_confidence >= 0.80
            and pitched_event >= 0.92
            and sustained_tonal >= 0.90
            and f0_voiced >= 0.78
            and 0.18 <= low_ratio <= 0.48
            and mid_ratio >= 0.60
            and high_ratio <= 0.035
            and flatness <= 0.030
            and percussive <= 0.08
            and drumlike <= 0.08
            and keys_support >= 0.45
        )

    def _facts_support_measured_sax_loop(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        if facts is None:
            return False
        if self._facts_support_mixed_instrument_loop_sax_decoy(facts, winning_claim):
            return False
        vocal_layer_not_sax = bool(
            _shape_vote_from_facts(facts) == "vocal_phrase"
            and self._subpanel_score(facts, "human_spoken_voice_score") >= 0.80
            and self._subpanel_score(facts, "voice_score") >= self._subpanel_score(facts, "woodwind_sax_score") - 0.12
            and self._shape_number(facts, "mid_event_ratio") >= 0.80
            and self._shape_number(facts, "low_event_ratio") <= 0.08
            and self._shape_number(facts, "high_event_ratio") <= 0.06
        )
        if vocal_layer_not_sax:
            return False
        strong_tonal_alert_fx = bool(
            self._subpanel_score(facts, "tonal_alert_siren_score") >= 0.82
            and max(
                self._subpanel_score(facts, "fx_siren_score"),
                self._subpanel_score(facts, "fx_alarm_score"),
                self._subpanel_score(facts, "fx_designed_noise_fx_siren_long_fx_score"),
                self._subpanel_score(facts, "fx_designed_noise_fx_alarm_long_fx_score"),
            )
            >= 0.58
            and self._shape_number(facts, "true_repetition_score") >= 0.72
            and self._shape_number(facts, "onset_count") >= 12.0
        )
        if strong_tonal_alert_fx:
            return False
        decisive_physics_sax = self._facts_have_decisive_physics_sax_candidate(facts)
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        sax_panel = self._subpanel_score(facts, "woodwind_sax_score")
        reed_panel = self._subpanel_score(facts, "reed_wind_score")
        voice_panel = self._subpanel_score(facts, "voice_score")
        synth_pad_score = max(
            self._subpanel_score(facts, "synth_pad_score"),
            _feature_number_from_facts(facts, "synth_pad_score"),
            _feature_number_from_facts(facts, "instruments_synths_synth_pad_one_shots_score"),
        )
        synth_chord_score = max(
            self._subpanel_score(facts, "synth_chord_score"),
            _feature_number_from_facts(facts, "synth_chord_score"),
            _feature_number_from_facts(facts, "instruments_synths_synth_chord_one_shots_score"),
        )
        synth_lead_score = max(
            self._subpanel_score(facts, "synth_lead_score"),
            _feature_number_from_facts(facts, "synth_lead_score"),
            _feature_number_from_facts(facts, "instruments_synths_synth_lead_one_shots_score"),
        )
        synth_branch_score = max(
            self._subpanel_score(facts, "synth_tonal_source_score"),
            _feature_number_from_facts(facts, "synth_tonal_source_score"),
        )
        close_physics_sax_candidate = False
        if isinstance(getattr(facts, "evidence", None), dict):
            physics_result = facts.evidence.get("physics_vote_result")
            guesses = physics_result.get("top_guesses") if isinstance(physics_result, dict) else None
            if isinstance(guesses, list) and guesses:
                best_score = None
                for idx, guess in enumerate(guesses[:3], start=1):
                    if not isinstance(guess, dict):
                        continue
                    try:
                        score = float(guess.get("score", idx + 1.0) or idx + 1.0)
                    except Exception:
                        score = float(idx + 1.0)
                    if best_score is None:
                        best_score = score
                    path = str(guess.get("folder_path") or guess.get("label") or "").lower().replace("\\", "/")
                    if (
                        path.startswith("instruments/")
                        and ("sax" in path or "saxophone" in path)
                        and score <= 0.62
                        and best_score is not None
                        and score <= best_score + 0.10
                    ):
                        close_physics_sax_candidate = True
                        break
        bright_reed_sax_over_voice = bool(
            sax_panel >= 0.55
            and reed_panel >= 0.60
            and self._shape_number(facts, "high_event_ratio") >= 0.32
            and self._shape_number(facts, "spectral_flatness_mean") >= 0.20
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.80
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )
        sax_over_voice_false_positive = bool(
            ((sax_panel >= 0.66 or (close_physics_sax_candidate and sax_panel >= 0.62)) or bright_reed_sax_over_voice)
            and reed_panel >= 0.58
            and (
                decisive_physics_sax
                or (
                    close_physics_sax_candidate
                    and (voice_panel <= 0.70 or sax_panel >= voice_panel - 0.02 or bright_reed_sax_over_voice)
                )
                or bright_reed_sax_over_voice
                or (self._facts_have_strong_sax_candidate_evidence(facts) and sax_panel >= voice_panel - 0.02)
                or sax_panel >= voice_panel + 0.035
            )
            and shape
            in {
                "vocal_phrase",
                "pitched_phrase",
                "pitched_phrase_shape",
                "solo_phrase",
                "sustained_pad",
                "repeated_phrase_loop",
            }
            and shape_confidence >= 0.70
            and self._shape_number(facts, "pitched_event_ratio") >= 0.78
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.70
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )
        if self._facts_show_true_voice_or_vocal_one_shot(facts) and not (
            decisive_physics_sax or sax_over_voice_false_positive
        ):
            return False
        if self._facts_support_final_drum_loop(facts, winning_claim):
            return False
        close_physics_sax_candidate = False
        if isinstance(getattr(facts, "evidence", None), dict):
            physics_result = facts.evidence.get("physics_vote_result")
            guesses = physics_result.get("top_guesses") if isinstance(physics_result, dict) else None
            if isinstance(guesses, list) and guesses:
                best_score = None
                for idx, guess in enumerate(guesses[:3], start=1):
                    if not isinstance(guess, dict):
                        continue
                    try:
                        score = float(guess.get("score", idx + 1.0) or idx + 1.0)
                    except Exception:
                        score = float(idx + 1.0)
                    if best_score is None:
                        best_score = score
                    path = str(guess.get("folder_path") or guess.get("label") or "").lower().replace("\\", "/")
                    if (
                        path.startswith("instruments/")
                        and ("sax" in path or "saxophone" in path)
                        and score <= 0.62
                        and best_score is not None
                        and score <= best_score + 0.10
                    ):
                        close_physics_sax_candidate = True
                        break
        strong_direct_sax_candidate = self._facts_have_strong_sax_candidate_evidence(facts)
        near_sax_candidate = self._facts_have_near_sax_candidate_evidence(facts)
        reed_witness_score = self._reed_sax_physics_witness_score(facts)
        shared_sax_candidate = self._shared_candidate_has_top_family(
            winning_claim,
            ("sax", "saxophone"),
            top_family="Instruments",
            max_score=44.0,
            max_brain_rank=8,
            max_physics_rank=12,
        )
        sax_candidate = shared_sax_candidate or strong_direct_sax_candidate or close_physics_sax_candidate
        mid_ratio = self._shape_number(facts, "mid_event_ratio")
        high_ratio = self._shape_number(facts, "high_event_ratio")
        flatness = self._shape_number(facts, "spectral_flatness_mean")
        onset_count = self._shape_number(facts, "onset_count")
        f0_voiced = self._shape_number(facts, "f0_voiced_ratio")
        pitched_event = self._shape_number(facts, "pitched_event_ratio")
        sustained_tonal = self._shape_number(facts, "sustained_tonal_frame_ratio")
        percussive = self._shape_number(facts, "percussive_event_ratio")
        drumlike = self._shape_number(facts, "drumlike_frame_ratio")
        entropy = self._shape_number(facts, "spectral_entropy_mean")
        low_ratio = self._shape_number(facts, "low_event_ratio")
        moderate_candidate_sax_loop = bool(
            (shared_sax_candidate or strong_direct_sax_candidate or near_sax_candidate)
            and shape in {"repeated_phrase_loop", "pitched_phrase", "solo_phrase", "sustained_pad"}
            and shape_confidence >= 0.74
            and sax_panel >= 0.52
            and reed_panel >= 0.50
            and sax_panel >= voice_panel + 0.10
            and self._shape_number(facts, "pitched_event_ratio") >= 0.84
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.68
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )
        strong_synth_pad_body_decoy = bool(
            synth_pad_score >= 0.74
            and synth_branch_score >= 0.68
            and synth_pad_score >= max(sax_panel, reed_panel) + 0.06
            and synth_pad_score >= max(synth_chord_score, synth_lead_score) - 0.04
            and shape
            in {
                "vocal_phrase",
                "pitched_phrase",
                "pitched_phrase_shape",
                "repeated_phrase_loop",
                "sustained_pad",
                "bass_phrase",
            }
            and shape_confidence >= 0.78
            and pitched_event >= 0.88
            and sustained_tonal >= 0.86
            and f0_voiced >= 0.75
            and low_ratio >= 0.45
            and high_ratio <= 0.030
            and flatness <= 0.105
            and percussive <= 0.12
            and drumlike <= 0.12
        )
        if strong_synth_pad_body_decoy:
            return False

        broad_reed_body = bool(mid_ratio >= 0.30 and 0.035 <= high_ratio <= 0.14 and flatness >= 0.22)
        clean_mid_reed_body = bool(mid_ratio >= 0.55 and 0.040 <= high_ratio <= 0.16 and flatness >= 0.050)
        bright_reed_body = bool(high_ratio >= 0.25 and flatness >= 0.20 and reed_panel >= 0.60 and sax_panel >= 0.55)
        clean_tonal_sax_body = bool(
            not sax_candidate
            and shape in {"sustained_pad", "pitched_phrase", "vocal_phrase", "solo_phrase", "repeated_phrase_loop"}
            and shape_confidence >= 0.88
            and onset_count >= 18.0
            and pitched_event >= 0.90
            and f0_voiced >= 0.85
            and sustained_tonal >= 0.85
            and 0.32 <= mid_ratio <= 0.58
            and 0.025 <= high_ratio <= 0.085
            and 0.040 <= flatness <= 0.180
            and entropy <= 0.45
            and percussive <= 0.08
            and drumlike <= 0.08
        )
        candidate_free_reverb_sax_body = bool(
            not sax_candidate
            and shape in {"sustained_pad", "solo_phrase"}
            and shape_confidence >= 0.70
            and onset_count >= 8.0
            and pitched_event >= 0.95
            and f0_voiced >= 0.75
            and self._shape_number(facts, "pitch_confidence") >= 0.30
            and sustained_tonal >= 0.72
            and 0.30 <= mid_ratio <= 0.55
            and 0.035 <= high_ratio <= 0.09
            and flatness >= 0.24
            and entropy <= 0.42
            and percussive <= 0.08
            and drumlike <= 0.18
            and self._measured_role_value(facts, "vocal_music_phrase") <= 0.10
            and _feature_number_from_facts(facts, "formant_light_voice_identity") <= 0.10
        )
        measured_reed_body = bool(
            shape
            in {
                "sustained_pad",
                "pitched_phrase",
                "pitched_phrase_shape",
                "vocal_phrase",
                "solo_phrase",
                "repeated_phrase_loop",
            }
            and shape_confidence >= 0.70
            and pitched_event >= 0.66
            and f0_voiced >= 0.70
            and (broad_reed_body or clean_mid_reed_body or bright_reed_body)
            and percussive <= 0.16
            and drumlike <= 0.20
        )
        low_ratio = self._shape_number(facts, "low_event_ratio")
        bright_bell_like = bool(high_ratio >= 0.30 and high_ratio > mid_ratio and flatness <= 0.14 and entropy <= 0.50)
        narrow_synth_like = bool(
            low_ratio <= 0.04 and mid_ratio >= 0.60 and high_ratio <= 0.18 and entropy <= 0.36 and flatness <= 0.18
        )
        clean_keys_like_loop = bool(mid_ratio >= 0.80 and high_ratio <= 0.025 and flatness <= 0.065 and entropy <= 0.38)
        layer = (
            facts.evidence.get("physics_layer_decision") if isinstance(getattr(facts, "evidence", None), dict) else {}
        )
        if not isinstance(layer, dict):
            layer = {}
        physics_branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
        woodwind_source = bool(
            layer.get("instrument_woodwind_source_signal") or layer.get("instrument_clean_tonal_reed_solo_signal")
        )
        clean_electric_keys_loop_signal = bool(layer.get("instrument_clean_electric_keys_loop_signal"))
        woodwind_branch = self._safe_float(layer.get("instrument_branch_Woodwinds"), 0.0)
        self._safe_float(layer.get("instrument_branch_KeysPiano"), 0.0)
        self._safe_float(layer.get("instrument_branch_Synth"), 0.0)
        voice_branch = self._safe_float(layer.get("instrument_branch_Voice"), 0.0)
        measured_reed_source = bool(layer.get("instrument_reed_woodwind_source_signal"))
        measured_plucked_source = bool(layer.get("instrument_plucked_string_source_signal"))
        selected_branch_confidence = self._safe_float(layer.get("instrument_branch_selected_confidence"), 0.0)
        compound_strength = self._safe_float(layer.get("compound_music_strength"), 0.0)
        selected_subpanel = str(layer.get(f"instrument_{physics_branch}_subpanel_selected") or "")
        selected_subpanel_confidence = self._safe_float(
            layer.get(f"instrument_{physics_branch}_subpanel_confidence"), 0.0
        )
        selected_subpanel_margin = self._safe_float(layer.get(f"instrument_{physics_branch}_subpanel_margin"), 0.0)
        fx_branch = str(layer.get("fx_branch_selected") or "")
        fx_branch_confidence = self._safe_float(layer.get("fx_branch_selected_confidence"), 0.0)
        fx_conflict = self._safe_float(layer.get("fx_role_conflict_strength"), 1.0)
        wood_sub = str(layer.get("instrument_Woodwinds_subpanel_selected") or "")
        wood_sub_score = self._safe_float(layer.get("instrument_Woodwinds_subpanel_confidence"), 0.0)
        wood_sub_margin = self._safe_float(layer.get("instrument_Woodwinds_subpanel_margin"), 0.0)
        pluck_sub = str(layer.get("instrument_PluckedString_subpanel_selected") or "")
        pluck_sub_score = self._safe_float(layer.get("instrument_PluckedString_subpanel_confidence"), 0.0)
        pluck_sub_margin = self._safe_float(layer.get("instrument_PluckedString_subpanel_margin"), 0.0)
        plucked_rival_subpanel_conflict = bool(
            not measured_reed_source
            and pluck_sub
            in {
                "AcousticGuitar",
                "ElectricGuitar",
                "NylonOrSoftPluck",
                "WorldPluck",
            }
            and pluck_sub_score >= 0.66
            and (pluck_sub_margin >= 0.055 or (wood_sub != "Sax" and pluck_sub_score >= wood_sub_score + 0.02))
            and not bool(layer.get("instrument_dark_low_mid_reed_loop_signal"))
            and not (
                wood_sub == "Sax"
                and wood_sub_score >= 0.74
                and wood_sub_margin >= 0.09
                and pluck_sub_score <= wood_sub_score - 0.08
            )
        )
        woodwind_sax_subpanel_authority = bool(
            not woodwind_source
            and wood_sub == "Sax"
            and wood_sub_score >= 0.74
            and wood_sub_margin >= 0.09
            and shape in {"pitched_phrase", "solo_phrase", "sustained_pad", "bass_phrase", "repeated_phrase_loop"}
            and shape_confidence >= 0.80
            and pitched_event >= 0.90
            and f0_voiced >= 0.82
            and sustained_tonal >= 0.84
            and mid_ratio >= 0.30
            and high_ratio <= 0.10
            and 0.08 <= flatness <= 0.36
            and entropy <= 0.46
            and percussive <= 0.10
            and drumlike <= 0.12
        )
        dark_low_mid_sax_subpanel_authority = bool(
            not woodwind_source
            and wood_sub == "Sax"
            and wood_sub_score >= 0.84
            and wood_sub_margin >= 0.080
            and woodwind_branch >= 0.62
            and shape in {"bass_phrase", "pitched_phrase", "solo_phrase", "sustained_pad", "repeated_phrase_loop"}
            and shape_confidence >= 0.80
            and pitched_event >= 0.90
            and f0_voiced >= 0.75
            and sustained_tonal >= 0.82
            and low_ratio >= 0.50
            and high_ratio <= 0.045
            and 0.012 <= flatness <= 0.120
            and entropy <= 0.36
            and percussive <= 0.10
            and drumlike <= 0.12
        )
        strong_low_mid_sax_subpanel_authority = bool(
            not woodwind_source
            and wood_sub == "Sax"
            and wood_sub_score >= 0.82
            and wood_sub_margin >= 0.045
            and pluck_sub_score <= 0.58
            and woodwind_branch >= 0.70
            and shape in {"bass_phrase", "pitched_phrase", "solo_phrase", "sustained_pad", "repeated_phrase_loop"}
            and shape_confidence >= 0.80
            and pitched_event >= 0.90
            and f0_voiced >= 0.75
            and sustained_tonal >= 0.82
            and low_ratio >= 0.35
            and high_ratio <= 0.13
            and 0.12 <= flatness <= 0.36
            and entropy <= 0.42
            and percussive <= 0.10
            and drumlike <= 0.12
        )
        sax_subpanel_authority = bool(
            woodwind_sax_subpanel_authority
            or strong_low_mid_sax_subpanel_authority
            or dark_low_mid_sax_subpanel_authority
            or (
                close_physics_sax_candidate
                and sax_panel >= 0.62
                and reed_panel >= 0.58
                and shape in {"pitched_phrase", "solo_phrase", "sustained_pad", "bass_phrase", "repeated_phrase_loop"}
                and shape_confidence >= 0.78
                and pitched_event >= 0.90
                and f0_voiced >= 0.75
                and percussive <= 0.10
                and drumlike <= 0.12
            )
        )
        sax_candidate_without_physics_reed = bool(
            sax_candidate
            and not shared_sax_candidate
            and not (
                woodwind_source
                or measured_reed_source
                or sax_subpanel_authority
                or (physics_branch in {"Woodwinds", "ReedWoodwind"} and woodwind_branch >= 0.72)
            )
        )
        strong_fx_role_competes_with_sax = bool(
            not (woodwind_source or sax_subpanel_authority)
            and fx_branch
            in {
                "SirenAlarm",
                "TextureAmbience",
                "MachineMechanical",
                "FoleyMaterial",
                "SmallObjectCluster",
                "HumanCreatureFX",
                "FormantFX",
                "RadioElectrical",
                "DesignedNoiseHybrid",
                "BlipBeep",
                "GlitchStutter",
            }
            and fx_branch_confidence >= 0.70
            and fx_branch_confidence >= woodwind_branch + 0.08
            and fx_conflict < 0.58
        )
        strong_non_woodwind_subpanel = bool(
            selected_subpanel and selected_subpanel_confidence >= 0.76 and selected_subpanel_margin >= 0.075
        )
        measured_branch_identity_conflict = bool(
            not (woodwind_source or sax_subpanel_authority)
            and physics_branch
            and physics_branch not in {"Woodwinds", "ReedWoodwind"}
            and selected_branch_confidence >= 0.70
            and (
                strong_non_woodwind_subpanel
                or (physics_branch == "MixedInstrument" and compound_strength >= 0.54)
                or (
                    physics_branch in {"MalletBell", "Synth", "KeysPiano", "Strings", "Brass", "Bass"}
                    and selected_branch_confidence >= max(0.80, woodwind_branch + 0.02)
                )
            )
        )
        clean_low_mid_keys_like_loop = bool(
            shape in {"pitched_phrase", "repeated_phrase_loop", "solo_phrase", "sustained_pad", "bass_phrase"}
            and shape_confidence >= 0.70
            and pitched_event >= 0.88
            and sustained_tonal >= 0.80
            and 0.18 <= low_ratio <= 0.70
            and mid_ratio >= 0.42
            and high_ratio <= 0.045
            and flatness <= 0.085
            and (
                _feature_number_from_facts(facts, "presence_ratio_2000_8000hz")
                + _feature_number_from_facts(facts, "air_ratio_gt_8000hz")
            )
            <= 0.045
            and max(
                _feature_number_from_facts(facts, "body_noise_ratio"),
                _feature_number_from_facts(facts, "tail_noise_ratio"),
            )
            <= 0.28
            and percussive <= 0.12
            and drumlike <= 0.12
        )
        clean_synth_pad_like_loop = bool(
            shape in {"pitched_phrase", "vocal_phrase", "repeated_phrase_loop", "sustained_pad", "bass_phrase"}
            and shape_confidence >= 0.70
            and pitched_event >= 0.88
            and sustained_tonal >= 0.82
            and low_ratio >= 0.48
            and high_ratio <= 0.020
            and flatness <= 0.095
            and (
                _feature_number_from_facts(facts, "presence_ratio_2000_8000hz")
                + _feature_number_from_facts(facts, "air_ratio_gt_8000hz")
            )
            <= 0.022
            and percussive <= 0.12
            and drumlike <= 0.12
        )
        bright_synth_lead_like_loop = bool(
            shape in {"pitched_phrase", "vocal_phrase"}
            and shape_confidence >= 0.70
            and pitched_event >= 0.88
            and sustained_tonal >= 0.82
            and low_ratio <= 0.03
            and mid_ratio >= 0.62
            and high_ratio >= 0.20
            and flatness <= 0.16
            and entropy <= 0.42
            and _feature_number_from_facts(facts, "fundamental_dominance_ratio") >= 0.45
            and percussive <= 0.12
            and drumlike <= 0.12
        )
        mixed_compound_loop_decoy = bool(
            not (woodwind_source or sax_subpanel_authority)
            and physics_branch == "MixedInstrument"
            and compound_strength >= 0.58
            and pitched_event >= 0.88
            and sustained_tonal >= 0.82
            and high_ratio <= 0.09
            and self._shape_number(facts, "instrument_plus_fx_loop_score") >= 0.54
        )
        physics_branch_says_keys_or_synth = bool(
            not woodwind_source
            and physics_branch in {"KeysPiano", "Synth"}
            and selected_branch_confidence >= woodwind_branch + 0.12
            and pitched_event >= 0.86
            and sustained_tonal >= 0.82
            and high_ratio <= 0.18
            and flatness <= 0.12
        )
        vocal_branch_competes_with_sax = bool(
            not woodwind_source
            and shape == "vocal_phrase"
            and voice_branch >= 0.60
            and voice_branch >= woodwind_branch - 0.02
        )
        bright_vocal_shot_decoy = bool(
            not woodwind_source
            and shape == "vocal_phrase"
            and physics_branch == "MalletBell"
            and voice_branch >= 0.58
            and high_ratio >= 0.28
        )
        synthetic_vocal_phrase_decoy = bool(
            not woodwind_source
            and shape == "vocal_phrase"
            and physics_branch == "Woodwinds"
            and low_ratio <= 0.22
            and mid_ratio >= 0.64
            and 0.08 <= high_ratio <= 0.18
            and flatness >= 0.20
            and entropy >= 0.48
            and max(
                _feature_number_from_facts(facts, "body_noise_ratio"),
                _feature_number_from_facts(facts, "tail_noise_ratio"),
            )
            >= 0.32
            and _feature_number_from_facts(facts, "harmonic_energy_ratio") <= 0.48
        )
        clean_mid_vocal_phrase_decoy = bool(
            not woodwind_source
            and shape == "vocal_phrase"
            and shape_confidence >= 0.88
            and pitched_event >= 0.92
            and sustained_tonal >= 0.90
            and f0_voiced >= 0.75
            and low_ratio <= 0.46
            and mid_ratio >= 0.55
            and high_ratio <= 0.030
            and percussive <= 0.08
            and drumlike <= 0.08
            and _feature_number_from_facts(facts, "harmonic_energy_ratio") <= 0.42
        )
        clean_vintage_keys_loop_decoy = self._facts_support_clean_vintage_keys_loop(facts)
        weak_woodwind_authority = not (woodwind_source or measured_reed_source or sax_subpanel_authority)
        sax_identity_decoy = bool(
            (
                weak_woodwind_authority
                and (
                    clean_low_mid_keys_like_loop
                    or clean_synth_pad_like_loop
                    or bright_synth_lead_like_loop
                    or mixed_compound_loop_decoy
                    or physics_branch_says_keys_or_synth
                    or vocal_branch_competes_with_sax
                    or bright_vocal_shot_decoy
                    or synthetic_vocal_phrase_decoy
                    or clean_mid_vocal_phrase_decoy
                    or clean_vintage_keys_loop_decoy
                    or clean_electric_keys_loop_signal
                    or measured_branch_identity_conflict
                    or strong_fx_role_competes_with_sax
                )
            )
            or sax_candidate_without_physics_reed
            or measured_plucked_source
            or plucked_rival_subpanel_conflict
        )
        clean_piano_like_loop = bool(
            low_ratio <= 0.40 and mid_ratio >= 0.55 and high_ratio <= 0.025 and flatness <= 0.020
        )
        clean_electric_piano_like_loop = bool(
            low_ratio <= 0.35 and mid_ratio >= 0.70 and high_ratio <= 0.025 and flatness <= 0.045 and entropy <= 0.44
        )
        low_heavy_weak_sax_evidence = bool(
            low_ratio >= 0.50
            and mid_ratio <= 0.42
            and high_ratio <= 0.16
            and entropy >= 0.42
            and not strong_direct_sax_candidate
            and not (near_sax_candidate and reed_witness_score >= 0.62)
        )
        low_synth_pad_like = bool(low_ratio >= 0.75 and high_ratio <= 0.015 and flatness <= 0.045 and entropy <= 0.34)
        sax_body_energy = bool(mid_ratio >= 0.05 and 0.0 <= high_ratio <= 0.70 and (mid_ratio + high_ratio) >= 0.14)
        candidate_reed_body = bool(
            sax_candidate
            and shape
            in {
                "pitched_phrase",
                "sustained_pad",
                "vocal_phrase",
                "bass_phrase",
                "hit_with_tail",
                "solo_phrase",
                "repeated_phrase_loop",
            }
            and shape_confidence >= 0.70
            and pitched_event >= 0.66
            and f0_voiced >= 0.64
            and sax_body_energy
            and percussive <= 0.20
            and drumlike <= 0.22
            and not bright_bell_like
            and not narrow_synth_like
            and not clean_keys_like_loop
            and not clean_low_mid_keys_like_loop
            and not sax_identity_decoy
            and not clean_piano_like_loop
            and not clean_electric_piano_like_loop
            and not low_heavy_weak_sax_evidence
            and not low_synth_pad_like
        )
        reed_witness_sax_body = bool(
            near_sax_candidate
            and reed_witness_score >= 0.62
            and not bright_bell_like
            and not narrow_synth_like
            and not clean_keys_like_loop
            and not clean_low_mid_keys_like_loop
            and not sax_identity_decoy
            and not clean_piano_like_loop
            and not clean_electric_piano_like_loop
            and not low_synth_pad_like
        )
        return (
            moderate_candidate_sax_loop
            or (sax_over_voice_false_positive and not clean_vintage_keys_loop_decoy)
            or reed_witness_sax_body
            or candidate_reed_body
            or woodwind_sax_subpanel_authority
            or strong_low_mid_sax_subpanel_authority
            or dark_low_mid_sax_subpanel_authority
            or clean_tonal_sax_body
            or candidate_free_reverb_sax_body
        ) or (
            sax_candidate
            and measured_reed_body
            and not bright_bell_like
            and not narrow_synth_like
            and not clean_keys_like_loop
            and not clean_low_mid_keys_like_loop
            and not sax_identity_decoy
            and not clean_piano_like_loop
            and not clean_electric_piano_like_loop
            and not low_heavy_weak_sax_evidence
            and not low_synth_pad_like
        )

    def _facts_support_clean_pitched_instrument_loop(
        self,
        facts: SharedAudioFacts | None,
        winning_claim: ConsensusClaim,
    ) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        instrument_support_count = self._shared_candidate_count_top_family(
            winning_claim,
            ("guitar", "synth", "keys", "piano", "strings", "cello", "woodwind", "sax", "brass", "voice", "vocal"),
            top_family="Instruments",
            max_score=38.0,
            max_brain_rank=14,
            max_physics_rank=14,
        )
        best_fx_score = self._shared_candidate_best_score_top_family(
            winning_claim,
            ("siren", "alarm", "beep", "blip", "glitch", "stutter", "impact", "riser", "drop", "downlifter"),
            top_family="FX",
        )
        best_instrument_score = self._shared_candidate_best_score_top_family(
            winning_claim,
            ("guitar", "synth", "keys", "piano", "strings", "cello", "woodwind", "sax", "brass", "voice", "vocal"),
            top_family="Instruments",
        )
        strong_concrete_fx_identity = bool(
            (best_fx_score <= 6.0 and best_fx_score + 2.0 < best_instrument_score)
            or (best_fx_score < 10.0 and best_fx_score + 4.0 < best_instrument_score and instrument_support_count < 2)
        )
        return bool(
            shape in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad", "vocal_phrase", "bass_phrase"}
            and shape_confidence >= 0.70
            and self._measured_role_value(facts, "pitched_music_loop") >= 0.68
            and self._shape_number(facts, "pitched_event_ratio") >= 0.58
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.58
            and self._shape_number(facts, "percussive_event_ratio") <= 0.20
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.20
            and shape not in {"transition_riser", "transition_downlifter"}
            and instrument_support_count >= 1
            and not strong_concrete_fx_identity
        )

    def pick_winner(
        self,
        *,
        raw_claim: ConsensusClaim,
        claims: list[ConsensusClaim],
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim:
        """Return the strongest legal claim.

        Cross-family movement requires either a real voter-backed candidate or a
        formal review claim.  Inferred broad claims may still broaden within the
        same top family, but they cannot hijack a sample across families.
        """
        allowed_claims = [claim for claim in claims if self._claim_can_compete(raw_claim, claim, facts=facts)]
        supported_kick_claim = self._best_supported_kick_claim(allowed_claims, facts=facts)
        if supported_kick_claim is not None:
            return supported_kick_claim
        percussive_firewall = self._percussive_one_shot_parent_firewall(raw_claim, facts=facts)
        if percussive_firewall is not None:
            return percussive_firewall
        if not allowed_claims:
            ambiguous_review = self._review_for_ambiguous_instrument_hit(raw_claim, facts=facts)
            if ambiguous_review is not None:
                return ambiguous_review
            blocked_review = self._review_for_blocked_role_shape_conflict(raw_claim, claims, facts=facts)
            return blocked_review or raw_claim
        best_claim = self._choose_best_competing_claim(raw_claim, allowed_claims, facts=facts)
        if best_claim.is_review:
            return best_claim if best_claim.strength >= self.REVIEW_MIN_STRENGTH else raw_claim
        # Role/shape producers are witnesses, not emergency exits.
        # They may win only after _claim_can_compete() verifies real candidate
        # backing and raw-candidate contradiction rules.
        if best_claim.strength < raw_claim.strength and best_claim.family != raw_claim.family:
            if self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(
                raw_claim, best_claim, facts
            ):
                return best_claim
            if self._raw_generic_false_positive_may_be_replaced(raw_claim, best_claim):
                return best_claim
            if (
                best_claim.family == "Instruments"
                and best_claim.sub_family == "Instrument Loops"
                and best_claim.source == "ambiguous_fx_music_loop_broad_bucket"
                and raw_claim.family == "FX"
                and self._raw_fx_is_ambiguous_pitched_false_positive(raw_claim)
            ):
                return best_claim
            return raw_claim
        return best_claim

    def _best_supported_kick_claim(
        self,
        claims: list[ConsensusClaim],
        *,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Return a real kick claim before broad percussive fallback.

        The percussive parent firewall is intentionally conservative: when the
        raw winner is a non-drum and the audio is clearly a short percussive
        hit, it may move the sound into a broad Drums/Percussion parent.  That
        is safe for unknown hits, but it is too broad when an explicit,
        voter-backed kick claim already exists and the measured low-hit facts
        support it.
        """
        candidates = [
            claim
            for claim in claims
            if claim.family == "Drums"
            and claim.sub_family == "Kick One Shot"
            and claim.can_override
            and claim.source in self.PROFILE_SHORTCUT_SOURCES
            and self._facts_support_kick_one_shot_claim(claim, facts)
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda claim: (
                -(claim.strength or 0.0),
                self._score_or_default(claim.raw_candidate_score),
                claim.source,
            )
        )
        return candidates[0]

    def _review_for_blocked_role_shape_conflict(
        self,
        raw_claim: ConsensusClaim,
        claims: list[ConsensusClaim],
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim | None:
        """Return review when blocked role/shape evidence conflicts with raw.

        If the refactor blocks an old-style routing claim, leaving a concrete
        but contradicted raw winner in place would still be fake certainty.
        Strong blocked role/shape evidence belongs in review until a real
        candidate-backed claim decisively wins.
        """
        # If the raw winner is already a strong concrete voter-backed
        # candidate, blocked old-style routing evidence should not turn it into
        # review.  It should simply lose.  Review is for weak/ambiguous raw
        # winners where a blocked shortcut reveals a genuine conflict.
        if (
            self._raw_is_specific_candidate(raw_claim)
            and raw_claim.raw_candidate_score is not None
            and raw_claim.raw_candidate_score <= 8.0
        ):
            return None
        if self._raw_is_stable_parent_bucket(raw_claim):
            return None
        if (
            raw_claim.family == "FX"
            and raw_claim.sub_family == "Human and Voice FX"
            and (
                self._raw_human_voice_has_true_voice_role_support(raw_claim)
                or self._facts_support_true_voice_role(facts)
            )
        ):
            return None

        blocked = [
            claim
            for claim in claims
            if claim.can_override
            and not claim.is_review
            and claim.family != raw_claim.family
            and claim.strength >= 0.78
            and not (
                claim.family == "Drums"
                and claim.sub_family == "Drum Loops"
                and not self._facts_support_drum_loop_claim(claim, facts)
            )
            and not (claim.family == "FX" and claim.sub_family == "Human and Voice FX" and not claim.is_real_candidate)
            and (claim.source in self.ROLE_SHAPE_ROUTING_SOURCES or claim.source in self.PROFILE_SHORTCUT_SOURCES)
        ]
        if not blocked:
            return None
        blocked.sort(key=lambda claim: (-claim.strength, claim.raw_candidate_score or 9999.0, claim.source))
        strongest = blocked[0]
        if self._facts_support_measured_transition_fx(facts, raw_claim):
            shape = _shape_vote_from_facts(facts)
            if shape == "transition_riser":
                folder_path = "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
            else:
                folder_path = (
                    "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX"
                )
            return claim_from_folder_path(
                folder_path=folder_path,
                source="final_blocked_transition_fx_release_invariant",
                reason=(
                    "final structure invariant released blocked role conflict to "
                    "broad transition FX because measured transition shape was strong"
                ),
                shared=raw_claim.shared_candidates,
                raw_candidate_score=raw_claim.raw_candidate_score,
                brain_rank=raw_claim.brain_rank,
                physics_rank=raw_claim.physics_rank,
                shared_winner=raw_claim.shared_winner or raw_claim.folder_path,
                can_override=True,
                strength=max(0.90, strongest.strength),
                is_real_candidate=raw_claim.is_real_candidate,
            )
        if self._blocked_drum_loop_claim_is_safe(strongest, facts):
            return claim_from_folder_path(
                folder_path="Drums/Drum Loops/Loops",
                source="final_blocked_drum_loop_release_invariant",
                reason=(
                    "final structure invariant released a blocked broad Drum Loops "
                    "claim because measured loop facts and drum-family evidence agree"
                ),
                shared=raw_claim.shared_candidates,
                raw_candidate_score=strongest.raw_candidate_score,
                brain_rank=strongest.brain_rank,
                physics_rank=strongest.physics_rank,
                shared_winner=strongest.shared_winner or strongest.folder_path,
                can_override=True,
                strength=max(0.92, strongest.strength),
                is_real_candidate=strongest.is_real_candidate,
            )
        if self._blocked_broad_instrument_loop_claim_is_safe(raw_claim, strongest, facts):
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="final_blocked_instrument_loop_release_invariant",
                reason=(
                    "final structure invariant released a blocked broad Instrument "
                    "Loops claim because measured non-percussive pitched-loop facts "
                    "outweigh the contradicted raw leaf"
                ),
                shared=raw_claim.shared_candidates,
                raw_candidate_score=strongest.raw_candidate_score,
                brain_rank=strongest.brain_rank,
                physics_rank=strongest.physics_rank,
                shared_winner=strongest.shared_winner or strongest.folder_path,
                can_override=True,
                strength=max(0.91, strongest.strength),
                is_real_candidate=strongest.is_real_candidate,
            )
        if (
            raw_claim.family == "FX"
            and strongest.family == "Instruments"
            and strongest.sub_family == "Instrument Loops"
            and self._facts_support_blocked_fx_release(raw_claim, strongest, facts)
        ):
            fx_branch = self._measured_physics_fx_role_branch(facts)
            folder_path = self._measured_physics_fx_role_folder(fx_branch) if fx_branch else "FX/Hybrid Designed FX"
            source = "final_blocked_fx_release_invariant"
            reason = (
                "final structure invariant released a blocked role conflict to "
                "broad FX because the measured body was not safe as a clean "
                "Instrument Loops claim but the raw family and FX role evidence "
                "both stayed in FX"
            )
            if fx_branch in {"FormantFX", "HumanCreatureFX"}:
                folder_path = "Instruments/Instrument Loops/Loops"
                source = "final_blocked_formant_fx_instrument_loop_release"
                reason = (
                    "final structure invariant released a blocked role conflict to "
                    "broad Instrument Loops because the FX evidence was mainly "
                    "formant/human-shaped, while the measured body was a repeated "
                    "pitched musical phrase rather than a true human/voice FX source"
                )
            return claim_from_folder_path(
                folder_path=folder_path,
                source=source,
                reason=reason,
                shared=raw_claim.shared_candidates,
                raw_candidate_score=raw_claim.raw_candidate_score,
                brain_rank=raw_claim.brain_rank,
                physics_rank=raw_claim.physics_rank,
                shared_winner=raw_claim.shared_winner or raw_claim.folder_path,
                can_override=True,
                strength=max(0.90, raw_claim.strength, strongest.strength),
                is_real_candidate=raw_claim.is_real_candidate,
            )
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            reason=(
                "blocked old-style role/shape routing claim instead of forcing a folder: "
                f"raw={raw_claim.folder_path}; blocked={strongest.folder_path}; "
                f"source={strongest.source}; strength={strongest.strength:.2f}"
            ),
            source="blocked_role_shape_routing_review",
            shared=raw_claim.shared_candidates,
            winner=raw_claim,
            strength=max(0.82, strongest.strength),
        )

    def _facts_support_blocked_fx_release(
        self,
        raw_claim: ConsensusClaim,
        blocked_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when a blocked broad loop shortcut should fall back to FX.

        This is intentionally not a filename or leaf rescue.  It only prevents
        a weak, cross-family broad Instrument Loops shortcut from turning an FX
        family result into review when the measured body is not clean enough for
        the broad instrument-loop release.
        """
        if facts is None:
            return False
        if raw_claim.family != "FX":
            return False
        if blocked_claim.family != "Instruments" or blocked_claim.sub_family != "Instrument Loops":
            return False
        if self._blocked_broad_instrument_loop_claim_is_safe(raw_claim, blocked_claim, facts):
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape_confidence < 0.66:
            return False
        if shape in {"transition_riser", "transition_drop", "transition_downlifter"}:
            return True
        if shape not in {
            "repeated_phrase_loop",
            "mixed_instrument_loop",
            "compound_musical_loop",
            "hit_with_tail",
            "vocal_phrase",
            "pitched_phrase",
            "bass_phrase",
            "sustained_pad",
        }:
            return False
        fx_branch = self._measured_physics_fx_role_branch(facts)
        if fx_branch:
            return True
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        selected = parent.get("selected_top_families", []) if isinstance(parent, dict) else []
        if "FX" in selected and "Instruments" not in selected:
            return True
        physics_top = self._top_physics_guess_path(facts)
        return bool(physics_top.lower().startswith("fx/"))

    def _review_for_ambiguous_instrument_hit(
        self,
        raw_claim: ConsensusClaim,
        *,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Review weak Mallet/Bell one-shots instead of pretending source certainty.

        Bell-like spectra are a common false consensus target for short or
        noisy hits.  If no measured role confirms the source and nearby shared
        candidates disagree across Drums/FX, the honest phase-4 answer is
        review, not a concrete instrument leaf.
        """
        if raw_claim.family != "Instruments":
            return None
        raw_path = str(raw_claim.folder_path or raw_claim.label or "").lower().replace("\\", "/")
        if "one shot" not in raw_path and "one shots" not in raw_path:
            return None
        if "mallets and bells" not in raw_path and "bells and mallets" not in raw_path:
            return None
        if raw_claim.raw_candidate_score is not None and raw_claim.raw_candidate_score > 10.0:
            return None
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape_confidence >= 0.72 and shape in {"single_hit", "hit_with_tail"}:
            percussive_strength = max(
                role_strength(
                    facts.evidence.get("measured_roles", {})
                    if facts is not None and isinstance(facts.evidence, dict)
                    else {},
                    "percussive_one_shot",
                ),
                role_strength(
                    facts.evidence.get("measured_roles", {})
                    if facts is not None and isinstance(facts.evidence, dict)
                    else {},
                    "protected_percussive_one_shot",
                ),
            )
            if percussive_strength >= 0.74:
                return None
        has_cross_family_conflict = self._has_close_raw_candidate(
            raw_claim,
            fragments=("drums/", "fx/", "percussion", "reverse", "engine", "blip", "glitch"),
            margin=42.0,
        )
        if not has_cross_family_conflict:
            return None
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            reason=(
                "ambiguous Mallet/Bell one-shot consensus lacked measured source-role "
                f"confirmation: raw={raw_claim.folder_path}; shape={shape}:{shape_confidence:.2f}"
            ),
            source="ambiguous_instrument_hit_review",
            shared=raw_claim.shared_candidates,
            winner=raw_claim,
            strength=0.84,
        )

    def _percussive_one_shot_parent_firewall(
        self,
        raw_claim: ConsensusClaim,
        *,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Keep measured short percussive one-shots out of music/texture leaves.

        This is parent-family arbitration, not source-name tuning.  The voters
        may still disagree about clap versus rim versus bongo, but when the
        measured body is a short percussive hit, Instruments and broad/texture
        FX should not win unless the audio lacks any viable Drum candidate.
        """
        if raw_claim.family in {"Drums", "_TO_REVIEW"}:
            return None
        best_drum = self._best_drum_one_shot_candidate(raw_claim)
        if not self._facts_support_percussive_one_shot_firewall(raw_claim, facts, best_drum):
            return None

        raw_score = self._score_or_default(raw_claim.raw_candidate_score)
        if (
            raw_claim.family == "Instruments"
            and self._raw_is_specific_candidate(raw_claim)
            and raw_score <= 4.0
            and (best_drum is None or best_drum[0] > raw_score + 4.0)
        ):
            return None
        if best_drum is not None:
            drum_score, drum_path, brain_rank, physics_rank = best_drum
            if self._drum_candidate_can_beat_percussive_raw(raw_claim, drum_score):
                target_path = self._percussive_firewall_target_path(raw_claim, drum_path, facts=facts)
                return claim_from_folder_path(
                    folder_path=target_path,
                    source="percussive_one_shot_parent_firewall",
                    reason=(
                        "measured short percussive one-shot evidence kept a non-drum "
                        f"winner out of Instruments/FX: raw={raw_claim.folder_path}; "
                        f"drum_candidate={drum_path}"
                    ),
                    shared=raw_claim.shared_candidates,
                    raw_candidate_score=drum_score,
                    brain_rank=brain_rank,
                    physics_rank=physics_rank,
                    shared_winner=target_path,
                    can_override=True,
                    strength=0.91,
                    is_real_candidate=True,
                )

        if raw_claim.family == "Instruments":
            return review_claim(
                label="_TO_REVIEW/Measured Role Conflict",
                reason=(
                    "measured short percussive one-shot evidence blocked an Instrument "
                    f"winner without a decisive Drum candidate: raw={raw_claim.folder_path}"
                ),
                source="percussive_one_shot_parent_firewall_review",
                shared=raw_claim.shared_candidates,
                winner=raw_claim,
                strength=0.84,
            )
        if raw_claim.family == "FX" and not self._specific_fx_candidate_is_safely_ahead(raw_claim):
            return review_claim(
                label="_TO_REVIEW/Measured Role Conflict",
                reason=(
                    "measured short percussive one-shot evidence blocked broad/ambiguous "
                    f"FX certainty without a decisive Drum candidate: raw={raw_claim.folder_path}"
                ),
                source="percussive_one_shot_parent_firewall_review",
                shared=raw_claim.shared_candidates,
                winner=raw_claim,
                strength=0.82,
            )
        return None

    def _percussive_firewall_target_path(
        self,
        raw_claim: ConsensusClaim,
        drum_path: str,
        *,
        facts: SharedAudioFacts | None,
    ) -> str:
        """Return a conservative drum target for parent-family firewall moves.

        The parent firewall should be broad only when the leaf evidence is weak.
        If the product brain ensemble strongly agrees on a snare or clap leaf and
        measured shape says short percussive one-shot, keeping that leaf is safer
        than flattening the result to Generic Percussion.  This reads only
        internal voter output already stored in ``SharedAudioFacts``.
        """
        if raw_claim.family == "FX" and "kick" not in drum_path.lower():
            supported_leaf = self._strong_internal_drum_leaf_for_percussive_firewall(facts)
            if supported_leaf:
                return supported_leaf
            return "Drums/Percussion/Generic Percussion/One Shots"
        return drum_path

    @staticmethod
    def _strong_internal_drum_leaf_for_percussive_firewall(facts: SharedAudioFacts | None) -> str:
        """Return a strong snare/clap/rim leaf from internal voters, or blank.

        This is not a source-name shortcut.  It is a narrow leaf-preservation
        rule for cases where the brain ensemble already agrees on a concrete
        drum one-shot but physics temporarily pulls the raw winner toward FX.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return ""
        preferred = FamilyClaimArbiter._preferred_drum_leaf_order_from_physics(facts)
        candidates: list[tuple[int, int, float, str]] = []
        for key_order, key in enumerate(
            (
                "brain_ensemble_vote_result",
                "full_brain_vote_result",
                "core_baby_vote_result",
                "spread_baby_vote_result",
            )
        ):
            result = facts.evidence.get(key)
            if not isinstance(result, dict):
                continue
            guesses = result.get("top_guesses")
            if not isinstance(guesses, list):
                continue
            for fallback_rank, guess in enumerate(guesses[:6], start=1):
                if not isinstance(guess, dict):
                    continue
                path = str(guess.get("folder_path") or guess.get("label") or "").strip("/")
                normalized = path.lower().replace("\\", "/")
                if not normalized.startswith("drums/"):
                    continue
                if "/loops" in normalized or "drum loops" in normalized:
                    continue
                preference_index = next(
                    (idx for idx, fragment in enumerate(preferred) if fragment in normalized),
                    None,
                )
                if preference_index is None:
                    continue
                try:
                    rank = int(guess.get("rank", fallback_rank) or fallback_rank)
                except Exception:
                    rank = fallback_rank
                try:
                    score = float(guess.get("score", guess.get("ensemble_score", rank + 1.0)) or rank + 1.0)
                except Exception:
                    score = float(rank + 1.0)
                try:
                    support = float(guess.get("support", 0.0) or 0.0)
                except Exception:
                    support = 0.0
                if rank <= 3 and (score <= 1.0 or support >= 1.8):
                    candidates.append((preference_index, key_order, rank, score, path))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return candidates[0][4]

    @staticmethod
    def _preferred_drum_leaf_order_from_physics(facts: SharedAudioFacts | None) -> tuple[str, ...]:
        """Choose snare/clap/rim preference from measured physics branch evidence.

        The percussive firewall already decided the audio is a short drum-like
        hit.  When picking a concrete fallback leaf from the brain lanes, use
        PhysicsVoter's measured branch as the tie-breaker instead of always
        taking the first brain drum leaf.  This prevents a snare-branch hit from
        being flattened to clap merely because the ensemble listed clap one row
        above snare.
        """
        default = ("/snares/", "/claps snaps slaps/", "/rims and sticks/")
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return default
        result = facts.evidence.get("physics_vote_result")
        if not isinstance(result, dict):
            return default
        guesses = result.get("top_guesses")
        if not isinstance(guesses, list):
            return default
        for guess in guesses[:5]:
            if not isinstance(guess, dict):
                continue
            evidence = guess.get("evidence")
            if not isinstance(evidence, dict):
                continue
            branch = str(evidence.get("drum_branch_selected") or evidence.get("physics_layer_branch") or "")
            try:
                branch_conf = float(evidence.get("drum_branch_selected_confidence", 0.0) or 0.0)
            except Exception:
                branch_conf = 0.0
            top_family = str(evidence.get("physics_layer_top_family") or "")
            try:
                top_conf = float(evidence.get("physics_layer_top_confidence", 0.0) or 0.0)
            except Exception:
                top_conf = 0.0
            if top_family != "Drums" or top_conf < 0.70 or branch_conf < 0.38:
                continue
            if branch == "Snare":
                return ("/snares/", "/rims and sticks/", "/claps snaps slaps/")
            if branch == "RimOrStick":
                return ("/rims and sticks/", "/snares/", "/claps snaps slaps/")
            if branch == "Clap":
                return ("/claps snaps slaps/", "/snares/", "/rims and sticks/")
        return default

    def _facts_support_percussive_one_shot_firewall(
        self,
        raw_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
        best_drum: tuple[float, str, int | None, int | None] | None,
    ) -> bool:
        """Return True when measured facts describe a short percussive hit."""
        if facts is None:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        measured_role = _measured_role_from_facts(facts)
        roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(roles, dict):
            roles = {}
        percussive_strength = max(
            role_strength(roles, "percussive_one_shot"),
            role_strength(roles, "protected_percussive_one_shot"),
            role_strength(roles, "low_kick_like_hit"),
        )
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        low_event = _shape_metric_from_facts(facts, "low_event_ratio")
        non_event_tonal = _shape_metric_from_facts(facts, "non_event_tonal_ratio")
        attack = _shape_metric_from_facts(facts, "attack_rise_time_norm")

        short_shape = bool(
            shape in {"single_hit", "hit_with_tail"}
            or (shape in {"top_loop", "beat_loop", "drum_loop"} and duration > 0.0 and duration <= 0.35)
            or (shape == "bass_phrase" and best_drum is not None and low_event >= 0.88 and attack <= 0.06)
        )
        if not short_shape or shape_confidence < 0.58:
            return False
        if duration > 1.25 and not facts.is_short_hit_like:
            return False
        if event_count > 5.0 and duration > 0.35:
            return False
        if self._facts_support_true_voice_role(facts):
            return False

        role_says_percussion = bool(
            measured_role
            in {
                "percussive_one_shot",
                "protected_percussive_one_shot",
                "low_kick_like_hit",
            }
            or percussive_strength >= 0.62
        )
        physics_says_short_bright_hit = bool(high_event >= 0.72 and non_event_tonal <= 0.20 and duration <= 0.75)
        physics_says_low_hit = bool(
            shape == "bass_phrase" and low_event >= 0.88 and attack <= 0.06 and best_drum is not None
        )
        if raw_claim.family == "FX" and self._specific_fx_candidate_is_safely_ahead(raw_claim):
            return bool(
                best_drum is not None and best_drum[0] <= self._score_or_default(raw_claim.raw_candidate_score) + 12.0
            )
        return bool(role_says_percussion or physics_says_short_bright_hit or physics_says_low_hit)

    def _best_drum_one_shot_candidate(
        self,
        raw_claim: ConsensusClaim,
    ) -> tuple[float, str, int | None, int | None] | None:
        """Return the best non-loop Drums candidate from shared voter rows."""
        best: tuple[float, str, int | None, int | None] | None = None
        for row in raw_claim.shared_candidates or []:
            path = str(row.get("folder_path") or row.get("label") or "").strip("/")
            normalized = path.lower().replace("\\", "/")
            family = str(row.get("top_family") or (path.split("/", 1)[0] if path else ""))
            if family != "Drums" and not normalized.startswith("drums/"):
                continue
            if "drum loops" in normalized or "/loops" in normalized:
                continue
            score = _candidate_combined_score(row)
            candidate = (
                score,
                path or "Drums/Percussion/Generic Percussion/One Shots",
                self._int_or_none(row.get("brain_rank")),
                self._int_or_none(row.get("physics_rank")),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best

    def _drum_candidate_can_beat_percussive_raw(
        self,
        raw_claim: ConsensusClaim,
        drum_score: float,
    ) -> bool:
        """Return True when the Drum candidate is close enough to own the parent."""
        raw_score = self._score_or_default(raw_claim.raw_candidate_score)
        if raw_claim.family == "Instruments":
            return bool(drum_score <= raw_score + 32.0 or drum_score <= 42.0)
        if raw_claim.family == "FX":
            if self._specific_fx_candidate_is_safely_ahead(raw_claim):
                return drum_score <= raw_score + 12.0
            return bool(drum_score <= raw_score + 24.0 or drum_score <= 36.0)
        return False

    def _specific_fx_candidate_is_safely_ahead(self, raw_claim: ConsensusClaim) -> bool:
        """Return True for a concrete FX raw candidate that is not merely broad."""
        if raw_claim.family != "FX" or not self._raw_is_specific_candidate(raw_claim):
            return False
        raw_score = self._score_or_default(raw_claim.raw_candidate_score)
        return raw_score <= 8.0

    @staticmethod
    def _score_or_default(value: float | None) -> float:
        try:
            return float(value if value is not None else 9999.0)
        except Exception:
            return 9999.0

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return None

    def _claim_has_architecture_safe_routing_evidence(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None = None,
    ) -> bool:
        """Return True when a claim has enough evidence to route.

        This is the guardrail that removes the old rule-tree behavior from the
        refactor.  Role/shape producers may still emit claims for diagnostics,
        but a routing claim needs real voter-candidate backing.  A shortcut
        claim also cannot beat a strong specific raw candidate merely because
        its measured role/shape looks compatible.
        """
        if claim.is_review:
            return True
        if claim.family == raw_claim.family and claim.sub_family == raw_claim.sub_family:
            return True
        cross_family = claim.family != raw_claim.family
        if self._real_instrument_loop_parent_candidate_is_safe_against_fx_raw(
            raw_claim,
            claim,
            max_candidate_gap=6.00,
            minimum_strength=0.86,
        ):
            return True
        if not claim.is_real_candidate or claim.raw_candidate_score is None:
            if self._measured_synth_lead_claim_is_safe(raw_claim, claim, facts):
                return True
            if self._noncandidate_true_bucket_claim_is_safe(raw_claim, claim, facts):
                return True
            if cross_family and self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(
                raw_claim, claim, facts
            ):
                return True
            if not cross_family:
                return True
            if claim.family == "Drums":
                return False
            return self._inferred_broad_claim_is_safe_for_raw(raw_claim, claim, facts=facts)
        if raw_claim.raw_candidate_score is None:
            return True
        if self._facts_support_drum_loop_claim(claim, facts):
            return True
        if self._facts_support_kick_one_shot_claim(claim, facts):
            return True
        if self._facts_support_percussion_one_shot_claim(raw_claim, claim, facts):
            return True
        if self._facts_support_bass_loop_parent_claim(claim, facts):
            return True
        if (
            raw_claim.family == "FX"
            and claim.family == "Drums"
            and claim.sub_family == "Percussion One Shot"
            and claim.is_real_candidate
            and claim.raw_candidate_score is not None
            and raw_claim.raw_candidate_score is not None
            and claim.raw_candidate_score <= raw_claim.raw_candidate_score + self.CROSS_FAMILY_OVERRIDE_MARGIN
            and claim.strength >= 0.86
        ):
            return True
        if claim.source in self.PROFILE_SHORTCUT_SOURCES and claim.family != raw_claim.family:
            # Profile shortcut claims are old rescue behavior.  They may support
            # review or same-family broadening elsewhere, but they cannot move a
            # concrete raw winner across families.
            return False
        if self._same_family_loop_broadening_is_safe(raw_claim, claim, facts):
            return True
        if self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(raw_claim, claim, facts):
            return True
        if claim.source == "candidate_true_bucket_rescue":
            return claim.raw_candidate_score <= raw_claim.raw_candidate_score + self.CROSS_FAMILY_OVERRIDE_MARGIN
        if claim.source in self.ROLE_SHAPE_ROUTING_SOURCES and cross_family:
            if self._raw_is_specific_candidate(raw_claim):
                return self._candidate_decisively_beats_specific_raw(raw_claim, claim, margin=0.75)
        return True

    def _measured_synth_lead_claim_is_safe(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for the narrow measured Synth Lead specialist claim.

        This is the one non-candidate leaf exception allowed by the arbiter.
        It can only beat an FX/review false-positive when measured structure is
        a clean, non-percussive pitched phrase.  It cannot steal concrete
        Instrument siblings such as sax, voice, keys, or guitar.
        """
        if claim.source != "profile_candidate_measured_synth_lead_claim":
            return False
        if claim.family != "Instruments":
            return False
        if "synth lead" not in str(claim.folder_path or claim.label).lower():
            return False
        if raw_claim.family not in {"FX", "_TO_REVIEW", "Instruments"}:
            return False
        raw_path = str(raw_claim.folder_path or raw_claim.label or "").lower().replace("\\", "/")
        if raw_claim.family == "Instruments" and not ("instrument loops" in raw_path or "synth" in raw_path):
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        return bool(
            shape == "pitched_phrase"
            and shape_confidence >= 0.88
            and _shape_metric_from_facts(facts, "pitched_event_ratio") >= 0.80
            and _shape_metric_from_facts(facts, "f0_voiced_ratio") >= 0.78
            and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.08
            and _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.10
            and _shape_metric_from_facts(facts, "low_event_ratio") <= 0.06
            and _shape_metric_from_facts(facts, "mid_event_ratio") >= 0.58
            and _shape_metric_from_facts(facts, "high_event_ratio") <= 0.18
            and _shape_metric_from_facts(facts, "spectral_entropy_mean") <= 0.38
            and _shape_metric_from_facts(facts, "spectral_flatness_mean") <= 0.20
        )

    def _noncandidate_true_bucket_claim_is_safe(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for strong broad measured buckets that may beat review.

        This is deliberately narrower than old role routing: it never chooses a
        terminal leaf, and it only applies to the measured true-bucket producer
        when shape or role evidence says the raw FX leaf is structurally wrong.
        """
        if claim.is_real_candidate or claim.is_review:
            return False
        if (
            claim.source
            not in {
                "candidate_true_bucket_rescue",
                "parent_eligibility_broad_bucket",
                "role_claim_arbitration_broad_bucket",
            }
            or claim.family == raw_claim.family
        ):
            return False
        minimum_strength = 0.88
        if (
            claim.source == "parent_eligibility_broad_bucket"
            and claim.family == "Drums"
            and claim.sub_family == "Drum Loops"
        ):
            minimum_strength = 0.78
        if (
            claim.source == "parent_eligibility_broad_bucket"
            and claim.family == "Instruments"
            and claim.sub_family == "Brass Woodwinds"
        ):
            minimum_strength = 0.76
        if (
            claim.source == "parent_eligibility_broad_bucket"
            and claim.family == "Instruments"
            and claim.sub_family == "Bass Loops"
        ):
            minimum_strength = 0.78
        if (
            claim.source == "parent_eligibility_broad_bucket"
            and claim.family == "Drums"
            and claim.sub_family == "Percussion One Shot"
        ):
            minimum_strength = 0.86
        if claim.strength < minimum_strength:
            return False

        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        measured_role = _measured_role_from_facts(facts)
        roles = (
            facts.evidence.get("measured_roles", {}) if facts is not None and isinstance(facts.evidence, dict) else {}
        )
        if not isinstance(roles, dict):
            roles = {}
        if claim.family == "Drums" and claim.sub_family == "Drum Loops":
            return bool(
                (shape in {"beat_loop", "top_loop", "drum_loop"} and shape_confidence >= 0.80)
                or measured_role in {"drum_loop", "percussive_drum_loop", "bright_drum_loop", "low_rhythmic_drum_loop"}
                or self._facts_support_bass_shaped_drum_loop(facts)
            )
        if claim.family == "Drums" and claim.sub_family == "Percussion One Shot":
            percussive_hit_strength = max(
                role_strength(roles, "percussive_one_shot"),
                role_strength(roles, "protected_percussive_one_shot"),
            )
            duration = _feature_number_from_facts(facts, "duration_sec")
            event_count = max(
                _shape_metric_from_facts(facts, "onset_count"),
                _feature_number_from_facts(facts, "event_count_estimate"),
            )
            raw_path = str(raw_claim.folder_path or raw_claim.label or "").lower().replace("\\", "/")
            return bool(
                raw_claim.family == "FX"
                and ("glitch" in raw_path or "stutter" in raw_path)
                and shape in {"single_hit", "hit_with_tail"}
                and shape_confidence >= 0.74
                and percussive_hit_strength >= 0.86
                and duration <= 1.5
                and event_count <= 4.0
            )
        if claim.family == "Instruments" and claim.sub_family == "Brass Woodwinds":
            pitched_strength = max(
                role_strength(roles, "pitched_reed_or_instrument_loop"),
                role_strength(roles, "pitched_reed_or_instrument_phrase"),
                role_strength(roles, "pitched_music_loop"),
                role_strength(roles, "pitched_music_phrase"),
            )
            return bool(
                shape in {"pitched_phrase", "sustained_pad", "vocal_phrase", "bass_phrase"}
                and shape_confidence >= 0.74
                and (
                    pitched_strength >= 0.62
                    or measured_role
                    in {
                        "pitched_reed_or_instrument_loop",
                        "pitched_reed_or_instrument_phrase",
                        "pitched_music_loop",
                        "pitched_music_phrase",
                    }
                )
            )
        if claim.family == "Instruments" and claim.sub_family == "Instrument Loops":
            return bool(
                (
                    shape in {"pitched_phrase", "sustained_pad", "bass_phrase", "vocal_phrase"}
                    and shape_confidence >= 0.74
                )
                or measured_role
                in {
                    "pitched_music_loop",
                    "pitched_music_phrase",
                    "mixed_music_loop",
                    "clean_sustained_tonal_instrument_loop",
                }
            )
        if claim.family == "Instruments" and claim.sub_family == "Bass Loops":
            return bool(
                shape == "bass_phrase"
                and shape_confidence >= 0.80
                and (role_strength(roles, "bass_loop") >= 0.78 or measured_role == "bass_loop")
            )
        return False

    def _candidate_decisively_beats_specific_raw(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        *,
        margin: float,
    ) -> bool:
        """Return True only when a candidate-backed claim really beats raw."""
        if not self._raw_is_specific_candidate(raw_claim):
            return True
        if claim.raw_candidate_score is None or raw_claim.raw_candidate_score is None:
            return False
        return claim.raw_candidate_score <= raw_claim.raw_candidate_score - margin

    def _raw_is_specific_candidate(self, raw_claim: ConsensusClaim) -> bool:
        """Return True when raw is a concrete candidate, not a broad fallback."""
        return bool(
            raw_claim.is_real_candidate and not raw_claim.is_review and not self._is_generic_broad_claim(raw_claim)
        )

    def _raw_is_stable_parent_bucket(self, raw_claim: ConsensusClaim) -> bool:
        """Return True when raw is already a safe broad parent bucket.

        Blocked role/shape shortcuts should not turn a useful Drum Loop or
        Instrument Loop parent into review.  If measured evidence is strong
        enough to move that parent, the competing claim will pass normal
        arbitration instead of arriving here as a blocked shortcut.
        """
        return bool(
            raw_claim.is_real_candidate
            and not raw_claim.is_review
            and (raw_claim.family, raw_claim.sub_family)
            in {
                ("Drums", "Drum Loops"),
                ("Instruments", "Instrument Loops"),
                ("Instruments", "Bass Loops"),
            }
        )

    def _raw_generic_false_positive_may_be_replaced(
        self,
        raw_claim: ConsensusClaim,
        replacement_claim: ConsensusClaim,
    ) -> bool:
        """Allow safe measured claims to beat dangerous generic raw buckets.

        A raw Human/Voice or generic Instrument Loop result can be a broad
        false-positive bucket.  A measured Brass/Woodwinds or Bass broad claim
        is allowed to replace it when that claim passes the explicit safe
        cross-family policy, even if the normalized strength is slightly lower.
        """
        if not self._is_generic_broad_claim(raw_claim):
            return False
        if (
            replacement_claim.family == "Instruments"
            and replacement_claim.sub_family in {"Instrument Loops", "Brass Woodwinds"}
            and replacement_claim.source
            in {
                "parent_eligibility_broad_bucket",
                "candidate_true_bucket_rescue",
                "role_sanity_generic_pitched_broad_instrument_loop",
                "sustained_pitched_instrument_broad_bucket",
            }
            and replacement_claim.strength >= 0.74
        ):
            return True
        return self._inferred_cross_family_claim_is_safe(replacement_claim)

    def _choose_best_competing_claim(
        self,
        raw_claim: ConsensusClaim,
        allowed_claims: list[ConsensusClaim],
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim:
        """Choose the strongest legal claim with broad-bucket discipline.

        Generic broad buckets are useful fallbacks, but they must not swallow a
        more specific candidate-backed claim in the same family.  This is the
        key Claim-Arbiter rule that replaces branch-order rescues: broad
        evidence can keep a sound in the right parent, while concrete candidate
        evidence can still win the destination when it is close enough.
        """
        review_claims = [claim for claim in allowed_claims if claim.is_review]
        if review_claims:
            review_claims.sort(key=lambda claim: self._claim_sort_key(raw_claim, claim))
            best_review = review_claims[0]
            review_alternative = self._best_review_alternative(
                best_review,
                allowed_claims,
                raw_claim=raw_claim,
                facts=facts,
            )
            if review_alternative is None:
                return best_review
            return review_alternative

        specific_claim = self._best_specific_candidate_claim(raw_claim, allowed_claims)
        broad_claims = [claim for claim in allowed_claims if self._is_generic_broad_claim(claim)]
        if specific_claim is not None and broad_claims:
            best_broad = min(broad_claims, key=lambda claim: self._claim_sort_key(raw_claim, claim))
            if (
                best_broad.family == "Instruments"
                and best_broad.sub_family == "Instrument Loops"
                and best_broad.source
                in {
                    "instrument_sibling_conflict_parent_claim",
                    "role_sanity_generic_pitched_broad_instrument_loop",
                    "ambiguous_fx_music_loop_broad_bucket",
                    "sustained_pitched_instrument_broad_bucket",
                    "mixed_instrument_loop_role_claim",
                }
                and specific_claim.source.startswith("profile_candidate_")
                and (best_broad.source == "instrument_sibling_conflict_parent_claim" or raw_claim.family == "FX")
                and best_broad.strength >= specific_claim.strength - 0.08
                and not (
                    specific_claim.source == "profile_candidate_voice_claim"
                    and (
                        self._facts_support_true_voice_role(facts)
                        or (
                            _shape_vote_from_facts(facts) == "vocal_phrase"
                            and _shape_confidence_from_facts(facts) >= 0.88
                        )
                    )
                )
                and not (
                    specific_claim.source == "profile_candidate_sax_leaf_claim"
                    and specific_claim.strength >= best_broad.strength - 0.12
                )
                and not (
                    specific_claim.source == "profile_candidate_synth_claim"
                    and "synth" in str(specific_claim.folder_path or specific_claim.label).lower()
                    and "synth" in str(raw_claim.folder_path or raw_claim.label or raw_claim.sub_family).lower()
                    and specific_claim.strength >= best_broad.strength - 0.12
                )
                and not (
                    specific_claim.brain_rank is not None
                    and specific_claim.brain_rank <= 3
                    and "synth pad" in str(specific_claim.folder_path or specific_claim.label).lower()
                    and "synth" in str(raw_claim.folder_path or raw_claim.label or raw_claim.sub_family).lower()
                    and specific_claim.strength >= best_broad.strength - 0.02
                )
                and not (
                    specific_claim.brain_rank is not None
                    and specific_claim.brain_rank <= 2
                    and specific_claim.strength >= best_broad.strength + 0.04
                )
            ):
                return best_broad
            if (
                best_broad.family == "Instruments"
                and best_broad.sub_family == "Instrument Loops"
                and best_broad.source
                in {
                    "parent_eligibility_broad_bucket",
                    "sustained_pitched_instrument_broad_bucket",
                    "placement_depth_family_rescue_broad_bucket",
                    "ambiguous_fx_music_loop_broad_bucket",
                    "role_sanity_generic_pitched_broad_instrument_loop",
                    "mixed_instrument_loop_role_claim",
                    "shape_sanity_consensus",
                    "candidate_true_bucket_rescue",
                }
                and best_broad.strength >= 0.86
                and "one shot" in str(specific_claim.folder_path or specific_claim.label).lower()
            ):
                return best_broad
            if (
                best_broad.family == "Instruments"
                and best_broad.sub_family == "Instrument Loops"
                and specific_claim.source == "baby_recall_brass_woodwind_claim"
                and best_broad.source
                in {
                    "shape_sanity_consensus",
                    "role_sanity_generic_pitched_broad_instrument_loop",
                    "candidate_true_bucket_rescue",
                    "top_family_sanity_consensus",
                }
                and best_broad.strength >= specific_claim.strength - 0.05
            ):
                return best_broad
            if (
                specific_claim.family == best_broad.family
                and specific_claim.strength >= best_broad.strength - 0.15
                and (
                    best_broad.source != "placement_depth_family_rescue_broad_bucket"
                    or self._specific_claim_may_override_placement_depth(specific_claim)
                )
            ):
                return specific_claim
        mixed_musical_claims = [
            claim
            for claim in allowed_claims
            if claim.source == "profile_candidate_mixed_musical_loop_claim"
            and claim.family == "Instruments"
            and claim.is_real_candidate
        ]
        if mixed_musical_claims:
            mixed_musical_claims.sort(key=lambda claim: self._claim_sort_key(raw_claim, claim))
            return mixed_musical_claims[0]

        bass_claims = [
            claim
            for claim in allowed_claims
            if claim.family == "Instruments"
            and claim.sub_family == "Bass Loops"
            and claim.source in {"role_sanity_broad_bucket", "same_family_role_broad_bucket"}
            and not claim.is_real_candidate
        ]
        instrument_loop_claims = [
            claim
            for claim in allowed_claims
            if claim.family == "Instruments"
            and claim.sub_family == "Instrument Loops"
            and claim.source == "parent_eligibility_broad_bucket"
        ]
        if bass_claims and instrument_loop_claims:
            best_bass = min(bass_claims, key=lambda claim: self._claim_sort_key(raw_claim, claim))
            best_loop = min(instrument_loop_claims, key=lambda claim: self._claim_sort_key(raw_claim, claim))
            if best_loop.strength >= best_bass.strength - 0.04:
                return best_loop

        supported_drum_parent_claims = [
            claim
            for claim in allowed_claims
            if self._facts_support_drum_loop_claim(claim, facts)
            or self._facts_support_kick_one_shot_claim(claim, facts)
            or self._facts_support_percussion_one_shot_claim(raw_claim, claim, facts)
        ]
        if supported_drum_parent_claims:
            supported_drum_parent_claims.sort(key=lambda claim: self._claim_sort_key(raw_claim, claim))
            return supported_drum_parent_claims[0]

        supported_instrument_loop_claims = [
            claim
            for claim in allowed_claims
            if self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(raw_claim, claim, facts)
        ]
        if supported_instrument_loop_claims:
            supported_instrument_loop_claims.sort(key=lambda claim: self._claim_sort_key(raw_claim, claim))
            return supported_instrument_loop_claims[0]

        sibling_conflict_claims = [
            claim
            for claim in allowed_claims
            if claim.source in {"instrument_sibling_conflict_parent_claim", "mixed_instrument_loop_role_claim"}
            and claim.family == "Instruments"
            and claim.sub_family == "Instrument Loops"
            and claim.strength >= 0.90
        ]
        if sibling_conflict_claims:
            sibling_conflict_claims.sort(key=lambda claim: self._claim_sort_key(raw_claim, claim))
            return sibling_conflict_claims[0]

        placement_depth_claims = [
            claim
            for claim in allowed_claims
            if claim.source == "placement_depth_family_rescue_broad_bucket" and claim.strength >= 0.84
        ]
        if placement_depth_claims:
            placement_depth_claims.sort(key=lambda claim: self._claim_sort_key(raw_claim, claim))
            best_depth = placement_depth_claims[0]
            if (
                specific_claim is not None
                and self._specific_claim_may_override_placement_depth(specific_claim)
                and specific_claim.strength >= best_depth.strength - 0.20
            ):
                return specific_claim
            return best_depth
        allowed_claims.sort(key=lambda claim: self._claim_sort_key(raw_claim, claim))
        best_claim = allowed_claims[0]
        if best_claim.is_review:
            review_alternative = self._best_review_alternative(
                best_claim,
                allowed_claims,
                raw_claim=raw_claim,
                facts=facts,
            )
            if review_alternative is not None:
                return review_alternative
        return best_claim

    def _best_review_alternative(
        self,
        review_claim: ConsensusClaim,
        claims: list[ConsensusClaim],
        *,
        raw_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Return a strong non-review alternative to an over-eager review.

        Review remains the default for true conflicts.  A non-review claim may
        beat review only when it is strong, candidate-backed or an explicitly
        safe broad parent bucket, and close enough to the review strength.
        """
        alternatives = [
            claim
            for claim in claims
            if not claim.is_review
            and self._review_alternative_is_safe(claim, raw_claim=raw_claim, facts=facts)
            and claim.strength >= review_claim.strength - self.REVIEW_ALTERNATIVE_MARGIN
        ]
        if not alternatives:
            return None
        alternatives.sort(key=lambda claim: self._claim_sort_key(review_claim, claim))
        return alternatives[0]

    def _review_alternative_is_safe(
        self,
        claim: ConsensusClaim,
        *,
        raw_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for claims that may rescue an over-eager review.

        A broad measured role alone is not allowed to erase review.  Review is
        where unresolved role/shape contradiction belongs.  Only real voter
        candidate evidence may rescue review.
        """
        if self._measured_synth_lead_claim_is_safe(raw_claim, claim, facts):
            return True
        if self._noncandidate_true_bucket_claim_is_safe(raw_claim, claim, facts):
            return True
        if self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(raw_claim, claim, facts):
            return True
        if (
            claim.source == "profile_candidate_synth_claim"
            and raw_claim.family == "FX"
            and (
                _measured_role_from_facts(facts)
                in {
                    "pitched_reed_or_instrument_loop",
                    "pitched_reed_or_instrument_phrase",
                    "pitched_music_loop",
                    "pitched_music_phrase",
                }
                or _shape_vote_from_facts(facts) in {"vocal_phrase", "pitched_phrase"}
            )
        ):
            return False
        if not claim.is_real_candidate:
            return False
        if self._real_instrument_loop_parent_candidate_is_safe_against_fx_raw(
            raw_claim,
            claim,
            max_candidate_gap=6.00,
            minimum_strength=0.86,
        ):
            return True
        if claim.source == "candidate_true_bucket_rescue":
            if claim.raw_candidate_score is None or raw_claim.raw_candidate_score is None:
                return False
            if (
                claim.family == raw_claim.family == "FX"
                and claim.raw_candidate_score <= raw_claim.raw_candidate_score + self.CROSS_FAMILY_OVERRIDE_MARGIN
                and claim.folder_path != raw_claim.folder_path
            ):
                return True
            return bool(
                claim.raw_candidate_score <= raw_claim.raw_candidate_score + self.CROSS_FAMILY_OVERRIDE_MARGIN
                and (claim.family, claim.sub_family)
                in {
                    ("Drums", "Drum Loops"),
                    ("Drums", "Percussion One Shot"),
                    ("Instruments", "Instrument Loops"),
                    ("Instruments", "Bass Loops"),
                    ("Instruments", "Brass Woodwinds"),
                    ("FX", "Human and Voice FX"),
                }
            )
        if claim.source in self.ROLE_SHAPE_ROUTING_SOURCES or claim.source in self.PROFILE_SHORTCUT_SOURCES:
            return False
        if not self._is_generic_broad_claim(claim):
            return True
        return (claim.family, claim.sub_family) in {
            ("Drums", "Drum Loops"),
            ("Instruments", "Brass Woodwinds"),
            ("Instruments", "Instrument Loops"),
        }

    def _real_instrument_loop_parent_candidate_is_safe_against_fx_raw(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        *,
        max_candidate_gap: float,
        minimum_strength: float,
    ) -> bool:
        """Return True for voter-backed broad Instrument Loop parent claims.

        This is parent-level arbitration, not source identity promotion.  It lets
        a real shared candidate for the broad Instrument Loops bucket challenge a
        likely FX false-positive leaf only when the measured claim is strong and
        the candidate rank is still near the raw FX winner.
        """
        if raw_claim.family != "FX":
            return False
        if claim.family != "Instruments" or claim.sub_family != "Instrument Loops":
            return False
        if not claim.is_real_candidate:
            return False
        if claim.source not in {
            "parent_eligibility_broad_bucket",
            "ambiguous_fx_music_loop_broad_bucket",
            "role_sanity_generic_pitched_broad_instrument_loop",
            "shape_sanity_consensus",
            "sustained_pitched_instrument_broad_bucket",
            "candidate_true_bucket_rescue",
        }:
            return False
        if claim.strength < minimum_strength:
            return False
        if claim.raw_candidate_score is None or raw_claim.raw_candidate_score is None:
            return False
        if self._raw_concrete_fx_winner_should_be_preserved(raw_claim):
            return False
        return claim.raw_candidate_score <= raw_claim.raw_candidate_score + max_candidate_gap

    def _best_specific_candidate_claim(
        self,
        raw_claim: ConsensusClaim,
        claims: list[ConsensusClaim],
    ) -> ConsensusClaim | None:
        """Return the best real non-generic candidate claim, if one exists."""
        specific_claims = [claim for claim in claims if self._is_specific_candidate_claim(claim)]
        if not specific_claims:
            return None
        specific_claims.sort(key=lambda claim: self._claim_sort_key(raw_claim, claim))
        return specific_claims[0]

    def _is_specific_candidate_claim(self, claim: ConsensusClaim) -> bool:
        """Return True for real candidate claims that are not generic buckets."""
        return bool(
            claim.is_real_candidate
            and claim.source != "placement_depth_broad_bucket"
            and not claim.is_review
            and not self._is_generic_broad_claim(claim)
        )

    @staticmethod
    def _specific_claim_may_override_placement_depth(claim: ConsensusClaim) -> bool:
        """Return True for explicit profile/baby claims allowed to beat broad depth.

        Ordinary role/shape sanity claims can be terminal-looking artifacts of a
        family rescue.  PlacementDepth exists to stop those from becoming brittle
        terminal placements.  Explicit profile/baby claims, however, were built
        by a narrow claim producer for a known failure class such as Kick,
        Drum Loop, Bass, or Brass/Woodwinds, so they may compete with the broad
        depth fallback.
        """
        return str(claim.source).startswith(("profile_candidate_", "baby_recall_"))

    def _is_generic_broad_claim(self, claim: ConsensusClaim) -> bool:
        """Return True for broad buckets that should lose to close specifics."""
        return (claim.family, claim.sub_family) in self.GENERIC_BROAD_BUCKETS

    def _strong_consensus_blocks_identity_claim(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None = None,
    ) -> bool:
        """Return True when strong committee consensus blocks identity rescues.

        Identity rescues are tie-breakers and conflict probes.  They must not
        override a strong concrete committee decision.  Broad unresolved parent
        buckets, such as Instruments/Instrument Loops, may still accept a
        same-family leaf probe because the committee has not decided the source
        leaf yet.  Final structure invariants run after arbitration and are not
        blocked here.
        """
        if not self._claim_is_identity_rescue_or_probe(claim):
            return False
        if self._decisive_real_candidate_relaxes_identity_firewall(raw_claim, claim):
            return False
        if self._dry_wet_probe_relaxes_identity_firewall(raw_claim, claim, facts):
            return False
        consensus_family = self._strong_committee_top_family(raw_claim)
        if not consensus_family:
            return False
        if self._raw_allows_same_family_identity_probe(raw_claim, consensus_family):
            return claim.family != consensus_family
        return True

    def _calibrated_panel_blocks_decoy_claim(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Block known decoy steals only when a calibration panel is authoritative.

        The panel layer is intentionally negative authority only.  It may reject
        a competing claim that matches a known decoy pattern, but it never
        creates or chooses the target folder by itself.
        """
        if claim.is_review or not claim.can_override:
            return False
        panels = build_voter_calibration_panels(claims=[raw_claim, claim], facts=facts)
        for panel in panels:
            if panel.target == "mixed_instrument_loop":
                if self._mixed_loop_panel_blocks_specific_instrument_leaf(panel, claim):
                    return True
                continue
            if panel.authority != "safe_to_block_decoys":
                continue
            if self._source_panel_blocks_claim(panel.target, claim):
                return True
        return False

    def _mixed_loop_panel_blocks_specific_instrument_leaf(
        self,
        panel: VoterCalibrationPanel,
        claim: ConsensusClaim,
    ) -> bool:
        """Return True when mixed-loop authority blocks a too-specific leaf."""
        if panel.authority != "safe_to_broaden_only":
            return False
        path = self._norm_claim_path(claim)
        if claim.family != "Instruments":
            return False
        if "instrument loops" in path or "mixed musical" in path or "multi instrument" in path:
            return False
        if "loop" not in path:
            return False
        return any(
            token in path
            for token in (
                "sax",
                "woodwind",
                "brass",
                "keys",
                "piano",
                "synth",
                "guitar",
                "string",
                "voice",
                "vocal",
            )
        )

    def _source_panel_blocks_claim(self, target: str, claim: ConsensusClaim) -> bool:
        """Return True when an authoritative source panel rejects a decoy claim."""
        path = self._norm_claim_path(claim)
        if target == "synth_pad":
            if "synth" in path or "pad" in path:
                return False
            return any(token in path for token in ("keys", "piano", "sax", "woodwind", "voice", "vocal", "guitar"))
        if target == "sax":
            if "sax" in path or "woodwind" in path:
                return False
            return any(token in path for token in ("synth", "pad", "keys", "piano", "voice", "vocal"))
        if target == "voice":
            if "voice" in path or "vocal" in path or "spoken" in path:
                return False
            return any(token in path for token in ("sax", "woodwind", "synth", "pad", "bell", "metallic"))
        if target == "blip":
            if "blip" in path or "beep" in path or "ui" in path:
                return False
            return any(token in path for token in ("snare", "clap", "rim", "stick", "drum"))
        return False

    def _enforce_final_invariant_consensus_firewall(
        self,
        raw_claim: ConsensusClaim,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Block late final invariants from crossing strong top-family consensus.

        Final invariants may still enforce hard measured contradictions, such as
        a clear drum one-shot, drum loop, transition FX, or clean pitched
        instrument body. They may not act as a second classifier that steals a
        strong committee winner into a different top family.
        """
        if winning_claim is raw_claim:
            return winning_claim
        if winning_claim.is_review:
            return winning_claim
        if winning_claim.family == raw_claim.family:
            return winning_claim
        if not str(winning_claim.source or "").startswith("final_"):
            return winning_claim
        protected_family = self._strong_committee_top_family(raw_claim)
        if protected_family != raw_claim.family:
            return winning_claim
        if protected_family not in {"Drums", "Instruments", "Textures", "FX"}:
            return winning_claim
        if self._final_cross_family_claim_has_hard_measured_authority(
            raw_claim,
            winning_claim,
            facts,
        ):
            return winning_claim
        return raw_claim

    def _final_cross_family_claim_has_hard_measured_authority(
        self,
        raw_claim: ConsensusClaim,
        winning_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True only for final cross-family moves backed by hard facts."""
        trusted_release_sources = {
            "final_review_bass_candidate_release_invariant",
            "final_shape_review_bass_loop_invariant",
            "final_shape_review_broad_drum_loop_invariant",
            "final_shape_review_broad_instrument_loop_invariant",
            "final_shape_review_voice_phrase_invariant",
            "final_blocked_drum_loop_release_invariant",
            "final_blocked_instrument_loop_release_invariant",
            "final_pitched_hit_broad_instrument_release_invariant",
            "final_false_voice_loop_broad_instrument_invariant",
        }
        if winning_claim.source in trusted_release_sources:
            return True
        if winning_claim.family == "Drums":
            return bool(
                self._facts_support_strong_drum_one_shot(facts)
                or self._facts_support_final_drum_loop(facts, winning_claim)
                or self._facts_support_drum_loop_claim(winning_claim, facts)
            )
        if winning_claim.family == "FX":
            return bool(
                self._facts_support_measured_transition_fx(facts, winning_claim)
                or self._facts_support_measured_voiced_fx_one_shot(facts, winning_claim)
                or (
                    winning_claim.source == "final_measured_tonal_alert_siren_invariant"
                    and self._facts_have_concrete_fx_lane_agreement(facts)
                )
            )
        if winning_claim.family == "Instruments":
            return bool(
                self._facts_support_clean_tonal_instrument_phrase(facts)
                or self._facts_support_clean_bass_loop(facts, winning_claim)
                or self._facts_support_synth_loop(facts, winning_claim)
                or self._facts_support_clean_keys_loop(facts, winning_claim)
                or self._facts_support_measured_sax_loop(facts, winning_claim)
                or self._facts_support_final_voice_instrument(facts, winning_claim)
                or self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(
                    raw_claim,
                    winning_claim,
                    facts,
                )
            )
        return False

    def _decisive_real_candidate_relaxes_identity_firewall(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
    ) -> bool:
        """Allow a real candidate to compete when it decisively beats raw.

        This is deliberately narrow.  It does not release inferred parent probes,
        role-shape shortcuts, or source-name guesses.  It only prevents the
        identity firewall from blocking a concrete candidate whose own score is
        much stronger than the raw candidate.
        """
        if not claim.is_real_candidate:
            return False
        if claim.raw_candidate_score is None or raw_claim.raw_candidate_score is None:
            return False
        if not self._raw_is_specific_candidate(raw_claim):
            return False
        if claim.strength < 0.88:
            return False
        # Lower raw_candidate_score is better.  Require a real gap so this is
        # not just a close tie dressed up as a rescue.
        return bool(claim.raw_candidate_score + 6.0 <= raw_claim.raw_candidate_score)

    def _dry_wet_probe_relaxes_identity_firewall(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Allow a specialist tie-breaker when a wet/dry probe weakens raw identity.

        This does not create a new route by itself.  It only lets an already
        produced, architecture-safe claim compete when the full-file committee is
        wet/reverb-conflicted and the harmonic-core diagnostic agrees with that
        claim's broad source neighborhood.
        """
        probe = self._dry_wet_probe_from_facts(facts)
        if not probe.get("enabled"):
            return False
        if not self._raw_identity_is_wet_conflict_candidate(raw_claim):
            return False
        return self._dry_wet_probe_supports_claim_bucket(probe, claim)

    def _dry_wet_probe_blocks_wet_voice_claim(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Block Voice rescue claims when dry-core evidence points away from voice."""
        if not self._path_is_voice(claim.folder_path or claim.label or ""):
            return False
        probe = self._dry_wet_probe_from_facts(facts)
        if not probe.get("enabled"):
            return False
        if not bool(probe.get("dry_has_non_voice_instrument")):
            return False
        if (
            bool(probe.get("dry_has_voice"))
            and not bool(probe.get("dry_has_synth_or_bells"))
            and not bool(probe.get("dry_has_sax_or_reed"))
        ):
            return False
        # A real true-voice raw candidate with measured voice support should keep
        # its normal protections.  This blocker is for wet formant false positives.
        return not (
            self._facts_support_true_voice_role(facts) and self._raw_human_voice_has_true_voice_role_support(raw_claim)
        )

    @staticmethod
    def _dry_wet_probe_from_facts(facts: SharedAudioFacts | None) -> dict[str, object]:
        if facts is None or not isinstance(facts.evidence, dict):
            return {}
        probe = facts.evidence.get("dry_wet_conflict_probe")
        return probe if isinstance(probe, dict) else {}

    def _raw_identity_is_wet_conflict_candidate(self, raw_claim: ConsensusClaim) -> bool:
        """Return True for raw identities commonly distorted by wet tails."""
        path = self._norm_claim_path(raw_claim)
        if self._is_generic_broad_claim(raw_claim):
            return True
        return bool(
            self._path_is_voice(path)
            or self._path_is_strings(path)
            or self._path_is_sax_or_reed(path)
            or self._path_is_synth(path)
            or self._path_is_mallet_or_bell(path)
            or "harmonica" in path
        )

    def _dry_wet_probe_supports_claim_bucket(
        self,
        probe: dict[str, object],
        claim: ConsensusClaim,
    ) -> bool:
        """Return True when dry-core candidate buckets support this claim."""
        path = self._norm_claim_path(claim)
        buckets = set(str(x) for x in probe.get("dry_source_buckets", []) if x)
        dry_paths = [str(x).lower().replace("\\", "/") for x in probe.get("dry_top_paths", []) if x]
        if self._path_is_voice(path):
            return "voice" in buckets and not bool(probe.get("dry_has_non_voice_instrument"))
        if self._path_is_sax_or_reed(path) or claim.sub_family == "Brass Woodwinds":
            return bool(
                "sax_reed" in buckets
                or "brass_woodwind" in buckets
                or any(("sax" in p or "saxophone" in p or "woodwind" in p or "harmonica" in p) for p in dry_paths)
            )
        if self._path_is_synth(path):
            return "synth" in buckets
        if self._path_is_mallet_or_bell(path):
            return "mallet_bells" in buckets
        if self._path_is_bass(path):
            return "bass" in buckets
        if claim.family == "Instruments" and claim.sub_family == "Instrument Loops":
            return bool(probe.get("dry_has_non_voice_instrument"))
        return bool(
            claim.family == "Instruments"
            and bool(probe.get("dry_has_non_voice_instrument"))
            and not self._path_is_voice(path)
        )

    @staticmethod
    def _norm_claim_path(claim: ConsensusClaim | str) -> str:
        raw = claim if isinstance(claim, str) else claim.folder_path or claim.label or claim.sub_family or claim.family
        return str(raw or "").lower().replace("\\", "/")

    @classmethod
    def _path_is_voice(cls, path: str) -> bool:
        normalized = cls._norm_claim_path(path)
        return "voice" in normalized or "vocal" in normalized or "spoken" in normalized

    @classmethod
    def _path_is_sax_or_reed(cls, path: str) -> bool:
        normalized = cls._norm_claim_path(path)
        return any(
            token in normalized for token in ("sax", "saxophone", "harmonica", "clarinet", "bassoon", "woodwind")
        )

    @classmethod
    def _path_is_synth(cls, path: str) -> bool:
        normalized = cls._norm_claim_path(path)
        return "synth" in normalized or "/lead" in normalized or "/pad" in normalized

    @classmethod
    def _path_is_mallet_or_bell(cls, path: str) -> bool:
        normalized = cls._norm_claim_path(path)
        return any(token in normalized for token in ("mallet", "bell", "vibraphone", "marimba"))

    @classmethod
    def _path_is_strings(cls, path: str) -> bool:
        normalized = cls._norm_claim_path(path)
        return any(token in normalized for token in ("string", "violin", "cello"))

    @classmethod
    def _path_is_bass(cls, path: str) -> bool:
        normalized = cls._norm_claim_path(path)
        return "/bass" in normalized or "bass/" in normalized or "808" in normalized

    def _claim_is_identity_rescue_or_probe(self, claim: ConsensusClaim) -> bool:
        """Return True for old-style rescue/probe claims, not final invariants."""
        source = str(claim.source or "")
        if source in self.ROLE_SHAPE_ROUTING_SOURCES or source in self.PROFILE_SHORTCUT_SOURCES:
            return True
        if source.startswith(self.IDENTITY_RESCUE_SOURCE_PREFIXES):
            return True
        return any(marker in source for marker in self.IDENTITY_RESCUE_SOURCE_MARKERS)

    def _strong_committee_top_family(self, raw_claim: ConsensusClaim) -> str | None:
        """Return the top family when the committee has a strong concrete consensus."""
        if raw_claim.is_review:
            return None
        if self._raw_is_specific_candidate(raw_claim) and self._score_or_default(raw_claim.raw_candidate_score) <= 12.0:
            return raw_claim.family

        rows = raw_claim.shared_candidates or []
        family_votes: dict[str, int] = {}
        raw_score = self._score_or_default(raw_claim.raw_candidate_score)
        for row in rows:
            family = self._candidate_row_top_family(row)
            if family not in {"Drums", "Instruments", "Textures", "FX"}:
                continue
            if self._candidate_row_is_near_committee_top(row, raw_score):
                family_votes[family] = family_votes.get(family, 0) + 1
        if not family_votes:
            return None
        ranked = sorted(family_votes.items(), key=lambda item: (-item[1], item[0]))
        winner, winner_count = ranked[0]
        runner_up_count = ranked[1][1] if len(ranked) > 1 else 0
        if winner_count >= 4 and winner_count >= runner_up_count + 2:
            return winner
        if winner_count >= 3 and runner_up_count == 0:
            return winner
        return None

    def _raw_allows_same_family_identity_probe(self, raw_claim: ConsensusClaim, consensus_family: str) -> bool:
        """Return True when raw is a broad unresolved parent, not a concrete identity."""
        if raw_claim.family != consensus_family:
            return True
        if raw_claim.is_review:
            return True
        if self._is_generic_broad_claim(raw_claim):
            return True
        return bool(
            raw_claim.family == "Instruments"
            and raw_claim.sub_family in {"Instrument Loops", "Brass Woodwinds"}
            and not self._raw_is_specific_candidate(raw_claim)
        )

    @staticmethod
    def _candidate_row_top_family(row: dict[str, object]) -> str:
        """Read a candidate row top family without trusting source names."""
        family = str(row.get("top_family") or "").strip()
        if family:
            return family
        path = str(row.get("folder_path") or row.get("label") or "").replace("\\", "/").strip("/")
        return path.split("/", 1)[0] if path else ""

    def _candidate_row_is_near_committee_top(self, row: dict[str, object], raw_score: float) -> bool:
        """Return True for candidate rows that represent top committee evidence."""
        brain_rank = self._int_or_none(row.get("brain_rank"))
        physics_rank = self._int_or_none(row.get("physics_rank"))
        if brain_rank is not None and brain_rank <= 3:
            return True
        if physics_rank is not None and physics_rank <= 3:
            return True
        score = self._score_or_default(_candidate_combined_score(row))
        return score <= raw_score + 8.0

    def _claim_can_compete(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        *,
        facts: SharedAudioFacts | None = None,
    ) -> bool:
        """Return True when a claim is legally allowed to challenge raw."""
        if not claim.can_override:
            return False
        if claim.is_review:
            return True
        # Measured true-bucket corrections are not old identity shortcuts.
        # Let their own safety rules evaluate before the identity firewall.
        if self._noncandidate_true_bucket_claim_is_safe(raw_claim, claim, facts):
            return True
        if self._facts_support_drum_loop_claim(claim, facts):
            return True
        if self._facts_support_kick_one_shot_claim(claim, facts):
            return True
        if self._facts_support_percussion_one_shot_claim(raw_claim, claim, facts):
            return True
        if self._facts_support_bass_loop_parent_claim(claim, facts):
            return True
        if (
            claim.source == "profile_candidate_voice_claim"
            and claim.family == "Instruments"
            and self._facts_support_true_voice_role(facts)
            and claim.strength >= 0.90
        ):
            return True
        if (
            claim.source == "shape_vocal_true_bucket_rescue"
            and claim.family == "Instruments"
            and claim.sub_family == "Voice"
            and claim.strength >= 0.90
            and self._facts_support_true_voice_role(facts)
        ):
            return True
        if self._calibrated_panel_blocks_decoy_claim(raw_claim, claim, facts):
            return False
        if self._strong_consensus_blocks_identity_claim(raw_claim, claim, facts=facts):
            return False
        if self._dry_wet_probe_blocks_wet_voice_claim(raw_claim, claim, facts):
            return False
        if not self._claim_has_architecture_safe_routing_evidence(raw_claim, claim, facts=facts):
            return False
        if (
            claim.family == "FX"
            and claim.sub_family == "Human and Voice FX"
            and raw_claim.family == "FX"
            and raw_claim.sub_family == "Human and Voice FX"
            and not self._raw_human_voice_has_true_voice_role_support(raw_claim)
            and not self._facts_support_true_voice_role(facts)
            and claim.source
            not in {
                "shape_vocal_true_bucket_rescue",
                "shape_vocal_candidate_rescue",
                "human_voice_true_bucket_rescue",
                "vocal_true_bucket_rescue",
            }
        ):
            return False
        if (
            claim.family == "Instruments"
            and claim.sub_family == "Instrument Loops"
            and raw_claim.family == "FX"
            and claim.source
            in {
                "parent_eligibility_broad_bucket",
                "ambiguous_fx_music_loop_broad_bucket",
                "role_sanity_generic_pitched_broad_instrument_loop",
                "sustained_pitched_instrument_broad_bucket",
            }
            and self._raw_concrete_fx_winner_should_be_preserved(raw_claim)
            and not self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(
                raw_claim, claim, facts
            )
        ):
            return False
        if (
            claim.family == "Instruments"
            and claim.sub_family == "Instrument Loops"
            and raw_claim.family == "FX"
            and raw_claim.sub_family == "Human and Voice FX"
            and (
                self._raw_human_voice_has_true_voice_role_support(raw_claim)
                or self._facts_support_true_voice_role(facts)
            )
        ):
            return False
        if self._real_instrument_loop_parent_candidate_is_safe_against_fx_raw(
            raw_claim,
            claim,
            max_candidate_gap=1.50,
            minimum_strength=0.90,
        ):
            return True
        if not self._inferred_broad_claim_is_safe_for_raw(raw_claim, claim, facts=facts):
            return False
        if self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(raw_claim, claim, facts):
            return True
        if claim.family == raw_claim.family:
            return True
        if self._measured_synth_lead_claim_is_safe(raw_claim, claim, facts):
            return True
        if claim.source == "human_voice_true_bucket_rescue" and claim.strength >= 0.90:
            return True
        if (
            claim.source == "profile_candidate_voice_claim"
            and claim.family == "Instruments"
            and self._facts_support_true_voice_role(facts)
            and self._raw_fx_is_ambiguous_pitched_false_positive(raw_claim)
            and claim.raw_candidate_score is not None
            and raw_claim.raw_candidate_score is not None
            and claim.raw_candidate_score <= raw_claim.raw_candidate_score + 14.0
            and claim.strength >= 0.90
        ):
            return True
        if claim.source in {"shape_sanity_consensus", "role_sanity_consensus"} and claim.strength >= 0.85:
            return True
        if claim.source == "placement_depth_family_rescue_broad_bucket" and claim.strength >= 0.84:
            return True
        if (
            claim.family == "Drums"
            and claim.sub_family == "Drum Loops"
            and claim.source == "parent_eligibility_broad_bucket"
            and claim.strength >= 0.76
            and raw_claim.family == "FX"
        ):
            return True
        if (
            claim.family == "Instruments"
            and claim.sub_family == "Instrument Loops"
            and claim.source
            in {
                "parent_eligibility_broad_bucket",
                "ambiguous_fx_music_loop_broad_bucket",
                "role_sanity_generic_pitched_broad_instrument_loop",
            }
            and raw_claim.family == "FX"
            and claim.strength >= 0.60
            and self._raw_fx_is_ambiguous_pitched_false_positive(raw_claim)
        ):
            return True
        if claim.raw_candidate_score is None or not claim.is_real_candidate:
            return self._inferred_cross_family_claim_is_safe(claim)
        if raw_claim.raw_candidate_score is None:
            return True
        return claim.raw_candidate_score <= raw_claim.raw_candidate_score + self.CROSS_FAMILY_OVERRIDE_MARGIN

    def _inferred_broad_claim_is_safe_for_raw(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None = None,
    ) -> bool:
        """Apply raw-context safety rules to inferred broad claims.

        Broad claims without a real candidate are useful for measured-role
        corrections, but they must not overwrite an already plausible concrete
        raw family.  This method keeps Bass, Drum Loop, and Reed rescues from
        becoming new broad-bucket vacuums.
        """
        if claim.is_real_candidate or claim.is_review:
            return True
        if claim.family == "Instruments" and claim.sub_family == "Bass Loops":
            if claim.source in {
                "role_sanity_broad_bucket",
                "same_family_role_broad_bucket",
                "parent_eligibility_broad_bucket",
            }:
                return claim.strength >= 0.90
            if raw_claim.family == "FX" and raw_claim.sub_family == "Human and Voice FX":
                return claim.strength >= 0.88
            if (
                raw_claim.family == "Instruments"
                and raw_claim.sub_family == "Voice"
                and _measured_role_from_facts(facts) == "bass_loop"
                and role_strength(
                    facts.evidence.get("measured_roles", {})
                    if facts is not None and isinstance(facts.evidence, dict)
                    else {},
                    "bass_loop",
                )
                >= 0.88
                and not self._facts_support_true_voice_role(facts)
            ):
                return True
            return bool(
                raw_claim.family == "Instruments"
                and self._has_any_raw_candidate(raw_claim, fragments=("bass", "808", "sub bass"))
            )
        if claim.family == "Instruments" and claim.sub_family == "Instrument Loops":
            if self._inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(raw_claim, claim, facts):
                return True
            if (
                raw_claim.family == "FX"
                and raw_claim.sub_family == "Human and Voice FX"
                and (
                    self._raw_human_voice_has_true_voice_role_support(raw_claim)
                    or self._facts_support_true_voice_role(facts)
                )
            ):
                return False
            if (
                raw_claim.family == "FX"
                and claim.source
                in {
                    "parent_eligibility_broad_bucket",
                    "ambiguous_fx_music_loop_broad_bucket",
                    "role_sanity_generic_pitched_broad_instrument_loop",
                    "sustained_pitched_instrument_broad_bucket",
                }
                and self._raw_concrete_fx_winner_should_be_preserved(raw_claim)
            ):
                return False
            if raw_claim.family == "Instruments":
                return True
            if (
                raw_claim.family == "FX"
                and claim.source
                in {
                    "parent_eligibility_broad_bucket",
                    "ambiguous_fx_music_loop_broad_bucket",
                    "role_sanity_generic_pitched_broad_instrument_loop",
                }
                and claim.strength >= 0.60
                and self._raw_fx_is_ambiguous_pitched_false_positive(raw_claim)
            ):
                return True
            if (
                raw_claim.family == "FX"
                and raw_claim.sub_family == "Human and Voice FX"
                and claim.source
                in {
                    "role_sanity_broad_bucket",
                    "parent_eligibility_broad_bucket",
                    "top_family_sanity_broad_bucket",
                }
                and not self._raw_human_voice_has_true_voice_role_support(raw_claim)
                and not self._facts_support_true_voice_role(facts)
            ):
                return claim.strength >= 0.65
            if claim.source == "sustained_pitched_instrument_broad_bucket":
                return claim.strength >= 0.74
            if claim.source == "top_family_sanity_broad_bucket":
                return claim.strength >= 0.88
            if self._is_generic_broad_claim(raw_claim) and claim.source == "parent_eligibility_broad_bucket":
                return claim.strength >= 0.76
            return self._is_generic_broad_claim(raw_claim) and claim.strength >= 0.88
        if claim.family == "Instruments" and claim.sub_family == "Brass Woodwinds":
            if raw_claim.family == "Instruments":
                return True
            return self._is_generic_broad_claim(raw_claim) and claim.strength >= 0.76
        if claim.family == "Drums" and claim.sub_family == "Drum Loops":
            if raw_claim.family == "Drums":
                return True
            if self._has_close_raw_candidate(
                raw_claim,
                fragments=("human and voice", "voice", "vocal", "crowd"),
                margin=0.5,
            ):
                return False
        return True

    def _same_family_loop_broadening_is_safe(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when measured loop structure may broaden a one-shot leaf."""
        if claim.family != raw_claim.family:
            return False
        if claim.source != "candidate_true_bucket_rescue":
            return False
        if claim.sub_family != "Instrument Loops":
            return False
        raw_path = str(raw_claim.folder_path or raw_claim.label or "").lower().replace("\\", "/")
        if "one shot" not in raw_path and "one shots" not in raw_path:
            return False
        roles = (
            facts.evidence.get("measured_roles", {}) if facts is not None and isinstance(facts.evidence, dict) else {}
        )
        if not isinstance(roles, dict):
            roles = {}
        pitched_strength = max(
            role_strength(roles, "pitched_music_loop"),
            role_strength(roles, "pitched_music_phrase"),
            role_strength(roles, "pitched_reed_or_instrument_loop"),
            role_strength(roles, "pitched_reed_or_instrument_phrase"),
            role_strength(roles, "mixed_music_loop"),
        )
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        return bool(
            pitched_strength >= 0.74
            or (shape in {"pitched_phrase", "sustained_pad", "bass_phrase"} and shape_confidence >= 0.74)
        )

    def _facts_support_drum_loop_claim(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when a broad Drum Loops claim has measured loop support."""
        if claim.family != "Drums" or claim.sub_family != "Drum Loops":
            return False
        if claim.source not in {
            "parent_eligibility_broad_bucket",
            "candidate_true_bucket_rescue",
            "role_claim_arbitration_broad_bucket",
            "profile_candidate_drum_loop_claim",
        }:
            return False
        if claim.strength < 0.78:
            return False
        if self._facts_support_bass_shaped_drum_loop(facts):
            return True
        roles = (
            facts.evidence.get("measured_roles", {}) if facts is not None and isinstance(facts.evidence, dict) else {}
        )
        if not isinstance(roles, dict):
            roles = {}
        drum_strength = max(
            role_strength(roles, "drum_loop"),
            role_strength(roles, "low_rhythmic_drum_loop"),
            role_strength(roles, "percussive_drum_loop"),
            role_strength(roles, "bright_drum_loop"),
        )
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        return bool(
            drum_strength >= 0.70
            and shape in {"bass_phrase", "beat_loop", "top_loop", "drum_loop"}
            and shape_confidence >= 0.70
        )

    def _facts_support_bass_loop_parent_claim(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when measured bass-loop evidence can beat a false voice leaf."""
        if claim.family != "Instruments" or claim.sub_family != "Bass Loops":
            return False
        if claim.source not in {
            "parent_eligibility_broad_bucket",
            "role_sanity_broad_bucket",
            "same_family_role_broad_bucket",
            "final_measured_bass_loop_invariant",
        }:
            return False
        if claim.strength < 0.88:
            return False
        roles = (
            facts.evidence.get("measured_roles", {}) if facts is not None and isinstance(facts.evidence, dict) else {}
        )
        if not isinstance(roles, dict):
            roles = {}
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        return bool(
            shape == "bass_phrase"
            and shape_confidence >= 0.82
            and role_strength(roles, "bass_loop") >= 0.78
            and not self._facts_support_true_voice_role(facts)
        )

    def _facts_support_percussion_one_shot_claim(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when measured hit facts can beat an FX glitch one-shot."""
        if claim.family != "Drums" or claim.sub_family != "Percussion One Shot":
            return False
        if claim.source not in {"parent_eligibility_broad_bucket", "candidate_true_bucket_rescue"}:
            return False
        if claim.strength < 0.86:
            return False
        roles = (
            facts.evidence.get("measured_roles", {}) if facts is not None and isinstance(facts.evidence, dict) else {}
        )
        if not isinstance(roles, dict):
            roles = {}
        percussive_hit_strength = max(
            role_strength(roles, "percussive_one_shot"),
            role_strength(roles, "protected_percussive_one_shot"),
        )
        duration = _feature_number_from_facts(facts, "duration_sec")
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        raw_path = str(raw_claim.folder_path or raw_claim.label or "").lower().replace("\\", "/")
        return bool(
            raw_claim.family == "FX"
            and ("glitch" in raw_path or "stutter" in raw_path)
            and shape in {"single_hit", "hit_with_tail"}
            and shape_confidence >= 0.74
            and percussive_hit_strength >= 0.86
            and duration <= 1.5
            and event_count <= 4.0
        )

    def _facts_support_kick_one_shot_claim(
        self,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when a kick claim has real low-hit measurements."""
        if claim.family != "Drums" or claim.sub_family != "Kick One Shot":
            return False
        if claim.source not in {
            "profile_candidate_kick_claim",
            "baby_recall_kick_claim",
            "candidate_true_bucket_rescue",
        }:
            return False
        minimum_strength = 0.84 if claim.source == "candidate_true_bucket_rescue" else 0.90
        if not claim.is_real_candidate or claim.strength < minimum_strength:
            return False
        low_pitched_hit = _feature_number_from_facts(facts, "low_pitched_hit_raw")
        low_total = _feature_number_from_facts(facts, "low_total")
        high_total = _feature_number_from_facts(facts, "high_total")
        attack = _shape_metric_from_facts(facts, "attack_rise_time_norm")
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        duration = _feature_number_from_facts(facts, "duration_sec")
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        bass_shaped_kick = bool(
            shape == "bass_phrase"
            and shape_confidence >= 0.88
            and low_total >= 0.90
            and high_total <= 0.08
            and attack <= 0.05
            and (event_count <= 3.0 or event_count == 0.0)
            and (duration <= 1.25 or duration == 0.0)
        )
        return bool(
            bass_shaped_kick
            or (
                shape in {"single_hit", "hit_with_tail"}
                and shape_confidence >= 0.70
                and low_pitched_hit >= 0.72
                and low_total >= 0.60
                and low_total >= high_total
                and (event_count <= 2.0 or event_count == 0.0)
                and (duration <= 1.25 or duration == 0.0)
            )
        )

    @staticmethod
    def _facts_support_bass_shaped_drum_loop(facts: SharedAudioFacts | None) -> bool:
        """Return True for bass-heavy measured loops with drum-loop structure."""
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        roles = facts.evidence.get("measured_roles", {})
        if not isinstance(roles, dict):
            roles = {}
        drum_strength = max(
            role_strength(roles, "drum_loop"),
            role_strength(roles, "low_rhythmic_drum_loop"),
            role_strength(roles, "percussive_drum_loop"),
            role_strength(roles, "bright_drum_loop"),
        )
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        onset_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        onset_span = _feature_number_from_facts(facts, "onset_span_ratio")
        if onset_span <= 0.0:
            onset_span = _shape_metric_from_facts(facts, "onset_span_ratio")
        return bool(
            shape == "bass_phrase"
            and shape_confidence >= 0.74
            and onset_count >= 12.0
            and onset_span >= 0.45
            and drum_strength >= 0.40
        )

    def _facts_show_true_voice_or_vocal_one_shot(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for measured voice that specialist witnesses must not steal.

        This is intentionally stricter than general voice routing.  It protects
        obvious vocal shots/phrases from the sax and keys witnesses without
        blocking piano or sax loops that merely get a broad ``vocal_phrase``
        shape label.
        """
        if facts is None:
            return False
        voice_strength = max(
            self._measured_role_value(facts, "vocal_music_phrase"),
            self._measured_role_value(facts, "vocal_phrase"),
            self._measured_role_value(facts, "vocal_one_shot"),
            self._measured_role_value(facts, "voiced_one_shot"),
        )
        voice_identity = _feature_number_from_facts(facts, "formant_light_voice_identity")
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        percussive = self._shape_number(facts, "percussive_event_ratio")
        drumlike = self._shape_number(facts, "drumlike_frame_ratio")
        onset_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        return bool(
            (
                voice_strength >= 0.62
                or voice_identity >= 0.64
                or (shape == "vocal_phrase" and voice_strength >= 0.34 and shape_confidence >= 0.72)
                or (
                    shape in {"vocal_phrase", "hit_with_tail"}
                    and getattr(facts, "is_single_event_like", False)
                    and voice_strength >= 0.28
                )
            )
            and percussive <= 0.42
            and drumlike <= 0.42
            and onset_count <= 48.0
        )

    def _facts_support_true_voice_role(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when measured role and shape evidence support real voice."""
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
        voice_panel_score = max(
            _feature_number_from_facts(facts, "voice_score"),
            _feature_number_from_facts(facts, "human_spoken_voice_score"),
            _feature_number_from_facts(facts, "human_breath_mouth_score"),
            self._subpanel_score(facts, "voice_score"),
            self._subpanel_score(facts, "human_spoken_voice_score"),
            self._subpanel_score(facts, "human_breath_mouth_score"),
        )
        drum_hit_score = max(
            _feature_number_from_facts(facts, "drum_hit_score"),
            self._subpanel_score(facts, "drum_hit_score"),
        )
        f0_voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        layer = facts.evidence.get("physics_layer_decision")
        if isinstance(layer, dict):
            try:
                human_voice_texture = float(layer.get("instrument_human_voice_texture", 0.0) or 0.0)
            except Exception:
                human_voice_texture = 0.0
            branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
            branch_confidence_raw = layer.get("physics_layer_branch_confidence") or layer.get(
                "instrument_branch_selected_confidence"
            )
            try:
                branch_confidence = float(branch_confidence_raw or 0.0)
            except Exception:
                branch_confidence = 0.0
            if (
                branch == "Voice"
                and branch_confidence >= 0.56
                and bool(layer.get("instrument_human_voice_phrase_signal"))
                and human_voice_texture >= 0.66
                and f0_voiced >= 0.55
                and shape
                in {
                    "vocal_phrase",
                    "pitched_phrase",
                    "repeated_phrase_loop",
                    "mixed_instrument_loop",
                    "compound_musical_loop",
                    "sustained_pad",
                    "bass_phrase",
                }
            ):
                return True
        return bool(
            (
                shape in {"vocal_phrase", "vocal_one_shot", "hit_with_tail"}
                and shape_confidence >= 0.70
                and (
                    voice_strength >= 0.80 or (voice_identity >= 0.65 and f0_voiced >= 0.72 and voice_strength >= 0.55)
                )
            )
            or (
                shape == "single_hit"
                and shape_confidence >= 0.72
                and f0_voiced >= 0.70
                and voice_panel_score >= 0.70
                and drum_hit_score <= 0.55
            )
        )

    @staticmethod
    def _raw_concrete_fx_has_lane_confirmation(
        raw_claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when raw concrete FX is confirmed by multiple brain lanes.

        This protects real sirens/alarms/glitches/risers from being flattened
        into Instrument Loops just because their tones look pitched.  It still
        lets sax/synth false positives escape when only one side of the brain
        family is calling concrete FX.
        """
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        raw_path = str(raw_claim.folder_path or raw_claim.label or "").lower().replace("\\", "/")
        keys = (
            "siren",
            "alarm",
            "glitch",
            "stutter",
            "riser",
            "build",
            "reverse",
            "impact",
            "boom",
            "whoosh",
            "sweep",
        )
        raw_keys = [key for key in keys if key in raw_path]
        if not raw_keys:
            return False
        matched_lanes: set[str] = set()
        for lane_key, lane_name in (
            ("full_brain_vote_result", "full"),
            ("core_baby_vote_result", "core"),
            ("spread_baby_vote_result", "spread"),
            ("outlier_baby_vote_result", "outlier"),
        ):
            result = facts.evidence.get(lane_key)
            if not isinstance(result, dict):
                continue
            guesses = result.get("top_guesses")
            if not isinstance(guesses, list) or not guesses:
                continue
            top = guesses[0] if isinstance(guesses[0], dict) else {}
            path = str(top.get("folder_path") or top.get("label") or "").lower().replace("\\", "/")
            if path.startswith("fx/") and any(key in path for key in raw_keys):
                matched_lanes.add(lane_name)
        try:
            raw_score = float(raw_claim.raw_candidate_score if raw_claim.raw_candidate_score is not None else 9999.0)
        except (TypeError, ValueError):
            raw_score = 9999.0
        return bool(
            len(matched_lanes) >= 3
            or ("outlier" in matched_lanes and len(matched_lanes) >= 2)
            or ("full" in matched_lanes and len(matched_lanes) >= 2 and raw_score < 5.0)
        )

    def _inferred_instrument_loop_claim_is_supported_by_strong_pitched_evidence(
        self,
        raw_claim: ConsensusClaim,
        claim: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when broad Instrument Loops may beat a false FX leaf.

        This is not a return to score shaping.  The raw brain/physics rows stay
        unchanged.  The arbiter only uses measured role/shape evidence to judge
        whether an FX leaf such as Siren, Bird, Glitch, Riser, or Short Impact is
        physically incompatible with a strong sustained pitched music phrase.
        In that case the safe destination is a broad instrument-loop bucket or
        review, not a fake concrete FX folder.
        """
        if claim.family != "Instruments" or claim.sub_family != "Instrument Loops":
            return False
        if claim.source not in {
            "ambiguous_fx_music_loop_broad_bucket",
            "parent_eligibility_broad_bucket",
            "role_sanity_generic_pitched_broad_instrument_loop",
            "shape_sanity_consensus",
            "sustained_pitched_instrument_broad_bucket",
            "top_family_sanity_consensus",
            "candidate_true_bucket_rescue",
        }:
            return False
        if raw_claim.family != "FX":
            return False
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        if self._raw_concrete_fx_has_lane_confirmation(raw_claim, facts):
            return False
        if self._raw_concrete_fx_winner_is_too_strong_for_inferred_loop(raw_claim):
            return False
        roles = facts.evidence.get("measured_roles", {})
        if not isinstance(roles, dict):
            return False
        parent_role = str(
            roles.get("primary_roles", [""])[0]
            if isinstance(roles.get("primary_roles"), list) and roles.get("primary_roles")
            else ""
        )
        if not parent_role:
            parent_role = detected_parent_role_name(roles)
        role_strength_value = role_strength(roles, parent_role) if parent_role != "unknown" else 0.0
        pitched_strength = max(
            role_strength(roles, "pitched_music_loop"),
            role_strength(roles, "pitched_music_phrase"),
            role_strength(roles, "pitched_reed_or_instrument_loop"),
            role_strength(roles, "pitched_reed_or_instrument_phrase"),
            role_strength(roles, "mixed_music_loop"),
            role_strength(roles, "clean_sustained_tonal_instrument_loop"),
            role_strength_value
            if parent_role
            in {
                "pitched_music_loop",
                "pitched_music_phrase",
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
                "mixed_music_loop",
                "clean_sustained_tonal_instrument_loop",
            }
            else 0.0,
        )
        if pitched_strength < 0.74 and claim.strength < 0.86:
            return False
        shape = facts.evidence.get("shape_vote", {})
        primary_shape = ""
        shape_confidence = 0.0
        if isinstance(shape, dict):
            primary_shape = str(shape.get("primary_shape") or "")
            try:
                shape_confidence = float(shape.get("confidence", 0.0) or 0.0)
            except Exception:
                shape_confidence = 0.0
        if primary_shape in {"transition_riser", "transition_drop"} and shape_confidence >= 0.86:
            return False
        if primary_shape in {"beat_loop", "drum_loop", "top_loop"} and shape_confidence >= 0.80:
            return False
        raw_path = str(raw_claim.folder_path or raw_claim.label or "").lower().replace("\\", "/")
        false_fx_fragments = (
            "animals",
            "bird",
            "cat",
            "dog",
            "designed noise",
            "siren",
            "alarm",
            "beep",
            "glitch",
            "stutter",
            "hybrid designed",
            "short impact",
            "impact",
            "riser",
            "build",
            "reverse",
            "keys coins",
            "small objects",
            "human and voice",
            "voice",
            "vocal",
            "machine",
            "motor",
            "engine",
            "natural ambience",
            "water",
            "wind",
            "ocean",
        )
        return any(fragment in raw_path for fragment in false_fx_fragments)

    @staticmethod
    def _raw_fx_is_ambiguous_pitched_false_positive(raw_claim: ConsensusClaim) -> bool:
        """Return True for FX leaves that are often false positives for music loops."""
        if raw_claim.family != "FX":
            return False
        path = str(raw_claim.folder_path or "").lower().replace("\\", "/")
        ambiguous_fragments = (
            "human and voice",
            "voice",
            "vocal",
            "crowd",
            "keys coins",
            "coins",
            "small objects",
            "animals",
            "dog",
            "cat",
            "bird",
            "cricket",
            "siren",
            "alarm",
            "beep",
            "machines",
            "motor",
            "engine",
            "natural ambience",
            "water",
            "waves",
            "wind",
        )
        return any(fragment in path for fragment in ambiguous_fragments)

    @classmethod
    def _raw_concrete_fx_winner_should_be_preserved(cls, raw_claim: ConsensusClaim) -> bool:
        """Return True when a strong concrete FX raw winner should not be broadened.

        Generic Instrument Loops is a useful fallback for animal/coin/ambience
        false positives, but it must not wash out a concrete FX candidate such
        as Siren, Alarm, or Glitch unless an Instrument Loop candidate is close.
        This keeps the arbiter evidence-based: measured pitched structure alone
        cannot erase a strong concrete FX vote.
        """
        if raw_claim.family != "FX":
            return False
        try:
            raw_score = float(raw_claim.raw_candidate_score if raw_claim.raw_candidate_score is not None else 9999.0)
        except Exception:
            raw_score = 9999.0
        if raw_score >= 999.0:
            return False
        path = str(raw_claim.folder_path or raw_claim.label or "").lower().replace("\\", "/")
        concrete_fx_fragments = (
            "designed noise",
            "siren",
            "alarm",
            "blip",
            "glitch",
            "stutter",
            "hybrid designed",
        )
        if not any(fragment in path for fragment in concrete_fx_fragments):
            return False
        if raw_score >= 8.0:
            return False
        return not cls._has_close_raw_candidate(
            raw_claim,
            fragments=("instrument loops", "mixed musical loops"),
            margin=4.0,
        )

    @staticmethod
    def _raw_concrete_fx_winner_is_too_strong_for_inferred_loop(raw_claim: ConsensusClaim) -> bool:
        """Return True when raw concrete FX is too strong for inferred broadening.

        Strong confirmed raw FX candidates should survive.  We still allow a
        medium-confidence siren/alarm-style false positive to escape when the
        measured pitched-loop evidence is strong and the concrete FX lanes are
        not confirmed by the baby-brain family.
        """
        if not FamilyClaimArbiter._raw_concrete_fx_winner_should_be_preserved(raw_claim):
            return False
        try:
            raw_score = float(raw_claim.raw_candidate_score if raw_claim.raw_candidate_score is not None else 9999.0)
        except Exception:
            raw_score = 9999.0
        return raw_score <= 3.5

    @staticmethod
    def _has_any_raw_candidate(
        raw_claim: ConsensusClaim,
        *,
        fragments: tuple[str, ...],
    ) -> bool:
        """Return True when any raw candidate path contains a fragment."""
        for row in raw_claim.shared_candidates:
            path = str(row.get("folder_path") or row.get("label") or "").lower()
            if any(fragment in path for fragment in fragments):
                return True
        return False

    @staticmethod
    def _has_close_raw_candidate(
        raw_claim: ConsensusClaim,
        *,
        fragments: tuple[str, ...],
        margin: float,
    ) -> bool:
        """Return True when raw candidates contain a close matching path."""
        if raw_claim.raw_candidate_score is None:
            return False
        ceiling = raw_claim.raw_candidate_score + margin
        for row in raw_claim.shared_candidates:
            path = str(row.get("folder_path") or row.get("label") or "").lower()
            if not any(fragment in path for fragment in fragments):
                continue
            try:
                score = float(row.get("combined_rank_score", 9999.0))
            except (TypeError, ValueError):
                score = 9999.0
            if score <= ceiling:
                return True
        return False

    @staticmethod
    def _raw_human_voice_has_true_voice_role_support(raw_claim: ConsensusClaim) -> bool:
        """Return True when a Human/Voice raw winner has real vocal-role evidence.

        Sax, reeds, keys, and some synths can rank near Human/Voice because of
        formant-like tone color.  This helper does not use producer names.  It
        only asks whether the candidate role signature actually supports vocal
        identity strongly enough to deserve protection from a measured
        Instruments parent claim.
        """
        if raw_claim.family != "FX" or raw_claim.sub_family != "Human and Voice FX":
            return False
        voice_fragments = (
            "human and voice",
            "voice",
            "vocal",
            "vox",
            "spoken",
            "choir",
            "breath",
            "crowd",
            "mouth",
        )
        for row in raw_claim.shared_candidates or []:
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if not any(fragment in path for fragment in voice_fragments):
                continue
            evidence = row.get("brain_evidence")
            measured_role_strengths = {}
            detected_role = ""
            if isinstance(evidence, dict):
                strengths = evidence.get("measured_role_strengths")
                if isinstance(strengths, dict):
                    measured_role_strengths = strengths
                detected_role = str(evidence.get("detected_parent_role") or "")

            def measured_value(role: str, strengths: dict = measured_role_strengths) -> float:
                try:
                    return float(strengths.get(role, 0.0) or 0.0)
                except Exception:
                    return 0.0

            measured_voice_strength = max(
                measured_value("vocal_music_phrase"),
                measured_value("vocal_phrase"),
                measured_value("vocal_one_shot"),
                measured_value("voiced_one_shot"),
            )
            if measured_voice_strength >= 0.82 or (
                detected_role in {"vocal_music_phrase", "vocal_phrase", "vocal_one_shot", "voiced_one_shot"}
                and measured_voice_strength >= 0.72
            ):
                return True
            signature = row.get("candidate_role_signature")
            if not isinstance(signature, dict):
                for key in ("brain_evidence", "physics_evidence", "evidence"):
                    evidence = row.get(key)
                    if isinstance(evidence, dict) and isinstance(evidence.get("candidate_role_signature"), dict):
                        signature = evidence["candidate_role_signature"]
                        break
            if not isinstance(signature, dict):
                continue

            def value(role: str, candidate_signature: dict = signature) -> float:
                try:
                    return float(candidate_signature.get(role, 0.0) or 0.0)
                except Exception:
                    return 0.0

            voice_strength = max(
                value("vocal_music_phrase"),
                value("vocal_phrase"),
                value("vocal_one_shot"),
                value("voiced_one_shot"),
            )
            if voice_strength >= 0.70:
                return True
        return False

    @staticmethod
    def _inferred_cross_family_claim_is_safe(claim: ConsensusClaim) -> bool:
        """Return whether non-candidate evidence may cross families.

        Architecture rule: measured roles and shapes are evidence, not routing.
        A non-review claim with no real voter-candidate backing cannot move a
        file across top families.  It may still appear in reports and it may
        contribute to review, but it must not become the final folder.
        """
        return False

    def _claim_sort_key(self, raw_claim: ConsensusClaim, claim: ConsensusClaim) -> tuple[float, float, int, str]:
        """Sort stronger claims first while keeping candidate rank useful."""
        same_family_bonus = 0.05 if claim.family == raw_claim.family else 0.0
        review_bonus = 0.04 if claim.is_review else 0.0
        real_candidate_bonus = 0.03 if claim.is_real_candidate else 0.0
        placement_depth_penalty = 0.02 if claim.source.startswith("placement_depth") else 0.0
        generic_broad_penalty = 0.12 if self._is_generic_broad_claim(claim) else 0.0
        effective_strength = (
            claim.strength
            + same_family_bonus
            + review_bonus
            + real_candidate_bonus
            - placement_depth_penalty
            - generic_broad_penalty
        )
        candidate_score = claim.raw_candidate_score if claim.raw_candidate_score is not None else 9999.0
        cross_family_penalty = 1 if claim.family != raw_claim.family and not claim.is_review else 0
        return (-effective_strength, candidate_score, cross_family_penalty, claim.source)


# v31.90_real_fx_one_by_one_claim_arbiter_cleanup
