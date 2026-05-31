# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Top-family physics layer."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_drum_layer import PhysicsDrumLayer
from aaron_sound_sorter.voters.physics_fx_layer import PhysicsFXRoleLayer
from aaron_sound_sorter.voters.physics_instrument_layer import PhysicsInstrumentLayer
from aaron_sound_sorter.voters.physics_layer_utils import *


class PhysicsTopFamilyLayer:
    """First physics layer: Drums, Instruments, FX, or unresolved."""

    def __init__(
        self,
        drum_layer: PhysicsDrumLayer | None = None,
        instrument_layer: PhysicsInstrumentLayer | None = None,
        fx_layer: PhysicsFXRoleLayer | None = None,
    ) -> None:
        self.drum_layer = drum_layer or PhysicsDrumLayer()
        self.instrument_layer = instrument_layer or PhysicsInstrumentLayer()
        self.fx_layer = fx_layer or PhysicsFXRoleLayer()

    def decide(self, facts: SharedAudioFacts) -> tuple[str, float, dict[str, Any]]:
        shape = shape_vote(facts)
        roles = measured_roles(facts)
        values = facts.feature_values_by_name or {}
        drum_branch, drum_branch_confidence, drum_evidence = self.drum_layer.decide(facts)
        instrument_branch, instrument_branch_confidence, instrument_evidence = self.instrument_layer.decide(facts)
        drum_anchor = safe_float(drum_evidence.get("drum_anchor_strength", 0.0), 0.0)
        instrument_anchor = safe_float(instrument_evidence.get("instrument_anchor_strength", 0.0), 0.0)
        compound_strength = safe_float(instrument_evidence.get("compound_music_strength", 0.0), 0.0)
        compound_prefer_broad_loop = bool(instrument_evidence.get("compound_music_prefer_broad_loop"))
        fx_branch, fx_confidence, fx_evidence = self.fx_layer.decide(
            facts,
            drum_anchor=drum_anchor,
            drum_branch_confidence=drum_branch_confidence,
            instrument_anchor=instrument_anchor,
            instrument_branch_confidence=instrument_branch_confidence,
            compound_strength=compound_strength,
            compound_prefer_broad_loop=compound_prefer_broad_loop,
        )
        fx_allows = bool(fx_evidence.get("fx_role_allows_fx"))
        fx_action = safe_float(fx_evidence.get("fx_action_strength", 0.0), 0.0)
        fx_conflict = safe_float(fx_evidence.get("fx_role_conflict_strength", 0.0), 0.0)
        tonal_non_drum_penalty = safe_float(
            drum_evidence.get("drum_anchor_tonal_non_drum_hit_penalty", 0.0),
            0.0,
        )
        shape_name = str(shape.get("primary_shape", ""))
        shape_confidence = number(shape, "confidence", 0.0)
        shape_scores = shape.get("shape_scores", [])
        top_shape_scores = {
            str(item[0]): safe_float(item[1], 0.0)
            for item in shape_scores
            if isinstance(item, (list, tuple)) and len(item) >= 2
        }
        beat_loop_score = max(
            top_shape_scores.get("beat_loop", 0.0),
            shape_confidence if shape_name == "beat_loop" else 0.0,
        )
        loop_percussive = number(values, "loop_percussive_event_ratio", number(shape, "percussive_event_ratio", 0.0))
        event_low = number(values, "loop_mean_event_low_ratio", number(shape, "low_event_ratio", 0.0))
        drum_role = max(
            role_value(roles, "percussive_one_shot"),
            role_value(roles, "bright_drum_loop"),
            role_value(roles, "percussive_drum_loop"),
        )
        low_drum_role = role_value(roles, "low_rhythmic_drum_loop")
        instrument_role = max(
            role_value(roles, "bass_loop"),
            role_value(roles, "pitched_music_loop"),
            role_value(roles, "pitched_music_phrase"),
            role_value(roles, "vocal_music_phrase"),
            role_value(roles, "voiced_one_shot"),
        )
        parent = (
            facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(getattr(facts, "evidence", None), dict)
            else {}
        )
        allowed = parent.get("allowed_top_families", []) if isinstance(parent, dict) else []
        parent_role_name = str(parent.get("role_name") or "") if isinstance(parent, dict) else ""
        parent_protected_drum = bool(
            parent_role_name in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}
            and "Drums" in allowed
            and "Instruments" not in allowed
            and drum_anchor >= 0.50
            and drum_branch_confidence >= 0.46
        )
        if parent_protected_drum:
            return (
                "Drums",
                max(drum_anchor, drum_branch_confidence),
                {
                    **drum_evidence,
                    "physics_top_layer_source": "parent_protected_drum_measurement",
                    "physics_top_layer_parent_role": parent_role_name,
                    "physics_top_layer_drum_branch": drum_branch,
                    "physics_top_layer_drum_branch_confidence": round(float(drum_branch_confidence), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                    "physics_top_layer_instrument_role": round(float(instrument_role), 6),
                },
            )
        strong_measured_impact_fx = bool(
            fx_branch == "ImpactHit"
            and fx_confidence >= 0.77
            and safe_float(fx_evidence.get("fx_impact_envelope", 0.0), 0.0) >= 0.70
            and fx_conflict < 0.56
            and drum_anchor < 0.68
        )
        strong_measured_transition_fx = bool(
            fx_branch
            in {
                "RiserBuild",
                "DropDownlifter",
                "WhooshSweep",
                "ReverseSwell",
                "GlitchStutter",
                "BlipBeep",
            }
            and (
                fx_confidence >= 0.78
                or (
                    safe_float(fx_evidence.get("fx_transition_shape_support", 0.0), 0.0) >= 0.68
                    and fx_confidence >= 0.60
                )
            )
            and (fx_allows or bool(fx_evidence.get("fx_designed_motion_loop_exception")))
            and (
                drum_anchor < 0.76
                or (
                    bool(fx_evidence.get("fx_designed_motion_loop_exception"))
                    and shape_name
                    in {"transition_riser", "transition_drop", "transition_downlifter", "reverse_swell", "whoosh_sweep"}
                    and shape_confidence >= 0.86
                )
            )
            and not (instrument_anchor >= 0.72 and fx_action < 0.78)
        )
        if strong_measured_impact_fx or strong_measured_transition_fx:
            return (
                "FX",
                fx_confidence,
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "fx_role_layer",
                    "physics_top_layer_fx_branch": fx_branch,
                    "physics_top_layer_fx_branch_confidence": round(float(fx_confidence), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                    "physics_top_layer_instrument_anchor": round(float(instrument_anchor), 6),
                    "physics_top_layer_shape": shape_name,
                    "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                },
            )
        strong_cymbal_branch = (
            drum_branch in {"Cymbal", "Hat", "MetallicPercussion"}
            and drum_anchor >= 0.58
            and drum_branch_confidence >= 0.58
        )
        if (drum_anchor >= 0.74 and drum_branch_confidence >= 0.56) or strong_cymbal_branch:
            return (
                "Drums",
                drum_anchor,
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "drum_anchor_layer",
                    "physics_top_layer_drum_branch": drum_branch,
                    "physics_top_layer_drum_branch_confidence": round(float(drum_branch_confidence), 6),
                    "physics_top_layer_instrument_branch": instrument_branch,
                    "physics_top_layer_instrument_branch_confidence": round(float(instrument_branch_confidence), 6),
                    "physics_top_layer_shape": shape_name,
                    "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                    "physics_top_layer_instrument_role": round(float(instrument_role), 6),
                },
            )
        kick_anchored_beat_loop = bool(
            drum_branch == "Kick"
            and drum_anchor >= 0.47
            and drum_branch_confidence >= 0.43
            and beat_loop_score >= 0.68
            and loop_percussive >= 0.20
            and event_low >= 0.55
        )
        if kick_anchored_beat_loop:
            return (
                "Drums",
                max(drum_anchor, drum_branch_confidence, beat_loop_score),
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "kick_anchored_beat_loop",
                    "physics_top_layer_drum_branch": drum_branch,
                    "physics_top_layer_drum_branch_confidence": round(float(drum_branch_confidence), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                    "physics_top_layer_beat_loop_score": round(float(beat_loop_score), 6),
                    "physics_top_layer_loop_percussive": round(float(loop_percussive), 6),
                    "physics_top_layer_event_low": round(float(event_low), 6),
                    "physics_top_layer_instrument_branch": instrument_branch,
                    "physics_top_layer_instrument_branch_confidence": round(float(instrument_branch_confidence), 6),
                    "physics_top_layer_instrument_anchor": round(float(instrument_anchor), 6),
                },
            )
        drum_loop_source = safe_float(instrument_evidence.get("instrument_subpanel_drum_loop_source_score", 0.0), 0.0)
        measured_drum_loop_source = bool(
            drum_loop_source >= 0.50
            and max(
                loop_percussive,
                safe_float(drum_evidence.get("drum_loop_drumlike_frame_ratio", 0.0), 0.0),
                drum_role,
                low_drum_role,
            )
            >= 0.22
            and beat_loop_score >= 0.48
            and instrument_anchor < max(0.86, drum_anchor + 0.22)
            and fx_conflict < 0.62
        )
        if measured_drum_loop_source:
            return (
                "Drums",
                max(drum_anchor, drum_branch_confidence, drum_loop_source, beat_loop_score),
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "measured_drum_loop_source_layer",
                    "physics_top_layer_drum_branch": drum_branch,
                    "physics_top_layer_drum_branch_confidence": round(float(drum_branch_confidence), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                    "physics_top_layer_drum_loop_source_score": round(float(drum_loop_source), 6),
                    "physics_top_layer_beat_loop_score": round(float(beat_loop_score), 6),
                    "physics_top_layer_loop_percussive": round(float(loop_percussive), 6),
                    "physics_top_layer_instrument_branch": instrument_branch,
                    "physics_top_layer_instrument_branch_confidence": round(float(instrument_branch_confidence), 6),
                    "physics_top_layer_instrument_anchor": round(float(instrument_anchor), 6),
                },
            )
        if instrument_anchor >= 0.68 and instrument_anchor >= drum_anchor + 0.08:
            return (
                "Instruments",
                max(instrument_anchor, instrument_role),
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "instrument_anchor_layer",
                    "physics_top_layer_instrument_branch": instrument_branch,
                    "physics_top_layer_instrument_branch_confidence": round(float(instrument_branch_confidence), 6),
                    "physics_top_layer_instrument_anchor": round(float(instrument_anchor), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                    "physics_top_layer_instrument_role": round(float(instrument_role), 6),
                    "physics_top_layer_shape": shape_name,
                    "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                },
            )
        # Shape can still identify real music/FX, but not before a strong drum
        # anchor has had a chance to explain short struck material.
        if (
            shape_name in {"bass_phrase", "vocal_phrase", "pitched_phrase", "pitched_phrase_shape", "sustained_pad"}
            and shape_confidence >= 0.70
        ):
            return (
                "Instruments",
                shape_confidence,
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "shape_music_family",
                    "physics_top_layer_shape": shape_name,
                    "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                },
            )
        fx_branch_strength = safe_float(
            fx_evidence.get("fx_branch_selected_confidence", fx_confidence),
            fx_confidence,
        )
        strong_clean_mid_blip_tone = bool(
            fx_branch == "BlipBeep"
            and fx_allows
            and fx_branch_strength >= 0.78
            and fx_action >= 0.78
            and safe_float(fx_evidence.get("fx_clean_mid_blip_tone", 0.0), 0.0) >= 0.58
            and safe_float(fx_evidence.get("fx_blip_band_support", 0.0), 0.0) >= 0.48
            and fx_conflict < 0.42
        )
        if strong_clean_mid_blip_tone:
            return (
                "FX",
                max(fx_confidence, fx_action),
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "clean_mid_blip_fx_preempts_solo_phrase",
                    "physics_top_layer_fx_branch": fx_branch,
                    "physics_top_layer_fx_branch_confidence": round(float(fx_branch_strength), 6),
                    "physics_top_layer_fx_role_confidence": round(float(fx_confidence), 6),
                    "physics_top_layer_fx_action": round(float(fx_action), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                    "physics_top_layer_instrument_anchor": round(float(instrument_anchor), 6),
                    "physics_top_layer_shape": shape_name,
                    "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                },
            )
        clean_solo_music_shape = bool(
            shape_name == "solo_phrase"
            and shape_confidence >= 0.84
            and (instrument_role >= 0.38 or instrument_anchor >= 0.58)
            and (
                drum_anchor < 0.64
                or (
                    max(loop_percussive, number(shape, "percussive_event_ratio", 0.0)) <= 0.08
                    and number(shape, "drumlike_frame_ratio", 0.0) <= 0.08
                    and number(shape, "pitched_event_ratio", 0.0) >= 0.90
                    and number(shape, "sustained_tonal_frame_ratio", 0.0) >= 0.82
                )
            )
            and not bool(drum_evidence.get("drum_anchor_low_sub_kick_exception"))
        )
        if clean_solo_music_shape:
            return (
                "Instruments",
                max(shape_confidence, instrument_anchor, instrument_role),
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "clean_solo_music_shape_family",
                    "physics_top_layer_shape": shape_name,
                    "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                    "physics_top_layer_instrument_role": round(float(instrument_role), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                },
            )
        measured_plucked_source = bool(instrument_evidence.get("instrument_plucked_string_source_signal"))
        measured_fx_should_preempt_pluck = bool(
            fx_allows
            and fx_branch
            in {
                "BlipBeep",
                "SirenAlarm",
                "GlitchStutter",
                "WhooshSweep",
                "ReverseSwell",
                "RiserBuild",
                "DropDownlifter",
            }
            and fx_action >= 0.82
            and fx_confidence >= 0.70
            and fx_conflict < 0.42
        )
        if (
            measured_plucked_source
            and instrument_anchor >= 0.62
            and drum_anchor < 0.70
            and not measured_fx_should_preempt_pluck
        ):
            return (
                "Instruments",
                max(instrument_anchor, instrument_branch_confidence, instrument_role),
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "measured_plucked_string_source_family",
                    "physics_top_layer_instrument_branch": instrument_branch,
                    "physics_top_layer_instrument_branch_confidence": round(float(instrument_branch_confidence), 6),
                    "physics_top_layer_instrument_anchor": round(float(instrument_anchor), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                    "physics_top_layer_instrument_role": round(float(instrument_role), 6),
                },
            )
        if (
            fx_allows
            and fx_confidence >= 0.70
            and drum_anchor < (0.60 if fx_branch == "ImpactHit" else 0.72)
            and instrument_anchor < 0.68
        ):
            return (
                "FX",
                fx_confidence,
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "fx_role_layer_unopposed",
                    "physics_top_layer_fx_branch": fx_branch,
                    "physics_top_layer_fx_branch_confidence": round(float(fx_confidence), 6),
                    "physics_top_layer_shape": shape_name,
                    "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                },
            )
        if (
            shape_name
            in {
                "transition_riser",
                "transition_downlifter",
                "transition_drop",
                "impact",
                "impact_with_tail",
                "glitch_stutter",
                "ui_blip",
                "siren_alarm_tone",
                "texture_bed",
                "foley_action",
                "mechanical_motion",
                "noise_texture",
                "whoosh_sweep",
                "hybrid_fx_motion",
            }
            and shape_confidence >= 0.72
            and fx_conflict < 0.62
            and not bool(fx_evidence.get("fx_transition_loop_decoy_guard"))
            and (fx_allows or fx_action >= 0.72)
        ):
            return (
                "FX",
                max(shape_confidence, fx_confidence),
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "shape_fx_family_with_fx_role_support",
                    "physics_top_layer_shape": shape_name,
                    "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                    "physics_top_layer_fx_branch": fx_branch,
                    "physics_top_layer_fx_branch_confidence": round(float(fx_confidence), 6),
                },
            )
        drum_family_measure = max(drum_role, low_drum_role, drum_anchor)
        drum_measure_floor = 0.68
        if tonal_non_drum_penalty >= 0.34 and drum_anchor < 0.78:
            # Some vocal/tonal one-shots are falsely marked as percussive by
            # coarse role facts because they have one sharp attack.  Let the
            # measured tonal-non-drum penalty cap and raise the drum-role
            # shortcut floor so a vocal shot does not become a rimshot.
            drum_family_measure = min(drum_family_measure, drum_anchor)
            drum_measure_floor = 0.78
        if drum_family_measure >= drum_measure_floor and drum_family_measure >= instrument_role + 0.12:
            return (
                "Drums",
                drum_family_measure,
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "measured_drum_role_or_anchor",
                    "physics_top_layer_drum_role": round(float(max(drum_role, low_drum_role)), 6),
                    "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                    "physics_top_layer_tonal_non_drum_penalty": round(float(tonal_non_drum_penalty), 6),
                    "physics_top_layer_instrument_role": round(float(instrument_role), 6),
                },
            )
        if instrument_role >= 0.60 and drum_anchor < 0.68:
            return (
                "Instruments",
                max(instrument_role, instrument_anchor),
                {
                    **drum_evidence,
                    **instrument_evidence,
                    **fx_evidence,
                    "physics_top_layer_source": "measured_instrument_role",
                    "physics_top_layer_instrument_role": round(float(instrument_role), 6),
                    "physics_top_layer_drum_role": round(float(max(drum_role, low_drum_role)), 6),
                },
            )
        return (
            "Unresolved",
            0.0,
            {
                **drum_evidence,
                **instrument_evidence,
                **fx_evidence,
                "physics_top_layer_source": "weak_or_conflicting_physics",
                "physics_top_layer_shape": shape_name,
                "physics_top_layer_shape_confidence": round(float(shape_confidence), 6),
                "physics_top_layer_drum_role": round(float(max(drum_role, low_drum_role)), 6),
                "physics_top_layer_drum_anchor": round(float(drum_anchor), 6),
                "physics_top_layer_instrument_role": round(float(instrument_role), 6),
            },
        )
