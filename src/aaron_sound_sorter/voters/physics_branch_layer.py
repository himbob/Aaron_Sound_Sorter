# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Branch physics layer."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_drum_layer import PhysicsDrumLayer
from aaron_sound_sorter.voters.physics_fx_layer import PhysicsFXRoleLayer
from aaron_sound_sorter.voters.physics_instrument_layer import PhysicsInstrumentLayer
from aaron_sound_sorter.voters.physics_layer_utils import *


class PhysicsBranchLayer:
    """Second physics layer: broad branch inside the chosen top family."""

    def __init__(
        self,
        drum_layer: PhysicsDrumLayer | None = None,
        instrument_layer: PhysicsInstrumentLayer | None = None,
        fx_layer: PhysicsFXRoleLayer | None = None,
    ) -> None:
        self.drum_layer = drum_layer or PhysicsDrumLayer()
        self.instrument_layer = instrument_layer or PhysicsInstrumentLayer()
        self.fx_layer = fx_layer or PhysicsFXRoleLayer()

    def decide(self, top_family: str, facts: SharedAudioFacts) -> tuple[str, float, dict[str, Any]]:
        if top_family == "Drums":
            branch, strength, evidence = self.drum_layer.decide(facts)
            anchor = safe_float(evidence.get("drum_anchor_strength", 0.0), 0.0)
            # Strong drum-anchor evidence should still get a branch safeguard
            # when the best branch is slightly below the normal confidence
            # floor.  Otherwise a measured snare/wire/rim hit can leave the
            # branch unresolved, and close FX texture leaves can win on raw
            # profile distance despite the top-family layer saying Drums.
            branch_floor = 0.46 if anchor >= 0.82 else 0.50
            if branch == "Snare" and anchor >= 0.40:
                branch_floor = min(branch_floor, 0.34)
            if branch == "Kick" and anchor >= 0.47 and strength >= 0.43:
                shape = shape_vote(facts)
                shape_scores = shape.get("shape_scores", []) if isinstance(shape, dict) else []
                top_shape_scores = {
                    str(item[0]): safe_float(item[1], 0.0)
                    for item in shape_scores
                    if isinstance(item, (list, tuple)) and len(item) >= 2
                }
                shape_name = str(shape.get("primary_shape", "")) if isinstance(shape, dict) else ""
                beat_loop_score = max(
                    top_shape_scores.get("beat_loop", 0.0),
                    safe_float(shape.get("confidence", 0.0), 0.0) if shape_name == "beat_loop" else 0.0,
                )
                if beat_loop_score >= 0.68:
                    branch_floor = min(branch_floor, 0.43)
            if strength < branch_floor:
                return (
                    "Unresolved",
                    strength,
                    {
                        **evidence,
                        "physics_branch_layer_source": "weak_or_conflicting_drum_branch",
                        "physics_branch_layer_selected": branch,
                        "physics_branch_layer_branch_floor": round(float(branch_floor), 6),
                    },
                )
            return (
                branch,
                strength,
                {
                    **evidence,
                    "physics_branch_layer_source": "measured_drum_branch_physics",
                    "physics_branch_layer_selected": branch,
                },
            )
        if top_family == "FX":
            branch, strength, evidence = self.fx_layer.decide(facts)
            role_strength = safe_float(evidence.get("fx_role_strength", strength), strength)
            branch_floor = 0.54 if role_strength >= 0.76 else 0.60
            if branch in {
                "SirenAlarm",
                "FormantFX",
                "RadioElectrical",
                "TextureAmbience",
                "MachineMechanical",
                "FoleyMaterial",
                "SmallObjectCluster",
                "HumanCreatureFX",
                "DesignedNoiseHybrid",
            }:
                branch_floor = min(branch_floor, 0.58)
            if strength < branch_floor:
                return (
                    "Unresolved",
                    strength,
                    {
                        **evidence,
                        "physics_branch_layer_source": "weak_or_conflicting_fx_branch",
                        "physics_branch_layer_selected": branch,
                        "physics_branch_layer_branch_floor": round(float(branch_floor), 6),
                    },
                )
            return (
                branch,
                strength,
                {
                    **evidence,
                    "physics_branch_layer_source": "measured_fx_role_branch_physics",
                    "physics_branch_layer_selected": branch,
                },
            )
        if top_family != "Instruments":
            return "Unresolved", 0.0, {"physics_branch_layer_source": "top_family_not_instruments"}
        branch, strength, evidence = self.instrument_layer.decide(facts)
        anchor = safe_float(evidence.get("instrument_anchor_strength", 0.0), 0.0)
        branch_floor = 0.46 if anchor >= 0.82 else 0.52
        if (
            branch == "Voice"
            and safe_float(evidence.get("instrument_rap_voice_texture", 0.0), 0.0) >= 0.56
            and anchor >= 0.64
        ):
            branch_floor = min(branch_floor, 0.46)
        if strength < branch_floor:
            return (
                "Unresolved",
                strength,
                {
                    **evidence,
                    "physics_branch_layer_source": "weak_or_conflicting_instrument_branch",
                    "physics_branch_layer_selected": branch,
                    "physics_branch_layer_branch_floor": round(float(branch_floor), 6),
                },
            )
        return (
            branch,
            strength,
            {
                **evidence,
                "physics_branch_layer_source": "measured_instrument_branch_physics",
                "physics_branch_layer_selected": branch,
            },
        )

    def bass_branch_strength(self, shape: dict[str, Any], roles: dict[str, Any]) -> float:
        if str(shape.get("primary_shape", "")) == "bass_phrase":
            return clamp01(max(number(shape, "confidence", 0.0), role_value(roles, "bass_loop")))
        low_event = number(shape, "low_event_ratio", 0.0)
        pitch = number(shape, "pitch_confidence", 0.0)
        pitched_event = number(shape, "pitched_event_ratio", 0.0)
        percussive = number(shape, "percussive_event_ratio", 0.0)
        drumlike = number(shape, "drumlike_frame_ratio", 0.0)
        return clamp01(
            0.40 * ramp(low_event, 0.62, 0.92)
            + 0.25 * ramp(pitch, 0.58, 0.90)
            + 0.20 * ramp(pitched_event, 0.70, 0.98)
            + 0.15 * inverse_ramp(max(percussive, drumlike), 0.04, 0.30)
        )

    def voice_branch_strength(self, shape: dict[str, Any], roles: dict[str, Any], values: dict[str, Any]) -> float:
        if str(shape.get("primary_shape", "")) != "vocal_phrase":
            return 0.0
        onset_count = number(shape, "onset_count", 0.0)
        flatness = number(shape, "spectral_flatness_mean", number(values, "spectral_flatness_mean", 0.0))
        mid_event = number(shape, "mid_event_ratio", 0.0)
        presence_event = number(shape, "high_event_ratio", 0.0)
        voice_role = role_value(roles, "vocal_music_phrase")
        dense_articulation = (
            0.38 * ramp(onset_count, 34.0, 68.0)
            + 0.28 * ramp(flatness, 0.24, 0.42)
            + 0.20 * ramp(mid_event, 0.34, 0.58)
            + 0.14 * ramp(presence_event, 0.08, 0.22)
        )
        return clamp01(max(0.52 * voice_role + 0.48 * dense_articulation, dense_articulation))

    def woodwind_branch_strength(self, shape: dict[str, Any], values: dict[str, Any]) -> float:
        shape_name = str(shape.get("primary_shape", ""))
        if shape_name not in {"vocal_phrase", "pitched_phrase", "sustained_pad"}:
            return 0.0
        harmonic = number(values, "harmonic_energy_ratio", 0.0)
        presence = number(values, "presence_ratio_2000_8000hz", 0.0)
        flatness = number(values, "spectral_flatness_mean", 0.0)
        pitch = number(shape, "pitch_confidence", number(values, "pitch_confidence", 0.0))
        pitched_event = number(shape, "pitched_event_ratio", number(values, "loop_pitched_event_ratio", 0.0))
        event_count = number(shape, "onset_count", 0.0)
        sparse_phrase = inverse_ramp(event_count, 4.0, 82.0)
        reed_tone = max(ramp(presence, 0.04, 0.36), ramp(flatness, 0.16, 0.34))
        return clamp01(
            0.28 * ramp(harmonic, 0.34, 0.70)
            + 0.24 * reed_tone
            + 0.22 * ramp(pitch, 0.62, 0.90)
            + 0.16 * ramp(pitched_event, 0.82, 1.0)
            + 0.10 * sparse_phrase
        )
