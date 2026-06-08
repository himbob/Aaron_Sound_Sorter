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


class MeasuredTransitionFxClaimProducer:
    """Emit broad transition-FX claims before final arbitration."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return a measured transition-FX claim, or no claim when support is weak."""
        claim = self._produce_claim(context)
        return [] if claim is None else [claim]

    def _produce_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        raw = context.raw
        facts = context.facts
        if not self._facts_support_measured_transition_fx(facts, raw):
            return None

        branch = self._measured_physics_fx_role_branch(facts)
        if not branch and self._parent_music_loop_release_should_block_transition(raw, facts):
            return None
        if branch:
            folder_path = self._measured_physics_fx_role_folder(branch)
            reason_tail = f"PhysicsFXRoleLayer measured {branch}"
        else:
            shape = _shape_vote_from_facts(facts)
            if shape == "transition_riser":
                folder_path = "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
                reason_tail = "ShapeVoter measured rising transition motion"
            else:
                folder_path = (
                    "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX"
                )
                reason_tail = "ShapeVoter measured falling/down transition motion"

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
        if raw.family == "Drums" and not self._facts_support_measured_transition_body(facts):
            return False

        branch = self._measured_physics_fx_role_branch(facts)
        if branch:
            if self._facts_support_true_voice_role(facts):
                return False
            if raw.family == "FX" and branch not in MOTION_BRANCHES:
                return False
            if raw.family in {"FX", "_TO_REVIEW"}:
                return True
            return self._shared_candidate_has_top_family(
                raw,
                self._measured_physics_fx_role_fragments(branch),
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
            raw,
            ("riser", "build", "drop", "downlifter", "transition", "sweep", "whoosh"),
            top_family="FX",
            max_score=18.0,
            max_brain_rank=10,
            max_physics_rank=10,
        )
        measured_transition_body = self._facts_support_measured_transition_body(facts)
        if not measured_transition_body:
            shape_only_transition_rehome = bool(
                raw.family == "FX"
                and transition_candidate
                and shape_confidence >= 0.78
                and shape in {"transition_riser", "transition_downlifter", "transition_drop"}
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

    def _facts_support_rhythmic_drum_loop_over_transition(self, facts: SharedAudioFacts) -> bool:
        """Return True when transition slope is really a rhythmic drum loop."""
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
        return not (
            self._shape_number(facts, "pitched_event_ratio") < 0.45
            and self._shape_number(facts, "sustained_tonal_frame_ratio") < 0.35
        )

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
