# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Compatibility scorer that applies top, branch, and leaf physics layers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_branch_layer import PhysicsBranchLayer
from aaron_sound_sorter.voters.physics_category_layer import PhysicsCategoryLayer
from aaron_sound_sorter.voters.physics_drum_layer import PhysicsDrumLayer
from aaron_sound_sorter.voters.physics_fx_layer import PhysicsFXRoleLayer
from aaron_sound_sorter.voters.physics_instrument_layer import PhysicsInstrumentLayer
from aaron_sound_sorter.voters.physics_layer_types import PhysicsLayerDecision
from aaron_sound_sorter.voters.physics_layer_utils import *
from aaron_sound_sorter.voters.physics_leaf_layer import PhysicsLeafLayer
from aaron_sound_sorter.voters.physics_top_family_layer import PhysicsTopFamilyLayer


class LayeredPhysicsScorer:
    """Apply readable top/branch/leaf physics layers to candidate scores."""

    def __init__(self) -> None:
        self.drum_layer = PhysicsDrumLayer()
        self.instrument_layer = PhysicsInstrumentLayer()
        self.fx_layer = PhysicsFXRoleLayer()
        self.top_layer = PhysicsTopFamilyLayer(self.drum_layer, self.instrument_layer, self.fx_layer)
        self.branch_layer = PhysicsBranchLayer(self.drum_layer, self.instrument_layer, self.fx_layer)
        self.leaf_layer = PhysicsLeafLayer()
        self.category_layer = PhysicsCategoryLayer()

    def analyze(self, facts: SharedAudioFacts) -> PhysicsLayerDecision:
        top_family, top_confidence, top_evidence = self.top_layer.decide(facts)
        if top_family == "FX" and "fx_branch_selected" in top_evidence:
            branch = str(top_evidence.get("fx_branch_selected") or "Unresolved")
            branch_confidence = safe_float(
                top_evidence.get("fx_branch_selected_confidence", top_confidence), top_confidence
            )
            branch_evidence = {
                **top_evidence,
                "physics_branch_layer_source": "measured_fx_role_branch_physics",
                "physics_branch_layer_selected": branch,
            }
        else:
            branch, branch_confidence, branch_evidence = self.branch_layer.decide(top_family, facts)
        category_evidence = self.category_panel_evidence(facts)
        memory_evidence = self.voter_memory_evidence(facts)
        physics_memory_evidence = self.physics_memory_evidence(facts)
        base_decision = PhysicsLayerDecision(
            top_family=top_family,
            top_confidence=top_confidence,
            branch=branch,
            branch_confidence=branch_confidence,
            leaf_strategy="",
            evidence={
                **top_evidence,
                **branch_evidence,
                **category_evidence,
                **memory_evidence,
                **physics_memory_evidence,
            },
        )
        learned_decision = self.apply_physics_memory_decision(base_decision)
        return replace(learned_decision, leaf_strategy=self.leaf_layer.strategy(learned_decision))

    def category_panel_evidence(self, facts: SharedAudioFacts) -> dict[str, Any]:
        """Extract category panel evidence from SharedAudioFacts for leaf scoring."""
        evidence = getattr(facts, "evidence", {})
        if not isinstance(evidence, dict):
            return {"physics_category_panel_flat": {}, "physics_category_panel_coverage": {}}
        subpanels = evidence.get("physics_subpanels", {})
        if not isinstance(subpanels, dict):
            return {"physics_category_panel_flat": {}, "physics_category_panel_coverage": {}}
        category_panels = subpanels.get("category_panels", {})
        if not isinstance(category_panels, dict):
            return {"physics_category_panel_flat": {}, "physics_category_panel_coverage": {}}
        flat = category_panels.get("flat", {})
        coverage = category_panels.get("coverage", {})
        return {
            "physics_category_panel_flat": flat if isinstance(flat, dict) else {},
            "physics_category_panel_coverage": coverage if isinstance(coverage, dict) else {},
        }

    def voter_memory_evidence(self, facts: SharedAudioFacts) -> dict[str, Any]:
        """Extract learned voter-memory evidence from shared facts."""
        evidence = getattr(facts, "evidence", {})
        if not isinstance(evidence, dict):
            return {"learned_voter_memory": {"enabled": False}}
        memory = evidence.get("learned_voter_memory", {})
        if not isinstance(memory, dict):
            return {"learned_voter_memory": {"enabled": False}}
        return {"learned_voter_memory": memory}

    def physics_memory_evidence(self, facts: SharedAudioFacts) -> dict[str, Any]:
        """Extract learned physics-memory evidence from shared facts."""
        evidence = getattr(facts, "evidence", {})
        if not isinstance(evidence, dict):
            return {"learned_physics_memory": {"enabled": False}}
        memory = evidence.get("learned_physics_memory", {})
        if not isinstance(memory, dict):
            return {"learned_physics_memory": {"enabled": False}}
        return {"learned_physics_memory": memory}

    def apply_physics_memory_decision(self, decision: PhysicsLayerDecision) -> PhysicsLayerDecision:
        """Let trained physics memory calibrate the top/branch decision.

        The memory brain is allowed to correct weak or unresolved static
        physics decisions. It is deliberately more conservative when static
        physics has a strong contradictory top-family read, unless the match is
        an exact human-taught fingerprint.
        """
        memory = decision.evidence.get("learned_physics_memory", {})
        if not isinstance(memory, dict) or not bool(memory.get("matched")):
            return decision
        confidence = safe_float(memory.get("confidence"), 0.0)
        if confidence < 0.74:
            return decision
        target_top = str(memory.get("top_family", ""))
        target_branch = str(memory.get("branch", ""))
        match_kind = str(memory.get("match_kind", ""))
        exact_or_strong = bool(match_kind == "fingerprint" or confidence >= 0.88)
        top_can_change = bool(
            target_top in {"Drums", "Instruments", "FX"}
            and (
                target_top == decision.top_family
                or decision.top_family not in {"Drums", "Instruments", "FX"}
                or decision.top_confidence < 0.86
                or exact_or_strong
            )
        )
        top_family = target_top if top_can_change else decision.top_family
        top_confidence = float(decision.top_confidence)
        branch = str(decision.branch)
        branch_confidence = float(decision.branch_confidence)
        reasons: list[str] = []
        if top_can_change:
            if target_top != decision.top_family:
                reasons.append(f"learned_physics_memory_top:{decision.top_family}->{target_top}")
            static_top_confidence = top_confidence if target_top == decision.top_family else 0.0
            top_confidence = max(
                static_top_confidence,
                min(0.97, 0.72 + 0.23 * confidence),
            )
        if (
            target_branch
            and target_top == top_family
            and (branch == target_branch or branch == "Unresolved" or branch_confidence < 0.78 or exact_or_strong)
        ):
            if target_branch != branch:
                reasons.append(f"learned_physics_memory_branch:{branch}->{target_branch}")
            branch = target_branch
            static_branch_confidence = branch_confidence if branch == decision.branch else 0.0
            branch_confidence = max(
                static_branch_confidence,
                min(0.97, 0.70 + 0.24 * confidence),
            )
        if not reasons:
            return decision
        evidence = {
            **decision.evidence,
            "physics_memory_decision_calibration": "applied",
            "physics_memory_decision_reasons": reasons,
            "physics_memory_decision_static_top_family": decision.top_family,
            "physics_memory_decision_static_branch": decision.branch,
        }
        return replace(
            decision,
            top_family=top_family,
            top_confidence=round(float(top_confidence), 6),
            branch=branch,
            branch_confidence=round(float(branch_confidence), 6),
            evidence=evidence,
        )

    def apply(self, folder_path: str, score: float, decision: PhysicsLayerDecision) -> tuple[float, dict[str, Any]]:
        folder = normalized_path(folder_path)
        candidate_top_family = folder.split("/", 1)[0] if folder else ""
        adjusted = float(score)
        reasons: list[str] = []
        adjusted, memory_reasons, memory_evidence = self.apply_voter_memory_calibration(
            folder,
            adjusted,
            decision,
        )
        reasons.extend(memory_reasons)
        adjusted, physics_memory_reasons, physics_memory_evidence = self.apply_physics_memory_calibration(
            folder,
            adjusted,
            decision,
        )
        reasons.extend(physics_memory_reasons)
        top_family_gate = (
            0.68 if decision.top_family == "Drums" else (0.74 if decision.top_family == "Instruments" else 0.78)
        )
        if decision.top_family in {"Drums", "Instruments", "FX"} and decision.top_confidence >= top_family_gate:
            if candidate_top_family != decision.top_family:
                penalty = 0.52 if decision.top_family == "Instruments" else 0.40
                if decision.top_family == "Instruments":
                    source = str(decision.evidence.get("physics_top_layer_source", ""))
                    if source == "clean_solo_music_shape_family" and decision.top_confidence >= 0.92:
                        penalty = 2.25
                    elif (
                        decision.branch == "MixedInstrument"
                        and bool(decision.evidence.get("compound_music_prefer_broad_loop"))
                        and decision.top_confidence >= 0.80
                    ):
                        penalty = 1.20
                    elif decision.top_confidence >= 0.90:
                        penalty = 0.95
                    elif decision.top_confidence >= 0.82:
                        penalty = 0.78
                if decision.top_family == "Drums":
                    # Strong measured drum anchors are more trustworthy than a
                    # close non-drum residual match.  This is still score shaping,
                    # not hard routing: true FX can win when the drum anchor is
                    # weak, but Ocean/Animal/Voice/Blip leaves should not beat a
                    # short struck drum event after the top physics layer says
                    # Drums.
                    if decision.top_confidence >= 0.90:
                        penalty = 0.80
                    elif decision.top_confidence >= 0.82:
                        penalty = 0.70
                    else:
                        penalty = 0.50
                adjusted += penalty
                reasons.append(f"top_family_mismatch:+{penalty:.2f}")
        if (
            candidate_top_family == "FX"
            and decision.top_family != "FX"
            and not bool(decision.evidence.get("fx_role_allows_fx"))
        ):
            fx_shape = str(decision.evidence.get("fx_shape_primary") or "")
            fx_conflict = safe_float(decision.evidence.get("fx_role_conflict_strength", 0.0), 0.0)
            non_fx_shape = fx_shape in {
                "beat_loop",
                "top_loop",
                "bass_phrase",
                "vocal_phrase",
                "pitched_phrase",
                "sustained_pad",
                "solo_phrase",
                "repeated_phrase_loop",
                "mixed_instrument_loop",
            }
            non_fx_family = decision.top_family in {"Drums", "Instruments"}
            if non_fx_shape or non_fx_family or fx_conflict >= 0.48:
                penalty = 0.62
                if non_fx_shape:
                    penalty = 0.92
                if fx_conflict >= 0.58:
                    penalty = max(penalty, 0.84)
                if bool(decision.evidence.get("fx_transition_loop_decoy_guard")):
                    penalty = max(penalty, 1.25)
                adjusted += penalty
                reasons.append(f"fx_role_rejected_by_measurements:+{penalty:.2f}")
        if decision.top_family == "Drums" and decision.branch in set(PhysicsDrumLayer.BRANCHES):
            adjusted, branch_reasons = self.apply_drum_branch(folder, adjusted, decision)
            reasons.extend(branch_reasons)
        if decision.top_family == "Instruments" and decision.branch in set(PhysicsInstrumentLayer.BRANCHES):
            adjusted, branch_reasons = self.apply_instrument_branch(folder, adjusted, decision)
            reasons.extend(branch_reasons)
        if decision.top_family == "FX" and decision.branch in set(PhysicsFXRoleLayer.BRANCHES):
            adjusted, branch_reasons = self.apply_fx_branch(folder, adjusted, decision)
            reasons.extend(branch_reasons)
        adjusted, category_reasons, category_evidence = self.category_layer.apply(folder, adjusted, decision)
        reasons.extend(category_reasons)
        return max(0.0, adjusted), {
            **category_evidence,
            **memory_evidence,
            **physics_memory_evidence,
            **decision.evidence,
            "physics_layer_top_family": decision.top_family,
            "physics_layer_top_confidence": round(float(decision.top_confidence), 6),
            "physics_layer_branch": decision.branch,
            "physics_layer_branch_confidence": round(float(decision.branch_confidence), 6),
            "physics_layer_leaf_strategy": decision.leaf_strategy,
            "physics_layer_score_before": round(float(score), 6),
            "physics_layer_score_after": round(float(adjusted), 6),
            "physics_layer_adjustment_reasons": reasons,
        }

    def apply_voter_memory_calibration(
        self,
        folder: str,
        score: float,
        decision: PhysicsLayerDecision,
    ) -> tuple[float, list[str], dict[str, Any]]:
        """Blend learned role memory into physics scoring without routing.

        The learned voter-memory lane is a calibration witness.  It can make
        same-family candidates slightly easier to rank and make cross-family
        candidates slightly harder when the fingerprint is very close to a
        human-taught role.  It cannot force a folder or bypass eligibility.
        """
        memory = decision.evidence.get("learned_voter_memory", {})
        if not isinstance(memory, dict) or not bool(memory.get("matched")):
            return score, [], {"voter_memory_physics_calibration": "not_matched"}
        confidence = safe_float(memory.get("confidence"), 0.0)
        if confidence < 0.72:
            return score, [], {"voter_memory_physics_calibration": "below_confidence_gate"}
        target_top = str(memory.get("top_family", ""))
        target_label = normalized_path(str(memory.get("label", "")))
        role = str(memory.get("role", ""))
        candidate_top = folder.split("/", 1)[0] if folder else ""
        adjusted = float(score)
        reasons: list[str] = []
        if target_top and candidate_top == target_top:
            pull = 0.10 + 0.22 * min(1.0, confidence)
            if target_label and folder == target_label and confidence >= 0.78:
                pull += 0.18
            adjusted = max(0.0, adjusted - pull)
            reasons.append(f"learned_voter_memory_same_family:{role}:-{pull:.2f}")
        elif target_top and confidence >= 0.86 and target_top in {"Drums", "Instruments", "FX"}:
            penalty = 0.10 + 0.24 * min(1.0, confidence)
            adjusted += penalty
            reasons.append(f"learned_voter_memory_family_conflict:{role}:+{penalty:.2f}")
        return (
            adjusted,
            reasons,
            {
                "voter_memory_physics_calibration": "applied" if reasons else "no_family_target",
                "voter_memory_physics_score_before": round(float(score), 6),
                "voter_memory_physics_score_after": round(float(adjusted), 6),
                "voter_memory_physics_role": role,
                "voter_memory_physics_confidence": round(float(confidence), 6),
            },
        )

    def apply_physics_memory_calibration(
        self,
        folder: str,
        score: float,
        decision: PhysicsLayerDecision,
    ) -> tuple[float, list[str], dict[str, Any]]:
        """Blend learned PhysicsVoter memory into candidate scoring."""
        memory = decision.evidence.get("learned_physics_memory", {})
        if not isinstance(memory, dict) or not bool(memory.get("matched")):
            return score, [], {"physics_memory_score_calibration": "not_matched"}
        confidence = safe_float(memory.get("confidence"), 0.0)
        if confidence < 0.70:
            return score, [], {"physics_memory_score_calibration": "below_confidence_gate"}
        target_top = str(memory.get("top_family", ""))
        target_branch = str(memory.get("branch", ""))
        target_label = normalized_path(str(memory.get("label", "")))
        match_kind = str(memory.get("match_kind", ""))
        candidate_top = folder.split("/", 1)[0] if folder else ""
        adjusted = float(score)
        reasons: list[str] = []
        if target_label and folder == target_label:
            pull = 0.24 + 0.42 * min(1.0, confidence)
            adjusted = max(0.0, adjusted - pull)
            if match_kind == "fingerprint" or confidence >= 0.92:
                teacher_score = max(0.015, 0.16 * (1.0 - min(1.0, confidence)))
                adjusted = min(adjusted, teacher_score)
                reasons.append(f"learned_physics_memory_exact_teacher:{target_branch}:={teacher_score:.3f}")
            else:
                reasons.append(f"learned_physics_memory_exact_label:{target_branch}:-{pull:.2f}")
        elif target_top and candidate_top == target_top:
            branch_matches = self.candidate_matches_physics_memory_branch(folder, target_top, target_branch)
            pull = 0.12 + 0.22 * min(1.0, confidence)
            if branch_matches:
                pull += 0.14
            adjusted = max(0.0, adjusted - pull)
            reason_name = "branch" if branch_matches else "same_family"
            reasons.append(f"learned_physics_memory_{reason_name}:{target_branch}:-{pull:.2f}")
        elif target_top and confidence >= 0.78 and target_top in {"Drums", "Instruments", "FX"}:
            penalty = 0.12 + 0.28 * min(1.0, confidence)
            adjusted += penalty
            reasons.append(f"learned_physics_memory_family_conflict:{target_branch}:+{penalty:.2f}")
        return (
            adjusted,
            reasons,
            {
                "physics_memory_score_calibration": "applied" if reasons else "no_family_target",
                "physics_memory_score_before": round(float(score), 6),
                "physics_memory_score_after": round(float(adjusted), 6),
                "physics_memory_score_branch": target_branch,
                "physics_memory_score_confidence": round(float(confidence), 6),
            },
        )

    def candidate_matches_physics_memory_branch(self, folder: str, top_family: str, branch: str) -> bool:
        """Return whether a candidate label matches the learned branch."""
        if top_family == "Drums":
            return drum_candidate_matches_branch(folder, branch)
        if top_family == "Instruments":
            return instrument_candidate_matches_branch(folder, branch)
        if top_family == "FX":
            return fx_candidate_matches_branch(folder, branch)
        return False

    def apply_instrument_branch(
        self, folder: str, score: float, decision: PhysicsLayerDecision
    ) -> tuple[float, list[str]]:
        branch = decision.branch
        confidence = float(decision.branch_confidence)
        if confidence < 0.50:
            return score, []
        adjusted = float(score)
        reasons: list[str] = []
        if not folder.startswith("Instruments/"):
            if confidence >= 0.72:
                if branch == "MixedInstrument" and bool(decision.evidence.get("compound_music_prefer_broad_loop")):
                    penalty = 6.00
                else:
                    penalty = 1.00 if branch == "MixedInstrument" else 0.55
                adjusted += penalty
                if branch == "MixedInstrument":
                    reasons.append(f"mixed_compound_branch_blocks_non_instrument:+{penalty:.2f}")
                elif branch == "Bass":
                    reasons.append("bass_branch_blocks_non_instrument:+0.55")
                else:
                    reasons.append("instrument_branch_blocks_non_instrument:+0.55")
            return adjusted, reasons
        compatible = instrument_candidate_matches_branch(folder, branch)
        broad = candidate_is_broad_instrument(folder)
        if compatible:
            target = max(0.12, 0.70 - 0.50 * min(1.0, confidence))
            if branch == "Bass":
                target = max(0.10, 0.68 - 0.52 * min(1.0, confidence))
            elif branch in {"Woodwinds", "KeysPiano", "Voice"}:
                target = max(0.12, 0.66 - 0.50 * min(1.0, confidence))
            elif branch == "Synth":
                target = max(0.12, 0.62 - 0.55 * min(1.0, confidence))
            elif branch == "MixedInstrument":
                target = max(0.14, 0.70 - 0.55 * min(1.0, confidence))
            subpanel = str(decision.evidence.get(f"instrument_{branch}_subpanel_selected") or "")
            sub_confidence = safe_float(decision.evidence.get(f"instrument_{branch}_subpanel_confidence"), 0.0)
            sub_margin = safe_float(decision.evidence.get(f"instrument_{branch}_subpanel_margin"), 0.0)
            subpanel_authoritative = bool(subpanel and sub_confidence >= 0.76 and sub_margin >= 0.10)
            if subpanel_authoritative and instrument_candidate_matches_subpanel(folder, branch, subpanel):
                target = max(0.10, target - 0.04)
                reasons.append(f"instrument_{branch}_{subpanel}_subpanel_target:{target:.3f}")
            elif subpanel_authoritative and instrument_candidate_has_specific_subbranch(folder, branch):
                adjusted += 0.18
                reasons.append(f"instrument_{branch}_{subpanel}_subpanel_blocks_neighbor:+0.18")
            adjusted = min(adjusted, target)
            if branch == "Bass":
                reasons.append(f"bass_branch_target:{target:.3f}")
            else:
                reasons.append(f"instrument_{branch}_branch_target:{target:.3f}")
        elif broad:
            target = max(0.24, 0.80 - 0.32 * min(1.0, confidence))
            adjusted = min(adjusted, target)
            reasons.append(f"instrument_{branch}_broad_bucket_target:{target:.3f}")
        elif confidence >= 0.70 or (
            branch == "PluckedString"
            and confidence >= 0.64
            and bool(decision.evidence.get("instrument_plucked_string_source_signal"))
        ):
            if branch == "Voice" and instrument_candidate_matches_branch(folder, "Woodwinds"):
                penalty = 0.95
                adjusted += penalty
                reasons.append("dense_voice_branch_blocks_sax:+0.95")
            elif branch == "MixedInstrument":
                penalty = 0.55 if confidence >= 0.86 else 0.42
                adjusted += penalty
                reasons.append(f"mixed_compound_branch_blocks_specific_leaf:+{penalty:.2f}")
            else:
                penalty = 0.30 if confidence >= 0.82 else 0.22
                adjusted += penalty
                reasons.append(f"instrument_branch_mismatch:+{penalty:.2f}")
        return adjusted, reasons

    def apply_fx_branch(self, folder: str, score: float, decision: PhysicsLayerDecision) -> tuple[float, list[str]]:
        branch = decision.branch
        confidence = float(decision.branch_confidence)
        if confidence < 0.54:
            return score, []
        adjusted = float(score)
        reasons: list[str] = []
        if not folder.startswith("FX/"):
            return adjusted, reasons
        compatible = fx_candidate_matches_branch(folder, branch)
        broad_fx = candidate_is_broad_fx(folder)
        human_voice_signal = max(
            safe_float(decision.evidence.get("instrument_subpanel_voice_score"), 0.0),
            safe_float(decision.evidence.get("instrument_subpanel_human_spoken_voice_score"), 0.0),
            safe_float(decision.evidence.get("instrument_subpanel_human_breath_mouth_score"), 0.0),
            safe_float(decision.evidence.get("instrument_human_voice_texture"), 0.0),
            safe_float(decision.evidence.get("fx_formant_motion"), 0.0),
        )
        if branch == "HumanCreatureFX" and folder.startswith("FX/Animals") and human_voice_signal >= 0.62:
            penalty = 0.42
            adjusted += penalty
            reasons.append(f"human_voice_signal_blocks_animal_fx_leaf:+{penalty:.2f}")
            return adjusted, reasons
        if compatible:
            target = max(0.12, 0.66 - 0.46 * min(1.0, confidence))
            if branch == "ImpactHit":
                target = max(0.10, 0.62 - 0.46 * min(1.0, confidence))
            elif branch in {"RiserBuild", "DropDownlifter", "WhooshSweep", "ReverseSwell"}:
                target = max(0.11, 0.64 - 0.48 * min(1.0, confidence))
            elif branch in {"GlitchStutter", "BlipBeep", "SirenAlarm"}:
                target = max(0.12, 0.60 - 0.45 * min(1.0, confidence))
            elif branch in {
                "TextureAmbience",
                "MachineMechanical",
                "FoleyMaterial",
                "SmallObjectCluster",
                "HumanCreatureFX",
            }:
                target = max(0.16, 0.70 - 0.40 * min(1.0, confidence))
            adjusted = min(adjusted, target)
            reasons.append(f"fx_{branch}_branch_target:{target:.3f}")
        elif broad_fx:
            target = max(0.24, 0.76 - 0.30 * min(1.0, confidence))
            adjusted = min(adjusted, target)
            reasons.append(f"fx_{branch}_broad_bucket_target:{target:.3f}")
        elif confidence >= 0.76:
            penalty = 0.16 if branch in {"DesignedNoiseHybrid", "RadioElectrical"} else 0.22
            adjusted += penalty
            reasons.append(f"fx_branch_mismatch:+{penalty:.2f}")
        return adjusted, reasons

    def apply_drum_branch(self, folder: str, score: float, decision: PhysicsLayerDecision) -> tuple[float, list[str]]:
        branch = decision.branch
        confidence = float(decision.branch_confidence)
        kick_anchored_loop = bool(decision.evidence.get("physics_top_layer_source") == "kick_anchored_beat_loop")
        confidence_floor = 0.43 if kick_anchored_loop and branch == "Kick" else (0.34 if branch == "Snare" else 0.54)
        if confidence < confidence_floor:
            return score, []
        adjusted = float(score)
        reasons: list[str] = []
        if not folder.startswith("Drums/"):
            return adjusted, reasons
        compatible = drum_candidate_matches_branch(folder, branch)
        broad_drum = candidate_is_broad_drum(folder)
        if kick_anchored_loop and branch == "Kick":
            if "/drum loops/" in folder.lower():
                target = max(0.12, 0.68 - 0.50 * min(1.0, confidence))
                adjusted = min(adjusted, target)
                reasons.append(f"kick_anchored_loop_drum_loop_target:{target:.3f}")
                return adjusted, reasons
            if compatible:
                penalty = 0.86
                adjusted += penalty
                reasons.append(f"kick_anchored_loop_blocks_kick_one_shot:+{penalty:.2f}")
                return adjusted, reasons
        if compatible:
            target = max(0.10, 0.62 - 0.42 * min(1.0, confidence))
            if branch == "Kick":
                target = max(0.08, 0.58 - 0.44 * min(1.0, confidence))
            elif branch == "Snare":
                target = max(0.16, 0.56 - 0.60 * min(1.0, confidence))
            elif branch in {"Hat", "Cymbal"}:
                target = max(0.12, 0.66 - 0.42 * min(1.0, confidence))
            adjusted = min(adjusted, target)
            reasons.append(f"drum_{branch}_branch_target:{target:.3f}")
        elif broad_drum:
            target = max(0.24, 0.76 - 0.30 * min(1.0, confidence))
            adjusted = min(adjusted, target)
            reasons.append(f"drum_{branch}_broad_bucket_target:{target:.3f}")
        elif confidence >= 0.70:
            penalty = 0.08 if branch in {"Snare", "Clap", "RimOrStick"} else 0.14
            adjusted += penalty
            reasons.append(f"drum_branch_mismatch:+{penalty:.2f}")
        return adjusted, reasons
