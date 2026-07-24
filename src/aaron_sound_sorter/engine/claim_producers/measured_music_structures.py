# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured music-structure claim producer.

This module downshifts coherent musical-structure invariants out of the final
arbiter.  It owns broad measured roles that ShapeVoter and the physics panels
already know before arbitration: true voice/tonal stabs, rhythmic break loops,
confirmed tonal-alert FX, and clean pitched musical-loop depth.

It deliberately emits broad, source-name-blind claims.  It does not choose pack
identity from names and it does not replace the final arbiter; it just gives the
arbiter typed evidence earlier so the top layer does not need late rescue passes.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_contracts import is_rank_one_concrete_non_sax_instrument_consensus
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _feature_number_from_facts,
    _instrument_paths_share_source_branch,
    _instrument_source_branch_names,
    _measured_role_from_facts,
    _norm_path,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
    _top_physics_guess_path_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path
from aaron_sound_sorter.engine.learned_memory_contracts import has_voice_memory_match
from aaron_sound_sorter.engine.voice_source_contracts import processed_voice_source_owned
from aaron_sound_sorter.voters.scoring_tools import role_strength


class MeasuredMusicStructureClaimProducer:
    """Emit measured music/voice/FX structure claims before arbitration."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return measured structural claims in conservative priority order."""
        claims: list[ConsensusClaim] = []
        for builder in (
            self._confirmed_tonal_alert_siren_claim,
            self._true_voice_loop_claim,
            self._tonal_chord_or_voice_stab_claim,
            self._rhythmic_break_loop_claim,
            self._musical_loop_depth_claim,
        ):
            claim = builder(context)
            if claim is not None:
                claims.append(claim)
        return claims

    def _confirmed_tonal_alert_siren_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Preserve measured siren/alarm FX before pitched-loop broadening."""
        raw = context.raw
        if raw.family == "FX" and _path_has_any(raw.folder_path, ("siren", "alarm")):
            return None
        if not self._facts_support_confirmed_tonal_alert_siren(context.facts):
            return None
        return claim_from_folder_path(
            folder_path="FX/Designed Noise FX/Siren/Long FX",
            source="final_measured_tonal_alert_siren_invariant",
            reason=(
                "measured tonal-alert/siren claim: multi-lane FX and tonal-alert "
                "evidence were established before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=1.0,
            is_real_candidate=False,
        )

    def _true_voice_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Keep measured real voice under Instruments/Voice.

        Processed vocals can carry ``designed_*`` shape character because reverb,
        filtering, and stutter tails look like FX.  The source family is still
        Voice only when the parent role and voice panels agree while drum
        material is weak.
        """
        facts = context.facts
        if self._branch_consensus_blocks_weak_voice_claim(context):
            return None
        if not self._facts_support_voice_instrument_loop(facts):
            return None
        raw = context.raw
        raw_path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Instruments" and "voice" in raw_path:
            return None
        duration = self._shape_number(facts, "duration_sec") or _feature_number_from_facts(facts, "duration_sec")
        onset_count = self._shape_number(facts, "onset_count")
        folder_path = "Instruments/Voice/Vocal Loops/Loops"
        if duration <= 1.35 and onset_count <= 3.0 and not bool(getattr(facts, "is_loop_like", False)):
            folder_path = "Instruments/Voice/Phrase/One Shots"
        return self._claim(
            context,
            folder_path=folder_path,
            source="final_measured_voice_invariant",
            reason=(
                "measured voice claim: parent vocal role and voice/formant panels "
                "beat FX Human/Voice or broad instrument fallbacks before final arbitration"
            ),
            strength=0.98,
        )

    def _branch_consensus_blocks_weak_voice_claim(self, context: DecisionContext) -> bool:
        """Return True when non-voice instrument branch agreement outranks weak Voice.

        Args:
            context: Current decision context with raw Brain winner and shared
                measured facts.

        Returns:
            True when the raw Brain winner is a concrete non-voice instrument,
            PhysicsVoter's top guess supports the same broad source branch, and
            the measured voice evidence is not strong enough to override that
            branch agreement.

        Side Effects:
            None.
        """
        facts = context.facts
        if facts is None or context.raw.family != "Instruments":
            return False
        raw_path = _norm_path(context.raw.folder_path or context.raw.label)
        raw_branches = _instrument_source_branch_names(raw_path)
        if not raw_branches or "voice" in raw_branches or "instrument loops" in raw_path:
            return False
        physics_path = _top_physics_guess_path_from_facts(facts)
        if not physics_path or not _instrument_paths_share_source_branch(raw_path, physics_path):
            return False
        if context.raw.brain_rank > 2:
            return False
        if self._safe_float(context.raw.raw_candidate_score, 9999.0) > 18.0:
            return False
        if self._facts_support_true_voice_role(facts):
            return False
        if self._facts_have_matched_voice_memory(facts):
            return False
        direct_voice = self._measured_score(
            facts,
            "voice_score",
            "human_spoken_voice_score",
            "human_breath_mouth_score",
        )
        core_voice = max(
            self._measured_score(facts, "voice_score", "human_breath_mouth_score"),
            _feature_number_from_facts(facts, "formant_light_voice_identity"),
        )
        roles = self._roles(facts)
        vocal_role = max(
            role_strength(roles, "vocal_music_phrase"),
            role_strength(roles, "vocal_phrase"),
            role_strength(roles, "vocal_one_shot"),
            role_strength(roles, "voiced_one_shot"),
        )
        branch_score = self._compatible_instrument_branch_score(facts, raw_branches)
        strong_voice = bool(core_voice >= 0.68 or (direct_voice >= 0.82 and vocal_role >= 0.50))
        return bool(not strong_voice and vocal_role < 0.62 and branch_score >= 0.54)

    def _facts_have_matched_voice_memory(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when learned user memory explicitly matches Voice."""
        return has_voice_memory_match(facts)

    def _facts_have_processed_voice_source_owner(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when processed-formant evidence is owned by Voice.

        Processed tonal FX, sax-like material, and spoken/rap vocals can all
        share formant-like panels.  This method is the source-ownership contract:
        broad processed shapes may become Instruments/Voice only when a vocal
        role, human correction memory, or a decisive rank-one Voice brain lane
        owns the source first.
        """
        return processed_voice_source_owned(facts)

    def _compatible_instrument_branch_score(self, facts: SharedAudioFacts | None, branches: set[str]) -> float:
        """Return measured branch support for a broad instrument branch set."""
        score_names_by_branch = {
            "brass_woodwinds": (
                "reed_wind_score",
                "reed_wind_authority_score",
                "woodwind_sax_score",
                "instrument_branch_Woodwinds",
                "instrument_branch_Brass",
            ),
            "synth": ("synth_tonal_source_score", "synth_lead_score", "synth_chord_score", "synth_pad_score"),
            "bass": ("bass_loop_score", "bass_synth_score", "bass_sub_score", "low_end_source_score"),
            "keys": ("struck_keys_score", "struck_keys_authority_score", "keys_piano_score"),
            "plucked": ("plucked_string_score", "plucked_string_authority_score", "guitar_acoustic_score"),
            "strings": ("bowed_string_score", "strings_source_score", "instrument_branch_Strings"),
            "mallet_bell": ("mallet_bell_score", "pitched_mallet_instrument_score", "instrument_branch_MalletBell"),
        }
        best = 0.0
        for branch_name in branches:
            best = max(best, self._measured_score(facts, *score_names_by_branch.get(branch_name, ())))
        return best

    def _tonal_chord_or_voice_stab_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Keep clean tonal/voice stabs out of drum and brittle plucked leaves."""
        raw = context.raw
        facts = context.facts
        if facts is None or raw.family not in {"Drums", "Instruments"}:
            return None
        raw_path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Instruments" and not _path_has_any(raw_path, ("guitar", "plucked", "strings")):
            return None
        duration = self._shape_number(facts, "duration_sec") or _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 1.80:
            return None
        shape = _shape_vote_from_facts(facts)
        if shape not in {"solo_phrase", "echo_tail_hit", "hit_with_tail", "single_hit", "pitched_phrase"}:
            return None
        if self._facts_support_voice_before_tonal_stab(facts):
            return self._claim(
                context,
                folder_path="Instruments/Voice/Phrase/One Shots",
                source="final_measured_voice_before_tonal_stab_invariant",
                reason=(
                    "measured tonal-stab claim deferred to Voice branch: voiced body "
                    "evidence beat a drum-leaf false positive before final arbitration"
                ),
                strength=0.94,
            )
        tonal_source = max(
            self._measured_score(facts, "synth_tonal_source_score", "synth_chord_score"),
            self._measured_score(
                facts,
                "struck_keys_score",
                "struck_keys_authority_score",
                "keys_tonal_decay_score",
                "keys_piano_score",
                "keys_electric_piano_score",
            ),
            self._shape_number(facts, "sustained_tonal_frame_ratio"),
        )
        keys_source = self._measured_score(
            facts,
            "struck_keys_authority_score",
            "struck_keys_score",
            "keys_tonal_decay_score",
            "keys_piano_score",
            "keys_electric_piano_score",
        )
        synth_source = self._measured_score(facts, "synth_tonal_source_score", "synth_chord_score")
        drum_source = max(
            self._measured_score(facts, "drum_hit_score"),
            self._measured_score(facts, "drum_snare_source_score"),
            self._measured_score(facts, "drum_clap_source_score"),
            self._measured_score(facts, "drum_tom_conga_source_score"),
        )
        if tonal_source < 0.54 or drum_source > 0.46:
            return None
        if self._parent_role(facts) in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}:
            plucked_source = self._measured_score(
                facts,
                "plucked_string_score",
                "guitar_acoustic_score",
                "guitar_electric_score",
                "guitar_nylon_score",
            )
            if max(keys_source, synth_source) < plucked_source + 0.02:
                return None
        target = "Instruments/Synths/Synth Chord/One Shots"
        if keys_source >= synth_source + 0.05 or keys_source >= tonal_source - 0.03:
            target = "Instruments/Keys/Rhodes/One Shots"
        return self._claim(
            context,
            folder_path=target,
            source="final_measured_tonal_chord_stab_invariant",
            reason=(
                "measured tonal-stab claim: clean pitched tonal body beat a drum-leaf "
                "false positive before final arbitration"
            ),
            strength=0.92,
        )

    def _rhythmic_break_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Keep measured rhythmic drum/break loops under Drum Loops."""
        raw = context.raw
        facts = context.facts
        if facts is None:
            return None
        path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Drums" and "drum loops" in path:
            return None
        duration = self._shape_number(facts, "duration_sec") or _feature_number_from_facts(facts, "duration_sec")
        if duration < 1.35:
            return None
        shape = _shape_vote_from_facts(facts)
        if shape not in {"beat_loop", "drum_loop", "top_loop", "bass_phrase", "repeated_phrase_loop"}:
            return None
        if not self._facts_support_measured_drum_loop_authority(facts):
            return None
        role_loop = self._measured_score(facts, "role_loop_score")
        role_phrase = self._measured_score(facts, "role_phrase_score")
        drum_loop_panel = self._measured_score(facts, "drum_loop_source_score", "rhythmic_break_loop_score")
        voice_like_panel = self._measured_score(facts, "voice_score", "human_spoken_voice_score", "formant_fx_score")
        if drum_loop_panel < 0.70 and role_loop < 0.50 and role_phrase >= role_loop + 0.14 and voice_like_panel >= 0.60:
            return None
        if self._facts_support_designed_motion_fx_over_rhythmic_break(facts):
            return None
        return self._claim(
            context,
            folder_path="Drums/Drum Loops/Loops",
            source="final_measured_rhythmic_break_loop_invariant",
            reason=(
                "measured rhythmic-break claim: repeated percussive/break structure "
                "and drum-loop authority were available before final arbitration"
            ),
            strength=0.93,
        )

    def _musical_loop_depth_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Restore measured musical-loop depth before late broadening/firewalls."""
        raw = context.raw
        facts = context.facts
        if facts is None:
            return None
        if is_rank_one_concrete_non_sax_instrument_consensus(raw):
            return None
        path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Instruments" and "voice" in path:
            return None
        if self._facts_support_true_voice_role(facts) or self._facts_support_voice_instrument_loop(facts):
            return None
        if not self._facts_support_clean_pitched_loop_body(facts):
            return None
        target = self._measured_loop_target(facts)
        if target is None:
            return None
        if raw.family == "Instruments" and self._raw_already_has_target_depth(path, target):
            return None
        source = "final_measured_musical_loop_depth_invariant"
        strength = 0.93
        if "Synths" in target:
            source = "final_measured_synth_loop_invariant"
            strength = 0.97
        elif "Keys" in target:
            source = "final_clean_keys_loop_invariant"
            strength = 1.0
        elif "Saxophone" in target or "Woodwinds" in target:
            source = "final_measured_sax_loop_invariant"
            strength = 0.95
        elif "Bass" in target:
            source = "final_measured_bass_loop_invariant"
            strength = 0.96
        return self._claim(
            context,
            folder_path=target,
            source=source,
            reason=(
                "measured musical-loop claim: clean pitched loop body and branch/panel "
                "evidence restored depth before late broadening"
            ),
            strength=strength,
        )

    def _raw_already_has_target_depth(self, normalized_raw_path: str, target: str) -> bool:
        """Return True when the raw path already has the measured target depth.

        Broad parent folders such as Brass/Woodwinds/Loops still need sax depth
        restoration, so this check must be narrower than a generic fragment match.
        """
        target_path = _norm_path(target)
        raw_path = _norm_path(normalized_raw_path)
        if "saxophone" in target_path or "/sax" in target_path:
            return "sax" in raw_path or "saxophone" in raw_path
        if "bass" in target_path:
            return "bass" in raw_path or "808" in raw_path or "sub" in raw_path
        if "keys" in target_path or "piano" in target_path:
            return "keys" in raw_path or "piano" in raw_path or "rhodes" in raw_path or "electric piano" in raw_path
        if "synth" in target_path:
            return "synth" in raw_path or "pad" in raw_path or "lead" in raw_path
        return "instrument loops" in raw_path

    def _measured_loop_target(self, facts: SharedAudioFacts) -> str | None:
        layer = self._physics_layer(facts)
        branch = str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
        keys = self._measured_score(facts, "struck_keys_score", "struck_keys_authority_score", "keys_tonal_decay_score")
        synth = self._measured_score(facts, "synth_tonal_source_score", "synth_lead_score", "synth_pad_score")
        sax = self._measured_score(facts, "woodwind_sax_score", "reed_wind_authority_score", "reed_wind_score")
        bass = self._measured_score(
            facts, "bass_sub_score", "bass_synth_score", "bass_electric_score", "low_end_source_score"
        )
        strong_synth_pad = bool(
            self._measured_score(facts, "synth_pad_score") >= 0.72
            and synth >= 0.62
            and self._shape_number(facts, "pitched_event_ratio") >= 0.80
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.74
            and self._shape_number(facts, "percussive_event_ratio") <= 0.16
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.16
        )
        if strong_synth_pad and sax < synth + 0.06 and keys < synth + 0.16:
            return "Instruments/Synths/Pads/Loops"
        if bass >= 0.62 and branch == "Bass":
            return "Instruments/Bass/Bass Loops"
        dark_low_mid_reed_branch = bool(
            branch in {"Woodwinds", "ReedWoodwind"}
            and (
                bool(layer.get("instrument_dark_low_mid_reed_loop_signal"))
                or (
                    str(layer.get("instrument_Woodwinds_subpanel_selected") or "") == "Sax"
                    and self._safe_float(layer.get("instrument_Woodwinds_subpanel_confidence"), 0.0) >= 0.74
                    and self._shape_number(facts, "low_event_ratio") >= 0.45
                )
            )
        )
        if (sax >= 0.60 and branch in {"Woodwinds", "ReedWoodwind"}) or dark_low_mid_reed_branch:
            return "Instruments/Woodwinds/Saxophone/Loops"
        if keys >= 0.60 and branch in {"Keys", "Piano", "ElectricPiano", "KeysPiano", ""}:
            if self._facts_support_acoustic_piano_loop(facts):
                return "Instruments/Keys/Piano/Loops"
            return "Instruments/Keys/Electric Piano/Loops"
        if sax >= 0.64 and branch == "":
            return "Instruments/Woodwinds/Saxophone/Loops"
        if synth >= 0.62 and branch in {"Synths", "Synth", "Synthesizer", "Electronic", "", "Strings", "BowedStrings"}:
            if self._measured_score(facts, "synth_pad_score") >= max(0.72, synth - 0.04):
                return "Instruments/Synths/Pads/Loops"
            if self._measured_score(facts, "synth_lead_score") >= max(0.72, synth - 0.04):
                return "Instruments/Synths/Synth Lead/Loops"
            return "Instruments/Synths/Synth Loops"
        if max(keys, synth, sax, bass) >= 0.56:
            return "Instruments/Instrument Loops/Loops"
        return None

    def _facts_support_acoustic_piano_loop(self, facts: SharedAudioFacts) -> bool:
        """Return True for source-blind acoustic-piano loop physics.

        Electric keys can score high on the general keys panel.  Acoustic piano
        needs the narrower low/mid chord body, hammer/struck evidence, low
        inharmonicity, and very little drum/high-band event behavior that the
        old arbiter rescue used to check late.
        """
        shape = _shape_vote_from_facts(facts)
        if shape not in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad"}:
            return False
        if _shape_confidence_from_facts(facts) < 0.78:
            return False
        keys_body_dominates_reed_decoy = bool(
            self._measured_score(facts, "keys_tonal_decay_score") >= 0.80
            and self._shape_number(facts, "mid_event_ratio") >= 0.80
            and self._shape_number(facts, "high_event_ratio") <= 0.03
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
        )
        reed_decoy_score = self._measured_score(
            facts, "woodwind_sax_score", "reed_wind_score", "reed_wind_authority_score"
        )
        return bool(
            self._measured_score(facts, "keys_tonal_decay_score") >= 0.78
            and self._measured_score(facts, "keys_hammer_attack_score") >= 0.40
            and self._measured_score(facts, "struck_keys_score") >= 0.56
            and self._measured_score(facts, "keys_partial_inharmonicity_score") <= 0.36
            and (reed_decoy_score < 0.66 or keys_body_dominates_reed_decoy)
        )

    def _facts_support_clean_pitched_loop_body(self, facts: SharedAudioFacts) -> bool:
        shape = _shape_vote_from_facts(facts)
        confidence = _shape_confidence_from_facts(facts)
        parent = self._parent_role(facts)
        duration = self._shape_number(facts, "duration_sec") or _feature_number_from_facts(facts, "duration_sec")
        onset_count = self._shape_number(facts, "onset_count")
        loop_body = duration >= 1.25 or onset_count >= 8.0
        if not loop_body:
            return False
        allowed_parent = parent in {
            "pitched_music_loop",
            "pitched_music_phrase",
            "pitched_reed_or_instrument_loop",
            "pitched_percussion_conflict_loop",
            "clean_sustained_tonal_instrument_loop",
            "mixed_music_loop",
        }
        strong_branch_panel = (
            max(
                self._measured_score(facts, "struck_keys_score", "keys_tonal_decay_score"),
                self._measured_score(facts, "synth_tonal_source_score", "synth_lead_score", "synth_pad_score"),
                self._measured_score(facts, "woodwind_sax_score", "reed_wind_authority_score"),
                self._measured_score(facts, "bass_sub_score", "bass_synth_score", "low_end_source_score"),
            )
            >= 0.58
        )
        if not (allowed_parent or strong_branch_panel):
            return False
        if shape not in {
            "pitched_phrase",
            "pitched_phrase_shape",
            "repeated_phrase_loop",
            "sustained_pad",
            "bass_phrase",
            "transition_drop",
            "designed_tonal_fx",
            "designed_low_fx",
        }:
            return False
        if shape in {"designed_tonal_fx", "designed_low_fx"} and not self._facts_support_clean_designed_music_loop(
            facts
        ):
            return False
        return bool(
            confidence >= 0.72
            and self._shape_number(facts, "pitched_event_ratio") >= 0.60
            and self._shape_number(facts, "percussive_event_ratio") <= 0.28
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.22
        )

    def _facts_support_voice_before_tonal_stab(self, facts: SharedAudioFacts) -> bool:
        # Voice wins only when struck-percussion material is not equally concrete.
        # Percussion-pack bell/block/rim hits often have stable F0 and formant-like
        # spacing, but compact struck material plus metallic/drum support should
        # remain Drums rather than Voice.
        compact_struck = self._measured_score(facts, "compact_struck_tonal_percussion_score")
        struck_material = max(
            self._measured_score(facts, "pitched_metal_percussion_score"),
            self._measured_score(facts, "struck_wood_score"),
            self._measured_score(facts, "hand_drum_membrane_score"),
        )
        drum_material = max(
            self._measured_score(facts, "drum_hit_score"),
            self._measured_score(facts, "drum_metallic_percussion_source_score"),
            self._measured_score(facts, "drum_cymbal_source_score"),
        )
        if (
            compact_struck >= 0.70
            and struck_material >= 0.64
            and drum_material >= 0.55
            and self._shape_number(facts, "duration_sec") <= 0.65
        ):
            return False
        return bool(
            self._facts_support_true_voice_role(facts)
            and max(
                self._measured_score(facts, "voice_score"),
                self._measured_score(facts, "human_spoken_voice_score"),
                self._measured_score(facts, "human_breath_mouth_score"),
            )
            >= 0.62
        )

    def _facts_support_true_voice_role(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        roles = self._roles(facts)
        role_voice = max(
            role_strength(roles, "vocal_music_phrase"),
            role_strength(roles, "vocal_phrase"),
            role_strength(roles, "vocal_one_shot"),
            role_strength(roles, "voiced_one_shot"),
        )
        if role_voice >= 0.72:
            return True
        shape = _shape_vote_from_facts(facts)
        return bool(
            shape
            in {
                "vocal_phrase",
                "vocal_one_shot",
                "hit_with_tail",
                "echo_tail_hit",
                "designed_tonal_fx",
                "designed_low_fx",
            }
            and self._measured_score(facts, "voice_score", "human_spoken_voice_score", "human_breath_mouth_score")
            >= 0.74
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.72
            and self._shape_number(facts, "percussive_event_ratio") <= 0.14
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.14
        )

    def _facts_support_voice_instrument_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        roles = self._roles(facts)
        voice_role = max(
            role_strength(roles, "vocal_music_phrase"),
            role_strength(roles, "vocal_phrase"),
            role_strength(roles, "vocal_one_shot"),
            role_strength(roles, "voiced_one_shot"),
        )
        detected_parent = self._parent_role(facts)
        voice_panel = self._measured_score(
            facts,
            "voice_score",
            "human_spoken_voice_score",
            "human_breath_mouth_score",
            "voice_choir_score",
        )
        layer = self._physics_layer(facts)
        if isinstance(layer, dict):
            branch = str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
            branch_confidence = self._safe_float(
                layer.get("instrument_branch_selected_confidence", layer.get("physics_layer_branch_confidence")),
                0.0,
            )
            rap_voice_texture = self._safe_float(layer.get("instrument_rap_voice_texture"), 0.0)
            human_voice_texture = self._safe_float(layer.get("instrument_human_voice_texture"), 0.0)
            source_owner = bool(layer.get("instrument_voice_source_owner_confirmed"))
            layer_voice_loop = bool(
                branch == "Voice"
                and branch_confidence >= 0.56
                and (source_owner or rap_voice_texture >= 0.58 or human_voice_texture >= 0.66)
                and self._shape_number(facts, "f0_voiced_ratio") >= 0.58
                and self._shape_number(facts, "pitched_event_ratio") >= 0.54
                and self._shape_number(facts, "percussive_event_ratio") <= 0.45
                and self._shape_number(facts, "drumlike_frame_ratio") <= 0.45
                and self._measured_score(facts, "drum_loop_source_score") <= 0.45
                and self._measured_score(facts, "drum_hit_score") <= 0.58
                and self._measured_score(facts, "fx_motion_score", "fx_transition_authority_score") <= 0.62
            )
            if layer_voice_loop:
                return True
        human_spoken = self._measured_score(facts, "human_spoken_voice_score")
        processed_spoken_voice_loop = bool(
            self._facts_have_internal_voice_candidate(facts, max_rank=4, max_score=1.35)
            and self._facts_have_processed_voice_source_owner(facts)
            and human_spoken >= 0.74
            and voice_panel >= 0.66
            and self._shape_number(facts, "pitched_event_ratio") >= 0.54
            and self._shape_number(facts, "percussive_event_ratio") <= 0.45
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.45
            and self._measured_score(facts, "drum_loop_source_score") <= 0.45
            and self._measured_score(facts, "drum_hit_score") <= 0.58
            and self._measured_score(facts, "fx_motion_score", "fx_transition_authority_score") <= 0.62
        )
        if processed_spoken_voice_loop:
            return True
        return bool(
            (voice_role >= 0.68 or detected_parent in {"vocal_music_phrase", "vocal_phrase", "voiced_one_shot"})
            and voice_panel >= 0.74
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.58
            and self._shape_number(facts, "pitched_event_ratio") >= 0.55
            and self._shape_number(facts, "percussive_event_ratio") <= 0.18
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.18
            and self._measured_score(facts, "drum_loop_source_score") <= 0.40
            and self._measured_score(facts, "drum_hit_score") <= 0.55
        )

    def _facts_support_clean_designed_music_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        fx_motion = max(
            self._measured_score(
                facts,
                "fx_motion_score",
                "fx_transition_authority_score",
                "fx_whoosh_sweep_score",
                "fx_drop_downlifter_score",
                "fx_riser_build_score",
                "fx_reverse_score",
                "fx_glitch_stutter_score",
                "fx_radio_electrical_score",
            ),
            self._shape_score(facts, "designed_motion_fx_loop"),
            self._shape_score(facts, "hybrid_fx_motion"),
            self._shape_score(facts, "transition_riser"),
            self._shape_score(facts, "transition_drop"),
        )
        return bool(
            _shape_confidence_from_facts(facts) >= 0.72
            and fx_motion < 0.62
            and self._shape_number(facts, "pitched_event_ratio") >= 0.85
            and max(
                self._shape_number(facts, "sustained_tonal_frame_ratio"),
                self._shape_number(facts, "non_event_tonal_ratio"),
            )
            >= 0.80
            and self._shape_number(facts, "percussive_event_ratio") <= 0.16
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.16
        )

    def _facts_support_confirmed_tonal_alert_siren(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        branch = str(self._physics_layer(facts).get("fx_branch_selected") or "")
        branch_conf = self._safe_float(self._physics_layer(facts).get("fx_branch_confidence"), 0.0)
        siren_panel = max(
            self._measured_score(facts, "tonal_alert_siren_score"),
            self._measured_score(facts, "fx_siren_score"),
            self._measured_score(facts, "fx_alarm_score"),
        )
        siren_shape = 0.78 if _shape_vote_from_facts(facts) == "siren_alarm_tone" else 0.0
        if max(siren_panel, siren_shape) < 0.70:
            return False
        if siren_panel < 0.55 and not siren_shape:
            return False
        if self._shape_number(facts, "percussive_event_ratio") > 0.18:
            return False
        if self._shape_number(facts, "drumlike_frame_ratio") > 0.18:
            return False
        fx_lane_votes = int(siren_panel >= 0.70) + int(branch == "SirenAlarm" and branch_conf >= 0.52)
        fx_lane_votes += int(self._shape_number(facts, "pitch_confidence") >= 0.45)
        return fx_lane_votes >= 2

    def _facts_support_measured_drum_loop_authority(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        roles = self._roles(facts)
        role_score = max(
            role_strength(roles, "drum_loop"),
            role_strength(roles, "low_rhythmic_drum_loop"),
            role_strength(roles, "percussive_drum_loop"),
            role_strength(roles, "bright_drum_loop"),
        )
        drum_panel = max(
            self._measured_score(facts, "drum_loop_source_score"),
            self._measured_score(facts, "drum_hit_score"),
            self._measured_score(facts, "drum_closed_hat_source_score"),
            self._measured_score(facts, "drum_cymbal_source_score"),
            self._measured_score(facts, "rhythmic_break_loop_score"),
        )
        repeat = max(
            self._shape_number(facts, "true_repetition_score"),
            self._shape_number(facts, "loop_tempo_confidence"),
            self._shape_number(facts, "pulse_regularity"),
            self._shape_number(facts, "loop_onset_periodicity"),
            # Required librosa rhythm evidence: used as supporting measured input,
            # never as a folder owner.  This strengthens loop-vs-one-shot voting
            # before the arbiter sees candidates.
            self._shape_number(facts, "librosa_loop_confidence"),
        )
        percussive = max(
            self._shape_number(facts, "percussive_event_ratio"),
            self._measured_score(facts, "onset_percussive_onset_score"),
            self._shape_number(facts, "librosa_percussive_confidence"),
        )
        pitched = max(
            self._measured_score(facts, "onset_pitched_onset_score"),
            self._measured_score(facts, "pitched_repetition_phrase_score"),
            self._shape_number(facts, "pitched_event_ratio"),
            self._shape_number(facts, "librosa_tonal_confidence"),
        )
        return bool(
            role_score >= 0.58
            or (
                drum_panel >= 0.58 and repeat >= 0.55 and percussive >= 0.32 and pitched <= max(0.86, percussive + 0.36)
            )
        )

    def _facts_support_designed_motion_fx_over_rhythmic_break(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for rhythmic sweeps/stutters that are not real breaks.

        Drum/break-loop evidence is allowed to win unless the shape stack shows
        high-confidence designed motion with steep spectral movement and FX
        subpanel support.  This prevents pulsed down-sweeps and lasers from
        being locked as drum loops only because they have many transients.
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
            self._measured_score(facts, "fx_glitch_stutter_score"),
            self._measured_score(facts, "fx_radio_electrical_score"),
            self._measured_score(facts, "fx_whoosh_sweep_score"),
            self._measured_score(facts, "fx_motion_score"),
            self._measured_score(facts, "fx_transition_authority_score"),
            self._measured_score(facts, "fx_reverse_score"),
            self._measured_score(facts, "fx_siren_score"),
            self._measured_score(facts, "fx_alarm_score"),
        )
        slope = abs(self._shape_number(facts, "centroid_slope_norm"))
        pulse = self._shape_number(facts, "pulse_regularity")
        return bool(motion_shape >= 0.78 and fx_panel >= 0.70 and slope >= 0.35 and pulse <= 0.48)

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
                    return self._safe_float(item[1], 0.0)
        return 0.0

    def _claim(
        self,
        context: DecisionContext,
        *,
        folder_path: str,
        source: str,
        reason: str,
        strength: float,
    ) -> ConsensusClaim:
        raw = context.raw
        return claim_from_folder_path(
            folder_path=folder_path,
            source=source,
            reason=reason,
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(strength, raw.strength),
            is_real_candidate=False,
        )

    def _shape_number(self, facts: SharedAudioFacts | None, name: str) -> float:
        return max(_shape_metric_from_facts(facts, name), _feature_number_from_facts(facts, name))

    def _measured_score(self, facts: SharedAudioFacts | None, *names: str) -> float:
        best = 0.0
        for name in names:
            best = max(best, _feature_number_from_facts(facts, name), _shape_metric_from_facts(facts, name))
            if facts is not None and isinstance(getattr(facts, "evidence", None), dict):
                panels = facts.evidence.get("physics_subpanels", {})
                flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
                if isinstance(flat, dict):
                    best = max(best, self._safe_float(flat.get(name), 0.0))
        return best

    def _parent_role(self, facts: SharedAudioFacts | None) -> str:
        if facts is None:
            return ""
        if isinstance(getattr(facts, "evidence", None), dict):
            parent_dict = facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(parent_dict, dict):
                parent_role = str(parent_dict.get("role_name") or parent_dict.get("detected_parent_role") or "")
                if parent_role in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}:
                    return parent_role
        parent = _measured_role_from_facts(facts)
        if parent:
            return parent
        if isinstance(getattr(facts, "evidence", None), dict):
            parent_dict = facts.evidence.get("parent_eligibility_v2", {})
            if isinstance(parent_dict, dict):
                return str(parent_dict.get("role_name") or parent_dict.get("detected_parent_role") or "")
        return ""

    def _roles(self, facts: SharedAudioFacts | None) -> dict:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return {}
        roles = facts.evidence.get("measured_roles", {})
        return roles if isinstance(roles, dict) else {}

    def _facts_have_internal_voice_candidate(
        self,
        facts: SharedAudioFacts | None,
        *,
        max_rank: int,
        max_score: float,
    ) -> bool:
        """Return True when internal voter output exposes an Instruments/Voice candidate."""
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False

        def row_matches(row: object, fallback_rank: int) -> bool:
            if not isinstance(row, dict):
                return False
            path = str(row.get("folder_path") or row.get("label") or "").lower().replace("\\", "/")
            if not path.startswith("instruments/voice"):
                row_top = str(row.get("top_family") or "").lower()
                if row_top != "instruments" or not any(
                    fragment in path for fragment in ("voice", "vocal", "choir", "spoken")
                ):
                    return False
            try:
                rank = int(row.get("rank", fallback_rank) or fallback_rank)
            except Exception:
                rank = fallback_rank
            try:
                score = float(row.get("score", row.get("combined_rank_score", rank + 1.0)) or rank + 1.0)
            except Exception:
                score = float(rank + 1.0)
            try:
                confidence = float(row.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            try:
                support = float(row.get("support", row.get("ensemble_support", 0.0)) or 0.0)
            except Exception:
                support = 0.0
            return bool(rank <= max_rank and (score <= max_score or confidence >= 0.50 or support >= 1.0))

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
            for index, guess in enumerate(guesses[:12], start=1):
                if row_matches(guess, index):
                    return True
        return False

    def _physics_layer(self, facts: SharedAudioFacts | None) -> dict:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return {}
        evidence = facts.evidence
        layer = evidence.get("physics_layer_decision", {})
        if not isinstance(layer, dict):
            layer = {}
        merged = dict(layer)
        # Some debug/acceptance paths carry selected branch facts flattened next
        # to the physics-layer packet.  Preserve source-blind behavior while
        # making the lower claim producers robust to either representation.
        for key in (
            "physics_layer_branch",
            "physics_layer_branch_confidence",
            "physics_top_layer_instrument_branch",
            "physics_top_layer_instrument_branch_confidence",
        ):
            if key in evidence and key not in merged:
                merged[key] = evidence[key]
        if "instrument_branch_selected" not in merged:
            merged["instrument_branch_selected"] = merged.get("physics_layer_branch") or merged.get(
                "physics_top_layer_instrument_branch"
            )
        if "instrument_branch_selected_confidence" not in merged:
            merged["instrument_branch_selected_confidence"] = merged.get(
                "physics_layer_branch_confidence"
            ) or merged.get("physics_top_layer_instrument_branch_confidence", 0.0)
        return merged

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            number = float(value)  # type: ignore[arg-type]
        except Exception:
            return default
        return default if number != number else number


def _path_has_any(path: str, fragments: tuple[str, ...]) -> bool:
    low = _norm_path(path)
    return any(fragment in low for fragment in fragments)


def _target_fragments(target: str) -> tuple[str, ...]:
    low = _norm_path(target)
    if "bass" in low:
        return ("bass", "808", "sub")
    if "keys" in low or "piano" in low or "rhodes" in low:
        return ("keys", "piano", "rhodes", "electric piano")
    if "sax" in low or "woodwind" in low:
        return ("sax", "woodwind", "brass")
    if "synth" in low:
        return ("synth", "pad", "lead")
    return ("instrument loops",)
