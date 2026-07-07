# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured instrument-branch claim producer.

This module moves the safest part of the old final instrument-loop invariants
into claim production.  It emits source-name-blind measured claims for cases
where ShapeVoter/physics subpanels already identify a stable bass, synth, keys,
or sax loop before the final arbiter runs.

The claims are deliberately narrow.  The older arbiter firewalls still exist as
fallbacks for direct unit tests and for edge cases not yet proven safe to
lower.  The goal is to make normal DecisionCore routing depend on measured
claims instead of late post-winner rescue calls.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_contracts import is_rank_one_concrete_non_sax_instrument_consensus
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _candidate_combined_score,
    _feature_number_from_facts,
    _norm_path,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path
from aaron_sound_sorter.voters.scoring_tools import role_strength

SYNTH_PATH_FRAGMENTS = ("synth", "electronic")
KEYS_PATH_FRAGMENTS = ("/keys/", "piano", "rhodes", "electric piano")
SAX_PATH_FRAGMENTS = ("sax", "saxophone", "woodwind")
BASS_PATH_FRAGMENTS = ("/bass/", "bass loops", "synth bass", "sub bass", "808")


class MeasuredInstrumentBranchClaimProducer:
    """Emit measured instrument branch claims before final arbitration."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return measured bass/synth/keys/sax claims in conservative order."""
        claims: list[ConsensusClaim] = []
        for builder in (
            self._short_synth_one_shot_claim,
            self._one_shot_leaf_measured_branch_loop_claim,
            self._false_sax_leaf_non_woodwind_branch_claim,
            self._mixed_low_melody_loop_claim,
            self._mixed_reed_woodwind_loop_claim,
            self._bass_loop_claim,
            self._synth_loop_claim,
            self._keys_loop_claim,
            self._sax_loop_claim,
        ):
            claim = builder(context)
            if claim is not None:
                claims.append(claim)
        return claims

    def _short_synth_one_shot_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Preserve short measured synth hits before loop/reed broadening.

        This downshifts the safe part of the old final short-instrument
        invariant.  The lower layers already know the event is short, pitched,
        non-drumlike, and synth-dominant while reed authority is weak.  Emit the
        one-shot claim here instead of waiting for a late arbiter rescue.
        """
        raw = context.raw
        path = _norm_path(raw.folder_path or raw.label)
        if raw.family != "Instruments" or "loop" not in path:
            return None
        if self._facts_support_true_voice_role(context.facts):
            return None
        duration = _feature_number_from_facts(context.facts, "duration_sec")
        if duration <= 0.0 or duration > 1.25:
            return None
        loop_role = max(
            self._role_value(context.facts, "pitched_music_loop"),
            self._role_value(context.facts, "vocal_music_loop"),
            self._role_value(context.facts, "drum_loop"),
            self._role_value(context.facts, "low_rhythmic_drum_loop"),
            self._shape_number(context.facts, "loop_pulse_strength"),
            self._shape_number(context.facts, "true_repetition_score"),
            self._shape_number(context.facts, "onset_periodicity_score"),
            self._shape_number(context.facts, "librosa_loop_confidence"),
        )
        if loop_role >= 0.55 and bool(getattr(context.facts, "is_loop_like", False)):
            return None
        synth_identity_score = self._measured_score(
            context.facts,
            "synth_tonal_source_score",
            "synth_lead_score",
            "synth_pad_score",
            "synth_chord_score",
        )
        weak_reed_authority = self._measured_score(context.facts, "reed_wind_authority_score") <= 0.58
        if not (
            synth_identity_score >= 0.62
            and synth_identity_score >= self._subpanel_score(context.facts, "woodwind_sax_score") + 0.06
            and weak_reed_authority
            and self._shape_number(context.facts, "pitched_event_ratio") >= 0.88
            and self._shape_number(context.facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(context.facts, "drumlike_frame_ratio") <= 0.12
        ):
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Synths/Synth One Shots",
            source="final_short_synth_one_shot_invariant",
            reason=(
                "measured short synth one-shot claim: short pitched non-drum event "
                "and synth-panel evidence beat weak reed/loop routing before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(0.92, raw.strength),
            is_real_candidate=raw.is_real_candidate,
        )

    def _one_shot_leaf_measured_branch_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Broaden contradicted instrument one-shot leaves to measured branch loops.

        A raw guitar/sax/etc. one-shot leaf can win even when ShapeVoter and the
        physics branch layer say the audio body is a long/repeated musical loop.
        The old arbiter repaired that after winner selection.  This producer
        emits the measured branch loop claim early when the branch layer is
        decisive and a matching candidate exists.
        """
        raw = context.raw
        raw_path = _norm_path(raw.folder_path or raw.label)
        if raw.family != "Instruments" or ("one shot" not in raw_path and "one shots" not in raw_path):
            return None
        if self._facts_support_true_voice_role(context.facts):
            return None
        if not self._facts_support_measured_loop_body(context.facts):
            return None
        if self._shape_number(context.facts, "drumlike_frame_ratio") > 0.20:
            return None
        if self._shape_number(context.facts, "percussive_event_ratio") > 0.45:
            return None
        target = self._measured_loop_branch_target(context)
        if (
            not target
            and self._raw_path_is_brass_or_woodwind(raw_path)
            and not self._facts_have_non_woodwind_branch_identity_conflict(context.facts)
            and (
                self._facts_support_repeated_tonal_hit_phrase(context.facts)
                or self._facts_support_measured_loop_body(context.facts)
            )
        ):
            target = "Instruments/Brass and Woodwinds/Loops"
        if not target and self._facts_support_repeated_pitched_loop_phrase(context.facts):
            target = "Instruments/Instrument Loops/Loops"
        if not target:
            return None
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_branch_loop_broad_bucket",
            reason=(
                "measured instrument-branch loop claim: long/repeated pitched body "
                "and branch-layer evidence contradicted a one-shot leaf before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(0.94, raw.strength),
            is_real_candidate=False,
        )

    def _false_sax_leaf_non_woodwind_branch_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Broaden a raw sax loop leaf when measured branch evidence is non-woodwind.

        ShapeVoter and the physics branch layer already know the audio is a clean
        pitched loop, but not a sax/reed body.  This claim replaces the old final
        arbiter sax-decoy rescue with an early measured broad-loop claim.
        """
        raw = context.raw
        raw_path = _norm_path(raw.folder_path or raw.label)
        if raw.family != "Instruments":
            return None
        if "sax" not in raw_path or "loop" not in raw_path or "one shot" in raw_path or "one shots" in raw_path:
            return None
        if not self._facts_support_measured_loop_body(context.facts):
            return None
        if not self._facts_have_non_woodwind_branch_identity_conflict(context.facts):
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Instrument Loops/Loops",
            source="final_false_sax_leaf_broad_instrument_loop_invariant",
            reason=(
                "measured non-woodwind branch claim: raw sax loop leaf was broadened "
                "because shape and branch evidence supported a clean pitched loop but not a sax/reed body"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(0.91, raw.strength),
            is_real_candidate=False,
        )

    def _mixed_low_melody_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Emit a broad mixed-instrument loop claim for low melody loops.

        A loop can have a huge bass/sub body and still function as a melodic
        multi-sample instrumental loop. The raw candidate window often exposes
        this as a Bass *one-shot* leaf because the low band dominates. This
        lower claim keeps the decision at the musical role level when measured
        loop evidence is strong and non-bass tonal panels also agree.
        """
        raw = context.raw
        raw_path = _norm_path(raw.folder_path or raw.label)
        if raw.family != "Instruments" or "one shot" not in raw_path:
            return None
        if self._facts_support_clean_keys_loop(context.facts) or self._facts_support_clean_vintage_keys_loop(
            context.facts
        ):
            return None
        if not self._facts_support_low_mixed_melodic_loop(context.facts):
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Instrument Loops/Loops",
            source="mixed_instrument_loop_role_claim",
            reason=(
                "measured mixed-instrument loop claim: low/bass body was present, "
                "but strong pitched-loop evidence and non-bass tonal panels showed "
                "a broad melodic instrumental loop before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=0.98,
            is_real_candidate=False,
        )

    def _mixed_reed_woodwind_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Emit a brass/woodwind loop claim for ensemble-like reed loops.

        Some sax/brass ensemble loops are detected by the physics branch as
        MixedInstrument because multiple tonal sources are active. When the
        parent role is a pitched reed/instrument loop and the woodwind/reed
        panels are credible, emit a branch-level claim instead of letting a weak
        FX or generic review row win.
        """
        raw = context.raw
        raw_path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Instruments" and ("brass" in raw_path or "woodwind" in raw_path or "sax" in raw_path):
            return None
        if not self._facts_support_mixed_reed_woodwind_loop(context.facts):
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Brass and Woodwinds/Loops",
            source="final_measured_branch_loop_broad_bucket",
            reason=(
                "measured reed/woodwind ensemble loop claim: pitched loop body, "
                "reed/woodwind panel support, and mixed-instrument branch evidence "
                "were available before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=0.97,
            is_real_candidate=False,
        )

    @staticmethod
    def _raw_path_is_brass_or_woodwind(path: str) -> bool:
        return any(
            token in path
            for token in (
                "brass",
                "woodwind",
                "sax",
                "trumpet",
                "trombone",
                "horn",
                "flute",
                "clarinet",
                "reed",
            )
        )

    def _facts_support_repeated_tonal_hit_phrase(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        return bool(
            _shape_vote_from_facts(facts) == "hit_with_tail"
            and _shape_confidence_from_facts(facts) >= 0.70
            and self._shape_number(facts, "onset_count") >= 2.8
            and self._shape_number(facts, "pitched_event_ratio") >= 0.80
            and self._shape_number(facts, "sustain_ratio") >= 0.70
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
        )

    def _facts_support_repeated_pitched_loop_phrase(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        return bool(
            duration >= 2.0
            and _shape_vote_from_facts(facts) in {"pitched_repetition_phrase", "repeated_phrase_loop"}
            and _shape_confidence_from_facts(facts) >= 0.78
            and self._shape_number(facts, "true_repetition_score") >= 0.55
            and self._shape_number(facts, "onset_span_ratio") >= 0.50
            and self._shape_number(facts, "pitched_event_ratio") >= 0.80
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.70
            and self._shape_number(facts, "percussive_event_ratio") <= 0.16
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.16
        )

    def _facts_support_measured_loop_body(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        if self._facts_support_repeated_tonal_hit_phrase(facts):
            return True
        if self._facts_support_repeated_pitched_loop_phrase(facts):
            return True
        if self._facts_support_third_party_pitched_loop_body(facts):
            return True
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        detected_role = str(self._roles(facts).get("detected_parent_role") or "")
        detected_loop_role = (
            1.0
            if detected_role
            in {
                "pitched_music_loop",
                "pitched_reed_or_instrument_loop",
                "bass_loop",
                "synth_loop",
                "vocal_music_loop",
            }
            else 0.0
        )
        measured_loop = max(
            detected_loop_role,
            self._role_value(facts, "pitched_music_loop"),
            self._role_value(facts, "pitched_reed_or_instrument_loop"),
            self._role_value(facts, "bass_loop"),
            self._role_value(facts, "synth_loop"),
            self._shape_number(facts, "librosa_loop_confidence"),
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
            and bool(getattr(facts, "is_loop_like", False))
            and self._shape_number(facts, "true_repetition_score") >= 0.55
            and self._shape_number(facts, "onset_span_ratio") >= 0.50
        )
        return bool(measured_loop >= 0.86 or structural_loop_body or repeated_pitched_loop_body)

    def _facts_support_third_party_pitched_loop_body(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when installed third-party DSP confirms a tonal loop body."""
        if facts is None:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        loop_confidence = self._shape_number(facts, "librosa_loop_confidence")
        tonal_confidence = self._shape_number(facts, "librosa_tonal_confidence")
        onset_events = self._shape_number(facts, "librosa_onset_event_count")
        pitch_support = max(
            self._shape_number(facts, "pitched_event_ratio"),
            self._shape_number(facts, "f0_voiced_ratio"),
            self._shape_number(facts, "pitch_confidence"),
            tonal_confidence,
        )
        drumlike = max(
            self._shape_number(facts, "drumlike_frame_ratio"),
            self._shape_number(facts, "percussive_event_ratio"),
        )
        return bool(
            duration >= 1.35
            and loop_confidence >= 0.64
            and onset_events >= 2.0
            and pitch_support >= 0.35
            and tonal_confidence >= 0.30
            and drumlike <= 0.42
        )

    def _measured_loop_branch_target(self, context: DecisionContext) -> str:
        layer = self._physics_layer(context.facts)
        branch = str(
            layer.get("instrument_branch_selected")
            or layer.get("physics_layer_branch")
            or layer.get("instrument_branch")
            or ""
        )
        branch_confidence = self._safe_float(
            layer.get("instrument_branch_selected_confidence", layer.get("physics_layer_branch_confidence")),
            0.0,
        )
        raw_path = _norm_path(context.raw.folder_path or context.raw.label)
        if branch == "Woodwinds":
            has_woodwind_signal = bool(
                layer.get("instrument_woodwind_source_signal") or layer.get("instrument_reed_woodwind_source_signal")
            )
            if (
                branch_confidence >= 0.88
                and has_woodwind_signal
                and self._has_candidate_support(
                    context.raw,
                    SAX_PATH_FRAGMENTS,
                    top_family="Instruments",
                    max_score=18.0,
                    max_brain_rank=8,
                    max_physics_rank=8,
                )
            ):
                return "Instruments/Woodwinds/Saxophone/Loops"
        if self._facts_support_synth_loop(context) and (
            branch == "Synth"
            or "synth" in raw_path
            or self._measured_score(context.facts, "synth_tonal_source_score") >= 0.58
        ):
            return self._measured_synth_loop_target_path(context)
        return ""

    def _bass_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        raw = context.raw
        path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Instruments" and "bass" in path and "loop" in path:
            return None
        if not self._facts_support_clean_bass_loop(context.facts):
            return None
        if self._facts_have_measured_drum_loop_authority(context.facts):
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Bass/Bass Loops",
            source="final_measured_bass_loop_invariant",
            reason=(
                "measured bass-loop claim: ShapeVoter bass phrase and measured "
                "bass-role evidence were strong before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=0.94,
            is_real_candidate=False,
        )

    def _synth_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        raw = context.raw
        path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Instruments" and "synth" in path and "loop" in path:
            return None
        if raw.family == "Instruments" and "instrument loops" not in path:
            can_correct_keys_decoy = bool(
                ("/keys/" in path or "piano" in path)
                and self._facts_support_strong_synth_pad_loop(context.facts)
                and self._facts_support_synth_loop(context)
            )
            can_correct_synth_one_shot_leaf = bool(
                "synth" in path
                and ("one shot" in path or "one shots" in path)
                and self._facts_support_synth_loop(context)
            )
            if not (can_correct_keys_decoy or can_correct_synth_one_shot_leaf):
                return None
        if not self._facts_support_synth_loop(context):
            return None
        target = self._measured_synth_loop_target_path(context)
        return claim_from_folder_path(
            folder_path=target,
            source="final_measured_synth_loop_invariant",
            reason=(
                "measured synth-loop claim: stable pitched-loop body and synth "
                "subpanel evidence were available before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=0.96,
            is_real_candidate=False,
        )

    def _keys_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        raw = context.raw
        path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Instruments" and ("/keys" in path or "piano" in path):
            return None
        if not (
            raw.family in {"FX", "_TO_REVIEW"}
            or "sax" in path
            or "woodwind" in path
            or "synth" in path
            or "instrument loops" in path
            or "measured role conflict" in path
        ):
            return None
        if self._facts_support_strong_synth_pad_loop(context.facts):
            return None
        layer = self._physics_layer(context.facts)
        branch = str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
        if (
            branch in {"Woodwinds", "ReedWoodwind"}
            and self._measured_score(context.facts, "woodwind_sax_score", "reed_wind_score") >= 0.64
            and self._measured_score(context.facts, "keys_hammer_attack_score") < 0.40
            and not self._facts_support_designed_tonal_keys_loop(context.facts)
        ):
            return None
        if not (
            self._facts_support_clean_keys_loop(context.facts)
            or self._facts_support_clean_vintage_keys_loop(context.facts)
        ):
            return None
        target = "Instruments/Keys/Electric Piano/Loops"
        if self._facts_support_acoustic_piano_loop(context):
            target = "Instruments/Keys/Piano/Loops"
        return claim_from_folder_path(
            folder_path=target,
            source="final_clean_keys_loop_invariant",
            reason=(
                "measured keys-loop claim: clean tonal keys body and keys-panel "
                "evidence were available before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=0.97,
            is_real_candidate=False,
        )

    def _sax_loop_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        parent_role = str(getattr(context.eligibility, "role_name", "") or "")
        if parent_role in {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}:
            return None
        raw = context.raw
        if self._raw_rank_one_concrete_non_sax_instrument(raw):
            return None
        path = _norm_path(raw.folder_path or "")
        if raw.family == "Instruments" and "sax" in path and "loop" in path and "one shot" not in path:
            return None
        layer = self._physics_layer(context.facts)
        branch = str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
        decisive_woodwind_branch = bool(
            branch in {"Woodwinds", "ReedWoodwind"}
            and self._measured_score(context.facts, "woodwind_sax_score", "reed_wind_score") >= 0.64
        )
        if self._facts_support_clean_keys_loop(context.facts) and not decisive_woodwind_branch:
            return None
        if self._facts_support_synth_loop(context) and not decisive_woodwind_branch:
            return None
        if not self._facts_support_measured_sax_loop(context):
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Woodwinds/Saxophone/Loops",
            source="final_measured_sax_loop_invariant",
            reason=(
                "measured sax-loop claim: reed/woodwind branch and sax-panel "
                "evidence were available before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=0.94,
            is_real_candidate=False,
        )

    def _raw_rank_one_concrete_non_sax_instrument(self, raw: ConsensusClaim) -> bool:
        """Return whether a concrete rank-one instrument claim should stand.

        Args:
            raw: Raw shared voter claim before measured branch producers add
                synthetic safety claims.

        Returns:
            True when both Brain and Physics already agree on the same concrete
            non-sax instrument label at rank one.

        Side Effects:
            None.

        Raises:
            No intentional exceptions.

        Important Constraints:
            This uses internal candidate metadata only.  It must not inspect
            producer filenames or source folder paths.
        """
        return is_rank_one_concrete_non_sax_instrument_consensus(raw)

    def _facts_support_low_mixed_melodic_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {
            "beat_loop",
            "bass_phrase",
            "pitched_repetition_phrase",
            "repeated_phrase_loop",
            "designed_low_fx",
            "designed_tonal_fx",
        }:
            return False
        minimum_confidence = 0.82 if shape in {"designed_low_fx", "designed_tonal_fx"} else 0.84
        if _shape_confidence_from_facts(facts) < minimum_confidence:
            return False
        if max(self._role_value(facts, "pitched_music_loop"), self._role_value(facts, "bass_loop")) < 0.82:
            return False
        bass_support = max(
            self._measured_score(facts, "bass_synth_score"),
            self._measured_score(facts, "bass_sub_score"),
            self._measured_score(facts, "bass_electric_score"),
            self._measured_score(facts, "low_end_source_score"),
        )
        non_bass_tonal = max(
            self._measured_score(facts, "keys_tonal_decay_score", "struck_keys_score", "keys_chord_density_score"),
            self._measured_score(
                facts, "synth_tonal_source_score", "synth_pad_score", "synth_lead_score", "synth_chord_score"
            ),
            self._measured_score(facts, "pitched_mallet_instrument_score"),
            self._measured_score(facts, "string_violin_score", "string_cello_score", "bowed_string_score"),
        )
        return bool(
            bass_support >= 0.62
            and non_bass_tonal >= 0.66
            and self._shape_number(facts, "pitched_event_ratio") >= 0.88
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.82
            and self._shape_number(facts, "non_event_tonal_ratio") >= 0.82
            and self._shape_number(facts, "onset_count") >= 6.0
            and self._shape_number(facts, "onset_span_ratio") >= 0.50
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
            and self._measured_score(facts, "drum_loop_source_score") <= 0.70
            and not self._facts_support_true_voice_role(facts)
        )

    def _facts_support_mixed_reed_woodwind_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {"pitched_repetition_phrase", "repeated_phrase_loop", "pitched_phrase", "sustained_pad"}:
            return False
        if _shape_confidence_from_facts(facts) < 0.78:
            return False
        if (
            max(
                self._role_value(facts, "pitched_reed_or_instrument_loop"),
                self._role_value(facts, "pitched_music_loop"),
            )
            < 0.76
        ):
            return False
        layer = self._physics_layer(facts)
        branch = str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
        woodwind_branch = self._safe_float(layer.get("instrument_branch_Woodwinds"), 0.0)
        compound_strength = self._safe_float(layer.get("compound_music_strength"), 0.0)
        wood_sub = str(layer.get("instrument_Woodwinds_subpanel_selected") or "")
        wood_sub_score = self._safe_float(layer.get("instrument_Woodwinds_subpanel_confidence"), 0.0)
        reed_score = max(
            self._measured_score(facts, "reed_wind_score"),
            self._measured_score(facts, "woodwind_sax_score"),
            self._safe_float(layer.get("instrument_panel_Woodwinds_Sax"), 0.0),
        )
        voice_like = max(
            self._measured_score(facts, "voice_score"),
            self._measured_score(facts, "human_spoken_voice_score"),
            self._measured_score(facts, "human_breath_mouth_score"),
        )
        if voice_like >= 0.82 and reed_score < voice_like - 0.10 and woodwind_branch < 0.66:
            return False
        credible_reed = bool(
            reed_score >= 0.54
            or woodwind_branch >= 0.58
            or (wood_sub in {"Sax", "AiryWoodwind", "Clarinet"} and wood_sub_score >= 0.66)
        )
        mixed_or_woodwind_branch = bool(
            branch in {"Woodwinds", "ReedWoodwind", "Brass"}
            or (branch == "MixedInstrument" and compound_strength >= 0.60 and woodwind_branch >= 0.55)
        )
        return bool(
            credible_reed
            and mixed_or_woodwind_branch
            and self._shape_number(facts, "pitched_event_ratio") >= 0.86
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.70
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.80
            and self._shape_number(facts, "onset_count") >= 6.0
            and self._shape_number(facts, "onset_span_ratio") >= 0.45
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
            and self._measured_score(facts, "drum_loop_source_score") <= 0.35
        )

    def _facts_support_clean_bass_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {"bass_phrase", "beat_loop", "pitched_repetition_phrase"}:
            return False
        if _shape_confidence_from_facts(facts) < 0.82:
            return False
        bass_identity = max(
            self._measured_score(facts, "bass_synth_score"),
            self._measured_score(facts, "bass_sub_score"),
            self._measured_score(facts, "bass_electric_score"),
            self._measured_score(facts, "low_end_source_score"),
        )
        layer = self._physics_layer(facts)
        branch = str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
        if branch in {"Woodwinds", "ReedWoodwind"} and (
            bool(layer.get("instrument_dark_low_mid_reed_loop_signal"))
            or self._safe_float(layer.get("instrument_branch_Woodwinds"), 0.0) >= 0.86
        ):
            return False
        bass_role = self._role_value(facts, "bass_loop")
        if max(bass_role, bass_identity) < 0.58:
            return False
        pure_low_bass_body = (
            self._shape_number(facts, "low_event_ratio") >= 0.88
            and self._shape_number(facts, "mid_event_ratio") <= 0.12
            and self._shape_number(facts, "high_event_ratio") <= 0.08
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.05
            and bass_identity >= 0.62
        )
        if bass_role < 0.55 and not pure_low_bass_body:
            return False
        if shape != "bass_phrase":
            # A ShapeVoter beat-loop can still be a bass loop when the audio is
            # extremely low, clean, pitched, and bass-branch dominated.  Do not
            # accept generic beat-loop shape alone; require weak drum-loop source
            # evidence so true drum loops stay protected.
            if not (
                pure_low_bass_body
                and bass_role >= 0.80
                and self._shape_number(facts, "pitched_event_ratio") >= 0.90
                and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.86
                and self._shape_number(facts, "non_event_tonal_ratio") >= 0.86
                and self._measured_score(facts, "drum_loop_source_score") <= 0.24
                and self._measured_score(facts, "rhythmic_break_loop_score") <= 0.44
            ):
                return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        compact_struck = self._measured_score(facts, "compact_struck_tonal_percussion_score")
        struck_material = max(
            self._measured_score(facts, "pitched_metal_percussion_score"),
            self._measured_score(facts, "struck_wood_score"),
            self._measured_score(facts, "hand_drum_membrane_score"),
        )
        drum_panel = max(
            self._measured_score(facts, "drum_hit_score"),
            self._measured_score(facts, "drum_kick_source_score"),
            self._measured_score(facts, "drum_snare_source_score"),
            self._measured_score(facts, "drum_tom_conga_source_score"),
            self._measured_score(facts, "drum_rim_stick_source_score"),
            self._measured_score(facts, "drum_metallic_percussion_source_score"),
        )
        if (
            duration > 0.0
            and duration <= 1.50
            and event_count <= 8.0
            and compact_struck >= 0.72
            and struck_material >= 0.62
            and drum_panel >= 0.32
        ):
            return False
        return bool(
            self._shape_number(facts, "low_event_ratio") >= 0.72
            and self._shape_number(facts, "high_event_ratio") <= 0.18
            and self._shape_number(facts, "percussive_event_ratio") <= 0.18
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.18
            and not self._facts_support_true_voice_role(facts)
        )

    def _facts_support_synth_loop(self, context: DecisionContext) -> bool:
        facts = context.facts
        if facts is None:
            return False
        if self._facts_have_measured_drum_loop_authority(facts):
            return False
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)
        if (
            _feature_number_from_facts(facts, "duration_sec") <= 0.90
            and shape in {"single_hit", "hit_with_tail", "solo_phrase", "pitched_phrase", "ui_blip"}
            and self._measured_score(facts, "compact_struck_tonal_percussion_score") >= 0.78
            and max(
                self._measured_score(facts, "hand_drum_membrane_score"),
                self._measured_score(facts, "pitched_metal_percussion_score"),
                self._measured_score(facts, "struck_wood_score"),
            )
            >= 0.70
            and max(
                self._measured_score(facts, "drum_hit_score"),
                self._measured_score(facts, "drum_tom_conga_source_score"),
                self._measured_score(facts, "drum_rim_stick_source_score"),
                self._measured_score(facts, "drum_snare_source_score"),
                self._measured_score(facts, "drum_metallic_percussion_source_score"),
            )
            >= 0.34
        ):
            return False
        if shape not in {
            "bass_phrase",
            "beat_loop",
            "pitched_repetition_phrase",
            "repeated_phrase_loop",
            "pitched_phrase",
            "pitched_phrase_shape",
            "sustained_pad",
            "vocal_phrase",
            "designed_low_fx",
            "designed_tonal_fx",
        }:
            return False
        if shape_conf < 0.70:
            return False
        synth_identity = self._measured_score(
            facts,
            "synth_tonal_source_score",
            "synth_lead_score",
            "synth_pad_score",
            "synth_chord_score",
        )
        has_synth_candidate = self._has_candidate_support(context.raw, SYNTH_PATH_FRAGMENTS, top_family="Instruments")
        if synth_identity < 0.58 and not has_synth_candidate:
            return False
        if self._measured_score(facts, "reed_wind_authority_score", "woodwind_sax_score") >= max(
            0.62, synth_identity - 0.04
        ):
            return False
        if self._measured_score(
            facts, "struck_keys_score", "struck_keys_authority_score", "keys_tonal_decay_score"
        ) >= max(0.70, synth_identity + 0.04):
            return False
        if self._measured_score(facts, "voice_score", "human_spoken_voice_score") >= max(0.62, synth_identity + 0.05):
            return False
        if shape == "bass_phrase":
            return bool(
                shape_conf >= 0.80
                and max(self._role_value(facts, "pitched_music_loop"), self._role_value(facts, "pitched_music_phrase"))
                >= 0.78
                and self._shape_number(facts, "pitched_event_ratio") >= 0.80
                and self._shape_number(facts, "low_event_ratio") >= 0.55
                and self._shape_number(facts, "percussive_event_ratio") <= 0.14
                and self._shape_number(facts, "drumlike_frame_ratio") <= 0.14
            )
        return bool(
            max(self._role_value(facts, "pitched_music_loop"), self._role_value(facts, "pitched_music_phrase")) >= 0.70
            and self._shape_number(facts, "pitched_event_ratio") >= 0.80
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.70
            and self._shape_number(facts, "percussive_event_ratio") <= 0.14
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.14
        )

    def _measured_synth_loop_target_path(self, context: DecisionContext) -> str:
        if self._facts_support_strong_synth_pad_loop(context.facts):
            return "Instruments/Synths/Pads/Loops"
        if self._facts_support_synth_lead_loop(context):
            return "Instruments/Synths/Synth Lead/Loops"
        return "Instruments/Synths/Synth Loops"

    def _facts_support_strong_synth_pad_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        return bool(
            shape in {"pitched_phrase_shape", "pitched_phrase", "sustained_pad", "vocal_phrase"}
            and _shape_confidence_from_facts(facts) >= 0.86
            and self._measured_score(facts, "synth_pad_score") >= 0.72
            and self._measured_score(facts, "synth_tonal_source_score") >= 0.60
            and self._shape_number(facts, "pitched_event_ratio") >= 0.86
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.78
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )

    def _facts_support_synth_lead_loop(self, context: DecisionContext) -> bool:
        facts = context.facts
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {"pitched_phrase", "pitched_phrase_shape", "solo_phrase"}:
            return False
        lead = self._measured_score(facts, "synth_lead_score")
        synth = self._measured_score(facts, "synth_tonal_source_score")
        return bool(
            _shape_confidence_from_facts(facts) >= 0.70
            and lead >= 0.72
            and synth >= 0.56
            and self._shape_number(facts, "pitched_event_ratio") >= 0.80
            and self._shape_number(facts, "percussive_event_ratio") <= 0.12
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.12
        )

    def _facts_support_clean_keys_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        if self._facts_support_true_voice_role(facts):
            return False
        shape = _shape_vote_from_facts(facts)
        if shape not in {
            "pitched_phrase",
            "pitched_phrase_shape",
            "pitched_repetition_phrase",
            "repeated_phrase_loop",
            "sustained_pad",
            "vocal_phrase",
            "designed_tonal_fx",
        }:
            return False
        keys_support = max(
            self._measured_score(facts, "struck_keys_score"),
            self._measured_score(facts, "keys_tonal_decay_score"),
            self._measured_score(facts, "struck_keys_authority_score"),
        )
        if self._facts_support_designed_tonal_keys_loop(facts):
            return True
        return bool(
            _shape_confidence_from_facts(facts) >= 0.78
            and keys_support >= 0.54
            and self._shape_number(facts, "pitched_event_ratio") >= 0.82
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.72
            and self._shape_number(facts, "low_event_ratio") >= 0.06
            and 0.50 <= self._shape_number(facts, "mid_event_ratio") <= 0.90
            and self._shape_number(facts, "high_event_ratio") <= 0.18
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.11
            and self._shape_number(facts, "percussive_event_ratio") <= 0.14
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.14
            and self._measured_score(facts, "synth_pad_score", "synth_lead_score") < keys_support + 0.20
            and self._measured_score(facts, "woodwind_sax_score", "reed_wind_score") < keys_support + 0.18
        )

    def _facts_support_designed_tonal_keys_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        keys_support = max(
            self._measured_score(facts, "struck_keys_score"),
            self._measured_score(facts, "keys_tonal_decay_score"),
            self._measured_score(facts, "struck_keys_authority_score"),
        )
        return bool(
            _shape_vote_from_facts(facts) == "designed_tonal_fx"
            and _shape_confidence_from_facts(facts) >= 0.74
            and keys_support >= 0.54
            and self._measured_score(facts, "keys_tonal_decay_score") >= 0.70
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.86
            and self._shape_number(facts, "non_event_tonal_ratio") >= 0.86
            and 0.50 <= self._shape_number(facts, "mid_event_ratio") <= 0.90
            and self._shape_number(facts, "high_event_ratio") <= 0.05
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.03
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.10
        )

    def _facts_support_clean_vintage_keys_loop(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        return bool(
            shape
            in {
                "vocal_phrase",
                "pitched_phrase",
                "repeated_phrase_loop",
                "sustained_pad",
                "pitched_repetition_phrase",
                "designed_tonal_fx",
            }
            and _shape_confidence_from_facts(facts) >= 0.80
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.78
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.78
            and self._shape_number(facts, "low_event_ratio") >= 0.08
            and 0.60 <= self._shape_number(facts, "mid_event_ratio") <= 0.90
            and self._shape_number(facts, "high_event_ratio") <= 0.18
            and self._shape_number(facts, "spectral_flatness_mean") <= 0.09
            and self._shape_number(facts, "percussive_event_ratio") <= 0.10
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.10
            and max(
                self._measured_score(facts, "struck_keys_score"), self._measured_score(facts, "keys_tonal_decay_score")
            )
            >= 0.45
        )

    def _facts_support_acoustic_piano_loop(self, context: DecisionContext) -> bool:
        facts = context.facts
        if facts is None:
            return False
        return bool(
            _shape_vote_from_facts(facts) in {"pitched_phrase", "pitched_phrase_shape", "sustained_pad", "vocal_phrase"}
            and _shape_confidence_from_facts(facts) >= 0.78
            and self._measured_score(facts, "keys_tonal_decay_score") >= 0.78
            and self._measured_score(facts, "keys_hammer_attack_score") >= 0.40
            and self._measured_score(facts, "struck_keys_score") >= 0.56
            and self._measured_score(facts, "woodwind_sax_score", "reed_wind_score") < 0.62
        )

    def _facts_support_measured_sax_loop(self, context: DecisionContext) -> bool:
        facts = context.facts
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)
        layer = self._physics_layer(facts)
        sax_panel = max(
            self._measured_score(facts, "woodwind_sax_score"),
            self._safe_float(layer.get("instrument_panel_Woodwinds_Sax"), 0.0),
            self._safe_float(layer.get("instruments_woodwinds_saxophone_one_shots_score"), 0.0),
        )
        reed_panel = max(
            self._measured_score(facts, "reed_wind_score", "reed_wind_authority_score"),
            self._safe_float(layer.get("instrument_subpanel_reed_wind_score"), 0.0),
            self._safe_float(layer.get("instrument_subpanel_reed_wind_authority_score"), 0.0),
        )
        synth_panel = self._measured_score(facts, "synth_tonal_source_score", "synth_lead_score", "synth_pad_score")
        keys_panel = self._measured_score(facts, "struck_keys_score", "keys_tonal_decay_score")
        siren_alarm_tonal_reed_loop = bool(
            shape == "siren_alarm_tone"
            and shape_conf >= 0.68
            and self._shape_number(facts, "pitched_event_ratio") >= 0.92
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.86
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.86
            and self._shape_number(facts, "non_event_tonal_ratio") >= 0.86
            and self._shape_number(facts, "true_repetition_score") >= 0.70
            and self._measured_score(facts, "loop_pulse_clarity", "loop_tempo_confidence") >= 0.62
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and max(sax_panel, reed_panel) >= 0.58
            and self._measured_score(facts, "fx_motion_score", "fx_transition_authority_score") <= 0.38
            and synth_panel < sax_panel + 0.16
            and keys_panel < sax_panel + 0.16
        )
        if (
            shape
            not in {
                "pitched_repetition_phrase",
                "pitched_phrase",
                "vocal_phrase",
                "repeated_phrase_loop",
                "transition_drop",
                "transition_riser",
            }
            and not siren_alarm_tonal_reed_loop
        ):
            return False
        branch = str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
        dark_low_mid_reed_branch = bool(
            branch in {"Woodwinds", "ReedWoodwind"}
            and bool(layer.get("instrument_dark_low_mid_reed_loop_signal"))
            and str(layer.get("instrument_Woodwinds_subpanel_selected") or "") == "Sax"
            and self._safe_float(layer.get("instrument_Woodwinds_subpanel_confidence"), 0.0) >= 0.74
        )
        if not dark_low_mid_reed_branch and (sax_panel < 0.54 or reed_panel < 0.50):
            return False
        clean_loop_shape = bool(
            shape_conf >= 0.72
            and self._shape_number(facts, "pitched_event_ratio") >= 0.80
            and self._shape_number(facts, "f0_voiced_ratio") >= 0.70
            and self._shape_number(facts, "percussive_event_ratio") <= 0.16
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.16
        )
        if dark_low_mid_reed_branch and clean_loop_shape:
            return True
        if siren_alarm_tonal_reed_loop:
            return True
        if branch in {"Woodwinds", "ReedWoodwind"} and clean_loop_shape and sax_panel >= 0.64:
            return True
        if (
            branch in {"Woodwinds", "ReedWoodwind"}
            and clean_loop_shape
            and sax_panel >= 0.62
            and reed_panel >= 0.50
            and str(layer.get("instrument_Woodwinds_subpanel_selected") or "") in {"Sax", "AiryWoodwind", "Clarinet"}
            and self._safe_float(layer.get("third_party_api_bowed_string_support"), 0.0) <= 0.24
            and self._safe_float(layer.get("third_party_api_synth_support"), 0.0) <= 0.42
        ):
            return True
        if synth_panel >= sax_panel + 0.10 or keys_panel >= sax_panel + 0.08:
            return False
        return bool(
            clean_loop_shape
            and sax_panel >= 0.60
            and reed_panel >= 0.54
            and self._has_candidate_support(
                context.raw,
                SAX_PATH_FRAGMENTS,
                top_family="Instruments",
                max_score=18.0,
                max_brain_rank=6,
                max_physics_rank=10,
            )
        )

    def _facts_have_measured_drum_loop_authority(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        roles = self._roles(facts)
        measured_drum_loop = max(
            role_strength(roles, "drum_loop"),
            role_strength(roles, "low_rhythmic_drum_loop"),
            role_strength(roles, "percussive_drum_loop"),
            role_strength(roles, "bright_drum_loop"),
            self._measured_score(facts, "drum_loop_source_score"),
        )
        onset_count = self._shape_number(facts, "onset_count")
        true_repetition = self._shape_number(facts, "true_repetition_score")
        low_event = self._shape_number(facts, "low_event_ratio")
        high_event = self._shape_number(facts, "high_event_ratio")
        bass_counter_witness = bool(
            _shape_vote_from_facts(facts) in {"beat_loop", "pitched_repetition_phrase", "bass_phrase"}
            and self._role_value(facts, "bass_loop") >= 0.88
            and self._shape_number(facts, "low_event_ratio") >= 0.88
            and self._shape_number(facts, "pitched_event_ratio") >= 0.90
            and self._shape_number(facts, "sustained_tonal_frame_ratio") >= 0.86
            and self._shape_number(facts, "percussive_event_ratio") <= 0.08
            and self._shape_number(facts, "drumlike_frame_ratio") <= 0.08
            and self._measured_score(facts, "drum_loop_source_score") <= 0.24
            and self._measured_score(facts, "rhythmic_break_loop_score") <= 0.44
            and self._measured_score(
                facts,
                "bass_synth_score",
                "bass_sub_score",
                "bass_electric_score",
                "low_end_source_score",
                "bass_808_score",
            )
            >= 0.62
        )
        if bass_counter_witness:
            return False
        return bool(
            measured_drum_loop >= 0.58
            and onset_count >= 6.0
            and (true_repetition >= 0.50 or low_event >= 0.70 or high_event >= 0.45)
        )

    def _facts_support_true_voice_role(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        roles = self._roles(facts)
        return bool(
            max(
                role_strength(roles, "vocal_music_phrase"),
                role_strength(roles, "vocal_phrase"),
                role_strength(roles, "voiced_one_shot"),
                self._measured_score(facts, "voice_score", "human_spoken_voice_score"),
            )
            >= 0.72
        )

    def _shape_number(self, facts: SharedAudioFacts | None, name: str) -> float:
        return max(_shape_metric_from_facts(facts, name), _feature_number_from_facts(facts, name))

    def _measured_score(self, facts: SharedAudioFacts | None, *names: str) -> float:
        best = 0.0
        for name in names:
            best = max(
                best,
                _feature_number_from_facts(facts, name),
                _shape_metric_from_facts(facts, name),
                self._subpanel_score(facts, name),
            )
        return best

    def _subpanel_score(self, facts: SharedAudioFacts | None, name: str) -> float:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return 0.0
        panels = facts.evidence.get("physics_subpanels", {})
        flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
        value = flat.get(name) if isinstance(flat, dict) else None
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    def _role_value(self, facts: SharedAudioFacts | None, role_name: str) -> float:
        return role_strength(self._roles(facts), role_name)

    def _roles(self, facts: SharedAudioFacts | None) -> dict:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return {}
        roles = facts.evidence.get("measured_roles", {})
        return roles if isinstance(roles, dict) else {}

    def _physics_layer(self, facts: SharedAudioFacts | None) -> dict:
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return {}
        evidence = facts.evidence
        merged: dict = {}

        # Product runs store the richest layer packet inside the top PhysicsVoter
        # guess evidence.  Direct tests often pass a smaller physics_layer_decision
        # packet or flattened branch keys.  Merge all source-blind internal forms
        # so lower claim producers see the same measured branch facts the manifest
        # already reports.
        physics_vote = evidence.get("physics_vote_result", {})
        top_guesses = physics_vote.get("top_guesses", []) if isinstance(physics_vote, dict) else []
        if isinstance(top_guesses, list) and top_guesses:
            top_evidence = top_guesses[0].get("evidence", {}) if isinstance(top_guesses[0], dict) else {}
            if isinstance(top_evidence, dict):
                merged.update(top_evidence)

        layer = evidence.get("physics_layer_decision", {})
        if isinstance(layer, dict):
            merged.update(layer)

        # Some debug/acceptance paths carry selected branch facts flattened next
        # to the physics-layer packet. Preserve source-blind behavior while
        # making the lower claim producers robust to either representation.
        for key in (
            "physics_layer_branch",
            "physics_layer_branch_confidence",
            "physics_top_layer_instrument_branch",
            "physics_top_layer_instrument_branch_confidence",
            "instrument_branch_selected",
            "instrument_branch_selected_confidence",
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

    def _facts_have_non_woodwind_branch_identity_conflict(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when layered physics selected a strong non-woodwind branch."""
        layer = self._physics_layer(facts)
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
        confidence = self._safe_float(
            layer.get("physics_layer_branch_confidence", layer.get("instrument_branch_selected_confidence")),
            0.0,
        )
        if confidence < 0.72 and not (branch == "MixedInstrument" and confidence >= 0.70):
            return False
        woodwind_branch = self._safe_float(layer.get("instrument_branch_Woodwinds"), 0.0)
        subpanel = str(layer.get(f"instrument_{branch}_subpanel_selected") or "")
        subpanel_confidence = self._safe_float(layer.get(f"instrument_{branch}_subpanel_confidence"), 0.0)
        subpanel_margin = self._safe_float(layer.get(f"instrument_{branch}_subpanel_margin"), 0.0)
        compound_strength = self._safe_float(layer.get("compound_music_strength"), 0.0)
        return bool(
            (subpanel and subpanel_confidence >= 0.76 and subpanel_margin >= 0.075)
            or (branch == "MixedInstrument" and compound_strength >= 0.54)
            or (
                branch in {"MalletBell", "Synth", "KeysPiano", "Strings", "Brass", "Bass"}
                and confidence >= max(0.80, woodwind_branch + 0.02)
            )
        )

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
            score = _candidate_combined_score(row)
            brain_rank = self._safe_int(row.get("brain_rank"), 999)
            physics_rank = self._safe_int(row.get("physics_rank"), 999)
            if score <= max_score or brain_rank <= max_brain_rank or physics_rank <= max_physics_rank:
                return True
        return False

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except Exception:
            return default

    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        try:
            return int(float(value))  # type: ignore[arg-type]
        except Exception:
            return default
