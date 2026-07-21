# SOURCE-NAME BLINDNESS INVARIANT:
# Profile candidate producers may inspect voter output and measured audio facts.
# They must never inspect source filenames or source folders.
"""Instrument-family profile candidate claims."""

from __future__ import annotations

from dataclasses import dataclass

from aaron_sound_sorter.domain.models import SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_voice_source_score_from_facts,
    _feature_number_from_facts,
    _measured_role_from_facts,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
    _safe_float,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
    _voice_claim_has_tonal_instrument_conflict,
    _voice_identity_from_facts,
)
from aaron_sound_sorter.engine.family_claims import (
    ConsensusClaim,
    claim_from_category_guess,
    claim_from_folder_path,
)

REED_FRAGMENTS = (
    "sax",
    "saxophone",
    "brass",
    "woodwind",
    "woodwinds",
    "flute",
    "clarinet",
    "horn",
    "trumpet",
    "trombone",
    "reed",
)

SAX_REED_FRAGMENTS = ("sax", "saxophone")

NON_SAX_REED_LEAF_FRAGMENTS = (
    "brass/horns",
    "brass/horn",
    "horns",
    "horn",
    "trumpet",
    "trombone",
    "harmonica",
    "brass",
)

SYNTH_FRAGMENTS = (
    "synths",
    "synth",
    "synth pad",
    "synth lead",
    "synth loop",
    "synth loops",
    "synth pluck",
    "synth chord",
    "synth arp",
    "synth one shot",
    "electronic",
)

KEYS_FRAGMENTS = (
    "keys",
    "piano",
    "electric piano",
    "rhodes",
    "wurlitzer",
    "organ",
    "processed keys",
    "harpsichord",
)

MALLET_BELL_FRAGMENTS = (
    "mallets and bells",
    "bells and mallets",
    "vibraphone",
    "marimba",
    "glockenspiel",
    "celesta",
    "mallet",
    "bell",
)

STRING_GUITAR_FRAGMENTS = (
    "strings",
    "string loops",
    "string drones",
    "string plucks",
    "violin",
    "viola",
    "cello",
    "guitar",
    "guitar loops",
    "guitar plucks",
    "nylon guitar",
    "electric guitar",
    "acoustic guitar",
)

VOICE_FRAGMENTS = (
    "voice",
    "vocal",
    "vocals",
    "choir",
    "spoken",
    "processed voice",
    "breath",
    "mouth",
    "human and voice",
)

BASS_FRAGMENTS = (
    "bass",
    "808",
    "sub bass",
    "synth bass",
    "electric bass",
    "upright bass",
)

PROTECTED_NON_REED_INSTRUMENT_BRANCH_FRAGMENTS = (
    STRING_GUITAR_FRAGMENTS + KEYS_FRAGMENTS + SYNTH_FRAGMENTS + MALLET_BELL_FRAGMENTS
)

STABLE_PITCHED_SHAPES = {"pitched_phrase", "sustained_pad", "vocal_phrase", "bass_phrase"}
STABLE_PITCHED_ROLES = {
    "pitched_music_loop",
    "pitched_music_phrase",
    "pitched_reed_or_instrument_loop",
    "pitched_reed_or_instrument_phrase",
    "mixed_music_loop",
    "clean_sustained_tonal_instrument_loop",
}


@dataclass(frozen=True)
class ReedCandidateSupport:
    """Measured-safe reed/wind candidate evidence from internal voter lanes."""

    path: str
    rank: int
    lane_count: int
    has_product_support: bool
    has_primary_lane_support: bool
    shared_score: float | None = None

    @property
    def is_strong(self) -> bool:
        return bool(
            self.has_product_support
            or self.has_primary_lane_support
            or self.lane_count >= 2
            or (self.shared_score is not None and self.shared_score <= 16.0)
        )


@dataclass(frozen=True)
class InstrumentSiblingSupport:
    """Candidate support for a concrete non-reed instrument sibling."""

    path: str
    rank: int
    shared_score: float | None
    lane_count: int = 0

    @property
    def is_present(self) -> bool:
        return bool(self.path or self.shared_score is not None or self.rank < 999)


@dataclass(frozen=True)
class MixedInstrumentLoopSpreadSupport:
    """Candidate spread evidence for broad mixed-instrument loop fallback."""

    groups: tuple[str, ...]
    best_score: float
    has_physics_broad_loop: bool
    has_mixed_branch_evidence: bool

    @property
    def group_count(self) -> int:
        return len(self.groups)


class ProfileInstrumentClaimMixin:
    """Produce instrument-family claims from voter candidates and measured facts."""

    def brass_woodwind_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        physics_result: VoterResult | None,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_path: str,
        raw_score: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Return a brass/woodwind loop claim from a real BrainVoter candidate."""
        specific_reed_role = role_name in {
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
        }
        shape_supports_reed = shape_name in STABLE_PITCHED_SHAPES and shape_confidence >= 0.70
        if not shape_supports_reed and shape_name not in {"hit_with_tail", "single_hit"}:
            return None
        raw_is_dangerous_fx = raw.final_top == "FX" and _path_has_any(
            raw_path,
            (
                "alarm",
                "siren",
                "beep",
                "blip",
                "human and voice",
                "glitch",
                "stutter",
                "riser",
                "build",
                "animals",
                "creatures",
                "bird",
                "cat",
                "dog",
            ),
        )
        raw_is_generic_instrument = raw.final_top == "Instruments" and _path_has_any(
            raw_path,
            ("instrument loops", "mixed musical loops"),
        )
        raw_is_reed = raw.final_top == "Instruments" and _path_has_any(raw_path, REED_FRAGMENTS)
        shared_reed = self.best_shared_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=REED_FRAGMENTS,
        )
        shared_reed_score = shared_reed[0] if shared_reed is not None else 9999.0
        reed_support = self._reed_candidate_support(
            brain_result=brain_result,
            facts=facts,
            shared_reed_score=None if shared_reed is None else shared_reed_score,
        )
        has_reed_witness = reed_support is not None and reed_support.is_strong
        sax_like_reed_body = self._facts_support_sax_like_reed_loop(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        )
        raw_is_protected_sibling_instrument = self._raw_is_protected_non_reed_instrument_branch(
            raw_path=raw_path,
            raw=raw,
            raw_is_generic_instrument=raw_is_generic_instrument,
            raw_is_reed=raw_is_reed,
        )
        protected_raw_group = self._instrument_branch_group(raw_path)
        early_measured_reed_parent_only = self._facts_support_measured_reed_parent_loop(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        )
        early_sax_leaf = self._best_sax_leaf_support(
            raw=raw,
            brain_result=brain_result,
            facts=facts,
        )
        early_safe_sax_leaf_depth = self._safe_sax_leaf_depth_claim(
            raw=raw,
            raw_path=raw_path,
            raw_score=raw_score,
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
            sax_leaf=early_sax_leaf,
        )
        if raw_is_protected_sibling_instrument and protected_raw_group == "string_guitar":
            return None
        if (
            raw_is_protected_sibling_instrument
            and not early_safe_sax_leaf_depth
            and not (sax_like_reed_body and has_reed_witness)
        ):
            if (
                early_measured_reed_parent_only
                and not early_safe_sax_leaf_depth
                and protected_raw_group in {"synth", "mallet_bell", "keys", "voice"}
            ):
                return self._measured_reed_parent_claim(
                    raw=raw,
                    role_name=role_name,
                    shape_name=shape_name,
                    shape_confidence=shape_confidence,
                    support_path="measured_reed_parent",
                    support_score=raw_score + 1.0,
                    rank=None,
                )
            return None
        raw_is_review = raw.final_top == "_TO_REVIEW"
        raw_is_wrong_specific_instrument = (
            raw.final_top == "Instruments"
            and not raw_is_generic_instrument
            and not raw_is_reed
            and not _path_has_any(raw_path, BASS_FRAGMENTS)
            and self._measured_has_stable_music_body(facts)
            and not self._raw_voice_has_true_voice_support(raw_path, facts)
        )
        sax_leaf = self._best_sax_leaf_support(
            raw=raw,
            brain_result=brain_result,
            facts=facts,
        )
        safe_sax_leaf_depth = self._safe_sax_leaf_depth_claim(
            raw=raw,
            raw_path=raw_path,
            raw_score=raw_score,
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
            sax_leaf=sax_leaf,
        )
        measured_reed_parent_only = early_measured_reed_parent_only
        sibling_support = self._best_protected_sibling_support(
            raw=raw,
            brain_result=brain_result,
            shared_reed_score=shared_reed_score,
        )
        if (
            not safe_sax_leaf_depth
            and not sax_like_reed_body
            and self._sibling_support_blocks_reed_claim(
                sibling_support=sibling_support,
                reed_support=reed_support,
                specific_reed_role=specific_reed_role,
                shared_reed_score=shared_reed_score,
            )
        ):
            return None
        safe_generic_reed_depth = bool(
            raw_is_generic_instrument
            and specific_reed_role
            and shape_supports_reed
            and shared_reed is not None
            and shared_reed_score <= raw_score - 2.0
        )
        raw_is_unsafe_generic = raw_is_dangerous_fx and not _path_has_any(
            raw_path,
            ("brass", "woodwind", "sax", "reed"),
        )
        raw_fx_has_instrument_witness = bool(
            raw_is_unsafe_generic and has_reed_witness and self._physics_top_is_instrument(physics_result)
        )
        if (
            not raw_is_unsafe_generic
            and not safe_generic_reed_depth
            and not raw_is_wrong_specific_instrument
            and not raw_is_review
            and not safe_sax_leaf_depth
            and not (sax_like_reed_body and has_reed_witness)
        ):
            return None
        if not specific_reed_role and not has_reed_witness and not safe_sax_leaf_depth:
            return None

        best_guess = self.best_brain_guess(
            brain_result,
            fragments=REED_FRAGMENTS,
            max_rank=20,
            include_top={"Instruments"},
        )
        if best_guess is None and reed_support is None and not safe_sax_leaf_depth:
            if measured_reed_parent_only:
                return self._measured_reed_parent_claim(
                    raw=raw,
                    role_name=role_name,
                    shape_name=shape_name,
                    shape_confidence=shape_confidence,
                    support_path="measured_reed_parent",
                    support_score=raw_score + 1.0,
                    rank=None,
                )
            return None
        if raw_is_generic_instrument and not safe_generic_reed_depth and not safe_sax_leaf_depth:
            if measured_reed_parent_only:
                return self._measured_reed_parent_claim(
                    raw=raw,
                    role_name=role_name,
                    shape_name=shape_name,
                    shape_confidence=shape_confidence,
                    support_path="measured_reed_parent",
                    support_score=raw_score + 1.0,
                    rank=None,
                )
            return None
        if (
            not safe_sax_leaf_depth
            and _shape_metric_from_facts(facts, "high_event_ratio") >= 0.24
            and _shape_metric_from_facts(facts, "low_event_ratio") <= 0.03
        ):
            return None
        if raw_is_dangerous_fx and not raw_fx_has_instrument_witness and shared_reed_score > raw_score + 8.0:
            return None

        if best_guess is not None:
            rank = int(getattr(best_guess, "rank", 999))
        elif reed_support is not None:
            rank = reed_support.rank
        elif safe_sax_leaf_depth and sax_leaf is not None:
            rank = sax_leaf[2]
        else:
            rank = 999
        support_path = str(
            getattr(best_guess, "folder_path", "")
            or getattr(best_guess, "label", "")
            or (reed_support.path if reed_support is not None else "")
            or (sax_leaf[1] if safe_sax_leaf_depth and sax_leaf is not None else "")
        )
        support_score = min(raw_score + 2.0, float(rank) + 2.0)
        if reed_support is not None and reed_support.shared_score is not None:
            support_score = min(support_score, reed_support.shared_score)

        physics_path = ""
        if physics_result is not None and getattr(physics_result, "guesses", None):
            physics_guess = physics_result.guesses[0]
            physics_path = str(getattr(physics_guess, "folder_path", "") or getattr(physics_guess, "label", ""))
        physics_supports_reed = _path_has_any(_norm_path(physics_path), REED_FRAGMENTS)
        strict_reed_identity = bool(
            safe_sax_leaf_depth
            or raw_is_reed
            or physics_supports_reed
            or (sax_like_reed_body and has_reed_witness)
            or (
                specific_reed_role
                and shape_name == "pitched_phrase"
                and shape_confidence >= 0.93
                and shared_reed_score <= raw_score - 4.0
                and reed_support is not None
                and reed_support.lane_count >= 2
            )
        )
        if not strict_reed_identity:
            if measured_reed_parent_only:
                return self._measured_reed_parent_claim(
                    raw=raw,
                    role_name=role_name,
                    shape_name=shape_name,
                    shape_confidence=shape_confidence,
                    support_path=support_path or physics_path or "measured_reed_parent",
                    support_score=min(support_score, (raw.raw_candidate_score or 9999.0) + 1.0),
                    rank=rank,
                )
            return claim_from_folder_path(
                folder_path="Instruments/Instrument Loops/Loops",
                source="profile_candidate_ambiguous_reed_like_parent_claim",
                reason=(
                    "nearby brass/woodwind candidate was not specific enough to own "
                    "the family; broadened to Instruments/Instrument Loops instead: "
                    f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                    f"raw={raw.folder_path}; support={support_path}; physics={physics_path}"
                ),
                shared=raw.shared_candidates,
                raw_candidate_score=min(support_score, (raw.raw_candidate_score or 9999.0) + 1.0),
                brain_rank=rank,
                physics_rank=None,
                shared_winner="Instruments/Instrument Loops/Loops",
                can_override=True,
                strength=0.84,
                is_real_candidate=bool(
                    reed_support is not None
                    and (
                        reed_support.lane_count >= 2
                        or reed_support.has_primary_lane_support
                        or raw_fx_has_instrument_witness
                    )
                ),
            )

        if sax_like_reed_body and has_reed_witness and not safe_sax_leaf_depth:
            # This broader sax-body rescue is only safe over broad/generic or
            # reed-ish raw decisions.  Do not let it steal clean concrete Keys,
            # Voice, or Synth branches when the measured role itself is not a
            # reed-specific role.  Those cases are exactly where wet electric
            # piano, vocal loops, and synth poly patches were being scattered
            # into Saxophone.
            allow_false_voice_sax_body = bool(
                protected_raw_group == "voice"
                and shape_name == "pitched_phrase"
                and shape_confidence >= 0.88
                and not self._raw_voice_has_true_voice_support(raw_path, facts)
            )
            if protected_raw_group in {"keys", "synth", "string_guitar"} and role_name not in {
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
            }:
                return None
            if protected_raw_group == "voice" and not (
                allow_false_voice_sax_body
                or role_name
                in {
                    "pitched_reed_or_instrument_loop",
                    "pitched_reed_or_instrument_phrase",
                }
            ):
                return None
            if protected_raw_group == "voice" and shape_name == "vocal_phrase" and shape_confidence >= 0.74:
                return None
            sax_source = support_path or (reed_support.path if reed_support is not None else "")
            if _path_has_any(_norm_path(sax_source), ("flute",)) and not _path_has_any(raw_path, SAX_REED_FRAGMENTS):
                return None
            sax_folder = "Instruments/Woodwinds/Saxophone/Loops"
            return claim_from_folder_path(
                folder_path=sax_folder,
                source="profile_candidate_sax_leaf_claim",
                reason=(
                    "sax-like reed loop body and internal reed/wind candidates preserved "
                    "a sax leaf over a brittle sibling decision: "
                    f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                    f"raw={raw.folder_path}; support={sax_source}"
                ),
                shared=raw.shared_candidates,
                raw_candidate_score=min(support_score, raw_score + 3.0),
                brain_rank=rank,
                physics_rank=None,
                shared_winner=sax_folder,
                can_override=True,
                strength=0.95,
                is_real_candidate=True,
            )

        if safe_sax_leaf_depth and sax_leaf is not None:
            sax_score, sax_path, sax_rank = sax_leaf
            sax_folder = self._normalize_sax_leaf_folder_path(
                sax_path,
                shape_name=shape_name,
            )
            return claim_from_folder_path(
                folder_path=sax_folder,
                source="profile_candidate_sax_leaf_claim",
                reason=(
                    "candidate-backed sax/woodwind leaf preserved over a brittle "
                    "horns/generic/instrument sibling decision: "
                    f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                    f"raw={raw.folder_path}; support={sax_path}"
                ),
                shared=raw.shared_candidates,
                raw_candidate_score=sax_score,
                brain_rank=sax_rank,
                physics_rank=None,
                shared_winner=sax_folder,
                can_override=True,
                strength=max(0.94, min(0.99, 1.02 - 0.02 * max(0, sax_rank - 1))),
                is_real_candidate=True,
            )

        return claim_from_folder_path(
            folder_path="Instruments/Brass and Woodwinds/Loops",
            source="profile_candidate_brass_woodwind_claim",
            reason=(
                "BrainVoter had a nearby brass/woodwind profile candidate and measured "
                f"audio supported pitched/reed instrument structure: role={role_name}, "
                f"shape={shape_name}:{shape_confidence:.2f}; raw={raw.folder_path}; "
                f"support={support_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=support_score,
            brain_rank=rank,
            physics_rank=None,
            shared_winner=support_path or "Instruments/Brass and Woodwinds/Loops",
            can_override=True,
            strength=max(0.88, min(0.96, 1.02 - 0.03 * max(0, rank - 1))),
            is_real_candidate=True,
        )

    def voice_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_path: str,
        raw_score: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Return an Instruments/Voice claim from real voice candidate evidence.

        This protects real vocal one-shots and phrases from Synth, Mallet/Bell,
        Brass/Woodwind, or FX false positives, but it does not let the broad
        Voice branch become a vacuum.  A winning claim needs a nearby real
        Voice candidate and measured vocal or voiced-one-shot support.
        """
        raw_is_voice = raw.final_top == "Instruments" and _path_has_any(raw_path, VOICE_FRAGMENTS)
        raw_is_reed = raw.final_top == "Instruments" and _path_has_any(raw_path, REED_FRAGMENTS)
        raw_is_non_voice_instrument = raw.final_top == "Instruments" and not raw_is_voice
        raw_is_fx_voice = raw.final_top == "FX" and _path_has_any(raw_path, VOICE_FRAGMENTS)
        raw_is_ambiguous_fx = raw.final_top == "FX" and _path_has_any(
            raw_path,
            ("beep", "blip", "riser", "build", "siren", "alarm", "glitch", "stutter", "human and voice"),
        )
        specific_reed_role = role_name in {
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
        }
        if (
            specific_reed_role
            and not raw_is_voice
            and self._voice_claim_is_blocked_by_reed_lane_support(
                raw=raw,
                brain_result=brain_result,
                facts=facts,
            )
        ):
            return None
        if not (raw_is_non_voice_instrument or raw_is_fx_voice or raw_is_ambiguous_fx):
            return None
        if self._instrument_branch_group(raw_path) == "string_guitar" and raw_score <= 12.0:
            return None

        shared_voice = self.best_shared_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=VOICE_FRAGMENTS,
        )
        if shared_voice is None:
            return None
        voice_score, voice_row = shared_voice
        if raw_is_reed and self._sax_or_reed_candidate_should_block_voice(
            raw=raw,
            brain_result=brain_result,
            voice_score=voice_score,
            facts=facts,
        ):
            return None
        voice_strength = self._shared_candidate_voice_strength(voice_row)
        measured_voice_strength = max(
            _role_strength_from_facts(facts, "vocal_music_phrase"),
            _role_strength_from_facts(facts, "vocal_phrase"),
            _role_strength_from_facts(facts, "vocal_one_shot"),
            _role_strength_from_facts(facts, "voiced_one_shot"),
        )
        formant_identity = self._formant_voice_identity_from_facts(facts)
        vocal_shape = shape_name in {"vocal_phrase", "vocal_one_shot", "hit_with_tail"} and shape_confidence >= 0.70
        supported_voice = bool(
            voice_strength >= 0.74
            or measured_voice_strength >= 0.78
            or (formant_identity >= 0.65 and vocal_shape)
            or (
                vocal_shape
                and shape_confidence >= 0.90
                and _shape_metric_from_facts(facts, "f0_voiced_ratio") >= 0.88
                and voice_score <= raw_score + 12.0
                and not raw_is_reed
            )
        )
        if not supported_voice:
            return None
        strong_measured_voice = bool(
            measured_voice_strength >= 0.80
            or (formant_identity >= 0.65 and _shape_metric_from_facts(facts, "f0_voiced_ratio") >= 0.72 and vocal_shape)
            or (
                vocal_shape
                and shape_confidence >= 0.90
                and _shape_metric_from_facts(facts, "f0_voiced_ratio") >= 0.88
                and not raw_is_reed
            )
        )
        if (
            voice_score > raw_score + 3.0
            and not (voice_strength >= 0.90 and voice_score <= raw_score + 8.0)
            and not (strong_measured_voice and voice_score <= raw_score + 14.0)
        ):
            return None

        folder_path = str(voice_row.get("folder_path") or voice_row.get("label") or "Instruments/Voice/Loops")
        if not folder_path.startswith("Instruments/Voice"):
            folder_path = (
                "Instruments/Voice/Phrase/One Shots"
                if shape_name in {"vocal_one_shot", "hit_with_tail"}
                else "Instruments/Voice/Loops"
            )
        return claim_from_folder_path(
            folder_path=folder_path,
            source="profile_candidate_voice_claim",
            reason=(
                "real Instruments/Voice candidate had measured vocal or voiced-one-shot support; "
                f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}; raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=voice_score,
            brain_rank=self._int_from_row(voice_row.get("brain_rank")),
            physics_rank=self._int_from_row(voice_row.get("physics_rank")),
            shared_winner=folder_path,
            can_override=True,
            strength=max(0.90, min(0.98, 0.78 + 0.18 * max(voice_strength, measured_voice_strength, formant_identity))),
            is_real_candidate=True,
        )

    def synth_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_path: str,
        raw_score: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Return a Synths claim when synth evidence competes with broad/reed/voice claims."""
        if not self._shape_or_role_supports_stable_music(
            role_name,
            shape_name,
            shape_confidence,
            facts,
        ):
            return None
        if self._facts_have_measured_drum_loop_authority(facts):
            return None
        raw_is_generic = raw.final_top == "Instruments" and _path_has_any(
            raw_path,
            ("instrument loops",),
        )
        raw_is_mixed_loop = raw.final_top == "Instruments" and _path_has_any(
            raw_path,
            ("mixed musical loops", "multi instrument"),
        )
        raw_is_reed = raw.final_top == "Instruments" and _path_has_any(raw_path, REED_FRAGMENTS)
        raw_is_voice = raw.final_top == "Instruments" and _path_has_any(raw_path, VOICE_FRAGMENTS)
        raw_is_synth = raw.final_top == "Instruments" and _path_has_any(raw_path, SYNTH_FRAGMENTS)
        raw_is_mallet_bell = raw.final_top == "Instruments" and _path_has_any(raw_path, MALLET_BELL_FRAGMENTS)
        raw_is_plucked_or_string = raw.final_top == "Instruments" and _path_has_any(
            raw_path,
            ("plucked strings", "harp", "banjo", "koto", "guitar", "guitar loops", "string plucks"),
        )
        raw_is_ambiguous_fx = raw.final_top == "FX" and _path_has_any(
            raw_path,
            ("siren", "alarm", "beep", "human and voice", "glitch", "stutter", "riser", "build"),
        )
        if self._facts_support_strong_fx_role_layer(facts):
            return None
        synth_lead_like_body = self._facts_support_synth_lead_like_body(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        )
        raw_is_non_sax_reed_synth_false_positive = bool(
            raw_is_reed and not _path_has_any(raw_path, SAX_REED_FRAGMENTS) and synth_lead_like_body
        )
        # Mallet/Bell is a concrete sibling, so Synth cannot steal it merely
        # because both are pitched.  It may compete only when the measured body
        # is sustained/non-percussive and therefore structurally unlike a real
        # struck mallet or bell loop.
        raw_is_synth_like_mallet_false_positive = bool(
            raw_is_mallet_bell and (self._raw_mallet_bell_lacks_struck_event_body(facts) or synth_lead_like_body)
        )
        raw_is_synth_like_plucked_false_positive = bool(
            raw_is_plucked_or_string and self._raw_plucked_string_lacks_pluck_body(facts)
        )
        if not (
            raw_is_generic
            or raw_is_voice
            or raw_is_synth
            or raw_is_ambiguous_fx
            or raw_is_synth_like_mallet_false_positive
            or raw_is_synth_like_plucked_false_positive
            or raw_is_non_sax_reed_synth_false_positive
        ):
            return None
        if raw_is_reed and not raw_is_non_sax_reed_synth_false_positive or raw_is_mixed_loop:
            # Concrete reed/brass/woodwind and mixed-instrument parents are
            # sibling branches, not weak placeholders.  Synth may not steal
            # them merely because both families are pitched/tonal.
            return None
        if _path_has_any(raw_path, VOICE_FRAGMENTS) and self._raw_voice_has_true_voice_support(raw_path, facts):
            return None
        if self._facts_have_strong_human_voice_identity(facts) and not synth_lead_like_body:
            return None
        if (
            shape_name == "solo_phrase"
            and not synth_lead_like_body
            and not self._facts_support_decisive_synth_source_body(facts)
        ):
            return None
        if self._facts_support_clean_designed_tonal_keys_loop(facts):
            return None
        if (
            raw_is_synth_like_plucked_false_positive
            and shape_name == "solo_phrase"
            and not synth_lead_like_body
            and not self._facts_support_decisive_synth_source_body(facts)
        ):
            return None

        shared_synth = self.best_shared_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=SYNTH_FRAGMENTS,
        )
        best_guess = self.best_brain_guess(
            brain_result,
            fragments=SYNTH_FRAGMENTS,
            max_rank=12,
            include_top={"Instruments"},
        )
        if shared_synth is None and best_guess is None:
            return None
        if raw_is_synth and shape_name == "vocal_phrase" and shape_confidence >= 0.88:
            voice_guess = self.best_brain_guess(
                brain_result,
                fragments=VOICE_FRAGMENTS,
                max_rank=4,
                include_top={"Instruments"},
            )
            if voice_guess is not None:
                return None
        shared_score = shared_synth[0] if shared_synth is not None else 9999.0
        brain_rank = int(getattr(best_guess, "rank", 999)) if best_guess is not None else 999
        if raw_is_synth and self._facts_support_crowded_non_synth_instrument_loop(
            raw=raw,
            facts=facts,
            raw_score=raw_score,
            shared_synth_score=shared_score,
        ):
            return None
        if raw_is_synth and role_name in {"pitched_reed_or_instrument_loop", "pitched_reed_or_instrument_phrase"}:
            reed_shared = self.best_shared_candidate(
                raw,
                include_top={"Instruments"},
                include_fragments=REED_FRAGMENTS,
            )
            reed_score = reed_shared[0] if reed_shared is not None else 9999.0
            if reed_score <= shared_score + 8.0:
                return None
        if (
            raw_is_synth
            and _path_has_any(raw_path, ("synth pluck", "synths/synth pluck"))
            and role_name in {"pitched_reed_or_instrument_loop", "pitched_reed_or_instrument_phrase"}
            and not synth_lead_like_body
        ):
            return None
        if raw_is_synth and shape_name == "vocal_phrase":
            voice_shared = self.best_shared_candidate(
                raw,
                include_top={"Instruments"},
                include_fragments=VOICE_FRAGMENTS,
            )
            voice_score = voice_shared[0] if voice_shared is not None else 9999.0
            if voice_score <= shared_score + 12.0:
                return None
        sax_leaf = self._best_sax_leaf_support(
            raw=raw,
            brain_result=brain_result,
            facts=facts,
        )
        if (
            (not raw_is_synth or (raw_is_synth and not synth_lead_like_body))
            and self._safe_sax_leaf_depth_claim(
                raw=raw,
                raw_path=raw_path,
                raw_score=raw_score,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                facts=facts,
                sax_leaf=sax_leaf,
            )
            and self._sax_leaf_is_close_enough_to_block_synth(
                raw=raw,
                sax_leaf=sax_leaf,
                shared_synth=shared_synth,
                best_guess=best_guess,
                facts=facts,
            )
        ):
            return None

        # A low-ranked Synth neighbor is not enough to steal another concrete
        # instrument branch.  It must either be the current raw branch, a real
        # broad/generic rescue target, or a top/decisive synth candidate.
        synth_has_primary_support = bool(
            raw_is_synth
            or raw_is_generic
            or raw_is_ambiguous_fx
            or raw_is_synth_like_mallet_false_positive
            or raw_is_synth_like_plucked_false_positive
            or brain_rank <= 3
            or (synth_lead_like_body and brain_rank <= 6)
            or shared_score <= raw_score - 1.0
        )
        if not synth_has_primary_support:
            return None
        if shared_score > raw_score + 14.0 and brain_rank > 6:
            return None
        if raw_is_voice and not raw_is_synth:
            # A synth sibling may beat a non-true-voice false positive only when
            # it has strong direct support.  Concrete reed/brass raw winners are
            # protected above so Synth cannot become the new vacuum.
            if shared_score > raw_score + 8.0 and brain_rank > 3:
                return None
        if raw_is_synth_like_mallet_false_positive and brain_rank > 3 and shared_score > raw_score + 6.0:
            return None
        if raw_is_synth_like_plucked_false_positive and brain_rank > 2 and shared_score > raw_score + 4.0:
            return None

        if synth_lead_like_body:
            folder_path = "Instruments/Synths/Synth Lead/Loops"
        else:
            folder_path = self._preferred_folder_path_from_shared_or_brain(
                shared_candidate=shared_synth,
                best_guess=best_guess,
                prefer_brain_when_rank_at_most=2,
            )
        if raw_is_ambiguous_fx and brain_rank > 3 and "one shot" in folder_path.lower():
            return None
        support_score = min(
            shared_score,
            raw_score + 2.0,
            float(brain_rank) + 2.0 if brain_rank < 999 else 9999.0,
        )
        return claim_from_folder_path(
            folder_path=folder_path,
            source="profile_candidate_synth_claim",
            reason=(
                "Synth sibling candidate had direct voter support and measured audio "
                f"supported stable pitched instrument structure: role={role_name}, "
                f"shape={shape_name}:{shape_confidence:.2f}; raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=support_score,
            brain_rank=None if brain_rank >= 999 else brain_rank,
            physics_rank=None,
            shared_winner=folder_path,
            can_override=True,
            strength=max(0.89, min(0.97, 1.03 - 0.03 * max(0, brain_rank - 1))),
            is_real_candidate=True,
        )

    def _facts_support_crowded_non_synth_instrument_loop(
        self,
        *,
        raw: ConsensusClaim,
        facts: SharedAudioFacts | None,
        raw_score: float,
        shared_synth_score: float,
    ) -> bool:
        """Return True when a Synth loop claim is only one crowded sibling.

        This keeps broad/mixed instrument loops from being narrowed to Synth
        when Physics and shared overlap are actually pointing at Keys, Plucked
        Strings, Reed/Winds, or Mixed Musical Loops.  A decisive measured Synth
        body still wins.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        if self._facts_support_decisive_synth_source_body(facts):
            return False
        shape_name = _shape_vote_from_facts(facts)
        if shape_name not in {"bass_phrase", "pitched_repetition_phrase", "mixed_instrument_loop"}:
            return False
        physics_top = self._physics_top_path_from_facts(facts)
        non_synth_fragments = (
            KEYS_FRAGMENTS
            + STRING_GUITAR_FRAGMENTS
            + REED_FRAGMENTS
            + ("mixed musical loops", "multi instrument", "instrument loops")
        )
        if physics_top and not _path_has_any(_norm_path(physics_top), non_synth_fragments):
            return False
        close_non_synth_count = 0
        mixed_or_reed_close = False
        for row in raw.shared_candidates:
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            if not path.startswith("instruments/") or _path_has_any(path, SYNTH_FRAGMENTS):
                continue
            if not _path_has_any(path, non_synth_fragments):
                continue
            score = _safe_float(row.get("combined_rank_score"))
            if score <= 0.0:
                score = 9999.0
            if score <= raw_score + 10.0 and score <= shared_synth_score + 8.0:
                close_non_synth_count += 1
            if _path_has_any(path, ("mixed musical loops", "multi instrument") + REED_FRAGMENTS) and score <= (
                raw_score + 8.0
            ):
                mixed_or_reed_close = True
        return bool(close_non_synth_count >= 2 or mixed_or_reed_close)

    @staticmethod
    def _physics_top_path_from_facts(facts: SharedAudioFacts) -> str:
        """Return the compact Physics top path from fact evidence."""
        evidence = facts.evidence if isinstance(getattr(facts, "evidence", None), dict) else {}
        digest = evidence.get("physics_vote_result")
        if isinstance(digest, dict):
            guesses = digest.get("top_guesses", [])
            if isinstance(guesses, list) and guesses and isinstance(guesses[0], dict):
                return str(guesses[0].get("folder_path") or guesses[0].get("label") or "")
        compact = evidence.get("physics_vote_1")
        if isinstance(compact, dict):
            return str(compact.get("folder_path") or compact.get("label") or "")
        return str(compact or "")

    def _facts_support_decisive_synth_source_body(self, facts: SharedAudioFacts | None) -> bool:
        """Return True when synth panels clearly beat sibling instrument panels."""
        synth_score = max(
            self._score_from_facts(facts, "synth_tonal_source_score"),
            self._score_from_facts(facts, "synth_lead_score"),
            self._score_from_facts(facts, "synth_pad_score"),
            self._score_from_facts(facts, "synth_chord_score"),
            self._score_from_facts(facts, "synth_pluck_score"),
            self._score_from_facts(facts, "synth_arp_score"),
            self._score_from_facts(facts, "instrument_panel_Synth_synth_pad"),
            self._score_from_facts(facts, "instrument_panel_Synth_synth_chord"),
            self._score_from_facts(facts, "instrument_panel_Synth_synth_lead"),
        )
        sibling_score = max(
            self._score_from_facts(facts, "struck_keys_score"),
            self._score_from_facts(facts, "keys_tonal_decay_score"),
            self._score_from_facts(facts, "plucked_string_score"),
            self._score_from_facts(facts, "guitar_acoustic_score"),
            self._score_from_facts(facts, "guitar_electric_score"),
            self._score_from_facts(facts, "woodwind_sax_score"),
            self._score_from_facts(facts, "reed_wind_score"),
            self._score_from_facts(facts, "voice_score"),
            self._score_from_facts(facts, "human_spoken_voice_score"),
            self._score_from_facts(facts, "voice_choir_score"),
            self._score_from_facts(facts, "bass_synth_score"),
            self._score_from_facts(facts, "bass_sub_score"),
            self._score_from_facts(facts, "bass_electric_score"),
            self._score_from_facts(facts, "mallet_bell_score"),
            self._score_from_facts(facts, "drum_hit_score"),
        )
        return bool(synth_score >= 0.74 or (synth_score >= 0.60 and synth_score >= sibling_score + 0.16))

    def measured_synth_lead_claim(
        self,
        *,
        raw: ConsensusClaim,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_path: str,
        raw_score: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Return a measured synth-lead claim when no leaf candidate wins.

        This is an abstaining specialist probe.  It does not inspect filenames.
        It fires only for a clean, non-percussive, mid-band dominant pitched
        phrase where the raw winner is a likely broad/FX false positive and no
        stronger true voice or sax leaf blocker is present.
        """
        if shape_name != "pitched_phrase":
            return None
        if not self._facts_support_synth_lead_like_body(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        ):
            return None
        raw_group = self._instrument_branch_group(raw_path)
        raw_is_eligible_false_positive = bool(
            raw.final_top == "FX"
            or (raw.final_top == "_TO_REVIEW")
            or (raw.final_top == "Instruments" and raw_group in {"generic", "synth"})
        )
        if not raw_is_eligible_false_positive:
            return None
        if self._facts_support_short_true_voice_one_shot(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        ):
            return None
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        low_event = _shape_metric_from_facts(facts, "low_event_ratio")
        mid_event = _shape_metric_from_facts(facts, "mid_event_ratio")
        entropy = _shape_metric_from_facts(facts, "spectral_entropy_mean")
        flatness = _shape_metric_from_facts(facts, "spectral_flatness_mean")
        # Reedy/airy instruments usually carry more high-event breath/noise.
        # Clean lead-synth evidence here is a tight mid-band pitched line.
        if high_event > 0.18 or low_event > 0.06 or mid_event < 0.58:
            return None
        if entropy > 0.38 or flatness > 0.20:
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Synths/Synth Lead/Loops",
            source="profile_candidate_measured_synth_lead_claim",
            reason=(
                "measured synth-lead specialist found a clean non-percussive "
                "pitched phrase after raw candidate conflict; "
                f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                f"raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=min(raw_score + 2.0, 18.0),
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner="Instruments/Synths/Synth Lead/Loops",
            can_override=True,
            strength=0.92,
            is_real_candidate=False,
        )

    def concrete_instrument_sibling_conflict_parent_claim(
        self,
        *,
        raw: ConsensusClaim,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_path: str,
        raw_score: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Broaden sibling-instrument fights to the Instrument Loops parent.

        This is the anti-vacuum rule for the instrument family.  If a concrete
        sibling leaf such as Vibraphone, Synth, Voice, Keys, Strings, or Brass
        is contradicted by another close sibling and the measured body is a
        stable pitched loop/phrase, the safe producer-facing answer is the
        broad instrument-loop parent.  It avoids swapping one wrong leaf for
        another wrong leaf.
        """
        if raw.final_top != "Instruments":
            return None
        raw_group = self._instrument_branch_group(raw_path)
        if raw_group in {"", "bass", "generic", "mixed"}:
            return None
        if self._facts_support_tonal_alert_siren_fx(role_name, facts):
            return None
        if self._facts_support_short_keys_or_synth_tonal_stab(facts):
            return None
        if not self._shape_or_role_supports_stable_music(
            role_name,
            shape_name,
            shape_confidence,
            facts,
        ):
            return None
        if self._facts_support_short_rhythmic_struck_percussion_phrase(facts):
            return None
        sustained_mallet_pad_false_positive = bool(
            raw_group == "mallet_bell"
            and self._raw_mallet_bell_has_sustained_non_struck_pad_body(
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                facts=facts,
            )
        )
        if self._raw_has_direct_terminal_identity(
            raw=raw,
            raw_path=raw_path,
            raw_group=raw_group,
            raw_score=raw_score,
        ):
            return None
        conflicts = self._close_instrument_sibling_groups(raw, raw_group=raw_group, raw_score=raw_score)
        struck_mallet_false_positive = bool(
            raw_group == "mallet_bell" and self._raw_mallet_bell_lacks_struck_event_body(facts)
        )
        unsupported_world_special = bool(
            raw_group == "world_special" and self._measured_has_stable_or_staggered_music_body(facts)
        )
        if raw_group == "voice":
            return None
        if (
            raw_group == "reed"
            and _path_has_any(raw_path, SAX_REED_FRAGMENTS)
            and self._shape_or_role_supports_stable_music(role_name, shape_name, shape_confidence, facts)
            and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.12
            and _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.16
        ):
            return None
        if len(conflicts) < 1 and not (
            struck_mallet_false_positive or unsupported_world_special or sustained_mallet_pad_false_positive
        ):
            return None
        strongest_score = min((score for _, score, _ in conflicts), default=raw_score + 9.0)
        # Strong voice one-shot evidence should use the voice claim, not broad loops.
        if raw_group == "voice" and self._raw_voice_has_true_voice_support(raw_path, facts):
            return None
        strength = 0.91
        if (
            strongest_score <= raw_score + 1.0
            or struck_mallet_false_positive
            or unsupported_world_special
            or sustained_mallet_pad_false_positive
        ):
            strength = 0.95
        return claim_from_folder_path(
            folder_path="Instruments/Instrument Loops/Loops",
            source="instrument_sibling_conflict_parent_claim",
            reason=(
                "concrete instrument sibling evidence conflicted across branches; "
                "broadened to Instruments/Instrument Loops instead of choosing a brittle leaf: "
                f"raw={raw.folder_path}; raw_group={raw_group}; conflicts={','.join(g for g, _, _ in conflicts)}; "
                f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=None,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner="Instruments/Instrument Loops/Loops",
            can_override=True,
            strength=strength,
            is_real_candidate=False,
        )

    def _facts_support_short_keys_or_synth_tonal_stab(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 1.80:
            return False
        if _shape_vote_from_facts(facts) not in {
            "solo_phrase",
            "echo_tail_hit",
            "hit_with_tail",
            "single_hit",
            "pitched_phrase",
        }:
            return False
        keys_score = max(
            self._score_from_facts(facts, "struck_keys_score"),
            self._score_from_facts(facts, "struck_keys_authority_score"),
            self._score_from_facts(facts, "keys_tonal_decay_score"),
            self._score_from_facts(facts, "keys_piano_score"),
            self._score_from_facts(facts, "keys_electric_piano_score"),
        )
        synth_score = max(
            self._score_from_facts(facts, "synth_tonal_source_score"),
            self._score_from_facts(facts, "synth_chord_score"),
        )
        plucked_score = max(
            self._score_from_facts(facts, "plucked_string_score"),
            self._score_from_facts(facts, "guitar_acoustic_score"),
            self._score_from_facts(facts, "guitar_electric_score"),
            self._score_from_facts(facts, "guitar_nylon_score"),
        )
        drum_score = max(
            self._score_from_facts(facts, "drum_hit_score"),
            self._score_from_facts(facts, "drum_snare_source_score"),
            self._score_from_facts(facts, "drum_clap_source_score"),
            self._score_from_facts(facts, "drum_tom_conga_source_score"),
        )
        return bool(
            max(keys_score, synth_score) >= 0.54
            and drum_score <= 0.46
            and max(keys_score, synth_score) >= plucked_score + 0.02
        )

    def _facts_support_tonal_alert_siren_fx(self, role_name: str, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        tonal_alert = max(
            self._score_from_facts(facts, "tonal_alert_siren_score"),
            self._score_from_facts(facts, "fx_siren_score"),
            self._score_from_facts(facts, "fx_alarm_score"),
        )
        if role_name == "fx_tonal_alert_or_siren" and tonal_alert >= 0.62:
            return True
        return bool(
            tonal_alert >= 0.82
            and max(
                self._score_from_facts(facts, "fx_siren_score"),
                self._score_from_facts(facts, "fx_alarm_score"),
            )
            >= 0.60
        )

    def _score_from_facts(self, facts: SharedAudioFacts | None, *names: str) -> float:
        best = 0.0
        for name in names:
            best = max(best, _feature_number_from_facts(facts, name), _shape_metric_from_facts(facts, name))
            if facts is not None and isinstance(getattr(facts, "evidence", None), dict):
                panels = facts.evidence.get("physics_subpanels", {})
                flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
                if isinstance(flat, dict):
                    best = max(best, _safe_float(flat.get(name)))
        return best

    def non_voice_instrument_parent_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        raw_path: str,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Broaden false-positive Instrument/Voice winners to Instrument Loops.

        This is not a voice taxonomy deletion.  It only blocks a Voice leaf when
        measured voice identity is absent and the audio body is a stable pitched
        or staggered instrument loop.  True vocals still keep the Voice branch.
        """
        if raw.final_top != "Instruments" or not _path_has_any(raw_path, VOICE_FRAGMENTS):
            return None
        if self._raw_voice_has_true_voice_support(raw_path, facts):
            return None
        if self._raw_voice_has_primary_product_brain_support(
            raw=raw, brain_result=brain_result
        ) and not self._full_brain_top_is_non_voice_instrument(facts):
            return None
        if self._raw_voice_has_rank_one_brain_and_body_support(
            raw=raw,
            brain_result=brain_result,
            facts=facts,
        ):
            return None
        if not self._measured_has_stable_or_staggered_music_body(facts):
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Instrument Loops/Loops",
            source="profile_candidate_non_voice_instrument_parent_claim",
            reason=(
                "raw Voice candidate lacked measured true-voice support while the "
                "audio body measured as a stable pitched/staggered instrument loop"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=(raw.raw_candidate_score or 9999.0) + 1.0,
            brain_rank=None,
            physics_rank=None,
            shared_winner="Instruments/Instrument Loops/Loops",
            can_override=True,
            strength=0.84,
            is_real_candidate=False,
        )

    def _facts_support_short_rhythmic_struck_percussion_phrase(self, facts: SharedAudioFacts | None) -> bool:
        """Return True for short rhythmic struck material that must not broaden to Instrument Loops.

        This protects percussion-pack wood/metal/membrane phrases that have repeated pitched
        resonances.  They can look stable and musical to the instrument sibling rule, but the
        measured body is a compact struck/percussive phrase, not a playable instrument loop.
        """
        if facts is None:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        event_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "event_count_estimate"),
        )
        shape = _shape_vote_from_facts(facts)
        compact_struck = _feature_number_from_facts(facts, "compact_struck_tonal_percussion_score")
        struck_material = max(
            _feature_number_from_facts(facts, "struck_wood_score"),
            _feature_number_from_facts(facts, "hand_drum_membrane_score"),
            _feature_number_from_facts(facts, "pitched_metal_percussion_score"),
        )
        onset_percussive = _feature_number_from_facts(facts, "onset_percussive_onset_score")
        low_event = _shape_metric_from_facts(facts, "low_event_ratio")
        attack = _shape_metric_from_facts(facts, "attack_rise_time_norm")
        rhythm = max(
            _feature_number_from_facts(facts, "rhythmic_break_loop_score"),
            _shape_metric_from_facts(facts, "true_repetition_score"),
        )
        voice_body = max(
            _feature_number_from_facts(facts, "voice_score"),
            _feature_number_from_facts(facts, "human_spoken_voice_score"),
            _feature_number_from_facts(facts, "human_breath_mouth_score"),
        )
        return bool(
            0.0 < duration <= 1.25
            and event_count <= 8.0
            and shape
            in {"pitched_repetition_phrase", "repeated_phrase_loop", "beat_loop", "bass_phrase", "solo_phrase"}
            and onset_percussive >= 0.70
            and rhythm >= 0.42
            and max(struck_material, compact_struck) >= 0.62
            and low_event >= 0.68
            and attack <= 0.03
            and voice_body <= 0.68
        )

    def _best_sax_leaf_support(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        facts: SharedAudioFacts | None,
    ) -> tuple[float, str, int] | None:
        """Return the strongest sax/woodwind leaf witness from voter candidates.

        This inspects only internal voter candidates. It does not inspect source
        filenames or source folders. The goal is to preserve a concrete sax leaf
        when the product/lane voters already expose one, rather than flattening
        it to Horns, generic Brass/Woodwinds, or Instrument Loops.
        """
        shared = self.best_shared_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=SAX_REED_FRAGMENTS,
        )
        best_guess = self.best_brain_guess(
            brain_result,
            fragments=SAX_REED_FRAGMENTS,
            max_rank=12,
            include_top={"Instruments"},
        )
        fact_guess = self._best_sax_leaf_from_fact_vote(facts)
        if shared is None and best_guess is None and fact_guess is None:
            return None

        shared_score = shared[0] if shared is not None else 9999.0
        shared_path = ""
        shared_rank = 999
        if shared is not None:
            row = shared[1]
            shared_path = str(row.get("folder_path") or row.get("label") or "")
            shared_rank = self._int_from_row(row.get("brain_rank")) or 999

        guess_rank = int(getattr(best_guess, "rank", 999)) if best_guess is not None else 999
        guess_path = ""
        if best_guess is not None:
            guess_path = str(getattr(best_guess, "folder_path", "") or getattr(best_guess, "label", ""))

        candidates: list[tuple[float, str, int]] = []
        if shared_path:
            candidates.append((shared_score, shared_path, shared_rank))
        if guess_path:
            candidates.append((min(float(guess_rank) + 1.0, shared_score), guess_path, guess_rank))
        if fact_guess is not None:
            candidates.append(fact_guess)
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[2], item[0], item[1]))
        return candidates[0]

    @staticmethod
    def _best_sax_leaf_from_fact_vote(facts: SharedAudioFacts | None) -> tuple[float, str, int] | None:
        """Return sax evidence from serialized internal voter/audit guesses.

        This is a specialist evidence probe, not a filename rescue.  Some real
        wet sax loops expose Saxophone only in the broader brain/role audit
        window while the shared brain-physics list is pulled toward keys,
        mallets, or synth siblings.  Use those existing internal candidates so
        the later arbiter can decide whether a sax leaf is safe.
        """
        if facts is None or not isinstance(facts.evidence, dict):
            return None
        candidates: list[tuple[float, str, int]] = []

        for key in (
            "physics_vote_result",
            "dry_core_physics_vote_result",
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
            for idx, guess in enumerate(guesses[:24], start=1):
                candidate = ProfileInstrumentClaimMixin._sax_leaf_tuple_from_guess_dict(guess, idx)
                if candidate is not None:
                    candidates.append(candidate)

        parent_audit = facts.evidence.get("parent_role_audit")
        if isinstance(parent_audit, dict):
            for key in ("brain_top_20", "physics_top_20"):
                guesses = parent_audit.get(key)
                if not isinstance(guesses, list):
                    continue
                for idx, guess in enumerate(guesses[:24], start=1):
                    candidate = ProfileInstrumentClaimMixin._sax_leaf_tuple_from_guess_dict(guess, idx)
                    if candidate is not None:
                        candidates.append(candidate)

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[2], item[0], item[1]))
        return candidates[0]

    @staticmethod
    def _sax_leaf_tuple_from_guess_dict(guess: object, fallback_rank: int) -> tuple[float, str, int] | None:
        """Read one sax candidate tuple from a serialized internal guess row."""
        if not isinstance(guess, dict):
            return None
        path = str(guess.get("folder_path") or guess.get("label") or "")
        if not _path_has_any(_norm_path(path), SAX_REED_FRAGMENTS):
            return None
        try:
            rank = int(guess.get("rank", fallback_rank) or fallback_rank)
        except Exception:
            rank = fallback_rank
        try:
            score = float(guess.get("score", rank + 1.0) or rank + 1.0)
        except Exception:
            score = float(rank + 1.0)
        return (min(score, float(rank) + 1.0), path, rank)

    @staticmethod
    def _normalize_sax_leaf_folder_path(path: str, *, shape_name: str) -> str:
        """Normalize candidate sax labels to the sax loop folder when appropriate."""
        normalized = _norm_path(path)
        if not _path_has_any(normalized, SAX_REED_FRAGMENTS):
            return "Instruments/Woodwinds/Saxophone/Loops"
        if "one shot" in normalized and shape_name in STABLE_PITCHED_SHAPES:
            return "Instruments/Woodwinds/Saxophone/Loops"
        if normalized.startswith("instruments/woodwinds/saxophone"):
            return path.strip("/")
        if normalized.startswith("instruments/") and _path_has_any(normalized, SAX_REED_FRAGMENTS):
            return "Instruments/Woodwinds/Saxophone/Loops"
        return "Instruments/Woodwinds/Saxophone/Loops"

    def _safe_sax_leaf_depth_claim(
        self,
        *,
        raw: ConsensusClaim,
        raw_path: str,
        raw_score: float,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
        sax_leaf: tuple[float, str, int] | None,
    ) -> bool:
        """Return True when a sax leaf is safer than broad sibling flattening.

        The rule is intentionally evidence based: a sax candidate must already
        exist inside the voter candidate set, and the measured body must be a
        stable pitched musical loop/phrase with weak drum/percussion identity.
        """
        if sax_leaf is None:
            return False
        sax_score, sax_path, sax_rank = sax_leaf
        if not _path_has_any(_norm_path(sax_path), SAX_REED_FRAGMENTS):
            return False
        if not self._shape_or_role_supports_stable_music(role_name, shape_name, shape_confidence, facts):
            return False
        if self._facts_support_short_true_voice_one_shot(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        ):
            return False

        percussive_event = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        pitched_event = _shape_metric_from_facts(facts, "pitched_event_ratio")
        voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        if percussive_event >= 0.18 or drumlike >= 0.18:
            return False
        if pitched_event < 0.72 and voiced < 0.70:
            return False
        if self._facts_support_sax_identity_decoy(
            shape_name=shape_name,
            facts=facts,
        ):
            return False

        raw_group = self._instrument_branch_group(raw_path)
        specific_reed_role = role_name in {
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
        }
        raw_is_generic_or_broad = bool(
            raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("instrument loops", "brass and woodwinds", "mixed musical loops"))
        )
        raw_is_review = raw.final_top == "_TO_REVIEW"
        raw_is_non_sax_reed_leaf = bool(
            raw.final_top == "Instruments"
            and not _path_has_any(raw_path, SAX_REED_FRAGMENTS)
            and _path_has_any(raw_path, NON_SAX_REED_LEAF_FRAGMENTS)
        )
        raw_is_unsupported_world_special = bool(
            raw.final_top == "Instruments" and _path_has_any(raw_path, ("world and special", "harmonica"))
        )
        raw_is_generic_voice_or_synth_conflict = bool(
            raw.final_top == "Instruments"
            and _path_has_any(
                raw_path,
                VOICE_FRAGMENTS + SYNTH_FRAGMENTS + KEYS_FRAGMENTS + MALLET_BELL_FRAGMENTS + STRING_GUITAR_FRAGMENTS,
            )
            and not self._raw_voice_has_true_voice_support(raw_path, facts)
        )

        # Do not let the sax rescue become a new vacuum.  The rescue may deepen
        # broad/generic or reed-like candidates to Saxophone, and it may correct
        # concrete siblings only when the measured role itself is reed-specific.
        # Otherwise clean electric piano, synth-poly, and true vocal loops can be
        # stolen by a low-ranked sax neighbor just because all are stable pitched
        # music with wet/formant-like tails.
        synth_lead_like_body = self._facts_support_synth_lead_like_body(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        )
        allow_synth_to_sax_correction = bool(
            raw_group == "synth"
            and not specific_reed_role
            and not synth_lead_like_body
            and (
                (shape_name == "pitched_phrase" and shape_confidence >= 0.88)
                or (
                    shape_name == "bass_phrase"
                    and shape_confidence >= 0.88
                    and self._wetness_score_from_facts(facts) >= 0.50
                )
            )
            and sax_rank <= 8
            and sax_score <= raw_score + 12.0
        )
        allow_false_voice_to_sax_correction = bool(
            raw_group == "voice"
            and not specific_reed_role
            and shape_name == "pitched_phrase"
            and shape_confidence >= 0.88
            and not self._raw_voice_has_true_voice_support(raw_path, facts)
            and sax_rank <= 8
            and sax_score <= 32.0
        )
        if raw_group in {"keys", "string_guitar"} and not specific_reed_role:
            return False
        if raw_group == "voice" and not (specific_reed_role or allow_false_voice_to_sax_correction):
            return False
        if raw_group == "synth" and not (specific_reed_role or allow_synth_to_sax_correction):
            return False
        if raw_group == "voice" and shape_name == "vocal_phrase" and shape_confidence >= 0.78:
            return False
        if raw_group == "keys" and not specific_reed_role and sax_rank > 1:
            return False

        if not (
            raw_is_generic_or_broad
            or raw_is_review
            or raw_is_non_sax_reed_leaf
            or raw_is_unsupported_world_special
            or raw_is_generic_voice_or_synth_conflict
        ):
            return False

        # A rank-1 sax candidate from the product brain is strong enough even if
        # physics is pulled toward mallets/bells by the reed's harmonic peaks.
        if sax_rank <= 1:
            return True
        if sax_rank <= 4 and shape_name == "pitched_phrase" and shape_confidence >= 0.90:
            return True
        if sax_rank <= 5 and sax_score <= raw_score + 12.0 and shape_name == "pitched_phrase":
            return True
        if sax_score <= 14.0 and role_name in {
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
        }:
            return True
        return bool(
            sax_rank <= 8
            and sax_score <= raw_score + 26.0
            and shape_name in {"pitched_phrase", "sustained_pad", "vocal_phrase", "bass_phrase"}
            and shape_confidence >= 0.74
            and self._wetness_score_from_facts(facts) >= 0.35
            and self._measured_has_stable_or_staggered_music_body(facts)
        )

    @staticmethod
    def _facts_support_sax_identity_decoy(
        *,
        shape_name: str,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for clean pad/keys/voice loops that can mimic sax.

        This mirrors the PhysicsVoter sax-identity anti-steal witness.  The
        claim layer may preserve a sax leaf only after the voter says sax is
        plausible; it should not re-promote a sax candidate when measured facts
        and the layered physics branch point to synth pad, keys, or voice.
        """
        if facts is None:
            return False
        layer = (
            facts.evidence.get("physics_layer_decision") if isinstance(getattr(facts, "evidence", None), dict) else {}
        )
        if not isinstance(layer, dict):
            layer = {}
        branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
        woodwind_source = bool(
            layer.get("instrument_woodwind_source_signal") or layer.get("instrument_clean_tonal_reed_solo_signal")
        )
        woodwind_branch = _safe_float(layer.get("instrument_branch_Woodwinds"))
        _safe_float(layer.get("instrument_branch_KeysPiano"))
        _safe_float(layer.get("instrument_branch_Synth"))
        voice_branch = _safe_float(layer.get("instrument_branch_Voice"))
        selected_confidence = _safe_float(layer.get("instrument_branch_selected_confidence"))
        compound_strength = _safe_float(layer.get("compound_music_strength"))
        selected_subpanel = str(layer.get(f"instrument_{branch}_subpanel_selected") or "")
        selected_subpanel_confidence = _safe_float(layer.get(f"instrument_{branch}_subpanel_confidence"))
        selected_subpanel_margin = _safe_float(layer.get(f"instrument_{branch}_subpanel_margin"))
        close_sax_physics_candidate = False
        if isinstance(getattr(facts, "evidence", None), dict):
            physics_vote = facts.evidence.get("physics_vote_result")
            guesses = physics_vote.get("top_guesses") if isinstance(physics_vote, dict) else None
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
                    path = _norm_path(str(guess.get("folder_path") or guess.get("label") or ""))
                    if (
                        path.startswith("instruments/")
                        and _path_has_any(path, SAX_REED_FRAGMENTS)
                        and best_score is not None
                        and score <= best_score + 0.10
                    ):
                        close_sax_physics_candidate = True
                        break
        strong_non_woodwind_subpanel = bool(
            selected_subpanel and selected_subpanel_confidence >= 0.76 and selected_subpanel_margin >= 0.075
        )
        measured_branch_identity_conflict = bool(
            not woodwind_source
            and not (branch == "Brass" and close_sax_physics_candidate)
            and branch
            and branch not in {"Woodwinds", "ReedWoodwind"}
            and selected_confidence >= 0.70
            and (
                strong_non_woodwind_subpanel
                or (branch == "MixedInstrument" and compound_strength >= 0.54)
                or (
                    branch in {"MalletBell", "Synth", "KeysPiano", "Strings", "Brass", "Bass"}
                    and selected_confidence >= max(0.80, woodwind_branch + 0.02)
                )
            )
        )
        pitch_conf = max(
            _shape_metric_from_facts(facts, "pitch_confidence"),
            _feature_number_from_facts(facts, "pitch_confidence"),
        )
        pitched_event = max(
            _shape_metric_from_facts(facts, "pitched_event_ratio"),
            _feature_number_from_facts(facts, "loop_pitched_event_ratio"),
        )
        sustained = max(
            _shape_metric_from_facts(facts, "sustained_tonal_frame_ratio"),
            _feature_number_from_facts(facts, "loop_sustained_tonal_frame_ratio"),
        )
        low_event = _shape_metric_from_facts(facts, "low_event_ratio")
        if low_event <= 0.0:
            low_event = max(
                _feature_number_from_facts(facts, "loop_mean_event_low_ratio"),
                _feature_number_from_facts(facts, "low_event_ratio"),
            )
        mid_event = _shape_metric_from_facts(facts, "mid_event_ratio")
        if mid_event <= 0.0:
            mid_event = max(
                _feature_number_from_facts(facts, "mid_event_ratio"),
                _feature_number_from_facts(facts, "mid_ratio_500_2000hz"),
            )
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        if high_event <= 0.0:
            high_event = max(
                _feature_number_from_facts(facts, "loop_mean_event_high_ratio"),
                _feature_number_from_facts(facts, "high_event_ratio"),
            )
        flatness = max(
            _shape_metric_from_facts(facts, "spectral_flatness_mean"),
            _feature_number_from_facts(facts, "spectral_flatness_mean"),
            _feature_number_from_facts(facts, "body_flatness"),
            _feature_number_from_facts(facts, "tail_flatness"),
        )
        presence = _feature_number_from_facts(facts, "presence_ratio_2000_8000hz")
        air = _feature_number_from_facts(facts, "air_ratio_gt_8000hz")
        noise = max(
            _feature_number_from_facts(facts, "body_noise_ratio"),
            _feature_number_from_facts(facts, "tail_noise_ratio"),
        )
        percussive = max(
            _shape_metric_from_facts(facts, "percussive_event_ratio"),
            _feature_number_from_facts(facts, "loop_percussive_event_ratio"),
        )
        return bool(
            (
                shape_name in {"pitched_phrase", "repeated_phrase_loop", "solo_phrase", "sustained_pad", "bass_phrase"}
                and pitch_conf >= 0.50
                and max(pitched_event, sustained) >= 0.88
                and 0.18 <= low_event <= 0.70
                and mid_event >= 0.42
                and high_event <= 0.045
                and (presence + air) <= 0.045
                and flatness <= 0.085
                and noise <= 0.28
                and percussive <= 0.12
            )
            or (
                shape_name in {"pitched_phrase", "vocal_phrase", "repeated_phrase_loop", "sustained_pad", "bass_phrase"}
                and pitch_conf >= 0.56
                and max(pitched_event, sustained) >= 0.88
                and low_event >= 0.48
                and high_event <= 0.020
                and (presence + air) <= 0.022
                and flatness <= 0.095
                and percussive <= 0.12
            )
            or (
                shape_name in {"pitched_phrase", "vocal_phrase"}
                and pitch_conf >= 0.62
                and max(pitched_event, sustained) >= 0.88
                and low_event <= 0.03
                and mid_event >= 0.62
                and high_event >= 0.20
                and flatness <= 0.16
                and _feature_number_from_facts(facts, "fundamental_dominance_ratio") >= 0.45
                and percussive <= 0.12
            )
            or (
                not woodwind_source
                and branch == "MixedInstrument"
                and compound_strength >= 0.58
                and max(pitched_event, sustained) >= 0.88
                and high_event <= 0.09
                and _shape_metric_from_facts(facts, "instrument_plus_fx_loop_score") >= 0.54
            )
            or (
                not woodwind_source
                and branch in {"KeysPiano", "Synth"}
                and selected_confidence >= woodwind_branch + 0.12
                and max(pitched_event, sustained) >= 0.86
                and high_event <= 0.18
                and flatness <= 0.12
            )
            or (
                not woodwind_source
                and shape_name == "vocal_phrase"
                and voice_branch >= 0.60
                and voice_branch >= woodwind_branch - 0.02
            )
            or (
                not woodwind_source
                and shape_name == "vocal_phrase"
                and branch == "MalletBell"
                and voice_branch >= 0.58
                and high_event >= 0.28
            )
            or (
                not woodwind_source
                and shape_name == "vocal_phrase"
                and branch == "Woodwinds"
                and low_event <= 0.22
                and mid_event >= 0.64
                and 0.08 <= high_event <= 0.18
                and flatness >= 0.20
                and noise >= 0.32
                and _feature_number_from_facts(facts, "harmonic_energy_ratio") <= 0.48
            )
            or (
                not woodwind_source
                and shape_name == "vocal_phrase"
                and pitch_conf >= 0.72
                and _shape_metric_from_facts(facts, "f0_voiced_ratio") >= 0.75
                and max(pitched_event, sustained) >= 0.92
                and low_event <= 0.46
                and mid_event >= 0.55
                and high_event <= 0.030
                and percussive <= 0.08
                and _feature_number_from_facts(facts, "harmonic_energy_ratio") <= 0.42
            )
            or measured_branch_identity_conflict
        )

    @staticmethod
    def _facts_support_measured_reed_parent_loop(
        *,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when measured facts support a broad reed/woodwind parent.

        This is deliberately broader than the sax-leaf rescue and narrower than
        generic Instrument Loops.  It handles cases where the role detector says
        reed-like musical loop but the candidate set does not expose a safe sax
        terminal.  It should not force Saxophone; it only keeps the sample out
        of Synth Pluck or unrelated generic loop buckets.
        """
        if facts is None:
            return False
        if role_name not in {"pitched_reed_or_instrument_loop", "pitched_reed_or_instrument_phrase"}:
            return False
        if shape_name not in {"pitched_phrase", "sustained_pad", "vocal_phrase"} or shape_confidence < 0.72:
            return False
        if ProfileInstrumentClaimMixin._facts_support_short_true_voice_one_shot(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        ):
            return False
        pitched_event = _shape_metric_from_facts(facts, "pitched_event_ratio")
        voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        percussive_event = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        return bool(pitched_event >= 0.70 and voiced >= 0.70 and percussive_event <= 0.12 and drumlike <= 0.16)

    def _measured_reed_parent_claim(
        self,
        *,
        raw: ConsensusClaim,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        support_path: str,
        support_score: float,
        rank: int | None,
    ) -> ConsensusClaim:
        """Build a broad Brass/Woodwinds claim for measured reed-like loops."""
        folder = "Instruments/Brass and Woodwinds/Loops"
        return claim_from_folder_path(
            folder_path=folder,
            source="profile_candidate_measured_reed_parent_claim",
            reason=(
                "measured reed-like musical loop evidence was too specific for "
                "generic Instrument Loops but not safe enough for a sax leaf: "
                f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                f"raw={raw.folder_path}; support={support_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=support_score,
            brain_rank=rank,
            physics_rank=None,
            shared_winner=folder,
            can_override=True,
            strength=0.88,
            is_real_candidate=False,
        )

    @staticmethod
    def _facts_support_short_true_voice_one_shot(
        *,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for short one-event vocal shots that must not become sax loops.

        This is a conditional voice-vs-sax conflict probe.  It does not inspect
        filenames.  It uses measured formant/light-voice identity, voiced F0,
        event count, and wet-tail evidence.  The goal is to let processed vocal
        shots stay in Voice while still allowing long wet sax phrases to keep
        the sax leaf.
        """
        if facts is None:
            return False
        vocal_shape = bool(
            shape_name in {"vocal_phrase", "vocal_one_shot", "hit_with_tail"} and shape_confidence >= 0.84
        )
        voice_identity = max(
            _voice_identity_from_facts(facts),
            ProfileInstrumentClaimMixin._formant_voice_identity_from_facts(facts),
        )
        voiced_ratio = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        onset_count = _shape_metric_from_facts(facts, "onset_count")
        onset_span = _shape_metric_from_facts(facts, "onset_span_ratio")
        pulse = _shape_metric_from_facts(facts, "pulse_regularity")
        tail_ratio = _shape_metric_from_facts(facts, "tail_ratio")
        duration = _feature_number_from_facts(facts, "duration_sec")
        measured_voice = max(
            _role_strength_from_facts(facts, "vocal_one_shot"),
            _role_strength_from_facts(facts, "voiced_one_shot"),
            _role_strength_from_facts(facts, "vocal_phrase"),
            _role_strength_from_facts(facts, "vocal_music_phrase"),
        )
        short_or_single_event = bool(
            (duration > 0.0 and duration <= 3.25) or (onset_count <= 1.25 and onset_span <= 0.08 and pulse <= 0.12)
        )
        low_tail_single_shot = bool(tail_ratio <= 0.12 and onset_count <= 1.25)
        return bool(
            vocal_shape
            and short_or_single_event
            and low_tail_single_shot
            and voice_identity >= 0.62
            and voiced_ratio >= 0.72
            and measured_voice >= 0.30
        )

    @staticmethod
    def _facts_support_sax_like_reed_loop(
        *,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for wet/broadband sustained reed-loop body evidence.

        This is not filename evidence.  It catches sax/reed loops whose reverb,
        delay, and formant structure make them look like voice, keys, harmonica,
        or generic instrument loops.  It deliberately rejects narrow mid-only
        synth leads.
        """
        if facts is None:
            return False
        if ProfileInstrumentClaimMixin._facts_support_short_true_voice_one_shot(
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        ):
            return False
        stable = bool(
            role_name in STABLE_PITCHED_ROLES
            or (shape_name in {"pitched_phrase", "sustained_pad", "vocal_phrase"} and shape_confidence >= 0.74)
        )
        if (
            shape_name == "vocal_phrase"
            and role_name not in {"pitched_reed_or_instrument_loop", "pitched_reed_or_instrument_phrase"}
            and ProfileInstrumentClaimMixin._formant_voice_identity_from_facts(facts) >= 0.42
        ):
            return False
        if not stable:
            return False
        low_event = _shape_metric_from_facts(facts, "low_event_ratio")
        mid_event = _shape_metric_from_facts(facts, "mid_event_ratio")
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        entropy = _shape_metric_from_facts(facts, "spectral_entropy_mean")
        flatness = _shape_metric_from_facts(facts, "spectral_flatness_mean")
        pitched_event = _shape_metric_from_facts(facts, "pitched_event_ratio")
        voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        percussive_event = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        if pitched_event < 0.70 or voiced < 0.70:
            return False
        if percussive_event > 0.12 or drumlike > 0.16:
            return False
        if low_event < 0.07 or mid_event < 0.25:
            return False
        narrow_synth_mid_body = bool(low_event <= 0.04 and high_event <= 0.07 and entropy <= 0.28 and flatness <= 0.06)
        if narrow_synth_mid_body:
            return False

        # Piano/keys loops can be stable, pitched, wet, and mid-bodied enough
        # to look reed-like to the shape lane.  Do not call that sax unless
        # there is some high-band breath/noise or broader noisy flatness.
        low_air_clean_tonal_body = bool(high_event < 0.045 and flatness < 0.025)
        if low_air_clean_tonal_body:
            return False

        return bool(entropy >= 0.30 or flatness >= 0.08 or high_event >= 0.08)

    @staticmethod
    def _facts_have_measured_drum_loop_authority(facts: SharedAudioFacts | None) -> bool:
        """Return True when measured role/shape evidence already supports a drum loop.

        Profile candidate claims are allowed to use candidate output, but they
        should not convert a measured low-rhythmic drum loop into Synth merely
        because the loop is pitched or bass-heavy.  This reads only measured
        audio facts, not source names.
        """
        if facts is None:
            return False
        measured_drum_loop = max(
            _role_strength_from_facts(facts, "low_rhythmic_drum_loop"),
            _role_strength_from_facts(facts, "percussive_drum_loop"),
            _role_strength_from_facts(facts, "bright_drum_loop"),
            _feature_number_from_facts(facts, "drum_loop_source_score"),
        )
        onset_count = _shape_metric_from_facts(facts, "onset_count")
        true_repetition = _shape_metric_from_facts(facts, "true_repetition_score")
        low_event = _shape_metric_from_facts(facts, "low_event_ratio")
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        return bool(
            measured_drum_loop >= 0.58
            and onset_count >= 6.0
            and (true_repetition >= 0.50 or low_event >= 0.70 or high_event >= 0.45)
        )

    @staticmethod
    def _facts_have_strong_human_voice_identity(facts: SharedAudioFacts | None) -> bool:
        """Return True when source-name-blind facts strongly support human voice.

        This guards against a synth sibling rescue stealing vocal one-shots when
        the measured voice panel is already decisive.  It deliberately requires
        both voice identity and a voiced/vocal parent role so formant-like synths
        are not blocked by one isolated metric.
        """
        voice_identity = _voice_identity_from_facts(facts)
        voice_role = max(
            _role_strength_from_facts(facts, "vocal_music_phrase"),
            _role_strength_from_facts(facts, "vocal_phrase"),
            _role_strength_from_facts(facts, "vocal_one_shot"),
            _role_strength_from_facts(facts, "voiced_one_shot"),
        )
        f0_voiced = max(
            _shape_metric_from_facts(facts, "f0_voiced_ratio"),
            _feature_number_from_facts(facts, "f0_voiced_ratio"),
        )
        percussive = max(
            _shape_metric_from_facts(facts, "percussive_event_ratio"),
            _feature_number_from_facts(facts, "loop_percussive_event_ratio"),
        )
        return bool(voice_identity >= 0.72 and voice_role >= 0.68 and f0_voiced >= 0.60 and percussive <= 0.24)

    @staticmethod
    def _facts_support_synth_lead_like_body(
        *,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for clean, non-percussive synth-lead-like loop body."""
        if facts is None:
            return False
        if shape_name != "pitched_phrase":
            return False
        stable = bool(
            role_name in STABLE_PITCHED_ROLES or (shape_name == "pitched_phrase" and shape_confidence >= 0.88)
        )
        if not stable:
            return False
        low_event = _shape_metric_from_facts(facts, "low_event_ratio")
        mid_event = _shape_metric_from_facts(facts, "mid_event_ratio")
        entropy = _shape_metric_from_facts(facts, "spectral_entropy_mean")
        flatness = _shape_metric_from_facts(facts, "spectral_flatness_mean")
        pitched_event = _shape_metric_from_facts(facts, "pitched_event_ratio")
        voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        percussive_event = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        return bool(
            low_event <= 0.05
            and mid_event >= 0.60
            and entropy <= 0.34
            and flatness <= 0.18
            and pitched_event >= 0.80
            and voiced >= 0.78
            and percussive_event <= 0.08
            and drumlike <= 0.10
        )

    @staticmethod
    def _facts_support_strong_fx_role_layer(facts: SharedAudioFacts | None) -> bool:
        """Return True when measured FX-role physics should block synth rescue.

        This does not force an FX folder.  It only prevents the Synth sibling
        claim from overriding a lower voter that has already measured a valid
        FX role such as transition motion, machine/siren motion, texture bed,
        or foley action.  Synth claims still work when the FX layer is weak or
        conflicted.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        layer = facts.evidence.get("physics_layer_decision")
        if not isinstance(layer, dict):
            return False
        branch = str(layer.get("physics_layer_branch") or layer.get("fx_branch_selected") or "")
        top_family = str(layer.get("physics_layer_top_family") or "")
        if top_family != "FX":
            return False
        if not bool(layer.get("fx_role_allows_fx")):
            return False
        strength = _safe_float(layer.get("fx_role_strength"))
        conflict_raw = layer.get("fx_role_conflict_strength")
        conflict = _safe_float(conflict_raw) if conflict_raw is not None else 1.0
        branch_confidence = _safe_float(
            layer.get("physics_layer_branch_confidence") or layer.get("fx_branch_selected_confidence")
        )
        return bool(
            branch
            and strength >= 0.58
            and branch_confidence >= 0.58
            and conflict < 0.58
            and branch
            in {
                "RiserBuild",
                "DropDownlifter",
                "WhooshSweep",
                "ReverseSwell",
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
            }
        )

    def _sax_or_reed_candidate_should_block_voice(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        voice_score: float,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when sax/reed evidence makes Voice a formant false-positive."""
        sax_leaf = self._best_sax_leaf_support(raw=raw, brain_result=brain_result, facts=facts)
        if sax_leaf is not None:
            sax_score, sax_path, sax_rank = sax_leaf
            if _path_has_any(_norm_path(sax_path), SAX_REED_FRAGMENTS):
                return bool(sax_rank <= 6 and sax_score <= voice_score + 12.0)
        reed_support = self._reed_candidate_support(
            brain_result=brain_result,
            facts=facts,
            shared_reed_score=None,
        )
        return bool(
            reed_support is not None
            and reed_support.is_strong
            and self._facts_support_sax_like_reed_loop(
                role_name=_measured_role_from_facts(facts),
                shape_name=_shape_vote_from_facts(facts),
                shape_confidence=_shape_confidence_from_facts(facts),
                facts=facts,
            )
        )

    @staticmethod
    def _wetness_score_from_facts(facts: SharedAudioFacts | None) -> float:
        """Return wetness/reverb evidence from shared facts, if available."""
        if facts is None or not isinstance(facts.evidence, dict):
            return 0.0
        wetness = facts.evidence.get("wetness_profile")
        if isinstance(wetness, dict):
            try:
                return float(wetness.get("wetness_score", 0.0) or 0.0)
            except Exception:
                return 0.0
        return 0.0

    @staticmethod
    def _candidate_role_distance_from_shared_row(row: dict | None) -> float:
        """Read role-distance evidence from a shared candidate row."""
        if not isinstance(row, dict):
            return 9999.0
        for key in ("candidate_role_distance",):
            try:
                return float(row.get(key, 9999.0) or 9999.0)
            except Exception:
                pass
        for evidence_key in ("brain_evidence", "physics_evidence", "evidence"):
            evidence = row.get(evidence_key)
            if isinstance(evidence, dict):
                try:
                    value = evidence.get("candidate_role_distance")
                    if value is not None:
                        return float(value)
                except Exception:
                    continue
        return 9999.0

    def _sax_leaf_is_close_enough_to_block_synth(
        self,
        *,
        raw: ConsensusClaim,
        sax_leaf: tuple[float, str, int] | None,
        shared_synth: tuple[float, dict] | None,
        best_guess: object | None,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when a wet sax/reed witness should stop synth theft."""
        if sax_leaf is None:
            return False
        sax_score, sax_path, sax_rank = sax_leaf
        if not _path_has_any(_norm_path(sax_path), SAX_REED_FRAGMENTS):
            return False
        if sax_rank > 8 and sax_score > 28.0:
            return False
        synth_score = shared_synth[0] if shared_synth is not None else 9999.0
        synth_rank = int(getattr(best_guess, "rank", 999)) if best_guess is not None else 999
        sax_row = None
        synth_row = shared_synth[1] if shared_synth is not None else None
        for row in raw.shared_candidates or []:
            path = str(row.get("folder_path") or row.get("label") or "")
            if _path_has_any(_norm_path(path), SAX_REED_FRAGMENTS):
                sax_row = row
                break
        sax_role_distance = self._candidate_role_distance_from_shared_row(sax_row)
        synth_role_distance = self._candidate_role_distance_from_shared_row(synth_row)
        stable_body = self._measured_has_stable_or_staggered_music_body(facts)
        wetness = self._wetness_score_from_facts(facts)
        close_rank_score = sax_score <= synth_score + 8.0
        close_rank = sax_rank <= max(8, synth_rank + 5)
        better_role_fit = sax_role_distance <= synth_role_distance + 0.04
        return bool(stable_body and close_rank_score and close_rank and (better_role_fit or wetness >= 0.55))

    @staticmethod
    def _raw_is_protected_non_reed_instrument_branch(
        *,
        raw_path: str,
        raw: ConsensusClaim,
        raw_is_generic_instrument: bool,
        raw_is_reed: bool,
    ) -> bool:
        """Return True when a reed rescue would steal a concrete sibling branch."""
        if raw.final_top != "Instruments" or raw_is_generic_instrument or raw_is_reed:
            return False
        return _path_has_any(raw_path, PROTECTED_NON_REED_INSTRUMENT_BRANCH_FRAGMENTS)

    @staticmethod
    def _physics_top_is_instrument(physics_result: VoterResult | None) -> bool:
        """Return True when physics is at least pointing at Instruments."""
        if physics_result is None or not physics_result.guesses:
            return False
        return str(physics_result.guesses[0].top_family or "") == "Instruments"

    @staticmethod
    def _measured_has_stable_music_body(facts: SharedAudioFacts | None) -> bool:
        """Return True for sustained pitched musical body evidence."""
        return bool(
            _feature_number_from_facts(facts, "loop_pitched_event_ratio") >= 0.86
            and _feature_number_from_facts(facts, "loop_sustained_tonal_frame_ratio") >= 0.74
            and _feature_number_from_facts(facts, "loop_tonal_to_percussive_balance") >= 0.55
        )

    @staticmethod
    def _raw_mallet_bell_lacks_struck_event_body(facts: SharedAudioFacts | None) -> bool:
        """Return True when a Mallet/Bell raw winner looks synth-like.

        This protects future mallet/bell categories.  Synth may challenge a
        mallet/bell raw winner only when the measured audio has sustained tonal
        body, very weak percussive/high-event content, and no strong drumlike
        frame population.  In other words, this catches pad/lead/arp material
        that physics temporarily matched to bell-like spectra, without turning
        every real vibraphone, bell roll, or struck mallet loop into Synth.
        """
        if facts is None:
            return False
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        percussive_event = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        sustain = _shape_metric_from_facts(facts, "sustain_ratio")
        voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        pitched_events = _shape_metric_from_facts(facts, "pitched_event_ratio")
        stable_body = ProfileInstrumentClaimMixin._measured_has_stable_or_staggered_music_body(facts)
        return bool(
            stable_body
            and sustain >= 0.62
            and voiced >= 0.70
            and pitched_events >= 0.70
            and high_event <= 0.08
            and percussive_event <= 0.08
            and drumlike <= 0.10
        )

    @staticmethod
    def _raw_mallet_bell_has_sustained_non_struck_pad_body(
        *,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when a Mallet/Bell winner lacks struck-instrument physics.

        Bell/mallet leaves should require some struck or percussive body.  A
        sustained, voiced, non-percussive pad-like loop may share spectral
        color with vibraphone/bells, but the safer producer-facing placement is
        the broad Instrument Loops parent until a better source identity exists.
        """
        if facts is None:
            return False
        if shape_name != "sustained_pad" or shape_confidence < 0.70:
            return False
        pitched_role = max(
            _role_strength_from_facts(facts, "pitched_music_loop"),
            _role_strength_from_facts(facts, "pitched_music_phrase"),
        )
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        percussive_event = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        sustain = _shape_metric_from_facts(facts, "sustain_ratio")
        voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        pitched_events = _shape_metric_from_facts(facts, "pitched_event_ratio")
        attack = _shape_metric_from_facts(facts, "attack_rise_time_norm")
        return bool(
            (role_name in STABLE_PITCHED_ROLES or pitched_role >= 0.74)
            and sustain >= 0.62
            and voiced >= 0.70
            and pitched_events >= 0.70
            and percussive_event <= 0.04
            and drumlike <= 0.08
            and attack <= 0.10
            and high_event <= 0.38
        )

    @staticmethod
    def _raw_plucked_string_lacks_pluck_body(facts: SharedAudioFacts | None) -> bool:
        """Return True when a plucked/string raw winner lacks pluck evidence.

        This is intentionally narrow.  It does not say Synth beats Strings.  It
        says a Synth candidate may compete when the supposed plucked/string raw
        winner has sustained, non-percussive, fully voiced body evidence and no
        measurable high/percussive event population.
        """
        if facts is None:
            return False
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        percussive_event = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        attack = _shape_metric_from_facts(facts, "attack_rise_time_norm")
        sustain = _shape_metric_from_facts(facts, "sustain_ratio")
        voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        pitched_events = _shape_metric_from_facts(facts, "pitched_event_ratio")
        stable_body = ProfileInstrumentClaimMixin._measured_has_stable_or_staggered_music_body(facts)
        return bool(
            stable_body
            and sustain >= 0.68
            and voiced >= 0.78
            and pitched_events >= 0.78
            and attack >= 0.24
            and high_event <= 0.08
            and percussive_event <= 0.08
            and drumlike <= 0.10
        )

    @staticmethod
    def _measured_has_stable_or_staggered_music_body(facts: SharedAudioFacts | None) -> bool:
        """Return True for stable pitched music in full or direct/body facts."""
        if ProfileInstrumentClaimMixin._measured_has_stable_music_body(facts):
            return True
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        direct_view = facts.evidence.get("direct_body_view")
        if not isinstance(direct_view, dict) or not direct_view.get("available"):
            return False
        groups = direct_view.get("feature_groups")
        if not isinstance(groups, dict):
            return False
        loop_group = groups.get("loop_long_population")
        if not isinstance(loop_group, dict):
            return False

        def value(name: str) -> float:
            try:
                return float(loop_group.get(name, 0.0) or 0.0)
            except Exception:
                return 0.0

        return bool(
            value("loop_pitched_event_ratio") >= 0.45
            and value("loop_sustained_tonal_frame_ratio") >= 0.72
            and value("loop_tonal_to_percussive_balance") >= 0.62
        )

    @staticmethod
    def _full_brain_top_is_non_voice_instrument(facts: SharedAudioFacts | None) -> bool:
        """Return True when the full brain top is a non-voice instrument.

        Product/ensemble voice support is allowed to protect real vocals, but
        it should not block a broad Instrument Loops rescue when the full
        brain's nearest concrete candidate is an instrument branch and the
        measured facts lack true-voice identity.
        """
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        result = facts.evidence.get("full_brain_vote_result")
        if not isinstance(result, dict):
            return False
        guesses = result.get("top_guesses")
        if not isinstance(guesses, list) or not guesses or not isinstance(guesses[0], dict):
            return False
        folder_path = (
            str(guesses[0].get("folder_path") or guesses[0].get("path") or guesses[0].get("label") or "")
            .replace("\\", "/")
            .lower()
        )
        if not folder_path or not folder_path.startswith("instruments/"):
            return False
        return not _path_has_any(folder_path, VOICE_FRAGMENTS)

    @staticmethod
    def _raw_voice_has_true_voice_support(raw_path: str, facts: SharedAudioFacts | None) -> bool:
        """Return True when a raw voice leaf has measured true-voice support."""
        if not _path_has_any(raw_path, VOICE_FRAGMENTS):
            return False
        voice_strength = max(
            _role_strength_from_facts(facts, "vocal_music_phrase"),
            _role_strength_from_facts(facts, "vocal_phrase"),
            _role_strength_from_facts(facts, "vocal_one_shot"),
            _role_strength_from_facts(facts, "voiced_one_shot"),
        )
        if voice_strength >= 0.78:
            return True
        voice_identity = _voice_identity_from_facts(facts)
        if voice_identity >= 0.68:
            return True
        # Some real rap/sung voice loops measure as generic pitched or bass
        # phrases rather than the vocal parent role.  Do not demote a Voice
        # branch on voiced+sustained evidence alone, though: synths and reeds
        # can also be highly voiced.  Require at least some voice-specific
        # identity before preserving the leaf.
        f0_voiced = _shape_metric_from_facts(facts, "f0_voiced_ratio")
        sustain = _shape_metric_from_facts(facts, "sustain_ratio")
        percussive = _shape_metric_from_facts(facts, "percussive_event_ratio")
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        if (
            f0_voiced >= 0.70
            and sustain >= 0.55
            and percussive <= 0.16
            and high_event <= 0.18
            and voice_identity >= 0.45
        ):
            return True
        return bool(voice_identity >= 0.68 and f0_voiced >= 0.72)

    def _voice_claim_is_blocked_by_reed_lane_support(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when reed/wind lanes make Voice a likely formant false-positive.

        Sax and reed instruments can look voice-like in formant features.  The
        measured role names the broader physical role, while the brain lanes
        provide independent source-family witnesses.  When the role is already
        reed/instrument and multiple lanes see reed/wind material, Voice should
        not steal the placement just because a shared Voice candidate is close.
        """
        reed_support = self._reed_candidate_support(
            brain_result=brain_result,
            facts=facts,
            shared_reed_score=None,
        )
        if reed_support is None or not reed_support.is_strong:
            return False
        shared_voice = self.best_shared_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=VOICE_FRAGMENTS,
        )
        shared_voice_score = shared_voice[0] if shared_voice is not None else 9999.0
        return bool(
            reed_support.lane_count >= 2
            or reed_support.has_primary_lane_support
            or (reed_support.has_product_support and shared_voice_score >= 12.0)
        )

    @staticmethod
    def _raw_voice_has_primary_product_brain_support(
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
    ) -> bool:
        """Return True when the actual product brain's top read is Voice.

        This protects real rap/sung voice loops whose measurement can look like
        a generic pitched loop after reverb, compression, or sparse phrasing.
        It does not preserve arbitrary Voice false positives: the product brain
        itself must put Voice at the top, and the shared consensus must already
        have the Voice leaf near the front.
        """
        if not brain_result.guesses:
            return False
        top_guess = brain_result.guesses[0]
        top_path = _norm_path(str(top_guess.folder_path or top_guess.label or ""))
        if not _path_has_any(top_path, VOICE_FRAGMENTS):
            return False
        try:
            raw_score = float(raw.raw_candidate_score or 9999.0)
        except Exception:
            raw_score = 9999.0
        try:
            support = float(top_guess.evidence.get("support", top_guess.evidence.get("ensemble_support", 0.0)) or 0.0)
        except Exception:
            support = 0.0
        try:
            confidence = float(getattr(top_guess, "confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        try:
            score = float(getattr(top_guess, "score", 9999.0) or 9999.0)
        except Exception:
            score = 9999.0
        decisive_top_voice = bool(confidence >= 0.72 and (support >= 2.20 or score <= 0.65))
        return bool(raw_score <= 10.0 or (raw_score <= 16.0 and decisive_top_voice))

    def _raw_voice_has_rank_one_brain_and_body_support(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when a raw Voice claim has rank-one brain/body support.

        This is a source-name-blind protection for processed vocal loops whose
        measured role looks like a generic pitched or designed tonal phrase.
        The product brain must rank Voice first, and measured voice/formant body
        evidence must be present without hard drum or transition-FX proof.
        """
        if not brain_result.guesses:
            return False
        top_guess = brain_result.guesses[0]
        top_path = _norm_path(str(top_guess.folder_path or top_guess.label or ""))
        if not _path_has_any(top_path, VOICE_FRAGMENTS):
            return False
        try:
            support = float(top_guess.evidence.get("support", top_guess.evidence.get("ensemble_support", 0.0)) or 0.0)
        except Exception:
            support = 0.0
        try:
            confidence = float(getattr(top_guess, "confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        try:
            score = float(getattr(top_guess, "score", 9999.0) or 9999.0)
        except Exception:
            score = 9999.0
        if confidence < 0.72 or (support < 2.20 and score > 0.65):
            return False
        try:
            raw_score = float(raw.raw_candidate_score or 9999.0)
        except Exception:
            raw_score = 9999.0
        if raw_score > 16.0:
            return False
        voice_body = max(
            self._score_from_facts(facts, "voice_score"),
            self._score_from_facts(facts, "human_spoken_voice_score"),
            self._score_from_facts(facts, "human_breath_mouth_score"),
            self._score_from_facts(facts, "voice_choir_score"),
            self._score_from_facts(facts, "fx_formant_score"),
        )
        vocal_role = max(
            _role_strength_from_facts(facts, "vocal_music_phrase"),
            _role_strength_from_facts(facts, "voiced_one_shot"),
        )
        if _direct_voice_source_score_from_facts(facts) < 0.62 and vocal_role < 0.40:
            return False
        if _voice_claim_has_tonal_instrument_conflict(facts):
            return False
        hard_drum = max(
            self._score_from_facts(facts, "drum_hit_score"),
            self._score_from_facts(facts, "drum_loop_source_score"),
            self._score_from_facts(facts, "drum_kick_source_score"),
            self._score_from_facts(facts, "drum_snare_source_score"),
            self._score_from_facts(facts, "drum_clap_source_score"),
        )
        transition_fx = max(
            self._score_from_facts(facts, "fx_motion_score"),
            self._score_from_facts(facts, "fx_transition_authority_score"),
            self._score_from_facts(facts, "fx_riser_build_score"),
            self._score_from_facts(facts, "fx_whoosh_sweep_score"),
        )
        return bool(voice_body >= 0.42 and hard_drum <= 0.58 and transition_fx <= 0.62)

    @staticmethod
    def _int_from_row(value: object) -> int | None:
        """Return an int row value for claim diagnostics."""
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return None

    @staticmethod
    def _shared_candidate_voice_strength(row: dict) -> float:
        """Return vocal-role strength encoded on a shared candidate row."""
        signatures = []
        direct = row.get("candidate_role_signature")
        if isinstance(direct, dict):
            signatures.append(direct)
        for key in ("brain_evidence", "physics_evidence", "evidence"):
            evidence = row.get(key)
            if isinstance(evidence, dict):
                sig = evidence.get("candidate_role_signature")
                if isinstance(sig, dict):
                    signatures.append(sig)
                strengths = evidence.get("measured_role_strengths")
                if isinstance(strengths, dict):
                    signatures.append(strengths)
        best = 0.0
        for signature in signatures:
            for role in ("vocal_music_phrase", "vocal_phrase", "vocal_one_shot", "voiced_one_shot"):
                try:
                    best = max(best, float(signature.get(role, 0.0) or 0.0))
                except Exception:
                    continue
        return best

    @staticmethod
    def _formant_voice_identity_from_facts(facts: SharedAudioFacts | None) -> float:
        """Return measured formant-like voice identity from shared facts."""
        if facts is None or not isinstance(facts.evidence, dict):
            return 0.0
        roles = facts.evidence.get("measured_roles")
        if isinstance(roles, dict):
            evidence = roles.get("evidence")
            if isinstance(evidence, dict):
                try:
                    return float(evidence.get("formant_light_voice_identity", 0.0) or 0.0)
                except Exception:
                    return 0.0
        return 0.0

    @staticmethod
    def _shape_or_role_supports_stable_music(
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for stable pitched instrument loop/phrase evidence."""
        return bool(
            role_name in STABLE_PITCHED_ROLES
            or (shape_name in STABLE_PITCHED_SHAPES and shape_confidence >= 0.70)
            or ProfileInstrumentClaimMixin._measured_has_stable_or_staggered_music_body(facts)
        )

    @staticmethod
    def _instrument_branch_group(path: str) -> str:
        """Return broad sibling branch name for an instrument path."""
        normalized = _norm_path(path)
        if "instrument loops" in normalized:
            return "generic"
        if "mixed musical" in normalized or "multi instrument" in normalized:
            return "mixed"
        if _path_has_any(normalized, BASS_FRAGMENTS):
            return "bass"
        if _path_has_any(normalized, REED_FRAGMENTS):
            return "reed"
        if _path_has_any(normalized, SYNTH_FRAGMENTS):
            return "synth"
        if _path_has_any(normalized, MALLET_BELL_FRAGMENTS):
            return "mallet_bell"
        if _path_has_any(normalized, KEYS_FRAGMENTS):
            return "keys"
        if _path_has_any(normalized, STRING_GUITAR_FRAGMENTS):
            return "string_guitar"
        if _path_has_any(normalized, VOICE_FRAGMENTS):
            return "voice"
        if "world and special" in normalized or "harmonica" in normalized or "accordion" in normalized:
            return "world_special"
        return ""

    def _close_instrument_sibling_groups(
        self,
        raw: ConsensusClaim,
        *,
        raw_group: str,
        raw_score: float,
    ) -> list[tuple[str, float, str]]:
        """Return close competing instrument branch groups from shared rows."""
        groups: dict[str, tuple[float, str]] = {}
        for row in raw.shared_candidates or []:
            path = str(row.get("folder_path") or row.get("label") or "")
            if str(row.get("top_family") or "Instruments") != "Instruments" and not path.startswith("Instruments/"):
                continue
            group = self._instrument_branch_group(path)
            if group in {"", "generic", "mixed", "bass", raw_group}:
                continue
            try:
                score = float(row.get("combined_rank_score", 9999.0) or 9999.0)
            except Exception:
                score = 9999.0
            try:
                brain_rank = int(row.get("brain_rank", 999) or 999)
            except Exception:
                brain_rank = 999
            close_by_score = score <= raw_score + 8.0
            strong_brain_witness = brain_rank <= 3 and score <= raw_score + 18.0
            if not (close_by_score or strong_brain_witness):
                continue
            if group not in groups or score < groups[group][0]:
                groups[group] = (score, path)
        return [(group, score, path) for group, (score, path) in sorted(groups.items(), key=lambda item: item[1][0])]

    @staticmethod
    def _raw_has_direct_terminal_identity(
        *,
        raw: ConsensusClaim,
        raw_path: str,
        raw_group: str,
        raw_score: float,
    ) -> bool:
        """Return True when brain and physics both directly support the raw leaf.

        The sibling-conflict parent exists to avoid brittle terminal guesses.
        It should not erase a clean terminal identity where both primary voters
        already agree, or where one primary voter is first and the other is
        still close.  This is branch-agnostic: it protects Rhodes, Saxophone,
        Horns, Synth Pad, and other concrete leaves by the same evidence rule.
        """
        normalized_raw = _norm_path(raw_path)
        if not normalized_raw or raw_group in {"generic", "mixed", "mallet_bell"}:
            return False
        for row in raw.shared_candidates or []:
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            if path != normalized_raw:
                continue
            try:
                score = float(row.get("combined_rank_score", raw_score) or raw_score)
            except Exception:
                score = raw_score
            try:
                brain_rank = int(row.get("brain_rank", 999) or 999)
            except Exception:
                brain_rank = 999
            try:
                physics_rank = int(row.get("physics_rank", 999) or 999)
            except Exception:
                physics_rank = 999
            return bool(
                score <= 3.0
                and brain_rank <= 2
                and physics_rank <= 2
                or (score <= 6.0 and brain_rank <= 2 and physics_rank <= 5 and raw_group in {"reed", "keys", "synth"})
                or (raw_group in {"reed", "keys", "synth"} and score <= 12.0 and brain_rank <= 5 and physics_rank <= 5)
                or (raw_group in {"keys", "synth"} and score <= 16.0 and brain_rank <= 3 and physics_rank <= 12)
            )
        return False

    def _best_protected_sibling_support(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        shared_reed_score: float,
    ) -> InstrumentSiblingSupport | None:
        """Return best non-reed sibling support that can block reed vacuum claims."""
        fragments = SYNTH_FRAGMENTS + KEYS_FRAGMENTS + MALLET_BELL_FRAGMENTS + STRING_GUITAR_FRAGMENTS
        shared = self.best_shared_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=fragments,
        )
        brain_guess = self.best_brain_guess(
            brain_result,
            fragments=fragments,
            max_rank=8,
        )
        if shared is None and brain_guess is None:
            return None
        shared_score = shared[0] if shared is not None else None
        rank = int(getattr(brain_guess, "rank", 999)) if brain_guess is not None else 999
        path = ""
        if shared is not None:
            path = str(shared[1].get("folder_path") or shared[1].get("label") or "")
        if not path and brain_guess is not None:
            path = str(brain_guess.folder_path or brain_guess.label or "")
        return InstrumentSiblingSupport(path=path, rank=rank, shared_score=shared_score)

    @staticmethod
    def _sibling_support_blocks_reed_claim(
        *,
        sibling_support: InstrumentSiblingSupport | None,
        reed_support: ReedCandidateSupport | None,
        specific_reed_role: bool,
        shared_reed_score: float,
    ) -> bool:
        """Return True when another instrument sibling should stop reed routing.

        This is the second half of the anti-vacuum rule.  It catches cases where
        the raw winner is generic or voice/mallet/FX, but the candidate list has
        strong synth/keys/strings support competing with a weaker reed rescue.
        """
        if sibling_support is None or not sibling_support.is_present:
            return False
        sibling_score = sibling_support.shared_score if sibling_support.shared_score is not None else 9999.0
        reed_rank = reed_support.rank if reed_support is not None else 999
        if specific_reed_role:
            return bool(sibling_score <= shared_reed_score - 1.5 and sibling_support.rank <= reed_rank)
        return bool(sibling_score <= shared_reed_score + 2.0 or sibling_support.rank <= max(2, reed_rank))

    @staticmethod
    def _preferred_folder_path_from_shared_or_brain(
        *,
        shared_candidate: tuple[float, dict] | None,
        best_guess: object | None,
        prefer_brain_when_rank_at_most: int,
    ) -> str:
        """Choose the concrete internal candidate path for a profile claim."""
        if best_guess is not None and int(getattr(best_guess, "rank", 999)) <= prefer_brain_when_rank_at_most:
            path = str(getattr(best_guess, "folder_path", "") or getattr(best_guess, "label", ""))
            if path:
                return path
        if shared_candidate is not None:
            row = shared_candidate[1]
            path = str(row.get("folder_path") or row.get("label") or "")
            if path:
                return path
        if best_guess is not None:
            return str(getattr(best_guess, "folder_path", "") or getattr(best_guess, "label", ""))
        return "Instruments/Instrument Loops/Loops"

    def _reed_candidate_support(
        self,
        *,
        brain_result: VoterResult,
        facts: SharedAudioFacts | None,
        shared_reed_score: float | None,
    ) -> ReedCandidateSupport | None:
        """Return the best reed/wind witness from product and lane voters."""
        best_path = ""
        best_rank = 999
        lanes: set[str] = set()
        has_product = False
        for guess in brain_result.guesses[:8]:
            path = _norm_path(str(guess.folder_path or guess.label or ""))
            if not _path_has_any(path, REED_FRAGMENTS):
                continue
            has_product = True
            lanes.add("product")
            if guess.rank < best_rank:
                best_rank = int(guess.rank)
                best_path = str(guess.folder_path or guess.label or "")
            break

        if facts is not None and isinstance(facts.evidence, dict):
            for key, lane in (
                ("full_brain_vote_result", "full"),
                ("core_baby_vote_result", "core_baby"),
                ("spread_baby_vote_result", "spread_baby"),
                ("outlier_baby_vote_result", "outlier_baby"),
            ):
                result = facts.evidence.get(key)
                if not isinstance(result, dict):
                    continue
                guesses = result.get("top_guesses")
                if not isinstance(guesses, list):
                    continue
                for idx, guess in enumerate(guesses[:5], start=1):
                    if not isinstance(guess, dict):
                        continue
                    path = _norm_path(str(guess.get("folder_path") or guess.get("label") or ""))
                    if not _path_has_any(path, REED_FRAGMENTS):
                        continue
                    lanes.add(lane)
                    if idx < best_rank:
                        best_rank = idx
                        best_path = str(guess.get("folder_path") or guess.get("label") or "")
                    break

        if not lanes and shared_reed_score is None:
            return None
        return ReedCandidateSupport(
            path=best_path or "Instruments/Brass and Woodwinds/Loops",
            rank=best_rank,
            lane_count=len(lanes),
            has_product_support=has_product,
            has_primary_lane_support="full" in lanes,
            shared_score=shared_reed_score,
        )

    def mixed_musical_loop_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        physics_result: VoterResult | None,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_path: str,
        raw_score: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Return a broad mixed-instrument-loop role claim.

        The first branch preserves the existing direct mixed-musical-loop
        candidate behavior.  The second branch is the v31114 architecture step:
        when the measured role is a musical loop, PhysicsVoter exposes broad
        Instrument Loops or a mixed-instrument branch, and source identity is
        split across three or more instrument branches, the safe result is the
        broad Instrument Loops role bucket.  This avoids letting Sax, Voice,
        Keys, Guitar, or Synth steal a compound loop when none owns the source
        identity by itself.
        """
        if raw.final_top != "Instruments":
            return None
        if _path_has_any(raw_path, ("mixed musical", "multi instrument")):
            return None
        if _path_has_any(raw_path, BASS_FRAGMENTS) and raw_score <= 8.0:
            return None
        if not self._shape_or_role_supports_stable_music(
            role_name,
            shape_name,
            shape_confidence,
            facts,
        ):
            return None
        if self._facts_have_measured_drum_loop_authority(facts):
            return None
        if self._facts_support_clean_designed_tonal_keys_loop(facts):
            return None
        if self._processed_voice_candidate_blocks_mixed_loop(
            brain_result=brain_result,
            facts=facts,
        ):
            return None

        best_guess = self.best_brain_guess(
            brain_result,
            fragments=("mixed musical", "multi instrument"),
            max_rank=2,
            include_top={"Instruments"},
        )
        if best_guess is not None:
            rank = int(getattr(best_guess, "rank", 999))
            if rank <= 1 or shape_name != "bass_phrase":
                folder_path = "Instruments/Mixed Musical Loops/Multi Instrument/Loops"
                return ConsensusClaim(
                    family="Instruments",
                    sub_family="Mixed Musical Loops",
                    strength=1.0,
                    can_override=True,
                    reason=(
                        "product brain exposed a nearby mixed musical loop candidate, "
                        "so bass/keys/single-source loop evidence was kept as a component "
                        f"instead of the whole identity: role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                        f"raw={raw.folder_path}"
                    ),
                    source="profile_candidate_mixed_musical_loop_claim",
                    raw_candidate_score=min(float(rank) + 1.0, raw_score + 1.0),
                    label=folder_path,
                    folder_path=folder_path,
                    brain_rank=rank,
                    physics_rank=None,
                    shared_winner=folder_path,
                    shared_candidates=raw.shared_candidates,
                    is_real_candidate=True,
                )

        spread = self._mixed_instrument_loop_spread_support(
            raw=raw,
            brain_result=brain_result,
            physics_result=physics_result,
            role_name=role_name,
            shape_name=shape_name,
            shape_confidence=shape_confidence,
            facts=facts,
        )
        if spread is None:
            return None

        raw_group = self._instrument_branch_group(raw_path)
        if self._raw_has_direct_terminal_identity(
            raw=raw,
            raw_path=raw_path,
            raw_group=raw_group,
            raw_score=raw_score,
        ):
            return None

        folder_path = "Instruments/Instrument Loops/Loops"
        strength = min(0.98, max(0.91, 0.84 + 0.03 * spread.group_count))
        return ConsensusClaim(
            family="Instruments",
            sub_family="Instrument Loops",
            strength=strength,
            can_override=True,
            reason=(
                "measured mixed/pitched musical loop role had split source-identity "
                "evidence across instrument branches, so the arbiter kept the safe "
                f"broad Instrument Loops role bucket: groups={','.join(spread.groups)}; "
                f"physics_broad={spread.has_physics_broad_loop}; mixed_branch={spread.has_mixed_branch_evidence}; "
                f"role={role_name}; shape={shape_name}:{shape_confidence:.2f}; raw={raw.folder_path}"
            ),
            source="mixed_instrument_loop_role_claim",
            raw_candidate_score=min(spread.best_score + 1.0, raw_score + 1.0),
            label=folder_path,
            folder_path=folder_path,
            brain_rank=None,
            physics_rank=1 if spread.has_physics_broad_loop else None,
            shared_winner=folder_path,
            shared_candidates=raw.shared_candidates,
            is_real_candidate=False,
        )

    @staticmethod
    def _facts_support_clean_designed_tonal_keys_loop(facts: SharedAudioFacts | None) -> bool:
        """Return True when split candidates should yield to a clean keys loop.

        This is a source-name-blind stand-down for the broad mixed-loop claim.
        Some electric/acoustic piano loops expose guitar, reed, synth, and voice
        neighbors in candidate space, but the measured body is a very clean,
        low-flatness, mid-band tonal keys loop with no drum or FX motion.
        """
        if facts is None:
            return False

        def subpanel_value(name: str) -> float:
            if not isinstance(getattr(facts, "evidence", None), dict):
                return 0.0
            panels = facts.evidence.get("physics_subpanels", {})
            flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
            if not isinstance(flat, dict):
                return 0.0
            return _safe_float(flat.get(name))

        keys_support = max(
            _feature_number_from_facts(facts, "struck_keys_score"),
            _feature_number_from_facts(facts, "keys_tonal_decay_score"),
            _feature_number_from_facts(facts, "struck_keys_authority_score"),
            subpanel_value("struck_keys_score"),
            subpanel_value("keys_tonal_decay_score"),
            subpanel_value("struck_keys_authority_score"),
        )
        fx_motion = max(
            _feature_number_from_facts(facts, "fx_motion_score"),
            _feature_number_from_facts(facts, "fx_transition_authority_score"),
            subpanel_value("fx_motion_score"),
            subpanel_value("fx_transition_authority_score"),
        )
        return bool(
            _shape_vote_from_facts(facts) == "designed_tonal_fx"
            and _shape_confidence_from_facts(facts) >= 0.74
            and keys_support >= 0.54
            and max(
                _feature_number_from_facts(facts, "keys_tonal_decay_score"),
                subpanel_value("keys_tonal_decay_score"),
            )
            >= 0.70
            and _shape_metric_from_facts(facts, "pitched_event_ratio") >= 0.90
            and _shape_metric_from_facts(facts, "sustained_tonal_frame_ratio") >= 0.86
            and _shape_metric_from_facts(facts, "non_event_tonal_ratio") >= 0.86
            and 0.50 <= _shape_metric_from_facts(facts, "mid_event_ratio") <= 0.90
            and _shape_metric_from_facts(facts, "high_event_ratio") <= 0.05
            and _shape_metric_from_facts(facts, "spectral_flatness_mean") <= 0.03
            and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.10
            and _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.10
            and fx_motion < 0.50
        )

    def _mixed_instrument_loop_spread_support(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        physics_result: VoterResult | None,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
    ) -> MixedInstrumentLoopSpreadSupport | None:
        """Return split-identity support for broad Instrument Loops.

        This helper uses only internal voter candidates and measured facts.  It
        deliberately ignores filenames and source folders.  The broad fallback
        becomes eligible only when structure/role says music loop and the source
        candidates disagree across several instrument branches.
        """
        if not self._mixed_loop_role_or_shape_is_supported(role_name, shape_name, shape_confidence, facts):
            return None
        group_scores = self._instrument_identity_group_scores(
            raw,
            brain_result,
            physics_result,
        )
        concrete_groups = tuple(
            sorted(group for group in group_scores if group not in {"", "generic", "mixed", "bass", "voice"})
        )
        if len(concrete_groups) < 3:
            return None
        has_physics_broad_loop = self._physics_has_broad_instrument_loop(raw, physics_result)
        has_mixed_branch = self._facts_have_mixed_instrument_branch(facts)
        if not (has_physics_broad_loop or has_mixed_branch):
            return None
        best_score = min(group_scores.values()) if group_scores else 9999.0
        return MixedInstrumentLoopSpreadSupport(
            groups=concrete_groups,
            best_score=best_score,
            has_physics_broad_loop=has_physics_broad_loop,
            has_mixed_branch_evidence=has_mixed_branch,
        )

    @staticmethod
    def _mixed_loop_role_or_shape_is_supported(
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True for measured mixed or pitched loop role support."""
        if role_name == "mixed_music_loop":
            return True
        if role_name in {"pitched_music_loop", "pitched_music_phrase"} and shape_confidence >= 0.72:
            return True
        if (
            shape_name in {"mixed_instrument_loop", "compound_musical_loop", "instrument_plus_fx_loop"}
            and shape_confidence >= 0.66
        ):
            return True
        if shape_name == "repeated_phrase_loop" and shape_confidence >= 0.74:
            return True
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        roles = facts.evidence.get("measured_roles")
        if isinstance(roles, dict):
            return bool(
                _safe_float(roles.get("mixed_music_loop")) >= 0.58
                or _safe_float(roles.get("pitched_music_loop")) >= 0.74
                or _safe_float(roles.get("pitched_music_phrase")) >= 0.78
            )
        return False

    def _instrument_identity_group_scores(
        self,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        physics_result: VoterResult | None,
    ) -> dict[str, float]:
        """Return best internal candidate score per instrument branch."""
        groups: dict[str, float] = {}

        def add_path(path: str, score: float, *, top_family: str = "Instruments") -> None:
            if top_family != "Instruments" and not str(path).startswith("Instruments/"):
                return
            group = self._instrument_branch_group(path)
            if not group:
                return
            current = groups.get(group)
            if current is None or score < current:
                groups[group] = score

        raw_path = str(raw.folder_path or raw.label or "")
        add_path(raw_path, float(raw.raw_candidate_score or 9999.0), top_family=raw.family)

        for row in raw.shared_candidates or []:
            path = str(row.get("folder_path") or row.get("label") or "")
            top = str(row.get("top_family") or path.split("/", 1)[0])
            score = _safe_float(row.get("combined_rank_score")) or 9999.0
            brain_rank = int(_safe_float(row.get("brain_rank")) or 999.0)
            physics_rank = int(_safe_float(row.get("physics_rank")) or 999.0)
            if score <= 24.0 or brain_rank <= 10 or physics_rank <= 10:
                add_path(path, score, top_family=top)

        for guess in brain_result.guesses:
            if int(getattr(guess, "rank", 999)) <= 10:
                add_path(
                    str(getattr(guess, "folder_path", "") or getattr(guess, "label", "")),
                    float(getattr(guess, "rank", 999)) + 2.0,
                    top_family=str(getattr(guess, "top_family", "")),
                )

        if physics_result is not None:
            for guess in physics_result.guesses:
                if int(getattr(guess, "rank", 999)) <= 10:
                    add_path(
                        str(getattr(guess, "folder_path", "") or getattr(guess, "label", "")),
                        float(getattr(guess, "rank", 999)) + 2.5,
                        top_family=str(getattr(guess, "top_family", "")),
                    )
        return groups

    @staticmethod
    def _physics_has_broad_instrument_loop(
        raw: ConsensusClaim,
        physics_result: VoterResult | None,
    ) -> bool:
        """Return True when PhysicsVoter exposes broad Instrument Loops."""
        if physics_result is not None and physics_result.guesses:
            top_guess = min(
                physics_result.guesses,
                key=lambda guess: int(getattr(guess, "rank", 999)),
            )
            top_path = _norm_path(str(top_guess.folder_path or top_guess.label))
            if top_guess.top_family == "Instruments" and _path_has_any(
                top_path, ("instrument loops", "mixed musical", "multi instrument")
            ):
                return True
        for row in raw.shared_candidates or []:
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            if str(row.get("top_family") or "") != "Instruments" and not path.startswith("instruments/"):
                continue
            physics_rank = int(_safe_float(row.get("physics_rank")) or 999.0)
            if physics_rank <= 2 and _path_has_any(path, ("instrument loops", "mixed musical", "multi instrument")):
                return True
        return False

    @staticmethod
    def _facts_have_mixed_instrument_branch(facts: SharedAudioFacts | None) -> bool:
        """Return True when measured physics already names the mixed branch."""
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        evidence = facts.evidence
        branch_values = (
            evidence.get("physics_layer_branch"),
            evidence.get("instrument_branch"),
            evidence.get("dominant_instrument_branch"),
        )
        if any(str(value) == "MixedInstrument" for value in branch_values):
            return True
        return bool(
            _safe_float(evidence.get("instrument_branch_MixedInstrument")) >= 0.66
            or _feature_number_from_facts(facts, "instrument_branch_MixedInstrument") >= 0.66
            or _feature_number_from_facts(facts, "compound_music_prefer_broad_loop") >= 0.50
        )

    def _processed_voice_candidate_blocks_mixed_loop(
        self,
        *,
        brain_result: VoterResult,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when measured spoken-voice evidence should own the source."""
        voice_guess = self.best_brain_guess(
            brain_result,
            fragments=("voice", "vocal", "choir", "spoken"),
            max_rank=4,
            include_top={"Instruments"},
        )
        if voice_guess is None:
            return False
        human_spoken = self._score_from_facts(facts, "human_spoken_voice_score")
        voice_panel = max(
            human_spoken,
            self._score_from_facts(facts, "voice_score"),
            self._score_from_facts(facts, "human_breath_mouth_score"),
            self._score_from_facts(facts, "voice_choir_score"),
        )
        drum_panel = max(
            self._score_from_facts(facts, "drum_hit_score"),
            self._score_from_facts(facts, "drum_loop_source_score"),
            self._score_from_facts(facts, "drum_snare_source_score"),
            self._score_from_facts(facts, "drum_cymbal_source_score"),
        )
        fx_transition = max(
            self._score_from_facts(facts, "fx_motion_score"),
            self._score_from_facts(facts, "fx_transition_authority_score"),
            self._score_from_facts(facts, "fx_riser_build_score"),
            self._score_from_facts(facts, "fx_whoosh_sweep_score"),
        )
        return bool(
            human_spoken >= 0.74
            and voice_panel >= 0.66
            and _shape_metric_from_facts(facts, "pitched_event_ratio") >= 0.54
            and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.45
            and _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.45
            and drum_panel <= 0.58
            and fx_transition <= 0.62
        )

    def bass_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_path: str,
        raw_score: float,
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim | None:
        """Return a bass-loop claim from a real BrainVoter bass candidate."""
        role_supports_bass = bool(
            (role_name == "bass_loop" and shape_name == "bass_phrase" and shape_confidence >= 0.80)
            or self._facts_support_measured_bass_loop_body(
                facts, shape_name=shape_name, shape_confidence=shape_confidence
            )
        )
        if not role_supports_bass:
            return None
        best_guess = self.best_brain_guess(
            brain_result,
            fragments=BASS_FRAGMENTS,
            max_rank=8,
            include_top={"Instruments"},
        )
        if best_guess is None:
            return None
        if not self._bass_claim_may_override_raw_path(raw=raw, raw_path=raw_path, best_guess_rank=best_guess.rank):
            return None
        return claim_from_category_guess(
            guess=best_guess,
            folder_path="Instruments/Bass/Bass Loops",
            source="profile_candidate_bass_claim",
            reason=(
                "BrainVoter had a nearby bass profile candidate and measured audio "
                f"supported bass-loop structure: role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                f"raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=min(raw_score + 2.0, float(best_guess.rank) + 2.0),
            can_override=True,
            strength=max(0.88, min(0.96, 1.02 - 0.03 * max(0, best_guess.rank - 1))),
        )

    def _bass_claim_may_override_raw_path(
        self,
        *,
        raw: ConsensusClaim,
        raw_path: str,
        best_guess_rank: int,
    ) -> bool:
        """Return True when a bass witness can safely refine the raw winner."""
        if _path_has_any(raw_path, BASS_FRAGMENTS):
            return False
        if raw.final_top != "Instruments":
            return True
        if raw_path in {"instruments/instrument loops/loops", "instruments/instrument loops"}:
            return True
        return int(best_guess_rank) <= 1

    def _facts_support_measured_bass_loop_body(
        self,
        facts: SharedAudioFacts | None,
        *,
        shape_name: str,
        shape_confidence: float,
    ) -> bool:
        """Return True when measured low/pitch evidence supports a bass loop."""
        if facts is None:
            return False
        if shape_name not in {"bass_phrase", "solo_phrase", "pitched_phrase", "pitched_repetition_phrase"}:
            return False
        if shape_confidence < 0.70:
            return False
        roles = facts.evidence.get("measured_roles", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
        nested_roles = roles.get("evidence", {}) if isinstance(roles, dict) else {}
        role_owner = max(
            _safe_float(roles.get("bass_loop")) if isinstance(roles, dict) else 0.0,
            _safe_float(roles.get("pitched_music_loop")) if isinstance(roles, dict) else 0.0,
            _safe_float(roles.get("low_rhythmic_drum_loop")) if isinstance(roles, dict) else 0.0,
            _safe_float(nested_roles.get("bass_loop")) if isinstance(nested_roles, dict) else 0.0,
            _safe_float(nested_roles.get("pitched_music_loop")) if isinstance(nested_roles, dict) else 0.0,
            _safe_float(nested_roles.get("low_rhythmic_drum_loop")) if isinstance(nested_roles, dict) else 0.0,
        )
        low_body = max(
            _feature_number_from_facts(facts, "low_total"),
            _shape_metric_from_facts(facts, "low_event_ratio"),
        )
        pitch_confidence = max(
            _feature_number_from_facts(facts, "pitch_confidence"),
            _shape_metric_from_facts(facts, "pitch_confidence"),
        )
        drum_body = max(
            self._score_from_facts(facts, "drum_hit_score"),
            self._score_from_facts(facts, "drum_loop_source_score"),
            _shape_metric_from_facts(facts, "drumlike_frame_ratio"),
            _shape_metric_from_facts(facts, "percussive_event_ratio"),
        )
        return bool(role_owner >= 0.62 and low_body >= 0.70 and pitch_confidence >= 0.70 and drum_body <= 0.46)
