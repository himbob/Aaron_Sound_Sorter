# SOURCE-NAME BLINDNESS INVARIANT:
# Measured broad-bucket logic may inspect voter output and measured facts.
# It must never inspect producer filenames, source folders, or ZIP member names.
"""Measured-role broad-bucket claim producer."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.claim_tools import BroadBucketClaimProducerBase, BroadBucketClaimTools
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _is_concrete_fx_path,
    _measured_role_from_facts,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class MeasuredBroadBucketClaimProducer(BroadBucketClaimProducerBase):
    """Recover broad measured-role buckets from over-eager review paths."""

    def __init__(self, tools: BroadBucketClaimTools) -> None:
        super().__init__(tools)

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return a measured broad-bucket claim, if one is justified."""
        claim = self.measured_broad_bucket_claim(context.raw, context.eligibility, context.facts)
        return [] if claim is None else [claim]

    def measured_broad_bucket_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Recover curated/stable smoke examples from over-eager review.

        This guard uses only measured roles, measured shape labels, raw voter
        candidates, and the raw consensus output.  It never reads producer
        filenames or source folders.  It is intentionally broad: when the
        measured structure is clear enough to choose a useful parent family,
        choose that broad parent instead of dumping a curated example into
        ``_TO_REVIEW/Measured Role Conflict``.
        """
        if raw.final_top == "_TO_REVIEW":
            return None
        role = str(eligibility.role_name or "")
        measured_role = _measured_role_from_facts(facts)
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)
        raw_path = _norm_path(raw.folder_path)
        broad_path = _norm_path(eligibility.broad_folder_path)

        # Stability rule: this smoke-stability guard is allowed to recover from
        # review/unsafe broad conflicts, but it is not allowed to demote an
        # already useful real bucket to a generic bucket.  The previous version
        # broke Bass Loops and Drum Loops by flattening raw Bass/Drum winners to
        # Instruments/Instrument Loops.
        if raw.final_top == "Instruments" and _path_has_any(raw_path, ("bass", "808", "sub bass", "synth bass")):
            return None
        if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops", "percussion")):
            return None
        if raw.final_top == "FX" and _path_has_any(raw_path, ("human and voice", "voice", "vocal")):
            # Keep true vocal/voice broad winners, but do not let a synthetic
            # Human/Voice bucket mask protected percussive hits or stable pitched
            # instruments.  Those cases must continue into the eligibility
            # arbiter, where they can be broadened to Drums/Instruments or review.
            direct_pitched_strength = max(
                _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
                _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
            )
            measured_vocal_strength = max(
                _role_strength_from_facts(facts, "vocal_music_phrase"),
                _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
            )
            if (
                role not in {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}
                and measured_role
                not in {
                    "pitched_music_loop",
                    "pitched_music_phrase",
                    "pitched_reed_or_instrument_loop",
                    "pitched_reed_or_instrument_phrase",
                }
                and shape not in {"pitched_phrase", "bass_phrase", "sustained_pad"}
                and not (direct_pitched_strength >= 0.84 and measured_vocal_strength < 0.55)
            ):
                return None

        # Pitched musical loop/phrase evidence is a better broad answer than
        # review for sax/reed/pad/string/key loops, even when the raw leaf is a
        # wrong FX neighborhood.  Exact FX tonal-build rescues already run in
        # _true_bucket_for_candidate_conflict before this method is relevant in
        # normal conflicted cases; this guard only prevents review as the answer.
        if (
            role == "bass_loop"
            and shape == "bass_phrase"
            and shape_conf >= 0.86
            and broad_path
            and "bass" in broad_path
        ):
            if raw.final_top == "Instruments" and _path_has_any(
                raw_path, ("instrument loops", "mixed musical loops", "multi instrument")
            ):
                return None
            return self._broaden_from_raw(
                raw,
                eligibility,
                "measured bass-loop structure beat unsafe non-bass family",
            )

        pitched_roles = {
            "pitched_music_loop",
            "pitched_music_phrase",
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
            "mixed_music_loop",
            "clean_sustained_tonal_instrument_loop",
        }
        pitched_shapes = {"pitched_phrase", "bass_phrase", "sustained_pad"}
        if role in pitched_roles or (shape in pitched_shapes and shape_conf >= 0.72):
            # A pitched musical structure can rescue non-instrument false
            # positives back to Instruments, but it must not demote already
            # useful Bass/Drum/Voice buckets to generic Instrument Loops.
            if raw.final_top == "Instruments":
                if (
                    shape in pitched_shapes
                    and shape_conf >= 0.72
                    and _path_has_any(raw_path, ("one shots", "one shot"))
                    and not _path_has_any(
                        raw_path, ("bass", "808", "drum", "percussion", "voice", "vocal", "human and voice")
                    )
                ):
                    broad_loop_path = "Instruments/Instrument Loops/Loops"
                    if _path_has_any(
                        raw_path,
                        (
                            "sax",
                            "saxophone",
                            "woodwind",
                            "woodwinds",
                            "brass",
                            "trumpet",
                            "trombone",
                            "horn",
                            "flute",
                            "clarinet",
                            "reed",
                        ),
                    ):
                        broad_loop_path = "Instruments/Brass and Woodwinds/Loops"
                    return self._redirect_from_raw(
                        raw,
                        broad_loop_path,
                        "measured pitched/bass phrase shape prevented exact one-shot leaf from replacing broad loop bucket",
                    )
                return None
            if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops", "percussion")):
                return None
            if raw.final_top == "FX" and _path_has_any(raw_path, ("human and voice", "voice", "vocal")):
                direct_pitched_strength = max(
                    _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
                    _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
                )
                measured_vocal_strength = max(
                    _role_strength_from_facts(facts, "vocal_music_phrase"),
                    _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
                )
                if (
                    role not in {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}
                    and measured_role
                    not in {
                        "pitched_music_loop",
                        "pitched_music_phrase",
                        "pitched_reed_or_instrument_loop",
                        "pitched_reed_or_instrument_phrase",
                    }
                    and shape not in {"pitched_phrase", "bass_phrase", "sustained_pad"}
                    and not (direct_pitched_strength >= 0.84 and measured_vocal_strength < 0.55)
                ):
                    return None
            if _is_concrete_fx_path(raw_path):
                transition_like_fx = _path_has_any(
                    raw_path,
                    ("riser", "build", "sweep", "whoosh", "drop", "downlifter", "transition"),
                )
                stable_instrument_role = role in {
                    "pitched_reed_or_instrument_loop",
                    "pitched_reed_or_instrument_phrase",
                    "clean_sustained_tonal_instrument_loop",
                }
                if not stable_instrument_role and not (transition_like_fx and shape in pitched_shapes):
                    return raw
            return self._redirect_from_raw(
                raw,
                "Instruments/Instrument Loops/Loops",
                "measured pitched musical structure kept curated smoke example out of review",
            )

        # Drum-top loops and beat loops should become the broad drum-loop bucket,
        # not FX/drop review, when the measured shape or role is clearly a loop.
        drum_roles = {
            "drum_loop",
            "percussive_drum_loop",
            "bright_drum_loop",
            "low_rhythmic_drum_loop",
        }
        drum_shapes = {"beat_loop", "top_loop", "drum_loop"}
        if role in drum_roles or (shape in drum_shapes and shape_conf >= 0.70):
            if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops")):
                return None
            if raw.final_top == "Instruments" and _path_has_any(raw_path, ("instrument loops", "mixed musical loops")):
                best_voice = self._best_candidate_score(
                    raw,
                    include_top={"FX", "Instruments"},
                    include_fragments=("human and voice", "voice", "vocal", "vox", "choir", "breath", "crowd"),
                )
                best_instrument = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
                best_drum = self._best_candidate_score(
                    raw,
                    include_top={"Drums"},
                    include_fragments=("drum loop", "drum loops", "kick", "percussion", "conga", "bongo", "tabla"),
                )
                if (
                    best_voice is not None
                    and best_instrument is not None
                    and best_voice <= best_instrument + 2.0
                    and (best_drum is None or best_drum >= best_voice + 6.0)
                ):
                    return self._review_from_raw(
                        raw,
                        eligibility,
                        "vocal loop conflict: drum-loop broadening lacked enough drum candidate support",
                    )
            if broad_path and not broad_path.startswith("_to_review"):
                return self._broaden_from_raw(
                    raw,
                    eligibility,
                    "measured drum/percussion loop structure kept curated smoke example out of review",
                )
            return self._redirect_from_raw(
                raw,
                "Drums/Drum Loops/Loops",
                "measured drum/percussion loop structure kept curated smoke example out of review",
            )

        # Vocal phrase shape should go to the broad Human/Voice bucket when the
        # alternative is review or an unsafe machine/riser/rimshot-like leaf.
        if shape == "vocal_phrase" and shape_conf >= 0.82:
            # Shape alone can be fooled by sax/flute-like instruments, so never
            # use this rescue when measured facts explicitly say pitched music.
            # Otherwise a very strong vocal_phrase shape is safer as broad
            # Human/Voice than as riser/rim/machine review.
            voice_roles = {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
            if measured_role in {
                "pitched_music_loop",
                "pitched_music_phrase",
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
                "bass_loop",
            }:
                return None
            stable_pitched_strength = max(
                _role_strength_from_facts(facts, "pitched_music_loop"),
                _role_strength_from_facts(facts, "pitched_music_phrase"),
                _role_strength_from_facts(facts, "pitched_reed_or_instrument_loop"),
                _role_strength_from_facts(facts, "pitched_reed_or_instrument_phrase"),
                _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
                _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
            )
            has_voice_candidate = self._has_voice_candidate_with_role(raw)
            if stable_pitched_strength >= 0.84 and not has_voice_candidate:
                best_instrument = self._best_candidate(
                    raw,
                    include_top={"Instruments"},
                    include_fragments=(
                        "brass",
                        "woodwind",
                        "sax",
                        "flute",
                        "clarinet",
                        "horn",
                        "instrument loops",
                        "synth",
                        "guitar",
                        "keys",
                        "piano",
                        "strings",
                    ),
                )
                if best_instrument is not None:
                    _instrument_score, instrument_path = best_instrument
                    return self._redirect_from_raw(
                        raw,
                        instrument_path,
                        "stable pitched instrument evidence blocked vocal-phrase shape rescue without real Human/Voice candidate",
                    )
                return self._review_from_raw(
                    raw,
                    eligibility,
                    "stable pitched instrument evidence blocked vocal-phrase shape rescue without real Human/Voice candidate",
                )
            if raw.final_top == "FX" and "human and voice" in raw_path:
                return None
            if shape_conf >= 0.95 or measured_role in voice_roles or role in voice_roles:
                return self._redirect_from_raw(
                    raw,
                    "FX/Human and Voice FX",
                    "measured vocal phrase structure kept curated smoke example out of review",
                )

        # If eligibility itself points to a real broad path, use that instead of
        # allowing a later conflict branch to turn the decision into review.
        if broad_path and not broad_path.startswith("_to_review"):
            if eligibility.confidence >= 0.76 and not eligibility.is_path_allowed(raw.folder_path, raw.final_top):
                return self._broaden_from_raw(
                    raw,
                    eligibility,
                    "measured broad parent was safer than review for curated/stable example",
                )

        return None
