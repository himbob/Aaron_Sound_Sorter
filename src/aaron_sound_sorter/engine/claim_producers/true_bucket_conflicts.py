# SOURCE-NAME BLINDNESS INVARIANT:
# True-bucket conflict logic may inspect voter output and measured facts.
# It must never inspect producer filenames, source folders, or ZIP member names.
"""Candidate-conflict claim producer for broad family safety."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.claim_tools import BroadBucketClaimProducerBase, BroadBucketClaimTools
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _candidate_role_strength,
    _direct_body_role_strength_from_facts,
    _feature_number_from_facts,
    _has_drum_loop_structure_support,
    _is_measured_loop_phrase_context,
    _measured_role_from_facts,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
    _shape_blocks_percussion_loop_rescue,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
    _stable_instrument_claim_strength,
    _voice_identity_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import (
    ConsensusClaim,
    build_human_voice_claim,
)


class TrueBucketConflictClaimProducer(BroadBucketClaimProducerBase):
    """Resolve strong candidate contradictions before broad fallbacks."""

    def __init__(self, tools: BroadBucketClaimTools) -> None:
        super().__init__(tools)

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return a true-bucket conflict claim, if one is justified."""
        claim = self.true_bucket_conflict_claim(context.raw, context.eligibility, context.facts)
        return [] if claim is None else [claim]

    def true_bucket_conflict_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim | None:
        """Choose a useful broad/near-true bucket when evidence is strong enough.

        Earlier conflict-resolver patches correctly stopped bad confident
        placements, but they sent too many confirmed cases to review.  This
        method is the next safety lattice step: it only redirects when the
        ranked candidate evidence clearly supports a safer real family.  It
        never reads source filenames or source folders.
        """
        if raw.final_top == "_TO_REVIEW":
            return None
        raw_path = _norm_path(raw.folder_path)
        measured_role = _measured_role_from_facts(facts)
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)

        early = self._early_candidate_adjudication(raw, eligibility, measured_role, shape, shape_conf, facts)
        if early is not None:
            return early

        # Stability guard: if the raw winner is already a strong broad drum-loop
        # bucket, do not let a noisy bass/pitched role broaden it to Instruments.
        # This is the exact failure seen in the FX smoke matrix where drum loops
        # were demoted to Instrument Loops/Bass.
        if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops")):
            best_drum = self._best_candidate_score(
                raw, include_top={"Drums"}, include_fragments=("drum", "drum loop", "drum loops")
            )
            best_bass = self._best_candidate_score(
                raw, include_top={"Instruments"}, include_fragments=("bass", "808", "sub bass", "synth bass")
            )
            if best_drum is not None and (best_bass is None or best_drum <= best_bass):
                return raw

        # Shape may look vocal for sax/reed/synth/string/bass phrases.  When
        # measured roles say pitched music and the raw winner is already a safe
        # Instrument Loop, do not allow a vocal eligibility fallback to steal it
        # into Human/Voice.
        if (
            raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("instrument loops", "mixed musical loops", "bass"))
            and eligibility.role_name in {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
            and measured_role
            in {
                "pitched_music_loop",
                "pitched_music_phrase",
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
                "bass_loop",
            }
        ):
            return raw

        # Decisive measured bass-loop evidence must not be flattened to generic
        # Instrument Loops or stolen by FX/Risers.  This rescue requires actual
        # Bass/808/sub/synth-bass candidate support, so synth leads, Rhodes, and
        # generic low-mid keys do not become Bass merely because a role score was
        # high.
        bass_rescue_role = (
            eligibility.role_name == "bass_loop"
            or (measured_role == "bass_loop" and eligibility.confidence >= 0.78)
            or (shape == "bass_phrase" and shape_conf >= 0.86 and eligibility.role_name == "bass_loop")
        )
        if bass_rescue_role and eligibility.confidence >= 0.78:
            best_bass = self._best_candidate(
                raw,
                include_top={"Instruments"},
                include_fragments=("bass", "808", "sub bass", "synth bass"),
            )
            best_nonbass_inst = self._best_candidate(
                raw,
                include_top={"Instruments"},
                include_fragments=(
                    "guitar",
                    "rhodes",
                    "piano",
                    "keys",
                    "organ",
                    "synth lead",
                    "strings",
                    "brass",
                    "woodwind",
                    "instrument loops",
                ),
                exclude_fragments=("bass", "808", "sub bass", "synth bass"),
            )
            if best_bass is not None:
                bass_score, _bass_path = best_bass
                nonbass_score = best_nonbass_inst[0] if best_nonbass_inst is not None else None
                try:
                    raw_score = float(raw.combined_rank_score or 9999.0)
                except Exception:
                    raw_score = 9999.0
                raw_is_unsafe = (
                    raw.final_top == "FX"
                    or (
                        raw.final_top == "Instruments"
                        and _path_has_any(raw_path, ("instrument loops", "mixed musical loops"))
                    )
                    or raw.final_top == "Drums"
                )
                bass_is_supported = bass_score <= raw_score + 8.0 and (
                    nonbass_score is None or bass_score <= nonbass_score + 3.0
                )
                if raw_is_unsafe and bass_is_supported:
                    return self._redirect_from_raw(
                        raw,
                        "Instruments/Bass/Bass Loops",
                        "bass-loop true-bucket rescue: measured bass_loop plus Bass candidate support beat unsafe generic/FX family",
                    )

        # If consensus produced only a synthetic generic Instrument Loop bucket
        # while the real ranked candidates are concrete FX leaves, do not let a
        # broad measured-role fallback wash obvious siren/glitch/whoosh evidence
        # into Instruments.  This is still candidate-evidence logic, not source
        # name logic: it fires only when no real Instrument candidate survived.
        if raw.final_top == "Instruments" and _path_has_any(raw_path, ("instrument loops", "mixed musical loops")):
            best_concrete_fx = self._best_candidate(
                raw,
                include_top={"FX"},
                include_fragments=(
                    "siren",
                    "alarm",
                    "glitch",
                    "stutter",
                    "designed noise",
                    "hybrid designed",
                    "whoosh",
                    "swoosh",
                    "swish",
                    "sweep",
                    "riser",
                    "build",
                    "drop",
                    "downlifter",
                    "transition",
                ),
            )
            best_real_instrument = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
            if best_concrete_fx is not None and best_real_instrument is None:
                _fx_score, fx_path = best_concrete_fx
                return self._redirect_from_raw(
                    raw,
                    fx_path,
                    "concrete FX true-bucket rescue: FX candidates survived while generic Instrument Loop was only a synthetic broad fallback",
                )

        # Clean pitched musical phrases should not be stolen into FX/Alarm or
        # FX/Risers from a weak tonal-alert role unless the measured shape is
        # actually a transition.  This protects piano/guitar/string/synth loops
        # from becoming risers while still allowing real transition shapes.
        if eligibility.role_name == "fx_tonal_alert_or_siren":
            shape = _shape_vote_from_facts(facts)
            shape_conf = _shape_confidence_from_facts(facts)
            transition_shape = shape in {"transition_riser", "transition_drop"} and shape_conf >= 0.84
            best_inst = self._best_candidate(
                raw,
                include_top={"Instruments"},
                include_fragments=(
                    "instrument loops",
                    "synth",
                    "lead",
                    "guitar",
                    "piano",
                    "keys",
                    "rhodes",
                    "strings",
                    "cello",
                    "brass",
                    "woodwind",
                ),
            )
            best_fx = self._best_candidate_score(raw, include_top={"FX"}, include_fragments=())
            if not transition_shape and best_inst is not None:
                inst_score, inst_path = best_inst
                if best_fx is None or inst_score <= best_fx + 5.0:
                    safe_path = inst_path
                    if not _path_has_any(
                        _norm_path(safe_path),
                        (
                            "instrument loops",
                            "synth",
                            "guitar",
                            "piano",
                            "keys",
                            "rhodes",
                            "strings",
                            "brass",
                            "woodwind",
                        ),
                    ):
                        safe_path = "Instruments/Instrument Loops/Loops"
                    return self._redirect_from_raw(
                        raw,
                        safe_path,
                        "musical-loop safeguard: tonal-alert role lacked transition shape and instrument candidate support beat unsafe FX alarm/riser family",
                    )

        # Confirmed kick/drum/percussion loops sometimes win as Bass Loops,
        # generic Instrument Loops, or Human/Voice FX.  Only allow this rescue
        # when measured structure is actually drum/percussion-like.  The prior
        # version let weak Drum candidates steal sax and vocal loops into
        # Drums/Drum Loops.
        drum_rescue_allowed = _has_drum_loop_structure_support(eligibility.role_name, measured_role, shape, shape_conf)
        best_drum_loop = (
            self._best_candidate(
                raw,
                include_top={"Drums"},
                include_fragments=(
                    "drum loop",
                    "drum loops",
                    "break",
                    "breaks",
                    "kick",
                    "percussion",
                    "conga",
                    "bongo",
                    "tabla",
                    "triangle",
                    "wood block",
                    "wood blocks",
                    "metallic percussion",
                ),
            )
            if drum_rescue_allowed
            else None
        )
        best_wrong_instrument = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
        best_voice = self._best_candidate_score(
            raw,
            include_top={"FX", "Instruments"},
            include_fragments=("human and voice", "voice", "vocal", "vox", "choir", "breath", "crowd"),
        )
        if (
            eligibility.role_name == "pitched_percussion_loop"
            and measured_role in {"pitched_music_loop", "pitched_music_phrase", "vocal_music_phrase", "bass_loop"}
            and raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("instrument loops", "mixed musical loops", "bass"))
        ):
            return raw

        if best_drum_loop is not None:
            drum_score, drum_path = best_drum_loop
            drum_path_low = _norm_path(drum_path)
            drum_rescue_reason = (
                "percussion-loop true-bucket rescue"
                if _path_has_any(
                    drum_path_low,
                    ("percussion", "conga", "bongo", "tabla", "triangle", "wood block", "wood blocks", "metallic"),
                )
                else "kick/drum-loop true-bucket rescue"
            )
            raw_is_wrong_loop = raw.final_top == "Instruments" and _path_has_any(
                raw_path, ("bass loops", "instrument loops", "mixed musical loops")
            )
            raw_is_human_voice_fx = raw.final_top == "FX" and "human and voice" in raw_path
            if raw_is_wrong_loop and (best_wrong_instrument is None or drum_score <= best_wrong_instrument + 5.0):
                return self._redirect_from_raw(
                    raw,
                    "Drums/Drum Loops/Loops",
                    f"{drum_rescue_reason}: close drum/percussion candidate beat unsafe instrument-loop family",
                )
            if raw_is_human_voice_fx and (best_voice is None or drum_score <= best_voice + 4.0):
                return self._redirect_from_raw(
                    raw,
                    "Drums/Drum Loops/Loops",
                    f"{drum_rescue_reason}: close drum/percussion candidate beat unsafe Human/Voice FX family",
                )

        # If a raw Human/Voice FX winner is supported by measured vocal facts,
        # keep it.  The previous instrument rescue stole real vocal shouts into
        # Synth Chord because a harmonic one-shot candidate ranked nearby.
        if raw.final_top == "FX" and "human and voice" in raw_path:
            voice_roles = {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
            protected_hit_roles = {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}
            if eligibility.role_name not in protected_hit_roles and (
                eligibility.role_name in voice_roles or measured_role in voice_roles
            ):
                return raw

        # If a raw FX one-shot contradicts very strong Keys/Rhodes/Guitar/etc.
        # candidate evidence, trust the best instrument candidate.  This covers
        # confirmed epchord/Rhodes-style one-shots without weakening drums or
        # stealing true Human/Voice winners.
        if raw.final_top == "FX":
            best_inst_hit = self._best_candidate(
                raw,
                include_top={"Instruments"},
                include_fragments=(
                    "keys",
                    "rhodes",
                    "electric piano",
                    "piano",
                    "organ",
                    "guitar",
                    "strum",
                    "chord",
                    "pluck",
                    "synth",
                    "strings",
                ),
            )
            if best_inst_hit is not None:
                inst_score, inst_path = best_inst_hit
                try:
                    raw_score = float(raw.combined_rank_score or 9999.0)
                except Exception:
                    raw_score = 9999.0
                if inst_score <= raw_score - 4.0:
                    return self._redirect_from_raw(
                        raw,
                        inst_path,
                        "instrument true-bucket rescue: instrument candidate evidence strongly beat unsafe FX one-shot family",
                    )

        # Human/Voice FX is a common false-positive sink.  When a non-voice FX
        # candidate is stronger, keep the file in FX but move it out of Voice.
        if raw.final_top == "FX" and "human and voice" in raw_path:
            best_nonvoice_fx = self._best_candidate(
                raw,
                include_top={"FX"},
                include_fragments=(
                    "whoosh",
                    "swoosh",
                    "swish",
                    "sweep",
                    "seagull",
                    "seaguls",
                    "bird",
                    "animal",
                    "clank",
                    "metallic",
                    "bell",
                    "chime",
                    "hybrid designed",
                    "designed noise",
                    "riser",
                    "build",
                    "transition",
                ),
                exclude_fragments=("human and voice", "voice", "vocal", "crowd", "breath"),
            )
            if best_nonvoice_fx is not None:
                nonvoice_score, nonvoice_path = best_nonvoice_fx
                if best_voice is None or nonvoice_score <= best_voice - 2.0:
                    return self._redirect_from_raw(
                        raw,
                        nonvoice_path,
                        "non-voice FX true-bucket rescue: stronger non-voice FX candidate beat Human/Voice FX",
                    )

        # Growing metallic/bell/build FX can be tonal and loop-like, but this
        # rescue used to steal ordinary piano/guitar/string loops into FX/Risers.
        # It now requires a measured transition shape and clearly better FX
        # evidence.  Plain pitched motion is not enough.
        if (
            eligibility.role_name in {"pitched_music_loop", "pitched_music_phrase", "mixed_music_loop"}
            and raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("instrument loops", "mixed musical loops"))
        ):
            shape = _shape_vote_from_facts(facts)
            shape_conf = _shape_confidence_from_facts(facts)
            if shape in {"transition_riser", "transition_drop"} and shape_conf >= 0.86:
                best_tonal_fx = self._best_candidate(
                    raw,
                    include_top={"FX"},
                    include_fragments=(
                        "bell",
                        "bells",
                        "chime",
                        "chimes",
                        "riser",
                        "build",
                        "sweep",
                        "whoosh",
                        "hybrid designed",
                        "designed tonal",
                        "metallic",
                    ),
                )
                best_inst = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
                if best_tonal_fx is not None:
                    fx_score, fx_path = best_tonal_fx
                    if best_inst is None or fx_score <= best_inst - 2.0:
                        return self._redirect_from_raw(
                            raw,
                            fx_path,
                            "tonal transition FX true-bucket rescue: transition shape and stronger FX candidate beat generic Instrument Loop",
                        )

        return None

    def _early_candidate_adjudication(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        measured_role: str,
        shape: str,
        shape_conf: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Resolve high-confidence candidate contradictions before broad fallbacks.

        This is the main safety lattice for historical regressions. It uses only
        voter candidate folders, measured role labels, and measured shape labels.
        It never reads source filenames or source folders.
        """
        raw_path = _norm_path(raw.folder_path)
        role = str(eligibility.role_name or "")
        try:
            raw_score = float(raw.combined_rank_score or 9999.0)
        except Exception:
            raw_score = 9999.0

        direct_voice_strength = max(
            _direct_body_role_strength_from_facts(facts, "voiced_one_shot"),
            _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
        )
        full_voice_strength = max(
            _role_strength_from_facts(facts, "voiced_one_shot"),
            _role_strength_from_facts(facts, "vocal_music_phrase"),
        )
        bass_loop_strength = max(
            _role_strength_from_facts(facts, "bass_loop"),
            _direct_body_role_strength_from_facts(facts, "bass_loop"),
        )

        # Positive role claims get first refusal before any Human/Voice or
        # short-hit rescue can move the file across families.  This prevents
        # later safety branches from demoting a confirmed drum loop into a bass
        # one-shot, or a confirmed bass loop into Human/Voice.
        early_best_drum_loop = self._best_candidate(
            raw,
            include_top={"Drums"},
            include_fragments=("drum loop", "drum loops", "break", "breaks"),
        )
        early_drum_loop_strength = max(
            _role_strength_from_facts(facts, "low_rhythmic_drum_loop"),
            _role_strength_from_facts(facts, "percussive_drum_loop"),
            _direct_body_role_strength_from_facts(facts, "low_rhythmic_drum_loop"),
            _direct_body_role_strength_from_facts(facts, "percussive_drum_loop"),
        )
        if (
            early_best_drum_loop is not None
            and early_best_drum_loop[0] <= 4.0
            and early_drum_loop_strength >= 0.55
            and not (shape == "bass_phrase" and bass_loop_strength >= early_drum_loop_strength + 0.12)
        ):
            return self._redirect_from_raw(
                raw,
                "Drums/Drum Loops/Loops",
                "positive drum-loop claim beat later bass/voice rescue branches",
            )

        if (
            raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("bass loops",))
            and shape == "bass_phrase"
            and bass_loop_strength >= 0.70
        ):
            return raw

        if (
            raw.final_top == "FX"
            and "human and voice" in raw_path
            and bass_loop_strength >= 0.86
            and shape == "bass_phrase"
            and shape_conf >= 0.82
            and max(full_voice_strength, direct_voice_strength) < 0.92
        ):
            return self._redirect_from_raw(
                raw,
                "Instruments/Bass/Bass Loops",
                "bass-loop claim blocked weak Human/Voice false positive",
            )

        if (
            raw.final_top == "FX"
            and "human and voice" in raw_path
            and max(full_voice_strength, direct_voice_strength) >= 0.70
        ):
            protected_hit_roles = {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}
            direct_pitched_strength = max(
                _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
                _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
            )
            vocal_phrase_strength = max(
                _role_strength_from_facts(facts, "vocal_music_phrase"),
                _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
            )
            if role not in protected_hit_roles and not (
                direct_pitched_strength >= 0.84 and vocal_phrase_strength < 0.55
            ):
                return raw
        if (
            direct_voice_strength >= 0.70
            and raw.final_top == "Drums"
            and _path_has_any(raw_path, ("percussion", "rim", "clap", "snare", "tom", "cymbal"))
        ):
            return self._redirect_from_raw(
                raw,
                "FX/Human and Voice FX",
                "direct/body voice evidence beat tail-heavy drum one-shot reading",
            )

        # Short-hit guard: shape alone can misread short percussive hits as
        # bass_phrase/hit_with_tail and the older shape sanity layer may then
        # choose a Bass/Guitar/Keys one-shot or generic Instrument Loops.  Do
        # not turn that into a musical loop.  If close drum one-shot evidence
        # exists, route to a broad drum one-shot bucket; otherwise review.
        # This uses only voter candidates and measured shape, never filenames.
        if (
            shape in {"bass_phrase", "hit_with_tail"}
            and shape_conf >= 0.72
            and not _is_measured_loop_phrase_context(shape, role, measured_role)
        ):
            raw_is_suspicious_one_shot = (
                shape == "hit_with_tail"
                or _path_has_any(raw_path, ("one shots", "one shot"))
                or (
                    raw.final_top == "Instruments"
                    and _path_has_any(raw_path, ("instrument loops", "mixed musical loops"))
                )
            )
            if raw_is_suspicious_one_shot and not _path_has_any(raw_path, ("drum loops", "looped")):
                # A safety guard may block a dangerous cross-family mistake, but
                # it must not demote a supported kick into a tom/percussion leaf.
                # If the raw consensus already landed on a kick, keep that raw
                # decision before considering nearby drum-family alternatives.
                if raw.final_top == "Drums" and _path_has_any(raw_path, ("kick", "kick drums")):
                    return raw

                # A measured voice-like one-shot with actual Human/Voice
                # candidate support should not be converted to generic
                # percussion by the short-hit guard.  If the voice support is
                # not strong enough to place, review is safer than a false drum
                # leaf.  Plain voiced/noisy percussion without a Human/Voice
                # candidate, such as small rim/stick hits, can still fall
                # through to the drum guard below.
                short_hit_voice_candidate = self._best_candidate(
                    raw,
                    include_top={"FX", "Instruments"},
                    include_fragments=(
                        "human and voice",
                        "voice",
                        "vocal",
                        "vox",
                        "spoken",
                        "choir",
                        "breath",
                        "crowd",
                    ),
                )
                short_hit_voice_strength = max(
                    _role_strength_from_facts(facts, "voiced_one_shot"),
                    _direct_body_role_strength_from_facts(facts, "voiced_one_shot"),
                    _role_strength_from_facts(facts, "vocal_music_phrase"),
                    _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
                )
                if short_hit_voice_candidate is not None and short_hit_voice_strength >= 0.70:
                    voice_score, _voice_path = short_hit_voice_candidate
                    competing_drum_for_voice = self._best_candidate(
                        raw,
                        include_top={"Drums"},
                        include_fragments=(
                            "kick",
                            "tom",
                            "snare",
                            "clap",
                            "rim",
                            "hat",
                            "cymbal",
                            "percussion",
                            "conga",
                            "bongo",
                            "tabla",
                        ),
                        exclude_fragments=("drum loop", "drum loops", "loops"),
                    )
                    drum_dominates_voice = bool(
                        competing_drum_for_voice is not None and voice_score > competing_drum_for_voice[0] + 18.0
                    )
                    if not drum_dominates_voice:
                        if voice_score <= raw_score + 25.0 or raw_score >= 999.0:
                            return self._redirect_from_raw(
                                raw,
                                "FX/Human and Voice FX",
                                "positive voice one-shot claim blocked short-hit percussion demotion",
                            )
                        return self._review_from_raw(
                            raw,
                            eligibility,
                            "voice one-shot claim blocked short-hit percussion demotion but candidate support was weak",
                        )

                best_drum_hit = self._best_candidate(
                    raw,
                    include_top={"Drums"},
                    include_fragments=(
                        "kick",
                        "tom",
                        "snare",
                        "clap",
                        "rim",
                        "hat",
                        "cymbal",
                        "percussion",
                        "conga",
                        "bongo",
                        "tabla",
                    ),
                    exclude_fragments=("drum loop", "drum loops", "loops"),
                )
                best_kick_hit = self._best_candidate(
                    raw,
                    include_top={"Drums"},
                    include_fragments=("kick",),
                    exclude_fragments=("drum loop", "drum loops", "loops"),
                )
                best_inst_hit_score = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
                compare_score = best_inst_hit_score if best_inst_hit_score is not None else raw_score
                if best_drum_hit is not None:
                    drum_score, drum_path = best_drum_hit
                    # For bass_phrase, only a clearly better drum candidate, or
                    # a close kick candidate, may redirect to Drums.  Otherwise
                    # use review to avoid stealing short pitched instrument hits.
                    drum_margin = 10.0 if shape == "hit_with_tail" else 2.5
                    kick_margin = 6.0 if shape == "bass_phrase" else 3.0
                    if drum_score <= compare_score + drum_margin:
                        if best_kick_hit is not None and best_kick_hit[0] <= min(
                            drum_score + 3.0, compare_score + kick_margin
                        ):
                            return self._redirect_from_raw(
                                raw,
                                "Drums/Kick Drums/Generic Kick/One Shots",
                                "short-hit guard: low/bass-shaped one-shot had close kick candidate evidence",
                            )
                        if _path_has_any(_norm_path(drum_path), ("snare", "clap", "rim", "hat", "cymbal", "tom")):
                            return self._redirect_from_raw(
                                raw,
                                drum_path,
                                "short-hit guard: measured hit shape had close drum one-shot candidate evidence",
                            )
                        return self._redirect_from_raw(
                            raw,
                            "Drums/Percussion/Generic Percussion/One Shots",
                            "short-hit guard: measured hit shape had close percussion candidate evidence",
                        )
                if raw.final_top == "Instruments" and _path_has_any(raw_path, ("bass", "808", "sub bass")):
                    return None
                if raw.final_top == "Instruments":
                    return self._review_from_raw(
                        raw,
                        eligibility,
                        "short-hit guard: measured hit shape prevented unsafe Instrument Loop/Bass one-shot rescue without enough drum evidence",
                    )

        voice_fragments = ("human and voice", "voice", "vocal", "vox", "choir", "spoken", "breath", "crowd")
        nonvoice_fragments = (
            "siren",
            "alarm",
            "beep",
            "blip",
            "chime",
            "bell",
            "whoosh",
            "swoosh",
            "swish",
            "sweep",
            "seagull",
            "seaguls",
            "bird",
            "animal",
            "clank",
            "metallic",
            "hybrid designed",
            "designed noise",
            "boom",
            "impact",
            "slam",
            "hit",
            "sub hit",
            "short impact",
            "trumpet",
            "rhodes",
            "keys",
            "guitar",
            "percussion",
            "conga",
            "tabla",
        )
        drum_loop_fragments = (
            "drum loop",
            "drum loops",
            "break",
            "breaks",
            "kick",
            "percussion",
            "conga",
            "bongo",
            "tabla",
            "triangle",
            "wood block",
            "wood blocks",
            "metallic percussion",
        )
        tonal_fx_fragments = (
            "bell",
            "bells",
            "chime",
            "chimes",
            "riser",
            "build",
            "sweep",
            "whoosh",
            "hybrid designed",
            "designed tonal",
            "metallic",
        )

        best_voice_candidate = self._best_candidate(
            raw, include_top={"FX", "Instruments"}, include_fragments=voice_fragments
        )
        best_voice = best_voice_candidate[0] if best_voice_candidate is not None else None
        best_voice_path = best_voice_candidate[1] if best_voice_candidate is not None else ""
        best_voice_role_strength = 0.0
        best_voice_candidate_role_strength = 0.0
        for _candidate in raw.shared_candidates or []:
            _folder = _norm_path(str(_candidate.get("folder_path") or _candidate.get("label") or ""))
            if _path_has_any(_folder, voice_fragments):
                candidate_voice_strength = max(
                    _candidate_role_strength(_candidate, "vocal_music_phrase"),
                    _candidate_role_strength(_candidate, "vocal_phrase"),
                    _candidate_role_strength(_candidate, "vocal_one_shot"),
                    _candidate_role_strength(_candidate, "voiced_one_shot"),
                    # Spoken/rap vocal loops often carry pitched-phrase structure
                    # rather than a dedicated vocal-role score. Treat that as
                    # structural support only when the candidate itself is already
                    # in a Human/Voice folder.
                    _candidate_role_strength(_candidate, "pitched_music_phrase"),
                )
                best_voice_role_strength = max(best_voice_role_strength, candidate_voice_strength)
                if best_voice_path and _norm_path(best_voice_path) == _folder:
                    best_voice_candidate_role_strength = max(
                        best_voice_candidate_role_strength, candidate_voice_strength
                    )
        best_nonvoice = self._best_candidate(
            raw, include_top={"FX", "Drums", "Instruments"}, include_fragments=nonvoice_fragments
        )
        best_drum_loop = self._best_candidate(raw, include_top={"Drums"}, include_fragments=drum_loop_fragments)
        best_inst_any = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
        best_fx_any = self._best_candidate_score(raw, include_top={"FX"}, include_fragments=())
        measured_voice_strength = max(
            _role_strength_from_facts(facts, "voiced_one_shot"),
            _role_strength_from_facts(facts, "vocal_music_phrase"),
        )
        direct_voice_strength = max(
            _direct_body_role_strength_from_facts(facts, "voiced_one_shot"),
            _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
        )
        best_competing_instrument = best_inst_any
        best_competing_drum = best_drum_loop[0] if best_drum_loop is not None else None
        best_competing_nonvoice = best_nonvoice[0] if best_nonvoice is not None else None
        best_competitor = min(
            [
                score
                for score in (best_competing_instrument, best_competing_drum, best_competing_nonvoice)
                if score is not None
            ],
            default=None,
        )
        voice_candidate_is_actual_voice_path = best_voice is not None and _path_has_any(
            _norm_path(best_voice_path),
            ("human and voice", "voice", "vocal", "vox", "spoken", "choir", "breath", "crowd"),
        )
        measured_voice_claim_strength = max(measured_voice_strength, direct_voice_strength)
        stable_instrument_claim_strength = _stable_instrument_claim_strength(
            role, measured_role, shape, shape_conf, facts
        )
        human_voice_claim = build_human_voice_claim(
            has_voice_candidate=voice_candidate_is_actual_voice_path,
            voice_candidate_score=best_voice,
            raw_score=raw_score,
            competitor_score=best_competitor,
            voice_role_strength=measured_voice_claim_strength,
            candidate_role_strength=max(best_voice_role_strength, best_voice_candidate_role_strength),
            stable_instrument_claim_strength=stable_instrument_claim_strength,
        )
        positive_voice_claim = human_voice_claim.can_override
        direct_one_shot_voice_claim = bool(
            voice_candidate_is_actual_voice_path
            and best_voice is not None
            and not bool(getattr(facts, "is_loop_like", False))
            and _direct_body_role_strength_from_facts(facts, "voiced_one_shot") >= 0.88
            and (raw_score >= 999.0 or best_voice <= raw_score + 25.0)
        )

        duration_sec = _feature_number_from_facts(facts, "duration_sec")
        event_count = _feature_number_from_facts(facts, "event_count_estimate")
        short_single_event = bool(
            duration_sec > 0.0
            and duration_sec <= 0.35
            and event_count <= 2.0
            and shape in {"hit_with_tail", "single_hit", "short_hit"}
        )
        if (
            short_single_event
            and raw.final_top == "FX"
            and "human and voice" in raw_path
            and not positive_voice_claim
            and not direct_one_shot_voice_claim
        ):
            return self._review_from_raw(
                raw,
                eligibility,
                "short single-event voice-like reading lacked a positive Human/Voice claim",
            )

        instrument_phrase_strength = max(
            _role_strength_from_facts(facts, "pitched_music_phrase"),
            _role_strength_from_facts(facts, "pitched_music_loop"),
            _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
            _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
        )
        weak_voice_false_positive = bool(
            raw.final_top == "FX"
            and "human and voice" in raw_path
            and instrument_phrase_strength >= 0.84
            and not positive_voice_claim
            and not direct_one_shot_voice_claim
        )
        if weak_voice_false_positive:
            best_instrument = self._best_candidate(
                raw,
                include_top={"Instruments"},
                include_fragments=(
                    "instrument loops",
                    "brass",
                    "woodwind",
                    "sax",
                    "trumpet",
                    "piano",
                    "keys",
                    "rhodes",
                    "guitar",
                    "strings",
                    "synth",
                ),
            )
            if best_instrument is not None:
                _instrument_score, instrument_path = best_instrument
                safe_path = instrument_path
                if not _path_has_any(
                    _norm_path(safe_path),
                    (
                        "brass",
                        "woodwind",
                        "sax",
                        "instrument loops",
                        "piano",
                        "keys",
                        "rhodes",
                        "guitar",
                        "strings",
                        "synth",
                    ),
                ):
                    safe_path = "Instruments/Instrument Loops/Loops"
                return self._redirect_from_raw(
                    raw,
                    safe_path,
                    "stable pitched instrument claim blocked weak Human/Voice false positive",
                )
            return self._redirect_from_raw(
                raw,
                "Instruments/Instrument Loops/Loops",
                "stable pitched instrument claim blocked weak Human/Voice false positive",
            )

        # Decisive Human/Voice candidate rescue. Some rap/vocal loops are
        # measured as pitched-percussion or generic pitched loops because they
        # are rhythmic and tonal. If the ranked candidates put Human/Voice far
        # ahead of the raw loop and far ahead of any drum-loop support, keep the
        # broad voice family instead of forcing Drums. This uses voter evidence,
        # not producer names.
        best_drum_for_voice_check = best_drum_loop[0] if best_drum_loop is not None else None
        voice_beats_raw = best_voice is not None and best_voice <= raw_score - 8.0
        voice_beats_drum = best_voice is not None and (
            best_drum_for_voice_check is None or best_voice <= best_drum_for_voice_check - 8.0
        )
        voice_beats_nonvoice = best_voice is not None and (
            best_nonvoice is None or best_voice <= best_nonvoice[0] - 6.0
        )
        decisive_voice_candidate = (
            positive_voice_claim and voice_beats_raw and voice_beats_drum and voice_beats_nonvoice
        )
        if decisive_voice_candidate and raw.final_top in {"Instruments", "Drums", "FX"}:
            return self._redirect_from_raw(
                raw,
                "FX/Human and Voice FX",
                "decisive Human/Voice candidate evidence beat false drum/instrument loop rescue",
            )

        # Direct/body voice rescue.  Full-file facts can hear a reverbed vocal
        # shot as percussion because the transient is sharp, while the direct
        # body view exposes the voiced/formant identity.  This redirect only
        # uses measured audio roles plus optional Human/Voice candidate support;
        # it does not read source filenames.
        if max(measured_voice_strength, direct_voice_strength) >= 0.72:
            raw_is_drum_or_generic = raw.final_top == "Drums" or _path_has_any(
                raw_path, ("percussion", "rim", "clap", "instrument loops", "mixed musical loops")
            )
            voice_role_names = {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
            strong_measured_vocal_shape = bool(
                shape == "vocal_phrase"
                and shape_conf >= 0.88
                and max(measured_voice_strength, direct_voice_strength) >= 0.84
                and (role in voice_role_names or measured_role in voice_role_names or positive_voice_claim)
            )
            strong_direct_vocal_hit = bool(
                shape in {"hit_with_tail", "single_hit"}
                and direct_voice_strength >= 0.90
                and not _has_drum_loop_structure_support(role, measured_role, shape, shape_conf)
            )
            if raw_is_drum_or_generic and (
                positive_voice_claim
                or direct_one_shot_voice_claim
                or strong_measured_vocal_shape
                or strong_direct_vocal_hit
            ):
                return self._redirect_from_raw(
                    raw,
                    "FX/Human and Voice FX",
                    "positive measured Human/Voice claim beat tail-heavy percussive or generic-loop reading",
                )

        if (
            role in {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
            and direct_voice_strength >= 0.85
            and shape == "vocal_phrase"
            and shape_conf >= 0.88
            and (positive_voice_claim or direct_one_shot_voice_claim)
        ):
            return self._redirect_from_raw(
                raw,
                "FX/Human and Voice FX",
                "positive Human/Voice claim beat tail-heavy non-voice candidate set",
            )

        low_rhythmic_drum_strength = max(
            _role_strength_from_facts(facts, "low_rhythmic_drum_loop"),
            _role_strength_from_facts(facts, "percussive_drum_loop"),
            _direct_body_role_strength_from_facts(facts, "low_rhythmic_drum_loop"),
            _direct_body_role_strength_from_facts(facts, "percussive_drum_loop"),
        )

        # Weak reed-like roles must not over-narrow raw FX/vocal material into
        # Brass/Woodwinds when the Brass/Woodwind candidate itself is distant.
        # A measured Saxophone loop claim is stronger than a generic reed-like
        # role and may rescue Human/Voice or alarm-like false positives into the
        # broad Saxophone loop bucket.
        if role in {"pitched_reed_or_instrument_loop", "pitched_reed_or_instrument_phrase"}:
            broad_path = _norm_path(eligibility.broad_folder_path)
            best_reed = self._best_candidate_score(
                raw, include_top={"Instruments"}, include_fragments=("brass", "woodwind", "sax", "reed")
            )
            if _path_has_any(broad_path, ("brass", "woodwind", "sax", "reed")) and raw.final_top != "Instruments":
                sax_claim = "woodwinds/saxophone" in broad_path or "saxophone" in broad_path
                raw_fx_false_positive = raw.final_top == "FX" and _path_has_any(
                    raw_path,
                    ("human and voice", "voice", "vocal", "siren", "alarm", "beep"),
                )
                if sax_claim and raw_fx_false_positive:
                    if best_reed is None or best_reed > raw_score + 10.0:
                        return self._review_from_raw(
                            raw,
                            eligibility,
                            "reed/woodwind over-narrowing conflict: sax-like measured role lacked close reed candidate support",
                        )
                    return self._broaden_from_raw(
                        raw,
                        eligibility,
                        "measured sax/woodwind claim beat non-reed FX or Human/Voice false positive",
                    )
                if best_reed is None or best_reed > raw_score + 10.0:
                    return self._review_from_raw(
                        raw,
                        eligibility,
                        "reed/woodwind over-narrowing conflict: weak reed-like role could not safely override raw non-reed candidate evidence",
                    )

        # Vocal/voice false-positive guard. If the eligibility path wants broad
        # Human/Voice but non-voice candidates are clearly stronger, review when
        # raw is not already Human/Voice; if raw is Human/Voice, redirect to the
        # best non-voice FX candidate when available.
        if role in {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}:
            voice_identity = _voice_identity_from_facts(facts)
            strong_measured_voice_broad = bool(
                eligibility.confidence >= 0.78
                and shape == "vocal_phrase"
                and shape_conf >= 0.78
                and max(measured_voice_strength, direct_voice_strength) >= 0.78
                and voice_identity >= 0.68
                and measured_role
                not in {
                    "pitched_reed_or_instrument_loop",
                    "pitched_reed_or_instrument_phrase",
                    "bass_loop",
                    "drum_loop",
                    "low_rhythmic_drum_loop",
                    "percussive_drum_loop",
                }
            )
            if positive_voice_claim or strong_measured_voice_broad:
                reason = (
                    "vocal true-bucket rescue: positive Human/Voice claim beat unsafe non-voice family"
                    if positive_voice_claim
                    else "measured vocal phrase/one-shot evidence beat unsafe non-voice candidate set"
                )
                return self._redirect_from_raw(
                    raw,
                    "FX/Human and Voice FX",
                    reason,
                )
            if best_nonvoice is not None:
                nonvoice_score, nonvoice_path = best_nonvoice
                if best_voice is None or nonvoice_score <= best_voice - 2.0:
                    nonvoice_low = _norm_path(nonvoice_path)
                    phrase_like = shape in {"vocal_phrase", "pitched_phrase"} and shape_conf >= 0.72
                    abstract_tone_fx = _path_has_any(nonvoice_low, ("beep", "siren", "alarm", "blip"))
                    concrete_fx = _path_has_any(
                        nonvoice_low,
                        (
                            "whoosh",
                            "swoosh",
                            "swish",
                            "sweep",
                            "seagull",
                            "seaguls",
                            "bird",
                            "animal",
                            "clank",
                            "metallic",
                            "bell",
                            "chime",
                            "hybrid designed",
                            "riser",
                            "build",
                            "transition",
                        ),
                    )
                    if raw.final_top == "FX" and "human and voice" in raw_path and nonvoice_low.startswith("fx/"):
                        if phrase_like and abstract_tone_fx and not concrete_fx:
                            return self._redirect_from_raw(
                                raw,
                                "Instruments/Brass and Woodwinds/Loops",
                                "reed-like phrase rescue: phrase-shaped Human/Voice false positive had only abstract beep/siren FX candidates",
                            )
                        return self._redirect_from_raw(
                            raw,
                            nonvoice_path,
                            "non-voice FX true-bucket rescue: stronger non-voice FX candidate beat Human/Voice FX",
                        )
                    return self._review_from_raw(
                        raw,
                        eligibility,
                        "voice false-positive conflict: broad vocal role had clearly stronger non-voice candidate evidence",
                    )

        if role == "drum_loop" and low_rhythmic_drum_strength >= 0.75 and shape != "vocal_phrase":
            return self._redirect_from_raw(
                raw,
                "Drums/Drum Loops/Loops",
                "low-rhythmic direct/full body evidence beat bass/FX loop reading",
            )

        # Vocal loop versus drum-loop conflict. Do not let repeated-event logic
        # force a structural vocal candidate set into Drum Loops. Weak Breath/Voice
        # candidates without vocal-role support must not veto a measured drum loop.
        if role == "drum_loop" and best_voice is not None:
            best_drum_score = (
                best_drum_loop[0]
                if best_drum_loop is not None
                else self._best_candidate_score(raw, include_top={"Drums"}, include_fragments=())
            )
            voice_is_structural = (
                best_voice_candidate_role_strength >= 0.70
                or measured_voice_strength >= 0.72
                or direct_voice_strength >= 0.72
            )
            voice_blocks_generic_instrument = (
                voice_is_structural
                and raw.final_top == "Instruments"
                and best_inst_any is not None
                and best_voice <= best_inst_any + 2.0
                and (best_drum_score is None or best_drum_score >= best_voice + 6.0)
            )
            if (voice_is_structural or voice_blocks_generic_instrument) and (
                best_drum_score is None or best_voice <= best_drum_score - 3.0
            ):
                return self._review_from_raw(
                    raw,
                    eligibility,
                    "vocal loop conflict: drum-loop role had stronger structural human/voice candidate evidence",
                )

        # Drum/kick/percussion loop true-bucket rescue. This intentionally runs
        # even when the measured role is bass_loop/pitched_music_loop, because
        # kick loops and pitched percussion were repeatedly flattened to Bass or
        # generic Instrument Loops. Voice-dominant candidate sets are protected
        # by the vocal conflict guard above.
        if best_drum_loop is not None:
            drum_score, drum_path = best_drum_loop
            raw_wrong_for_drum = (
                (
                    raw.final_top == "Instruments"
                    and _path_has_any(raw_path, ("bass loops", "instrument loops", "mixed musical loops"))
                )
                or (
                    raw.final_top == "Instruments"
                    and _path_has_any(raw_path, ("bass", "808", "sub bass", "synth bass"))
                    and _path_has_any(raw_path, ("one shot", "one shots"))
                    and _has_drum_loop_structure_support(role, measured_role, shape, shape_conf)
                )
                or (
                    raw.final_top == "FX"
                    and _path_has_any(raw_path, ("human and voice", "voice", "vocal", "breath", "crowd"))
                )
            )
            # Do not let a stray close drum candidate steal a sax/reed/vocal-like
            # pitched phrase into Drum Loops.  This guard protects the real FX
            # smoke sax failures while still allowing confirmed kick/drum loop
            # rescues when measured role/shape carries drum-loop evidence.
            drum_structure_support = _has_drum_loop_structure_support(role, measured_role, shape, shape_conf) or (
                shape == "bass_phrase" and best_drum_loop is not None and drum_score <= raw_score + 4.0
            )
            phrase_like_non_drum = (
                shape in {"vocal_phrase", "pitched_phrase", "sustained_pad"} and not drum_structure_support
            )
            if raw_wrong_for_drum and not phrase_like_non_drum:
                against = (
                    best_inst_any
                    if raw.final_top == "Instruments"
                    else (best_voice if best_voice is not None else best_fx_any)
                )
                if against is None:
                    against = raw_score
                if drum_score <= against + 5.0:
                    reason = (
                        "percussion-loop true-bucket rescue"
                        if _path_has_any(
                            _norm_path(drum_path),
                            ("percussion", "conga", "bongo", "tabla", "triangle", "wood block", "metallic"),
                        )
                        else "kick/drum-loop true-bucket rescue"
                    )
                    return self._redirect_from_raw(
                        raw,
                        "Drums/Drum Loops/Loops",
                        f"{reason}: close drum/percussion candidate beat unsafe generic/FX family",
                    )

        # A measured pitched percussion loop is Drums even when the raw
        # winner is a generic Instrument Loop and an FX riser/build candidate is
        # closer than the drum candidate.  This is a role-level structural fact,
        # not a filename rule.
        if role == "pitched_percussion_loop" and best_drum_loop is not None:
            if _shape_blocks_percussion_loop_rescue(shape, shape_conf):
                if raw.final_top in {"Instruments", "FX"} and not raw.folder_path.startswith("_TO_REVIEW"):
                    return raw
                return None
            return self._broaden_from_raw(
                raw,
                eligibility,
                "percussion-loop true-bucket rescue: measured pitched_percussion_loop overruled generic instrument/FX candidates",
            )

        # Raw Human/Voice FX false-positive rescue.  A sax/reed-like phrase can
        # be misread as broad Human/Voice while the nearest non-voice FX
        # candidates are Beep/Siren/Alarm.  If the measured shape is a phrase,
        # route to a broad brass/woodwind instrument bucket rather than FX.
        # Keep obvious non-voice FX such as seagulls, whooshes, clanks, bells,
        # or hybrid-designed sounds in FX.
        if raw.final_top == "FX" and "human and voice" in raw_path and best_nonvoice is not None:
            nonvoice_score, nonvoice_path = best_nonvoice
            nonvoice_low = _norm_path(nonvoice_path)
            phrase_like = shape in {"vocal_phrase", "pitched_phrase"} and shape_conf >= 0.72
            nonvoice_is_abstract_tone = _path_has_any(nonvoice_low, ("beep", "siren", "alarm", "blip"))
            nonvoice_is_concrete_fx = _path_has_any(
                nonvoice_low,
                (
                    "whoosh",
                    "swoosh",
                    "swish",
                    "sweep",
                    "seagull",
                    "seaguls",
                    "bird",
                    "animal",
                    "clank",
                    "metallic",
                    "bell",
                    "chime",
                    "hybrid designed",
                    "riser",
                    "build",
                    "transition",
                ),
            )
            if phrase_like and nonvoice_is_abstract_tone and not nonvoice_is_concrete_fx:
                return self._redirect_from_raw(
                    raw,
                    "Instruments/Brass and Woodwinds/Loops",
                    "reed-like phrase rescue: phrase-shaped Human/Voice false positive had only abstract beep/siren FX candidates",
                )
            if nonvoice_low.startswith("fx/") and (best_voice is None or nonvoice_score <= best_voice - 2.0):
                return self._redirect_from_raw(
                    raw,
                    nonvoice_path,
                    "non-voice FX true-bucket rescue: stronger non-voice FX candidate beat Human/Voice FX",
                )

        # Tonal metallic/build FX rescue. A generic Instrument Loop should not
        # be stolen by an ordinary close Build/Riser candidate. Require a
        # metallic/bell/chime/hybrid anchor candidate first, then choose the
        # best close tonal FX candidate as the broad FX landing.
        if raw.final_top == "Instruments" and _path_has_any(raw_path, ("instrument loops", "mixed musical loops")):
            anchor_fx = self._best_candidate(
                raw,
                include_top={"FX"},
                include_fragments=("bell", "bells", "chime", "chimes", "hybrid designed", "designed tonal", "metallic"),
            )
            best_tonal_fx = self._best_candidate(raw, include_top={"FX"}, include_fragments=tonal_fx_fragments)
            if anchor_fx is not None and best_tonal_fx is not None:
                anchor_score, _anchor_path = anchor_fx
                fx_score, fx_path = best_tonal_fx
                if anchor_score <= raw_score + 5.0 and fx_score <= raw_score + 5.0:
                    return self._redirect_from_raw(
                        raw,
                        fx_path,
                        "tonal FX true-bucket rescue: close tonal/designed FX candidate beat generic Instrument Loop",
                    )

        return None
