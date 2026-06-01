# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Parent-role eligibility policy for final placement safety.

This module is deliberately broad.  It does not decide that a sound is a sax,
dog, coin, tom, or vocal.  It only decides which parent families or branches are
physically eligible enough for final placement.
"""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.eligibility_decision import EligibilityDecision


def infer_parent_eligibility(facts: SharedAudioFacts) -> EligibilityDecision:
    """Infer broad parent eligibility from shared facts and measured roles.

    The thresholds here intentionally work at parent-role level.  They block
    catastrophic families such as drum loops landing in FX ambience or vocal
    one-shots landing in animal FX.  They do not identify the final leaf.
    """
    evidence = facts.evidence if isinstance(facts.evidence, dict) else {}
    roles = evidence.get("measured_roles", {})
    if not isinstance(roles, dict):
        roles = {}
    role_evidence = roles.get("evidence", {})
    if not isinstance(role_evidence, dict):
        role_evidence = {}
    feature_values = evidence.get("feature_values_by_name", {})
    if not isinstance(feature_values, dict):
        feature_values = facts.feature_values_by_name or {}

    duration = _num(evidence.get("duration_sec"), facts.feature_values_by_name.get("duration_sec", 0.0))
    event_count = _num(evidence.get("event_count_estimate"), 0.0)
    event_rate = _num(evidence.get("event_rate_hz"), 0.0)
    onset_span = _num(evidence.get("onset_span_ratio"), 0.0)
    attack_rise = _num(evidence.get("attack_rise_time_norm"), 0.0)
    temporal_centroid = _num(evidence.get("temporal_centroid_ratio"), 0.0)

    pitch_conf = _num(feature_values.get("pitch_confidence"), 0.0)
    f0_voiced = _num(feature_values.get("f0_voiced_ratio"), 0.0)
    formant = _num(feature_values.get("formant_like_peak_spacing"), 0.0)
    low_total = _num(feature_values.get("sub_bass_ratio_lt_150hz"), 0.0) + _num(
        feature_values.get("bass_ratio_150_500hz"), 0.0
    )
    high_total = _num(feature_values.get("presence_ratio_2000_8000hz"), 0.0) + _num(
        feature_values.get("air_ratio_gt_8000hz"), 0.0
    )
    mid_total_value = _num(feature_values.get("mid_ratio_500_2000hz"), 0.0) + _num(
        feature_values.get("presence_ratio_2000_8000hz"), 0.0
    )
    loop_percussive = _num(feature_values.get("loop_percussive_event_ratio"), 0.0)
    loop_drumlike = _num(feature_values.get("loop_drumlike_frame_ratio"), 0.0)
    loop_pitched = _num(feature_values.get("loop_pitched_event_ratio"), 0.0)
    loop_sustained = _num(feature_values.get("loop_sustained_tonal_frame_ratio"), 0.0)
    loop_non_event_tonal = _num(feature_values.get("loop_non_event_tonal_ratio"), 0.0)
    shape_vote = evidence.get("shape_vote", {})
    if not isinstance(shape_vote, dict):
        shape_vote = {}
    primary_shape = str(shape_vote.get("primary_shape", ""))
    shape_confidence = _num(shape_vote.get("confidence"), 0.0)
    tonal_alert_siren_score = _num(evidence.get("tonal_alert_siren_score"), 0.0)

    # A real transition FX needs more than a rising/falling centroid.
    # Drum loops with bass can show spectral slope over time, but they usually
    # remain low/body-heavy and rhythmically pulsed.  Require clear high/noisy
    # transition energy or a late/slow envelope so drum loops do not get stolen
    # by FX just because their spectrum drifts upward.
    fx_transition_motion = (
        primary_shape in {"transition_riser", "transition_drop"}
        and shape_confidence >= 0.80
        and duration >= 1.20
        and event_count >= 3.0
        and onset_span >= 0.45
        and loop_non_event_tonal <= 0.35
        and (high_total >= 0.35 or temporal_centroid >= 0.68 or attack_rise >= 0.45)
        and not (low_total >= 0.55 and (loop_drumlike >= 0.25 or loop_percussive >= 0.20) and high_total < 0.25)
    )

    voice_identity_score = max(
        _role(roles, "voiced_one_shot"),
        _num(role_evidence.get("formant_light_voice_identity"), 0.0),
    )
    clean_sustained_tonal_instrument_loop = (
        duration >= 2.5
        and f0_voiced >= 0.75
        and loop_pitched >= 0.85
        and loop_sustained >= 0.75
        and loop_non_event_tonal >= 0.75
        and loop_drumlike <= 0.12
        and loop_percussive <= 0.12
        and voice_identity_score < 0.45
    )
    short_front_loaded_tonal_hit = (
        duration <= 1.25
        and event_count <= 3.0
        and _num(feature_values.get("log_crest"), 0.0) >= 1.70
        and _num(feature_values.get("attack_rise_time_norm"), attack_rise) <= 0.12
        and _num(feature_values.get("temporal_centroid_ratio"), temporal_centroid) <= 0.20
        and _num(feature_values.get("tail_energy_ratio"), 0.0) <= 0.12
        and voice_identity_score < 0.62
    )
    processed_vocal_loop_or_stab = (
        duration >= 0.35
        and f0_voiced >= 0.60
        and loop_pitched >= 0.70
        and loop_sustained >= 0.60
        and loop_non_event_tonal >= 0.55
        and low_total <= 0.75
        and loop_drumlike <= 0.18
        and loop_percussive <= 0.22
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.28
        and formant >= 0.45
        and not clean_sustained_tonal_instrument_loop
        and not short_front_loaded_tonal_hit
        and (
            voice_identity_score >= 0.45
            or (
                duration <= 3.0
                and primary_shape in {"vocal_phrase", "vocal_one_shot", "hit_with_tail"}
                and shape_confidence >= 0.72
                and (mid_total_value + high_total) >= 0.68
            )
        )
        and (high_total >= 0.05 or _num(feature_values.get("air_ratio_gt_8000hz"), 0.0) >= 0.015 or duration <= 3.0)
        and not (
            duration <= 0.65
            and formant < 1.0
            and low_total >= 0.25
            and _num(feature_values.get("log_crest"), 0.0) >= 1.40
            and temporal_centroid <= 0.45
        )
    )
    tonal_alert_or_siren_fx = (
        duration >= 2.0
        and event_count >= 8.0
        and event_rate >= 4.5
        and f0_voiced >= 0.80
        and loop_pitched >= 0.85
        and loop_sustained >= 0.85
        and loop_non_event_tonal >= 0.80
        and loop_percussive <= 0.08
        and loop_drumlike <= 0.08
        and formant >= 2.0
        and low_total <= 0.28
        and (tonal_alert_siren_score >= 0.62 or not clean_sustained_tonal_instrument_loop)
        and (
            primary_shape not in {"vocal_phrase", "bass_phrase"}
            or (tonal_alert_siren_score >= 0.70 and duration >= 5.0 and low_total <= 0.05 and high_total <= 0.08)
        )
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.08
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.25
    )

    clean_tonal_low_loop = (
        duration >= 3.5
        and event_count >= 4.0
        and pitch_conf >= 0.55
        and low_total >= 0.55
        and loop_pitched >= 0.65
        and loop_sustained >= 0.55
        and loop_non_event_tonal >= 0.55
        and loop_percussive <= 0.08
        and loop_drumlike <= 0.08
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.06
    )
    choppy_music_like_loop = (
        duration >= 4.0
        and event_count >= 8.0
        and loop_pitched >= 0.40
        and formant >= 1.50
        and low_total <= 0.45
        and loop_drumlike <= 0.08
    )
    sustained_melodic_nonpercussive_loop = (
        duration >= 2.0
        and f0_voiced >= 0.70
        and loop_pitched >= 0.92
        and loop_sustained >= 0.72
        and loop_non_event_tonal >= 0.70
        and ((loop_percussive <= 0.08 and loop_drumlike <= 0.08) or (loop_percussive <= 0.04 and loop_drumlike <= 0.18))
    )

    clean_sustained_tonal_non_drum_loop = (
        duration >= 3.0
        and loop_pitched >= 0.75
        and loop_sustained >= 0.75
        and loop_non_event_tonal >= 0.75
        and f0_voiced >= 0.55
        and loop_percussive <= 0.20
        and loop_drumlike <= 0.20
        and not (low_total >= 0.55 and (loop_percussive >= 0.05 or loop_drumlike >= 0.05))
    )
    clean_low_tonal_non_drum_loop = (
        duration >= 3.0
        and low_total >= 0.55
        and _num(feature_values.get("sub_bass_ratio_lt_150hz"), 0.0) < 0.75
        and loop_pitched >= 0.90
        and loop_sustained >= 0.90
        and loop_non_event_tonal >= 0.90
        and loop_percussive <= 0.03
        and loop_drumlike <= 0.03
    )

    low_body_rhythmic_loop = (
        duration >= 3.5
        and event_count >= 10.0
        and event_rate >= 1.25
        and onset_span >= 0.65
        and low_total >= 0.55
        and _num(feature_values.get("log_crest"), 0.0) >= 1.25
        and not clean_tonal_low_loop
        and not clean_low_tonal_non_drum_loop
        and not clean_sustained_tonal_non_drum_loop
        and not choppy_music_like_loop
        and not sustained_melodic_nonpercussive_loop
        and not (
            loop_pitched >= 0.90
            and loop_sustained >= 0.90
            and loop_non_event_tonal >= 0.90
            and loop_percussive <= 0.05
            and loop_drumlike <= 0.05
            and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.12
        )
        and (
            loop_percussive >= 0.05
            or loop_drumlike >= 0.05
            or _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.16
            or high_total >= 0.035
        )
    )
    high_reed_loop = (
        duration >= 1.5
        and primary_shape in {"pitched_phrase", "sustained_pad", "bass_phrase"}
        and shape_confidence >= 0.70
        and pitch_conf >= 0.70
        and f0_voiced >= 0.72
        and loop_pitched >= 0.82
        and loop_sustained >= 0.65
        and loop_non_event_tonal >= 0.65
        and loop_percussive <= 0.10
        and loop_drumlike <= 0.14
        and low_total <= 0.28
        and 0.10 <= high_total <= 0.48
        and _num(feature_values.get("mid_ratio_500_2000hz"), 0.0) >= 0.25
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.38
    )
    low_mid_reed_loop = (
        duration >= 1.5
        and primary_shape in {"pitched_phrase", "sustained_pad", "bass_phrase", "vocal_phrase"}
        and shape_confidence >= 0.70
        and f0_voiced >= 0.80
        and loop_pitched >= 0.90
        and loop_sustained >= 0.76
        and loop_non_event_tonal >= 0.74
        and loop_percussive <= 0.08
        and loop_drumlike <= 0.16
        and _num(feature_values.get("sub_bass_ratio_lt_150hz"), 0.0) <= 0.06
        and 0.24 <= _num(feature_values.get("bass_ratio_150_500hz"), 0.0) <= 0.48
        and 0.34 <= _num(feature_values.get("mid_ratio_500_2000hz"), 0.0) <= 0.62
        and 0.03 <= high_total <= 0.22
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.36
        and _num(feature_values.get("event_rate_hz"), event_rate) <= 5.75
        and not (pitch_conf < 0.32 and _num(feature_values.get("spectral_flatness_mean"), 0.0) > 0.34)
    )
    dry_mid_reed_loop = (
        duration >= 1.5
        and primary_shape == "pitched_phrase"
        and shape_confidence >= 0.82
        and pitch_conf >= 0.74
        and f0_voiced >= 0.86
        and loop_pitched >= 0.90
        and loop_sustained >= 0.82
        and loop_non_event_tonal >= 0.82
        and loop_percussive <= 0.06
        and loop_drumlike <= 0.08
        and _num(feature_values.get("sub_bass_ratio_lt_150hz"), 0.0) <= 0.04
        and 0.055 <= _num(feature_values.get("bass_ratio_150_500hz"), 0.0) <= 0.22
        and _num(feature_values.get("mid_ratio_500_2000hz"), 0.0) >= 0.66
        and 0.035 <= high_total <= 0.18
        and _num(feature_values.get("air_ratio_gt_8000hz"), 0.0) <= 0.035
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.16
        and _num(feature_values.get("event_rate_hz"), event_rate) >= 4.0
    )
    sax_like_reed_loop = (
        (high_reed_loop or low_mid_reed_loop or dry_mid_reed_loop)
        and voice_identity_score < 0.45
        and not processed_vocal_loop_or_stab
        and not choppy_music_like_loop
    )
    reed_like_sustained_loop = (
        duration >= 1.25
        and (pitch_conf >= 0.12 or f0_voiced >= 0.75)
        and f0_voiced >= 0.52
        and loop_pitched >= 0.52
        and loop_sustained >= 0.38
        and loop_non_event_tonal >= 0.34
        and loop_percussive <= 0.18
        and loop_drumlike <= 0.24
        and low_total <= 0.72
        and high_total <= 0.42
        and (sax_like_reed_loop or _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.34)
        and (
            sax_like_reed_loop
            or (
                _num(feature_values.get("bass_ratio_150_500hz"), 0.0) >= 0.12
                and _num(feature_values.get("mid_ratio_500_2000hz"), 0.0) <= 0.76
                and event_rate <= 3.75
            )
            or (
                formant >= 1.15
                and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.10
                and event_rate <= 3.75
            )
        )
        and not processed_vocal_loop_or_stab
        and not clean_tonal_low_loop
        and not low_body_rhythmic_loop
        and not choppy_music_like_loop
    )
    rhythmic_dominant_loop = (
        duration >= 2.4
        and event_count >= 6.0
        and event_rate >= 0.65
        and onset_span >= 0.42
        and not clean_tonal_low_loop
        and not clean_low_tonal_non_drum_loop
        and not clean_sustained_tonal_non_drum_loop
        and not reed_like_sustained_loop
        and not choppy_music_like_loop
        and not sustained_melodic_nonpercussive_loop
        and (
            loop_percussive >= 0.12
            or loop_drumlike >= 0.12
            or (
                event_count >= 9.0
                and _num(feature_values.get("log_crest"), 0.0) >= 1.05
                and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.11
            )
            or (low_total >= 0.45 and high_total >= 0.025 and event_count >= 8.0 and loop_non_event_tonal <= 0.78)
        )
    )
    repeated_drum_event_loop = (
        duration >= 3.5
        and event_count >= 10.0
        and event_rate >= 1.25
        and onset_span >= 0.65
        and (loop_percussive >= 0.22 or loop_drumlike >= 0.20)
        and not (
            loop_percussive <= 0.05 and loop_pitched >= 0.85 and loop_sustained >= 0.65 and loop_non_event_tonal >= 0.60
        )
        and not clean_tonal_low_loop
        and not clean_low_tonal_non_drum_loop
        and not clean_sustained_tonal_non_drum_loop
        and not reed_like_sustained_loop
        and not choppy_music_like_loop
    )

    voiced_one_shot = _role(roles, "voiced_one_shot")
    pitched_music_loop = max(
        _role(roles, "pitched_music_loop"),
        _role(roles, "pitched_music_phrase"),
        _role(roles, "vocal_music_phrase"),
    )
    drum_loop = max(
        _role(roles, "bright_drum_loop"),
        _role(roles, "percussive_drum_loop"),
        _role(roles, "low_rhythmic_drum_loop"),
    )
    bass_loop = _role(roles, "bass_loop")
    percussive_one_shot = _role(roles, "percussive_one_shot")

    clean_pitched_phrase_context = (
        primary_shape in {"pitched_phrase", "vocal_phrase", "sustained_pad", "bass_phrase"}
        and shape_confidence >= 0.74
        and (pitch_conf >= 0.70 or (pitch_conf >= 0.62 and f0_voiced >= 0.88))
        and pitched_music_loop >= 0.78
        and drum_loop <= 0.36
        and loop_pitched >= 0.82
        and loop_sustained >= 0.72
        and loop_non_event_tonal >= 0.70
        and loop_percussive <= 0.10
        and loop_drumlike <= 0.10
    )
    pitched_percussion_conflict_loop = (
        duration >= 4.0
        and event_count >= 10.0
        and onset_span >= 0.65
        and _num(feature_values.get("log_crest"), 0.0) >= 2.35
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.16
        and loop_pitched >= 0.80
        and loop_sustained >= 0.75
        and loop_non_event_tonal >= 0.75
        and loop_percussive <= 0.08
        and loop_drumlike <= 0.08
        and not clean_pitched_phrase_context
    )
    pitched_percussion_loop = (
        duration >= 4.0
        and event_count >= 16.0
        and event_rate >= 3.0
        and onset_span >= 0.60
        and loop_pitched >= 0.75
        and loop_sustained >= 0.70
        and loop_non_event_tonal >= 0.70
        and (
            (pitch_conf <= 0.55 and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.16)
            or (high_total >= 0.70 and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.25)
            or (loop_drumlike >= 0.12 and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.18)
        )
        and not reed_like_sustained_loop
        and not (
            pitch_conf >= 0.75 and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.12 and event_rate <= 2.0
        )
    )

    # Real transition FX can look like repeated bright drum events because they
    # are noisy, rising/falling, and segmented.  If the shape voter finds a
    # strong transition envelope, protect it before drum-loop gates fire.
    if fx_transition_motion:
        broad_path = "FX/Structural and Transitional FX/Risers and Builds"
        role_name = "fx_transition_riser"
        if primary_shape == "transition_drop":
            broad_path = "FX/Structural and Transitional FX/Drops and Downlifters"
            role_name = "fx_transition_drop"
        return EligibilityDecision(
            role_name=role_name,
            confidence=max(shape_confidence, 0.82),
            allowed_top_families=("FX", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Drums",
                "Drum Loops",
                "Percussion",
                "Instruments",
                "Bass",
                "Voice",
                "Human",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Ocean",
                "Water",
                "Natural Ambience",
            ),
            broad_folder_path=broad_path,
            reason="measured transition FX envelope; drum, instrument, animal, voice, and ambience leaves are not eligible",
        )

    # Processed vocal loops/stabs are often tonal and can look like reeds, synths,
    # or generic instrument loops.  Protect them before reed/pitched fallbacks.
    if processed_vocal_loop_or_stab:
        return EligibilityDecision(
            role_name="vocal_phrase",
            confidence=max(voiced_one_shot, 0.78),
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Drums",
                "Percussion",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Machines",
                "Motor",
                "Engine",
                "Glitch",
                "Stutter",
                "Ocean",
                "Water",
                "Natural Ambience",
                "Drum Loops",
            ),
            broad_folder_path="FX/Human and Voice FX",
            reason="measured processed vocal loop/stab; drum, machine, animal, and ambience leaves are not eligible",
        )

    # Repeating tonal alert/siren/police-style FX should not be converted into
    # generic instrument loops only because they are pitched.
    if tonal_alert_or_siren_fx:
        return EligibilityDecision(
            role_name="fx_tonal_alert_or_siren",
            confidence=0.76,
            allowed_top_families=("FX", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Drums",
                "Drum Loops",
                "Percussion",
                "Instruments",
                "Bass",
                "Voice",
                "Human",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Ocean",
                "Water",
                "Natural Ambience",
            ),
            broad_folder_path="FX/Designed Noise FX/Alarm/Long FX",
            reason="measured repeated tonal alert/siren-like FX; drum, instrument, animal, and ambience leaves are not eligible",
        )

    if pitched_percussion_loop:
        return EligibilityDecision(
            role_name="pitched_percussion_loop",
            confidence=0.78,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=(
                "FX",
                "Human",
                "Voice",
                "Animals",
                "Dog",
                "Bird",
                "Water",
                "Natural Ambience",
                "Instruments",
            ),
            broad_folder_path="Drums/Drum Loops/Loops",
            reason="measured dense pitched percussion loop; generic instrument, voice, ambience, and FX leaves are not eligible",
        )

    if pitched_percussion_conflict_loop:
        return EligibilityDecision(
            role_name="pitched_percussion_conflict_loop",
            confidence=0.74,
            allowed_top_families=("Drums", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Human", "Voice", "Animals", "Dog", "Bird", "Water", "Natural Ambience"),
            broad_folder_path="_TO_REVIEW/Measured Role Conflict",
            reason="measured repeated high-crest pitched loop; could be pitched percussion or tonal loop, so do not force generic instruments",
        )

    # Sustained reed/brass/woodwind-like phrases can look like drum loops when
    # wet reverb creates repeated onsets.  Strong mid/high, clean, voiced reed
    # loops get a broad Saxophone loop bucket; weaker reed-like loops remain in
    # generic Instrument Loops so the parent seam does not over-narrow every
    # melodic loop.
    if reed_like_sustained_loop:
        broad_reed_path = (
            "Instruments/Woodwinds/Saxophone/Loops" if sax_like_reed_loop else "Instruments/Instrument Loops/Loops"
        )
        return EligibilityDecision(
            role_name="pitched_reed_or_instrument_loop",
            confidence=max(pitched_music_loop, 0.78),
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Drums",
                "Drum Loops",
                "Percussion",
                "FX",
                "Human",
                "Voice",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Machines",
                "Motor",
                "Ocean",
                "Water",
                "Natural Ambience",
            ),
            broad_folder_path=broad_reed_path,
            reason="measured sustained pitched reed/instrument loop; drum-loop, human voice, FX, animal, machine, and ambience leaves are not eligible",
        )

    # Structure-first protected drum loops.  This runs before pitched/bass loop
    # eligibility because low-heavy drum loops often look strongly pitched in
    # autocorrelation and centroid features.  The guard requires repeated events
    # spread across the file plus either drumlike/percussive loop evidence or
    # broadband/noisy body.  Clean tonal bass loops remain eligible for Bass.
    if low_body_rhythmic_loop or repeated_drum_event_loop or rhythmic_dominant_loop:
        return EligibilityDecision(
            role_name="drum_loop",
            confidence=max(drum_loop, 0.78 if low_body_rhythmic_loop else (0.74 if rhythmic_dominant_loop else 0.72)),
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=(
                "FX",
                "Instruments",
                "Bass",
                "Voice",
                "Human",
                "Guitar",
                "Keys",
                "Breath",
                "Spoken Voice",
                "Drops",
                "Downlifters",
                "Water",
                "Natural Ambience",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
            ),
            broad_folder_path="Drums/Drum Loops/Loops",
            reason="measured repeated rhythmic drum/bass-loop structure; pitched/bass/instrument/FX leaves require stronger non-drum evidence",
        )

    # Structure-first protected one-shots.  This catches short drum/percussion
    # hits even when pitch/formant estimation makes the measured role vector
    # uncertain.  FX and Instruments are not allowed to steal these unless a
    # separate, positive FX/instrument role is added later.
    short_front_loaded = (
        duration <= 1.10
        and event_count <= 4.0
        and _num(feature_values.get("log_crest"), 0.0) >= 1.45
        and _num(feature_values.get("attack_rise_time_norm"), attack_rise) <= 0.28
        and _num(feature_values.get("temporal_centroid_ratio"), temporal_centroid) <= 0.44
    )
    very_short_noisy_hit = (
        duration <= 0.35
        and event_count <= 3.0
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.25
        and high_total >= 0.18
    )
    hat_cymbal_tail_hit = (
        duration <= 0.90
        and event_count <= 4.0
        and high_total >= 0.32
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.28
        and _num(feature_values.get("formant_like_peak_spacing"), 0.0) < 1.05
    )
    clean_short_pitched_tonal_hit = (
        (
            duration <= 2.75
            or (duration <= 12.0 and onset_span <= 0.06 and _num(feature_values.get("tail_energy_ratio"), 0.0) <= 0.04)
        )
        and event_count <= 4.0
        and pitch_conf >= 0.62
        and loop_pitched >= 0.85
        and loop_sustained >= 0.75
        and loop_non_event_tonal >= 0.75
        and loop_percussive <= 0.08
        and loop_drumlike <= 0.08
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.16
        and _num(feature_values.get("log_crest"), 0.0) >= 1.45
        and _num(feature_values.get("attack_rise_time_norm"), attack_rise) <= 0.18
        and _num(feature_values.get("temporal_centroid_ratio"), temporal_centroid) <= 0.28
    )

    low_kick_hit = (
        duration <= 1.25
        and event_count <= 4.0
        and _num(feature_values.get("sub_bass_ratio_lt_150hz"), 0.0) >= 0.45
        and low_total >= 0.72
        and _num(feature_values.get("attack_rise_time_norm"), attack_rise) <= 0.10
        and _num(feature_values.get("temporal_centroid_ratio"), temporal_centroid) <= 0.22
    )
    strong_vocal_identity = (
        duration >= 0.35
        and event_count <= 6.0
        and low_total <= 0.32
        and loop_drumlike <= 0.12
        and loop_percussive <= 0.12
        and (
            (
                _num(feature_values.get("formant_like_peak_spacing"), 0.0) >= 1.05
                and _num(feature_values.get("f0_voiced_ratio"), 0.0) >= 0.55
            )
            or _num(role_evidence.get("formant_light_voice_identity"), 0.0) >= 0.62
            or (
                voiced_one_shot >= 0.62
                and _num(feature_values.get("f0_voiced_ratio"), 0.0) >= 0.78
                and _num(feature_values.get("pitch_confidence"), 0.0) >= 0.32
                and (mid_total_value + high_total) >= 0.70
                and loop_pitched >= 0.75
                and (
                    _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.18
                    or _num(feature_values.get("presence_ratio_2000_8000hz"), 0.0) >= 0.18
                )
            )
        )
    )
    clean_tonal_reed_phrase = (
        duration >= 1.45
        and event_count <= 6.0
        and f0_voiced >= 0.82
        and loop_pitched >= 0.90
        and loop_sustained >= 0.90
        and loop_non_event_tonal >= 0.90
        and loop_percussive <= 0.08
        and loop_drumlike <= 0.08
        and _num(feature_values.get("spectral_flatness_mean"), 0.0) <= 0.10
        and formant < 1.05
    )
    vocal_phrase_by_shape = (
        not clean_tonal_reed_phrase
        and primary_shape in {"vocal_phrase", "vocal_one_shot"}
        and shape_confidence >= 0.70
        and f0_voiced >= 0.72
        and pitch_conf >= 0.30
        and low_total <= 0.46
        and (mid_total_value + high_total) >= 0.58
        and loop_drumlike <= 0.12
        and loop_percussive <= 0.12
        and (voiced_one_shot >= 0.55 or _num(role_evidence.get("formant_light_voice_identity"), 0.0) >= 0.45)
    )
    if vocal_phrase_by_shape:
        return EligibilityDecision(
            role_name="vocal_phrase",
            confidence=max(voiced_one_shot, shape_confidence, 0.76),
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Drums",
                "Percussion",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Machines",
                "Motor",
                "Engine",
                "Glitch",
                "Stutter",
                "Ocean",
                "Water",
                "Natural Ambience",
                "Drum Loops",
            ),
            broad_folder_path="FX/Human and Voice FX",
            reason="measured vocal phrase/one-shot shape; drum, machine, animal, and ambience leaves are not eligible",
        )

    if clean_tonal_reed_phrase:
        return EligibilityDecision(
            role_name="pitched_reed_or_instrument_phrase",
            confidence=0.76,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Drums",
                "Percussion",
                "FX",
                "Human",
                "Voice",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Machines",
                "Motor",
                "Ocean",
                "Water",
                "Natural Ambience",
            ),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="measured clean tonal instrument phrase; human voice, drum, animal, machine, and ambience leaves are not eligible",
        )

    vocal_one_shot_by_facts = (
        0.35 <= duration <= 2.75
        and event_count <= 6.0
        and primary_shape in {"vocal_phrase", "vocal_one_shot", "pitched_phrase", "hit_with_tail"}
        and f0_voiced >= 0.82
        and (pitch_conf >= 0.30 or loop_pitched >= 0.75)
        and low_total <= 0.32
        and loop_drumlike <= 0.12
        and loop_percussive <= 0.12
        and (
            (formant >= 1.35 and (mid_total_value + high_total) >= 0.62)
            or _num(role_evidence.get("formant_light_voice_identity"), 0.0) >= 0.62
            or (
                (mid_total_value + high_total) >= 0.88
                and _num(feature_values.get("spectral_flatness_mean"), 0.0) >= 0.18
                and loop_sustained >= 0.75
            )
        )
    )
    if vocal_one_shot_by_facts:
        return EligibilityDecision(
            role_name="vocal_one_shot",
            confidence=max(voiced_one_shot, 0.80),
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Drums",
                "Percussion",
                "Coins",
                "Machines",
                "Motor",
                "Engine",
                "Glitch",
                "Stutter",
                "Ocean",
                "Water",
                "Natural Ambience",
                "Drum Loops",
            ),
            broad_folder_path="FX/Human and Voice FX",
            reason="measured formant-rich vocal one-shot/phrase; machine, animal, drum, and ambience leaves are not eligible",
        )
    if voiced_one_shot >= 0.62 and strong_vocal_identity:
        return EligibilityDecision(
            role_name="vocal_one_shot",
            confidence=max(voiced_one_shot, 0.74),
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Drums",
                "Percussion",
                "Coins",
                "Machines",
                "Motor",
                "Engine",
                "Glitch",
                "Stutter",
                "Ocean",
                "Water",
                "Natural Ambience",
                "Drum Loops",
            ),
            broad_folder_path="FX/Human and Voice FX",
            reason="measured voiced one-shot; machine, animal, drum, and ambience leaves are not eligible",
        )
    if clean_short_pitched_tonal_hit and not strong_vocal_identity:
        return EligibilityDecision(
            role_name="short_pitched_instrument_hit",
            confidence=0.76,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Drums",
                "Percussion",
                "Kick",
                "Tom",
                "Snare",
                "Clap",
                "FX",
                "Human",
                "Voice",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Ocean",
                "Water",
                "Natural Ambience",
                "Loops",
            ),
            broad_folder_path="_TO_REVIEW/Measured Role Conflict",
            reason="measured clean short pitched tonal hit; drum/percussion one-shot requires stronger positive drum evidence",
        )

    if low_kick_hit and not strong_vocal_identity:
        return EligibilityDecision(
            role_name="low_kick_like_hit",
            confidence=0.82,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=(
                "FX",
                "Instruments",
                "Drum Loops",
                "Long FX",
                "Ambience",
                "Water",
                "Animals",
                "Voice",
                "Guitar",
                "Bass",
            ),
            broad_folder_path="Drums/Kick Drums/Generic Kick/One Shots",
            reason="measured short low front-loaded one-shot; bass, loop, FX, and instrument leaves are not eligible",
        )
    vocal_shape_identity = primary_shape in {"vocal_phrase", "vocal_one_shot", "hit_with_tail"}
    if (short_front_loaded or very_short_noisy_hit or hat_cymbal_tail_hit) and not (
        strong_vocal_identity and vocal_shape_identity
    ):
        return EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=max(percussive_one_shot, 0.74),
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=(
                "FX",
                "Instruments",
                "Guitar",
                "Keys",
                "Bass",
                "Voice",
                "Human",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Ocean",
                "Water",
                "Natural Ambience",
                "Glitch",
                "Stutter",
                "Long FX",
                "Drum Loops",
            ),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="measured short percussive one-shot; FX/instrument/loop leaves require stronger positive evidence",
        )

    short_pitched_instrument_hit = (
        duration <= 1.25
        and event_count <= 8.0
        and pitch_conf >= 0.45
        and f0_voiced >= 0.50
        and loop_pitched >= 0.70
        and loop_sustained >= 0.60
        and loop_non_event_tonal >= 0.60
        and loop_drumlike <= 0.20
        and loop_percussive <= 0.20
        and temporal_centroid <= 0.20
        and _num(feature_values.get("tail_energy_ratio"), 0.0) <= 0.08
    )
    if short_pitched_instrument_hit and not vocal_one_shot_by_facts:
        short_hit_folder = "Instruments/Synths/Synth Lead/One Shots"
        short_hit_reason = "measured short pitched one-shot; loop, FX, drum, and human/animal leaves are not eligible"
        if low_total >= 0.55:
            short_hit_folder = "Instruments/Bass/Generic Bass/One Shots"
            short_hit_reason = (
                "measured short low pitched one-shot; loop, FX, drum, and human/animal leaves are not eligible"
            )
        return EligibilityDecision(
            role_name="short_pitched_instrument_hit",
            confidence=0.74,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Loops",
                "Long FX",
                "FX",
                "Drums",
                "Percussion",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Human",
                "Voice",
                "Breath",
                "Spoken Voice",
                "Ocean",
                "Water",
                "Natural Ambience",
            ),
            broad_folder_path=short_hit_folder,
            reason=short_hit_reason,
        )

    # Vocal stabs/shouts: allow broad human/voice buckets, block animal/drum leaves.
    if voiced_one_shot >= 0.62:
        return EligibilityDecision(
            role_name="vocal_one_shot",
            confidence=voiced_one_shot,
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("Animals", "Dog", "Bird", "Cat", "Drums", "Percussion", "Coins"),
            broad_folder_path="FX/Human and Voice FX",
            reason="measured voice-like one-shot; animal/drum/foley leaves are not eligible",
        )

    # Bass loops are a specific pitched-loop role.  Check them before the
    # generic pitched-loop fallback so clean sub-heavy phrases do not get
    # flattened into Instrument Loops when the broader pitch evidence is also
    # strong.
    if bass_loop >= 0.62 and not (low_body_rhythmic_loop or repeated_drum_event_loop):
        return EligibilityDecision(
            role_name="bass_loop",
            confidence=bass_loop,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("Drums/Kick", "Drums/Drum Loops", "FX", "Animals", "Bird", "Dog"),
            broad_folder_path="Instruments/Bass/Bass Loops",
            reason="measured bass loop; kick/drum/FX leaves are not eligible",
        )

    # Long pitched material or sustained melodic phrases: broad instrument loop/phrase.
    pitched_phrase_by_facts = (
        duration >= 1.5
        and pitch_conf >= 0.70
        and f0_voiced >= 0.75
        and loop_pitched >= 0.85
        and loop_sustained >= 0.70
        and loop_non_event_tonal >= 0.70
        and loop_percussive <= 0.18
    )
    pitched_loop_by_facts = (
        duration >= 4.0
        and event_count >= 6.0
        and pitch_conf >= 0.30
        and f0_voiced >= 0.65
        and loop_pitched >= 0.70
        and loop_sustained >= 0.55
        and loop_non_event_tonal >= 0.45
        and loop_drumlike <= 0.25
    )
    if (pitched_music_loop >= 0.58 or pitched_phrase_by_facts or pitched_loop_by_facts) and not (
        low_body_rhythmic_loop or repeated_drum_event_loop
    ):
        confidence = max(pitched_music_loop, 0.76 if pitched_phrase_by_facts else 0.66)
        return EligibilityDecision(
            role_name="pitched_music_loop",
            confidence=confidence,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Drums",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Keys Coins",
                "Breath",
                "Spoken Voice",
                "Drops",
                "Downlifters",
                "Water",
                "Natural Ambience",
            ),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="measured sustained pitched musical phrase/loop; drum, animal, foley, and long-FX leaves are not eligible",
        )

    # Long choppy music-like loop: when no sharper role wins, block tiny foley leaves.
    # This is checked before drum-loop fallback so a pitched/formant-rich
    # chopped musical phrase does not become a drum loop only because it has
    # repeated attacks.
    music_loop_by_facts = (
        duration >= 4.0
        and event_count >= 8.0
        and loop_pitched >= 0.45
        and (loop_sustained >= 0.30 or formant >= 1.50)
        and loop_drumlike <= 0.12
    )
    if music_loop_by_facts and not (low_body_rhythmic_loop or repeated_drum_event_loop):
        return EligibilityDecision(
            role_name="mixed_music_loop",
            confidence=0.62,
            allowed_top_families=("Instruments", "Drums", "_TO_REVIEW"),
            blocked_path_fragments=(
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Keys Coins",
                "Water",
                "Natural Ambience",
                "Breath",
                "Spoken Voice",
                "FX",
                "Human",
                "Voice",
            ),
            broad_folder_path="Instruments/Mixed Musical Loops",
            reason="measured long choppy music-like loop; animal, coin, and ambience leaves are not eligible",
        )

    # Drum/percussion loops: allow broad drum loops, block long-FX and voice/animal/texture leaves.
    drum_loop_by_facts = (
        duration >= 4.0
        and event_count >= 8.0
        and onset_span >= 0.50
        and event_rate >= 0.70
        and (
            loop_percussive >= 0.14
            or loop_drumlike >= 0.30
            or (low_total >= 0.70 and event_count >= 12.0 and loop_sustained <= 0.65)
            or (high_total >= 0.35 and loop_percussive >= 0.35)
        )
    )
    if (
        (drum_loop >= 0.30 or drum_loop_by_facts or rhythmic_dominant_loop)
        and not clean_sustained_tonal_non_drum_loop
        and not clean_low_tonal_non_drum_loop
    ):
        confidence = max(drum_loop, 0.72 if (drum_loop_by_facts or rhythmic_dominant_loop) else 0.60)
        return EligibilityDecision(
            role_name="drum_loop",
            confidence=confidence,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=(
                "FX",
                "Instruments/Voice",
                "Breath",
                "Spoken Voice",
                "Drops",
                "Downlifters",
                "Water",
                "Natural Ambience",
                "Animals",
                "Dog",
                "Bird",
                "Coins",
            ),
            broad_folder_path="Drums/Drum Loops/Loops",
            reason="measured repeated percussive/drumlike loop; FX, voice, animal, and foley leaves are not eligible",
        )

    # Single low/front-loaded transient: one-shot area, not loop area.
    if percussive_one_shot >= 0.70 and duration <= 3.5:
        return EligibilityDecision(
            role_name="percussive_one_shot",
            confidence=percussive_one_shot,
            allowed_top_families=("Drums", "FX", "_TO_REVIEW"),
            blocked_path_fragments=("Drum Loops", "Long FX", "Ambience", "Water"),
            broad_folder_path="Drums/Percussion/One Shots",
            reason="measured single percussive event; loop and long-FX leaves are not eligible",
        )

    return EligibilityDecision(role_name="unknown", confidence=0.0)


def _role(roles: dict[str, Any], name: str) -> float:
    return _num(roles.get(name), 0.0)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if number != number:
        return float(default)
    return number
