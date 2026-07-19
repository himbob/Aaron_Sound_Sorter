# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Temporal shape voter.

ShapeVoter is deliberately not a category voter.  It reads measured time-shape
features and returns structural shape claims such as ``beat_loop`` or
``pitched_phrase``.  Consensus may use those claims to reject fake certainty,
but this voter must not manufacture a destination folder by itself.

Important naming rule: shape labels must not look like source identity labels.
A formant/voiced musical phrase is reported as ``pitched_phrase_shape``, not
``vocal_phrase``.  Voice remains a source/role claim owned by measured voice
features and the identity voters, not by ShapeVoter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from aaron_audio_intelligence.shape_memory_brain import ShapeMemoryMatch, shape_memory_match_for_facts
from aaron_sound_sorter.domain.models import AudioPhysics, CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.domain.policies import ShapeVoterPolicy
from aaron_sound_sorter.score_math import average_score as average
from aaron_sound_sorter.voters.base import Voter, clamp01

SHAPE_TOP = "_SHAPE_DIAGNOSTIC"
PITCHED_PHRASE_SHAPE = "pitched_phrase_shape"
PITCHED_REPETITION_PHRASE = "pitched_repetition_phrase"
DESIGNED_LOW_FX_SHAPE = "designed_low_fx"
DESIGNED_MOTION_FX_LOOP = "designed_motion_fx_loop"
DESIGNED_TONAL_FX_SHAPE = "designed_tonal_fx"


@dataclass(frozen=True)
class ShapeEvidence:
    """Readable waveform/time-shape evidence used by ShapeVoter."""

    primary_shape: str
    secondary_shape: str
    confidence: float
    onset_count: float
    onset_density_hz: float
    pulse_regularity: float
    onset_span_ratio: float
    temporal_centroid_ratio: float
    attack_rise_time_norm: float
    sustain_ratio: float
    tail_ratio: float
    low_event_ratio: float
    mid_event_ratio: float
    high_event_ratio: float
    drumlike_frame_ratio: float
    pitched_event_ratio: float
    percussive_event_ratio: float
    sustained_tonal_frame_ratio: float
    non_event_tonal_ratio: float
    pitch_confidence: float
    f0_voiced_ratio: float
    spectral_flatness_mean: float
    spectral_entropy_mean: float
    centroid_slope_norm: float
    shape_scores: list[tuple[str, float]]
    solo_isolation_score: float
    layered_loop_score: float
    instrument_plus_fx_loop_score: float
    echo_tail_score: float
    true_repetition_score: float
    reason: str
    learned_shape_memory_enabled: bool = False
    learned_shape_memory_matched: bool = False
    learned_shape_memory_shape: str = ""
    learned_shape_memory_confidence: float = 0.0
    learned_shape_memory_nearest_distance: float = 0.0
    learned_shape_memory_threshold: float = 0.0
    learned_shape_memory_example_count: int = 0
    learned_shape_memory_effective_weight: int = 0
    learned_shape_memory_match_kind: str = "none"
    learned_shape_memory_policy: str = ""
    learned_voter_memory_enabled: bool = False
    learned_voter_memory_matched: bool = False
    learned_voter_memory_shape: str = ""
    learned_voter_memory_role: str = ""
    learned_voter_memory_confidence: float = 0.0
    learned_voter_memory_match_kind: str = "none"
    learned_voter_memory_policy: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 6)
        data["shape_scores"] = [[str(name), round(float(score), 6)] for name, score in self.shape_scores]
        for key in (
            "solo_isolation_score",
            "layered_loop_score",
            "instrument_plus_fx_loop_score",
            "echo_tail_score",
            "true_repetition_score",
            "learned_shape_memory_confidence",
            "learned_shape_memory_nearest_distance",
            "learned_shape_memory_threshold",
            "learned_voter_memory_confidence",
        ):
            data[key] = round(float(data[key]), 6)
        return data


class ShapeVoter(Voter):
    """Vote on structural waveform shape, not final category identity."""

    voter_name = "shape"

    def __init__(self, policy: ShapeVoterPolicy | None = None) -> None:
        self.policy = policy or ShapeVoterPolicy()
        super().__init__(self.policy.top_n)

    def vote(self, physics: AudioPhysics, facts: SharedAudioFacts, brain: dict[str, Any]) -> VoterResult:
        if facts.is_broken_or_tiny:
            evidence = self.broken_shape_evidence(physics, facts)
        else:
            evidence = classify_shape(facts.feature_values_by_name, facts=facts)
            evidence = apply_shape_memory(evidence, facts=facts, brain=brain)
            evidence = apply_voter_memory(evidence, facts=facts)
        label = f"shape/{evidence.primary_shape}"
        guess = CategoryGuess(
            label=label,
            folder_path=label,
            top_family=SHAPE_TOP,
            score=1.0 - clamp01(evidence.confidence),
            confidence=clamp01(evidence.confidence),
            rank=1,
            reason=evidence.reason,
            evidence={"shape_vote": evidence.to_dict(), "voter_is_category_router": False},
        )
        return VoterResult(
            voter_name=self.voter_name,
            guesses=[guess],
            diagnostics={"shape_vote": evidence.to_dict(), "voter_is_category_router": False},
        )

    @staticmethod
    def broken_shape_evidence(physics: AudioPhysics, facts: SharedAudioFacts) -> ShapeEvidence:
        return ShapeEvidence(
            primary_shape="broken_or_tiny",
            secondary_shape="unusable",
            confidence=1.0,
            onset_count=0.0,
            onset_density_hz=0.0,
            pulse_regularity=0.0,
            onset_span_ratio=0.0,
            temporal_centroid_ratio=0.0,
            attack_rise_time_norm=0.0,
            sustain_ratio=0.0,
            tail_ratio=0.0,
            low_event_ratio=0.0,
            mid_event_ratio=0.0,
            high_event_ratio=0.0,
            drumlike_frame_ratio=0.0,
            pitched_event_ratio=0.0,
            percussive_event_ratio=0.0,
            sustained_tonal_frame_ratio=0.0,
            non_event_tonal_ratio=0.0,
            pitch_confidence=0.0,
            f0_voiced_ratio=0.0,
            spectral_flatness_mean=0.0,
            spectral_entropy_mean=0.0,
            centroid_slope_norm=0.0,
            shape_scores=[("broken_or_tiny", 1.0)],
            solo_isolation_score=0.0,
            layered_loop_score=0.0,
            instrument_plus_fx_loop_score=0.0,
            echo_tail_score=0.0,
            true_repetition_score=0.0,
            reason=f"broken_or_tiny:{physics.read_status}",
        )


def classify_shape(values: Mapping[str, float], *, facts: SharedAudioFacts) -> ShapeEvidence:
    """Classify audio time shape from measured features only."""
    onset_count = max(0.0, expm1_safe(v(values, "log_transient_count")))
    onset_density = v(values, "event_rate_hz")
    regularity_raw = v(values, "onset_interval_regularity")
    pulse_regularity = clamp01(1.0 - regularity_raw)
    span = v(values, "onset_span_ratio")
    temporal = v(values, "temporal_centroid_ratio")
    attack = v(values, "attack_rise_time_norm")
    tail = v(values, "tail_energy_ratio")
    sustain = clamp01(
        v(values, "body_pitch_confidence") * 0.45
        + v(values, "tail_pitch_confidence") * 0.25
        + v(values, "loop_sustained_tonal_frame_ratio") * 0.30
    )

    low_event = v(values, "loop_mean_event_low_ratio")
    high_event = v(values, "loop_mean_event_high_ratio")
    mid_event = clamp01(1.0 - low_event - high_event)
    drumlike = v(values, "loop_drumlike_frame_ratio")
    pitched = v(values, "loop_pitched_event_ratio")
    percussive = v(values, "loop_percussive_event_ratio")
    sustained_tonal = v(values, "loop_sustained_tonal_frame_ratio")
    non_event_tonal = v(values, "loop_non_event_tonal_ratio")
    pitch_conf = v(values, "pitch_confidence")
    f0_voiced = v(values, "f0_voiced_ratio")
    flatness = v(values, "spectral_flatness_mean")
    entropy = v(values, "spectral_entropy_mean")
    body_noise = v(values, "body_noise_ratio")
    tail_noise = v(values, "tail_noise_ratio")
    body_flatness = v(values, "body_flatness", flatness)
    tail_flatness = v(values, "tail_flatness", flatness)
    harmonic = v(values, "harmonic_energy_ratio")
    hnr = v(values, "harmonic_to_noise_ratio")
    inharmonicity = v(values, "inharmonicity")
    stability = v(values, "spectral_peak_stability")
    stereo_width = v(values, "stereo_width")
    noisy_events = v(values, "loop_noisy_event_ratio")
    event_noise = v(values, "loop_mean_event_noise_ratio")
    timbre_diversity = v(values, "loop_event_timbre_diversity")
    slope = v(values, "centroid_slope_norm")
    formant = v(values, "formant_like_peak_spacing")
    zcr = v(values, "zcr_mean")
    flux = v(values, "spectral_flux_mean")
    flux_var = v(values, "spectral_flux_variance")
    tonal_balance = v(values, "loop_tonal_to_percussive_balance")
    pitched_onset = evidence_number(facts, "onset_pitched_onset_score")
    percussive_onset = evidence_number(facts, "onset_percussive_onset_score")
    plucked_authority = evidence_number(facts, "plucked_string_authority_score")
    synth_source = evidence_number(facts, "synth_tonal_source_score")
    keys_authority = evidence_number(facts, "struck_keys_authority_score")
    measured_fx_motion = measured_number(facts, "fx_motion_score")
    measured_transition = measured_number(facts, "fx_transition_authority_score")
    measured_texture = max(
        measured_number(facts, "texture_bed_score"),
        measured_number(facts, "texture_water_ocean_score"),
        measured_number(facts, "texture_rain_score"),
        measured_number(facts, "texture_wind_score"),
        measured_number(facts, "texture_fire_score"),
        measured_number(facts, "texture_thunder_score"),
        measured_number(facts, "texture_noise_static_score"),
    )
    measured_fx_tail = max(
        measured_number(facts, "fx_impact_score"),
        measured_number(facts, "fx_boom_score"),
        measured_number(facts, "fx_slam_score"),
        measured_number(facts, "fx_sub_hit_score"),
        measured_number(facts, "fx_reverse_score"),
        measured_number(facts, "fx_whoosh_sweep_score"),
        measured_number(facts, "fx_riser_build_score"),
        measured_number(facts, "fx_drop_downlifter_score"),
        measured_transition,
    )
    raw_measured_drum_loop = max(
        measured_number(facts, "drum_loop_source_score"),
        role_number(facts, "percussive_drum_loop"),
        role_number(facts, "bright_drum_loop"),
    )
    measured_low_rhythmic_role = role_number(facts, "low_rhythmic_drum_loop")
    duration = float(facts.evidence.get("duration_sec", 0.0) or 0.0) if isinstance(facts.evidence, dict) else 0.0

    low_total = clamp01(v(values, "sub_bass_ratio_lt_150hz") + v(values, "bass_ratio_150_500hz"))
    high_total = clamp01(v(values, "presence_ratio_2000_8000hz") + v(values, "air_ratio_gt_8000hz"))
    mid_total = clamp01(v(values, "mid_ratio_500_2000hz"))
    bass_identity_source = max(
        measured_number(facts, "bass_synth_score"),
        measured_number(facts, "bass_sub_score"),
        measured_number(facts, "bass_808_score"),
        measured_number(facts, "bass_electric_score"),
        measured_number(facts, "low_end_source_score"),
    )
    pure_low_bass_loop_body = bool(
        bass_identity_source >= 0.70
        and low_event >= 0.88
        and mid_event <= 0.08
        and high_event <= 0.06
        and max(sustained_tonal, non_event_tonal) >= 0.55
        and max(percussive, drumlike) <= 0.08
    )
    measured_drum_loop = raw_measured_drum_loop
    if pure_low_bass_loop_body:
        # A very pure low, pitched Bass-panel loop can trip generic drum-loop
        # source scores because it is repetitive and transient-rich.  Keep that
        # source score diagnostic, but do not let ShapeVoter turn it into a beat
        # loop unless there is additional non-bass rhythmic evidence.
        measured_drum_loop = min(measured_drum_loop, 0.42)
    else:
        measured_drum_loop = max(measured_drum_loop, measured_low_rhythmic_role)
    event_separation = max(low_event, high_event, mid_event) - min(low_event, high_event, mid_event)
    low_band_active = ramp(max(low_total, low_event), 0.12, 0.42)
    mid_band_active = ramp(max(mid_total, mid_event), 0.22, 0.58)
    high_band_active = ramp(max(high_total, high_event), 0.05, 0.24)
    band_spread = clamp01(
        0.50 * min(low_band_active, mid_band_active, high_band_active)
        + 0.30 * sorted((low_band_active, mid_band_active, high_band_active))[1]
        + 0.20 * average(low_band_active, mid_band_active, high_band_active)
    )
    tonal_presence = clamp01(
        0.34 * ramp(max(pitch_conf, f0_voiced), 0.42, 0.88)
        + 0.26 * ramp(max(sustained_tonal, non_event_tonal, pitched), 0.45, 0.95)
        + 0.20 * ramp(max(harmonic, hnr), 0.18, 0.72)
        + 0.20 * inverse_ramp(flatness, 0.04, 0.34)
    )
    noise_wash = clamp01(
        0.28 * ramp(max(flatness, body_flatness, tail_flatness), 0.16, 0.48)
        + 0.22 * ramp(max(body_noise, tail_noise, noisy_events, event_noise), 0.14, 0.48)
        + 0.20 * ramp(entropy, 0.36, 0.78)
        + 0.16 * ramp(tail, 0.28, 0.76)
        + 0.14 * ramp(stereo_width, 0.26, 0.72)
    )
    true_repetition = clamp01(
        0.26 * (1.0 if facts.is_loop_like else 0.0)
        + 0.22 * ramp(onset_count, 4.0, 16.0)
        + 0.20 * ramp(span, 0.36, 0.86)
        + 0.14 * ramp(pulse_regularity, 0.22, 0.68)
        + 0.10 * ramp(onset_density, 0.45, 2.2)
        + 0.08 * inverse_ramp(temporal, 0.22, 0.72)
    )
    echo_tail = clamp01(
        0.26 * (1.0 if facts.is_single_event_like else inverse_ramp(onset_count, 1.0, 4.0))
        + 0.20 * inverse_ramp(attack, 0.04, 0.22)
        + 0.18 * inverse_ramp(temporal, 0.16, 0.44)
        + 0.18 * ramp(tail, 0.24, 0.76)
        + 0.10 * inverse_ramp(span, 0.04, 0.34)
        + 0.08 * inverse_ramp(true_repetition, 0.16, 0.46)
    )
    layered_loop = clamp01(
        0.22 * true_repetition
        + 0.20 * band_spread
        + 0.16 * ramp(timbre_diversity, 0.16, 0.58)
        + 0.14 * ramp(max(percussive, drumlike), 0.10, 0.48)
        + 0.12 * ramp(max(pitched, sustained_tonal, non_event_tonal), 0.45, 0.95)
        + 0.10 * noise_wash
        + 0.06 * ramp(stereo_width, 0.30, 0.75)
        - 0.20 * echo_tail
    )
    instrument_plus_fx = clamp01(
        0.30 * tonal_presence
        + 0.26 * noise_wash
        + 0.18 * true_repetition
        + 0.14 * ramp(stereo_width, 0.30, 0.76)
        + 0.12 * ramp(tail, 0.28, 0.78)
        - 0.12 * echo_tail
    )
    solo_isolation = clamp01(
        0.26 * tonal_presence
        + 0.18 * ramp(stability, 0.10, 0.46)
        + 0.16 * inverse_ramp(band_spread, 0.20, 0.62)
        + 0.14 * inverse_ramp(max(body_noise, tail_noise, flatness), 0.08, 0.38)
        + 0.12 * inverse_ramp(timbre_diversity, 0.10, 0.48)
        + 0.08 * inverse_ramp(onset_count, 5.0, 34.0)
        + 0.06 * inverse_ramp(stereo_width, 0.12, 0.56)
    )

    scores: dict[str, float] = {}
    scores["transition_riser"] = average(
        ramp(slope, 0.05, 0.20),
        ramp(duration, 0.65, 2.0),
        ramp(tail, 0.20, 0.65),
        inverse_ramp(onset_count, 14.0, 32.0),
    )
    scores["transition_drop"] = average(
        ramp(-slope, 0.05, 0.20),
        ramp(duration, 0.65, 2.0),
        ramp(tail, 0.18, 0.65),
        inverse_ramp(onset_count, 14.0, 32.0),
    )
    scores["reverse_swell"] = clamp01(
        0.28 * ramp(attack, 0.14, 0.58)
        + 0.22 * ramp(temporal, 0.44, 0.78)
        + 0.18 * ramp(tail, 0.20, 0.72)
        + 0.16 * inverse_ramp(onset_count, 1.0, 6.0)
        + 0.10 * ramp(duration, 0.45, 1.8)
        + 0.06 * ramp(abs(slope), 0.04, 0.18)
    )
    scores["whoosh_sweep"] = clamp01(
        0.26 * noise_wash
        + 0.22 * ramp(abs(slope), 0.04, 0.20)
        + 0.17 * ramp(max(high_total, high_event), 0.10, 0.38)
        + 0.14 * ramp(stereo_width, 0.24, 0.76)
        + 0.11 * ramp(duration, 0.35, 1.6)
        + 0.10 * inverse_ramp(max(pitched, sustained_tonal), 0.12, 0.56)
    )
    scores["beat_loop"] = average(
        1.0 if facts.is_loop_like else 0.0,
        ramp(onset_count, 5.0, 14.0),
        ramp(span, 0.42, 0.82),
        ramp(onset_density, 0.65, 2.4),
        ramp(percussive, 0.35, 0.78),
        ramp(drumlike, 0.35, 0.78),
        ramp(event_separation, 0.12, 0.42),
        inverse_ramp(sustained_tonal, 0.18, 0.55),
    )
    scores["top_loop"] = average(
        1.0 if facts.is_loop_like else 0.0,
        ramp(high_total, 0.38, 0.75),
        ramp(high_event, 0.30, 0.70),
        ramp(percussive, 0.35, 0.75),
        inverse_ramp(low_total, 0.10, 0.36),
        inverse_ramp(sustained_tonal, 0.12, 0.42),
    )
    scores["bass_phrase"] = average(
        ramp(low_total, 0.35, 0.72),
        ramp(pitch_conf, 0.42, 0.85),
        ramp(pitched, 0.40, 0.88),
        ramp(sustained_tonal + non_event_tonal, 0.60, 1.35),
        inverse_ramp(drumlike, 0.10, 0.40),
        inverse_ramp(high_event, 0.08, 0.35),
    )
    # Bass phrase is allowed to be tonal, but it still needs a bass-band
    # owner.  Otherwise midrange voiced/formant phrases can win ``bass_phrase``
    # just because they are pitched, sustained, and not drumlike.
    bass_band_owner = max(low_total, low_event, v(values, "sub_bass_ratio_lt_150hz"))
    bass_identity_hint = max(measured_number(facts, "bass_sub_score"), measured_number(facts, "bass_synth_score"))
    if bass_band_owner < 0.26 and not (bass_identity_hint >= 0.70 and low_total >= 0.20):
        scores["bass_phrase"] = min(scores["bass_phrase"], 0.52)
    # This used to be named "vocal_phrase".  That was a category leak:
    # formant/voicing shape is useful structure evidence, but it is not proof
    # of source identity.  Keep the measured score, but use a source-safe name.
    scores[PITCHED_PHRASE_SHAPE] = average(
        ramp(formant, 1.05, 2.20),
        ramp(f0_voiced, 0.45, 0.85),
        ramp(pitch_conf, 0.35, 0.78),
        ramp(duration, 0.35, 1.5),
        inverse_ramp(drumlike, 0.08, 0.32),
    )
    source_safe_voiced_phrase_body = bool(
        formant >= 1.45
        and f0_voiced >= 0.72
        and pitch_conf >= 0.72
        and max(drumlike, percussive) <= 0.12
        and bass_band_owner < 0.30
    )
    if source_safe_voiced_phrase_body:
        voiced_floor = max(scores[PITCHED_PHRASE_SHAPE], 0.86)
        scores[PITCHED_PHRASE_SHAPE] = voiced_floor
        scores["bass_phrase"] = min(scores["bass_phrase"], voiced_floor - 0.08)
    scores["pitched_phrase"] = average(
        ramp(pitch_conf, 0.45, 0.85),
        ramp(f0_voiced, 0.42, 0.86),
        ramp(sustained_tonal + non_event_tonal, 0.55, 1.30),
        inverse_ramp(drumlike, 0.08, 0.32),
        inverse_ramp(flatness, 0.10, 0.45),
    )
    scores["sustained_pad"] = average(
        ramp(duration, 1.5, 5.5),
        ramp(sustain + tail, 0.55, 1.40),
        inverse_ramp(onset_density, 0.20, 1.10),
        ramp(non_event_tonal + sustained_tonal, 0.45, 1.20),
    )
    scores["noise_texture"] = average(
        ramp(duration, 1.2, 4.0),
        ramp(flatness, 0.25, 0.70),
        ramp(entropy, 0.55, 0.92),
        inverse_ramp(pitch_conf, 0.12, 0.45),
        inverse_ramp(onset_density, 0.25, 1.30),
    )
    scores["hit_with_tail"] = average(
        1.0 if facts.is_single_event_like else 0.35,
        inverse_ramp(attack, 0.06, 0.28),
        inverse_ramp(temporal, 0.18, 0.46),
        ramp(tail, 0.22, 0.70),
        inverse_ramp(onset_count, 1.5, 5.0),
    )
    scores["impact_with_tail"] = clamp01(
        0.38 * scores["hit_with_tail"]
        + 0.18 * ramp(max(low_total, high_total), 0.18, 0.62)
        + 0.16 * ramp(stereo_width, 0.20, 0.76)
        + 0.14 * ramp(noise_wash, 0.20, 0.66)
        + 0.14 * ramp(duration, 0.18, 1.4)
    )
    scores["single_hit"] = average(
        1.0 if facts.is_short_hit_like else 0.35,
        inverse_ramp(attack, 0.04, 0.22),
        inverse_ramp(temporal, 0.12, 0.38),
        inverse_ramp(tail, 0.08, 0.42),
        inverse_ramp(onset_count, 1.5, 4.0),
        inverse_ramp(sustain, 0.08, 0.45),
    )
    scores["glitch_stutter"] = clamp01(
        0.26 * ramp(onset_count, 4.0, 20.0)
        + 0.20 * ramp(onset_density, 1.6, 7.0)
        + 0.18 * inverse_ramp(pulse_regularity, 0.08, 0.46)
        + 0.16 * ramp(max(flux, flux_var * 10.0), 0.10, 0.58)
        + 0.12 * ramp(noise_wash, 0.22, 0.70)
        + 0.08 * ramp(span, 0.12, 0.74)
    )
    scores["ui_blip"] = clamp01(
        0.26 * (1.0 if facts.is_short_hit_like else inverse_ramp(duration, 0.08, 0.48))
        + 0.22 * ramp(max(pitch_conf, stability), 0.46, 0.92)
        + 0.18 * inverse_ramp(flatness, 0.02, 0.24)
        + 0.14 * inverse_ramp(tail, 0.02, 0.34)
        + 0.12 * inverse_ramp(onset_count, 1.0, 3.0)
        + 0.08 * ramp(f0_voiced, 0.18, 0.78)
        - 0.24 * ramp(low_total, 0.38, 0.82)
    )
    scores["static_bed"] = clamp01(
        0.30 * noise_wash
        + 0.22 * ramp(duration, 1.4, 4.8)
        + 0.18 * inverse_ramp(abs(slope), 0.02, 0.12)
        + 0.16 * inverse_ramp(onset_density, 0.12, 1.0)
        + 0.14 * ramp(max(flatness, zcr), 0.24, 0.66)
    )
    scores["texture_bed"] = clamp01(
        0.34 * max(scores["noise_texture"], scores["static_bed"])
        + 0.22 * noise_wash
        + 0.16 * ramp(duration, 1.2, 4.8)
        + 0.12 * inverse_ramp(onset_count, 1.0, 9.0)
        + 0.10 * inverse_ramp(abs(slope), 0.02, 0.16)
        + 0.06 * inverse_ramp(tonal_presence, 0.28, 0.72)
    )
    scores["foley_action"] = clamp01(
        0.28 * max(scores["single_hit"], scores["hit_with_tail"], scores["impact_with_tail"])
        + 0.20 * noise_wash
        + 0.16 * ramp(max(inharmonicity, high_total), 0.18, 0.62)
        + 0.14 * inverse_ramp(max(pitch_conf, harmonic), 0.22, 0.72)
        + 0.12 * inverse_ramp(low_total, 0.12, 0.54)
        + 0.10 * ramp(max(flux, flux_var * 8.0), 0.12, 0.58)
    )
    scores["mechanical_motion"] = clamp01(
        0.26 * noise_wash
        + 0.20 * true_repetition
        + 0.16 * ramp(max(flux, flux_var * 8.0), 0.12, 0.62)
        + 0.14 * ramp(max(mid_total, low_total), 0.24, 0.72)
        + 0.12 * ramp(onset_density, 0.35, 3.6)
        + 0.12 * inverse_ramp(tonal_presence, 0.34, 0.82)
    )
    scores["siren_alarm_tone"] = clamp01(
        0.30 * ramp(max(pitch_conf, stability), 0.48, 0.92)
        + 0.20 * ramp(abs(slope), 0.025, 0.16)
        + 0.18 * ramp(duration, 0.55, 2.4)
        + 0.14 * ramp(max(f0_voiced, harmonic), 0.24, 0.82)
        + 0.10 * inverse_ramp(flatness, 0.04, 0.34)
        + 0.08 * inverse_ramp(low_total, 0.42, 0.82)
    )
    if source_safe_voiced_phrase_body:
        scores["siren_alarm_tone"] = min(scores["siren_alarm_tone"], scores[PITCHED_PHRASE_SHAPE] - 0.07)
    scores["hybrid_fx_motion"] = clamp01(
        0.28 * noise_wash
        + 0.22 * ramp(abs(slope), 0.035, 0.18)
        + 0.16 * ramp(stereo_width, 0.24, 0.78)
        + 0.14 * ramp(max(formant, timbre_diversity), 0.20, 0.72)
        + 0.12
        * ramp(
            max(
                scores["glitch_stutter"],
                scores["whoosh_sweep"],
                scores["reverse_swell"],
                scores["mechanical_motion"],
                scores["siren_alarm_tone"],
            ),
            0.30,
            0.78,
        )
        + 0.08 * inverse_ramp(max(pitched, harmonic), 0.20, 0.76)
    )
    fx_low_design = max(
        measured_number(facts, "fx_sub_hit_score"),
        measured_number(facts, "fx_boom_score"),
        measured_number(facts, "fx_impact_score"),
        measured_number(facts, "fx_drop_downlifter_score"),
        measured_number(facts, "fx_siren_score"),
        measured_number(facts, "fx_formant_score"),
        measured_number(facts, "fx_whoosh_sweep_score"),
        measured_fx_motion,
        measured_transition,
    )
    fx_tonal_design = max(
        measured_number(facts, "fx_formant_score"),
        measured_number(facts, "fx_siren_score"),
        measured_number(facts, "fx_alarm_score"),
        measured_number(facts, "fx_glitch_stutter_score"),
        measured_number(facts, "fx_radio_electrical_score"),
        measured_number(facts, "fx_reverse_score"),
        measured_fx_motion,
        measured_transition,
    )
    synthetic_alert_values = [
        measured_number(facts, "tonal_alert_siren_score"),
        measured_number(facts, "fx_siren_score"),
        measured_number(facts, "fx_alarm_score"),
        measured_number(facts, "fx_boom_score"),
        measured_number(facts, "fx_sub_hit_score"),
        measured_number(facts, "fx_impact_score"),
        measured_number(facts, "fx_slam_score"),
    ]
    synthetic_alert_fx_pressure = max(synthetic_alert_values)
    synthetic_alert_fx_cluster = average(*sorted(synthetic_alert_values, reverse=True)[:3])
    fx_motion_design = max(
        measured_fx_motion,
        measured_transition,
        measured_fx_tail,
        measured_number(facts, "fx_whoosh_sweep_score"),
        measured_number(facts, "fx_glitch_stutter_score"),
        measured_number(facts, "fx_machine_mechanical_score"),
    )
    # These are still shape labels, not destinations.  They mark common FX
    # decoys where normal pitch/rhythm morphology is present, but the measured
    # spectrum/envelope behaves like designed motion, designed low-end, or
    # designed tonal/formant effects.  Later claim producers must still require
    # compatible FX evidence before routing to FX.
    scores[DESIGNED_LOW_FX_SHAPE] = clamp01(
        0.24 * ramp(low_event, 0.52, 0.88)
        + 0.22 * ramp(fx_low_design, 0.48, 0.72)
        + 0.16 * ramp(tail, 0.30, 0.82)
        + 0.14 * ramp(max(noise_wash, entropy), 0.34, 0.72)
        + 0.10 * ramp(duration, 0.35, 2.2)
        + 0.08 * ramp(abs(slope), 0.025, 0.14)
        + 0.06 * inverse_ramp(max(drumlike, percussive), 0.04, 0.30)
        - 0.14 * ramp(bass_identity_source, 0.68, 0.92)
    )
    scores[DESIGNED_MOTION_FX_LOOP] = clamp01(
        0.24 * true_repetition
        + 0.22 * ramp(fx_motion_design, 0.38, 0.72)
        + 0.16 * ramp(max(scores["whoosh_sweep"], scores["glitch_stutter"], scores["hybrid_fx_motion"]), 0.52, 0.82)
        + 0.12 * ramp(max(noise_wash, timbre_diversity), 0.38, 0.72)
        + 0.10 * ramp(stereo_width, 0.32, 0.78)
        + 0.08 * ramp(abs(slope), 0.025, 0.16)
        + 0.08 * inverse_ramp(max(drumlike, percussive), 0.32, 0.76)
    )
    scores[DESIGNED_TONAL_FX_SHAPE] = clamp01(
        0.24 * tonal_presence
        + 0.22 * ramp(fx_tonal_design, 0.46, 0.74)
        + 0.16 * ramp(max(scores["siren_alarm_tone"], scores["glitch_stutter"], scores["hybrid_fx_motion"]), 0.50, 0.82)
        + 0.12 * ramp(max(noise_wash, inharmonicity, timbre_diversity), 0.24, 0.66)
        + 0.10 * ramp(tail, 0.22, 0.78)
        + 0.08 * ramp(duration, 0.35, 2.4)
        + 0.08 * inverse_ramp(max(drumlike, percussive), 0.08, 0.36)
        - 0.10 * ramp(max(plucked_authority, keys_authority), 0.58, 0.88)
    )
    scores["echo_tail_hit"] = echo_tail
    repeated_phrase_score = true_repetition
    if max(layered_loop, instrument_plus_fx) >= 0.66 and band_spread >= 0.45:
        repeated_phrase_score *= 0.80
    scores["repeated_phrase_loop"] = repeated_phrase_score
    pitched_repetition_body = clamp01(
        0.22 * true_repetition
        + 0.20 * tonal_presence
        + 0.16 * ramp(max(pitched, pitch_conf, f0_voiced), 0.45, 0.88)
        + 0.14 * ramp(max(sustained_tonal, non_event_tonal, tonal_balance), 0.48, 0.92)
        + 0.12 * ramp(max(pitched_onset - percussive_onset, tonal_balance), 0.04, 0.38)
        + 0.10 * ramp(max(plucked_authority, synth_source, keys_authority), 0.32, 0.70)
        + 0.08 * inverse_ramp(max(percussive, drumlike), 0.12, 0.36)
        + 0.08 * inverse_ramp(flatness, 0.10, 0.40)
        - 0.08 * noise_wash
    )
    if onset_count < 3.0 or true_repetition < 0.32:
        pitched_repetition_body *= 0.45
    if max(percussive, drumlike) >= 0.42 and pitched_onset < percussive_onset + 0.06:
        pitched_repetition_body *= 0.55
    scores[PITCHED_REPETITION_PHRASE] = pitched_repetition_body
    scores["solo_phrase"] = clamp01(
        0.52 * solo_isolation
        + 0.24 * tonal_presence
        + 0.16 * ramp(max(pitched, sustained_tonal, non_event_tonal), 0.55, 0.98)
        + 0.08 * inverse_ramp(max(percussive, drumlike), 0.04, 0.30)
    )
    scores["layered_phrase"] = clamp01(
        0.48 * layered_loop + 0.24 * true_repetition + 0.16 * tonal_presence + 0.12 * band_spread
    )
    scores["instrument_plus_fx_loop"] = instrument_plus_fx
    scores["mixed_instrument_loop"] = clamp01(
        0.42 * layered_loop
        + 0.24 * band_spread
        + 0.18 * true_repetition
        + 0.16 * inverse_ramp(solo_isolation, 0.20, 0.62)
    )
    scores["compound_musical_loop"] = clamp01(
        0.40 * layered_loop
        + 0.24 * instrument_plus_fx
        + 0.18 * true_repetition
        + 0.12 * band_spread
        + 0.06 * inverse_ramp(solo_isolation, 0.24, 0.70)
    )
    clean_bass_shape_evidence = bool(
        low_total >= 0.42
        and pitch_conf >= 0.58
        and (f0_voiced >= 0.28 or bass_identity_source >= 0.60)
        and max(sustained_tonal, non_event_tonal, pitched) >= 0.62
        and flatness <= 0.24
        and noise_wash <= 0.46
        and measured_texture < 0.52
        and measured_fx_motion < 0.36
    )
    compact_struck_percussion = evidence_number(facts, "compact_struck_tonal_percussion_score")
    struck_percussion_material = max(
        evidence_number(facts, "hand_drum_membrane_score"),
        evidence_number(facts, "pitched_metal_percussion_score"),
        evidence_number(facts, "struck_wood_score"),
    )
    drum_material_branch = max(
        evidence_number(facts, "drum_kick_source_score"),
        evidence_number(facts, "drum_snare_source_score"),
        evidence_number(facts, "drum_clap_source_score"),
        evidence_number(facts, "drum_tom_conga_source_score"),
        evidence_number(facts, "drum_rim_stick_source_score"),
        evidence_number(facts, "drum_cymbal_source_score"),
        evidence_number(facts, "drum_guiro_scrape_source_score"),
        evidence_number(facts, "drum_metallic_percussion_source_score"),
        evidence_number(facts, "drum_shaker_tambourine_source_score"),
    )
    pitched_repetition_shape_counter = bool(
        scores.get(PITCHED_REPETITION_PHRASE, 0.0) >= 0.60
        and max(pitched_onset, tonal_balance, pitched, pitch_conf) >= max(percussive_onset, percussive, drumlike) + 0.04
        and max(plucked_authority, synth_source, keys_authority, tonal_presence) >= 0.42
        and max(percussive, drumlike) <= 0.38
    )
    repeated_noisy_drum_loop_body = bool(
        onset_count >= 8.0
        and true_repetition >= 0.68
        and span >= 0.62
        and drum_material_branch >= 0.50
        and (
            drumlike >= 0.12
            or percussive >= 0.52
            or percussive_onset >= pitched_onset - 0.02
            or measured_drum_loop >= 0.66
        )
        and (
            measured_drum_loop >= 0.48
            or max(drum_material_branch, percussive_onset) >= 0.58
            or (high_event >= 0.72 and drum_material_branch >= 0.68)
        )
        and not clean_bass_shape_evidence
        and not pitched_repetition_shape_counter
    )
    directional_fx_motion = max(
        measured_fx_motion,
        measured_transition,
        measured_number(facts, "fx_whoosh_sweep_score"),
        measured_number(facts, "fx_glitch_stutter_score"),
        measured_number(facts, "fx_drop_downlifter_score"),
        measured_number(facts, "fx_riser_build_score"),
        measured_number(facts, "fx_reverse_score"),
        measured_number(facts, "fx_radio_electrical_score"),
    )
    strong_directional_fx_motion = bool(
        abs(slope) >= 0.30
        and directional_fx_motion >= 0.68
        and max(scores["whoosh_sweep"], scores["glitch_stutter"], scores["hybrid_fx_motion"]) >= 0.62
    )
    if strong_directional_fx_motion:
        # Pulsed sweeps/downlifters can contain many sharp broadband events, so
        # generic drum-loop detectors often fire.  A strong monotonic spectral
        # move with FX-motion support is a stronger structural owner than the
        # repeated-noisy drum-loop shortcut.  Later claim producers still need
        # compatible FX evidence before routing this to FX.
        repeated_noisy_drum_loop_body = False
    real_drum_material_body = bool(
        drum_material_branch >= 0.58
        and max(percussive, drumlike, percussive_onset) >= 0.44
        and not strong_directional_fx_motion
        and directional_fx_motion < 0.58
    )
    synthetic_alert_fx_body = bool(
        duration >= 1.20
        and synthetic_alert_fx_pressure >= 0.54
        and synthetic_alert_fx_cluster >= 0.49
        and max(
            scores["hybrid_fx_motion"],
            scores[DESIGNED_TONAL_FX_SHAPE],
            scores["siren_alarm_tone"],
            scores["impact_with_tail"],
        )
        >= 0.50
        and max(percussive, drumlike) <= 0.14
        and f0_voiced <= 0.38
        and not source_safe_voiced_phrase_body
        and not real_drum_material_body
    )
    designed_low_fx_body = bool(
        scores[DESIGNED_LOW_FX_SHAPE] >= 0.60
        and fx_low_design >= 0.56
        and not real_drum_material_body
        and not (clean_bass_shape_evidence and fx_low_design < bass_identity_source + 0.08)
    )
    designed_motion_directional_or_unpitched = bool(
        abs(slope) >= 0.22
        or pitch_conf <= 0.35
        or pitched <= 0.35
        or (fx_motion_design >= 0.74 and max(sustained_tonal, non_event_tonal) <= 0.48)
    )
    designed_motion_fx_loop_body = bool(
        scores[DESIGNED_MOTION_FX_LOOP] >= 0.62
        and fx_motion_design >= 0.48
        and true_repetition >= 0.50
        and designed_motion_directional_or_unpitched
        and not real_drum_material_body
    )
    designed_tonal_fx_body = bool(
        not source_safe_voiced_phrase_body
        and scores[DESIGNED_TONAL_FX_SHAPE] >= 0.62
        and fx_tonal_design >= 0.54
        and not real_drum_material_body
        and not (
            max(plucked_authority, keys_authority, synth_source) >= 0.72
            and noise_wash <= 0.34
            and max(measured_fx_motion, measured_transition) < 0.42
            and synthetic_alert_fx_pressure < 0.56
        )
    )
    if synthetic_alert_fx_body:
        synthetic_alert_floor = min(
            0.94,
            max(
                scores[DESIGNED_TONAL_FX_SHAPE],
                scores["hybrid_fx_motion"] + 0.08,
                scores["siren_alarm_tone"] + 0.12,
                synthetic_alert_fx_pressure + 0.12,
            ),
        )
        scores[DESIGNED_TONAL_FX_SHAPE] = max(scores[DESIGNED_TONAL_FX_SHAPE], synthetic_alert_floor)
        scores["hybrid_fx_motion"] = max(scores["hybrid_fx_motion"], min(0.93, synthetic_alert_floor - 0.02))
        scores["siren_alarm_tone"] = max(scores["siren_alarm_tone"], min(0.92, synthetic_alert_floor - 0.04))
        scores["bass_phrase"] = min(scores["bass_phrase"], synthetic_alert_floor - 0.03)
        scores[PITCHED_REPETITION_PHRASE] = min(
            scores[PITCHED_REPETITION_PHRASE],
            synthetic_alert_floor - 0.04,
        )
        scores["repeated_phrase_loop"] = min(scores["repeated_phrase_loop"], synthetic_alert_floor - 0.05)
    if designed_low_fx_body:
        designed_low_floor = min(0.94, max(scores[DESIGNED_LOW_FX_SHAPE], fx_low_design + 0.08))
        scores[DESIGNED_LOW_FX_SHAPE] = max(scores[DESIGNED_LOW_FX_SHAPE], designed_low_floor)
        scores["bass_phrase"] = min(scores["bass_phrase"], designed_low_floor - 0.03)
        scores[PITCHED_REPETITION_PHRASE] = min(scores[PITCHED_REPETITION_PHRASE], designed_low_floor - 0.04)
        scores["beat_loop"] = min(scores["beat_loop"], designed_low_floor - 0.05)
    if designed_motion_fx_loop_body:
        designed_motion_floor = min(0.95, max(scores[DESIGNED_MOTION_FX_LOOP], fx_motion_design + 0.08))
        scores[DESIGNED_MOTION_FX_LOOP] = max(scores[DESIGNED_MOTION_FX_LOOP], designed_motion_floor)
        scores["hybrid_fx_motion"] = max(scores["hybrid_fx_motion"], min(0.93, designed_motion_floor - 0.02))
        scores["beat_loop"] = min(scores["beat_loop"], designed_motion_floor - 0.04)
        scores["top_loop"] = min(scores["top_loop"], designed_motion_floor - 0.04)
        scores["repeated_phrase_loop"] = min(scores["repeated_phrase_loop"], designed_motion_floor - 0.03)
    if designed_tonal_fx_body:
        designed_tonal_floor = min(0.94, max(scores[DESIGNED_TONAL_FX_SHAPE], fx_tonal_design + 0.06))
        scores[DESIGNED_TONAL_FX_SHAPE] = max(scores[DESIGNED_TONAL_FX_SHAPE], designed_tonal_floor)
        scores["pitched_phrase"] = min(scores["pitched_phrase"], designed_tonal_floor - 0.03)
        scores[PITCHED_PHRASE_SHAPE] = min(scores[PITCHED_PHRASE_SHAPE], designed_tonal_floor - 0.03)
        scores[PITCHED_REPETITION_PHRASE] = min(scores[PITCHED_REPETITION_PHRASE], designed_tonal_floor - 0.03)
    voice_body = max(
        evidence_number(facts, "voice_score"),
        evidence_number(facts, "human_spoken_voice_score"),
        evidence_number(facts, "human_breath_mouth_score"),
    )
    short_percussive_hit_body = bool(
        onset_count <= 3.0
        and true_repetition <= 0.24
        and evidence_number(facts, "role_one_shot_score") >= 0.72
        and evidence_number(facts, "drum_hit_score") >= 0.62
        and percussive_onset >= 0.62
        and pitched <= 0.20
        and pitch_conf <= 0.35
    )
    repeated_percussive_drum_loop_body = bool(
        onset_count >= 6.0
        and true_repetition >= 0.62
        and measured_drum_loop >= 0.54
        and max(percussive, drumlike, percussive_onset) >= 0.50
        and measured_texture < 0.82
        and not strong_directional_fx_motion
        and not clean_bass_shape_evidence
        and not pitched_repetition_shape_counter
    )
    tonal_struck_percussive_hit_body = bool(
        onset_count <= 5.0
        and true_repetition <= 0.38
        and evidence_number(facts, "role_one_shot_score") >= 0.58
        and compact_struck_percussion >= 0.74
        and struck_percussion_material >= 0.56
        and drum_material_branch >= 0.50
        and max(percussive_onset, evidence_number(facts, "drum_hit_score"), drum_material_branch) >= 0.52
        and voice_body < 0.62
        and not clean_bass_shape_evidence
        and not (
            keys_authority >= 0.70
            and keys_authority >= drum_material_branch + 0.18
            and struck_percussion_material < 0.70
        )
    )
    event_texture_body = bool(
        measured_texture >= 0.58
        and not clean_bass_shape_evidence
        and not short_percussive_hit_body
        and not tonal_struck_percussive_hit_body
        and not repeated_percussive_drum_loop_body
        and not repeated_noisy_drum_loop_body
    )
    low_tail_fx_body = bool(
        measured_fx_tail >= 0.50
        and tail >= 0.30
        and not facts.is_loop_like
        and onset_count <= 8.0
        and not clean_bass_shape_evidence
    )
    weak_loop_periodicity = max(
        evidence_number(facts, "role_loop_score"),
        v(values, "loop_onset_periodicity"),
        v(values, "loop_pulse_clarity"),
        v(values, "librosa_loop_confidence"),
    )
    impact_tail_fx_body = bool(
        measured_fx_tail >= 0.52
        and tail >= 0.24
        and onset_count <= 12.0
        and f0_voiced <= 0.25
        and weak_loop_periodicity <= 0.36
        and not clean_bass_shape_evidence
        and not repeated_percussive_drum_loop_body
        and not repeated_noisy_drum_loop_body
    )
    motion_texture_body = bool(
        max(measured_fx_motion, measured_transition, measured_fx_tail) >= 0.38
        and max(noise_wash, measured_texture) >= 0.42
        and not clean_bass_shape_evidence
        and not (true_repetition >= 0.72 and measured_drum_loop >= 0.62)
    )
    if event_texture_body:
        texture_shape_floor = min(0.96, max(measured_texture + 0.06, scores["texture_bed"]))
        scores["texture_bed"] = max(scores["texture_bed"], texture_shape_floor)
        scores["noise_texture"] = max(scores["noise_texture"], min(0.94, measured_texture + 0.04))
        scores["static_bed"] = max(scores["static_bed"], min(0.92, 0.72 * measured_texture + 0.22 * noise_wash))
        scores["beat_loop"] = min(scores["beat_loop"], scores["texture_bed"] - 0.03)
        scores["top_loop"] = min(scores["top_loop"], scores["texture_bed"] - 0.04)
        scores["repeated_phrase_loop"] = min(scores["repeated_phrase_loop"], scores["texture_bed"] - 0.02)
    if repeated_percussive_drum_loop_body or repeated_noisy_drum_loop_body:
        drum_loop_floor = min(
            0.95,
            max(
                scores["beat_loop"],
                0.42 * true_repetition
                + 0.24 * max(measured_drum_loop, drum_material_branch)
                + 0.34 * max(percussive, drumlike, percussive_onset, drum_material_branch),
            ),
        )
        scores["beat_loop"] = max(scores["beat_loop"], drum_loop_floor)
        if high_event >= 0.50 and high_total >= 0.16:
            scores["top_loop"] = max(scores["top_loop"], min(0.94, drum_loop_floor - 0.02))
        scores["texture_bed"] = min(scores["texture_bed"], drum_loop_floor - 0.04)
        scores["noise_texture"] = min(scores["noise_texture"], drum_loop_floor - 0.05)
        scores["static_bed"] = min(scores["static_bed"], drum_loop_floor - 0.06)
        scores[PITCHED_REPETITION_PHRASE] = min(scores[PITCHED_REPETITION_PHRASE], drum_loop_floor - 0.04)
    if short_percussive_hit_body:
        hit_floor = min(0.96, max(scores["single_hit"], percussive_onset + 0.04))
        scores["single_hit"] = max(scores["single_hit"], hit_floor)
        scores["hit_with_tail"] = max(scores["hit_with_tail"], min(0.90, hit_floor - 0.03))
        scores["texture_bed"] = min(scores["texture_bed"], hit_floor - 0.04)
        scores["noise_texture"] = min(scores["noise_texture"], hit_floor - 0.05)
        scores["static_bed"] = min(scores["static_bed"], hit_floor - 0.06)
    if tonal_struck_percussive_hit_body:
        struck_hit_floor = min(
            0.96,
            max(
                scores["single_hit"],
                scores["hit_with_tail"],
                0.50 * compact_struck_percussion
                + 0.28 * struck_percussion_material
                + 0.22 * max(percussive_onset, drum_material_branch),
            ),
        )
        if tail >= 0.14 or duration >= 0.42:
            scores["hit_with_tail"] = max(scores["hit_with_tail"], struck_hit_floor)
            scores["single_hit"] = max(scores["single_hit"], min(0.92, struck_hit_floor - 0.05))
        else:
            scores["single_hit"] = max(scores["single_hit"], struck_hit_floor)
            scores["hit_with_tail"] = max(scores["hit_with_tail"], min(0.90, struck_hit_floor - 0.04))
        scores["solo_phrase"] = min(scores["solo_phrase"], struck_hit_floor - 0.03)
        scores["pitched_phrase"] = min(scores["pitched_phrase"], struck_hit_floor - 0.04)
        scores[PITCHED_PHRASE_SHAPE] = min(scores[PITCHED_PHRASE_SHAPE], struck_hit_floor - 0.04)
        scores[PITCHED_REPETITION_PHRASE] = min(scores[PITCHED_REPETITION_PHRASE], struck_hit_floor - 0.04)
        scores["bass_phrase"] = min(scores["bass_phrase"], struck_hit_floor - 0.03)
        scores["texture_bed"] = min(scores["texture_bed"], struck_hit_floor - 0.04)
        scores["noise_texture"] = min(scores["noise_texture"], struck_hit_floor - 0.05)
        scores["static_bed"] = min(scores["static_bed"], struck_hit_floor - 0.06)
    if low_tail_fx_body:
        tail_shape_floor = min(0.94, max(scores["impact_with_tail"], measured_fx_tail + 0.12))
        scores["impact_with_tail"] = max(scores["impact_with_tail"], tail_shape_floor)
        scores["hit_with_tail"] = max(scores["hit_with_tail"], min(0.90, tail_shape_floor - 0.03))
        scores["bass_phrase"] = min(scores["bass_phrase"], tail_shape_floor - 0.02)
        scores[PITCHED_REPETITION_PHRASE] = min(scores[PITCHED_REPETITION_PHRASE], tail_shape_floor - 0.03)
    if impact_tail_fx_body:
        impact_tail_floor = min(0.94, max(scores["impact_with_tail"], measured_fx_tail + 0.18))
        scores["impact_with_tail"] = max(scores["impact_with_tail"], impact_tail_floor)
        scores["hit_with_tail"] = max(scores["hit_with_tail"], min(0.90, impact_tail_floor - 0.03))
        scores["bass_phrase"] = min(scores["bass_phrase"], impact_tail_floor - 0.02)
        scores[PITCHED_REPETITION_PHRASE] = min(
            scores[PITCHED_REPETITION_PHRASE],
            impact_tail_floor - 0.03,
        )
        scores["repeated_phrase_loop"] = min(scores["repeated_phrase_loop"], impact_tail_floor - 0.04)
    if motion_texture_body:
        motion_shape_floor = min(
            0.93, max(scores["hybrid_fx_motion"], measured_fx_tail + 0.08, measured_transition + 0.10)
        )
        scores["hybrid_fx_motion"] = max(scores["hybrid_fx_motion"], motion_shape_floor)
        scores["bass_phrase"] = min(scores["bass_phrase"], motion_shape_floor - 0.02)
        scores[PITCHED_PHRASE_SHAPE] = min(scores[PITCHED_PHRASE_SHAPE], motion_shape_floor - 0.03)
        scores["pitched_phrase"] = min(scores["pitched_phrase"], motion_shape_floor - 0.03)
    drum_repetition_loop = clamp01(
        0.42 * true_repetition
        + 0.24 * ramp(max(percussive, drumlike), 0.18, 0.62)
        + 0.16 * ramp(onset_count, 6.0, 24.0)
        + 0.10 * ramp(span, 0.60, 0.92)
        + 0.08 * inverse_ramp(sustained_tonal, 0.45, 0.85)
    )
    rhythmic_break_loop = clamp01(
        0.30 * true_repetition
        + 0.18 * ramp(onset_count, 6.0, 28.0)
        + 0.16 * ramp(span, 0.42, 0.90)
        + 0.16 * max(ramp(low_event, 0.62, 0.94), ramp(high_event, 0.32, 0.86))
        + 0.14 * inverse_ramp(attack, 0.006, 0.12)
        + 0.06 * ramp(onset_density, 0.8, 5.5)
    )
    # Low-end breakbeats and synthetic hat loops often look like pitched
    # phrases because their events have clear pitch centers.  Shape should not
    # require the source panel's drumlike/percussive labels to have already
    # fired.  Repeated fast-attacked low/high event streams are structural drum
    # loops at the shape layer; slower attacked multi-instrument loops remain
    # pitched or mixed musical phrases.
    clean_tonal_low_arp_body = bool(
        low_event >= 0.62
        and max(pitched, pitch_conf) >= 0.82
        and max(sustained_tonal, non_event_tonal) >= 0.82
        and max(percussive, drumlike) <= 0.08
    )
    pitched_repetition_drum_decoy = pitched_repetition_shape_counter
    if pitched_repetition_drum_decoy:
        scores["repeated_phrase_loop"] = min(
            scores["repeated_phrase_loop"],
            max(0.0, scores[PITCHED_REPETITION_PHRASE] - 0.02),
        )
        scores["beat_loop"] = min(scores["beat_loop"], max(0.0, scores[PITCHED_REPETITION_PHRASE] - 0.04))
        scores["top_loop"] = min(scores["top_loop"], max(0.0, scores[PITCHED_REPETITION_PHRASE] - 0.05))
    rhythmic_break_has_drum_body = bool(
        not pitched_repetition_drum_decoy
        and not strong_directional_fx_motion
        and (
            max(percussive, drumlike) >= 0.52
            or (high_event >= 0.42 and high_total >= 0.18 and percussive_onset >= pitched_onset - 0.04)
            or (low_event >= 0.62 and max(percussive, drumlike) >= 0.18 and not clean_tonal_low_arp_body)
            or repeated_noisy_drum_loop_body
        )
    )
    measured_low_rhythmic_drum_loop = bool(
        measured_drum_loop >= 0.62
        and true_repetition >= 0.62
        and onset_count >= 6.0
        and low_event >= 0.70
        and max(mid_event, high_event) <= 0.30
    )
    if measured_low_rhythmic_drum_loop:
        rhythmic_break_has_drum_body = True
        rhythmic_break_loop = max(rhythmic_break_loop, measured_drum_loop, true_repetition)
    if rhythmic_break_loop >= 0.62 and true_repetition >= 0.62 and onset_count >= 6.0 and rhythmic_break_has_drum_body:
        preferred_drum_shape = "top_loop" if high_event >= 0.42 and high_total >= 0.18 else "beat_loop"
        strongest_transition = max(scores.get("transition_riser", 0.0), scores.get("transition_drop", 0.0))
        scores[preferred_drum_shape] = max(
            scores[preferred_drum_shape],
            rhythmic_break_loop + 0.03,
            strongest_transition + 0.02,
            scores.get(PITCHED_REPETITION_PHRASE, 0.0) + 0.02,
        )
        scores["bass_phrase"] = min(scores["bass_phrase"], rhythmic_break_loop - 0.02)
        scores["pitched_phrase"] = min(scores["pitched_phrase"], rhythmic_break_loop - 0.03)
        scores[PITCHED_PHRASE_SHAPE] = min(scores[PITCHED_PHRASE_SHAPE], rhythmic_break_loop - 0.03)
        scores["transition_riser"] = min(scores["transition_riser"], scores[preferred_drum_shape] - 0.04)
        scores["transition_drop"] = min(scores["transition_drop"], scores[preferred_drum_shape] - 0.04)
    if (
        scores["repeated_phrase_loop"] >= 0.72
        and drum_repetition_loop >= 0.62
        and max(percussive, drumlike) >= 0.18
        and max(scores["beat_loop"], scores["top_loop"]) >= 0.60
    ):
        # A busy drum loop can have enough pitch/tonal body to look like a
        # repeated phrase.  Preserve the structural loop read, but prefer the
        # drum-loop shape when measured percussive/drumlike activity is present.
        drum_loop_shape = (
            "top_loop" if scores["top_loop"] >= scores["beat_loop"] and high_total >= 0.36 else "beat_loop"
        )
        scores[drum_loop_shape] = max(scores[drum_loop_shape], scores["repeated_phrase_loop"] + 0.01)
    if scores["glitch_stutter"] >= 0.70 and noise_wash >= 0.58:
        scores["repeated_phrase_loop"] *= 0.70
        scores["beat_loop"] *= 0.85

    single_event_gate = 1.0 if (facts.is_single_event_like or onset_count <= 1.5 or span <= 0.05) else 0.0
    if single_event_gate:
        # A low, pitched, single transient can be a kick/sub hit.  It should
        # not become a bass phrase or pad merely because its body is tonal.
        phrase_suppression = 0.35 if duration < 1.2 else 0.60
        scores["bass_phrase"] *= phrase_suppression
        scores["pitched_phrase"] *= phrase_suppression
        scores[PITCHED_PHRASE_SHAPE] *= phrase_suppression
        scores["sustained_pad"] *= 0.55
        low_single_hit = bool(
            low_event >= 0.82
            and max(mid_event, high_event) <= 0.16
            and attack <= 0.06
            and temporal <= 0.48
            and (tail >= 0.18 or sustain >= 0.64 or low_total >= 0.42)
        )
        if low_single_hit:
            # Do not let a resonant kick/sub hit become a structural solo phrase
            # only because the tail is pitched.  This is still a shape decision:
            # the compatible family remains broad Drums/FX/Instruments.
            scores["hit_with_tail"] = max(scores["hit_with_tail"], scores["solo_phrase"] + 0.01)
            scores["solo_phrase"] = min(
                scores["solo_phrase"], max(scores["single_hit"], scores["hit_with_tail"]) - 0.02
            )

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = sorted_scores[0]
    secondary = sorted_scores[1][0]
    if primary_score < 0.42:
        primary = "ambiguous_shape"
        secondary = sorted_scores[0][0]
    reason = (
        f"shape={primary} confidence={primary_score:.2f} "
        f"events={onset_count:.1f} span={span:.2f} pulse={pulse_regularity:.2f} "
        f"drumlike={drumlike:.2f} pitched={pitched:.2f} slope={slope:.2f} "
        f"solo={solo_isolation:.2f} layered={layered_loop:.2f} echo={echo_tail:.2f}"
    )
    return ShapeEvidence(
        primary_shape=primary,
        secondary_shape=secondary,
        confidence=round(clamp01(primary_score), 6),
        onset_count=round(onset_count, 6),
        onset_density_hz=round(onset_density, 6),
        pulse_regularity=round(pulse_regularity, 6),
        onset_span_ratio=round(span, 6),
        temporal_centroid_ratio=round(temporal, 6),
        attack_rise_time_norm=round(attack, 6),
        sustain_ratio=round(sustain, 6),
        tail_ratio=round(tail, 6),
        low_event_ratio=round(low_event, 6),
        mid_event_ratio=round(mid_event, 6),
        high_event_ratio=round(high_event, 6),
        drumlike_frame_ratio=round(drumlike, 6),
        pitched_event_ratio=round(pitched, 6),
        percussive_event_ratio=round(percussive, 6),
        sustained_tonal_frame_ratio=round(sustained_tonal, 6),
        non_event_tonal_ratio=round(non_event_tonal, 6),
        pitch_confidence=round(pitch_conf, 6),
        f0_voiced_ratio=round(f0_voiced, 6),
        spectral_flatness_mean=round(flatness, 6),
        spectral_entropy_mean=round(entropy, 6),
        centroid_slope_norm=round(slope, 6),
        shape_scores=[(name, round(clamp01(score), 6)) for name, score in sorted_scores],
        solo_isolation_score=round(solo_isolation, 6),
        layered_loop_score=round(layered_loop, 6),
        instrument_plus_fx_loop_score=round(instrument_plus_fx, 6),
        echo_tail_score=round(echo_tail, 6),
        true_repetition_score=round(true_repetition, 6),
        reason=reason,
    )


def apply_shape_memory(
    evidence: ShapeEvidence,
    *,
    facts: SharedAudioFacts,
    brain: dict[str, Any],
) -> ShapeEvidence:
    """Return shape evidence adjusted by learned GUI correction memory."""
    if evidence.primary_shape == "broken_or_tiny":
        return evidence
    match = shape_memory_match_for_facts(brain, facts.feature_vector)
    if match.example_count <= 0:
        return evidence
    memory_fields = shape_memory_fields(match)
    if not match.matched or not match.shape:
        return replace(evidence, **memory_fields)

    scores = {str(name): float(score) for name, score in evidence.shape_scores}
    current = float(scores.get(match.shape, 0.0))
    teacher_score = float(match.confidence)
    if match.match_kind == "fingerprint":
        teacher_score = max(teacher_score, min(0.98, float(evidence.confidence) + 0.035))
    scores[match.shape] = max(current, teacher_score)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = sorted_scores[0]
    secondary = sorted_scores[1][0] if len(sorted_scores) > 1 else evidence.secondary_shape
    if primary_score < 0.42:
        primary = "ambiguous_shape"
        secondary = sorted_scores[0][0]
    reason = (
        f"{evidence.reason}; learned_shape_memory={match.shape} "
        f"kind={match.match_kind} confidence={match.confidence:.2f} "
        f"distance={match.nearest_distance:.2f}"
    )
    return replace(
        evidence,
        primary_shape=primary,
        secondary_shape=secondary,
        confidence=round(clamp01(primary_score), 6),
        shape_scores=[(name, round(clamp01(score), 6)) for name, score in sorted_scores],
        reason=reason,
        **memory_fields,
    )


def shape_memory_fields(match: ShapeMemoryMatch) -> dict[str, Any]:
    """Return ``ShapeEvidence`` keyword fields for learned shape diagnostics."""
    return {
        "learned_shape_memory_enabled": match.example_count > 0,
        "learned_shape_memory_matched": bool(match.matched),
        "learned_shape_memory_shape": str(match.shape),
        "learned_shape_memory_confidence": float(match.confidence),
        "learned_shape_memory_nearest_distance": float(match.nearest_distance),
        "learned_shape_memory_threshold": float(match.threshold),
        "learned_shape_memory_example_count": int(match.example_count),
        "learned_shape_memory_effective_weight": int(match.effective_weight),
        "learned_shape_memory_match_kind": str(match.match_kind),
        "learned_shape_memory_policy": "shape_voter_teacher_fingerprint_memory",
    }


def apply_voter_memory(
    evidence: ShapeEvidence,
    *,
    facts: SharedAudioFacts,
) -> ShapeEvidence:
    """Return shape evidence adjusted by learned voter-role memory."""
    if evidence.primary_shape == "broken_or_tiny":
        return evidence
    memory = facts.evidence.get("learned_voter_memory", {}) if isinstance(facts.evidence, dict) else {}
    if not isinstance(memory, Mapping) or not bool(memory.get("enabled")):
        return evidence
    memory_fields = voter_memory_shape_fields(memory)
    if not bool(memory.get("matched")):
        return replace(evidence, **memory_fields)
    target_shape = str(memory.get("shape", ""))
    confidence = clamp01(v(memory, "confidence", 0.0))
    if not target_shape or confidence < 0.72:
        return replace(evidence, **memory_fields)

    scores = {str(name): float(score) for name, score in evidence.shape_scores}
    current = float(scores.get(target_shape, 0.0))
    teacher_score = max(current, min(0.95, confidence + 0.02))
    scores[target_shape] = teacher_score
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = sorted_scores[0]
    secondary = sorted_scores[1][0] if len(sorted_scores) > 1 else evidence.secondary_shape
    reason = (
        f"{evidence.reason}; learned_voter_memory_shape={target_shape} "
        f"role={memory.get('role', '')} confidence={confidence:.2f}"
    )
    return replace(
        evidence,
        primary_shape=primary,
        secondary_shape=secondary,
        confidence=round(clamp01(primary_score), 6),
        shape_scores=[(name, round(clamp01(score), 6)) for name, score in sorted_scores],
        reason=reason,
        **memory_fields,
    )


def voter_memory_shape_fields(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``ShapeEvidence`` keyword fields for voter-memory diagnostics."""
    return {
        "learned_voter_memory_enabled": bool(memory.get("enabled")),
        "learned_voter_memory_matched": bool(memory.get("matched")),
        "learned_voter_memory_shape": str(memory.get("shape", "")),
        "learned_voter_memory_role": str(memory.get("role", "")),
        "learned_voter_memory_confidence": v(memory, "confidence", 0.0),
        "learned_voter_memory_match_kind": str(memory.get("match_kind", "none")),
        "learned_voter_memory_policy": str(memory.get("policy", "")),
    }


def v(values: Mapping[str, float], name: str, default: float = 0.0) -> float:
    try:
        return float(values.get(name, default) or default)
    except Exception:
        return float(default)


def evidence_number(facts: SharedAudioFacts, name: str, default: float = 0.0) -> float:
    """Read a finite numeric value from the shared evidence dictionary."""
    evidence = getattr(facts, "evidence", {})
    if not isinstance(evidence, Mapping):
        return float(default)
    try:
        return float(evidence.get(name, default) or default)
    except Exception:
        return float(default)


def subpanel_number(facts: SharedAudioFacts, name: str, default: float = 0.0) -> float:
    """Read one measured low-level physics subpanel score, if present."""
    evidence = getattr(facts, "evidence", {})
    if not isinstance(evidence, Mapping):
        return float(default)
    subpanels = evidence.get("physics_subpanels")
    if isinstance(subpanels, Mapping):
        flat = subpanels.get("flat")
        if isinstance(flat, Mapping):
            try:
                return float(flat.get(name, default) or default)
            except Exception:
                return float(default)
    return float(default)


def measured_number(facts: SharedAudioFacts, name: str, default: float = 0.0) -> float:
    """Read a numeric feature from either top-level evidence or subpanel evidence."""
    return max(evidence_number(facts, name, default), subpanel_number(facts, name, default))


def role_number(facts: SharedAudioFacts, name: str, default: float = 0.0) -> float:
    """Read measured-role strength from the nested role packet."""
    evidence = getattr(facts, "evidence", {})
    if not isinstance(evidence, Mapping):
        return float(default)
    roles = evidence.get("measured_roles")
    if not isinstance(roles, Mapping):
        return float(default)
    try:
        return float(roles.get(name, default) or default)
    except Exception:
        return float(default)


def expm1_safe(value: float) -> float:
    import math

    try:
        return float(math.expm1(max(0.0, float(value))))
    except Exception:
        return 0.0


def ramp(number: float, start: float, full: float) -> float:
    if full <= start:
        return 0.0
    return clamp01((float(number) - float(start)) / (float(full) - float(start)))


def inverse_ramp(number: float, good_at_or_below: float, bad_at_or_above: float) -> float:
    if bad_at_or_above <= good_at_or_below:
        return 0.0
    return clamp01((float(bad_at_or_above) - float(number)) / (float(bad_at_or_above) - float(good_at_or_below)))


def shape_compatible_tops(primary_shape: str) -> list[str]:
    """Return broad top families compatible with a shape claim."""
    if primary_shape in {"beat_loop", "top_loop"}:
        return ["Drums"]
    if primary_shape in {
        "bass_phrase",
        "pitched_phrase",
        PITCHED_PHRASE_SHAPE,
        PITCHED_REPETITION_PHRASE,
        "sustained_pad",
        "solo_phrase",
        "layered_phrase",
        "repeated_phrase_loop",
        "compound_musical_loop",
        "mixed_instrument_loop",
        "instrument_plus_fx_loop",
    }:
        return ["Instruments", "Textures"]
    if primary_shape == "vocal_phrase":
        # Backward compatibility for older synthetic tests that inject the old
        # source-sounding shape name directly.  ShapeVoter no longer emits it.
        return ["Instruments", "FX"]
    if primary_shape in {
        "transition_riser",
        "transition_drop",
        "reverse_swell",
        "whoosh_sweep",
        "glitch_stutter",
        "ui_blip",
        "siren_alarm_tone",
        "texture_bed",
        "foley_action",
        "mechanical_motion",
        "hybrid_fx_motion",
        DESIGNED_LOW_FX_SHAPE,
        DESIGNED_MOTION_FX_LOOP,
        DESIGNED_TONAL_FX_SHAPE,
    }:
        return ["FX"]
    if primary_shape in {"noise_texture", "static_bed"}:
        return ["Textures", "FX"]
    if primary_shape in {"single_hit", "hit_with_tail", "impact_with_tail", "echo_tail_hit"}:
        return ["Drums", "FX", "Instruments"]
    return []
