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
from dataclasses import asdict, dataclass
from typing import Any

from aaron_sound_sorter.domain.models import AudioPhysics, CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.domain.policies import ShapeVoterPolicy
from aaron_sound_sorter.voters.base import Voter, clamp01

SHAPE_TOP = "_SHAPE_DIAGNOSTIC"
PITCHED_PHRASE_SHAPE = "pitched_phrase_shape"


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
    duration = float(facts.evidence.get("duration_sec", 0.0) or 0.0) if isinstance(facts.evidence, dict) else 0.0

    low_total = clamp01(v(values, "sub_bass_ratio_lt_150hz") + v(values, "bass_ratio_150_500hz"))
    high_total = clamp01(v(values, "presence_ratio_2000_8000hz") + v(values, "air_ratio_gt_8000hz"))
    mid_total = clamp01(v(values, "mid_ratio_500_2000hz"))
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
    scores["echo_tail_hit"] = echo_tail
    repeated_phrase_score = true_repetition
    if max(layered_loop, instrument_plus_fx) >= 0.66 and band_spread >= 0.45:
        repeated_phrase_score *= 0.80
    scores["repeated_phrase_loop"] = repeated_phrase_score
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
    rhythmic_break_has_drum_body = bool(
        max(percussive, drumlike) >= 0.52
        or (high_event >= 0.42 and high_total >= 0.18)
        or (
            low_event >= 0.62
            and (max(percussive, drumlike) >= 0.18 or pulse_regularity >= 0.38)
        )
    )
    if rhythmic_break_loop >= 0.62 and true_repetition >= 0.62 and onset_count >= 6.0 and rhythmic_break_has_drum_body:
        preferred_drum_shape = "top_loop" if high_event >= 0.42 and high_total >= 0.18 else "beat_loop"
        strongest_transition = max(scores.get("transition_riser", 0.0), scores.get("transition_drop", 0.0))
        scores[preferred_drum_shape] = max(
            scores[preferred_drum_shape],
            rhythmic_break_loop + 0.03,
            strongest_transition + 0.02,
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


def v(values: Mapping[str, float], name: str, default: float = 0.0) -> float:
    try:
        return float(values.get(name, default) or default)
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


def average(*parts: float) -> float:
    usable = [clamp01(part) for part in parts]
    return sum(usable) / len(usable) if usable else 0.0


def shape_compatible_tops(primary_shape: str) -> list[str]:
    """Return broad top families compatible with a shape claim."""
    if primary_shape in {"beat_loop", "top_loop"}:
        return ["Drums"]
    if primary_shape in {
        "bass_phrase",
        "pitched_phrase",
        PITCHED_PHRASE_SHAPE,
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
    }:
        return ["FX"]
    if primary_shape in {"noise_texture", "static_bed"}:
        return ["Textures", "FX"]
    if primary_shape in {"single_hit", "hit_with_tail", "impact_with_tail", "echo_tail_hit"}:
        return ["Drums", "FX", "Instruments"]
    return []
