# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""PhysicsVoter implementation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aaron_sound_sorter.domain.models import AudioPhysics, SharedAudioFacts, VoterResult
from aaron_sound_sorter.domain.policies import ConsensusPolicy, PhysicsVoterPolicy
from aaron_sound_sorter.features import (
    HOP,
    TARGET_SR,
    clean_onset_peaks,
    frame_audio,
    read_audio,
    resample_linear,
    trim_and_normalize,
)
from aaron_sound_sorter.voters.base import Voter
from aaron_sound_sorter.voters.physics_layers import LayeredPhysicsScorer
from aaron_sound_sorter.voters.scoring_tools import (
    label_maps,
    profile_residual_score,
    role_compatibility_adjustment,
    role_strength,
    structure_penalty,
)


class PhysicsVoter(Voter):
    """Rank categories by full learned physics profile fit."""

    voter_name = "physics"

    def __init__(self, policy: PhysicsVoterPolicy | None = None) -> None:
        self.policy = policy or PhysicsVoterPolicy()
        self.layered_scorer = LayeredPhysicsScorer()
        super().__init__(self.policy.top_n)

    def vote(self, physics: AudioPhysics, facts: SharedAudioFacts, brain: dict[str, Any]) -> VoterResult:
        if facts.is_broken_or_tiny:
            return self.review_result(ConsensusPolicy().broken_or_tiny_label, "broken_or_tiny")
        labels = [str(label) for label in brain.get("labels", []) if str(label)]
        role_gate = facts.evidence.get("dynamic_role_gate", {}) if isinstance(facts.evidence, dict) else {}
        gate_can_filter = bool(isinstance(role_gate, dict) and role_gate.get("final_effects_enabled"))
        allowed = set(role_gate.get("allowed_labels", []) or []) if gate_can_filter else set()
        if gate_can_filter and allowed:
            labels = [label for label in labels if label in allowed]
        profiles = (
            brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles", {}), dict) else {}
        )
        if not labels or not profiles:
            return self.review_result(ConsensusPolicy().no_consensus_label, "physics_has_no_profiles")
        rows = self.score_labels(physics, facts, brain, labels, profiles)
        folder_map, top_map = label_maps(brain)
        guesses = self.ranked_guesses(
            rows,
            label_to_folder=folder_map,
            label_to_top=top_map,
            lower_score_is_better=True,
            default_reason="physics_profile_match",
        )
        diagnostics = {
            "candidate_count": len(labels),
            "profile_count": len(profiles),
            "returned_count": len(guesses),
            "role_gate_applied": bool(gate_can_filter and allowed),
            "role_gate_mode": str(role_gate.get("mode", "missing")) if isinstance(role_gate, dict) else "missing",
        }
        if isinstance(facts.evidence, dict):
            facts.evidence["physics_vote_result"] = {
                "diagnostics": diagnostics,
                "top_guesses": [
                    {
                        "label": guess.label,
                        "folder_path": guess.folder_path,
                        "top_family": guess.top_family,
                        "score": round(float(guess.score), 6),
                        "confidence": round(float(guess.confidence), 6),
                        "rank": int(guess.rank),
                        "reason": guess.reason,
                        "evidence": guess.evidence,
                    }
                    for guess in guesses[:100]
                ],
            }
        return VoterResult(
            voter_name=self.voter_name,
            guesses=guesses,
            diagnostics=diagnostics,
        )

    def apply_direct_body_profile_check(
        self,
        *,
        label: str,
        brain: dict[str, Any],
        facts: SharedAudioFacts,
        profile: dict[str, Any],
        full_profile_score: float,
        direct_vector: np.ndarray | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Blend in the tail-reduced direct/body view when it is compatible.

        The direct/body view is not a second hidden classifier. It can only make
        a label modestly more competitive when the direct source body fits that
        label better than the full reverberant/decaying file and the measured
        direct role is compatible with the candidate's broad family.
        """
        view = facts.evidence.get("direct_body_view", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(view, dict) or not view.get("available"):
            return full_profile_score, {"direct_body_profile_check": "not_available"}
        direct_values = view.get("feature_values_by_name", {})
        if not isinstance(direct_values, dict):
            return full_profile_score, {"direct_body_profile_check": "missing_values"}
        if direct_vector is None:
            direct_vector = np.asarray(
                [float(direct_values.get(name, 0.0) or 0.0) for name in brain.get("feature_names", [])],
                dtype=np.float32,
            )
        if direct_vector.size == 0:
            return full_profile_score, {"direct_body_profile_check": "missing_feature_names"}
        direct_score, direct_profile_evidence = profile_residual_score(direct_vector, profile, self.policy)
        if direct_score == float("inf") or direct_score >= full_profile_score:
            return full_profile_score, {
                "direct_body_profile_check": "not_better",
                "direct_body_profile_score": round(float(direct_score), 6) if direct_score != float("inf") else "inf",
            }
        roles = view.get("measured_roles", {})
        if not isinstance(roles, dict):
            roles = {}
        folder_map, top_map = label_maps(brain)
        folder = str(folder_map.get(label, label))
        top = str(top_map.get(label, folder.split("/", 1)[0]))
        compatible = self.direct_body_role_is_compatible(top, folder, roles)
        if not compatible:
            return full_profile_score, {
                "direct_body_profile_check": "better_but_role_incompatible",
                "direct_body_profile_score": round(float(direct_score), 6),
            }
        blended = min(full_profile_score, 0.70 * float(full_profile_score) + 0.30 * float(direct_score))
        return blended, {
            "direct_body_profile_check": "compatible_score_blend",
            "direct_body_profile_score": round(float(direct_score), 6),
            "direct_body_blended_score": round(float(blended), 6),
            "direct_body_used_feature_count": direct_profile_evidence.get("used_feature_count", 0),
        }

    def direct_body_role_is_compatible(self, top_family: str, folder_path: str, roles: dict[str, Any]) -> bool:
        """Return whether a candidate may benefit from the direct/body view."""
        low_path = str(folder_path).replace("\\", "/").lower()
        percussive = max(
            role_strength(roles, "percussive_one_shot"),
            role_strength(roles, "bright_drum_loop"),
            role_strength(roles, "percussive_drum_loop"),
            role_strength(roles, "low_rhythmic_drum_loop"),
        )
        voiced = max(
            role_strength(roles, "voiced_one_shot"),
            role_strength(roles, "vocal_music_phrase"),
        )
        pitched = max(
            role_strength(roles, "pitched_music_phrase"),
            role_strength(roles, "pitched_music_loop"),
        )
        bass = role_strength(roles, "bass_loop")
        if top_family == "Drums" and percussive >= 0.55:
            return True
        if top_family in {"FX", "Instruments"} and "voice" in low_path and voiced >= 0.55:
            return True
        if top_family == "Instruments" and pitched >= 0.65 and percussive < 0.60:
            return True
        return bool(top_family == "Instruments" and "bass" in low_path and bass >= 0.6 and percussive < 0.6)

    def apply_mallet_bell_eligibility_adjustment(
        self,
        *,
        label: str,
        folder_path: str,
        facts: SharedAudioFacts,
        current_score: float,
        first_arrival_telemetry: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Down-rank weak Mallet/Bell physics matches that lack struck evidence.

        This is a physics-voter scoring adjustment only.  It does not route to a
        different category and it never uses filenames.  Mallet/Bell candidates
        remain competitive when the measured audio has a struck attack plus a
        resonant body/tail.  Sustained wet pitched phrases, sax/reed-like loops,
        pads, and vocal-ish loops may still be tonal and bright, but they should
        not get a strong Mallet/Bell physics vote unless a struck-event body is
        actually present.
        """
        if not self.path_is_mallet_or_bell(folder_path):
            return current_score, {"mallet_bell_physics_check": "not_mallet_bell_candidate"}
        evidence = self.mallet_bell_eligibility_evidence(facts)
        float(evidence["mallet_bell_eligibility"])
        weak = bool(evidence["weak_mallet_bell_eligibility"])
        # v31.103: keep the Bell/Mallet struck-resonance measurements as
        # diagnostics only.  The v31.102 hard penalty moved too much weight out
        # of the physics voter and caused unrelated category regressions.
        # Final routing can still read these evidence fields later, but this
        # voter no longer changes the candidate score here.
        return current_score, {
            **evidence,
            "mallet_bell_physics_check": ("diagnostic_only_weak" if weak else "diagnostic_only_eligible"),
            "mallet_bell_physics_penalty": 0.0,
            "mallet_bell_score_before_penalty": round(float(current_score), 6),
            "mallet_bell_score_after_penalty": round(float(current_score), 6),
        }

    @classmethod
    def path_is_mallet_or_bell(cls, folder_path: str) -> bool:
        """Return True for Mallet/Bell candidate folders."""
        normalized = str(folder_path or "").replace("\\", "/").lower()
        return any(
            token in normalized for token in ("mallet", "bell", "vibraphone", "marimba", "glockenspiel", "celesta")
        )

    @classmethod
    def mallet_bell_eligibility_evidence(cls, facts: SharedAudioFacts) -> dict[str, Any]:
        """Return measured struck-resonance evidence for Mallet/Bell candidates."""
        feature_values = facts.feature_values_by_name or {}
        direct_view = facts.evidence.get("direct_body_view", {}) if isinstance(facts.evidence, dict) else {}
        direct_values = direct_view.get("feature_values_by_name", {}) if isinstance(direct_view, dict) else {}

        attack = cls.fact_value(feature_values, "attack_rise_time_norm", 1.0)
        temporal = cls.fact_value(feature_values, "temporal_centroid_ratio", 0.5)
        tail = cls.fact_value(feature_values, "tail_energy_ratio", 0.0)
        event_count = cls.expm1_value(cls.fact_value(feature_values, "log_transient_count", 0.0))
        high_event = cls.fact_value(feature_values, "loop_mean_event_high_ratio", 0.0)
        percussive_event = cls.fact_value(feature_values, "loop_percussive_event_ratio", 0.0)
        drumlike = cls.fact_value(feature_values, "loop_drumlike_frame_ratio", 0.0)
        sustained_tonal = cls.fact_value(feature_values, "loop_sustained_tonal_frame_ratio", 0.0)
        tonal_balance = cls.fact_value(feature_values, "loop_tonal_to_percussive_balance", 0.0)
        pitched_event = cls.fact_value(feature_values, "loop_pitched_event_ratio", 0.0)
        non_event_tonal = cls.fact_value(feature_values, "loop_non_event_tonal_ratio", 0.0)
        f0_voiced = cls.fact_value(feature_values, "f0_voiced_ratio", 0.0)
        inharmonicity = cls.fact_value(feature_values, "inharmonicity", 0.0)
        harmonic_energy = cls.fact_value(feature_values, "harmonic_energy_ratio", 0.0)
        attack_high = cls.fact_value(feature_values, "attack_high_ratio", 0.0)
        body_high = cls.fact_value(feature_values, "body_high_ratio", 0.0)
        tail_high = cls.fact_value(feature_values, "tail_high_ratio", 0.0)
        body_flatness = cls.fact_value(feature_values, "body_flatness", 0.0)
        direct_attack = cls.fact_value(direct_values, "attack_rise_time_norm", attack)
        direct_high_event = cls.fact_value(direct_values, "loop_mean_event_high_ratio", high_event)
        direct_percussive = cls.fact_value(direct_values, "loop_percussive_event_ratio", percussive_event)

        fast_attack = cls.inverse_ramp(min(attack, direct_attack), 0.045, 0.24)
        cls.inverse_ramp(temporal, 0.20, 0.52)
        event_presence = cls.ramp(event_count, 1.0, 6.0)
        bright_attack = max(
            cls.ramp(attack_high, 0.10, 0.38),
            cls.ramp(high_event, 0.08, 0.32),
            cls.ramp(direct_high_event, 0.08, 0.32),
        )
        percussive_body = max(
            cls.ramp(percussive_event, 0.04, 0.22),
            cls.ramp(direct_percussive, 0.04, 0.22),
            cls.ramp(drumlike, 0.08, 0.32),
        )
        struck_onset_score = max(
            0.70 * fast_attack + 0.30 * bright_attack,
            0.55 * fast_attack + 0.25 * event_presence + 0.20 * percussive_body,
        )
        struck_onset_score = min(1.0, max(0.0, struck_onset_score))

        resonant_tail_score = min(
            1.0,
            max(
                0.0,
                (
                    0.35 * cls.ramp(tail, 0.18, 0.70)
                    + 0.25 * cls.ramp(harmonic_energy, 0.20, 0.55)
                    + 0.20 * cls.ramp(max(body_high, tail_high), 0.05, 0.30)
                    + 0.20 * cls.inverse_ramp(body_flatness, 0.10, 0.42)
                ),
            ),
        )
        inharmonic_ring_score = min(
            1.0, max(0.0, (0.65 * cls.ramp(inharmonicity, 0.12, 0.36) + 0.35 * cls.ramp(harmonic_energy, 0.20, 0.55)))
        )
        wet_phrase_like = bool(
            sustained_tonal >= 0.72
            and pitched_event >= 0.78
            and f0_voiced >= 0.72
            and percussive_event <= 0.08
            and high_event <= 0.18
            and tonal_balance >= 0.52
        )
        sustained_non_struck_loop = bool(
            non_event_tonal >= 0.70 and struck_onset_score < 0.52 and percussive_event <= 0.08 and drumlike <= 0.16
        )
        wet_phrase_penalty = 0.28 if wet_phrase_like else 0.0
        sustained_non_struck_penalty = 0.18 if sustained_non_struck_loop else 0.0
        mallet_bell_eligibility = min(
            1.0,
            max(
                0.0,
                (
                    0.48 * struck_onset_score
                    + 0.30 * resonant_tail_score
                    + 0.22 * inharmonic_ring_score
                    - wet_phrase_penalty
                    - sustained_non_struck_penalty
                ),
            ),
        )
        weak = bool(
            mallet_bell_eligibility < 0.46
            or (struck_onset_score < 0.42 and (wet_phrase_like or sustained_non_struck_loop))
        )
        return {
            "mallet_bell_eligibility": round(float(mallet_bell_eligibility), 6),
            "weak_mallet_bell_eligibility": bool(weak),
            "struck_onset_score": round(float(struck_onset_score), 6),
            "resonant_tail_score": round(float(resonant_tail_score), 6),
            "inharmonic_ring_score": round(float(inharmonic_ring_score), 6),
            "wet_phrase_like": bool(wet_phrase_like),
            "sustained_non_struck_loop": bool(sustained_non_struck_loop),
            "mallet_attack_rise_time_norm": round(float(attack), 6),
            "mallet_event_count_estimate": round(float(event_count), 6),
            "mallet_high_event_ratio": round(float(high_event), 6),
            "mallet_percussive_event_ratio": round(float(percussive_event), 6),
            "mallet_sustained_tonal_frame_ratio": round(float(sustained_tonal), 6),
            "mallet_inharmonicity": round(float(inharmonicity), 6),
        }

    @staticmethod
    def fact_value(values: dict[str, Any], name: str, default: float = 0.0) -> float:
        """Return a finite feature value from a dictionary."""
        import math

        try:
            number = float(values.get(name, default) or default)
            if math.isfinite(number):
                return number
        except Exception:
            pass
        return float(default)

    @staticmethod
    def expm1_value(value: float) -> float:
        """Safe expm1 for log-count features."""
        import math

        try:
            return float(math.expm1(max(0.0, float(value))))
        except Exception:
            return 0.0

    @staticmethod
    def ramp(number: float, start: float, full: float) -> float:
        """Linear 0..1 ramp."""
        if full <= start:
            return 0.0
        return min(1.0, max(0.0, (float(number) - float(start)) / (float(full) - float(start))))

    @staticmethod
    def inverse_ramp(number: float, good_at_or_below: float, bad_at_or_above: float) -> float:
        """Inverse linear 0..1 ramp."""
        if bad_at_or_above <= good_at_or_below:
            return 0.0
        return min(
            1.0,
            max(
                0.0,
                1.0 - ((float(number) - float(good_at_or_below)) / (float(bad_at_or_above) - float(good_at_or_below))),
            ),
        )

    def extract_first_arrival_telemetry(
        self,
        physics: AudioPhysics,
        facts: SharedAudioFacts,
    ) -> dict[str, Any]:
        """Measure reusable first-arrival source evidence before reverb dominates.

        This hook is intentionally source-name blind. It reads the audio samples
        from ``AudioPhysics.source_path`` only as signal data, finds onset-centered
        first-arrival windows, and reports anonymous temporal/cepstral metrics.
        Sax/reed is the first consumer, but the same payload can later support
        plucked, brass, and struck-resonator witnesses.
        """
        cached = facts.evidence.get("first_arrival_telemetry") if isinstance(facts.evidence, dict) else None
        if isinstance(cached, dict):
            return cached
        try:
            y, sr0 = read_audio(physics.source_path)
            y, sr = resample_linear(y, sr0, TARGET_SR)
            y, status = trim_and_normalize(y)
        except Exception as exc:
            return {"status": "audio_read_error", "error": str(exc)[:120]}
        if status != "ok" or y.size == 0:
            return {"status": status or "empty"}
        samples = np.asarray(y, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        mono = np.mean(samples, axis=1).astype(np.float32)
        if mono.size < max(128, int(0.08 * sr)) or float(np.max(np.abs(mono))) <= 1e-9:
            return {"status": "too_short_or_silent"}
        mono = mono - float(np.mean(mono))
        peak = float(np.max(np.abs(mono)))
        if peak > 1e-9:
            mono = mono / peak
        frames = frame_audio(mono)
        onset_frames = clean_onset_peaks(frames, sr)
        if not onset_frames:
            env = np.sqrt(np.mean(frames**2, axis=1)) if frames.size else np.asarray([], dtype=np.float32)
            if env.size:
                onset_frames = [int(np.argmax(env))]
        if not onset_frames:
            return {"status": "no_onsets"}

        # Keep this deterministic and bounded.  Use the strongest few events so
        # loops cannot spend all work on silence, but still include early notes.
        env = np.sqrt(np.mean(frames**2, axis=1)) if frames.size else np.asarray([], dtype=np.float32)
        ranked = []
        for frame_idx in onset_frames:
            amp = float(env[min(max(0, int(frame_idx)), max(0, env.size - 1))]) if env.size else 0.0
            ranked.append((amp, int(frame_idx)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = [frame for _, frame in ranked[: min(10, len(ranked))]]
        selected.sort()

        f0_hz = self.fact_value(facts.feature_values_by_name or {}, "f0_median_hz", 0.0)
        pitch_conf = self.fact_value(facts.feature_values_by_name or {}, "pitch_confidence", 0.0)
        if f0_hz <= 35.0 or f0_hz >= 1800.0 or pitch_conf < 0.15:
            f0_hz = self.estimate_f0_from_audio(mono, sr)

        pre_ms = 35.0
        first_ms = 125.0
        tail_start_ms = 150.0
        tail_end_ms = 520.0
        pre_metrics: list[dict[str, float]] = []
        first_metrics: list[dict[str, float]] = []
        strike_metrics: list[dict[str, float]] = []
        settle_metrics: list[dict[str, float]] = []
        tail_metrics: list[dict[str, float]] = []
        cepstral_values: list[float] = []
        strike_inharmonic_values: list[float] = []
        settle_inharmonic_values: list[float] = []
        strike_flatness_values: list[float] = []
        settle_flatness_values: list[float] = []
        strike_high_values: list[float] = []
        settle_high_values: list[float] = []
        strike_entropy_values: list[float] = []
        settle_entropy_values: list[float] = []
        spectral_darkening_values: list[float] = []
        harmonic_ratios: list[float] = []
        bite_values: list[float] = []
        pre_diffuse_values: list[float] = []
        tail_diffuse_values: list[float] = []
        stochastic_values: list[float] = []
        formant_centers: list[float] = []
        event_count = 0
        # Parallelization note: selected first-arrival events are independent, but
        # this release keeps them synchronous so decoder and evidence order stay deterministic.
        for frame_idx in selected:
            center = int(frame_idx * HOP)
            pre = self.slice_mono(mono, sr, center, -pre_ms, 0.0)
            first = self.slice_mono(mono, sr, center, 0.0, first_ms)
            strike = self.slice_mono(mono, sr, center, 0.0, 25.0)
            settle = self.slice_mono(mono, sr, center, 25.0, 120.0)
            tail = self.slice_mono(mono, sr, center, tail_start_ms, tail_end_ms)
            if first.size < 96:
                continue
            event_count += 1
            pre_desc = self.segment_physics(pre, sr, f0_hz)
            first_desc = self.segment_physics(first, sr, f0_hz)
            strike_desc = self.segment_physics(strike, sr, f0_hz)
            settle_desc = self.segment_physics(settle, sr, f0_hz)
            tail_desc = self.segment_physics(tail, sr, f0_hz)
            pre_metrics.append(pre_desc)
            first_metrics.append(first_desc)
            strike_metrics.append(strike_desc)
            settle_metrics.append(settle_desc)
            tail_metrics.append(tail_desc)
            if strike_desc["inharmonic_energy_ratio"] > 0.0:
                strike_inharmonic_values.append(strike_desc["inharmonic_energy_ratio"])
            if settle_desc["inharmonic_energy_ratio"] > 0.0:
                settle_inharmonic_values.append(settle_desc["inharmonic_energy_ratio"])
            if strike_desc["flatness"] > 0.0:
                strike_flatness_values.append(strike_desc["flatness"])
            if settle_desc["flatness"] > 0.0:
                settle_flatness_values.append(settle_desc["flatness"])
            if strike_desc["high_ratio"] > 0.0:
                strike_high_values.append(strike_desc["high_ratio"])
            if settle_desc["high_ratio"] > 0.0:
                settle_high_values.append(settle_desc["high_ratio"])
            if strike_desc["entropy"] > 0.0:
                strike_entropy_values.append(strike_desc["entropy"])
            if settle_desc["entropy"] > 0.0:
                settle_entropy_values.append(settle_desc["entropy"])
            if strike_desc["high_ratio"] > 0.0 and settle_desc["high_ratio"] > 0.0:
                spectral_darkening_values.append(
                    10.0 * math.log10((strike_desc["high_ratio"] + 1e-9) / (settle_desc["high_ratio"] + 1e-9))
                )
            if first_desc["cepstral_peak_coherence"] > 0.0:
                cepstral_values.append(first_desc["cepstral_peak_coherence"])
            if first_desc["conical_even_odd_fit"] > 0.0:
                harmonic_ratios.append(first_desc["conical_even_odd_fit"])
            if first_desc["presence_contrast_db"] > 0.0:
                bite_values.append(first_desc["presence_contrast_db"])
            if pre_desc["rms"] > 0.0:
                pre_diffuse = min(1.0, pre_desc["rms"] / (first_desc["rms"] + 1e-9)) * max(
                    pre_desc["flatness"], pre_desc["high_ratio"]
                )
                pre_diffuse_values.append(float(pre_diffuse))
            if tail_desc["rms"] > 0.0:
                tail_diffuse = max(0.0, tail_desc["flatness"] - first_desc["flatness"]) + max(
                    0.0, tail_desc["entropy"] - first_desc["entropy"]
                )
                tail_diffuse_values.append(float(min(1.0, tail_diffuse)))
            if first_desc["stochastic_to_harmonic_coherence"] > 0.0:
                stochastic_values.append(first_desc["stochastic_to_harmonic_coherence"])
            if first_desc["presence_peak_frequency_hz"] > 0.0:
                formant_centers.append(first_desc["presence_peak_frequency_hz"])

        if event_count <= 0:
            return {"status": "no_valid_first_arrival_windows"}
        cep = self.safe_mean(cepstral_values)
        conical = self.safe_mean(harmonic_ratios)
        bite = self.safe_mean(bite_values)
        pre_diffuse = self.safe_mean(pre_diffuse_values)
        tail_diffuse = self.safe_mean(tail_diffuse_values)
        stochastic = self.safe_mean(stochastic_values)
        formant_mean = self.safe_mean(formant_centers)
        formant_std = self.safe_std(formant_centers)
        strike_inharmonic = self.safe_mean(strike_inharmonic_values)
        settle_inharmonic = self.safe_mean(settle_inharmonic_values)
        strike_flatness = self.safe_mean(strike_flatness_values)
        settle_flatness = self.safe_mean(settle_flatness_values)
        strike_high = self.safe_mean(strike_high_values)
        settle_high = self.safe_mean(settle_high_values)
        strike_entropy = self.safe_mean(strike_entropy_values)
        settle_entropy = self.safe_mean(settle_entropy_values)
        spectral_darkening = self.safe_mean(spectral_darkening_values)
        strike_to_settle_flatness_drop = max(0.0, strike_flatness - settle_flatness)
        strike_to_settle_inharmonic_drop = max(0.0, strike_inharmonic - settle_inharmonic)
        return {
            "status": "ok",
            "selected_event_count": int(event_count),
            "f0_hz_used": round(float(f0_hz), 3),
            "cepstral_pitch_period_coherence": round(float(cep), 6),
            "first_arrival_conical_balance": round(float(conical), 6),
            "first_arrival_presence_contrast_db": round(float(bite), 6),
            "pre_onset_diffuse_ratio": round(float(pre_diffuse), 6),
            "is_convoluted_transient": bool(pre_diffuse >= 0.13),
            "tail_diffusion_ratio": round(float(tail_diffuse), 6),
            "stochastic_modulation_coherence": round(float(stochastic), 6),
            "formant_center_mean_hz": round(float(formant_mean), 3),
            "formant_center_std_hz": round(float(formant_std), 3),
            "formant_stability_score": round(float(self.inverse_ramp(formant_std, 40.0, 700.0)), 6),
            "strike_inharmonic_energy_ratio": round(float(strike_inharmonic), 6),
            "settle_inharmonic_energy_ratio": round(float(settle_inharmonic), 6),
            "strike_to_settle_inharmonic_drop": round(float(strike_to_settle_inharmonic_drop), 6),
            "strike_flatness": round(float(strike_flatness), 6),
            "settle_flatness": round(float(settle_flatness), 6),
            "strike_to_settle_flatness_drop": round(float(strike_to_settle_flatness_drop), 6),
            "strike_high_ratio": round(float(strike_high), 6),
            "settle_high_ratio": round(float(settle_high), 6),
            "spectral_darkening_db": round(float(spectral_darkening), 6),
            "strike_entropy": round(float(strike_entropy), 6),
            "settle_entropy": round(float(settle_entropy), 6),
        }

    @staticmethod
    def slice_mono(mono: np.ndarray, sr: int, center: int, start_ms: float, stop_ms: float) -> np.ndarray:
        """Slice a mono window around an onset center in milliseconds."""
        start = max(0, int(center + (start_ms / 1000.0) * sr))
        stop = min(int(mono.size), int(center + (stop_ms / 1000.0) * sr))
        if stop <= start:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(mono[start:stop], dtype=np.float32)

    @classmethod
    def segment_physics(cls, segment: np.ndarray, sr: int, f0_hz: float) -> dict[str, float]:
        """Small first-arrival segment descriptor used by family witnesses."""
        x = np.asarray(segment, dtype=np.float32).reshape(-1)
        if x.size < 64 or float(np.max(np.abs(x))) <= 1e-9:
            return {
                "rms": 0.0,
                "flatness": 0.0,
                "entropy": 0.0,
                "high_ratio": 0.0,
                "conical_even_odd_fit": 0.0,
                "cepstral_peak_coherence": 0.0,
                "presence_contrast_db": 0.0,
                "presence_peak_frequency_hz": 0.0,
                "stochastic_to_harmonic_coherence": 0.0,
                "harmonic_energy_ratio": 0.0,
                "inharmonic_energy_ratio": 0.0,
            }
        x = x - float(np.mean(x))
        rms = float(np.sqrt(np.mean(x**2)))
        n_fft = 1
        while n_fft < max(512, min(4096, int(x.size))):
            n_fft *= 2
        y = np.pad(x, (0, n_fft - x.size)) if x.size < n_fft else x[:n_fft]
        win = np.hanning(n_fft).astype(np.float32)
        spec = np.abs(np.fft.rfft(y * win)) + 1e-12
        power = spec**2
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        total = float(np.sum(power)) + 1e-12
        flatness = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-12))
        probs = power / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)) / max(1e-9, math.log2(max(2, probs.size))))
        high_ratio = cls.band_energy_ratio(power, freqs, 4000.0, sr / 2.0)
        conical_fit = cls.conical_even_odd_fit(power, freqs, f0_hz)
        harmonic_support = cls.harmonic_support_ratio(power, freqs, f0_hz)
        inharmonic_ratio = max(0.0, 1.0 - harmonic_support) if harmonic_support > 0.0 else 0.0
        cep_coh = cls.cepstral_period_coherence(spec, sr, f0_hz)
        presence_contrast, presence_peak = cls.presence_band_contrast(power, freqs)
        stochastic = cls.stochastic_to_harmonic_coherence(x, sr, f0_hz)
        return {
            "rms": rms,
            "flatness": flatness,
            "entropy": entropy,
            "high_ratio": high_ratio,
            "conical_even_odd_fit": conical_fit,
            "cepstral_peak_coherence": cep_coh,
            "presence_contrast_db": presence_contrast,
            "presence_peak_frequency_hz": presence_peak,
            "stochastic_to_harmonic_coherence": stochastic,
            "harmonic_energy_ratio": harmonic_support,
            "inharmonic_energy_ratio": inharmonic_ratio,
        }

    @staticmethod
    def safe_mean(values: list[float]) -> float:
        finite = [float(v) for v in values if np.isfinite(float(v))]
        return float(np.mean(finite)) if finite else 0.0

    @staticmethod
    def safe_std(values: list[float]) -> float:
        finite = [float(v) for v in values if np.isfinite(float(v))]
        return float(np.std(finite)) if finite else 0.0

    @staticmethod
    def band_energy_ratio(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
        mask = (freqs >= float(lo)) & (freqs < float(hi))
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask]) / (np.sum(power) + 1e-12))

    @classmethod
    def harmonic_band_energy(cls, power: np.ndarray, freqs: np.ndarray, target_hz: float) -> float:
        if target_hz <= 0.0:
            return 0.0
        width = max(18.0, target_hz * 0.045)
        mask = (freqs >= target_hz - width) & (freqs <= target_hz + width)
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask]))

    @classmethod
    def harmonic_support_ratio(cls, power: np.ndarray, freqs: np.ndarray, f0_hz: float) -> float:
        """Energy near harmonic bins as a fraction of total local spectrum."""
        if f0_hz <= 35.0 or f0_hz >= 1800.0:
            return 0.0
        nyquist = float(freqs[-1]) if freqs.size else 0.0
        total = float(np.sum(power)) + 1e-12
        support = 0.0
        count = 0
        for harmonic in range(1, 10):
            target = f0_hz * harmonic
            if target >= nyquist:
                break
            support += cls.harmonic_band_energy(power, freqs, target)
            count += 1
        if count < 3:
            return 0.0
        return float(max(0.0, min(1.0, support / total)))

    @classmethod
    def conical_even_odd_fit(cls, power: np.ndarray, freqs: np.ndarray, f0_hz: float) -> float:
        """Balanced even/odd harmonic support from a first-arrival window."""
        if f0_hz <= 35.0 or f0_hz >= 1800.0:
            return 0.0
        nyquist = float(freqs[-1]) if freqs.size else 0.0
        harmonics = {n: cls.harmonic_band_energy(power, freqs, f0_hz * n) for n in range(1, 7) if f0_hz * n < nyquist}
        if len(harmonics) < 4:
            return 0.0
        even_sum = harmonics.get(2, 0.0) + harmonics.get(4, 0.0) + 0.35 * harmonics.get(6, 0.0)
        odd_sum = harmonics.get(3, 0.0) + harmonics.get(5, 0.0) + 1e-12
        upper_sum = even_sum + odd_sum
        total = sum(harmonics.values()) + 1e-12
        ratio = even_sum / odd_sum
        balance = 1.0 - min(1.0, abs(math.log(max(ratio, 1e-6))) / math.log(3.2))
        support = min(1.0, upper_sum / total * 2.4)
        return float(max(0.0, min(1.0, balance * support)))

    @staticmethod
    def cepstral_period_coherence(spec: np.ndarray, sr: int, f0_hz: float) -> float:
        """High-quefrency pitch-period peak after low-quefrency liftering."""
        if f0_hz <= 35.0 or f0_hz >= 1800.0 or spec.size < 16:
            return 0.0
        log_spec = np.log(np.asarray(spec, dtype=np.float64) + 1e-12)
        cep = np.fft.irfft(log_spec)
        if cep.size < 8:
            return 0.0
        # High-pass lifter: ignore the smooth spectral envelope / room coloration.
        min_q = max(1, int(0.0007 * sr))
        max_q = min(int(0.026 * sr), cep.size - 1)
        target = int(round(sr / max(f0_hz, 1.0)))
        lo = max(min_q, target - max(2, int(0.13 * target)))
        hi = min(max_q, target + max(2, int(0.13 * target)))
        if hi <= lo:
            return 0.0
        high = np.abs(cep[min_q:max_q])
        target_band = np.abs(cep[lo:hi])
        if target_band.size == 0 or high.size == 0:
            return 0.0
        peak = float(np.max(target_band))
        floor = float(np.median(high)) + 1e-12
        return float(min(1.0, max(0.0, (peak / floor - 1.0) / 8.0)))

    @staticmethod
    def presence_band_contrast(power: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
        mask = (freqs >= 900.0) & (freqs <= 3400.0)
        if not np.any(mask):
            return 0.0, 0.0
        vals = np.asarray(power[mask], dtype=np.float64) + 1e-18
        db = 10.0 * np.log10(vals)
        contrast = float(np.percentile(db, 95) - np.percentile(db, 12))
        sub_freqs = freqs[mask]
        peak_freq = float(sub_freqs[int(np.argmax(vals))]) if sub_freqs.size else 0.0
        return max(0.0, min(40.0, contrast)), peak_freq

    @staticmethod
    def stochastic_to_harmonic_coherence(segment: np.ndarray, sr: int, f0_hz: float) -> float:
        """Approximate airflow modulation coupling in a tiny local window."""
        x = np.asarray(segment, dtype=np.float32).reshape(-1)
        if x.size < int(0.09 * sr) or f0_hz <= 35.0:
            return 0.0
        win = max(128, int(0.020 * sr))
        hop = max(32, int(0.006 * sr))
        hi_env: list[float] = []
        harm_env: list[float] = []
        for start in range(0, max(1, x.size - win + 1), hop):
            seg = x[start : start + win]
            if seg.size < 64:
                continue
            n_fft = 1
            while n_fft < max(256, seg.size):
                n_fft *= 2
            y = np.pad(seg, (0, max(0, n_fft - seg.size)))[:n_fft]
            spec = np.abs(np.fft.rfft(y * np.hanning(n_fft))) ** 2 + 1e-12
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            hi_env.append(float(np.sum(spec[freqs >= 8000.0])))
            h2 = PhysicsVoter.harmonic_band_energy(spec, freqs, f0_hz * 2.0)
            h3 = PhysicsVoter.harmonic_band_energy(spec, freqs, f0_hz * 3.0)
            harm_env.append(float(h2 + h3))
            if len(hi_env) >= 24:
                break
        if len(hi_env) < 6 or len(harm_env) < 6:
            return 0.0
        a = np.asarray(hi_env, dtype=np.float64)
        b = np.asarray(harm_env, dtype=np.float64)
        if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
            return 0.0
        corr = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return float(max(0.0, min(1.0, (corr + 1.0) * 0.5)))

    @staticmethod
    def estimate_f0_from_audio(mono: np.ndarray, sr: int) -> float:
        """Small autocorrelation pitch fallback for first-arrival telemetry."""
        x = np.asarray(mono, dtype=np.float32).reshape(-1)
        if x.size < int(0.08 * sr):
            return 0.0
        max_len = min(x.size, int(1.2 * sr))
        x = x[:max_len]
        x = x - float(np.mean(x))
        if float(np.sqrt(np.mean(x**2))) <= 1e-6:
            return 0.0
        n = 1
        while n < x.size * 2:
            n *= 2
        sp = np.fft.rfft(x, n=n)
        corr = np.fft.irfft(sp * np.conj(sp), n=n)[: x.size]
        corr0 = float(corr[0]) + 1e-12
        min_lag = max(1, int(sr / 1500.0))
        max_lag = min(int(sr / 35.0), corr.size - 1)
        if max_lag <= min_lag:
            return 0.0
        corr[:min_lag] = 0.0
        lag = int(np.argmax(corr[min_lag:max_lag])) + min_lag
        strength = float(corr[lag] / corr0) if lag > 0 else 0.0
        if strength < 0.18:
            return 0.0
        return float(sr / lag) if lag > 0 else 0.0

    def apply_piano_identity_adjustment(
        self,
        *,
        label: str,
        folder_path: str,
        facts: SharedAudioFacts,
        current_score: float,
        first_arrival_telemetry: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Lift measured acoustic-piano / struck-resonator identity in PhysicsVoter.

        This is the piano analogue of the sax first-arrival witness. It does not
        route by names. It only lets Keys/Rhodes/Piano candidates compete when
        the onset-local physics show a hammer-like inharmonic strike followed by
        rapid tonal settling and spectral darkening.
        """
        if not self.path_is_piano_or_keys_candidate(folder_path):
            return current_score, {}
        evidence = self.piano_struck_identity_evidence(facts, first_arrival_telemetry or {})
        identity = float(evidence["piano_struck_identity_score"])
        eligible = bool(evidence["piano_struck_identity_eligible"])
        if not eligible:
            return current_score, {
                **evidence,
                "piano_struck_physics_check": "diagnostic_only_not_eligible",
                "piano_struck_score_before_adjustment": round(float(current_score), 6),
                "piano_struck_score_after_adjustment": round(float(current_score), 6),
            }
        # Candidate score target: enough to beat broad synth/FX confusion, but
        # not enough to steal non-key sounds without the measured struck evidence.
        target_score = max(0.16, 0.80 - 0.82 * identity)
        adjusted = min(float(current_score), float(target_score))
        return adjusted, {
            **evidence,
            "piano_struck_physics_check": "first_arrival_struck_resonator_identity_lift",
            "piano_struck_score_before_adjustment": round(float(current_score), 6),
            "piano_struck_score_after_adjustment": round(float(adjusted), 6),
            "piano_struck_identity_target_score": round(float(target_score), 6),
        }

    @classmethod
    def path_is_piano_or_keys_candidate(cls, folder_path: str) -> bool:
        """Return True for internal Keys/Piano/Rhodes candidates."""
        normalized = str(folder_path or "").replace("\\", "/").lower()
        return (
            any(token in normalized for token in ("/keys/", "piano", "rhodes", "electric piano"))
            and "coins" not in normalized
        )

    @classmethod
    def piano_struck_identity_evidence(
        cls,
        facts: SharedAudioFacts,
        first_arrival_telemetry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return source-name-blind piano/struck-string identity evidence."""
        values = facts.feature_values_by_name or {}
        first_arrival_telemetry = first_arrival_telemetry or {}
        pitch_conf = cls.fact_value(values, "pitch_confidence", 0.0)
        f0_voiced = cls.fact_value(values, "f0_voiced_ratio", 0.0)
        pitched_event = cls.fact_value(values, "loop_pitched_event_ratio", 0.0)
        sustained_tonal = cls.fact_value(values, "loop_sustained_tonal_frame_ratio", 0.0)
        percussive_event = cls.fact_value(values, "loop_percussive_event_ratio", 0.0)
        drumlike = cls.fact_value(values, "loop_drumlike_frame_ratio", 0.0)
        high_event = cls.fact_value(values, "loop_mean_event_high_ratio", 0.0)
        mid_event = cls.fact_value(
            values, "loop_mean_event_mid_ratio", cls.fact_value(values, "mid_ratio_500_2000hz", 0.0)
        )
        low_event = cls.fact_value(
            values, "loop_mean_event_low_ratio", cls.fact_value(values, "bass_ratio_150_500hz", 0.0)
        )
        flatness = cls.fact_value(values, "spectral_flatness_mean", 0.0)
        entropy = cls.fact_value(values, "spectral_entropy_mean", 0.0)
        harmonic_energy = cls.fact_value(values, "harmonic_energy_ratio", 0.0)
        fundamental_dominance = cls.fact_value(values, "fundamental_dominance_ratio", 0.0)
        onset_count = cls.expm1_value(cls.fact_value(values, "log_transient_count", 0.0))
        shape_blob = facts.evidence.get("shape_vote", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
        shape_name = str(shape_blob.get("primary_shape", "")) if isinstance(shape_blob, dict) else ""
        voice_identity = cls.fact_value(values, "formant_light_voice_identity", 0.0)
        vocal_role = max(
            cls.fact_value_from_evidence(facts, "vocal_music_phrase", 0.0),
            cls.fact_value_from_evidence(facts, "vocal_phrase", 0.0),
            cls.fact_value_from_evidence(facts, "vocal_one_shot", 0.0),
        )
        voiced_role = cls.fact_value_from_evidence(facts, "voiced_one_shot", 0.0)
        formant_like = cls.fact_value(values, "formant_like_peak_spacing", 0.0)
        voiced_role_is_vocal = bool(
            shape_name == "vocal_phrase" or voice_identity >= 0.55 or formant_like >= 1.05 or vocal_role >= 0.28
        )
        voice_role = max(vocal_role, voiced_role if voiced_role_is_vocal else 0.0)

        fa_status = str(first_arrival_telemetry.get("status", "missing"))
        strike_inharmonic = float(first_arrival_telemetry.get("strike_inharmonic_energy_ratio", 0.0) or 0.0)
        settle_inharmonic = float(first_arrival_telemetry.get("settle_inharmonic_energy_ratio", 0.0) or 0.0)
        inharmonic_drop = float(first_arrival_telemetry.get("strike_to_settle_inharmonic_drop", 0.0) or 0.0)
        strike_flatness = float(first_arrival_telemetry.get("strike_flatness", 0.0) or 0.0)
        settle_flatness = float(first_arrival_telemetry.get("settle_flatness", 0.0) or 0.0)
        flatness_drop = float(first_arrival_telemetry.get("strike_to_settle_flatness_drop", 0.0) or 0.0)
        strike_high = float(first_arrival_telemetry.get("strike_high_ratio", 0.0) or 0.0)
        settle_high = float(first_arrival_telemetry.get("settle_high_ratio", 0.0) or 0.0)
        spectral_darkening = float(first_arrival_telemetry.get("spectral_darkening_db", 0.0) or 0.0)
        fa_cepstral = float(first_arrival_telemetry.get("cepstral_pitch_period_coherence", 0.0) or 0.0)
        event_count = float(first_arrival_telemetry.get("selected_event_count", 0.0) or 0.0)

        pitch_score = min(
            1.0,
            max(
                cls.ramp(pitch_conf, 0.34, 0.82),
                0.55 * cls.ramp(f0_voiced, 0.55, 0.94) + 0.45 * cls.ramp(pitched_event, 0.70, 0.96),
            ),
        )
        tonal_loop_score = min(
            1.0,
            max(
                0.0,
                0.35 * cls.ramp(sustained_tonal, 0.55, 0.95)
                + 0.30 * cls.ramp(pitched_event, 0.65, 0.96)
                + 0.20 * cls.ramp(onset_count, 4.0, 24.0)
                + 0.15 * cls.inverse_ramp(percussive_event, 0.04, 0.26),
            ),
        )
        mid_body_score = min(
            1.0,
            max(
                0.0,
                0.60 * cls.ramp(mid_event, 0.45, 0.86)
                + 0.25 * cls.inverse_ramp(abs(low_event - 0.12), 0.02, 0.35)
                + 0.15 * cls.inverse_ramp(high_event, 0.018, 0.12),
            ),
        )
        hammer_shock_score = min(
            1.0,
            max(
                0.0,
                0.34 * cls.ramp(strike_inharmonic, 0.18, 0.72)
                + 0.24 * cls.ramp(flatness_drop, 0.045, 0.22)
                + 0.20 * cls.ramp(inharmonic_drop, 0.04, 0.30)
                + 0.14 * cls.ramp(strike_flatness, 0.08, 0.42)
                + 0.08 * cls.ramp(event_count, 1.0, 6.0),
            ),
        )
        damping_score = min(
            1.0,
            max(
                0.0,
                0.50 * cls.ramp(spectral_darkening, 2.0, 15.0)
                + 0.25 * cls.ramp(max(0.0, strike_high - settle_high), 0.005, 0.10)
                + 0.25 * cls.ramp(max(0.0, strike_flatness - settle_flatness), 0.04, 0.24),
            ),
        )
        resonant_string_score = min(
            1.0,
            max(
                0.0,
                0.30 * cls.ramp(harmonic_energy, 0.22, 0.62)
                + 0.25 * cls.inverse_ramp(flatness, 0.018, 0.080)
                + 0.20 * cls.ramp(fa_cepstral, 0.18, 0.70)
                + 0.15 * cls.ramp(entropy, 0.24, 0.55)
                + 0.10 * cls.inverse_ramp(fundamental_dominance, 0.10, 0.42),
            ),
        )
        raw_score = min(
            1.0,
            max(
                0.0,
                0.25 * pitch_score
                + 0.20 * tonal_loop_score
                + 0.17 * mid_body_score
                + 0.18 * hammer_shock_score
                + 0.12 * damping_score
                + 0.08 * resonant_string_score,
            ),
        )

        synth_penalty = 0.0
        if flatness < 0.018 and strike_flatness < 0.075 and hammer_shock_score < 0.42:
            synth_penalty += 0.30
        if high_event <= 0.006 and spectral_darkening <= 0.5 and hammer_shock_score < 0.48:
            synth_penalty += 0.18
        drum_or_bell_penalty = 0.0
        if drumlike >= 0.25 or percussive_event >= 0.32:
            drum_or_bell_penalty += 0.34
        reed_penalty = 0.0
        # If the sax witness is strong, do not let a keys witness steal it.
        if cls.fact_value_from_evidence(facts, "reed_sax_identity_score", 0.0) >= 0.62:
            reed_penalty += 0.42
        voice_penalty = 0.0
        # A struck-keys witness is allowed to help piano/EP loops, but it must
        # not steal obvious vocal shots or vocal phrases.  This is measured
        # audio evidence only: shape/role/formant facts, not source names.
        if voice_identity >= 0.55 or voice_role >= 0.52:
            voice_penalty += 0.55
        if shape_name == "vocal_phrase" and voice_role >= 0.28 and onset_count <= 48.0:
            voice_penalty += 0.40
        if percussive_event >= 0.34 and voice_role >= 0.26:
            voice_penalty += 0.28
        penalty = min(0.92, synth_penalty + drum_or_bell_penalty + reed_penalty + voice_penalty)
        identity = min(1.0, max(0.0, raw_score - penalty))
        eligible = bool(
            fa_status == "ok"
            and identity >= 0.53
            and pitch_score >= 0.62
            and tonal_loop_score >= 0.55
            and mid_body_score >= 0.45
            and max(hammer_shock_score, damping_score) >= 0.38
            and drum_or_bell_penalty < 0.34
            and reed_penalty < 0.42
            and voice_penalty < 0.40
        )
        return {
            "piano_struck_identity_score": round(float(identity), 6),
            "piano_struck_raw_score": round(float(raw_score), 6),
            "piano_struck_identity_penalty": round(float(penalty), 6),
            "piano_struck_identity_eligible": bool(eligible),
            "piano_pitch_score": round(float(pitch_score), 6),
            "piano_tonal_loop_score": round(float(tonal_loop_score), 6),
            "piano_mid_body_score": round(float(mid_body_score), 6),
            "piano_hammer_shock_score": round(float(hammer_shock_score), 6),
            "piano_damping_score": round(float(damping_score), 6),
            "piano_resonant_string_score": round(float(resonant_string_score), 6),
            "piano_synth_penalty": round(float(synth_penalty), 6),
            "piano_drum_or_bell_penalty": round(float(drum_or_bell_penalty), 6),
            "piano_reed_penalty": round(float(reed_penalty), 6),
            "piano_voice_penalty": round(float(voice_penalty), 6),
            "piano_strike_inharmonic_ratio": round(float(strike_inharmonic), 6),
            "piano_settle_inharmonic_ratio": round(float(settle_inharmonic), 6),
            "piano_spectral_darkening_db": round(float(spectral_darkening), 6),
        }

    @staticmethod
    def fact_value_from_evidence(facts: SharedAudioFacts, key: str, default: float = 0.0) -> float:
        """Search voter evidence blobs for a previously emitted metric."""
        if not isinstance(getattr(facts, "evidence", None), dict):
            return float(default)

        def walk(obj: Any) -> float | None:
            if isinstance(obj, dict):
                if key in obj:
                    try:
                        number = float(obj.get(key, default) or default)
                        if math.isfinite(number):
                            return number
                    except Exception:
                        return None
                for value in obj.values():
                    found = walk(value)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = walk(item)
                    if found is not None:
                        return found
            return None

        found = walk(facts.evidence)
        return float(default) if found is None else float(found)

    def apply_reed_sax_identity_adjustment(
        self,
        *,
        label: str,
        folder_path: str,
        facts: SharedAudioFacts,
        current_score: float,
        first_arrival_telemetry: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Lift strong measured sax/reed identity inside the PhysicsVoter.

        The production brain currently has a Saxophone one-shot profile but no
        dedicated Saxophone loop profile.  A wet sax phrase can therefore look
        structurally mismatched even when its anonymous audio physics show a
        stable pitched reed body.  This adjustment is structure-neutral identity
        evidence: it can make the Saxophone candidate competitive, but only when
        broad measured facts support a real reed-like source and anti-steal
        guards reject clean synth, electric-piano, bell, and drum bodies.
        """
        if not self.path_is_sax_candidate(folder_path):
            return current_score, {}

        evidence = self.reed_sax_identity_evidence(facts, first_arrival_telemetry or {})
        identity = float(evidence["reed_sax_identity_score"])
        eligible = bool(evidence["reed_sax_identity_eligible"])
        if not eligible:
            return current_score, {
                **evidence,
                "reed_sax_physics_check": "diagnostic_only_not_eligible",
                "reed_sax_score_before_adjustment": round(float(current_score), 6),
                "reed_sax_score_after_adjustment": round(float(current_score), 6),
            }

        # Convert measured identity into a candidate score target.  This is not
        # a route decision.  It only lets the PhysicsVoter say "Saxophone is a
        # good source-identity match" while Shape/arbiter logic still decides
        # loop versus one-shot depth.
        if bool(evidence.get("reed_sax_first_arrival_eligible")):
            target_score = max(0.18, 0.72 - 0.84 * identity)
        else:
            target_score = max(0.38, 1.05 - 0.74 * identity)
        adjusted = min(float(current_score), float(target_score))
        return adjusted, {
            **evidence,
            "reed_sax_physics_check": "structure_neutral_identity_lift",
            "reed_sax_score_before_adjustment": round(float(current_score), 6),
            "reed_sax_score_after_adjustment": round(float(adjusted), 6),
            "reed_sax_identity_target_score": round(float(target_score), 6),
        }

    @classmethod
    def path_is_sax_candidate(cls, folder_path: str) -> bool:
        """Return True only for Saxophone candidate folders."""
        normalized = str(folder_path or "").replace("\\", "/").lower()
        return "sax" in normalized or "saxophone" in normalized

    @classmethod
    def reed_sax_identity_evidence(
        cls, facts: SharedAudioFacts, first_arrival_telemetry: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return source-name-blind sax/reed identity evidence from stored physics.

        This intentionally reuses the existing 112-feature physics map.  It does
        not inspect filenames, source paths, folder names, or producer tokens.
        """
        values = facts.feature_values_by_name or {}
        first_arrival_telemetry = first_arrival_telemetry or {}
        direct_view = facts.evidence.get("direct_body_view", {}) if isinstance(facts.evidence, dict) else {}
        direct_values = direct_view.get("feature_values_by_name", {}) if isinstance(direct_view, dict) else {}
        if not isinstance(direct_values, dict):
            direct_values = {}

        pitch_conf = cls.fact_value(values, "pitch_confidence", 0.0)
        voiced = cls.fact_value(values, "f0_voiced_ratio", 0.0)
        loop_pitched = cls.fact_value(values, "loop_pitched_event_ratio", 0.0)
        sustained_tonal = cls.fact_value(values, "loop_sustained_tonal_frame_ratio", 0.0)
        percussive_loop = cls.fact_value(values, "loop_percussive_event_ratio", 0.0)
        harmonic_energy = cls.fact_value(values, "harmonic_energy_ratio", 0.0)
        inharmonicity = cls.fact_value(values, "inharmonicity", 0.0)
        fundamental_dominance = cls.fact_value(values, "fundamental_dominance_ratio", 0.0)
        body_flatness = cls.fact_value(values, "body_flatness", cls.fact_value(values, "spectral_flatness_mean", 0.0))
        tail_flatness = cls.fact_value(values, "tail_flatness", body_flatness)
        spectral_flatness = cls.fact_value(values, "spectral_flatness_mean", body_flatness)
        body_noise = cls.fact_value(values, "body_noise_ratio", body_flatness)
        tail_noise = cls.fact_value(values, "tail_noise_ratio", tail_flatness)
        body_entropy = cls.fact_value(values, "body_entropy", 0.0)
        tail_entropy = cls.fact_value(values, "tail_entropy", 0.0)
        bass_ratio = cls.fact_value(values, "bass_ratio_150_500hz", 0.0)
        mid_ratio = cls.fact_value(values, "mid_ratio_500_2000hz", 0.0)
        presence_ratio = cls.fact_value(values, "presence_ratio_2000_8000hz", 0.0)
        air_ratio = cls.fact_value(values, "air_ratio_gt_8000hz", 0.0)
        tail_energy = cls.fact_value(values, "tail_energy_ratio", 0.0)
        peak_stability = cls.fact_value(values, "spectral_peak_stability", 0.0)
        formant_spacing = cls.fact_value(values, "formant_like_peak_spacing", 0.0)
        onset_count = cls.expm1_value(cls.fact_value(values, "log_transient_count", 0.0))
        shape_blob = facts.evidence.get("shape_vote", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
        shape_name = str(shape_blob.get("primary_shape", "")) if isinstance(shape_blob, dict) else ""
        true_repetition = (
            cls.fact_value(shape_blob, "true_repetition_score", 0.0) if isinstance(shape_blob, dict) else 0.0
        )
        event_low = cls.fact_value(
            values, "loop_mean_event_low_ratio", cls.fact_value(shape_blob, "low_event_ratio", bass_ratio)
        )
        event_mid = cls.fact_value(
            values, "loop_mean_event_mid_ratio", cls.fact_value(shape_blob, "mid_event_ratio", mid_ratio)
        )
        event_high = cls.fact_value(
            values,
            "loop_mean_event_high_ratio",
            cls.fact_value(shape_blob, "high_event_ratio", presence_ratio + air_ratio),
        )
        layer = (
            facts.evidence.get("physics_layer_decision") if isinstance(getattr(facts, "evidence", None), dict) else {}
        )
        if not isinstance(layer, dict):
            layer = {}
        layer_branch = str(layer.get("physics_layer_branch") or layer.get("instrument_branch_selected") or "")
        layer_branch_confidence = cls.fact_value(
            layer,
            "physics_layer_branch_confidence",
            cls.fact_value(layer, "instrument_branch_selected_confidence", 0.0),
        )
        layer_woodwind_branch = cls.fact_value(layer, "instrument_branch_Woodwinds", 0.0)
        layer_subpanel = str(layer.get(f"instrument_{layer_branch}_subpanel_selected") or "")
        layer_subpanel_confidence = cls.fact_value(layer, f"instrument_{layer_branch}_subpanel_confidence", 0.0)
        layer_subpanel_margin = cls.fact_value(layer, f"instrument_{layer_branch}_subpanel_margin", 0.0)
        layer_compound_strength = cls.fact_value(layer, "compound_music_strength", 0.0)
        layer_woodwind_source = bool(
            layer.get("instrument_woodwind_source_signal")
            or layer.get("instrument_clean_tonal_reed_solo_signal")
            or layer.get("instrument_dark_low_mid_reed_loop_signal")
        )
        strong_non_woodwind_subpanel = bool(
            layer_subpanel and layer_subpanel_confidence >= 0.76 and layer_subpanel_margin >= 0.075
        )
        branch_identity_conflict = bool(
            not layer_woodwind_source
            and layer_branch
            and layer_branch not in {"Woodwinds", "ReedWoodwind"}
            and layer_branch_confidence >= 0.70
            and (
                strong_non_woodwind_subpanel
                or (layer_branch == "MixedInstrument" and layer_compound_strength >= 0.54)
                or (
                    layer_branch in {"MalletBell", "Synth", "KeysPiano", "Strings", "Brass", "Bass"}
                    and layer_branch_confidence >= max(0.80, layer_woodwind_branch + 0.02)
                )
            )
        )

        direct_body_flatness = cls.fact_value(direct_values, "body_flatness", body_flatness)
        direct_tail_flatness = cls.fact_value(direct_values, "tail_flatness", tail_flatness)
        direct_harmonic = cls.fact_value(direct_values, "harmonic_energy_ratio", harmonic_energy)
        direct_presence = cls.fact_value(direct_values, "presence_ratio_2000_8000hz", presence_ratio)

        fa_status = str(first_arrival_telemetry.get("status", "missing"))
        fa_cepstral = float(first_arrival_telemetry.get("cepstral_pitch_period_coherence", 0.0) or 0.0)
        fa_conical = float(first_arrival_telemetry.get("first_arrival_conical_balance", 0.0) or 0.0)
        fa_presence_db = float(first_arrival_telemetry.get("first_arrival_presence_contrast_db", 0.0) or 0.0)
        fa_pre_diffuse = float(first_arrival_telemetry.get("pre_onset_diffuse_ratio", 0.0) or 0.0)
        fa_tail_diffusion = float(first_arrival_telemetry.get("tail_diffusion_ratio", 0.0) or 0.0)
        fa_stochastic = float(first_arrival_telemetry.get("stochastic_modulation_coherence", 0.0) or 0.0)
        fa_formant_stability = float(first_arrival_telemetry.get("formant_stability_score", 0.0) or 0.0)
        fa_event_count = float(first_arrival_telemetry.get("selected_event_count", 0.0) or 0.0)

        pitch_anchor_score = cls.ramp(pitch_conf, 0.42, 0.82)
        loop_pitch_score = cls.ramp(max(loop_pitched, sustained_tonal), 0.56, 0.92)
        voiced_support_score = cls.ramp(voiced, 0.25, 0.82)
        pitch_score = min(
            1.0, max(0.0, 0.46 * pitch_anchor_score + 0.38 * loop_pitch_score + 0.16 * voiced_support_score)
        )
        non_percussive_score = cls.inverse_ramp(percussive_loop, 0.16, 0.42)
        reed_air_noise_score = max(
            cls.ramp(body_flatness, 0.050, 0.145),
            cls.ramp(tail_flatness, 0.080, 0.245),
            cls.ramp(spectral_flatness, 0.055, 0.165),
            cls.ramp(max(body_noise, tail_noise), 0.055, 0.180),
            cls.ramp(max(direct_body_flatness, direct_tail_flatness), 0.055, 0.165),
        )
        harmonic_score = 0.65 * cls.ramp(max(harmonic_energy, direct_harmonic), 0.22, 0.50)
        harmonic_score += 0.35 * cls.inverse_ramp(inharmonicity, 0.22, 0.55)
        harmonic_score = min(1.0, max(0.0, harmonic_score))
        low_body_score = cls.ramp(bass_ratio, 0.42, 0.82) * cls.inverse_ramp(air_ratio, 0.035, 0.18)
        presence_score = max(
            cls.ramp(mid_ratio + presence_ratio, 0.070, 0.240),
            cls.ramp(direct_presence, 0.015, 0.080),
        )
        wet_body_score = max(
            cls.ramp(tail_energy, 0.24, 0.52),
            cls.ramp(tail_entropy - body_entropy, 0.045, 0.180),
            cls.ramp(tail_flatness - body_flatness, 0.045, 0.160),
        )
        peak_score = max(
            cls.ramp(peak_stability, 0.16, 0.34),
            cls.inverse_ramp(abs(formant_spacing - 1.0), 0.0, 1.35),
        )
        first_arrival_cepstral_score = cls.ramp(fa_cepstral, 0.16, 0.56)
        first_arrival_conical_score = cls.ramp(fa_conical, 0.16, 0.64)
        first_arrival_presence_score = cls.ramp(fa_presence_db, 7.5, 22.0)
        first_arrival_tail_score = max(cls.ramp(fa_tail_diffusion, 0.035, 0.18), cls.ramp(tail_energy, 0.30, 0.58))
        first_arrival_stochastic_score = cls.ramp(fa_stochastic, 0.40, 0.78)
        first_arrival_formant_score = max(
            cls.ramp(fa_formant_stability, 0.30, 0.78),
            cls.ramp(fa_event_count, 1.0, 4.0) * cls.ramp(fa_presence_db, 9.0, 24.0),
        )
        # If the pre-onset frame is already contaminated by reverse reverb or
        # early reflections, raw harmonic ratios become less reliable.  Cepstral
        # pitch-period evidence is then weighted more heavily because it removes
        # smooth room coloration in the quefrency domain.
        pre_onset_trust = cls.inverse_ramp(fa_pre_diffuse, 0.11, 0.34)
        first_arrival_harmonic_identity_score = (
            (0.28 + 0.16 * (1.0 - pre_onset_trust)) * first_arrival_cepstral_score
            + (0.25 * pre_onset_trust + 0.10 * (1.0 - pre_onset_trust)) * first_arrival_conical_score
            + 0.18 * first_arrival_presence_score
            + 0.12 * first_arrival_tail_score
            + 0.10 * first_arrival_stochastic_score
            + 0.07 * first_arrival_formant_score
        )
        first_arrival_harmonic_identity_score = min(1.0, max(0.0, first_arrival_harmonic_identity_score))

        clean_synth_penalty = 0.0
        if max(body_flatness, tail_flatness, spectral_flatness) < 0.045 and tail_noise < 0.065:
            clean_synth_penalty += 0.32
        if formant_spacing >= 0.85 and max(body_flatness, tail_flatness) < 0.070:
            clean_synth_penalty += 0.22

        clean_keys_penalty = 0.0
        if fundamental_dominance < 0.080 and mid_ratio >= 0.45 and max(body_flatness, tail_flatness) < 0.075:
            clean_keys_penalty += 0.38
        if mid_ratio >= 0.62 and presence_ratio < 0.012 and max(body_flatness, tail_flatness) < 0.055:
            clean_keys_penalty += 0.18
        clean_synth_pad_loop = bool(
            shape_name in {"pitched_phrase", "vocal_phrase", "repeated_phrase_loop", "sustained_pad", "bass_phrase"}
            and pitch_conf >= 0.56
            and max(loop_pitched, sustained_tonal) >= 0.88
            and event_low >= 0.48
            and event_high <= 0.020
            and (presence_ratio + air_ratio) <= 0.022
            and max(body_flatness, tail_flatness, spectral_flatness) <= 0.095
            and tail_energy >= 0.35
            and percussive_loop <= 0.12
        )
        bright_synth_lead_loop = bool(
            shape_name in {"pitched_phrase", "vocal_phrase"}
            and pitch_conf >= 0.62
            and max(loop_pitched, sustained_tonal) >= 0.88
            and event_low <= 0.03
            and event_mid >= 0.62
            and event_high >= 0.20
            and spectral_flatness <= 0.16
            and max(body_entropy, tail_entropy) <= 0.42
            and fundamental_dominance >= 0.45
            and percussive_loop <= 0.12
        )
        clean_low_mid_keys_loop = bool(
            shape_name in {"pitched_phrase", "repeated_phrase_loop", "solo_phrase", "sustained_pad", "bass_phrase"}
            and pitch_conf >= 0.50
            and max(loop_pitched, sustained_tonal) >= 0.88
            and event_mid >= 0.42
            and 0.18 <= event_low <= 0.70
            and event_high <= 0.045
            and (presence_ratio + air_ratio) <= 0.045
            and max(body_flatness, tail_flatness, spectral_flatness) <= 0.085
            and max(body_noise, tail_noise) <= 0.28
            and percussive_loop <= 0.12
        )
        clean_highless_pitched_loop = bool(
            shape_name in {"pitched_phrase", "repeated_phrase_loop", "solo_phrase", "sustained_pad", "bass_phrase"}
            and pitch_conf >= 0.50
            and max(loop_pitched, sustained_tonal) >= 0.88
            and event_low >= 0.45
            and event_high <= 0.020
            and (presence_ratio + air_ratio) <= 0.030
            and max(body_flatness, tail_flatness, spectral_flatness) <= 0.045
            and max(body_noise, tail_noise) <= 0.30
            and percussive_loop <= 0.08
        )
        if clean_low_mid_keys_loop:
            clean_keys_penalty += 0.54
        if clean_highless_pitched_loop:
            clean_keys_penalty += 0.48
        if clean_synth_pad_loop:
            clean_synth_penalty += 0.50
        if bright_synth_lead_loop:
            clean_synth_penalty += 0.46
        synthetic_vocal_phrase_loop = bool(
            shape_name == "vocal_phrase"
            and pitch_conf >= 0.55
            and max(loop_pitched, sustained_tonal) >= 0.88
            and event_low <= 0.22
            and event_mid >= 0.64
            and 0.08 <= event_high <= 0.18
            and spectral_flatness >= 0.20
            and max(body_noise, tail_noise) >= 0.32
            and harmonic_energy <= 0.48
            and percussive_loop <= 0.12
        )
        clean_mid_vocal_phrase_loop = bool(
            shape_name == "vocal_phrase"
            and pitch_conf >= 0.72
            and voiced >= 0.75
            and max(loop_pitched, sustained_tonal) >= 0.92
            and event_mid >= 0.55
            and event_high <= 0.030
            and event_low <= 0.46
            and harmonic_energy <= 0.42
            and first_arrival_conical_score <= 0.18
            and percussive_loop <= 0.08
        )

        bell_or_drum_penalty = 0.0
        if inharmonicity >= 0.44:
            bell_or_drum_penalty += 0.30
        if percussive_loop >= 0.45:
            bell_or_drum_penalty += 0.35
        if pitch_conf < 0.45 or voiced < 0.65:
            bell_or_drum_penalty += 0.35

        bowed_string_penalty = 0.0
        if fundamental_dominance < 0.16 and presence_ratio >= 0.14 and body_flatness >= 0.18 and bass_ratio < 0.46:
            bowed_string_penalty += 0.46
        if formant_spacing >= 3.0 and fundamental_dominance < 0.20:
            bowed_string_penalty += 0.24

        weak_harmonic_reed_penalty = 0.0
        if harmonic_score < 0.38 and first_arrival_cepstral_score < 0.45:
            weak_harmonic_reed_penalty += 0.56
        if harmonic_energy < 0.08 and first_arrival_conical_score < 0.25 and first_arrival_stochastic_score < 0.36:
            weak_harmonic_reed_penalty += 0.56

        dense_vocal_phrase_penalty = 0.0
        voice_identity = cls.fact_value(values, "formant_light_voice_identity", 0.0)
        voice_role = max(
            cls.fact_value_from_evidence(facts, "vocal_music_phrase", 0.0),
            cls.fact_value_from_evidence(facts, "vocal_phrase", 0.0),
            cls.fact_value_from_evidence(facts, "vocal_one_shot", 0.0),
            cls.fact_value_from_evidence(facts, "voiced_one_shot", 0.0),
        )
        dense_articulated_vocal_phrase = bool(
            onset_count >= 45.0
            and spectral_flatness >= 0.28
            and mid_ratio >= 0.35
            and presence_ratio >= 0.08
            and air_ratio <= 0.25
        )
        if dense_articulated_vocal_phrase:
            dense_vocal_phrase_penalty += 0.62
        if voice_identity >= 0.55:
            dense_vocal_phrase_penalty += 0.62
        elif voice_role >= 0.54:
            # Shape/role voice can be a false positive on wet sax/reed loops.
            # Keep it as a soft caution unless the dedicated voice identity
            # feature also agrees.
            dense_vocal_phrase_penalty += 0.22
        if shape_name == "vocal_phrase" and voice_role >= 0.30:
            dense_vocal_phrase_penalty += 0.18 if voice_identity < 0.35 else 0.48
        if synthetic_vocal_phrase_loop:
            dense_vocal_phrase_penalty += 0.54
        if clean_mid_vocal_phrase_loop:
            dense_vocal_phrase_penalty += 0.58
        rhythmic_low_loop_decoy = bool(
            shape_name in {"bass_phrase", "beat_loop", "repeated_phrase_loop"}
            and onset_count >= 18.0
            and true_repetition >= 0.70
            and event_low >= 0.55
            and event_high <= 0.16
            and percussive_loop >= 0.10
        )

        role_only_vocal_phrase_penalty = bool(
            dense_vocal_phrase_penalty > 0.0
            and voice_identity < 0.35
            and not dense_articulated_vocal_phrase
            and not clean_mid_vocal_phrase_loop
            and not synthetic_vocal_phrase_loop
            and not branch_identity_conflict
            and first_arrival_harmonic_identity_score >= 0.40
            and pitch_score >= 0.64
            and non_percussive_score >= 0.70
        )
        if role_only_vocal_phrase_penalty:
            dense_vocal_phrase_penalty = min(dense_vocal_phrase_penalty, 0.22)

        global_reed_score = (
            0.27 * pitch_score
            + 0.23 * reed_air_noise_score
            + 0.15 * harmonic_score
            + 0.12 * low_body_score
            + 0.08 * presence_score
            + 0.10 * wet_body_score
            + 0.05 * peak_score
        )
        # v31.104: first-arrival/cepstral branch.  Global averages are kept,
        # but no longer allowed to bury a wet sax phrase when onset-local source
        # evidence is stronger than the reverb-smeared full file.
        first_arrival_reed_score = (
            0.43 * first_arrival_harmonic_identity_score
            + 0.24 * pitch_score
            + 0.13 * reed_air_noise_score
            + 0.10 * wet_body_score
            + 0.10 * non_percussive_score
        )
        raw_score = max(global_reed_score, first_arrival_reed_score)
        if (
            bowed_string_penalty > 0.24
            and first_arrival_harmonic_identity_score >= 0.86
            and first_arrival_presence_score >= 0.74
            and first_arrival_cepstral_score >= 0.74
        ):
            bowed_string_penalty = min(bowed_string_penalty, 0.24)

        penalty = min(
            0.90,
            clean_synth_penalty
            + clean_keys_penalty
            + bell_or_drum_penalty
            + bowed_string_penalty
            + weak_harmonic_reed_penalty
            + dense_vocal_phrase_penalty,
        )
        if rhythmic_low_loop_decoy:
            penalty = max(penalty, 0.88)
        if branch_identity_conflict:
            penalty = max(penalty, 0.82)
        first_arrival_penalty_relief = 0.42 * first_arrival_harmonic_identity_score
        effective_penalty = max(0.0, penalty - first_arrival_penalty_relief)
        if rhythmic_low_loop_decoy:
            effective_penalty = max(effective_penalty, 0.74)
        if branch_identity_conflict:
            effective_penalty = max(effective_penalty, 0.72)
        if dense_vocal_phrase_penalty > 0.0 and not role_only_vocal_phrase_penalty:
            effective_penalty = max(effective_penalty, dense_vocal_phrase_penalty)
        vocal_penalty_blocks_sax = bool(dense_vocal_phrase_penalty > 0.0 and not role_only_vocal_phrase_penalty)
        identity_score = min(1.0, max(0.0, raw_score - effective_penalty))
        global_eligible = bool(
            identity_score >= 0.62
            and pitch_score >= 0.78
            and non_percussive_score >= 0.70
            and reed_air_noise_score >= 0.45
            and harmonic_score >= 0.42
            and penalty <= 0.38
            and not branch_identity_conflict
            and not vocal_penalty_blocks_sax
        )
        first_arrival_eligible = bool(
            fa_status == "ok"
            and identity_score >= 0.58
            and first_arrival_harmonic_identity_score >= 0.40
            and pitch_score >= 0.64
            and non_percussive_score >= 0.70
            and max(
                reed_air_noise_score,
                first_arrival_tail_score,
                first_arrival_presence_score,
                first_arrival_stochastic_score,
            )
            >= 0.42
            and not (voiced < 0.35 and first_arrival_conical_score < 0.20)
            and not vocal_penalty_blocks_sax
            and not clean_low_mid_keys_loop
            and not clean_highless_pitched_loop
            and not clean_synth_pad_loop
            and not rhythmic_low_loop_decoy
            and not branch_identity_conflict
            and effective_penalty <= 0.52
            and not (
                clean_synth_penalty >= 0.32
                and first_arrival_cepstral_score < 0.48
                and first_arrival_conical_score < 0.48
                and first_arrival_presence_score < 0.52
            )
            and not (bell_or_drum_penalty >= 0.35 and first_arrival_harmonic_identity_score < 0.62)
        )
        eligible = bool(global_eligible or first_arrival_eligible)
        return {
            "reed_sax_identity_score": round(float(identity_score), 6),
            "reed_sax_identity_raw_score": round(float(raw_score), 6),
            "reed_sax_global_raw_score": round(float(global_reed_score), 6),
            "reed_sax_first_arrival_raw_score": round(float(first_arrival_reed_score), 6),
            "reed_sax_identity_penalty": round(float(penalty), 6),
            "reed_sax_identity_effective_penalty": round(float(effective_penalty), 6),
            "reed_sax_identity_eligible": bool(eligible),
            "reed_sax_global_eligible": bool(global_eligible),
            "reed_sax_first_arrival_eligible": bool(first_arrival_eligible),
            "reed_sax_first_arrival_status": fa_status,
            "reed_sax_pitch_score": round(float(pitch_score), 6),
            "reed_sax_non_percussive_score": round(float(non_percussive_score), 6),
            "reed_sax_air_noise_score": round(float(reed_air_noise_score), 6),
            "reed_sax_harmonic_score": round(float(harmonic_score), 6),
            "reed_sax_low_body_score": round(float(low_body_score), 6),
            "reed_sax_presence_score": round(float(presence_score), 6),
            "reed_sax_wet_body_score": round(float(wet_body_score), 6),
            "reed_sax_peak_score": round(float(peak_score), 6),
            "reed_sax_first_arrival_harmonic_identity_score": round(float(first_arrival_harmonic_identity_score), 6),
            "reed_sax_first_arrival_cepstral_score": round(float(first_arrival_cepstral_score), 6),
            "reed_sax_first_arrival_conical_score": round(float(first_arrival_conical_score), 6),
            "reed_sax_first_arrival_presence_score": round(float(first_arrival_presence_score), 6),
            "reed_sax_first_arrival_tail_score": round(float(first_arrival_tail_score), 6),
            "reed_sax_first_arrival_stochastic_score": round(float(first_arrival_stochastic_score), 6),
            "reed_sax_first_arrival_formant_score": round(float(first_arrival_formant_score), 6),
            "reed_sax_first_arrival_pre_onset_trust": round(float(pre_onset_trust), 6),
            "reed_sax_clean_synth_penalty": round(float(clean_synth_penalty), 6),
            "reed_sax_clean_keys_penalty": round(float(clean_keys_penalty), 6),
            "reed_sax_clean_low_mid_keys_loop": bool(clean_low_mid_keys_loop),
            "reed_sax_clean_highless_pitched_loop": bool(clean_highless_pitched_loop),
            "reed_sax_clean_synth_pad_loop": bool(clean_synth_pad_loop),
            "reed_sax_bright_synth_lead_loop": bool(bright_synth_lead_loop),
            "reed_sax_synthetic_vocal_phrase_loop": bool(synthetic_vocal_phrase_loop),
            "reed_sax_clean_mid_vocal_phrase_loop": bool(clean_mid_vocal_phrase_loop),
            "reed_sax_rhythmic_low_loop_decoy": bool(rhythmic_low_loop_decoy),
            "reed_sax_branch_identity_conflict": bool(branch_identity_conflict),
            "reed_sax_layer_branch": layer_branch,
            "reed_sax_layer_branch_confidence": round(float(layer_branch_confidence), 6),
            "reed_sax_layer_subpanel": layer_subpanel,
            "reed_sax_layer_subpanel_confidence": round(float(layer_subpanel_confidence), 6),
            "reed_sax_bell_or_drum_penalty": round(float(bell_or_drum_penalty), 6),
            "reed_sax_bowed_string_penalty": round(float(bowed_string_penalty), 6),
            "reed_sax_weak_harmonic_penalty": round(float(weak_harmonic_reed_penalty), 6),
            "reed_sax_dense_vocal_phrase_penalty": round(float(dense_vocal_phrase_penalty), 6),
            "reed_sax_role_only_vocal_phrase_penalty": bool(role_only_vocal_phrase_penalty),
            "reed_sax_vocal_penalty_blocks_sax": bool(vocal_penalty_blocks_sax),
        }

    def score_labels(
        self,
        physics: AudioPhysics,
        facts: SharedAudioFacts,
        brain: dict[str, Any],
        labels: list[str],
        profiles: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Score every label using full-profile residual fit. Candidates are pre-filtered by dynamic role gate."""
        rows: list[dict[str, Any]] = []
        folder_map, top_map = label_maps(brain)
        first_arrival_telemetry = self.extract_first_arrival_telemetry(physics, facts)
        if isinstance(facts.evidence, dict):
            facts.evidence["first_arrival_telemetry"] = first_arrival_telemetry
        layer_decision = self.layered_scorer.analyze(facts)
        if isinstance(facts.evidence, dict):
            facts.evidence["physics_layer_decision"] = {
                **layer_decision.evidence,
                "physics_layer_top_family": layer_decision.top_family,
                "physics_layer_top_confidence": round(float(layer_decision.top_confidence), 6),
                "physics_layer_branch": layer_decision.branch,
                "physics_layer_branch_confidence": round(float(layer_decision.branch_confidence), 6),
                "physics_layer_leaf_strategy": layer_decision.leaf_strategy,
            }
        direct_vector = self.direct_body_feature_vector(brain, facts)
        # Parallelization note: label scoring is independent after first-arrival and
        # layer decisions. This release keeps it synchronous for deterministic evidence logs.
        for label in labels:
            profile = profiles.get(label)
            if not isinstance(profile, dict):
                continue
            folder_path = str(folder_map.get(label, label))
            profile_score, profile_evidence = profile_residual_score(physics.fingerprint, profile, self.policy)
            if profile_score == float("inf"):
                continue
            profile_score, direct_evidence = self.apply_direct_body_profile_check(
                label=label,
                brain=brain,
                facts=facts,
                profile=profile,
                full_profile_score=profile_score,
                direct_vector=direct_vector,
            )
            mallet_bell_score, mallet_bell_evidence = self.apply_mallet_bell_eligibility_adjustment(
                label=label,
                folder_path=folder_path,
                facts=facts,
                current_score=profile_score,
            )
            profile_score = mallet_bell_score
            structure_delta, structure_reason = structure_penalty(label, brain, facts, self.policy)
            role_delta, role_evidence = role_compatibility_adjustment(label, brain, facts, voter_name=self.voter_name)
            score = max(0.0, float(profile_score + structure_delta + role_delta))
            score, piano_struck_evidence = self.apply_piano_identity_adjustment(
                label=label,
                folder_path=folder_path,
                facts=facts,
                current_score=score,
                first_arrival_telemetry=first_arrival_telemetry,
            )
            score, reed_sax_evidence = self.apply_reed_sax_identity_adjustment(
                label=label,
                folder_path=folder_path,
                facts=facts,
                current_score=score,
                first_arrival_telemetry=first_arrival_telemetry,
            )
            score, layered_evidence = self.layered_scorer.apply(folder_path, score, layer_decision)
            rows.append(
                {
                    "label": label,
                    "score": score,
                    "confidence": 1.0 / (1.0 + max(0.0, score)),
                    "reason": "physics_profile_match",
                    "evidence": {
                        **profile_evidence,
                        **direct_evidence,
                        **mallet_bell_evidence,
                        **piano_struck_evidence,
                        **reed_sax_evidence,
                        **layered_evidence,
                        "raw_profile_score": round(float(profile_score), 6),
                        "structure_penalty": round(float(structure_delta), 6),
                        "structure_penalty_reason": structure_reason,
                        "physics_score_after_adjustments": round(float(score), 6),
                        "score_is_raw_voter_score": structure_delta == 0.0 and abs(float(role_delta)) < 1e-9,
                        **role_evidence,
                        "structure_gate": "candidate_pre_filtered",
                    },
                }
            )
        return rows

    @staticmethod
    def direct_body_feature_vector(brain: dict[str, Any], facts: SharedAudioFacts) -> np.ndarray | None:
        """Build the direct/body feature vector once per sample."""
        view = facts.evidence.get("direct_body_view", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(view, dict) or not view.get("available"):
            return None
        direct_values = view.get("feature_values_by_name", {})
        if not isinstance(direct_values, dict):
            return None
        names = brain.get("feature_names", [])
        if not names:
            return None
        return np.asarray([float(direct_values.get(name, 0.0) or 0.0) for name in names], dtype=np.float32)
