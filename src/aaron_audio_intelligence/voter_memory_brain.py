"""Trainable voter-memory brain for low-level role calibration.

This brain is a deliberate next step toward fewer static voter rules.  It
stores human-approved fingerprints as reusable role evidence for BrainVoter,
PhysicsVoter, and ShapeVoter.  The stored label is the supervised target chosen
by the user, while runtime matching uses only numeric audio fingerprints.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaron_audio_intelligence.learned_memory_features import (
    PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
    pad_vector,
    scaled_vector_distance,
    weighted_normalized_signature_vector,
    weighted_normalized_vector,
)
from aaron_audio_intelligence.shape_memory_brain import (
    example_fingerprint,
    safe_int,
    shape_target_for_label,
)
from aaron_sound_sorter.core import FP_SIZE, FeatureRow
from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.training_labels import label_default_structure, public_label, top_for_public_label

VOTER_MEMORY_BRAIN_NAME = "stage4_voter_memory_brain.json"
VOTER_MEMORY_KEY = "voter_memory"
MAX_VOTER_MEMORY_EXAMPLES_PER_ROLE = 300

_METADATA_KEYS = (
    "version",
    "feature_names",
    "feature_size",
    "feature_weights",
    "scaler_mean",
    "scaler_std",
)

SOURCE_OWNER_MEMORY_FEATURES = frozenset(
    {
        "log_rolloff85_hz",
        "log_crest",
        "spectral_flux_mean",
        "spectral_flatness_mean",
        "spectral_entropy_mean",
        "log_decay_ratio",
        "stereo_width",
        "mid_side_ratio",
        "centroid_slope_norm",
        "log_transient_count",
        "sub_bass_ratio_lt_150hz",
        "bass_ratio_150_500hz",
        "mid_ratio_500_2000hz",
        "presence_ratio_2000_8000hz",
        "air_ratio_gt_8000hz",
        "zcr_mean",
        "temporal_centroid_ratio",
        "onset_interval_regularity",
        "attack_rise_time_norm",
        "onset_span_ratio",
        "event_rate_hz",
        "tail_energy_ratio",
        "spectral_flux_variance",
        "pitch_confidence",
        "f0_voiced_ratio",
        "f0_stability_cents",
        "f0_slope_cents_per_sec",
        "harmonic_to_noise_ratio",
        "harmonic_peak_count",
        "harmonic_energy_ratio",
        "inharmonicity",
        "fundamental_dominance_ratio",
        "overtone_slope",
        "attack_pitch_confidence",
        "body_pitch_confidence",
        "tail_pitch_confidence",
        "attack_flatness",
        "body_flatness",
        "tail_flatness",
        "attack_entropy",
        "body_entropy",
        "tail_entropy",
        "attack_zcr",
        "body_zcr",
        "tail_zcr",
        "attack_high_ratio",
        "body_high_ratio",
        "tail_high_ratio",
        "attack_low_ratio",
        "body_low_ratio",
        "tail_low_ratio",
        "noise_burst_duration_ms",
        "high_band_decay_slope",
        "spectral_centroid_decay_slope",
        "noise_tail_decay_slope",
        "attack_noise_ratio",
        "body_noise_ratio",
        "tail_noise_ratio",
        "low_peak_bandwidth_hz",
        "sub_attack_time_ms",
        "sub_decay_time_ms",
        "sub_to_click_offset_ms",
        "kick_pitch_drop_cents",
        "sub_sustain_ratio",
        "spectral_peak_count",
        "peak_bandwidth_mean_hz",
        "spectral_peak_stability",
        "formant_like_peak_spacing",
        "spectral_envelope_slope",
        "loop_pitched_event_ratio",
        "loop_percussive_event_ratio",
        "loop_noisy_event_ratio",
        "loop_event_timbre_diversity",
        "loop_sustained_tonal_frame_ratio",
        "loop_drumlike_frame_ratio",
        "loop_tonal_to_percussive_balance",
        "loop_mean_event_pitch_confidence",
        "loop_mean_event_noise_ratio",
        "loop_mean_event_low_ratio",
        "loop_mean_event_high_ratio",
        "loop_non_event_tonal_ratio",
    }
)

SOURCE_OWNER_MEMORY_PREFIXES = ("mfcc_mu_", "mfcc_std_")


@dataclass(frozen=True)
class VoterMemoryMatch:
    """Nearest learned low-level role match for a query fingerprint.

    Args:
        matched: True when the query is close enough to learned role evidence.
        role: Learned role target such as ``fx_impact`` or ``instrument_keys``.
        label: Human-approved taxonomy label for the nearest teacher example.
        top_family: Top family associated with the nearest teacher example.
        shape: Broad structural shape associated with the nearest example.
        nearest_distance: Weighted normalized distance to learned evidence.
        confidence: Calibration confidence contributed by this memory lane.
        example_count: Valid examples inspected for the winning role.
        effective_weight: Human correction support weight for the winning role.
        match_kind: Diagnostic match mode.
        threshold: Active maximum accepted distance.

    Side Effects:
        None.
    """

    matched: bool
    role: str
    label: str
    top_family: str
    shape: str
    nearest_distance: float
    confidence: float
    example_count: int
    effective_weight: int
    match_kind: str
    threshold: float

    def evidence(self) -> dict[str, Any]:
        """Return diagnostics for sorter manifests and voter layers."""
        nested = {
            "enabled": self.example_count > 0,
            "matched": bool(self.matched),
            "role": str(self.role),
            "label": str(self.label),
            "top_family": str(self.top_family),
            "shape": str(self.shape),
            "confidence": round(float(self.confidence), 6),
            "nearest_distance": round(float(self.nearest_distance), 6),
            "threshold": round(float(self.threshold), 6),
            "example_count": int(self.example_count),
            "effective_weight": int(self.effective_weight),
            "match_kind": str(self.match_kind),
            "policy": "teacher_fingerprint_role_calibration",
        }
        return {
            "learned_voter_memory": nested,
            "learned_voter_memory_enabled": nested["enabled"],
            "learned_voter_memory_matched": nested["matched"],
            "learned_voter_memory_role": nested["role"],
            "learned_voter_memory_label": nested["label"],
            "learned_voter_memory_top_family": nested["top_family"],
            "learned_voter_memory_shape": nested["shape"],
            "learned_voter_memory_confidence": nested["confidence"],
            "learned_voter_memory_nearest_distance": nested["nearest_distance"],
            "learned_voter_memory_threshold": nested["threshold"],
            "learned_voter_memory_example_count": nested["example_count"],
            "learned_voter_memory_effective_weight": nested["effective_weight"],
            "learned_voter_memory_match_kind": nested["match_kind"],
            "learned_voter_memory_policy": nested["policy"],
        }


def build_empty_voter_memory_brain(base_brain: dict[str, Any]) -> dict[str, Any]:
    """Return an empty brain JSON ready to store low-level role evidence.

    Args:
        base_brain: Active full brain used as the source of feature metadata.

    Returns:
        Brain-shaped dictionary with an empty voter-memory payload.

    Side Effects:
        None.
    """
    memory: dict[str, Any] = {key: deepcopy(base_brain[key]) for key in _METADATA_KEYS if key in base_brain}
    memory.update(
        {
            "brain_type": "voter_memory_brain",
            "labels": [],
            VOTER_MEMORY_KEY: empty_voter_memory_payload(),
            "voter_memory_policy": "gui_corrections_teach_low_level_voter_roles",
        }
    )
    return memory


def empty_voter_memory_payload() -> dict[str, Any]:
    """Return the empty nested voter-memory payload."""
    return {
        "version": "v1",
        "examples_by_role": {},
        "centroids_by_role": {},
        "counts_by_role": {},
        "effective_weights_by_role": {},
    }


def is_voter_memory_brain(brain: dict[str, Any] | None) -> bool:
    """Return true when ``brain`` is the dedicated voter-memory lane."""
    return bool(isinstance(brain, dict) and brain.get("brain_type") == "voter_memory_brain")


def merge_voter_memory_into_brain(
    brain: dict[str, Any],
    voter_memory_brain: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge standalone voter-memory evidence into an in-memory brain view.

    Args:
        brain: Loaded full brain used for the current sort run.
        voter_memory_brain: Optional loaded voter-memory brain.

    Returns:
        ``brain`` after in-memory merge.

    Side Effects:
        Mutates only the passed ``brain`` object for this run. No files are
        written.
    """
    if not is_voter_memory_brain(voter_memory_brain):
        return brain
    source_payload = voter_memory_brain.get(VOTER_MEMORY_KEY)
    if not isinstance(source_payload, dict):
        return brain
    target_payload = ensure_voter_memory_payload(brain)
    merge_voter_memory_payload(target_payload, source_payload)
    brain["_voter_memory_brain_loaded"] = True
    brain["_voter_memory_role_count"] = len(target_payload.get("examples_by_role", {}))
    return brain


def update_voter_memory_with_row(
    brain: dict[str, Any],
    row: FeatureRow,
    *,
    evidence_weight: int,
    memory_origin: str = "gui_correction",
) -> str:
    """Store one human correction as learned low-level voter role evidence.

    Args:
        brain: Brain dictionary to mutate.
        row: Measured correction row with a human-approved taxonomy label.
        evidence_weight: Bounded human-correction support weight.
        memory_origin: Audit source for this teacher example.

    Returns:
        Learned role touched by the update, or an empty string if no role was
        derivable from the supervised label.

    Side Effects:
        Mutates ``brain`` by adding one numeric fingerprint example and
        refreshing that role's centroid summary.
    """
    role = voter_memory_role_for_label(row.label, row.structure)
    if not role:
        return ""
    payload = ensure_voter_memory_payload(brain)
    examples_by_role = ensure_payload_map(payload, "examples_by_role")
    examples = examples_by_role.setdefault(role, [])
    if not isinstance(examples, list):
        examples = []
        examples_by_role[role] = examples
    existing = next(
        (
            example
            for example in examples
            if isinstance(example, dict) and str(example.get("source_path")) == str(row.path)
        ),
        None,
    )
    if isinstance(existing, dict):
        reinforce_voter_memory_example(existing, evidence_weight=evidence_weight)
    else:
        examples.append(
            voter_memory_example_from_row(
                row,
                role,
                evidence_weight=evidence_weight,
                memory_origin=memory_origin,
            )
        )
    del examples[:-MAX_VOTER_MEMORY_EXAMPLES_PER_ROLE]
    refresh_voter_memory_role(payload, role)
    if is_voter_memory_brain(brain):
        labels = [str(label) for label in brain.get("labels", []) if str(label)]
        if role not in labels:
            labels.append(role)
            brain["labels"] = sorted(labels)
    brain["voter_memory_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return role


def attach_voter_memory_to_facts(brain: dict[str, Any], facts: SharedAudioFacts) -> VoterMemoryMatch:
    """Attach the nearest learned role match to ``facts.evidence``.

    Args:
        brain: Active in-memory brain with optional merged voter memory.
        facts: Shared facts for the current audio file.

    Returns:
        The nearest voter-memory match.

    Side Effects:
        Updates ``facts.evidence`` with flat and nested diagnostics.
    """
    match = voter_memory_match_for_facts(brain, facts.feature_vector)
    if isinstance(getattr(facts, "evidence", None), dict):
        facts.evidence.update(match.evidence())
    return match


def voter_memory_match_for_facts(
    brain: dict[str, Any],
    feature_vector: tuple[float, ...] | list[float],
) -> VoterMemoryMatch:
    """Return the strongest learned voter-memory match for a query vector."""
    payload = brain.get(VOTER_MEMORY_KEY)
    if not isinstance(payload, dict):
        return no_voter_memory_match()
    examples_by_role = payload.get("examples_by_role")
    if not isinstance(examples_by_role, dict) or not examples_by_role:
        return no_voter_memory_match()
    query = weighted_normalized_vector(brain, feature_vector)
    if query.size <= 0:
        return no_voter_memory_match()
    owner_signature_query = weighted_normalized_signature_vector(
        brain,
        feature_vector,
        include_feature_names=SOURCE_OWNER_MEMORY_FEATURES,
        exclude_feature_names=PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
        include_feature_prefixes=SOURCE_OWNER_MEMORY_PREFIXES,
        minimum_feature_count=36,
    )

    best_role = ""
    best_label = ""
    best_top = ""
    best_shape = ""
    best_distance = float("inf")
    best_count = 0
    best_weight = 0
    best_match_kind = "teacher_role_cloud"
    for role, examples in examples_by_role.items():
        if not isinstance(examples, list):
            continue
        valid_count = 0
        effective_weight = 0
        nearest = float("inf")
        nearest_kind = "teacher_role_cloud"
        nearest_example: dict[str, Any] | None = None
        for example in examples:
            if not isinstance(example, dict):
                continue
            vector = example_fingerprint(example)
            if vector.size <= 0:
                continue
            weighted_example = weighted_normalized_vector(brain, vector)
            distance = scaled_vector_distance(query, weighted_example)
            owner_signature_example = weighted_normalized_signature_vector(
                brain,
                vector,
                include_feature_names=SOURCE_OWNER_MEMORY_FEATURES,
                exclude_feature_names=PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
                include_feature_prefixes=SOURCE_OWNER_MEMORY_PREFIXES,
                minimum_feature_count=36,
            )
            signature_distance = scaled_vector_distance(
                owner_signature_query,
                owner_signature_example,
                reference_size=query.size,
            )
            if signature_distance < distance:
                distance = signature_distance
                candidate_kind = "teacher_role_signature_cloud"
            else:
                candidate_kind = "teacher_role_cloud"
            if math.isfinite(distance):
                valid_count += 1
                effective_weight = max(effective_weight, safe_int(example.get("human_override_evidence_weight")))
                if distance < nearest:
                    nearest = distance
                    nearest_kind = candidate_kind
                    nearest_example = example
        if valid_count > 0 and nearest < best_distance:
            best_role = str(role)
            best_distance = nearest
            best_count = valid_count
            best_weight = effective_weight
            best_match_kind = nearest_kind
            best_label = str(nearest_example.get("approved_label", "")) if isinstance(nearest_example, dict) else ""
            best_top = str(nearest_example.get("top", "")) if isinstance(nearest_example, dict) else ""
            best_shape = str(nearest_example.get("target_shape", "")) if isinstance(nearest_example, dict) else ""

    if not best_role:
        return no_voter_memory_match()
    threshold = voter_memory_distance_threshold(best_weight, example_count=best_count)
    matched = bool(best_distance <= threshold)
    match_kind = "none"
    confidence = 0.0
    if matched:
        exact_threshold = min(1.25, max(0.22, 0.22 + math.log1p(max(1, best_weight)) / 11.0))
        match_kind = (
            "fingerprint"
            if best_match_kind == "teacher_role_cloud" and best_distance <= exact_threshold
            else best_match_kind
        )
        distance_ratio = min(1.0, best_distance / max(threshold, 1e-6))
        confidence = min(0.94, max(0.62, 0.94 - 0.25 * distance_ratio + math.log1p(max(1, best_count)) / 32.0))
    return VoterMemoryMatch(
        matched=matched,
        role=best_role,
        label=best_label,
        top_family=best_top,
        shape=best_shape,
        nearest_distance=best_distance,
        confidence=confidence,
        example_count=best_count,
        effective_weight=best_weight,
        match_kind=match_kind,
        threshold=threshold,
    )


def voter_memory_distance_threshold(effective_weight: int, *, example_count: int) -> float:
    """Return accepted weighted distance for learned voter-memory recall."""
    weight = max(1, int(effective_weight or 0))
    count = max(1, int(example_count or 0))
    return min(2.90, 0.92 + math.log1p(weight) / 7.2 + math.log1p(count) / 8.5)


def ensure_voter_memory_payload(brain: dict[str, Any]) -> dict[str, Any]:
    """Return a mutable voter-memory payload on ``brain``."""
    payload = brain.get(VOTER_MEMORY_KEY)
    if not isinstance(payload, dict):
        payload = empty_voter_memory_payload()
        brain[VOTER_MEMORY_KEY] = payload
    for key, value in empty_voter_memory_payload().items():
        if key not in payload:
            payload[key] = deepcopy(value)
    return payload


def merge_voter_memory_payload(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge one voter-memory payload into another."""
    target_examples = ensure_payload_map(target, "examples_by_role")
    source_examples = source.get("examples_by_role")
    if not isinstance(source_examples, dict):
        return
    for role, examples in source_examples.items():
        if not isinstance(examples, list):
            continue
        current = target_examples.setdefault(str(role), [])
        if not isinstance(current, list):
            current = []
            target_examples[str(role)] = current
        indexes_by_path = {
            str(example.get("source_path", "")): index
            for index, example in enumerate(current)
            if isinstance(example, dict) and str(example.get("source_path", ""))
        }
        for example in examples:
            if not isinstance(example, dict):
                continue
            example_path = str(example.get("source_path", ""))
            if example_path and example_path in indexes_by_path:
                existing_index = indexes_by_path[example_path]
                existing_weight = safe_int(current[existing_index].get("human_override_evidence_weight"))
                incoming_weight = safe_int(example.get("human_override_evidence_weight"))
                if incoming_weight > existing_weight:
                    current[existing_index] = deepcopy(example)
                continue
            if example_path:
                indexes_by_path[example_path] = len(current)
            current.append(deepcopy(example))
        del current[:-MAX_VOTER_MEMORY_EXAMPLES_PER_ROLE]
        refresh_voter_memory_role(target, str(role))


def ensure_payload_map(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested mutable mapping from a voter-memory payload."""
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    payload[key] = {}
    return payload[key]


def voter_memory_example_from_row(
    row: FeatureRow,
    role: str,
    *,
    evidence_weight: int,
    memory_origin: str = "gui_correction",
) -> dict[str, Any]:
    """Return one stored voter-memory example."""
    shape = shape_target_for_label(row.label, row.structure)
    return {
        "source_path": str(row.path),
        "approved_label": str(row.label),
        "target_role": str(role),
        "target_shape": str(shape),
        "top": str(row.top),
        "structure": str(row.structure),
        "duration_sec": float(row.duration_sec),
        "fingerprint": [float(value) for value in row.fingerprint],
        "memory_origin": str(memory_origin or "gui_correction"),
        "incremental_gui_correction": str(memory_origin or "gui_correction") == "gui_correction",
        "voter_memory_added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "human_override_confirmation_count": 1,
        "human_override_evidence_weight": safe_int(evidence_weight),
    }


def reinforce_voter_memory_example(example: dict[str, Any], *, evidence_weight: int) -> None:
    """Increase human support for an existing voter-memory example."""
    previous = safe_int(example.get("human_override_evidence_weight"))
    confirmation_count = safe_int(example.get("human_override_confirmation_count")) + 1
    example["human_override_confirmation_count"] = confirmation_count
    example["human_override_evidence_weight"] = max(previous, safe_int(evidence_weight))
    example["voter_memory_last_confirmed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def refresh_voter_memory_role(payload: dict[str, Any], role: str) -> None:
    """Refresh centroid/count summaries for one learned role."""
    examples = ensure_payload_map(payload, "examples_by_role").get(role, [])
    if not isinstance(examples, list):
        examples = []
    vectors: list[np.ndarray] = []
    weight_total = 0
    for example in examples:
        if not isinstance(example, dict):
            continue
        vector = example_fingerprint(example)
        if vector.size != FP_SIZE:
            continue
        weight = max(1, safe_int(example.get("human_override_evidence_weight")))
        vectors.append(vector * float(weight))
        weight_total += weight
    ensure_payload_map(payload, "counts_by_role")[role] = len(vectors)
    ensure_payload_map(payload, "effective_weights_by_role")[role] = int(weight_total)
    centroids = ensure_payload_map(payload, "centroids_by_role")
    if vectors and weight_total > 0:
        centroids[role] = (np.sum(np.vstack(vectors), axis=0) / float(weight_total)).astype(float).tolist()
    else:
        centroids[role] = []


def voter_memory_role_for_label(label: str, structure: str | None = None) -> str:
    """Return the low-level role target implied by a supervised label."""
    normalized = public_label(label)
    top = top_for_public_label(normalized)
    structure_name = str(structure or label_default_structure(normalized) or "").lower()
    parts = [part.strip().lower() for part in normalized.replace("\\", "/").split("/") if part.strip()]
    joined = "/".join(parts)
    suffix = "loop" if structure_name == "loop" else ("long_fx" if structure_name == "long_fx" else "one_shot")
    if top == "Drums":
        if "kick" in joined:
            return f"drum_kick_{suffix}"
        if "snare" in joined:
            return f"drum_snare_{suffix}"
        if "clap" in joined or "snap" in joined:
            return f"drum_clap_snap_{suffix}"
        if "hat" in joined:
            return f"drum_hat_{suffix}"
        if "cymbal" in joined:
            return f"drum_cymbal_{suffix}"
        if "tom" in joined or "conga" in joined or "bongo" in joined or "tabla" in joined:
            return f"drum_membrane_{suffix}"
        return f"drum_general_{suffix}"
    if top == "Instruments":
        if "voice" in joined or "vocal" in joined:
            return f"instrument_voice_{suffix}"
        if "bass" in joined or "808" in joined:
            return f"instrument_bass_{suffix}"
        if "synth" in joined:
            return f"instrument_synth_{suffix}"
        if any(token in joined for token in ("piano", "keys", "rhodes", "wurlitzer", "organ", "clav")):
            return f"instrument_keys_{suffix}"
        if any(token in joined for token in ("guitar", "koto", "plucked", "harp", "banjo", "mandolin")):
            return f"instrument_plucked_{suffix}"
        if any(token in joined for token in ("sax", "reed", "woodwind", "flute", "brass", "trumpet", "trombone")):
            return f"instrument_wind_{suffix}"
        if "strings" in joined or "violin" in joined or "cello" in joined:
            return f"instrument_strings_{suffix}"
        if "mixed musical" in joined or "instrument loops" in joined:
            return "instrument_mixed_loop"
        return f"instrument_general_{suffix}"
    if top == "FX":
        if any(
            token in joined
            for token in (
                "human and voice",
                "spoken",
                "mouth",
                "breath",
                "scream",
                "crowd",
                "applause",
            )
        ):
            return f"fx_human_voice_{suffix}"
        if any(token in joined for token in ("impact", "boom", "slam")):
            return "fx_impact"
        if "riser" in joined or "build" in joined:
            return "fx_riser"
        if "drop" in joined or "downlifter" in joined:
            return "fx_downlifter"
        if "whoosh" in joined or "sweep" in joined:
            return "fx_sweep"
        if "reverse" in joined or "swell" in joined:
            return "fx_reverse"
        if "glitch" in joined or "stutter" in joined:
            return "fx_glitch"
        if "blip" in joined or "beep" in joined:
            return "fx_blip"
        if any(token in joined for token in ("radio", "electrical", "electric", "zap")):
            return "fx_electrical"
        if any(token in joined for token in ("noise", "static", "hiss", "vinyl")):
            return "fx_noise_texture"
        if any(token in joined for token in ("water", "rain", "ocean", "wind", "fire", "thunder")):
            return "fx_natural_texture"
        if any(token in joined for token in ("animal", "bird", "cat", "dog", "cricket")):
            return "fx_biological"
        return "fx_general"
    return ""


def no_voter_memory_match() -> VoterMemoryMatch:
    """Return the empty learned voter-memory match."""
    return VoterMemoryMatch(
        matched=False,
        role="",
        label="",
        top_family="",
        shape="",
        nearest_distance=float("inf"),
        confidence=0.0,
        example_count=0,
        effective_weight=0,
        match_kind="none",
        threshold=voter_memory_distance_threshold(0, example_count=0),
    )


def finite_feature_vector(values: tuple[float, ...] | list[float] | np.ndarray) -> list[float]:
    """Return a JSON-safe finite fingerprint vector."""
    return pad_vector(np.asarray(values, dtype=np.float32), 0.0).astype(float).tolist()
