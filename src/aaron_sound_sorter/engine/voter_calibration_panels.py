# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Diagnostic calibration panels for weak voter-authority zones.

These panels do not route audio and do not change voter scores.  They expose
where existing voter and physics evidence is strong enough to block a dangerous
decoy, where it is only good enough to broaden, and where it must stay
informational.  The first targets are the current recurring failure zones:
sax, voice, blip, synth pad, and mixed instrumental loops.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


@dataclass(frozen=True)
class VoterCalibrationPanel:
    """One diagnostic authority panel for a recurring classifier failure zone."""

    target: str
    claim_type: str
    raw_support: float
    source_specific_support: float
    role_structure_support: float
    decoy_pressure: float
    margin: float
    evidence_keys: tuple[str, ...]
    decoy_keys: tuple[str, ...]
    supporting_claims: tuple[str, ...]
    decoy_claims: tuple[str, ...]
    authority: str
    confidence_tier: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly panel dictionary."""
        return asdict(self)

    def debug_line(self) -> str:
        """Return a stable compact debug line for claim trace files."""
        return (
            "CALIBRATION_PANEL "
            f"target={self.target} "
            f"type={self.claim_type} "
            f"authority={self.authority} "
            f"tier={self.confidence_tier} "
            f"support={self.raw_support:.3f} "
            f"source={self.source_specific_support:.3f} "
            f"role={self.role_structure_support:.3f} "
            f"decoy={self.decoy_pressure:.3f} "
            f"margin={self.margin:.3f} "
            f"claims={','.join(self.supporting_claims) or 'none'} "
            f"decoys={','.join(self.decoy_claims) or 'none'}"
        )


@dataclass(frozen=True)
class CalibrationTargetSpec:
    """Static source-blind panel recipe for one target area."""

    target: str
    claim_type: str
    source_specific_keys: tuple[str, ...]
    decoy_keys: tuple[str, ...]
    support_tokens: tuple[str, ...]
    decoy_tokens: tuple[str, ...]
    role_structure_keys: tuple[str, ...] = ()
    safe_block_threshold: float = 0.82
    safe_broaden_threshold: float = 0.72
    min_block_margin: float = 0.18
    min_broaden_margin: float = 0.10


CALIBRATION_TARGETS: tuple[CalibrationTargetSpec, ...] = (
    CalibrationTargetSpec(
        target="sax",
        claim_type="source_identity",
        source_specific_keys=(
            "woodwind_sax_score",
            "instrument_panel_Woodwinds_sax",
            "instrument_panel_Woodwinds_sax_loop",
            "instrument_subpanel_reed_wind_score",
            "instrument_subpanel_reed_wind_authority_score",
        ),
        decoy_keys=(
            "synth_tonal_source_score",
            "instrument_panel_Synth_synth_pad",
            "instrument_panel_Synth_synth_loop",
            "instrument_panel_KeysPiano_electric_piano",
            "struck_keys_score",
            "voice_score",
            "human_spoken_voice_score",
        ),
        support_tokens=("sax", "woodwind"),
        decoy_tokens=("synth", "pad", "keys", "piano", "voice", "vocal"),
    ),
    CalibrationTargetSpec(
        target="voice",
        claim_type="source_identity",
        source_specific_keys=(
            "voice_score",
            "instrument_subpanel_voice_score",
            "human_spoken_voice_score",
            "human_breath_mouth_score",
            "fx_formant_score",
        ),
        decoy_keys=(
            "woodwind_sax_score",
            "instrument_panel_Woodwinds_sax",
            "synth_tonal_source_score",
            "instrument_panel_Synth_synth_pad",
            "mallet_bell_source_score",
            "drum_metallic_percussion_source_score",
        ),
        support_tokens=("voice", "vocal", "spoken"),
        decoy_tokens=("sax", "woodwind", "synth", "pad", "bell", "metallic"),
    ),
    CalibrationTargetSpec(
        target="blip",
        claim_type="source_identity",
        source_specific_keys=(
            "fx_blip_beep_score",
            "fx_blip_score",
            "fx_beep_score",
            "instrument_panel_FX_blip",
        ),
        decoy_keys=(
            "drum_snare_source_score",
            "drum_clap_source_score",
            "drum_rim_stick_source_score",
            "drum_hit_score",
            "role_one_shot_score",
        ),
        support_tokens=("blip", "beep", "ui"),
        decoy_tokens=("snare", "clap", "rim", "stick", "drum"),
        safe_block_threshold=0.84,
    ),
    CalibrationTargetSpec(
        target="synth_pad",
        claim_type="source_identity",
        source_specific_keys=(
            "synth_tonal_source_score",
            "instrument_panel_Synth_synth_pad",
            "instrument_panel_Synth_synth_loop",
            "instrument_subpanel_synth_tonal_source_score",
        ),
        role_structure_keys=(
            "sustained_tonal_frame_ratio",
            "loop_sustained_tonal_frame_ratio",
            "pitched_event_ratio",
            "f0_voiced_ratio",
        ),
        decoy_keys=(
            "instrument_panel_KeysPiano_electric_piano",
            "struck_keys_score",
            "woodwind_sax_score",
            "instrument_panel_Woodwinds_sax",
            "voice_score",
            "plucked_string_authority_score",
        ),
        support_tokens=("synth", "pad"),
        decoy_tokens=("keys", "piano", "sax", "woodwind", "voice", "guitar"),
    ),
    CalibrationTargetSpec(
        target="mixed_instrument_loop",
        claim_type="role",
        source_specific_keys=(),
        role_structure_keys=(
            "role_loop_score",
            "role_phrase_score",
            "mixed_instrument_loop_role_score",
            "instrument_panel_MixedInstrument_loop",
            "instrument_subpanel_role_loop_score",
        ),
        decoy_keys=(
            "instrument_panel_Woodwinds_sax",
            "instrument_panel_KeysPiano_electric_piano",
            "instrument_panel_Synth_synth_pad",
            "instrument_panel_PluckedString_guitar_loop",
            "voice_score",
        ),
        support_tokens=("instrument loops", "mixed musical", "multi instrument"),
        decoy_tokens=("sax", "woodwind", "piano", "keys", "synth", "guitar", "voice"),
        safe_block_threshold=0.78,
        safe_broaden_threshold=0.68,
        min_block_margin=0.10,
        min_broaden_margin=0.04,
    ),
)


def build_voter_calibration_panels(
    *,
    claims: Iterable[ConsensusClaim],
    facts: SharedAudioFacts | None = None,
) -> list[VoterCalibrationPanel]:
    """Build diagnostic calibration panels for weak authority zones.

    The returned panels are evidence summaries only.  They are safe to write to
    debug traces, status reports, or future calibration reports.  They must not
    be used as a direct final-folder override.
    """
    fact_values = flatten_fact_values(facts)
    claim_rows = tuple(claims or ())
    panels: list[VoterCalibrationPanel] = []
    for spec in CALIBRATION_TARGETS:
        panels.append(build_one_panel(spec, fact_values, claim_rows))
    return panels


def build_one_panel(
    spec: CalibrationTargetSpec,
    fact_values: dict[str, float],
    claims: tuple[ConsensusClaim, ...],
) -> VoterCalibrationPanel:
    """Build one calibration panel from facts and current internal claims."""
    source_pairs = best_key_scores(fact_values, spec.source_specific_keys)
    role_pairs = best_key_scores(fact_values, spec.role_structure_keys)
    positive_pairs = tuple(list(source_pairs) + list(role_pairs))
    decoy_pairs = best_key_scores(fact_values, spec.decoy_keys)
    supporting_claims = matching_claim_sources(claims, spec.support_tokens)
    decoy_claims = matching_claim_sources(claims, spec.decoy_tokens)
    source_evidence_support = max((score for _, score in source_pairs), default=0.0)
    role_evidence_support = max((score for _, score in role_pairs), default=0.0)
    decoy_pressure = max((score for _, score in decoy_pairs), default=0.0)
    claim_support = claim_support_score(supporting_claims)
    claim_decoy = claim_support_score(decoy_claims)
    source_specific_support = clamp01(max(source_evidence_support, claim_support))
    role_structure_support = clamp01(role_evidence_support)
    raw_support = clamp01(max(source_specific_support, role_structure_support))
    decoy_score = clamp01(max(decoy_pressure, claim_decoy))
    margin = clamp01(raw_support - decoy_score)
    source_margin = clamp01(source_specific_support - decoy_score)
    authority = infer_panel_authority(
        spec,
        raw_support,
        decoy_score,
        margin,
        source_specific_support=source_specific_support,
        source_margin=source_margin,
    )
    tier = infer_confidence_tier(raw_support, margin)
    return VoterCalibrationPanel(
        target=spec.target,
        claim_type=spec.claim_type,
        raw_support=round(raw_support, 6),
        source_specific_support=round(source_specific_support, 6),
        role_structure_support=round(role_structure_support, 6),
        decoy_pressure=round(decoy_score, 6),
        margin=round(margin, 6),
        evidence_keys=tuple(key for key, _ in positive_pairs),
        decoy_keys=tuple(key for key, _ in decoy_pairs),
        supporting_claims=supporting_claims,
        decoy_claims=decoy_claims,
        authority=authority,
        confidence_tier=tier,
        explanation=panel_explanation(spec.target, authority),
    )


def flatten_fact_values(facts: SharedAudioFacts | None) -> dict[str, float]:
    """Flatten numeric evidence values from SharedAudioFacts.

    This can be called many times while the arbiter compares candidate claims for
    one file.  The facts object is immutable for the duration of that decision,
    so cache the flattened numeric map on the facts instance instead of walking
    large nested diagnostic dictionaries repeatedly.
    """
    if facts is None:
        return {}
    cached = getattr(facts, "_voter_calibration_flat_values_cache", None)
    if isinstance(cached, dict):
        return cached
    output: dict[str, float] = {}
    for key, value in getattr(facts, "feature_values_by_name", {}).items():
        add_numeric(output, str(key), value)
    for key, value in getattr(facts, "evidence", {}).items():
        # Private/cache entries are diagnostics for Python runtime, not audio
        # evidence.  Do not recursively flatten them into panel facts.
        if str(key).startswith("_"):
            continue
        flatten_value(output, str(key), value)
    try:
        object.__setattr__(facts, "_voter_calibration_flat_values_cache", output)
    except Exception:
        pass
    return output


def flatten_value(output: dict[str, float], prefix: str, value: Any) -> None:
    """Flatten dict/list scalar values into a single numeric lookup map."""
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            child_name = str(child_key)
            flatten_value(output, child_name, child_value)
            flatten_value(output, f"{prefix}_{child_name}", child_value)
        return
    if isinstance(value, (list, tuple)):
        for index, child_value in enumerate(value):
            flatten_value(output, f"{prefix}_{index}", child_value)
        return
    add_numeric(output, prefix, value)


def add_numeric(output: dict[str, float], key: str, value: Any) -> None:
    """Add a finite numeric or boolean value to a flat fact map."""
    if isinstance(value, bool):
        output[key] = 1.0 if value else 0.0
        return
    try:
        numeric = float(value)
    except Exception:
        return
    if numeric == numeric and numeric not in {float("inf"), float("-inf")}:
        output[key] = clamp01(numeric)


def best_key_scores(values: dict[str, float], keys: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    """Return present keys sorted by strongest score."""
    pairs = [(key, clamp01(values[key])) for key in keys if key in values]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return tuple(pairs[:5])


def matching_claim_sources(
    claims: tuple[ConsensusClaim, ...],
    tokens: tuple[str, ...],
) -> tuple[str, ...]:
    """Return claim sources whose internal label path matches target tokens."""
    matches: list[str] = []
    for claim in claims:
        text = " ".join(
            [
                str(claim.family or ""),
                str(claim.sub_family or ""),
                str(claim.label or ""),
                str(claim.folder_path or ""),
            ]
        ).lower()
        if any(token in text for token in tokens):
            source = str(claim.source or "claim")
            if source not in matches:
                matches.append(source)
    return tuple(matches[:6])


def claim_support_score(matches: tuple[str, ...]) -> float:
    """Convert independent matching claim count into bounded support."""
    if not matches:
        return 0.0
    return clamp01(0.34 + 0.16 * min(4, len(matches)))


def infer_panel_authority(
    spec: CalibrationTargetSpec,
    raw_support: float,
    decoy_score: float,
    margin: float,
    *,
    source_specific_support: float,
    source_margin: float,
) -> str:
    """Return the authority tier this panel is currently allowed to hold."""
    if spec.claim_type == "role":
        if raw_support >= spec.safe_broaden_threshold and margin >= spec.min_broaden_margin:
            return "safe_to_broaden_only"
        if decoy_score > raw_support:
            return "decoy_pressure_warning"
        if raw_support >= 0.55:
            return "diagnostic_support_only"
        return "uncalibrated_insufficient_evidence"
    if source_specific_support >= spec.safe_block_threshold and source_margin >= spec.min_block_margin:
        return "safe_to_block_decoys"
    if raw_support >= spec.safe_broaden_threshold and margin >= spec.min_broaden_margin:
        return "safe_to_broaden_only"
    if raw_support >= 0.55 and decoy_score <= raw_support:
        return "diagnostic_support_only"
    if decoy_score > raw_support:
        return "decoy_pressure_warning"
    return "uncalibrated_insufficient_evidence"


def infer_confidence_tier(raw_support: float, margin: float) -> str:
    """Return a compact confidence tier for reports."""
    if raw_support >= 0.84 and margin >= 0.18:
        return "strong"
    if raw_support >= 0.72 and margin >= 0.10:
        return "moderate"
    if raw_support >= 0.55:
        return "weak"
    return "low"


def panel_explanation(target: str, authority: str) -> str:
    """Return a stable human-readable calibration explanation."""
    if authority == "safe_to_block_decoys":
        return f"{target} panel is strong enough to block competing decoy identity claims."
    if authority == "safe_to_broaden_only":
        return f"{target} panel can support broadening but should not choose a leaf by itself."
    if authority == "decoy_pressure_warning":
        return f"{target} panel is under stronger decoy pressure and must not gain authority."
    if authority == "diagnostic_support_only":
        return f"{target} panel has useful evidence but is not calibrated for authority."
    return f"{target} panel has insufficient calibrated evidence."


def clamp01(value: float) -> float:
    """Clamp a numeric score to 0..1."""
    try:
        number = float(value)
    except Exception:
        return 0.0
    if number != number:
        return 0.0
    return max(0.0, min(1.0, number))
