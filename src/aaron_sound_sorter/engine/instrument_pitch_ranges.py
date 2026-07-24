# SOURCE-NAME BLINDNESS INVARIANT:
# This module may inspect only internal taxonomy/category labels and measured
# pitch facts. It must never inspect producer filenames, source folders, ZIP
# member names, sample-pack labels, or path tokens from the input file.
"""Measured pitch-range guards for narrow instrument category claims.

The sorter is brain-first, but a brain should not be allowed to place a sound
into a narrow acoustic instrument leaf when the measured fundamental is outside
the physically plausible range for that instrument family.  These rules are
conservative proof guards, not source identifiers: they only reject impossible
narrow leaves when pitch tracking is confident.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import _feature_number_from_facts, _norm_path
from aaron_sound_sorter.engine.measured_source_contracts import supports_clean_low_bass_phrase_owner


@dataclass(frozen=True)
class PitchRangeEvidence:
    """Confident measured fundamental-frequency evidence for one sound.

    Attributes:
        observed_hz: Best source-blind fundamental estimate in Hertz.
        confidence: Best pitch confidence score available from the feature
            extractor or loop note-event analysis.
        voiced_ratio: Fraction of frames treated as voiced by the pitch tracker.
        source: Name of the feature used for ``observed_hz``.

    Side Effects:
        None.
    """

    observed_hz: float
    confidence: float
    voiced_ratio: float
    source: str

    @property
    def is_trusted(self) -> bool:
        """Return True when the measured pitch is reliable enough to guard."""
        if self.observed_hz <= 0.0:
            return False
        if self.confidence >= 0.82 and self.voiced_ratio >= 0.24:
            return True
        return bool(self.confidence >= 0.72 and self.voiced_ratio >= 0.50)


@dataclass(frozen=True)
class InstrumentPitchRange:
    """Conservative pitch range metadata for one internal instrument family.

    Args:
        family_name: Human-readable internal instrument-family name.
        path_markers: Internal taxonomy path fragments covered by this rule.
        normal_min_hz: Published or common musical lower bound used only for
            diagnostics.
        normal_max_hz: Published or common musical upper bound used only for
            diagnostics.
        hard_min_hz: Wider lower bound.  Only below this value becomes a hard
            review claim.
        hard_max_hz: Wider upper bound.  Only above this value becomes a hard
            review claim.

    Side Effects:
        None.
    """

    family_name: str
    path_markers: tuple[str, ...]
    normal_min_hz: float
    normal_max_hz: float
    hard_min_hz: float
    hard_max_hz: float

    def matches_internal_path(self, folder_path: str) -> bool:
        """Return True when an internal taxonomy label belongs to this rule."""
        normalized = _norm_path(folder_path)
        return normalized.startswith("instruments/") and any(marker in normalized for marker in self.path_markers)


@dataclass(frozen=True)
class PitchRangeCheck:
    """Result of comparing measured pitch evidence to one instrument rule.

    Attributes:
        rule: Matched internal instrument-family rule.
        evidence: Trusted measured pitch evidence, when available.
        status: ``unknown``, ``inside``, ``soft_outside``,
            ``probable_harmonic_alias``, or ``hard_outside``.
        semitones_outside: Distance outside the normal range.  Zero means the
            pitch is inside the common range or no trusted pitch exists.

    Side Effects:
        None.
    """

    rule: InstrumentPitchRange | None
    evidence: PitchRangeEvidence | None
    status: str
    semitones_outside: float = 0.0

    @property
    def is_hard_violation(self) -> bool:
        """Return True when a narrow instrument leaf should not auto-place."""
        return self.status == "hard_outside"


INSTRUMENT_PITCH_RANGES: tuple[InstrumentPitchRange, ...] = (
    InstrumentPitchRange(
        family_name="Bass",
        path_markers=("instruments/bass/",),
        normal_min_hz=41.20,
        normal_max_hz=392.00,
        hard_min_hz=24.50,
        hard_max_hz=880.00,
    ),
    InstrumentPitchRange(
        family_name="Saxophone",
        path_markers=("instruments/woodwinds/saxophone",),
        normal_min_hz=58.27,
        normal_max_hz=1174.66,
        hard_min_hz=43.65,
        hard_max_hz=1760.00,
    ),
    InstrumentPitchRange(
        family_name="Flute",
        path_markers=("instruments/woodwinds/flute",),
        normal_min_hz=261.63,
        normal_max_hz=2349.32,
        hard_min_hz=174.61,
        hard_max_hz=3520.00,
    ),
    InstrumentPitchRange(
        family_name="Clarinet/Bassoon/Oboe",
        path_markers=(
            "instruments/woodwinds/clarinet",
            "instruments/woodwinds/bassoon",
            "instruments/woodwinds/oboe",
        ),
        normal_min_hz=58.27,
        normal_max_hz=1567.98,
        hard_min_hz=38.89,
        hard_max_hz=2637.02,
    ),
    InstrumentPitchRange(
        family_name="Brass",
        path_markers=("instruments/brass/", "instruments/brass and woodwinds/stabs"),
        normal_min_hz=43.65,
        normal_max_hz=1174.66,
        hard_min_hz=29.14,
        hard_max_hz=2093.00,
    ),
    InstrumentPitchRange(
        family_name="Guitar",
        path_markers=("instruments/guitar/",),
        normal_min_hz=82.41,
        normal_max_hz=1318.51,
        hard_min_hz=55.00,
        hard_max_hz=2637.02,
    ),
    InstrumentPitchRange(
        family_name="Plucked Strings",
        path_markers=("instruments/plucked strings/",),
        normal_min_hz=73.42,
        normal_max_hz=2093.00,
        hard_min_hz=43.65,
        hard_max_hz=3520.00,
    ),
    InstrumentPitchRange(
        family_name="Bowed Strings",
        path_markers=("instruments/strings bowed/", "instruments/strings/"),
        normal_min_hz=65.41,
        normal_max_hz=3135.96,
        hard_min_hz=32.70,
        hard_max_hz=4186.01,
    ),
    InstrumentPitchRange(
        family_name="Keys",
        path_markers=("instruments/keys/",),
        normal_min_hz=27.50,
        normal_max_hz=4186.01,
        hard_min_hz=20.60,
        hard_max_hz=6271.93,
    ),
    InstrumentPitchRange(
        family_name="Mallets and Bells",
        path_markers=("instruments/mallets and bells/",),
        normal_min_hz=65.41,
        normal_max_hz=4186.01,
        hard_min_hz=43.65,
        hard_max_hz=7040.00,
    ),
    InstrumentPitchRange(
        family_name="Voice",
        path_markers=("instruments/voice/",),
        normal_min_hz=65.41,
        normal_max_hz=1046.50,
        hard_min_hz=43.65,
        hard_max_hz=2093.00,
    ),
)


def pitch_range_for_internal_path(folder_path: str) -> InstrumentPitchRange | None:
    """Return the pitch-range rule for a narrow internal instrument path.

    Args:
        folder_path: Internal taxonomy/category label path from a voter or
            claim. This is not an input source path.

    Returns:
        Matching conservative range rule, or ``None`` for broad/unknown paths.

    Side Effects:
        None.
    """
    normalized = _norm_path(folder_path)
    if not normalized.startswith("instruments/"):
        return None
    broad_markers = (
        "instrument loops",
        "mixed musical loops",
        "brass and woodwinds/loops",
        "world and special instruments",
    )
    if any(marker in normalized for marker in broad_markers):
        return None
    for range_rule in INSTRUMENT_PITCH_RANGES:
        if range_rule.matches_internal_path(normalized):
            return range_rule
    return None


def pitch_range_evidence_from_facts(facts: SharedAudioFacts | None) -> PitchRangeEvidence | None:
    """Return the best measured F0 evidence available in shared audio facts.

    Args:
        facts: Shared source-blind audio measurements.

    Returns:
        A pitch-evidence object when an F0 estimate exists, otherwise ``None``.

    Side Effects:
        None.
    """
    observed_hz = _feature_number_from_facts(facts, "f0_median_hz")
    source = "f0_median_hz"
    if observed_hz <= 0.0:
        observed_hz = _feature_number_from_facts(facts, "librosa_yin_f0_median_hz")
        source = "librosa_yin_f0_median_hz"
    if observed_hz <= 0.0:
        return None
    confidence = max(
        _feature_number_from_facts(facts, "pitch_confidence"),
        _feature_number_from_facts(facts, "body_pitch_confidence"),
        _feature_number_from_facts(facts, "loop_mean_event_pitch_confidence"),
    )
    voiced_ratio = max(
        _feature_number_from_facts(facts, "f0_voiced_ratio"),
        _feature_number_from_facts(facts, "librosa_yin_voiced_ratio"),
    )
    return PitchRangeEvidence(
        observed_hz=observed_hz,
        confidence=confidence,
        voiced_ratio=voiced_ratio,
        source=source,
    )


def check_instrument_pitch_range(folder_path: str, facts: SharedAudioFacts | None) -> PitchRangeCheck:
    """Compare one internal instrument claim with confident measured pitch.

    Args:
        folder_path: Internal taxonomy/category label path proposed for final
            placement.
        facts: Shared source-blind audio measurements.

    Returns:
        Pitch-range check result.  ``hard_outside`` means the proposed narrow
        leaf is physically implausible and should review instead of auto-place.

    Side Effects:
        None.
    """
    range_rule = pitch_range_for_internal_path(folder_path)
    if range_rule is None:
        return PitchRangeCheck(rule=None, evidence=None, status="unknown")
    evidence = pitch_range_evidence_from_facts(facts)
    if evidence is None or not evidence.is_trusted:
        return PitchRangeCheck(rule=range_rule, evidence=evidence, status="unknown")
    observed_hz = evidence.observed_hz
    if observed_hz < range_rule.hard_min_hz or observed_hz > range_rule.hard_max_hz:
        if _bass_hard_high_pitch_is_probable_harmonic_alias(range_rule, observed_hz, facts):
            return PitchRangeCheck(
                rule=range_rule,
                evidence=evidence,
                status="probable_harmonic_alias",
                semitones_outside=_semitones_outside_normal_range(
                    observed_hz,
                    range_rule.normal_min_hz,
                    range_rule.normal_max_hz,
                ),
            )
        return PitchRangeCheck(
            rule=range_rule,
            evidence=evidence,
            status="hard_outside",
            semitones_outside=_semitones_outside_normal_range(
                observed_hz,
                range_rule.normal_min_hz,
                range_rule.normal_max_hz,
            ),
        )
    if observed_hz < range_rule.normal_min_hz or observed_hz > range_rule.normal_max_hz:
        return PitchRangeCheck(
            rule=range_rule,
            evidence=evidence,
            status="soft_outside",
            semitones_outside=_semitones_outside_normal_range(
                observed_hz,
                range_rule.normal_min_hz,
                range_rule.normal_max_hz,
            ),
        )
    return PitchRangeCheck(rule=range_rule, evidence=evidence, status="inside")


def _bass_hard_high_pitch_is_probable_harmonic_alias(
    range_rule: InstrumentPitchRange,
    observed_hz: float,
    facts: SharedAudioFacts | None,
) -> bool:
    """Return True when high Bass F0 is likely overtone tracking.

    Bass loops often expose stronger upper harmonics than fundamentals, so a
    pitch tracker can report a high partial while measured low-band energy and
    learned Bass memory are both correct. This is a brain-first escape hatch
    for that case, not a general Bass rescue.
    """
    if facts is None or range_rule.family_name != "Bass" or observed_hz <= range_rule.hard_max_hz:
        return False
    low_peak_hz = _feature_number_from_facts(facts, "low_peak_frequency_hz")
    if not range_rule.hard_min_hz <= low_peak_hz <= range_rule.normal_max_hz:
        return False
    if not _has_bass_owner_support(facts):
        return False
    if _trusted_low_yin_supports_bass_range(range_rule, low_peak_hz, facts):
        return True
    if supports_clean_low_bass_phrase_owner(facts):
        return True
    bass_body = max(
        _measured_score(facts, "bass_synth_score"),
        _measured_score(facts, "bass_sub_score"),
        _measured_score(facts, "bass_electric_score"),
        _measured_score(facts, "bass_808_score"),
        _measured_score(facts, "low_end_source_score"),
    )
    low_ratio = max(
        _feature_number_from_facts(facts, "low_event_ratio"),
        _feature_number_from_facts(facts, "loop_mean_event_low_ratio"),
        _role_evidence_number(facts, "low_total"),
    )
    high_ratio = max(
        _feature_number_from_facts(facts, "high_event_ratio"),
        _feature_number_from_facts(facts, "loop_mean_event_high_ratio"),
        _role_evidence_number(facts, "high_total"),
    )
    pitched_ratio = max(
        _feature_number_from_facts(facts, "pitched_event_ratio"),
        _feature_number_from_facts(facts, "loop_pitched_event_ratio"),
        _role_evidence_number(facts, "loop_pitched_event_ratio"),
    )
    tonal_ratio = max(
        _feature_number_from_facts(facts, "sustained_tonal_frame_ratio"),
        _feature_number_from_facts(facts, "loop_sustained_tonal_frame_ratio"),
        _feature_number_from_facts(facts, "non_event_tonal_ratio"),
        _role_evidence_number(facts, "loop_sustained_tonal_frame_ratio"),
        _role_evidence_number(facts, "loop_non_event_tonal_ratio"),
    )
    drum_body = max(
        _measured_score(facts, "drum_hit_score"),
        _measured_score(facts, "drum_loop_source_score"),
        _measured_score(facts, "drum_snare_source_score"),
        _measured_score(facts, "drum_clap_source_score"),
        _measured_score(facts, "drum_rim_stick_source_score"),
        _measured_score(facts, "compact_struck_tonal_percussion_score"),
    )
    fx_motion = max(
        _measured_score(facts, "fx_motion_score"),
        _measured_score(facts, "fx_transition_authority_score"),
        _measured_score(facts, "fx_riser_build_score"),
        _measured_score(facts, "fx_whoosh_sweep_score"),
    )
    return bool(
        bass_body >= 0.62
        and low_ratio >= 0.78
        and high_ratio <= 0.16
        and pitched_ratio >= 0.80
        and tonal_ratio >= 0.70
        and drum_body <= 0.48
        and fx_motion <= 0.46
    )


def _trusted_low_yin_supports_bass_range(
    range_rule: InstrumentPitchRange,
    low_peak_hz: float,
    facts: SharedAudioFacts,
) -> bool:
    """Return True when YIN and the low spectral peak contradict a high alias.

    The cheap frame autocorrelation tracker can lock onto a strong upper
    harmonic in plucked Bass material. A normal-range YIN estimate is enough
    to make that high reading non-conclusive when it also agrees with the
    measured low-band peak and a trained Bass owner.
    """
    yin_hz = _feature_number_from_facts(facts, "librosa_yin_f0_median_hz")
    yin_voiced_ratio = _feature_number_from_facts(facts, "librosa_yin_voiced_ratio")
    if not range_rule.normal_min_hz <= yin_hz <= range_rule.normal_max_hz:
        return False
    if yin_voiced_ratio < 0.24:
        return False
    ratio = max(yin_hz, low_peak_hz) / max(1e-9, min(yin_hz, low_peak_hz))
    return bool(ratio <= 2.05)


def _has_bass_owner_support(facts: SharedAudioFacts) -> bool:
    """Return True when a trained owner/memory lane points at Bass."""
    evidence = getattr(facts, "evidence", {})
    if not isinstance(evidence, dict):
        return False
    if _memory_supports_bass(evidence.get("learned_physics_memory"), minimum_confidence=0.78):
        return True
    if _memory_supports_bass(evidence.get("learned_voter_memory"), minimum_confidence=0.86):
        return True
    brain_vote = evidence.get("brain_ensemble_vote_1")
    if isinstance(brain_vote, str):
        return _norm_path(brain_vote).startswith("instruments/bass/")
    if isinstance(brain_vote, dict):
        label = str(brain_vote.get("folder_path") or brain_vote.get("label") or "")
        confidence = _safe_float(brain_vote.get("confidence"))
        support = _safe_float(brain_vote.get("support"))
        return bool(_norm_path(label).startswith("instruments/bass/") and (confidence >= 0.72 or support >= 0.50))
    return False


def _measured_score(facts: SharedAudioFacts, key: str) -> float:
    """Return a measured feature or flattened Physics subpanel score."""
    score = _feature_number_from_facts(facts, key)
    evidence = getattr(facts, "evidence", {})
    panels = evidence.get("physics_subpanels") if isinstance(evidence, dict) else {}
    flat = panels.get("flat") if isinstance(panels, dict) else {}
    if isinstance(flat, dict):
        score = max(score, _safe_float(flat.get(key)))
    return score


def _memory_supports_bass(memory: object, *, minimum_confidence: float) -> bool:
    """Return True when one learned-memory diagnostic owns Bass."""
    if not isinstance(memory, dict) or not bool(memory.get("matched")):
        return False
    label = _norm_path(str(memory.get("label") or ""))
    target = _norm_path(str(memory.get("target_key") or ""))
    branch = str(memory.get("branch") or "").strip().lower()
    confidence = _safe_float(memory.get("confidence"))
    return bool(
        confidence >= minimum_confidence
        and (
            label.startswith("instruments/bass/")
            or target == "instruments/bass"
            or branch == "bass"
            or str(memory.get("role") or "").strip().lower() == "instrument_bass_loop"
        )
    )


def _role_evidence_number(facts: SharedAudioFacts, key: str) -> float:
    """Return a numeric value from measured role diagnostic evidence."""
    evidence = getattr(facts, "evidence", {})
    roles = evidence.get("measured_roles") if isinstance(evidence, dict) else {}
    role_evidence = roles.get("evidence") if isinstance(roles, dict) else {}
    if not isinstance(role_evidence, dict):
        return 0.0
    return _safe_float(role_evidence.get(key))


def _safe_float(value: object) -> float:
    """Return ``value`` as a finite-ish float with malformed values as zero."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _semitones_outside_normal_range(observed_hz: float, min_hz: float, max_hz: float) -> float:
    """Return absolute semitone distance outside a normal range."""
    if observed_hz <= 0.0:
        return 0.0
    if observed_hz < min_hz:
        return abs(12.0 * log2(observed_hz / min_hz))
    if observed_hz > max_hz:
        return abs(12.0 * log2(observed_hz / max_hz))
    return 0.0
