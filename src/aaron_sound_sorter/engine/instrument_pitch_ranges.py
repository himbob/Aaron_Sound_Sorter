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
        status: ``unknown``, ``inside``, ``soft_outside``, or ``hard_outside``.
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


def _semitones_outside_normal_range(observed_hz: float, min_hz: float, max_hz: float) -> float:
    """Return absolute semitone distance outside a normal range."""
    if observed_hz <= 0.0:
        return 0.0
    if observed_hz < min_hz:
        return abs(12.0 * log2(observed_hz / min_hz))
    if observed_hz > max_hz:
        return abs(12.0 * log2(observed_hz / max_hz))
    return 0.0
