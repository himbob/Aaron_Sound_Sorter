from __future__ import annotations

from dataclasses import asdict, dataclass

# Shared public API for the componentized Stage 4 sorter.
# The original Stage 4 file had one global namespace. After splitting it into
# component modules, many functions still intentionally call helpers by their
# old unqualified names. This module wires the component namespaces together so
# each module can resolve the same public symbols without keeping a legacy blob.
from . import (
    brain,
    commands,
    committee,
    core,
    features,
    io_utils,
    preview,
    reports,
    selftests,
    training_labels,
    two_voter,
)

_MODULES = [
    core,
    io_utils,
    training_labels,
    features,
    brain,
    committee,
    two_voter,
    reports,
    preview,
    selftests,
    commands,
]

# Save original implementations before namespace wiring or test wrappers.
_ORIG_committee_agreement_choose_label = reports.committee_agreement_choose_label
_ORIG_choose_membership_safe_alternative = committee.choose_membership_safe_alternative
_ORIG_structure_conflict_warning = training_labels.structure_conflict_warning
_ORIG_repair_structure_lane_with_fingerprint = training_labels.repair_structure_lane_with_fingerprint

_PUBLIC = {}
for _module in _MODULES:
    for _name, _value in vars(_module).items():
        if _name.startswith("__"):
            continue
        _PUBLIC[_name] = _value

# Make split modules behave like the former single-file namespace.
for _module in _MODULES:
    _module.__dict__.update(_PUBLIC)
globals().update(_PUBLIC)

# Explicit aliases keep static quality checks honest while preserving the
# legacy namespace-bridge behavior above.
N_FFT = core.N_FFT
HOP = core.HOP
TARGET_SR = core.TARGET_SR
FP_SIZE = core.FP_SIZE
DEFAULT_ALLOWED_TOP = core.DEFAULT_ALLOWED_TOP


@dataclass(frozen=True)
class _BrainBuildConfig:
    n_fft: int = N_FFT
    hop: int = HOP
    target_sr: int = TARGET_SR
    fp_size: int = FP_SIZE


BRAIN_BUILD_CONFIG = _BrainBuildConfig()


def brain_runtime_config_warning(brain: dict) -> str:
    cfg = dict(brain.get("build_config") or {})
    expected = asdict(BRAIN_BUILD_CONFIG)
    for key, value in expected.items():
        if key in cfg and cfg.get(key) != value:
            return f"runtime_config_mismatch:{key}:brain={cfg.get(key)} expected={value}"
    return ""


def choose_broad_family_candidate_top(
    predicted_top: str,
    learned_top: str,
    learned_gap: float = 0.0,
    allowed_top=DEFAULT_ALLOWED_TOP,
    min_gap: float = 0.0,
    conflict_override_gap=None,
    review_reason: str = "",
):
    pred = str(predicted_top or "").strip()
    learned = str(learned_top or "").strip()
    if learned not in allowed_top:
        return "", "no_learned_top_candidate"
    if float(learned_gap) < float(min_gap):
        return "", f"learned_top_gap_below_min {float(learned_gap):.3f} < {float(min_gap):.3f}"
    if learned == pred:
        return learned, f"learned_top_candidate:{learned};gap={float(learned_gap):.3f}"
    reason_text = str(review_reason or "").lower()
    if (
        pred == "Drums"
        and learned in allowed_top
        and learned != pred
        and (
            "unsupported_drum" in reason_text
            or "drum_loop_rejected_by_non_drum_physics" in reason_text
            or "unsupported drum loop" in reason_text
        )
    ):
        return learned, (
            f"learned_top_used_after_raw_drum_rejected_by_physics raw_top={pred} "
            f"learned_top={learned} gap={float(learned_gap):.3f}"
        )
    if conflict_override_gap is not None and float(learned_gap) >= float(conflict_override_gap):
        return learned, f"learned_top_strong_override:{pred}->{learned};gap={float(learned_gap):.3f}"
    return (
        pred,
        f"fallback_to_raw_top:raw_exact_label_top_wins:predicted_top={pred};learned_top={learned};learned_gap={float(learned_gap):.3f}",
    )


def _sync_component_globals_from(api_globals: dict) -> None:
    # Do not overwrite these wrapper names inside implementation modules, or calls recurse.
    skip = {
        "committee_agreement_choose_label",
        "choose_membership_safe_alternative",
        "structure_conflict_warning",
        "repair_structure_lane_with_fingerprint",
    }
    for _m in _MODULES:
        for k, v in api_globals.items():
            if k not in skip:
                _m.__dict__[k] = v


def committee_agreement_choose_label(
    brain,
    fingerprint,
    candidate_rows=None,
    current_label="",
    top_scores=None,
    structure_scores=None,
    feature_names_list=None,
    rival_contrast_facts=None,
    **kwargs,
):
    rows = kwargs.get("rows", candidate_rows)
    if rows is None:
        rows = []
    _sync_component_globals_from(globals())
    return _ORIG_committee_agreement_choose_label(
        brain,
        fingerprint,
        rows,
        current_label,
        feature_names_list or [],
        top_scores or [],
        structure_scores or [],
        rival_contrast_facts or {},
    )


def choose_membership_safe_alternative(brain, fingerprint, current_label, top5, current_ev):
    _sync_component_globals_from(globals())
    return _ORIG_choose_membership_safe_alternative(brain, fingerprint, current_label, top5, current_ev)


def structure_conflict_warning(row, group_key, structure):
    if str(group_key).endswith("/_LOOPS") or "/_LOOPS/" in str(group_key):
        return ""
    return _ORIG_structure_conflict_warning(row, group_key, structure)


def repair_structure_lane_with_fingerprint(base_structure: str, fingerprint, duration_sec: float, group_key: str = ""):
    if base_structure == "loop" and not group_key:
        return ("loop", "")
    return _ORIG_repair_structure_lane_with_fingerprint(base_structure, fingerprint, duration_sec, group_key)


_PUBLIC.update(
    {
        "asdict": asdict,
        "BRAIN_BUILD_CONFIG": BRAIN_BUILD_CONFIG,
        "brain_runtime_config_warning": brain_runtime_config_warning,
        "choose_broad_family_candidate_top": choose_broad_family_candidate_top,
        "committee_agreement_choose_label": committee_agreement_choose_label,
        "choose_membership_safe_alternative": choose_membership_safe_alternative,
        "structure_conflict_warning": structure_conflict_warning,
        "repair_structure_lane_with_fingerprint": repair_structure_lane_with_fingerprint,
    }
)

globals().update(_PUBLIC)
for _module in _MODULES:
    _module.__dict__.update(_PUBLIC)

__all__ = sorted(k for k in _PUBLIC if not k.startswith("__"))

# Re-export private helper names too because the old single-file tests imported
# the runner as a module and intentionally asserted private tournament helpers.
__all__ = [k for k in globals() if not k.startswith("__")]
