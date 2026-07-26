"""Shared taxonomy-label contracts for training and GUI correction flows.

The sorter is moving toward brain-owned decisions, so category labels need a
small amount of grammar hygiene before they become supervised targets.  These
helpers inspect only internal taxonomy labels supplied by the brain, GUI, or
training tree.  They never inspect input filenames or source folders as sorting
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

TOP_LEVEL_TAXONOMY_FAMILIES = frozenset({"Drums", "Instruments", "FX", "_TO_REVIEW"})
TRAINABLE_TOP_LEVEL_FAMILIES = frozenset({"Drums", "Instruments", "FX"})
STRUCTURE_TERMINALS = frozenset({"One Shots", "Loops", "Long FX", "Long Running"})


@dataclass(frozen=True)
class TaxonomyLabelContract:
    """Validation and canonicalization result for one internal taxonomy label.

    Args:
        raw_label: Label string before normalization.
        normalized_label: Slash-normalized label with empty path parts removed.
        canonical_label: Label after safe grammar canonicalization.
        valid: Whether the canonical label is usable as a taxonomy path.
        reasons: Validation or canonicalization reasons.
        structure_terminal: Terminal structure folder when present.

    Side Effects:
        None.
    """

    raw_label: str
    normalized_label: str
    canonical_label: str
    valid: bool
    reasons: tuple[str, ...]
    structure_terminal: str


def normalize_taxonomy_path(value: object) -> str:
    """Return a slash-normalized internal taxonomy label.

    Args:
        value: Raw label-like value.

    Returns:
        Normalized slash-separated label.

    Side Effects:
        None.
    """
    text = str(value or "").replace("\\", "/").strip()
    return "/".join(part.strip() for part in text.split("/") if part.strip())


def taxonomy_label_contract(value: object) -> TaxonomyLabelContract:
    """Return a grammar contract for a trainable taxonomy label.

    Args:
        value: Internal taxonomy label from GUI, brain, catalog, or training
            metadata.

    Returns:
        Contract with a canonical label and validation reasons.

    Side Effects:
        None.
    """
    normalized = normalize_taxonomy_path(value)
    canonical, reasons = canonicalize_structure_terminal(normalized)
    valid_reasons = list(reasons)
    parts = canonical.split("/") if canonical else []
    if len(parts) < 2:
        valid_reasons.append("too_shallow")
    elif parts[0] not in TOP_LEVEL_TAXONOMY_FAMILIES:
        valid_reasons.append("unknown_top_family")
    if parts and any(part in {".", ".."} for part in parts):
        valid_reasons.append("unsafe_path_part")
    if parts and "." in parts[-1]:
        valid_reasons.append("terminal_looks_like_audio_filename")
    terminal = parts[-1] if parts and parts[-1] in STRUCTURE_TERMINALS else ""
    if parts and parts[0] == "FX" and terminal == "Loops":
        valid_reasons.append("fx_loop_terminal_not_trainable")
    return TaxonomyLabelContract(
        raw_label=str(value or ""),
        normalized_label=normalized,
        canonical_label=canonical,
        valid=not any(reason in invalid_reasons() for reason in valid_reasons),
        reasons=tuple(valid_reasons),
        structure_terminal=terminal,
    )


def canonicalize_taxonomy_label(value: object) -> str:
    """Return a safe canonical internal taxonomy label."""
    return taxonomy_label_contract(value).canonical_label


def is_valid_taxonomy_contract_label(value: object) -> bool:
    """Return whether ``value`` is a valid internal taxonomy label."""
    return taxonomy_label_contract(value).valid


def is_trainable_taxonomy_label(value: object) -> bool:
    """Return whether ``value`` can be used as supervised training evidence."""
    contract = taxonomy_label_contract(value)
    if not contract.valid:
        return False
    parts = contract.canonical_label.split("/")
    return bool(parts and parts[0] in TRAINABLE_TOP_LEVEL_FAMILIES and parts[-1] in STRUCTURE_TERMINALS)


def canonicalize_structure_terminal(label: str) -> tuple[str, list[str]]:
    """Repair obvious structure/category grammar contradictions.

    Args:
        label: Slash-normalized internal taxonomy label.

    Returns:
        ``(canonical_label, reasons)``.  The repair is intentionally limited to
        labels whose category name explicitly says one-shot or loop while the
        terminal structure says the opposite.

    Side Effects:
        None.
    """
    parts = label.split("/") if label else []
    if len(parts) < 2:
        return label, []
    terminal = parts[-1]
    body_text = " ".join(parts).lower().replace("_", " ").replace("-", " ")
    if terminal not in STRUCTURE_TERMINALS:
        if explicit_loop_category_text(body_text):
            return "/".join([*parts, "Loops"]), ["inferred_explicit_loop_category_terminal"]
        if explicit_one_shot_category_text(body_text):
            return "/".join([*parts, "One Shots"]), ["inferred_explicit_one_shot_category_terminal"]
        return label, []
    body_parts = parts[:-1]
    body_text = " ".join(body_parts).lower().replace("_", " ").replace("-", " ")
    reasons: list[str] = []
    if terminal == "One Shots" and explicit_loop_category_text(body_text):
        parts[-1] = "Loops"
        reasons.append("canonicalized_loop_category_terminal")
    elif terminal == "Loops" and explicit_one_shot_category_text(body_text):
        parts[-1] = "One Shots"
        reasons.append("canonicalized_one_shot_category_terminal")
    return "/".join(parts), reasons


def explicit_loop_category_text(body_text: str) -> bool:
    """Return true when category wording names a loop role."""
    return any(
        phrase in body_text
        for phrase in (
            "drum loop",
            "drum loops",
            "guitar loop",
            "guitar loops",
            "key loop",
            "keys loop",
            "keys loops",
            "string loop",
            "string loops",
            "synth loop",
            "synth loops",
            "vocal loop",
            "vocal loops",
            "instrument loop",
            "instrument loops",
            "mixed musical loop",
            "mixed musical loops",
        )
    )


def explicit_one_shot_category_text(body_text: str) -> bool:
    """Return true when category wording names a one-shot role."""
    return any(phrase in body_text for phrase in ("one shot", "one shots"))


def invalid_reasons() -> frozenset[str]:
    """Return validation reasons that make a label unusable."""
    return frozenset(
        {
            "too_shallow",
            "unknown_top_family",
            "unsafe_path_part",
            "terminal_looks_like_audio_filename",
            "fx_loop_terminal_not_trainable",
        }
    )
