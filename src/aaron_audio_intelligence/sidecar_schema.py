"""Versioned JSON sidecar schema for reusable audio intelligence evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aaron_audio_intelligence.owner_brains import OwnerBrainResult

AAI_SIDECAR_SCHEMA_VERSION = "aai_sidecar_v1"


@dataclass(frozen=True)
class AudioIntelligenceSidecar:
    """Reusable evidence sidecar for one analyzed audio file.

    Args:
        schema_version: Sidecar schema identifier.
        file_id: Stable content or path-derived ID for the file.
        source_path: Source path for audit/display only.
        analysis_version: Sorter or analysis version string.
        final_sort: Product adapter's final folder decision.
        roles: Owner/role evidence and blockers.
        source_like: Source identity evidence, independent of final folder.
        usable_as: Product-use role evidence.
        shapes: Shape/structure evidence.
        music: Read-only music property evidence.
        components: Read-only component inventory evidence.
        traits: Human-readable traits.
        warnings: Analysis warnings and safety notes.
        training: Suggested positive/negative learning targets.
        debug_refs: Pointers to heavier debug packets or manifests.
    """

    schema_version: str
    file_id: str
    source_path: str
    analysis_version: str
    final_sort: Mapping[str, Any]
    roles: Mapping[str, Any] = field(default_factory=dict)
    source_like: Mapping[str, float] = field(default_factory=dict)
    usable_as: Mapping[str, float] = field(default_factory=dict)
    shapes: Mapping[str, Any] = field(default_factory=dict)
    music: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, Any] = field(default_factory=dict)
    traits: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    training: Mapping[str, Any] = field(default_factory=dict)
    debug_refs: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return asdict(self)


def build_sidecar_from_sort_result(
    result: Any,
    *,
    owner_results: Sequence[OwnerBrainResult] = (),
    analysis_version: str = "sorter_current",
    debug_packet_ref: str = "",
) -> AudioIntelligenceSidecar:
    """Build a sidecar from the current sorter result without changing routing.

    Args:
        result: ``SortFileResult``-like object from the sorter.
        owner_results: Optional read-only trainable owner-brain results.
        analysis_version: Version string written into the sidecar.
        debug_packet_ref: Optional pointer to the heavier debug JSONL row/source.

    Returns:
        A versioned sidecar that separates final folder, source-like evidence,
        owner evidence, shapes, music placeholders, and training hints.
    """
    evidence = getattr(getattr(result, "facts", None), "evidence", {}) or {}
    decision = result.decision
    source_path = Path(result.source_path)
    owner_payload = {owner.owner_name: owner.as_dict() for owner in owner_results}
    primary_owner = primary_owner_name(owner_results)
    roles = {
        "primary_owner": primary_owner,
        "owner_scores": owner_payload,
        "blocked_by": role_blockers(owner_results),
    }
    sidecar = AudioIntelligenceSidecar(
        schema_version=AAI_SIDECAR_SCHEMA_VERSION,
        file_id=file_id_for_path(source_path),
        source_path=str(source_path),
        analysis_version=analysis_version,
        final_sort={
            "folder": str(getattr(decision, "folder_path", "")),
            "label": str(getattr(decision, "final_label", "")),
            "top": str(getattr(decision, "final_top", "")),
            "status": str(getattr(decision, "consensus_status", "")),
            "reason": str(getattr(decision, "reason", "")),
        },
        roles=roles,
        source_like=source_like_scores(evidence),
        usable_as=usable_as_scores(evidence),
        shapes=shape_payload(evidence),
        music=music_placeholders(evidence),
        components=component_placeholders(evidence),
        traits=trait_summary(evidence),
        warnings=warning_summary(evidence, owner_results),
        training=training_hints(owner_results),
        debug_refs={"debug_packet": debug_packet_ref} if debug_packet_ref else {},
    )
    return sidecar


def file_id_for_path(path: Path) -> str:
    """Return a stable file signature without re-reading audio content.

    The sorter has already paid the audio-loading cost before sidecars are
    written. This ID intentionally uses path and stat metadata so the sidecar
    layer does not add another full-file read across large sample folders.
    """
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        signature = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
        return "file-signature-sha256:" + hashlib.sha256(signature.encode("utf-8")).hexdigest()
    except OSError:
        return "path-sha256:" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def primary_owner_name(owner_results: Sequence[OwnerBrainResult]) -> str:
    """Return the highest-confidence owner name, if any."""
    if not owner_results:
        return ""
    best = max(owner_results, key=lambda result: (result.confidence, result.score))
    return best.owner_name


def role_blockers(owner_results: Sequence[OwnerBrainResult]) -> list[str]:
    """Return owner names whose negative evidence outweighs positive evidence."""
    return [result.owner_name for result in owner_results if result.score < 0 and result.confidence > 0]


def source_like_scores(evidence: Mapping[str, Any]) -> dict[str, float]:
    """Extract source-like evidence from existing sorter diagnostics."""
    keys = {
        "voice": "voice_score",
        "sax": "woodwind_sax_score",
        "flute": "woodwind_flute_score",
        "guitar_acoustic": "guitar_acoustic_score",
        "guitar_electric": "guitar_electric_score",
        "piano_keys": "struck_keys_score",
        "synth": "synth_tonal_source_score",
        "bass": "low_end_source_score",
        "drums": "drum_hit_score",
        "drum_loop": "drum_loop_source_score",
        "fx_motion": "fx_motion_score",
        "texture_bed": "texture_bed_score",
    }
    return compact_score_map(evidence, keys)


def usable_as_scores(evidence: Mapping[str, Any]) -> dict[str, float]:
    """Extract product-use role scores from existing sorter diagnostics."""
    keys = {
        "loop": "role_loop_score",
        "one_shot": "role_one_shot_score",
        "phrase": "role_phrase_score",
        "transition_fx": "fx_transition_authority_score",
        "texture_bed": "texture_bed_score",
        "low_end_foundation": "low_end_source_score",
    }
    return compact_score_map(evidence, keys)


def compact_score_map(evidence: Mapping[str, Any], keys: Mapping[str, str]) -> dict[str, float]:
    """Return rounded numeric scores for keys present in evidence."""
    flat = nested_flat_evidence(evidence)
    scores: dict[str, float] = {}
    for public_name, evidence_key in keys.items():
        value = flat.get(evidence_key, evidence.get(evidence_key, ""))
        try:
            scores[public_name] = round(float(value), 6)
        except (TypeError, ValueError):
            continue
    return scores


def nested_flat_evidence(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the low-level physics flat map when present."""
    subpanels = evidence.get("physics_subpanels", {})
    if isinstance(subpanels, Mapping):
        flat = subpanels.get("flat", {})
        if isinstance(flat, Mapping):
            return flat
    return {}


def shape_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return shape evidence in sidecar form."""
    shape = evidence.get("shape_vote", {})
    if isinstance(shape, Mapping):
        return dict(shape)
    return {}


def music_placeholders(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return read-only music fields, using not-applicable defaults."""
    return {
        "bpm": evidence.get("bpm", None),
        "bpm_confidence": evidence.get("bpm_confidence", 0.0),
        "tempo_candidates": evidence.get("tempo_candidates", []),
        "key": evidence.get("key", None),
        "key_confidence": evidence.get("key_confidence", 0.0),
        "chords": evidence.get("chords", []),
        "policy": "read_only_placeholder_until_music_property_phase",
    }


def component_placeholders(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return component inventory placeholders."""
    return {
        "drums": evidence.get("component_drums", {}),
        "instruments": evidence.get("component_instruments", {}),
        "fx": evidence.get("component_fx", {}),
        "ambience": evidence.get("component_ambience", {}),
        "policy": "read_only_placeholder_until_component_phase",
    }


def trait_summary(evidence: Mapping[str, Any]) -> list[str]:
    """Return compact trait strings from obvious existing scores."""
    source_scores = source_like_scores(evidence)
    usable_scores = usable_as_scores(evidence)
    traits = [f"source_like:{name}" for name, score in source_scores.items() if score >= 0.65]
    traits.extend(f"usable_as:{name}" for name, score in usable_scores.items() if score >= 0.65)
    return traits


def warning_summary(evidence: Mapping[str, Any], owner_results: Sequence[OwnerBrainResult]) -> list[str]:
    """Return sidecar warnings from conflict and owner evidence."""
    warnings: list[str] = []
    if evidence.get("harmonic_core_recall_enabled"):
        warnings.append("harmonic_core_recall_enabled")
    if role_blockers(owner_results):
        warnings.append("owner_negative_evidence_present")
    return warnings


def training_hints(owner_results: Sequence[OwnerBrainResult]) -> dict[str, Any]:
    """Return suggested positive/negative owner labels for future training."""
    if not owner_results:
        return {
            "safe_to_auto_train": False,
            "suggested_positive_owner": "",
            "suggested_negative_owners": [],
        }
    primary = primary_owner_name(owner_results)
    negatives = [result.owner_name for result in owner_results if result.score < 0]
    return {
        "safe_to_auto_train": False,
        "suggested_positive_owner": primary,
        "suggested_negative_owners": negatives,
    }
