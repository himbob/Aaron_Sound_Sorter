"""Sort output writer for manifests, summaries, folders, and ZIPs."""

from __future__ import annotations

import csv
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

from aaron_sound_sorter.domain.models import SortFileResult, SortSummary
from aaron_sound_sorter.domain.physics_category_panels import ALL_CATEGORY_SPECS
from aaron_sound_sorter.infrastructure.brain_lane_validation import write_brain_lane_validation_reports

SORTED_ROOT_NAME = "Aaron_Sorted_Sounds"
DEBUG_PACKET_SIDECAR_NAME = f"{SORTED_ROOT_NAME}_debug_packets.jsonl"

CATEGORY_PANEL_MANIFEST_FIELDS = [spec.score_key for spec in ALL_CATEGORY_SPECS]

PHYSICS_SUBPANEL_MANIFEST_FIELDS = [
    "role_loop_score",
    "role_one_shot_score",
    "role_phrase_score",
    "plucked_string_score",
    "plucked_string_authority_score",
    "reed_wind_score",
    "reed_wind_authority_score",
    "struck_keys_score",
    "struck_keys_authority_score",
    "synth_tonal_source_score",
    "bowed_string_score",
    "voice_score",
    "drum_hit_score",
    "drum_loop_source_score",
    "compact_struck_tonal_percussion_score",
    "hand_drum_membrane_score",
    "pitched_metal_percussion_score",
    "struck_wood_score",
    "pitched_mallet_instrument_score",
    "struck_percussion_guard_exception",
    "pitched_metal_material_evidence",
    "hand_drum_material_evidence",
    "struck_wood_material_evidence",
    "metallic_noise_score",
    "scrape_rasp_score",
    "fx_motion_score",
    "fx_transition_authority_score",
    "tonal_alert_siren_score",
    "texture_bed_score",
    "low_end_source_score",
    "physics_subpanel_event_count",
    "physics_subpanel_clean_tone",
    "physics_subpanel_noisy_air",
    "onset_percussive_onset_score",
    "onset_pitched_onset_score",
    "onset_scrape_onset_score",
    "onset_swell_onset_score",
    "onset_echo_tail_likelihood",
    "onset_true_repetition_likelihood",
    "loop_tempo_confidence",
    "loop_pulse_clarity",
    "loop_onset_periodicity",
    "loop_delay_tail_likelihood",
    "plucked_pluck_attack_score",
    "plucked_string_decay_score",
    "plucked_harmonic_stack_score",
    "plucked_pick_noise_score",
    "reed_reed_noise_score",
    "reed_formant_envelope_score",
    "reed_breath_attack_score",
    "keys_hammer_attack_score",
    "keys_tonal_decay_score",
    "keys_partial_inharmonicity_score",
    "keys_chord_density_score",
    "drum_kick_source_score",
    "drum_snare_source_score",
    "drum_clap_source_score",
    "drum_closed_hat_source_score",
    "drum_cymbal_source_score",
    "drum_tom_conga_source_score",
    "drum_rim_stick_source_score",
    "drum_shaker_tambourine_source_score",
    "drum_guiro_scrape_source_score",
    "drum_metallic_percussion_source_score",
    "fx_riser_build_score",
    "fx_drop_downlifter_score",
    "fx_whoosh_sweep_score",
    "fx_reverse_score",
    "fx_impact_score",
    "fx_glitch_stutter_score",
    "fx_blip_beep_score",
    "fx_formant_score",
    "fx_radio_electrical_score",
    "fx_machine_mechanical_score",
    "fx_foley_material_score",
    "fx_small_object_score",
    "texture_water_ocean_score",
    "texture_rain_score",
    "texture_wind_score",
    "texture_fire_score",
    "texture_thunder_score",
    "texture_noise_static_score",
    "texture_room_crowd_ambience_score",
    "human_breath_mouth_score",
    "human_spoken_voice_score",
    "human_scream_score",
    "human_applause_crowd_score",
    "animal_bird_score",
    "animal_voice_score",
    "animal_cricket_insect_score",
    "animal_cat_score",
    "animal_dog_score",
    "fx_door_foley_score",
    "fx_engine_machine_score",
    "fx_motor_machine_score",
    "fx_coin_object_score",
    "fx_key_object_score",
    "fx_boom_score",
    "fx_slam_score",
    "fx_sub_hit_score",
    "fx_alarm_score",
    "fx_siren_score",
    "bass_808_score",
    "bass_sub_score",
    "bass_synth_score",
    "bass_electric_score",
    "bass_upright_score",
    "guitar_acoustic_score",
    "guitar_electric_score",
    "guitar_nylon_score",
    "synth_lead_score",
    "synth_pad_score",
    "synth_chord_score",
    "brass_trumpet_score",
    "woodwind_flute_score",
    "woodwind_sax_score",
    "string_violin_score",
    "string_cello_score",
    "voice_choir_score",
    *CATEGORY_PANEL_MANIFEST_FIELDS,
]


class SortReportWriter:
    """Write user-facing sort outputs."""

    def __init__(self, output_dir: Path, write_zip: bool = True) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.sorted_root = self.output_dir / SORTED_ROOT_NAME
        self.write_zip = bool(write_zip)

    def start(self) -> None:
        """Create a clean output tree."""
        disable_macos_metadata_sidecars()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.sorted_root.exists():
            shutil.rmtree(self.sorted_root, ignore_errors=True)
        self.sorted_root.mkdir(parents=True, exist_ok=True)

    def place_file(self, result: SortFileResult) -> Path:
        """Place one audio file in the selected folder path."""
        folder = self.sorted_root / safe_folder_path(result.decision.folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        target = unique_path(folder / result.source_path.name)
        try:
            target.symlink_to(result.source_path.resolve())
        except Exception:
            shutil.copyfile(result.source_path, target)
        return target

    def finish(self, file_results: list[SortFileResult]) -> SortSummary:
        """Write manifest, summary, optional ZIP, and return summary."""
        manifest_path = self.output_dir / f"{SORTED_ROOT_NAME}_manifest.csv"
        summary_path = self.output_dir / f"{SORTED_ROOT_NAME}_summary.txt"
        write_manifest(manifest_path, file_results)
        write_debug_packets(self.output_dir / DEBUG_PACKET_SIDECAR_NAME, file_results)
        write_summary(summary_path, file_results)
        write_brain_ensemble_audit(self.output_dir / "Aaron_Brain_Ensemble_Audit.csv", file_results)
        write_brain_lane_validation_reports(self.output_dir, file_results)
        zip_path = self.write_sorted_zip() if self.write_zip else None
        return SortSummary(
            output_root=self.sorted_root,
            manifest_path=manifest_path,
            summary_path=summary_path,
            zip_path=zip_path,
            file_results=file_results,
        )

    def write_sorted_zip(self) -> Path:
        """Zip the sorted folder tree."""
        zip_path = self.output_dir / f"{SORTED_ROOT_NAME}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.sorted_root.rglob("*")):
                if path.is_dir() or path.name.startswith("._"):
                    continue
                archive.write(path, path.relative_to(self.output_dir))
        return zip_path


def disable_macos_metadata_sidecars() -> None:
    """Avoid AppleDouble files when writing sorted outputs on external drives."""
    os.environ.setdefault("COPYFILE_DISABLE", "1")


def write_manifest(path: Path, file_results: list[SortFileResult]) -> None:
    """Write the CSV manifest."""
    fields = [
        "source_path",
        "placed_path",
        "duration_sec",
        "read_status",
        "final_label",
        "final_top",
        "folder_path",
        "consensus_status",
        "decision_reason",
        "brain_ensemble_mode",
        "brain_ensemble_version",
        "brain_ensemble_vote_1",
        "brain_ensemble_vote_1_lanes",
        "brain_ensemble_vote_1_support",
        "brain_ensemble_vote_1_score",
        "brain_ensemble_weight_profile",
        "brain_ensemble_lane_trust_reason",
        "brain_ensemble_top5_json",
        "brain_lane_agreement_json",
        "final_agrees_with_brain_ensemble",
        "final_agrees_with_full_brain",
        "final_agrees_with_core_baby",
        "final_agrees_with_spread_baby",
        "final_agrees_with_outlier_baby",
        "brain_vote_1",
        "full_brain_vote_1",
        "baby_brain_vote_1",
        "baby_brain_enabled",
        "core_baby_vote_1",
        "core_baby_enabled",
        "spread_baby_vote_1",
        "spread_baby_enabled",
        "outlier_baby_vote_1",
        "outlier_baby_enabled",
        "harmonic_core_baby_vote_1",
        "harmonic_core_baby_enabled",
        "harmonic_spread_baby_vote_1",
        "harmonic_spread_baby_enabled",
        "harmonic_outlier_baby_vote_1",
        "harmonic_outlier_baby_enabled",
        "wetness_score",
        "harmonic_core_recall_enabled",
        "harmonic_core_recall_reason",
        "baby_brains_affect_product_vote",
        "harmonic_brains_affect_product_vote",
        "physics_vote_1",
        "shape_vote",
        "shape_confidence",
        "shape_vote_json",
        *PHYSICS_SUBPANEL_MANIFEST_FIELDS,
        "physics_subpanels_json",
        "parent_role_audit_json",
        "shared_facts_json",
        "shared_candidates_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in file_results:
            writer.writerow(manifest_row(result))


def physics_subpanel_manifest_values(facts_evidence: dict[str, Any]) -> dict[str, str]:
    """Return flattened low-level physics subpanel manifest columns."""
    subpanels = facts_evidence.get("physics_subpanels", {}) if isinstance(facts_evidence, dict) else {}
    flat = subpanels.get("flat", {}) if isinstance(subpanels, dict) else {}
    if not isinstance(flat, dict):
        flat = {}
    row: dict[str, str] = {}
    for field in PHYSICS_SUBPANEL_MANIFEST_FIELDS:
        value = flat.get(field, facts_evidence.get(field, "") if isinstance(facts_evidence, dict) else "")
        row[field] = str(value)
    return row


def compact_physics_subpanels(facts_evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a manifest-safe physics panel digest, not the full debug packet."""
    subpanels = facts_evidence.get("physics_subpanels", {}) if isinstance(facts_evidence, dict) else {}
    flat = subpanels.get("flat", {}) if isinstance(subpanels, dict) else {}
    if not isinstance(flat, dict):
        flat = {}
    compact_flat = {field: flat[field] for field in PHYSICS_SUBPANEL_MANIFEST_FIELDS if field in flat}
    digest: dict[str, Any] = {"flat": compact_flat}
    if isinstance(subpanels, dict):
        for key in ("selected", "summary", "warnings"):
            if key in subpanels:
                digest[key] = subpanels[key]
    return digest


def compact_parent_role_audit(facts_evidence: dict[str, Any]) -> dict[str, Any]:
    """Return the role-audit keys needed for manifest triage."""
    audit = facts_evidence.get("parent_role_audit", {}) if isinstance(facts_evidence, dict) else {}
    if not isinstance(audit, dict):
        return {}
    keep = (
        "detected_parent_role",
        "parent_role",
        "role_name",
        "broad_folder_path",
        "confidence",
        "winning_role",
        "role_score",
        "reason",
        "status",
    )
    return {key: audit[key] for key in keep if key in audit}


def compact_shared_candidates(candidates: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Return manifest-safe shared candidate rows."""
    rows: list[dict[str, Any]] = []
    for row in candidates[:limit]:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "folder_path": row.get("folder_path", ""),
                "label": row.get("label", ""),
                "top_family": row.get("top_family", ""),
                "combined_rank_score": row.get("combined_rank_score", ""),
                "brain_rank": row.get("brain_rank", ""),
                "physics_rank": row.get("physics_rank", ""),
            }
        )
    return rows


def compact_shared_facts(facts_evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a compact manifest digest of shared facts.

    The full debug packet can be huge. The CSV manifest should remain usable in
    Numbers, Excel, and quick Python scripts. Full evidence is written to the
    JSONL sidecar instead.
    """
    if not isinstance(facts_evidence, dict):
        return {"debug_packet_sidecar": DEBUG_PACKET_SIDECAR_NAME}
    digest: dict[str, Any] = {
        "debug_packet_sidecar": DEBUG_PACKET_SIDECAR_NAME,
        "measured_roles": facts_evidence.get("measured_roles", {}),
        "shape_vote": facts_evidence.get("shape_vote", {}),
        "parent_role_audit": compact_parent_role_audit(facts_evidence),
        "audio_analysis_packet_summary": facts_evidence.get("audio_analysis_packet_summary", {}),
        "audio_analysis_cache_stats": facts_evidence.get("audio_analysis_cache_stats", {}),
        "wetness_score": (facts_evidence.get("wetness_profile", {}) or {}).get("wetness_score", "")
        if isinstance(facts_evidence.get("wetness_profile", {}), dict)
        else "",
        "physics_vote_1": compact_top_guess(facts_evidence.get("physics_vote_result", {})),
        "brain_ensemble_vote_1": compact_top_guess(facts_evidence.get("brain_ensemble_vote_result", {})),
    }
    for key in (
        "harmonic_core_recall_enabled",
        "harmonic_core_recall_reason",
        "baby_brains_affect_product_vote",
        "harmonic_brains_affect_product_vote",
    ):
        if key in facts_evidence:
            digest[key] = facts_evidence[key]
    return digest


def write_debug_packets(path: Path, file_results: list[SortFileResult]) -> None:
    """Write full per-file evidence packets to a JSONL sidecar."""
    with path.open("w", encoding="utf-8") as handle:
        for result in file_results:
            payload = {
                "source_path": str(result.source_path),
                "placed_path": str(result.placed_path or ""),
                "folder_path": result.decision.folder_path,
                "consensus_status": result.decision.consensus_status,
                "decision_reason": result.decision.reason,
                "facts_evidence": result.facts.evidence,
                "shared_candidates": result.decision.shared_candidates,
            }
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def compact_top_guess(digest: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Return a compact top-guess dictionary from a voter digest."""
    if not isinstance(digest, dict):
        return {}
    guesses = digest.get("top_guesses", [])
    if not isinstance(guesses, list) or index >= len(guesses):
        return {}
    guess = guesses[index]
    if not isinstance(guess, dict):
        return {}
    evidence = guess.get("evidence", {}) if isinstance(guess.get("evidence", {}), dict) else {}
    return {
        "label": str(guess.get("label", "")),
        "folder_path": str(guess.get("folder_path", "")),
        "rank": guess.get("rank", ""),
        "score": guess.get("score", ""),
        "confidence": guess.get("confidence", ""),
        "lanes": evidence.get("brain_candidate_lanes", []),
        "support": evidence.get("brain_ensemble_support", ""),
        "ensemble_score": evidence.get("brain_ensemble_score", ""),
        "weight_profile": evidence.get("brain_ensemble_weight_profile", ""),
        "lane_trust_reason": evidence.get("brain_ensemble_lane_trust_reason", ""),
    }


def digest_top_label(digest: dict[str, Any]) -> str:
    """Return the top label for a voter digest."""
    return str(compact_top_guess(digest).get("label", "")) if isinstance(digest, dict) else ""


def final_agrees_with_label(final_label: str, final_folder: str, label: str) -> str:
    """Return a string bool showing whether a lane broadly agrees with final placement."""
    if not label:
        return ""
    final_parts = [part for part in str(final_folder or final_label).split("/") if part]
    label_parts = [part for part in str(label).split("/") if part]
    if not final_parts or not label_parts:
        return "False"
    if final_parts[0] != label_parts[0]:
        return "False"
    if len(final_parts) < 2 or len(label_parts) < 2:
        return "True"
    final_branch = final_parts[1].lower()
    label_branch = label_parts[1].lower()
    if final_branch == label_branch:
        return "True"
    brass_woodwind_terms = {"brass and woodwinds", "brass", "woodwinds"}
    if final_branch in brass_woodwind_terms and label_branch in brass_woodwind_terms:
        return "True"
    if final_branch == "instrument loops" and label_parts[0] == "Instruments":
        return "True"
    if final_branch == "drum loops" and label_branch in {
        "drum loops",
        "kick drums",
        "snares",
        "hi hats",
        "toms",
        "percussion",
    }:
        return "True"
    return "False"


def brain_lane_agreement(
    *,
    final_label: str,
    final_folder: str,
    ensemble_digest: dict[str, Any],
    full_digest: dict[str, Any],
    core_digest: dict[str, Any],
    spread_digest: dict[str, Any],
    outlier_digest: dict[str, Any],
) -> dict[str, Any]:
    """Return compact agreement facts for brain lane auditing."""
    lane_labels = {
        "ensemble": digest_top_label(ensemble_digest),
        "full": digest_top_label(full_digest),
        "core_baby": digest_top_label(core_digest),
        "spread_baby": digest_top_label(spread_digest),
        "outlier_baby": digest_top_label(outlier_digest),
    }
    return {
        "lane_top_labels": lane_labels,
        "final_agreement": {
            lane: final_agrees_with_label(final_label, final_folder, label) for lane, label in lane_labels.items()
        },
    }


def manifest_row(result: SortFileResult) -> dict[str, str]:
    """Return one manifest row."""
    brain_first = result.brain_votes.guesses[0].label if result.brain_votes.guesses else ""
    full_digest = (
        result.facts.evidence.get("full_brain_vote_result", {}) if isinstance(result.facts.evidence, dict) else {}
    )
    baby_digest = (
        result.facts.evidence.get("baby_brain_vote_result", {}) if isinstance(result.facts.evidence, dict) else {}
    )
    core_digest = (
        result.facts.evidence.get("core_baby_vote_result", {}) if isinstance(result.facts.evidence, dict) else {}
    )
    spread_digest = (
        result.facts.evidence.get("spread_baby_vote_result", {}) if isinstance(result.facts.evidence, dict) else {}
    )
    outlier_digest = (
        result.facts.evidence.get("outlier_baby_vote_result", {}) if isinstance(result.facts.evidence, dict) else {}
    )
    harmonic_core_digest = (
        result.facts.evidence.get("harmonic_core_baby_vote_result", {})
        if isinstance(result.facts.evidence, dict)
        else {}
    )
    harmonic_spread_digest = (
        result.facts.evidence.get("harmonic_spread_baby_vote_result", {})
        if isinstance(result.facts.evidence, dict)
        else {}
    )
    harmonic_outlier_digest = (
        result.facts.evidence.get("harmonic_outlier_baby_vote_result", {})
        if isinstance(result.facts.evidence, dict)
        else {}
    )
    ensemble_digest = (
        result.facts.evidence.get("brain_ensemble_vote_result", {}) if isinstance(result.facts.evidence, dict) else {}
    )
    full_top = ""
    baby_top = ""
    baby_enabled = ""
    core_top = ""
    core_enabled = ""
    spread_top = ""
    spread_enabled = ""
    outlier_top = ""
    outlier_enabled = ""
    harmonic_core_top = ""
    harmonic_core_enabled = ""
    harmonic_spread_top = ""
    harmonic_spread_enabled = ""
    harmonic_outlier_top = ""
    harmonic_outlier_enabled = ""
    ensemble_top_guess = compact_top_guess(ensemble_digest if isinstance(ensemble_digest, dict) else {})
    ensemble_diag = (
        ensemble_digest.get("diagnostics", {})
        if isinstance(ensemble_digest, dict) and isinstance(ensemble_digest.get("diagnostics", {}), dict)
        else {}
    )
    ensemble_top5 = []
    if isinstance(ensemble_digest, dict):
        for idx in range(5):
            compact = compact_top_guess(ensemble_digest, idx)
            if compact:
                ensemble_top5.append(compact)
    if isinstance(full_digest, dict):
        full_guesses = full_digest.get("top_guesses", [])
        if isinstance(full_guesses, list) and full_guesses:
            full_top = str(full_guesses[0].get("label", ""))

    def _lane_top_and_enabled(digest: dict) -> tuple[str, str]:
        top = ""
        enabled = ""
        if isinstance(digest, dict):
            guesses = digest.get("top_guesses", [])
            if isinstance(guesses, list) and guesses:
                top = str(guesses[0].get("label", ""))
            diag = digest.get("diagnostics", {})
            if isinstance(diag, dict):
                enabled = str(diag.get("enabled", ""))
        return top, enabled

    if isinstance(baby_digest, dict):
        baby_top, baby_enabled = _lane_top_and_enabled(baby_digest)
    core_top, core_enabled = _lane_top_and_enabled(core_digest)
    spread_top, spread_enabled = _lane_top_and_enabled(spread_digest)
    outlier_top, outlier_enabled = _lane_top_and_enabled(outlier_digest)
    harmonic_core_top, harmonic_core_enabled = _lane_top_and_enabled(harmonic_core_digest)
    harmonic_spread_top, harmonic_spread_enabled = _lane_top_and_enabled(harmonic_spread_digest)
    harmonic_outlier_top, harmonic_outlier_enabled = _lane_top_and_enabled(harmonic_outlier_digest)
    wet_profile = result.facts.evidence.get("wetness_profile", {}) if isinstance(result.facts.evidence, dict) else {}
    wetness_score = ""
    if isinstance(wet_profile, dict):
        wetness_score = str(wet_profile.get("wetness_score", ""))
    physics_first = result.physics_votes.guesses[0].label if result.physics_votes.guesses else ""
    shape_vote = result.facts.evidence.get("shape_vote", {}) if isinstance(result.facts.evidence, dict) else {}
    if not isinstance(shape_vote, dict):
        shape_vote = {}
    lane_agreement = brain_lane_agreement(
        final_label=result.decision.final_label,
        final_folder=result.decision.folder_path,
        ensemble_digest=ensemble_digest if isinstance(ensemble_digest, dict) else {},
        full_digest=full_digest if isinstance(full_digest, dict) else {},
        core_digest=core_digest if isinstance(core_digest, dict) else {},
        spread_digest=spread_digest if isinstance(spread_digest, dict) else {},
        outlier_digest=outlier_digest if isinstance(outlier_digest, dict) else {},
    )
    final_agreement = lane_agreement.get("final_agreement", {}) if isinstance(lane_agreement, dict) else {}
    return {
        "source_path": str(result.source_path),
        "placed_path": str(result.placed_path or ""),
        "duration_sec": f"{result.physics.duration_sec:.6f}",
        "read_status": result.physics.read_status,
        "final_label": result.decision.final_label,
        "final_top": result.decision.final_top,
        "folder_path": result.decision.folder_path,
        "consensus_status": result.decision.consensus_status,
        "decision_reason": result.decision.reason,
        "brain_ensemble_mode": str(ensemble_diag.get("architecture", "")),
        "brain_ensemble_version": str(ensemble_diag.get("ensemble_version", "")),
        "brain_ensemble_vote_1": str(ensemble_top_guess.get("label", "")),
        "brain_ensemble_vote_1_lanes": json.dumps(ensemble_top_guess.get("lanes", []), sort_keys=True),
        "brain_ensemble_vote_1_support": str(ensemble_top_guess.get("support", "")),
        "brain_ensemble_vote_1_score": str(ensemble_top_guess.get("ensemble_score", "")),
        "brain_ensemble_weight_profile": str(ensemble_top_guess.get("weight_profile", "")),
        "brain_ensemble_lane_trust_reason": str(ensemble_top_guess.get("lane_trust_reason", "")),
        "brain_ensemble_top5_json": json.dumps(ensemble_top5, sort_keys=True),
        "brain_lane_agreement_json": json.dumps(lane_agreement, sort_keys=True),
        "final_agrees_with_brain_ensemble": str(final_agreement.get("ensemble", "")),
        "final_agrees_with_full_brain": str(final_agreement.get("full", "")),
        "final_agrees_with_core_baby": str(final_agreement.get("core_baby", "")),
        "final_agrees_with_spread_baby": str(final_agreement.get("spread_baby", "")),
        "final_agrees_with_outlier_baby": str(final_agreement.get("outlier_baby", "")),
        "brain_vote_1": brain_first,
        "full_brain_vote_1": full_top,
        "baby_brain_vote_1": baby_top,
        "baby_brain_enabled": baby_enabled,
        "core_baby_vote_1": core_top,
        "core_baby_enabled": core_enabled,
        "spread_baby_vote_1": spread_top,
        "spread_baby_enabled": spread_enabled,
        "outlier_baby_vote_1": outlier_top,
        "outlier_baby_enabled": outlier_enabled,
        "harmonic_core_baby_vote_1": harmonic_core_top,
        "harmonic_core_baby_enabled": harmonic_core_enabled,
        "harmonic_spread_baby_vote_1": harmonic_spread_top,
        "harmonic_spread_baby_enabled": harmonic_spread_enabled,
        "harmonic_outlier_baby_vote_1": harmonic_outlier_top,
        "harmonic_outlier_baby_enabled": harmonic_outlier_enabled,
        "wetness_score": wetness_score,
        "harmonic_core_recall_enabled": str(result.facts.evidence.get("harmonic_core_recall_enabled", ""))
        if isinstance(result.facts.evidence, dict)
        else "",
        "harmonic_core_recall_reason": str(result.facts.evidence.get("harmonic_core_recall_reason", ""))
        if isinstance(result.facts.evidence, dict)
        else "",
        "baby_brains_affect_product_vote": str(result.facts.evidence.get("baby_brains_affect_product_vote", ""))
        if isinstance(result.facts.evidence, dict)
        else "",
        "harmonic_brains_affect_product_vote": str(result.facts.evidence.get("harmonic_brains_affect_product_vote", ""))
        if isinstance(result.facts.evidence, dict)
        else "",
        "physics_vote_1": physics_first,
        "shape_vote": str(shape_vote.get("primary_shape", "")),
        "shape_confidence": str(shape_vote.get("confidence", "")),
        "shape_vote_json": json.dumps(shape_vote, sort_keys=True),
        **physics_subpanel_manifest_values(result.facts.evidence if isinstance(result.facts.evidence, dict) else {}),
        "physics_subpanels_json": json.dumps(
            compact_physics_subpanels(result.facts.evidence if isinstance(result.facts.evidence, dict) else {}),
            sort_keys=True,
            default=str,
        ),
        "parent_role_audit_json": json.dumps(
            compact_parent_role_audit(result.facts.evidence if isinstance(result.facts.evidence, dict) else {}),
            sort_keys=True,
            default=str,
        ),
        "shared_facts_json": json.dumps(
            compact_shared_facts(result.facts.evidence if isinstance(result.facts.evidence, dict) else {}),
            sort_keys=True,
            default=str,
        ),
        "shared_candidates_json": json.dumps(
            compact_shared_candidates(result.decision.shared_candidates, limit=5),
            sort_keys=True,
            default=str,
        ),
    }


def write_brain_ensemble_audit(path: Path, file_results: list[SortFileResult]) -> None:
    """Write a compact audit CSV for brain-lane trust tuning.

    This file is intentionally small compared with the manifest.  It lets us
    inspect which brain lanes agreed with the final placement without opening
    giant JSON fields.
    """
    fields = [
        "source_path",
        "final_top",
        "folder_path",
        "consensus_status",
        "brain_ensemble_mode",
        "brain_ensemble_vote_1",
        "brain_ensemble_vote_1_lanes",
        "brain_ensemble_vote_1_support",
        "brain_ensemble_weight_profile",
        "brain_ensemble_lane_trust_reason",
        "full_brain_vote_1",
        "core_baby_vote_1",
        "spread_baby_vote_1",
        "outlier_baby_vote_1",
        "final_agrees_with_brain_ensemble",
        "final_agrees_with_full_brain",
        "final_agrees_with_core_baby",
        "final_agrees_with_spread_baby",
        "final_agrees_with_outlier_baby",
        "physics_vote_1",
        "shape_vote",
        "shape_confidence",
        "decision_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in file_results:
            row = manifest_row(result)
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary(path: Path, file_results: list[SortFileResult]) -> None:
    """Write a short human-readable summary."""
    total = len(file_results)
    review = sum(1 for result in file_results if result.decision.final_top == "_TO_REVIEW")
    by_top: dict[str, int] = {}
    for result in file_results:
        by_top[result.decision.final_top] = by_top.get(result.decision.final_top, 0) + 1
    lines = [
        "Aaron Sound Sorter Summary",
        f"Processed: {total}",
        f"Needs review: {review}",
        "",
        "Top folders:",
    ]
    for top, count in sorted(by_top.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {count:5d}  {top}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_folder_path(folder_path: str) -> Path:
    """Return a safe relative folder path."""
    parts = []
    for part in str(folder_path or "_TO_REVIEW/Unknown").replace("\\", "/").split("/"):
        cleaned = " ".join(part.replace(":", " ").split()).strip()
        if cleaned and cleaned not in {".", ".."}:
            parts.append(cleaned)
    return Path(*parts) if parts else Path("_TO_REVIEW", "Unknown")


def unique_path(path: Path) -> Path:
    """Return a unique path without overwriting existing output."""
    if not path.exists() and not path.is_symlink():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10000):
        candidate = path.with_name(f"{stem}_{index:04d}{suffix}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"Could not create unique path for {path}")
