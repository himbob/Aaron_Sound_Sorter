# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Shared measured source-owner contracts for claim producers.

The project is moving toward brain-owned source identity, but the measured
voters still need lightweight family firewalls.  This module keeps those
firewalls in one place so FX, instrument, and arbiter code do not each grow
slightly different copies of the same source-contract logic.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    _feature_number_from_facts,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.voters.scoring_tools import role_strength


def supports_measured_drum_loop_owner(facts: SharedAudioFacts | None) -> bool:
    """Return True when measured facts prove drum-loop ownership.

    Args:
        facts: Shared measured audio facts from the current sort.

    Returns:
        True when the audio behaves like a repeated percussive/drum loop and
        has direct drum-loop or drum-material evidence strong enough to make a
        weak tonal side-signal stand down.

    Side Effects:
        None.

    Raises:
        None.

    Constraints:
        This contract uses measured audio facts and internal voter scores only.
        It must not inspect source filenames, source folders, ZIP member paths,
        or producer pack names.
    """
    if facts is None:
        return False
    shape = _shape_vote_from_facts(facts)
    if shape not in {"beat_loop", "top_loop", "drum_loop", "repeated_phrase_loop"}:
        return False
    if _shape_confidence_from_facts(facts) < 0.78:
        return False
    drum_loop_body = max(
        _subpanel_score(facts, "drum_loop_source_score"),
        _subpanel_score(facts, "rhythmic_break_loop_score"),
        _feature_number_from_facts(facts, "drum_loop_source_score"),
    )
    drum_material = max(
        _subpanel_score(facts, "drum_hit_score"),
        _subpanel_score(facts, "drum_kick_source_score"),
        _subpanel_score(facts, "drum_snare_source_score"),
        _subpanel_score(facts, "drum_clap_source_score"),
        _subpanel_score(facts, "drum_closed_hat_source_score"),
        _subpanel_score(facts, "drum_cymbal_source_score"),
        _subpanel_score(facts, "drum_tom_conga_source_score"),
        _subpanel_score(facts, "drum_rim_stick_source_score"),
        _subpanel_score(facts, "drum_shaker_tambourine_source_score"),
        _subpanel_score(facts, "drum_guiro_scrape_source_score"),
        _subpanel_score(facts, "drum_metallic_percussion_source_score"),
        _subpanel_score(facts, "compact_struck_tonal_percussion_score"),
    )
    percussive_loop = max(
        _shape_metric_from_facts(facts, "percussive_event_ratio"),
        _shape_metric_from_facts(facts, "drumlike_frame_ratio"),
        _shape_metric_from_facts(facts, "loop_percussive_event_ratio"),
        _shape_metric_from_facts(facts, "loop_drumlike_frame_ratio"),
        _feature_number_from_facts(facts, "loop_percussive_event_ratio"),
        _feature_number_from_facts(facts, "loop_drumlike_frame_ratio"),
    )
    event_count = max(
        _shape_metric_from_facts(facts, "onset_count"),
        _feature_number_from_facts(facts, "onset_count"),
        _subpanel_score(facts, "physics_subpanel_event_count"),
    )
    looplike_body = bool(getattr(facts, "is_loop_like", False) or event_count >= 6.0)
    return bool(
        looplike_body
        and drum_loop_body >= 0.62
        and (drum_material >= 0.34 or percussive_loop >= 0.22)
        and _subpanel_score(facts, "synth_tonal_source_score") < drum_loop_body + 0.08
    )


def supports_clean_low_bass_phrase_owner(facts: SharedAudioFacts | None) -> bool:
    """Return True when measured facts prove clean low bass phrase ownership.

    Args:
        facts: Shared measured audio facts from the current sort.

    Returns:
        True when the audio behaves like a repeated/loop-like, low-band dominant,
        pitched bass phrase with weak drum and transition-FX evidence.

    Side Effects:
        None.

    Raises:
        None.

    Constraints:
        This contract uses measured audio facts and internal subpanel scores
        only.  It must not inspect source filenames, source folders, ZIP member
        paths, or producer pack names.
    """
    if facts is None:
        return False
    shape = _shape_vote_from_facts(facts)
    if shape not in {"bass_phrase", "solo_phrase", "beat_loop", "pitched_repetition_phrase"}:
        return False
    if _shape_confidence_from_facts(facts) < 0.78:
        return False

    roles = facts.evidence.get("measured_roles", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
    detected_role = str(roles.get("detected_parent_role") or "") if isinstance(roles, dict) else ""
    musical_role = max(
        role_strength(roles, "bass_loop") if isinstance(roles, dict) else 0.0,
        role_strength(roles, "pitched_music_loop") if isinstance(roles, dict) else 0.0,
        1.0 if detected_role in {"bass_loop", "pitched_music_loop", "synth_loop"} else 0.0,
    )
    bass_identity = max(
        _subpanel_score(facts, "bass_synth_score"),
        _subpanel_score(facts, "bass_sub_score"),
        _subpanel_score(facts, "bass_808_score"),
        _subpanel_score(facts, "bass_electric_score"),
        _subpanel_score(facts, "bass_upright_score"),
        _subpanel_score(facts, "low_end_source_score"),
        _subpanel_score(facts, "instruments_bass_generic_bass_one_shots_score"),
        _subpanel_score(facts, "instruments_bass_synth_bass_one_shots_score"),
        _subpanel_score(facts, "instruments_bass_sub_bass_one_shots_score"),
        _feature_number_from_facts(facts, "bass_synth_score"),
        _feature_number_from_facts(facts, "bass_sub_score"),
    )
    looplike_body = bool(
        getattr(facts, "is_loop_like", False)
        or (
            _shape_metric_from_facts(facts, "onset_count") >= 4.0
            and _shape_metric_from_facts(facts, "onset_span_ratio") >= 0.45
            and _shape_metric_from_facts(facts, "true_repetition_score") >= 0.45
        )
    )
    clean_low_tonal_body = bool(
        musical_role >= 0.78
        and looplike_body
        and bass_identity >= 0.62
        and max(
            _shape_metric_from_facts(facts, "pitch_confidence"),
            _feature_number_from_facts(facts, "pitch_confidence"),
        )
        >= 0.70
        and _shape_metric_from_facts(facts, "pitched_event_ratio") >= 0.88
        and max(
            _shape_metric_from_facts(facts, "sustained_tonal_frame_ratio"),
            _shape_metric_from_facts(facts, "non_event_tonal_ratio"),
        )
        >= 0.78
        and _shape_metric_from_facts(facts, "low_event_ratio") >= 0.84
        and _shape_metric_from_facts(facts, "mid_event_ratio") <= 0.18
        and _shape_metric_from_facts(facts, "high_event_ratio") <= 0.10
        and _shape_metric_from_facts(facts, "spectral_flatness_mean") <= 0.18
        and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.12
        and _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.12
    )
    if not clean_low_tonal_body:
        return False

    drum_material = max(
        _subpanel_score(facts, "drum_loop_source_score"),
        _subpanel_score(facts, "drum_hit_score"),
        _subpanel_score(facts, "drum_kick_source_score"),
        _subpanel_score(facts, "drum_snare_source_score"),
        _subpanel_score(facts, "drum_tom_conga_source_score"),
        _subpanel_score(facts, "drum_metallic_percussion_source_score"),
        _subpanel_score(facts, "rhythmic_break_loop_score"),
    )
    fx_motion = max(
        _subpanel_score(facts, "fx_motion_score"),
        _subpanel_score(facts, "fx_transition_authority_score"),
        _subpanel_score(facts, "fx_riser_build_score"),
        _subpanel_score(facts, "fx_drop_downlifter_score"),
        _subpanel_score(facts, "fx_whoosh_sweep_score"),
        _subpanel_score(facts, "fx_reverse_score"),
        _subpanel_score(facts, "fx_glitch_stutter_score"),
        _subpanel_score(facts, "fx_radio_electrical_score"),
    )
    return bool(drum_material <= 0.44 and fx_motion <= 0.42)


def supports_synth_pad_over_woodwind(facts: SharedAudioFacts | None) -> bool:
    """Return whether a clean synth-pad body contradicts woodwind ownership.

    Args:
        facts: Shared measured audio facts from the current sort.

    Returns:
        True when sustained synth-pad evidence decisively exceeds sax and reed
        evidence while the direct reed-authority signal remains weak.

    Side Effects:
        None.

    Raises:
        None.

    Constraints:
        This narrow owner contract uses measured audio facts only. It must not
        inspect source filenames, folders, ZIP paths, or sample-pack labels.
    """
    if facts is None:
        return False
    shape = _shape_vote_from_facts(facts)
    synth_pad = _subpanel_score(facts, "synth_pad_score")
    synth_source = _subpanel_score(facts, "synth_tonal_source_score")
    synth_siblings = max(
        _subpanel_score(facts, "synth_chord_score"),
        _subpanel_score(facts, "synth_lead_score"),
    )
    reed_identity = max(
        _subpanel_score(facts, "woodwind_sax_score"),
        _subpanel_score(facts, "reed_wind_score"),
    )
    return bool(
        shape
        in {
            "bass_phrase",
            "pitched_phrase",
            "pitched_phrase_shape",
            "repeated_phrase_loop",
            "sustained_pad",
            "vocal_phrase",
        }
        and _shape_confidence_from_facts(facts) >= 0.78
        and synth_pad >= 0.74
        and synth_source >= 0.68
        and synth_pad >= reed_identity + 0.06
        and synth_pad >= synth_siblings - 0.04
        and _subpanel_score(facts, "reed_wind_authority_score") <= 0.58
        and _shape_metric_from_facts(facts, "pitched_event_ratio") >= 0.88
        and max(
            _shape_metric_from_facts(facts, "sustained_tonal_frame_ratio"),
            _shape_metric_from_facts(facts, "non_event_tonal_ratio"),
        )
        >= 0.86
        and _shape_metric_from_facts(facts, "low_event_ratio") >= 0.45
        and _shape_metric_from_facts(facts, "high_event_ratio") <= 0.03
        and _shape_metric_from_facts(facts, "percussive_event_ratio") <= 0.12
        and _shape_metric_from_facts(facts, "drumlike_frame_ratio") <= 0.12
    )


def _subpanel_score(facts: SharedAudioFacts, key: str) -> float:
    subpanels = (
        facts.evidence.get("physics_subpanels", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
    )
    if isinstance(subpanels, dict):
        flat = subpanels.get("flat", {})
        if isinstance(flat, dict):
            try:
                return float(flat.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return _feature_number_from_facts(facts, key)
