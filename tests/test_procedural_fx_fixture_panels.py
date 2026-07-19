from __future__ import annotations

from pathlib import Path

import pytest

from aaron_sound_sorter import api as sorter_api
from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.models import AudioPhysics
from tests import synthetic_audio_fixtures as fixtures


def _fx_panel_scores(tmp_path: Path, fixture_name: str, samples) -> dict[str, float]:
    """Write one procedural FX fixture and return source-blind panel scores."""
    audio_path = fixtures.write_wav(tmp_path / f"{fixture_name}.wav", samples)
    fingerprint, duration_sec, read_status = sorter_api.make_fingerprint_safe(audio_path)
    assert read_status == "ok"
    facts = build_shared_audio_facts(AudioPhysics(audio_path, fingerprint, duration_sec, read_status))
    return facts.evidence["physics_subpanels"]["flat"]


@pytest.mark.parametrize(
    ("fixture_name", "samples", "score_key", "minimum_score"),
    [
        ("riser", fixtures.synth_fx_riser(), "fx_riser_build_score", 0.60),
        ("downlifter", fixtures.synth_fx_downlifter(), "fx_drop_downlifter_score", 0.34),
        ("whoosh", fixtures.synth_fx_whoosh(), "fx_whoosh_sweep_score", 0.54),
        ("impact_tail", fixtures.synth_fx_impact_tail(), "fx_impact_score", 0.50),
        ("beep_alarm", fixtures.synth_fx_beep_alarm(), "fx_blip_beep_score", 0.62),
        ("glitch_stutter", fixtures.synth_fx_glitch_stutter(), "fx_glitch_stutter_score", 0.64),
    ],
)
def test_procedural_fx_fixtures_exercise_expected_physics_panel(
    tmp_path: Path,
    fixture_name: str,
    samples,
    score_key: str,
    minimum_score: float,
) -> None:
    """Procedural FX fixtures should keep the matching FX panel alive."""
    scores = _fx_panel_scores(tmp_path, fixture_name, samples)

    assert scores[score_key] >= minimum_score


def test_procedural_riser_has_transition_authority_without_drum_loop_claim(tmp_path: Path) -> None:
    """A synthetic riser should look like FX motion, not a drum loop."""
    scores = _fx_panel_scores(tmp_path, "riser", fixtures.synth_fx_riser())

    assert scores["fx_transition_authority_score"] >= 0.55
    assert scores["fx_riser_build_score"] > scores["drum_loop_source_score"]


def test_procedural_wood_click_panel_prefers_one_shot_over_loop(tmp_path: Path) -> None:
    """The synthetic fixture library can also lock struck-wood one-shots."""
    scores = _fx_panel_scores(tmp_path, "wood_click", fixtures.synth_percussion_hit(dur=0.18))

    assert scores["role_one_shot_score"] > scores["role_loop_score"]
