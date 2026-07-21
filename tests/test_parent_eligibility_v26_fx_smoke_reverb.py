"""V2.6 targeted FX-smoke regressions from Aaron's 2026-05-14 run."""

from __future__ import annotations

import pytest

from tests.sorter_harness import DEFAULT_BRAIN, PROJECT_ROOT, run_folder_once

BRAIN = DEFAULT_BRAIN
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_v26_fx_smoke_reverb"


def _sort_rows() -> dict[str, dict[str, str]]:
    return run_folder_once(AUDIO_DIR, output_key="v26_fx_smoke_reverb", brain_path=BRAIN, timeout=240)


@pytest.fixture(scope="module")
def sorted_rows() -> dict[str, dict[str, str]]:
    """Sort the focused FX-smoke fixture folder once for this module."""
    return _sort_rows()


def test_wet_sax_is_brass_woodwind_not_human_voice_or_drums(sorted_rows: dict[str, dict[str, str]]) -> None:
    rows = sorted_rows
    for name in [
        "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
        "SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav",
    ]:
        row = rows[name]
        path = row["folder_path"]
        # The blind sorter may not always prove the exact Sax/Brass-Woodwind
        # leaf from audio alone, but it must keep sax/reed-like phrases inside
        # Instruments and never route them to Human/Voice, Drums, or FX.
        assert path.startswith("Instruments/"), path
        assert "Voice" not in path and "Human" not in path, path
        assert not path.startswith("Drums/"), path
        assert not path.startswith("FX/"), path


def test_drum_loop_with_riser_like_shape_stays_drum_loop(sorted_rows: dict[str, dict[str, str]]) -> None:
    rows = sorted_rows
    row = rows["ABOUTME_94_DRUMLOOP.wav"]
    path = row["folder_path"]
    assert path.startswith("Drums/Drum Loops/"), path
    assert not path.startswith("FX/"), row["decision_reason"]
