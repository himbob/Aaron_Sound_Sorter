"""V2.5 transition-FX and reverb-sax broadening regressions."""

from __future__ import annotations

import pytest

from tests.sorter_harness import DEFAULT_BRAIN, PROJECT_ROOT, row_label, run_folder_once

BRAIN = DEFAULT_BRAIN
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_v25_transition_reverb"


@pytest.fixture(scope="module")
def sorted_rows() -> dict[str, dict[str, str]]:
    """Sort the V2.5 transition/reverb fixture folder once."""
    return run_folder_once(
        AUDIO_DIR,
        output_key="parent_eligibility_v25_transition_reverb",
        brain_path=BRAIN,
        timeout=240,
    )


def label_for(rows: dict[str, dict[str, str]], name: str) -> str:
    """Return normalized final folder path for a fixture."""
    assert name in rows, f"Missing {name}; got {sorted(rows)}"
    return row_label(rows[name])


def test_reverb_sax_phrase_broadens_to_instruments_instead_of_review_or_drums(
    sorted_rows: dict[str, dict[str, str]],
) -> None:
    """A heavy-reverb sax/reed phrase may broaden, but not to drums/human/FX animals."""
    label = label_for(sorted_rows, "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav")
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    forbidden = ["Drums", "Human", "Voice", "Dog", "Cat", "Bird", "Percussion"]
    bad = [fragment for fragment in forbidden if fragment.lower() in label.lower()]
    assert not bad, f"Unexpected {bad} in {label}"


def test_real_short_riser_stays_fx_transition_not_drum_loop(sorted_rows: dict[str, dict[str, str]]) -> None:
    """A real riser can be noisy/percussive but must remain FX transition."""
    label = label_for(sorted_rows, "Riser Short Effect.wav")
    assert label.startswith("FX/Structural and Transitional FX/Risers and Builds"), label
    assert not label.startswith("Drums/"), label


def test_vocal_phrase_from_fx_pack_stays_human_voice_not_generic_instrument(
    sorted_rows: dict[str, dict[str, str]],
) -> None:
    """A voiced vocal phrase may be broad, but not generic synth/instrument loop."""
    label = label_for(sorted_rows, "US_CHV2_Vocal_female_shouts_processed_13.wav")
    assert (
        label.startswith("FX/Human and Voice FX")
        or label.startswith("Instruments/Voice")
        or label.startswith("_TO_REVIEW/")
    ), label
    forbidden = ["Drums", "Percussion", "Machines", "Motor", "Dog", "Cat", "Bird", "Instrument Loops", "Synths"]
    bad = [fragment for fragment in forbidden if fragment.lower() in label.lower()]
    assert not bad, f"Unexpected {bad} in {label}"


def test_piano_loop_does_not_get_stolen_by_vocal_phrase_guard(sorted_rows: dict[str, dict[str, str]]) -> None:
    """The vocal phrase guard must not steal clean piano/key loops."""
    label = label_for(sorted_rows, "Piano 1 - 80 Bpm - Key C.wav")
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    assert "Human and Voice" not in label and "Voice" not in label, label
