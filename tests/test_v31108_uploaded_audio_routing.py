from __future__ import annotations

from tests.sorter_harness import row_label, run_regression_sample


def _run_one(sample_name: str) -> str:
    return row_label(run_regression_sample(sample_name, output_key="v31108_uploaded_audio"))


def test_uploaded_synth_bells_loop_is_not_voice() -> None:
    label = _run_one("01_WCS_No_Safety_BPM92_D#min__Bells.wav")
    low = label.lower()
    assert "voice" not in low and "vocal" not in low, label
    assert label.startswith("Instruments/") or label.startswith("FX/"), label


def test_uploaded_belize_loop_is_not_voice() -> None:
    label = _run_one("belize87bpm_8bars_UNKWN (Bbm).wav")
    low = label.lower()
    assert "voice" not in low and "vocal" not in low, label
    assert label.startswith("Instruments/") or label.startswith("FX/"), label
