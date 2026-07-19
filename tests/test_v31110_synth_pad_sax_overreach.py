from __future__ import annotations

from tests.sorter_harness import row_label, run_regression_sample


def _run_neutralized_audio(sample_name: str) -> dict[str, str]:
    return run_regression_sample(
        sample_name,
        output_key="v31110_synth_pad_sax_overreach",
        neutralize_input_name=True,
    )


def test_uploaded_pad_loop_is_not_stolen_by_sax_invariant() -> None:
    row = _run_neutralized_audio("CS_NJ2_135bpm_Pad_Aster_Am.wav")
    label = row_label(row)
    low = label.lower()

    assert label.startswith("Instruments/Synths/"), label
    assert "pad" in low, label
    assert "sax" not in low and "woodwind" not in low, label
    assert row.get("consensus_status") == "final_measured_synth_loop_invariant"
