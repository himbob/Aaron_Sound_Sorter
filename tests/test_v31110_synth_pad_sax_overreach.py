from __future__ import annotations

from tests.sorter_harness import row_label, run_regression_sample


def _run_neutralized_audio(sample_name: str) -> dict[str, str]:
    return run_regression_sample(
        sample_name,
        output_key="v31110_synth_pad_sax_overreach",
        neutralize_input_name=True,
    )


def test_uploaded_pad_loop_memory_conflict_goes_to_review() -> None:
    row = _run_neutralized_audio("CS_NJ2_135bpm_Pad_Aster_Am.wav")
    label = row_label(row)
    low = label.lower()

    assert label.startswith("_TO_REVIEW/"), label
    assert "owner conflict" in low, label
    assert "sax" not in low and "woodwind" not in low, label
    assert row.get("consensus_status") == "measured_memory_owner_conflict_review"
