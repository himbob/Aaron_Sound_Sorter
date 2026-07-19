from __future__ import annotations

from tests.sorter_harness import row_label, run_regression_sample


def _run_neutralized_audio(sample_name: str) -> dict[str, str]:
    return run_regression_sample(
        sample_name,
        output_key="v31109_strong_drum_consensus",
        neutralize_input_name=True,
    )


def test_uploaded_snare_clap_is_not_released_to_blip_fx() -> None:
    row = _run_neutralized_audio("ES_TEDR2_Claps&Snares_76.wav")
    label = row_label(row)
    low = label.lower()

    assert label.startswith("Drums/"), label
    assert "blip" not in low, label
    assert not label.startswith("FX/Designed Noise FX/Blip"), label
    assert row.get("consensus_status") != "final_clean_tonal_non_drum_hit_fx_invariant"
