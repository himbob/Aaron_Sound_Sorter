from __future__ import annotations

from tools.fx_zip_batch_probe import audit_source_group, audit_status, external_expected_family


def test_fx_probe_marks_real_instrument_source_as_pass_when_sorted_to_instruments() -> None:
    member = "FX_Aaron2/Premium Bounce/Loop/Vocals/Money_vocals_female_rap_110bpm.wav"

    expected = external_expected_family(member)

    assert expected == "Instruments_or_FX_Human"
    assert audit_source_group(expected) == "source_instrument_or_hybrid"
    assert audit_status(member, "Instruments/Voice/Vocal Loops/Loops") == "PASS_INSTRUMENT_SOURCE"


def test_fx_probe_still_marks_true_fx_source_to_instruments_as_failure() -> None:
    member = "Apollo Studios SFX Library/One_Shot/APL_SFX_Whoosh_Roar_01.wav"

    assert external_expected_family(member) == "FX"
    assert audit_status(member, "Instruments/Bass/Bass Loops") == "FAIL_INSTRUMENT"


def test_fx_probe_normalizes_lowercase_drum_top_family() -> None:
    member = "Loop 2/Fx/CSWW_Pan_Pipe_12_Dm_105bpm.wav"

    assert audit_status(member, "drums/percussion/bells and metallic percussion/one shots") == "QUESTIONABLE_DRUM"
