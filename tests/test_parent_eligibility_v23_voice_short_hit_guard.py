"""Regression tests for V2.3 vocal and short-hit structure guards.

These are real uploaded failures from the current thread.  The tests lock broad
family safety only: vocal one-shots must not become machines/percussion, and
short pitched one-shots must not be sent to loop folders.
"""

from __future__ import annotations

import pytest

from tests.sorter_harness import DEFAULT_BRAIN, PROJECT_ROOT, row_label, run_folder_once

BRAIN = DEFAULT_BRAIN
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_v23_voice_short_hit_guard"


@pytest.fixture(scope="module")
def sorted_rows() -> dict[str, dict[str, str]]:
    """Sort this focused fixture folder once and return rows by filename."""
    return run_folder_once(
        AUDIO_DIR,
        output_key="parent_eligibility_v23_voice_short_hit_guard",
        brain_path=BRAIN,
        timeout=180,
    )


def final_label(row: dict[str, str]) -> str:
    """Return normalized final label."""
    return row_label(row)


def assert_not_in(label: str, *fragments: str) -> None:
    """Assert none of the fragments occur in a label."""
    low = label.lower()
    bad = [fragment for fragment in fragments if fragment.lower() in low]
    assert not bad, f"Unexpected {bad} in {label}"


def test_formant_rich_female_vocal_shot_does_not_go_to_machine_or_percussion(
    sorted_rows: dict[str, dict[str, str]],
) -> None:
    """A true vocal shot/Fx should stay in broad Human/Voice, not machines or percussion."""
    label = final_label(sorted_rows["DOJO_CGNB_Female_Vocal_Shot_01_D.wav"])
    assert (
        label.startswith("FX/Human and Voice FX")
        or label.startswith("Instruments/Voice")
        or label.startswith("_TO_REVIEW/")
    ), label
    assert_not_in(label, "Machine", "Motor", "Engine", "Drums", "Percussion", "Dog", "Cat", "Bird")


def test_short_low_pitched_hit_does_not_go_to_loop_folder(sorted_rows: dict[str, dict[str, str]]) -> None:
    """A short pitched hit may be a bass/instrument one-shot, but not a loop."""
    label = final_label(sorted_rows["29793.wav"])
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    assert_not_in(label, "Loops", "Long FX", "Drum Loops")


def test_short_percussive_hit_still_stays_out_of_keys_after_vocal_fix(sorted_rows: dict[str, dict[str, str]]) -> None:
    """The vocal fix must not reopen the old Keys/Instrument theft bug."""
    label = final_label(sorted_rows["16791.wav"])
    assert label.startswith("Drums/") or label.startswith("_TO_REVIEW/"), label
    assert_not_in(label, "Keys", "Instruments", "FX", "Human", "Voice")
