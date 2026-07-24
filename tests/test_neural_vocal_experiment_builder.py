import pytest

from tools.build_neural_vocal_experiment import broad_training_label


@pytest.mark.parametrize(
    ("trusted_label", "broad_label"),
    [
        ("Drums/Drum Loops/Loops", "NonVoice/Drums and Percussion"),
        ("FX/Designed Noise FX/Siren/Long FX", "NonVoice/Designed FX"),
        ("Instruments/Voice/Vocal Loops/Loops", "Voice/Musical Vocal"),
        ("Instruments/Woodwinds/Saxophone/Loops", "NonVoice/Sax and Woodwind"),
        ("Instruments/Synths/Synth Pad/Loops", "NonVoice/Synth Pad and Keys"),
        ("Instruments/Keys/Piano/Loops", "NonVoice/Synth Pad and Keys"),
        ("Instruments/Guitar/Guitar Loops/Loops", "NonVoice/Other Instruments"),
    ],
)
def test_broad_training_label_uses_only_explicit_trusted_taxonomy(
    trusted_label: str,
    broad_label: str,
) -> None:
    assert broad_training_label(trusted_label) == broad_label
