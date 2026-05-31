from aaron_sound_sorter.features import dry_core_augmentation_kind_for_training_label


def test_dry_core_policy_is_off_by_default() -> None:
    assert (
        dry_core_augmentation_kind_for_training_label(
            group_key="Instruments/Voice/Vocal Loops",
            label="Instruments/Voice/Vocal Loops/Loops",
            public_label_text="Instruments/Voice/Vocal Loops/Loops",
            top="Instruments",
            policy="",
        )
        == ""
    )


def test_dry_core_policy_matches_voice_training_label() -> None:
    assert (
        dry_core_augmentation_kind_for_training_label(
            group_key="Instruments/Voice/Phrase/One Shots",
            label="Instruments/Voice/Phrase/One Shots",
            public_label_text="Instruments/Voice/Phrase/One Shots",
            top="Instruments",
            policy="voice_sax",
        )
        == "voice"
    )


def test_dry_core_policy_matches_sax_training_label() -> None:
    assert (
        dry_core_augmentation_kind_for_training_label(
            group_key="Instruments/Woodwinds/Saxophone/Loops",
            label="Instruments/Woodwinds/Saxophone/Loops",
            public_label_text="Instruments/Woodwinds/Saxophone/Loops",
            top="Instruments",
            policy="voice_sax",
        )
        == "sax"
    )


def test_dry_core_policy_does_not_match_non_instrument_fx_voice_bucket() -> None:
    assert (
        dry_core_augmentation_kind_for_training_label(
            group_key="FX/Human and Voice FX/Formant FX",
            label="FX/Human and Voice FX/Formant FX/One Shots",
            public_label_text="FX/Human and Voice FX/Formant FX/One Shots",
            top="FX",
            policy="voice_sax",
        )
        == ""
    )


def test_dry_core_policy_does_not_use_source_filename() -> None:
    # The folder label is Keys.  A source filename containing sax would not be
    # passed to this helper, so it must not trigger augmentation.
    assert (
        dry_core_augmentation_kind_for_training_label(
            group_key="Instruments/Keys/Electric Piano/Loops",
            label="Instruments/Keys/Electric Piano/Loops",
            public_label_text="Instruments/Keys/Electric Piano/Loops",
            top="Instruments",
            policy="voice_sax",
        )
        == ""
    )
