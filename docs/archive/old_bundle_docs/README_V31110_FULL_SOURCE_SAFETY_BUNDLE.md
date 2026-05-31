# Aaron Sound Sorter v31110 Full Source Safety Bundle

This is a cumulative safety bundle for the v31106 through v31110 repair chain.
It is not just the last changed file. It includes every patched source file from the recent repair sequence, with the newest version winning when the same file changed more than once.

## Included patched source files

- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
  - latest v31110 version
  - includes strong drum-consensus firewall, false-voice fixes, sax/reed protection, and synth-pad authority over sax/keys overreach
- `src/aaron_sound_sorter/voters/physics_instrument_layer.py`
  - latest v31108/v31106 cumulative version
- `src/aaron_sound_sorter/engine/claim_producers/profile_candidate_instruments.py`
  - v31106 profile candidate instrument update
- `tools/run_pytest_node_by_node.py`
  - node-by-node pytest helper from v31108

## Included regression tests and uploaded audio fixtures

- `tests/test_v31106_voice_voter_invariant.py`
- `tests/test_v31108_non_voice_tonal_loop_guard.py`
- `tests/test_v31108_uploaded_audio_routing.py`
- `tests/test_v31109_strong_drum_consensus_firewall.py`
- `tests/test_v31110_synth_pad_sax_overreach.py`
- `tests/regression_audio/01_WCS_No_Safety_BPM92_D#min__Bells.wav`
- `tests/regression_audio/AV5_5_94bpm_Hit 2.wav`
- `tests/regression_audio/belize87bpm_8bars_UNKWN (Bbm).wav`
- `tests/regression_audio/ES_TEDR2_Claps&Snares_76.wav`
- `tests/regression_audio/CS_NJ2_135bpm_Pad_Aster_Am.wav`

## Included commands

- `commands/quality/RUN_V31106_FX_VOICE_SAX_SELECTOR_AUDIT.command`
- `commands/quality/RUN_V31108_FX_SAX_ONE_BY_ONE_AUDIT.command`
- `commands/quality/RUN_V31108_UPLOADED_SAMPLE_AUDIT.command`

## Install behavior

`INSTALL_FULL_SOURCE_SAFETY_BUNDLE.command` installs the included source, tools, tests, commands, and handoff docs into the current project folder.
It does not create backups. Use git if you want rollback.
