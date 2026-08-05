# AI Handoff v31108 False Voice, Sax, and FX Conflict Repair

## Purpose

This patch repairs three related routing failures without using filenames as classifier evidence.

1. Synth bell and nonvoice tonal loops were able to be stolen into Voice.
2. Some sax and wet reed material could be flattened or pushed into false Voice paths.
3. A musical hit with raw FX evidence and blocked broad Instrument Loop evidence fell into `_TO_REVIEW/Measured Role Conflict` or broad FX when the parent eligibility evidence described a mixed musical loop.

## Architecture rule followed

The changes are source-name blind. Filenames are used only in test selectors for choosing the real samples to test. Sorting decisions use measured shape, physics subpanels, parent eligibility, and voter evidence.

The fix is deliberately broad and voter-level:

- `PhysicsInstrumentLayer` now blocks Voice only when a repeated tonal loop has strong nonvoice tonal evidence such as synth, metallic, keys, or reed authority.
- Bowed/string-like evidence is no longer enough by itself to block Voice because real vocal phrases can measure as bowed/string-like through sustained formant motion.
- `FamilyClaimArbiter` now treats parent-eligible mixed musical loops as safe broad Instrument Loops instead of falling into measured role conflict or broad FX.
- Formant/human-shaped FX evidence no longer automatically wins broad FX when the measured body is a repeated pitched musical phrase and the blocked broad Instrument Loop is parent-eligible.

## Files changed

- `src/aaron_sound_sorter/voters/physics_instrument_layer.py`
- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
- `tools/run_pytest_node_by_node.py`
- `tests/test_v31108_non_voice_tonal_loop_guard.py`
- `tests/test_v31108_uploaded_audio_routing.py`
- `tests/regression_audio/01_WCS_No_Safety_BPM92_D#min__Bells.wav`
- `tests/regression_audio/belize87bpm_8bars_UNKWN (Bbm).wav`
- `tests/regression_audio/AV5_5_94bpm_Hit 2.wav`
- `commands/quality/RUN_V31108_FX_SAX_ONE_BY_ONE_AUDIT.command`
- `commands/quality/RUN_V31108_UPLOADED_SAMPLE_AUDIT.command`

## One-file-at-a-time sax audit already run

The FX zip was unpacked, filenames containing `sax` or `saxophone` were selected, then each selected file was copied to a neutral name and sorted one at a time.

Result from the completed sax pass before the final acceptance polish:

- Total selected sax files: 49
- Sax or woodwind pass: 44
- Non-sax but not Voice and not Review: 5
- Voice false positives: 0
- `_TO_REVIEW` false outcomes: 0

The 5 non-sax results landed in `Instruments/Instrument Loops/Loops`. They were mixed or low-heavy phrase material by measured evidence. They should not be forced into Sax based on filename alone.

## Uploaded audio regressions

The three uploaded samples were neutralized and tested without filename evidence.

Expected policy:

- `01_WCS_No_Safety_BPM92_D#min__Bells.wav` must not go to Voice.
- `belize87bpm_8bars_UNKWN (Bbm).wav` must not go to Voice.
- `AV5_5_94bpm_Hit 2.wav` must escape `_TO_REVIEW/Measured Role Conflict` and should land in broad musical Instrument Loops when parent eligibility says mixed music loop.

All three one-node pytest checks passed after the final patch.

## Acceptance tests run one case at a time

The locked smoke acceptance panel has 32 cases. Each case was run as its own command via `RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --case-id <id>`.

Failures found and fixed during this pass:

1. `vocal_money_female_rap` was incorrectly swallowed into Instrument Loops by the first nonvoice tonal loop guard. Fixed by removing bowed/string-like evidence as a standalone nonvoice blocker.
2. `vocal_or_brass_hit_av5` incorrectly released to broad FX. Fixed by treating parent-eligible mixed musical loop conflicts as broad Instrument Loops.

After those fixes, all 32 acceptance cases passed individually.

## Pytest status

Passed one at a time after final code changes:

- `tests/test_v31108_non_voice_tonal_loop_guard.py`
- `tests/test_v31108_uploaded_audio_routing.py::test_uploaded_synth_bells_loop_is_not_voice`
- `tests/test_v31108_uploaded_audio_routing.py::test_uploaded_belize_loop_is_not_voice`
- `tests/test_v31108_uploaded_audio_routing.py::test_uploaded_av5_hit_escapes_measured_role_conflict_review`
- `tests/test_v31106_voice_voter_invariant.py`
- `tests/test_no_source_name_sorting_invariant.py`
- `tests/test_physics_voter_reed_sax_identity.py`
- `tests/test_vocal_shape_true_bucket_policy.py`
- `tests/test_claim_arbiter_architecture.py`
- `tests/test_claim_arbiter_real_panel_surrogates.py`
- `tests/test_family_claim_arbitration_matrix.py`
- `tests/test_family_claim_stability_followup_matrix.py`
- `tests/test_family_claim_stability_v31_76_followup.py`
- `tests/test_parent_eligibility_fact_roles_v27.py`
- `tests/test_role_aware_brain_ensemble_policy.py`
- `tests/test_v3197_no_role_score_shaping_architecture.py`
- `tests/test_v3199_arbiter_authority_contract.py`
- `tests/stability/test_family_firewalls.py::test_accepted_folder_is_not_blocked_by_firewall`

The full node-by-node pytest runner was also started after fixing its nested-folder collection bug. It collected 718 nodes and passed the first five nodes before the container tool call timed out. This was an environment/tool-call limit, not a failing test result. Run it locally on the Mac with:

```bash
cd /path/to/Aaron_Sound_Sorter
PYTHON_BIN="$(command -v python3)" TIMEOUT_SEC=240 STOP_AFTER=1 ./commands/quality/RUN_ALL_PYTESTS_NODE_BY_NODE.command
```

## Install command

From the active project root:

```bash
cd /path/to/Aaron_Sound_Sorter
ZIP="$HOME/Downloads/Aaron_Sound_Sorter_v31108_false_voice_sax_fx_patch.zip"
TMP="$(mktemp -d)"
unzip -q "$ZIP" -d "$TMP"
PATCH_DIR="$(find "$TMP" -maxdepth 1 -type d -name 'Aaron_Sound_Sorter_v31108_false_voice_sax_fx_patch' | head -1)"
find "$PATCH_DIR" -name '._*' -type f -delete
find "$PATCH_DIR" -name '.DS_Store' -type f -delete
rsync -av "$PATCH_DIR/src/" ./src/
rsync -av "$PATCH_DIR/tests/" ./tests/
rsync -av "$PATCH_DIR/tools/" ./tools/
rsync -av "$PATCH_DIR/commands/" ./commands/
cp "$PATCH_DIR/AI_HANDOFF_V31108_FALSE_VOICE_SAX_FX_REPAIR.md" ./
python3 -m py_compile \
  src/aaron_sound_sorter/voters/physics_instrument_layer.py \
  src/aaron_sound_sorter/engine/family_claim_arbiter.py \
  tools/run_pytest_node_by_node.py
rm -rf "$TMP"
echo "Installed v31108 false voice, sax, and FX conflict patch."
```
