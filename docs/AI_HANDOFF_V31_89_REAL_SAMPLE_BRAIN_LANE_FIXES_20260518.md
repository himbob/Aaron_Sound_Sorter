# AI Handoff — v31.89 Real-Sample Brain-Lane Fixes

Date: 2026-05-18
Project root expected on Aaron's Mac: `/Volumes/T9/testbed/Aaron_Sound_Sorter`
Bundle name: `ASS_v3189_force_with_handoff.zip`

## Read this first

This project is now under the strict architecture rule from:

- `docs/REQUIRED_Aaron_SoundSorter_v3177_Architecture_Review.md`
- `docs/AI_MUST_READ_CURRENT_ARCHITECTURE.md`
- `docs/NO_SOURCE_NAME_SORTING_POLICY_20260515.md`
- `docs/TESTING_POLICY.md`

The controlling architecture rule is:

```text
brain lanes -> evidence / claims
physics, shape, role, eligibility -> evidence / claims
FamilyClaimArbiter -> one final decision
PlacementResolver -> public folder path
```

Do not add another hardcoded rescue tree. Do not use filenames in production logic. Filenames are allowed only to grade test output and decide what failure family needs a physics or claim fix.

## Why this patch exists

The v31.88 change helped some real sax cases, but it over-amplified the Sax / Brass / Woodwind path. The bad pattern was:

```text
baby/outlier reed recall + generic pitched material -> Brass/Woodwind/Sax claim wins too often
```

That caused non-sax material to get pulled toward Sax or Brass/Woodwinds:

- bells and keys
- pads
- strings
- mixed instrumental loops
- short brass/stab clusters
- bass-heavy mixed melody loops

Aaron correctly called this out as a Sax-area family collapse. The fix in v31.89 is not “make sax weaker everywhere.” The fix is: baby/reed recall can help only when the raw result is unsafe or clearly wrong, such as generic FX, Human/Voice false positive, Siren, Coins, Motors, Dog, etc. It should not narrow a safe broad `Instruments/Instrument Loops/Loops` result into Sax.

## What was changed

### 1. Baby-brain category competence was made narrower

The baby/outlier reed lane is no longer treated as a general authority on all pitched loops.

New intent:

```text
real sax or reed false positive trapped in FX / Human Voice / Siren -> reed recall may help
already safe broad Instrument Loop -> do not narrow to Sax just because reed recall fires
keys, bells, pads, strings, mixed loops -> prefer broad Instrument Loop unless stronger specific evidence exists
```

The baby brains are useful, but only where the audit shows they are good. They are not reliable enough to steer drum loops or generic pitched loops globally.

### 2. Ambiguous FX leaves can lose to measured musical-loop claims

A measured pitched or mixed musical loop can now escape unsafe FX leaf winners such as:

- Coins
- Siren / Alarm
- Motor / Engine
- Dog / Animals
- Human / Voice, when true voice evidence is weak
- Small object / foley leaves

This was needed for `AV5_5_94bpm_Hit 2.wav`, which Aaron described as a short strong brass/stab or brass-shot cluster, not coins.

### 3. Mixed melody with bass is no longer forced to Bass Loops

`SC_[[Challenger]]_Melody_Multi_Sample_2_130BPM_Amin.wav` is not just bass. It is a mixed instrumental loop with a lot of bass. v31.89 keeps this broad:

```text
Instruments/Instrument Loops/Loops
```

### 4. Drum loop with bass-heavy body is no longer Synth Bass

`QUp_DzU_DrumLp_03_100bpm.wav` is a real drum loop. The fix makes drum-loop role/shape evidence beat bass/synth-bass narrowing when the material has drum-loop structure.

### 5. Shape metric reading bug was fixed

One hidden issue: some measured shape metrics were stored in `shape_vote_json`, but the helper was reading an earlier dict and returning zero. That suppressed broad loop claims for some real sax/brass and pitched-loop samples. The patch reads shape metrics from the available real location instead of losing them.

## Real samples Aaron corrected and expected behavior

These were the direct user corrections during the v31.89 work:

| Sample | Aaron's correction | Acceptable target |
|---|---|---|
| `AV5_5_94bpm_Hit 2.wav` | short strong brass/stab cluster, maybe brass shots together in short loop | broad Instrument Loop, possibly Brass/Woodwinds if strongly supported |
| `RHSH_Brass_Saxophones_01_keyEm_103bpm.wav` | sax loop, mixed sax, broad normal instrument loop is okay | Brass/Woodwinds Loop or Instrument Loops |
| `SC_[[Challenger]]_Melody_Multi_Sample_2_130BPM_Amin.wav` | mixed instrumental loop with lots of bass, not just bass | Instrument Loops or Mixed Musical Loop, not Bass-only |
| `QUp_DzU_DrumLp_03_100bpm.wav` | real drum loop, not synth bass | Drums/Drum Loops/Loops |
| `01_WCS_No_Safety_BPM92_D#min__Bells.wav` | mixed bells/keys, not sax | Instrument Loops, Keys, Bells acceptable; not Sax |
| `02_CHAMPAPI_vol_6_Pad_F#m_122bpm.wav` | not sax | broad Instrument Loop or Synth/Pad; not Sax |
| `03.strings_77bpm_Ebm.wav` | real strings, not electric guitar | Strings or broad Instrument Loop; not Guitar, not FX/Siren |

## Real one-file checks run before this handoff

Each was run one file at a time using the uploaded real brains from `Archive.zip`. The current workspace produced these results before packaging v31.89:

| Sample | Observed v31.89 result |
|---|---|
| `RHSH_Brass_Saxophones_01_keyEm_103bpm.wav` | `Instruments/Instrument Loops/Loops` |
| `SC_[[Challenger]]_Melody_Multi_Sample_2_130BPM_Amin.wav` | `Instruments/Instrument Loops/Loops` |
| `QUp_DzU_DrumLp_03_100bpm.wav` | `Drums/Drum Loops/Loops` |
| `01_WCS_No_Safety_BPM92_D#min__Bells.wav` | `Instruments/Instrument Loops/Loops` |
| `02_CHAMPAPI_vol_6_Pad_F#m_122bpm.wav` | `Instruments/Instrument Loops/Loops` |
| `03.strings_77bpm_Ebm.wav` | `Instruments/Instrument Loops/Loops` |
| `AV5_5_94bpm_Hit 2.wav` | `Instruments/Instrument Loops/Loops` |
| `DOJO_CGNB_Female_Vocal_Shot_01_D.wav` | `FX/Human and Voice FX` |
| `US_CHV2_Vocal_female_shouts_processed_13.wav` | `FX/Human and Voice FX` |
| `AAJV_Kick_Happiness.wav` | `Drums/Kick Drums/Generic Kick/One Shots` |
| `CS_NE_Kick_OneShot_Monroe.wav` | `Drums/Kick Drums/Generic Kick/One Shots` |
| `DRUMS_LOOP_125BPM.wav` | `Drums/Drum Loops/Loops` |
| `AA_JBL_74bpm_Am_Sax_Loop_13.wav` | `Instruments/Brass and Woodwinds/Loops` |
| `AA_JBL_78bpm_Cm_Sax_Loop_1.wav` | `Instruments/Brass and Woodwinds/Loops` |
| `SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav` | `Instruments/Instrument Loops/Loops` |
| `AMV_VRNB1_121_brass_saxophone_loop_shot_Gm.wav` | `Instruments/Instrument Loops/Loops` |
| `CD4_Forbidden Romance_saxophone & keys_Part_1_Gm_78.wav` | `Instruments/Instrument Loops/Loops` |
| `GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav` | `Instruments/Instrument Loops/Loops` |
| `Riser Short Effect.wav` | `FX/Structural and Transitional FX/Risers and Builds/Short Riser/Long FX` |

Important: Aaron then said to stop testing and package/handoff. No additional validation was run after this handoff file was written.

## What remains risky

### 1. The code is improved but not architecturally clean enough yet

v31.89 follows the claim-arbiter direction better than v31.88, but the codebase still has legacy seams in `decision_core_v2.py`. Some claim producers still know too much about folder concepts. The next refactor should continue moving folder-path ownership into `PlacementResolver` and claim ownership into typed `FamilyClaim` objects.

### 2. Brain-lane competence is still partly hand-coded

The better long-term fix is not static lane weights. The next step should compute lane competence from uploaded audit logs:

```text
brain lane + filename verification family -> right / wrong / broad acceptable / bad family
```

Then generate a small competence table such as:

```text
full brain: strong for drum loops and broad loops
core baby: good for clean centered examples, kicks, some vocals
spread baby: useful recall but risky for broad loops
outlier baby: useful for sax/reed false-positive escapes, risky as a narrowing authority
```

Use that table to adjust claim confidence, not final folder paths.

### 3. Broad Instrument Loops is currently the safe fallback

This is intentional for now. It may feel less specific, but it is much better than Sax stealing pads, Strings becoming Siren, or drum loops becoming Synth Bass. Specific leaves should come later when the source identity evidence is stronger.

## Next recommended work

1. Install this bundle on Aaron's Mac.
2. Run only one validation command at a time.
3. Run a corrected-sample panel first, not the whole world.
4. Then run the full FX pack one file at a time if needed.
5. Upload the new manifest and brain ensemble audit.
6. Build a data-driven brain-lane competence table from that run.
7. Continue removing folder decisions from `decision_core_v2.py` and into claim/placement layers.

## Do not do next

- Do not globally boost baby brains.
- Do not make Sax stronger globally.
- Do not add filename production rules.
- Do not run all pytest files at once and assume it is safe.
- Do not bulk-sort the whole FX ZIP as one command when debugging. Extract, then one sample at a time.
- Do not narrow broad Instrument Loops unless the specific identity evidence is stronger than the broad loop evidence.

## Suggested validation commands after install

Run these one at a time:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
```

```bash
python3 Aaron_Sound_Sorter.py self-test
```

```bash
python3 -m pytest tests/test_v3189_real_audio_architecture_regressions.py -q
```

```bash
python3 -m pytest tests/test_v3188_fx_run_regression_safety.py -q
```

```bash
python3 -m pytest tests/test_v3187_architecture_doc_and_lane_competence.py -q
```

Then do real-sample testing one file at a time using the corrected sample list above.
