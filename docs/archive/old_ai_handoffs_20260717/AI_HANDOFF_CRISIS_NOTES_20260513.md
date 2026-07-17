# Aaron Sound Sorter Crisis Handoff, 2026-05-13

## Current status

Stop making broad architecture changes until the current failure is understood.
The code recently got much worse on obvious files. Several samples are being sent into impossible categories, especially FX animal/bird-like folders or kick loops.

The project goal is not perfect AI. The design document says the sorter should be a practical sound librarian, sort obvious files into sensible producer folders, send uncertain/conflicting files to review, keep top matches visible, and avoid one-file hacks. The current behavior violates that goal.

## Active project root

Use this path by default:

```bash
/Volumes/T9/testbed/Aaron_Sound_Sorter
```

Do not default to the old `/Users/aaron/Documents/Codex/...` path.

## What likely went wrong

The recent changes added several good ideas, but they may now be interacting badly:

1. **112-feature loop/long segment features** were added to improve loop/long understanding.
2. **Long FX label preservation** fixed `_LONG_FX` being rewritten as `/One Shots`.
3. **Sample-count-neutral physics buckets** fixed the one-feature physics winner bug.
4. **Dynamic role gate** and **branch-local identity** were then added to stop obvious cross-family errors.
5. The latest stabilized role gate may still be too blunt or too weak depending on family. It can either trap a sample in the wrong broad family or fail to rescue it from one.

The failure pattern now suggests the issue is not lack of features. The samples have enough evidence. The problem is how candidate labels are being filtered and how broad learned buckets are allowed to compete.

## Known bad regression samples included in this bundle

The files are in `regression_audio/`.

| File | Aaron's description | Desired regression behavior |
|---|---|---|
| `04_Dmn_176bpm_bass.wav` | bass loop, not kick loop | Must land under `Instruments/Bass`, not `Drums/Kick` |
| `04.bass_92bpm_Em.wav` | cool synth bass thing, not kick | Must land under `Instruments/Bass`, not `Drums/Kick` |
| `DOJO_CGNB_Female_Vocal_Shot_01_D.wav` | short one-shot female voice, not percussion | Must not land in Drums/Percussion; should be Voice/Vocal/Human-like |
| `Drumloop_hats_Dark_Rap_140BPM.wav` | hi-hat drum loop, close to shaker but not shaker | Must be Drums, preferably Hi Hats or Drum Loops, not Shaker/Tambourine |
| `GangstaFunk_Cmaj_96bpm.wav` | full music loop with synths and beat, not FX birds | Must not land in FX Animals/Bird; broad Drums/Drum Loops or Instruments/Mixed/Instrument Loops may be acceptable |
| `SCY095_03_Drums_Top_Loop_90bpm_01.wav` | drum loop, not FX | Must be Drums/Loop-like, not FX |
| `1.Saxophone_1_110bpm_Am.wav` | saxophone loop | Should land in Instruments/Woodwinds/Saxophone or at least Instruments/Woodwinds |
| `Compton_Fmin_100bpm.wav` | mixed West Coast loop with drums/horns/percs | Broad `Drums/Drum Loops` may be acceptable; do not narrow to Kick/Snare |
| `AV5_5_94bpm_Hit 2.wav` | went into FX keys, target not confirmed | Diagnostic only until Aaron gives the correct destination |

## Test file included

`tests/test_uploaded_regression_audio.py`

This is intentionally broad and role-level. Do not satisfy it with filename rules. Make the measured physics and learned role system satisfy it.

Install it with `INSTALL_CRISIS_HANDOFF_TEST_BUNDLE.command`, then run:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
python3 -m pytest -q tests/test_uploaded_regression_audio.py
```

## Latest brain

The latest uploaded brain is included as:

```text
brain/latest_stage4_folder_brain.json.zip
```

The installer backs up your existing `stage4_folder_brain.json` and installs this uploaded brain.

## Patch archives included

`patches_created/` contains the patch archives created during this work session:

- `LONG_FX_LABEL_PRESERVE`
- `LOOP_LONG_SEGMENT_FEATURES`
- `DYNAMIC_ROLE_GATE`
- `BRANCH_LOCAL_IDENTITY`
- `ROLE_GATE_STABILIZED`
- earlier physics bucket patches

These are included for forensic comparison. Do not blindly stack them again. The active repo may already contain some or all of them.

## Partial pytest status from the last pass

Only a small chunk was confirmed before this handoff was requested:

```text
tests/test_audio_analyzer.py: passed
tests/test_brain_and_committee.py: passed
tests/test_broad_family_coherence.py: passed
tests/test_dynamic_branch_identity_gate.py: passed
tests/test_legacy_cli_contract.py: passed
```

That does **not** mean the suite is clean. It only means those short chunks passed.

## What the next AI should do first

1. Install this test bundle.
2. Run only the uploaded regression test file first.
3. Run the sorter on each sample and inspect top 20 brain/physics voters plus role-gate diagnostics.
4. Do not patch one category.
5. Identify whether the failure is:
   - role gate too broad,
   - role gate too narrow,
   - branch-local identity over-filtering,
   - physics bucket ranges too permissive,
   - consensus selecting shared confusion,
   - bad/latest brain profile data,
   - missing/weak bass/voice/hat loop features.

## Likely architecture direction

The right fix is probably a diagnostic-first recovery of the dynamic role gate:

- print sample role vector,
- print allowed candidate labels after role gate,
- print blocked candidate labels and why,
- print branch-local candidate set,
- print top 20 before and after gating,
- then adjust the gating design generically.

Avoid:

- filename rules,
- hard-coded sample-specific rescues,
- hiding categories,
- adding secret penalties,
- returning to a giant first-match rule tree.

## Coverage idea

Aaron asked about `pytest-cov` and roughly 80% coverage. That is useful later, but not before the architecture stabilizes. First make short regression tests that run quickly. Then add coverage tracking:

```bash
python3 -m pip install pytest-cov
python3 -m pytest --cov=src/aaron_sound_sorter --cov-report=term-missing -q
```

Do not chase 80% by writing shallow tests. Use synthetic shape tests and these real regression files.
