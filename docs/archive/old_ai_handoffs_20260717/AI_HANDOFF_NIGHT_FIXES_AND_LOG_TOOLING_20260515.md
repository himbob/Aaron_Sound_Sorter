# AI Handoff: Night Fixes and Log-Only Regression Tooling

Date: 2026-05-15
Bundle scope: small architecture-safe fixes plus log-only testing tools.

## Mission

Aaron asked for two things:

1. Add enough testing/logging tools and commands so future debugging can be done from logs only, without uploading sound files.
2. Start carefully fixing the risky seams identified in the night audit, validating each fix before moving to the next one.

This bundle does both, but it deliberately avoids a broad rewrite.

## Important design stance

The repeated failure pattern is not a single bad threshold. It is weak measured descriptors being promoted too early into semantic buckets:

- formant-like spacing became Human/Voice FX
- low-mid tonal energy became Bass Loop
- repeated pitched/formant material became FX/Alarm
- short pitched percussion was stolen by Synth Lead logic

The safe correction is to require stronger positive identity for the semantic bucket, and otherwise fall back to a broader musical parent like `Instruments/Instrument Loops/Loops` or a broad drum one-shot bucket.

## Online DSP sanity check used

The DSP guidance supports this architecture: low-level descriptors are not source identity by themselves.

- Spectral flatness means tone-like versus noise-like, not voice versus sax versus strings.
- Spectral centroid is brightness/center of mass, not instrument identity.
- Descriptor libraries summarize low-level audio facts; the classifier must decide how much semantic weight those facts deserve.

This is why the code should not treat formant-like, low-mid-heavy, or bright repeated tonal facts as proof of a final semantic class.

## Files changed

### Code

```text
src/aaron_sound_sorter/domain/roles.py
src/aaron_sound_sorter/engine/eligibility.py
```

### Tests

```text
tests/test_parent_eligibility_synthetic_bass_alert_overreach_surrogates.py
```

Existing test from the earlier patch is still part of the regression panel:

```text
tests/test_parent_eligibility_synthetic_voice_false_positive_surrogates.py
```

### Tools

```text
tools/diff_sort_manifests.py
tools/run_single_file_log_panel.py
```

### Commands

```text
commands/quality/RUN_NIGHT_FIX_REGRESSION_TESTS.command
commands/quality/RUN_NIGHT_SINGLE_FILE_LOG_PANEL.command
```

## Fix 1: Voice/FX false positive protection

This was from the previous patch and remains in this bundle.

### Problem

These files were routed into `FX/Human and Voice FX`:

```text
03.strings_77bpm_Ebm.wav
RHSH_Saxophone_Ensemble_02_keyC_85bpm.wav
```

### Mechanism

Both measured as sustained pitched musical loops, but a vocal-like shape plus formant-like evidence triggered a broad Human/Voice FX fallback.

### Fix

`processed_vocal_loop_or_stab` now requires stronger positive voice identity and excludes clean sustained tonal instrument loops.

### Real validation

After patch:

```text
03.strings_77bpm_Ebm.wav => Instruments/Instrument Loops/Loops
RHSH_Saxophone_Ensemble_02_keyC_85bpm.wav => Instruments/Instrument Loops/Loops
```

This is not perfect deep identity yet. It is the safe parent-family correction.

## Fix 2: Bass-loop overreach on Rhodes and synth-lead loops

### Problem

These were being broadened into Bass Loops:

```text
VINTAGEDREAMS_KATO_rhodes_loop_amber_haze_86_C.wav
MS_TLV1_03_The Way It Is_Synth Lead 1_Eminor_98bpm_Wet.wav
```

### Mechanism

`bass_loop` role used total low energy, where total low energy was `sub_bass + bass_150_500`. Rhodes, organ, low keys, and synth leads can be very strong in 150-500 Hz without being foundational bass.

### Fix

`measured_roles_from_features()` now requires real sub/foundation energy before `bass_loop` can become decisive:

```text
sub_foundation_gate = ramp(sub_bass_ratio_lt_150hz, 0.35, 0.70)
```

This keeps true sub-heavy bass loops eligible, but stops low-mid instrument loops from being forced into Bass.

### Real validation

After patch:

```text
VINTAGEDREAMS_KATO_rhodes_loop_amber_haze_86_C.wav
=> Instruments/Instrument Loops/Loops

MS_TLV1_03_The Way It Is_Synth Lead 1_Eminor_98bpm_Wet.wav
=> Instruments/Instrument Loops/Loops
```

True bass controls still held:

```text
04.bass_92bpm_Em.wav
=> Instruments/Bass/Bass Loops

MS_TLV1_03_The Way It Is_Bass_Eminor_98bpm_Dry.wav
=> Instruments/Bass/Bass Loops
```

One bass sample still landed in generic instrument loops:

```text
03_bass_Emn_178bpm.wav
=> Instruments/Instrument Loops/Loops
```

That is acceptable for this patch because it is still inside Instruments and avoids a catastrophic wrong parent. A later deep-leaf improvement can handle this.

## Fix 3: Tonal alert/siren overreach on bright synth lead

### Problem

This file was broadened to Alarm:

```text
MS_NLS_02_Ghetto Millionaire_Synth Lead_Fminor_85bpm_Wet.wav
=> FX/Designed Noise FX/Alarm/Long FX
```

### Mechanism

The alert/siren branch fired from repeated pitched/formant-like tonal behavior. That is too broad, because synth leads and arps can look like repeated tonal alerts at the descriptor level.

### Fix

`tonal_alert_or_siren_fx` now excludes clean sustained tonal instrument loops and refuses to fire when the shape voter says `vocal_phrase` or `bass_phrase`.

### Real validation

After patch:

```text
MS_NLS_02_Ghetto Millionaire_Synth Lead_Fminor_85bpm_Wet.wav
=> Instruments/Instrument Loops/Loops
```

## Fix 4: Short pitched percussion stolen by Synth Lead logic

### Problem

Random one-shot percussion panel found:

```text
140923.wav
=> Instruments/Synths/Synth Lead/One Shots
```

The source ZIP was the one-shot percussion set, and the measured role was already `percussive_one_shot`. The final placement was wrong.

### Mechanism

`strong_vocal_identity` was too broad for a short resonant percussion hit. It blocked the protected percussion one-shot branch, but the file did not actually qualify as a vocal branch, so it fell through to `short_pitched_instrument_hit`.

### Fix

The protected percussion branch now only yields to strong vocal identity when the shape also supports a vocal shape:

```text
vocal_shape_identity = primary_shape in {"vocal_phrase", "vocal_one_shot", "hit_with_tail"}
```

A `single_hit` with formant-like resonance is still allowed to stay Drums.

### Real validation

Before:

```text
140923.wav => Instruments/Synths/Synth Lead/One Shots
```

After:

```text
140923.wav => Drums/Percussion/Generic Percussion/One Shots
```

Random one-shot percussion panel after fix:

```text
12/12 => Drums
0/12 => non-Drums
```

## New synthetic regression tests

New file:

```text
tests/test_parent_eligibility_synthetic_bass_alert_overreach_surrogates.py
```

Covers:

1. Rhodes low-mid tonal loop must not become Bass Loop.
2. Sub-foundation bass loop must still keep strong bass role.
3. Bright synth lead must not become FX/Alarm.
4. Short pitched percussion one-shot must stay Drums, not Synth Lead.

These are synthetic fact-vector surrogates. They do not store WAV files.

## New log-only tooling

### `tools/run_single_file_log_panel.py`

Runs a ZIP one selected audio file at a time. It extracts one temp file, runs the sorter, copies only logs/manifests/summaries, then deletes the temp audio.

Outputs:

```text
single_file_results.csv
single_file_suspicious.csv
single_file_summary.txt
cases/*/sorter_stdout.log
cases/*/Aaron_Sorted_Sounds_manifest.csv
cases/*/Aaron_Sorted_Sounds_summary.txt
single_file_log_panel_upload_back.zip
```

This is the tool Aaron needed for timeout-safe real-file testing without uploading audio.

Example:

```bash
python3 tools/run_single_file_log_panel.py \
  --project-root /path/to/Aaron_Sound_Sorter \
  --zip /path/to/sample-library/one_shot_percussive_sounds.zip \
  --expected-top Drums \
  --per-category 3 \
  --max-total 36 \
  --timeout-sec 75
```

### `tools/diff_sort_manifests.py`

Compares two manifests and writes:

```text
manifest_diff_all.csv
manifest_diff_changed.csv
manifest_diff_suspicious.csv
manifest_diff_summary.txt
```

This is the missing tool for answering: did this patch make anything crazy worse?

Example:

```bash
python3 tools/diff_sort_manifests.py \
  --old reports/before/Aaron_Sorted_Sounds_manifest.csv \
  --new reports/after/Aaron_Sorted_Sounds_manifest.csv \
  --out reports/diff_voice_fix
```

## Commands added

### Regression panel

```bash
./commands/quality/RUN_NIGHT_FIX_REGRESSION_TESTS.command
```

Runs py_compile plus the safe synthetic/non-audio pytest panel.

### Single-file log panel

```bash
./commands/quality/RUN_NIGHT_SINGLE_FILE_LOG_PANEL.command
```

Defaults to:

```text
ROOT=/path/to/Aaron_Sound_Sorter
ZIP_PATH=/path/to/sample-library/FX_Aaron2.zip
```

Examples:

```bash
# FX_Aaron2 default, 36 selected files
./commands/quality/RUN_NIGHT_SINGLE_FILE_LOG_PANEL.command

# One-shot percussion panel, assume everything should be Drums or Review
ZIP_PATH="/path/to/sample-library/one_shot_percussive_sounds.zip" \
EXPECTED_TOP="Drums" \
MAX_TOTAL=36 \
./commands/quality/RUN_NIGHT_SINGLE_FILE_LOG_PANEL.command
```

The command opens the report folder when done.

## Validation completed in sandbox

### Pytest panel

Command:

```bash
ROOT=/mnt/data/night_work/Aaron_Sound_Sorter \
./commands/quality/RUN_NIGHT_FIX_REGRESSION_TESTS.command
```

Result:

```text
56 passed
```

### Real-file focused checks

After patch:

```text
03.strings_77bpm_Ebm.wav
=> Instruments/Instrument Loops/Loops

RHSH_Saxophone_Ensemble_02_keyC_85bpm.wav
=> Instruments/Instrument Loops/Loops

VINTAGEDREAMS_KATO_rhodes_loop_amber_haze_86_C.wav
=> Instruments/Instrument Loops/Loops

MS_TLV1_03_The Way It Is_Synth Lead 1_Eminor_98bpm_Wet.wav
=> Instruments/Instrument Loops/Loops

MS_NLS_02_Ghetto Millionaire_Synth Lead_Fminor_85bpm_Wet.wav
=> Instruments/Instrument Loops/Loops

04.bass_92bpm_Em.wav
=> Instruments/Bass/Bass Loops

MS_TLV1_03_The Way It Is_Bass_Eminor_98bpm_Dry.wav
=> Instruments/Bass/Bass Loops
```

### Random one-shot percussion panel

Command used log-only tool with:

```text
ZIP: one_shot_percussive_sounds.zip
expected top: Drums
selected: 12
```

After patch:

```text
12/12 Drums
0 suspicious top-family misses
```

## Known limitations

1. Some real-file runs still take long in the sandbox. That is why the new log panel runs one file at a time and has per-file timeouts.
2. The fixes improve parent-family safety and broad bucket routing. They do not yet solve exact deep leaf identity such as exact Sax versus generic Instrument Loops.
3. `Needs review: 0` remains suspicious in broad mixed runs. This bundle adds the tooling needed to investigate that safely, but does not change review thresholds yet.
4. A file named `Kit 4 Sub Pad Key B minor 140 bpm.wav` still landed in `Instruments/Bass/Bass Loops` during the small log panel. That may be acceptable by physics because it is almost pure sub energy, but it should be listened to before changing code.
5. One true bass sample still landed in generic Instrument Loops. That is safer than wrong FX/Drums, but future deep-leaf work should improve Bass identity without reviving Rhodes/synth-lead bass steals.

## Next safe step

Run the new log panel on Aaron's Mac:

```bash
ZIP_PATH="/path/to/sample-library/FX_Aaron2.zip" \
MAX_TOTAL=60 \
./commands/quality/RUN_NIGHT_SINGLE_FILE_LOG_PANEL.command
```

Then run:

```bash
ZIP_PATH="/path/to/sample-library/one_shot_percussive_sounds.zip" \
EXPECTED_TOP="Drums" \
MAX_TOTAL=60 \
./commands/quality/RUN_NIGHT_SINGLE_FILE_LOG_PANEL.command
```

Upload only the generated `single_file_log_panel_upload_back.zip` files.

## One-sentence summary

This bundle stops the biggest remaining early semantic overreach patterns found tonight: low-mid Rhodes/synth leads becoming Bass Loops, bright synth leads becoming FX/Alarm, and short resonant percussion becoming Synth Lead, while adding log-only one-file-at-a-time test tooling so future regressions can be found without uploading audio.
