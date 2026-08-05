# AI READ THIS FIRST — Aaron Sound Sorter

This is the single current handoff document for future AI/code assistants.

Do not create a new bundle-specific handoff markdown file unless Aaron explicitly asks. Update this file instead.

## Current priority

Stop clever classifier patching. The project has repeatedly regressed because AI tools changed routing logic after narrow tests. The next safe work is QA, diagnostics, and tightly gated architecture changes.

## Non-negotiable release rule

Any AI that changes classifier behavior must run the locked real-audio smoke panel itself before returning code.

Required command:

```bash
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE.command
```

For behavior-changing work, also run the relevant pytest files. If the locked smoke panel has protected failures after the change, do not return a fixed bundle. Keep working or revert.

Do not say "Aaron should run it later" as proof. Aaron's run is final confirmation, not the AI's acceptance proof.

## What counts as behavior-changing code

Behavior-changing code includes edits to:

- classifier policy
- voters
- brains or brain loading
- feature extraction
- claim producers
- family arbiter
- placement resolver or folder mapping
- training or correction logic
- thresholds, gates, rescue logic, confidence rules

## What AI may safely do without smoke acceptance

QA-only work may still run its own tests but does not need classifier acceptance if it truly does not affect behavior:

- docs
- bundle tooling
- report formatting
- acceptance harness code
- manifest comparison tools
- static audits

## Architecture direction

The desired architecture is evidence first:

1. Feature extraction measures audio.
2. Brain/voters produce evidence.
3. Claim producers produce typed claims.
4. One arbiter owns final family decision.
5. One placement resolver maps final claims to public folders.
6. Ambiguous files go broad or review instead of a confident wrong leaf.

Do not add another hidden rescue branch that directly chooses a final folder.

## Smoke panel philosophy

The locked panel is small on purpose so AI tools can run it every time. It is a leash, not a full benchmark.

Use sentinel samples:

- one clear sample for stable critical categories
- two samples for fragile categories
- broad/review-accepted expectations for ambiguous files

Never use filenames as sorter evidence. Filenames are only stable test IDs for the acceptance harness.

## Bundle policy

Default bundles for AI handoff should include:

- source code
- tests
- commands
- tools
- docs
- active root brain JSONs
- acceptance harness and small locked smoke samples

Default bundles should exclude:

- training folders
- large sample libraries
- generated reports
- _real_sort_tests
- virtualenvs
- git folders
- old nested bundle ZIPs

## User project root

Default active project root:

```bash
/path/to/Aaron_Sound_Sorter
```

Do not write generated upload-back ZIPs, reports, or scratch output to the project root. Put them under `_reports`, `reports`, or another clear subfolder.

## If trained samples fail

Do not tune classifier rules first. Build or run a training round-trip audit and identify whether the failure is:

- feature/scaler mismatch
- brain exact memory failure
- label-map drift
- final arbiter override
- training contradiction
- disabled custom/exact routing

Trained-sample failure is a pipeline integrity issue, not a category-tuning issue.

## AI-safe locked smoke execution

The locked smoke gate is meant to be run by AI tools before they return behavior-changing code. If an AI wrapper times out, that is **not** a pass and not a reason to hand the work back to Aaron.

Preferred command for AI tools:

```bash
timeout 1200 ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command
```

The AI-safe command prints per-case progress immediately so the run does not look hung. It writes reports under:

```text
_reports/locked_smoke_acceptance/run_YYYYMMDD_HHMMSS/
```

A behavior-changing patch may only be returned as fixed when the final aggregate report shows:

- every expected case was attempted
- zero protected failures
- zero `FAIL_MISSING_SAMPLE`
- zero `FAIL_TIMEOUT`
- zero `FAIL_SORTER_ERROR`

If the AI tool cannot run a single long command, it must run chunks and report every chunk. Example 4-case chunks:

```bash
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 1 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 5 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 9 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 13 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 17 --max-cases 4
```

For debugging one failing sentinel:

```bash
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id kick_clean_cs_ne_monroe
```

Passing one case or one chunk is **not** acceptance. It only proves that part of the panel. For a classifier/routing patch, all protected cases must be accounted for before the patch can be called fixed.

Current case order:

1. `kick_clean_cs_ne_monroe` - `CS_NE_Kick_OneShot_Monroe.wav`
2. `snare_clean_jackbaby` - `JACKBABY_SNARE_11.wav`
3. `clap_clean_clap2` - `Clap2.wav`
4. `drum_loop_full_95` - `Drum_Full_06_95bpm.wav`
5. `bass_loop_way_it_is` - `MS_TLV1_03_The Way It Is_Bass_Eminor_98bpm_Dry.wav`
6. `piano_loop_key_c` - `Piano 1 - 80 Bpm - Key C.wav`
7. `strings_loop_77_ebm` - `03.strings_77bpm_Ebm.wav`
8. `synth_pad_e_pad_097` - `01 E Pad 097 Ebm.wav`
9. `synth_bells_no_safety` - `01_WCS_No_Safety_BPM92_D#min__Bells.wav`
10. `synth_lead_k_dot_fluteish` - `02_K_DOT_vol_4_Lead_Em_100bpm.wav`
11. `vocal_money_female_rap` - `Money_vocals_female_rap_110bpm.wav`
12. `vocal_female_shout_dojo` - `DOJO_FBP_Female_Vocal_Shout.wav`
13. `vocal_or_brass_hit_av5` - `AV5_5_94bpm_Hit 2.wav`
14. `sax_loop_aajbl_11` - `AA_JBL_74bpm_Am_Sax_Loop_11.wav`
15. `ambiguous_pitched_loop_she2` - `SHE2_loop 14_full_123 bpm_D#.wav`
16. `mixed_brass_sax_rnb` - `Brass_Saxophone_RnB_Multi_Instrument_F_Minor_80BPM.wav`
17. `fx_riser_short_effect` - `Riser Short Effect.wav`
