# locked_smoke_v1 acceptance panel

This folder is the small real-audio smoke gate for Aaron Sound Sorter.

It is intentionally small enough for AI tools to run before returning behavior-changing code. It is not a full validation library.

## Hard rule

Any AI or developer that changes classifier behavior must run:

```bash
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE.command
```

before returning a bundle. If this panel has protected failures after the change, the change is not accepted.

## What this panel protects

The panel protects catastrophic routing regressions across critical sentinels:

- kick
- snare
- clap
- drum loop
- bass
- piano/keys
- strings
- synth pad/lead/bell-like synth
- vocals
- sax/brass/woodwind
- mixed instrument loop
- riser FX

Some cases are strict. Some intentionally allow broad parent folders or review because the audio is ambiguous. Do not tighten ambiguous cases into brittle exact-leaf assertions unless Aaron has listened and approved the expectation.

## AI-safe execution when wrappers time out

Run the visible-progress wrapper first:

```bash
timeout 1200 ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command
```

If your AI/code tool cannot allow a long enough command timeout, run the panel in chunks. Do not claim acceptance until every expected case has been attempted and the aggregate result is known.

```bash
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 1 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 5 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 9 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 13 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 17 --max-cases 4
```

Single-case debugging:

```bash
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id synth_bells_no_safety
```

Rules for AI tools:

- A timeout is not a pass.
- A partial chunk is not full acceptance.
- A behavior-changing patch cannot be returned as fixed unless all protected cases are accounted for.
- If a wrapper times out, inspect `_reports/locked_smoke_acceptance/latest_run_path.txt` and the run folder before rerunning.
- Always include the final `summary.txt` and `failures.csv` in the handoff.

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

