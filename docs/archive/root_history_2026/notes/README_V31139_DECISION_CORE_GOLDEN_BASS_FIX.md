# Archived: V31.139 Decision Core Golden Bass-Loop Fix

## Problem fixed

`tests/test_decision_core_golden_failure_regressions.py` had one failing golden case:

```text
test_real_bass_loop_with_bass_candidate_must_not_flatten_to_instrument_loops_or_riser
```

The failure was:

```text
Expected: Instruments/Bass/Bass Loops
Actual:   Instruments/Instrument Loops/Loops
```

## Root cause

`DecisionCoreV2` correctly produced Bass Loop eligibility claims, including a real candidate-backed Bass claim.

`FamilyClaimArbiter` then blocked that Bass Loop parent claim because `_facts_support_bass_loop_parent_claim()` required a measured `bass_loop` role strength in `facts.evidence["measured_roles"]`.

The golden test intentionally provides minimal direct decision-core facts. It gives:

```text
shape_vote.primary_shape = bass_phrase
shape_vote.confidence = 1.0
real shared candidate = Instruments/Bass/Synth Bass/One Shots
```

That should be enough for the candidate-backed Bass Loop rescue. The arbiter was treating it like an unsupported broad inference, reviewing it, and then releasing the review to generic Instrument Loops.

## Architecture-safe fix

Changed:

```text
src/aaron_sound_sorter/engine/family_claim_arbiter.py
```

The Bass Loop parent claim support now accepts `candidate_true_bucket_rescue` when it is:

```text
candidate-backed
real-candidate marked
close enough by candidate score
paired with bass_phrase shape evidence
not contradicted by true voice evidence
```

This does not create a filename rule. It does not force every bass role into Bass Loops. The existing negative golden case still passes:

```text
test_false_bass_role_without_bass_candidate_must_not_force_synth_lead_to_bass_loops
```

## Also included

This bundle also includes the earlier V31.138 clean pitched stab guard and the test fixture hardening in `tests/conftest.py` so pytest fixture materialization uses byte-copy fallback instead of `shutil.copy2()` metadata copying.

## Tests run

Passed:

```text
python3 -m py_compile src/aaron_sound_sorter/engine/family_claim_arbiter.py
python3 -m pytest -q tests/test_decision_core_golden_failure_regressions.py
python3 -m pytest -q tests/test_decision_core_fx_smoke_matrix_hard_failures.py
python3 -m pytest -q tests/test_v31138_clean_pitched_stab_role_guard.py
./commands/quality/RUN_V31139_GOLDEN_BASS_LOOP_CHECKS.command
```

Also passed through the clean pitched-stab command up through the targeted tests:

```text
tests/test_v31138_clean_pitched_stab_role_guard.py
tests/test_parent_eligibility_percussion_oneshots.py
tests/test_phase4_pitched_percussion_guard.py
tests/test_physics_voter_piano_struck_identity.py
tests/test_v31116_synth_pad_keys_decoy_guard.py
tests/test_decision_core_golden_failure_regressions.py
```

No-source-name audit was run directly and passed:

```text
python3 tools/audit_no_source_name_sorting.py --project-root "$PWD" --verbose
```

Acceptance spot checks passed individually for:

```text
kick_clean_cs_ne_monroe
snare_clean_jackbaby
clap_clean_clap2
drum_loop_full_95
bass_loop_way_it_is
strings_loop_77_ebm
synth_loop_05_emn
```

The combined acceptance script was interrupted by the container timeout after five passing cases, so the remaining two were run individually.
