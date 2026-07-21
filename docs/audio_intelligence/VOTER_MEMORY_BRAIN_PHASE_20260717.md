# Voter Memory Brain Phase - 2026-07-17

## Goal

Move the sorter one step away from static rescue code by adding a trainable low-level voter memory lane. Human-approved corrections now teach reusable audio-fingerprint evidence for voter roles, not just final folder prototypes.

## What Changed

- Added `stage4_voter_memory_brain.json` as a dedicated memory brain.
- GUI incremental training now updates memory lanes by default:
  - `stage4_folder_brain_user_memory.json`
  - `stage4_shape_memory_brain.json`
  - `stage4_voter_memory_brain.json`
- Stable full/core/spread/outlier brains are no longer mutated by sparse GUI corrections by default.
- Runtime sorting loads voter memory from `config/gui_brains.yaml` and merges it into the in-memory brain view.
- ShapeVoter and PhysicsVoter read learned voter-memory evidence as calibration input.
- Manifest reports now include `learned_voter_memory_*` columns and nested JSON diagnostics.
- Added `commands/quality/RUN_TRUSTED_MEMORY_SEED.command` for command-line
  seeding of dedicated memory brains from explicit trusted panels.

## Runtime Rule

Voter memory uses measured numeric fingerprints only. Source filenames and source folders are used only for locating supervised training files and writing audit reports, never as runtime sorting evidence.

## Trusted Memory Seed Command

Run:

```bash
./commands/quality/RUN_TRUSTED_MEMORY_SEED.command --dry-run
```

The command reads a JSON or CSV panel and trains only rows with an explicit
`training_label` or `approved_label`. Broad acceptance prefixes are skipped on
purpose, because they are pass/fail rules, not teacher labels. This keeps the
memory brains from inventing broad fake taxonomy leaves.

## Seed Training Performed

Seed reports:

- `_reports/voter_memory_phase/train_20260717_172001`
- `_reports/voter_memory_phase/train_more_20260717_173332`
- `_reports/voter_memory_phase/acceptance_memory_seed_20260717_181956`
- `_reports/voter_memory_phase/phonk50_review_memory_seed_20260717_183211`

Roles seeded include:

- `fx_impact`
- `fx_riser`
- `instrument_bass_loop`
- `instrument_bass_one_shot`
- `instrument_keys_loop`
- `instrument_plucked_loop`
- `instrument_strings_loop`
- `instrument_synth_loop`
- `instrument_voice_loop`
- `drum_kick_one_shot`

## Validation

Focused tests passed:

- `make ai-test TEST='tests/test_gui_incremental_brain_update.py'`
- `make ai-test TEST='tests/test_gui_preview_service.py'`
- `make ai-test TEST='tests/test_gui_web_app.py'`
- `make ai-test TEST='tests/test_designed_fx_shape_claims.py'`
- `make ai-test TEST='tests/test_phase4_shape_voter.py'`
- `make ai-test TEST='tests/test_physics_layers.py'`
- `make ai-test TEST='tests/test_measured_instrument_loop_voter_claims.py'`
- `make ai-test TEST='tests/test_claim_arbiter_real_panel_surrogates.py'`
- `make ai-test TEST='tests/test_phase4_shape_voter.py::test_decisive_top_family_gate_needs_shape_or_role_support'`
- `make ai-test TEST='tests/test_parent_eligibility_full_thread_matrix.py::test_uploaded_voice_material_stays_voice_human_or_review'`

Project gate passed with sandbox-safe bytecode cache:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/aaron_sound_sorter_pycache make ai-check
```

Full pytest passed with sandbox-safe bytecode cache:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/aaron_sound_sorter_pycache make test
```

Locked acceptance passed:

- `_reports/locked_smoke_acceptance/run_20260717_182007`
- `_reports/locked_smoke_acceptance/run_20260717_182722`
- `_reports/locked_smoke_acceptance/run_20260717_183412`
- `_reports/locked_smoke_acceptance/run_20260717_190010`

Final locked run: 32 of 32 protected cases passed.

Real-pack recheck:

- `_reports/voter_memory_phase/phonk50_final_recheck_20260717_182950`: 50 processed, 3 review.
- `_reports/voter_memory_phase/phonk50_final_recheck_after_review_seed_20260717_183225`: 50 processed, 0 review.
- `_reports/voter_memory_phase/phonk50_post_fullsuite_recheck_20260717_192029`: 50 processed, 0 review.

Final Phonk run family distribution:

- Instruments: 34
- Drums: 9
- FX: 7
- Review: 0
- Learned voter-memory matches: 18

## Important Architecture Notes

- User corrections should keep teaching memory brains first.
- Full/core/spread/outlier brains should be rebuilt from curated training data, not edited by sparse GUI corrections.
- Voter memory and physics memory are no longer decorative telemetry. They may
  produce owner evidence when paired with measured body compatibility.
- Memory must not become a blind top-family override. It can own a deep label
  only when the broad measured family/body contract is compatible.
- When memory confidence is strong and low-level measured evidence agrees, the sorter should use it to deepen source identity.
- If memory and measured shape/family strongly disagree, review is still preferred over a forced folder.
- Static Python should block catastrophic contradictions and report why; it
  should not flatten strong learned memory into broad fallback buckets merely
  because old hand-coded identity thresholds are uncertain.

## Remaining Work

- Add a small report comparing memory-matched rows against non-memory rows by top family, review rate, and catastrophic steal count.
- Keep adding small supervised panels for hard areas:
  - vocal loops with heavy reverb
  - synth-bell loops
  - koto/plucked-string loops
  - risers versus cymbals
  - impacts versus kicks
- Long term: move more static ShapeVoter and PhysicsVoter branch thresholds into trainable memory-backed panels.
