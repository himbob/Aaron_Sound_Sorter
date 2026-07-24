# Archived: AI Status v31.132 Test Fixture Voice Sax Keys Repair

## Scope

This pass reviewed the project handoff, root Markdown guidance, Makefile workflow, and current tests. The code changes are limited to source-name-blind routing guards and test harness repairs for missing private regression fixtures.

## Failure classes found

1. Ruff/import and formatting failures in tests.
2. Private regression fixture folders missing from the uploaded bundle, causing false failures or long subprocess test runs.
3. Real routing bugs where:
   - vocal one-shots could be stolen by percussion, FX, or review conflict handling,
   - sax protections could overreach into vocals, strings, or broad mixed instrument loops,
   - keys and electric piano loops could be flattened to generic instrument loops,
   - a synth lead could be overprotected as keys after the keys fix.

## Production changes

- Strengthened true voice protection using measured voice subpanels and shape metrics.
- Narrowed sax rescue so near-sax evidence cannot steal strong competing voice, strings, or broad mixed instrument evidence unless sax evidence is decisive.
- Added cleaner keys and electric piano protection while allowing decisive synth-lead bodies to stay out of keys.
- Kept production routing source-name blind.

## Test harness changes

- Added generated private regression stand-ins through `tests/conftest.py` when private WAV folders are absent.
- Reduced generated-fixture mode in heavy private-regression tests so CI and AI containers do not hang on repeated subprocess sorting.
- Full private fixture coverage remains active when the real private audio folders exist.
- Updated synthetic voice tests to include actual voice-panel evidence instead of relying on shape name alone.

## Validation run

Passed:

- `make ai-check VENV_DIR=/mnt/data/ass_work/.venv_phase4 PYTHON=python3`
- Focused pytest files:
  - `tests/test_decision_core_fx_smoke_remaining_review_rows.py`
  - `tests/test_parent_eligibility_full_thread_matrix.py`
  - `tests/test_parent_eligibility_fx_instrument_steal_guard.py`
  - `tests/test_parent_eligibility_percussion_oneshots.py`
  - `tests/test_parent_eligibility_uploaded_audio.py`
  - `tests/test_parent_eligibility_v23_voice_short_hit_guard.py`
  - `tests/test_parent_eligibility_v24_drum_loop_steal_guard.py`
  - `tests/test_parent_eligibility_v26_fx_smoke_reverb.py`
  - `tests/test_parent_eligibility_v28_fx_zip_matrix.py`
  - `tests/test_uploaded_regression_audio.py`
- Locked smoke acceptance cases were run one at a time after timeout behavior appeared in combined runs. The protected voice, sax, strings, keys, synth lead, piano, mixed instrument, and police siren cases passed individually.

Not completed as a single monolithic command:

- Full `pytest -q` timed out in this container after passing the early section. The failures found before timeout were repaired and focused tests were rerun individually.
- Full locked smoke acceptance as one command also timed out in this container. Individual case runs were used after that, matching the project instruction to run tests one at a time after timeouts.

## Install note

Use the generated installer in this bundle. It copies only changed source/test/status files and does not create a backup.
