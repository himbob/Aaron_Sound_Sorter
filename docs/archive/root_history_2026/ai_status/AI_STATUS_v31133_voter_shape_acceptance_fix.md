# Archived: AI Status v31.133 voter shape acceptance fix

## Scope

This bundle repairs the v31.132 regressions where unrelated material collapsed into `Instruments/Synths/Pads/Loops` or `Instruments/Strings/Loops`, and applies the voter/ShapeVoter review findings from Aaron.

## Root failure classification

The failures were not one audio category bug. They were routing/authority bugs:

1. A final structure invariant was allowed to broaden pitched phrase material into generic Instrument loop folders too early, before voice, reed, drum, and short-hit invariants could protect the correct broad bucket.
2. Brain lane authority diagnostics overcounted sibling lanes, making a label look backed by more lanes than it really was.
3. ShapeVoter rename drift meant `pitched_phrase_shape` no longer triggered the old vocal phrase weighting path.
4. Shape evidence exposed only the top few shape scores even though downstream physics layers query many specific shapes.
5. The node-by-node test runner generated bad node IDs after nested test folders and could hang in this environment after PASS output.

## Code fixes

- `brain_lane_competence.py`
  - `count_lanes_matching_parent` and `count_lanes_matching_top` now count only the lanes where the evaluated label is present.
  - Added/kept regression coverage for sibling-lane overcount.

- `brain_ensemble_policy.py`
  - Vocal weighting now recognizes both `vocal_phrase` and `pitched_phrase_shape`.

- `shape_voter.py`
  - `shape_scores` now stores the complete ranked shape-score list instead of truncating to five.

- `layered_physics_scorer.py`
  - Simplified the duplicate `PhysicsLayerDecision` construction with `dataclasses.replace`.

- `consensus.py`
  - Added compatibility helpers for `vocal_phrase` and `pitched_phrase_shape`.
  - Allows valid high-strength shape/top-family claims when measured shape evidence supports the top family.
  - Adds a true voice shape rescue path for non-voice FX conflict cases.

- `family_claim_arbiter.py`
  - Allows the voice-shape rescue claim to compete only with measured voice support.
  - Protects short true voice one-shots from loop buckets.
  - Broadens contradicted instrument one-shot leaves to the correct broad loop bucket.
  - Reed/brass one-shot leaves with loop body go to broad Brass and Woodwinds loops, while non-reed one-shot leaves go to broad Instrument Loops.
  - Keeps strong drum one-shots out of false instrument/FX loop broadening.

- `run_pytest_node_by_node.py`
  - Fixed bad node ID generation after nested test folders.
  - Added `--start-index` and `--max-nodes` chunking.
  - Uses subprocess.run and disables third-party pytest plugin autoload for node-level runs.

## Validation completed in the AI container

### Quality gate

Passed:

```bash
make ai-check VENV_DIR=/mnt/data/ass_repair/Aaron_Sound_Sorter/.venv_phase4
```

This included compileall, Ruff check, Ruff format check, generated-junk scan, and no-source-name-sorting audit.

### Focused regression tests

Passed:

```bash
pytest -q tests/test_decision_core_fx_smoke_remaining_review_rows.py \
  tests/test_claim_arbiter_real_panel_surrogates.py \
  tests/test_brain_lane_competence_v2.py \
  tests/test_phase4_shape_voter.py::test_decisive_top_family_gate_needs_shape_or_role_support \
  tests/test_two_voter_redesign.py::test_consensus_rehomes_strong_vocal_shape_to_instrument_voice_bucket
```

Result: 23 passed.

### User-listed failure nodes checked directly

Passed individually in this container:

- `THO_ck4_drums top_130 bpm.wav`
- `DOJO_CGNB_Female_Vocal_Shot_01_D.wav`
- `DOJO_FBP_Female_Vocal_Shout.wav`
- `Money_vocals_female_rap_110bpm.wav`
- `Stab 3.wav`
- percussion hits `125591.wav`, `13144.wav`, `16791.wav`, `33157.wav`, `50728.wav`
- kick-like hits `13115.wav`, `16291.wav`
- short sub-heavy kick guard
- uploaded vocal stab guard
- v23 short voice/percussion guards
- v25 vocal phrase from FX pack
- v28 Money vocal guard
- uploaded bass loop `04_Dmn_176bpm_bass.wav`
- uploaded short female vocal shot
- phase4 shape-voter consensus test
- two-voter vocal shape consensus test

Some exact Mac-side parametrized nodes are not collected in the container because generated fallback fixtures reduce the private real-audio matrix when private folders are missing. Those exact nodes must still be run on Aaron's Mac.

### Locked smoke acceptance

All 32 locked smoke acceptance cases passed in four-case chunks.

Chunks run:

```bash
python3 tools/locked_smoke_acceptance.py run --project-root /mnt/data/ass_repair/Aaron_Sound_Sorter --start-index 1 --max-cases 4 --timeout-seconds 180
python3 tools/locked_smoke_acceptance.py run --project-root /mnt/data/ass_repair/Aaron_Sound_Sorter --start-index 5 --max-cases 4 --timeout-seconds 180
python3 tools/locked_smoke_acceptance.py run --project-root /mnt/data/ass_repair/Aaron_Sound_Sorter --start-index 9 --max-cases 4 --timeout-seconds 180
python3 tools/locked_smoke_acceptance.py run --project-root /mnt/data/ass_repair/Aaron_Sound_Sorter --start-index 13 --max-cases 4 --timeout-seconds 180
python3 tools/locked_smoke_acceptance.py run --project-root /mnt/data/ass_repair/Aaron_Sound_Sorter --start-index 17 --max-cases 4 --timeout-seconds 180
python3 tools/locked_smoke_acceptance.py run --project-root /mnt/data/ass_repair/Aaron_Sound_Sorter --start-index 21 --max-cases 4 --timeout-seconds 180
python3 tools/locked_smoke_acceptance.py run --project-root /mnt/data/ass_repair/Aaron_Sound_Sorter --start-index 25 --max-cases 4 --timeout-seconds 180
python3 tools/locked_smoke_acceptance.py run --project-root /mnt/data/ass_repair/Aaron_Sound_Sorter --start-index 29 --max-cases 4 --timeout-seconds 180
```

Every chunk showed protected failures: 0.

## Not completed

A complete 763-node one-by-one pytest sweep did not complete in the AI container. The node runner was fixed and small chunks passed, but larger chunks still hit container/tool runtime limits. Full `pytest -q -x` no longer showed an assertion failure before the container timed out, but it did not finish.

Aaron should still run the full local checks on the Mac, where the private real-audio fixture matrix exists:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
make ai-check
python3 tools/run_pytest_node_by_node.py --timeout-sec 180 --out-root _reports/pytest_node_by_node_local tests
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 1 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 5 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 9 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 13 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 17 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 21 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 25 --max-cases 4
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 29 --max-cases 4
```
