# Phase 4 Two-Voter Redesign Status

Date: 2026-05-12

## Purpose

This patch starts the architecture described in `Aaron_Sound_Sorter_Two_Voter_Redesign_Handoff_20260511.md`.

The goal is to stop adding hidden reroutes and make every future placement explainable through two explicit voters:

- `BrainVoter`: learned memory and similarity against the trained brain.
- `PhysicsVoter`: learned category fact-profile residuals against the same brain JSON.

Both voters return ranked `CategoryGuess` objects. `ConsensusRunner` only trusts a category when both voters share it strongly.

## Current Stage

This is a trace-first migration slice.

The existing sorter output is not made authoritative through the two-voter runner yet. Instead, `run_real_sort_preview()` now writes the two-voter trace into the manifest so a big smoke run can show where old placement and new consensus disagree.

## Changed Files

- `src/aaron_sound_sorter/two_voter.py`
  - Adds `SharedAudioFacts`, `CategoryGuess`, `VoterResult`, `ConsensusDecision`, and `TwoVoterTrace`.
  - Adds `BrainVoter`, `PhysicsVoter`, and `ConsensusRunner`.
  - Adds manifest serialization helpers for the required trace columns.

- `src/aaron_sound_sorter/preview.py`
  - Adds two-voter manifest columns to `real_sort_preview_manifest.csv`.
  - Keeps current final placement unchanged for this stage.

- `src/aaron_sound_sorter/api.py`
  - Exposes the new two-voter runtime through the existing public API wrapper.

- `tests/test_two_voter_redesign.py`
  - Adds regression tests for no hidden reroute, no overlap review, weak consensus review, broken/tiny review, physics profile ranking, and manifest trace JSON.

## New Manifest Columns

- `brain_vote_1` through `brain_vote_5`
- `physics_vote_1` through `physics_vote_5`
- `shared_winner`
- `shared_winner_brain_rank`
- `shared_winner_physics_rank`
- `shared_winner_combined_rank_score`
- `consensus_status`
- `final_decision_reason`
- `shared_facts_json`
- `vote_trace_json`

## Real-File Probe

Using the current `stage4_folder_brain.json`:

```text
Vocal Phrase We Up 140bpm.wav
two-voter consensus: _TO_REVIEW / No Strong Voter Consensus
reason: best shared category's top family matched neither voter's first-choice top family

DOJO_FBP_Female_Vocal_Shout.wav
two-voter consensus: _TO_REVIEW / No Strong Voter Consensus
reason: best shared category's top family matched neither voter's first-choice top family
```

This means the new trace layer no longer treats those vocal files as safe Drums or safe Risers. The current production placement path still needs to be compared before the two-voter result becomes authoritative.

## Tests Run

```text
python3 -m pytest -q tests/test_two_voter_redesign.py
8 passed

python3 -m pytest -q \
  tests/test_phase4_fx_transition_guard.py \
  tests/test_phase4_voice_guard.py \
  tests/test_stage4_dynamic_runtime_contracts.py \
  tests/test_broad_family_coherence.py
13 passed
```

`tests/test_stage4_code_sanity_audit.py` collected zero pytest tests in this checkout.

## Recommended Next Step

Run the real FX smoke command and compare:

- old final placement: `final_top`, `final_internal_label`, `review_reason`
- new trace: `consensus_status`, `shared_winner`, `final_decision_reason`, `vote_trace_json`

If the trace looks safer across the broad sample, the next migration stage is to make `ConsensusRunner` authoritative for final placement and quarantine the old hidden reroute guards.
