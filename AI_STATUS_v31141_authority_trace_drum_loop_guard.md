# AI Status v31.141 Authority Trace and Drum Loop Guard

## Purpose

This patch addresses the architecture problem where higher-level arbitration could hide or override lower-level voter behavior. The goal was not to retune drum thresholds. The goal was to make final authority visible and reduce broad drum-loop overreach when pitched repetition is the better measured role.

## Files changed

- `src/aaron_sound_sorter/domain/models.py`
- `src/aaron_sound_sorter/infrastructure/report_writer.py`
- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`

## Architecture changes

### 1. Added authority tracing to final decisions

`ConsensusDecision` now carries `authority_trace`.

The manifest now includes these columns:

- `raw_consensus_claim`
- `raw_consensus_family`
- `winner_after_pick`
- `winner_after_pick_family`
- `post_mutator_count`
- `post_mutators_fired_json`
- `final_claim_source`
- `authority_trace_json`

This exposes cases where the final placement was changed after the raw winner. That was the hidden failure mode.

### 2. Wrapped post-winner mutators

`FamilyClaimArbiter.adjudicate()` now records every post-winner mutator that actually changes the claim.

This does not remove all old mutators yet. It makes them visible so future fixes can be based on evidence instead of guessing.

### 3. Tightened broad drum-loop eligibility

Added a measured pitched-repetition decoy guard. Broad Drum Loops support is blocked when the sound is a clean pitched repeated phrase with weak drum-body evidence.

This protects sax, keys, synth, and other pitched repetitions from being promoted into Drum Loops just because they are rhythmic.

### 4. Blocked two bad post-winner releases for real drum one-shots

Random percussion validation exposed a snare that won a Drums claim and then got moved to FX Blip or broad Instrument Loops by later rescue logic.

The guard now blocks clean-tonal FX/instrument release when the current winning claim is a drum one-shot and measured evidence still supports a struck drum/percussion hit.

## Test results

### Compile and audits

```text
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
PASS

python3 tools/audit_no_source_name_sorting.py --project-root . --verbose
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
Scanned Python files: 94
```

### Pytest

```text
python3 -m pytest -q \
  tests/test_claim_arbiter_architecture.py \
  tests/test_v3197_no_role_score_shaping_architecture.py \
  tests/test_decision_core_golden_failure_regressions.py

11 passed
```

Earlier in this patch cycle before the final small guard, these also passed:

```text
tests/test_low_level_physics_subpanels.py  13 passed
tests/test_phase4_shape_voter.py          17 passed
```

The final guard only touches arbiter release conditions, not low-level physics or shape extraction.

### Locked smoke acceptance

All 32 locked smoke acceptance cases were run one at a time after the first wrapper timeout pattern showed up.

Result: 32 passed.

Important representative passes:

```text
kick_clean_cs_ne_monroe                    PASS -> Drums/Kick Drums/Generic Kick/One Shots
snare_clean_jackbaby                       PASS -> Drums/Snares/Acoustic Snare/One Shots
clap_clean_clap2                           PASS -> Drums/Claps Snaps Slaps/Hand Clap/One Shots
drum_loop_full_95                          PASS -> Drums/Drum Loops/Loops
sax_loop_aajbl_11                          PASS -> Instruments/Woodwinds/Saxophone/Loops
sax_loop_aajbl_78_cm                       PASS -> Instruments/Woodwinds/Saxophone/Loops
wet_sax_jazzhiphop_15                      PASS -> Instruments/Woodwinds/Saxophone/Loops
wet_sax_scy097_no_voice                    PASS -> Instruments/Woodwinds/Saxophone/Loops
wet_sax_hiphoptapes_no_voice               PASS -> Instruments/Woodwinds/Saxophone/Loops
mixed_piano_sax_chopart_never_voice        PASS -> Instruments/Instrument Loops/Loops
electric_keys_ews_not_sax_or_guitar        PASS -> Instruments/Keys/Electric Piano/Loops
police_fx_siren_not_piano_or_sax           PASS -> FX/Designed Noise FX/Siren/Long FX
```

### Random percussion ZIP validation

Used `/mnt/data/one_shot_percussive_sounds.zip` and created three fixed-seed random sections, 6 files each.

Initial result before final guard:

```text
section 1: 5 Drums, 0 Review, 1 Other -> FX/Designed Noise FX/Blip/One Shots
section 2: 5 Drums, 0 Review, 1 Other -> FX/Designed Noise FX/Blip/One Shots
section 3: 4 Drums, 2 Review, 0 Other
```

Inspection showed section 2's non-drum file was `200515.wav`, metadata name `SD13x03-Pearl-LP,TSn-HdC-v10.wav`, tagged snare/drum. It had won a Drums claim, then post-winner release moved it out. The final guard fixed that case:

```text
section 2 rerun after guard:
6 Drums, 0 Review, 0 Other
200515.wav -> Drums/Rims and Sticks/Rimshot/One Shots
```

Section 1 still has one suspicious non-drum output:

```text
158956.wav -> FX/Designed Noise FX/Blip/One Shots
metadata name: Jembay Hit 13 Rim.wav
```

That case did not win a Drums claim first. It came through the raw FX path. I did not force-fix it in this patch because that would require improving compact struck tonal percussion claim production, not just blocking a bad post-winner release.

## Known remaining issue

The arbiter still has too many post-winner mutators. This patch makes them visible and blocks the most obvious bad releases found during random percussion validation, but it does not finish the larger cleanup of converting every post-winner family move into a formal pre-pick claim.

The next architecture-safe step is to add tests that assert acceptable `post_mutator_count` and allowed mutator names for known regression cases.
