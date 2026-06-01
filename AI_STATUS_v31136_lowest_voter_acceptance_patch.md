# AI Status: v31136 lowest-voter acceptance patch

## Purpose

Emergency continuation patch after v31135.

This bundle includes the current code state plus a final arbiter safeguard for the acceptance failure:

- `strings_loop_77_ebm` / `03.strings_77bpm_Ebm.wav`
- Wrong result observed: `FX/Designed Noise FX/Siren/Long FX`
- Expected top family: `Instruments`
- Accepted: `Instruments/Strings`, `Instruments/Instrument Loops`, or `_TO_REVIEW`

## Code-level intent

The fix blocks late FX tonal-alert/siren invariants from overriding a measured nonpercussive pitched musical loop when internal instrument-family evidence exists.

This is not a filename rescue. It uses measured role/shape/body evidence plus internal instrument candidate or PhysicsVoter branch support.

## Files included

- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
- `src/aaron_sound_sorter/voters/physics_instrument_layer.py`
- `src/aaron_sound_sorter/voters/physics_top_family_layer.py`
- `src/aaron_sound_sorter/voters/shape_voter.py`
- `src/aaron_sound_sorter/domain/roles.py`
- `tests/acceptance/locked_smoke_v1/samples/HipHopTapes_28_Saxophone_D#m_90bpm.wav`
- `commands/quality/RUN_V31136_TARGETED_REPROS_ONE_BY_ONE.command`

## Testing status

No full pytest suite was run for this v31136 bundle because Aaron requested the code bundle immediately and no more testing.

I did run Python compile on the edited `family_claim_arbiter.py` only to catch syntax breakage.

## Install

```bash
cd ~/Downloads
unzip v31136_lowest_voter_acceptance_patch.zip
cd v31136_lowest_voter_acceptance_patch
./INSTALL_NO_BACKUP.command
```

## Suggested local test

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_V31136_TARGETED_REPROS_ONE_BY_ONE.command
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
```
