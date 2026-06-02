# Aaron Sound Sorter v31.137b Repack Full Install

This is a packaging-only repack of the v31.137 acceptance synth/siren/keys patch.

## Purpose

The previous patch bundle installed the source code correctly, but the helper command was packaged at:

```text
commands/RUN_V31137_TARGETED_CHECKS.command
```

while the instructions also referenced:

```text
commands/quality/RUN_V31137_TARGETED_CHECKS.command
```

This repack includes and installs both paths so either command works.

## Source files included

```text
src/aaron_sound_sorter/domain/roles.py
src/aaron_sound_sorter/engine/family_claim_arbiter.py
```

## Behavior fixes contained in the source files

- Adds/restores low rhythmic drum-loop role evidence so low tonal kick loops do not collapse into bass-loop policy.
- Protects measured tonal alert/siren FX from clean pitched loop flattening into generic Instrument Loops.
- Keeps synth lead restoration from stealing strings, pads, electric keys, and sax cases.
- Preserves acoustic piano vs electric piano routing behavior from the v31.137 patch.

## Expected confirmation after install

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
pytest
```

Expected result from Aaron's verified local run before this repack:

```text
725 passed, 1 xpassed
```

Run either helper command after install:

```bash
./commands/RUN_V31137_TARGETED_CHECKS.command
./commands/quality/RUN_V31137_TARGETED_CHECKS.command
```
