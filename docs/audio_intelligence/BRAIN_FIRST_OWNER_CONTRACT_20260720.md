# Brain-First Owner Contract - 2026-07-20

## Goal

Aaron Sound Sorter should become an AI-first audio tool. Static Python should
not keep growing into a hand-written category classifier. The preferred path is:

```text
measured audio fingerprint
  -> trainable shape memory
  -> trainable physics memory
  -> trainable voter-role memory
  -> trainable owner memory
  -> lightweight safety guardrails
  -> folder output or review
```

## Rule

When a GUI correction or trusted seed panel teaches a memory brain, that memory
is allowed to own source identity if measured top-family/body evidence is
compatible.

Static code may block only catastrophic contradictions, such as:

- learned Drum owner over clean sustained pitched instrument body
- learned Instrument owner over obvious hard drum body
- learned FX owner over obvious drum loop or clean source-like instrument,
  unless FX motion/body evidence is also present
- learned non-Voice instrument owner using only Voice body evidence
- narrow acoustic instrument leaf whose trusted measured F0 is outside a
  conservative physical range for that instrument family

Static code should not flatten high-confidence learned owner labels into broad
fallbacks like `Instruments/Instrument Loops/Loops` just because old panel
thresholds are undecided.

## Branch Agreement Rule

When the raw brain winner is a concrete non-Voice instrument and PhysicsVoter's
top internal label agrees with the same broad source branch, weak unmatched
side-signals must stand down.

Example:

```text
Brain:   Instruments/Woodwinds/Saxophone/Loops
Physics: Instruments/Brass and Woodwinds/Loops
Shape:   pitched loop / phrase body
Side:    weak Voice-like or siren-like score
Result:  keep the sax/woodwind owner unless learned Voice memory actually
         matches or measured vocal-role evidence is strong.
```

This is not a sax rescue. It is a voter contract: a side panel that is not
matched by learned memory and does not prove its own role cannot erase coherent
owner evidence from Brain plus Physics.

## Measured Pitch Range Guard

`instrument_pitch_ranges.py` is a lightweight proof guard, not an identity
voter. It may inspect only internal taxonomy labels and source-blind measured
pitch facts. It never looks at input filenames, source folders, or pack names.

The guard stands down for broad parent buckets and low-confidence pitch
tracking. It only emits `_TO_REVIEW/Instrument Pitch Range Conflict` when a
narrow instrument leaf, such as Saxophone or Flute, has trusted F0 evidence
outside a wide hard range. This keeps an impossible register from becoming a
confident sax/woodwind-style leaf while preserving normal sax, piano, guitar,
voice, and user-trained owner-memory routing.

## Current Product Increment

`LearnedOwnerAuthorityClaimProducer` now emits generic learned-owner claims for
matched memory labels across Drums, Instruments, and FX. Voice is still kept as
a named claim source for compatibility, but it is no longer the only trained
owner with arbitration authority.

The claim producer validates only broad body compatibility:

- Drums need drum/percussive body or shape.
- Instruments need non-Voice instrument body or pitched/tonal shape, unless
  the target is Voice and Voice body is validated separately.
- FX need FX motion, impact, texture, foley, or designed-sound body/shape.

The learned memory label owns the deep destination. This is the first concrete
step toward shrinking static leaf rules.

## Decision Ownership Audit

The project now has an explicit ownership report:

```bash
cd /path/to/Aaron_Sound_Sorter
./commands/quality/RUN_DECISION_OWNERSHIP_AUDIT.command path/to/Aaron_Sorted_Sounds_manifest.csv
```

This report classifies each row as:

- `learned_memory`
- `trained_brain_consensus`
- `trained_brain_assisted`
- `static_measured_contract`
- `review`
- `unowned_or_legacy`

A category fix is not fully brain-first until the relevant regression row is
owned by `learned_memory` or by the older trained brain lanes with compatible
measured evidence. Static contracts may still prevent catastrophic family
crossings, but they should be treated as training debt, not as the final product
shape.

The report also flags:

- `learned_memory_blocked=True`
- `static_over_brain=True`
- broad fallback/review placements

These flags are the first places to inspect when GUI corrections appear not to
matter.

## Current Model Limitation

The existing brains are not yet a modern embedding model. They are
feature-neighborhood memory lanes over engineered audio fingerprints. They can
generalize when the relevant timbre, envelope, onset, loop, and harmonic/noise
features are close enough, especially through the newer pitch/register-invariant
signature paths. They are weaker when a category has sparse examples, when a
teacher example is too far from nearby variations, or when a static boundary
contract reviews/overrides the row before memory owns it.

That means the near-term AI-first path is:

1. Audit ownership after every suspicious run.
2. Promote trusted corrections into owner, voter-role, physics, and shape
   memory brains.
3. Add small measured body contracts only when they prevent catastrophic family
   contradictions.
4. Once memory-backed tests cover a static contract, retire or weaken that
   static branch.

## Development Contract For Future AI Agents

- Add training or memory evidence before adding rescue code.
- Keep shape, physics, voter-role, and owner brains separate.
- Let static code guard families, not invent leaf identity.
- If a static guard and a trained memory disagree, inspect the measured body
  contract and memory confidence before changing thresholds.
- Add regression tests that prove memory generalizes beyond one corrected file.
- Keep runtime sorting source-name blind.
- Do not declare a category fixed until the ownership audit shows whether the
  final decision came from learned memory, old trained brains, or static code.

## Next Steps

1. Use the decision ownership audit on every GUI/smoke run and convert static
   successes into trusted memory seed panels.
2. Add trusted seed panels for common categories that still fall into broad
   buckets: Piano, Koto/plucked strings, Sax/Woodwinds, Synth Arps, Voice,
   Risers, Impacts, and Percussion Loops.
3. Move more ShapeVoter and PhysicsVoter leaf thresholds into trainable memory
   panels.
4. Retire static rescue branches only after a memory-backed test proves the new
   brain path covers the same safety behavior.
5. Keep branch-compatibility helpers centralized in `decision_helpers.py`; do
   not copy local parsers into claim producers or the arbiter.
