# Named Sample Coverage Panel - 2026-07-21

## Purpose

This pass adds a repeatable offline QA harness for broad real-sample coverage.
The harness may use filenames and source folders to select likely named samples,
but only as a post-sort oracle. The production sorter must remain source-name
blind.

The goal is to stop one-file whack-a-mole by producing a category failure map:

- expected named category
- final sorter folder
- brain top vote
- physics top vote
- shape vote
- final claim source

## Modern AI Direction

The design direction is brain-first with lightweight rails:

- Use hierarchy, not flat labels. AudioSet models sound events as a hierarchy
  across human, animal, instrument, environmental, and miscellaneous sounds:
  https://research.google.com/audioset/ontology/
- Use embeddings and trainable memories as first-class evidence. YAMNet maps
  short log-mel patches into embeddings and predicts AudioSet classes:
  https://www.tensorflow.org/hub/tutorials/yamnet
- Keep descriptor families broad and time-aware. Essentia's batch extractor
  computes spectral, time-domain, rhythm, tonal, and high-level descriptors:
  https://essentia.upf.edu/streaming_extractor_music.html
- For future mixed-loop understanding, source separation is the right research
  direction. Demucs-style models separate drums, bass, vocals, and other stems,
  with some variants adding guitar and piano:
  https://github.com/facebookresearch/demucs

The immediate implementation is not a deep model rewrite. It is a scaffold that
lets us measure failures, seed trusted memory, and verify that trained owner
memory beats unsafe static fallback code when the measured contract agrees.

## Files Added

- `tools/named_sample_coverage_panel.py`
- `commands/quality/BUILD_NAMED_SAMPLE_COVERAGE_PANEL.command`
- `commands/quality/RUN_NAMED_SAMPLE_COVERAGE_PANEL.command`
- `commands/quality/AUDIT_NAMED_SAMPLE_COVERAGE_PANEL.command`
- `tests/test_named_sample_coverage_panel.py`

## Full Panel Built

Latest cleaned full panel:

`_reports/named_sample_coverage/run_20260721_103325`

The cleaned selector found 2,871 strong named examples. It did not pad to 5,000
with generated `*_train_*` / `*_val_*` artifacts because those were not trusted
producer names.

## Micro-Panel Result

Cleaned 36-file micro-panel:

`_reports/named_sample_coverage/run_20260721_102225`

Before trusted memory seeding:

- 36 cases
- 8 pass
- 28 fail

After a 7-row explicit trusted seed:

- 36 cases
- 14 pass
- 22 fail

After adding the FX impact owner guard:

- 36 cases
- 15 pass
- 21 fail

## Trusted Seed Applied

Trusted memory seed report:

`_reports/trusted_memory_seed/run_20260721_102454`

Rows applied: 7

Immediate improvements:

- Hi-hat one-shot moved from guiro/scrape to `Drums/Hi Hats/Closed Hat/One Shots`
- Cymbal crash moved from metallic percussion to `Drums/Cymbals/Crash Cymbal/One Shots`
- Floor tom moved from kick to `Drums/Toms/Floor Tom/One Shots`
- Female vocal phrase moved from FX riser to `Instruments/Voice/Vocal Loops/Loops`
- Reverse crash moved from riser to `FX/Structural and Transitional FX/Reverses and Tails/Reverse Cymbal/Long FX`
- Vinyl noise moved from drum loop to `FX/Textures/Noise and Static/Vinyl Noise/Long FX`
- Boom impact required a code guard because learned memory, brain, and physics
  all agreed on FX/Boom, but an old low-kick release still stole it.

## Code Fixes From Findings

1. Sax ownership over synth-loop broadening

   A real sax loop had Brain and Physics both voting Sax, but a measured synth
   loop invariant won because the sax measured claim was not considered a real
   candidate and ShapeVoter called the sample `bass_phrase`.

   Fix:

   - Measured sax loop claims are real candidates when the taxonomy path is real.
   - Synth-loop measured claims stand down when Brain/Physics support Sax and
     reed/sax evidence is comparable to synth evidence.

2. FX impact ownership over low-kick release

   A boom impact had learned voter memory, learned physics memory, Brain, and
   Physics all voting `FX/Impacts and Hits/Boom/One Shots`, but a late low-kick
   parent release changed it to Kick.

   Fix:

   - The low-kick release now stands down when learned memory or Brain/Physics
     owns an FX impact and the shape is impact/hit-with-tail.

## Remaining Failure Clusters

These are not fixed yet:

- Snare vs hat/cymbal: snare one-shot went to closed hat.
- Rim/stick vs snare: sidestick/snap went to acoustic snare.
- Shaker loops: brain sees shaker loop, physics/final broadens to drum loops.
- Guitar and plucked strings: guitar/banjo often fall to broad loops or voice.
- Piano/electric piano: key loops are often broadened to generic instrument loops.
- Organ/synth stabs: short tonal stabs can still get stolen by drums or FX impact.
- FX drops/whooshes/glitches/textures: FX motion and texture roles need more
  curated memory and lower-level shape-voter training.
- Foley/animals: these are thin in trusted examples and should not be judged
  from sparse, ambiguous filenames.

## Next Giant Run

Run the cleaned full panel when you have time:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
AARON_SORT_WORKERS=6 python3 Aaron_Sound_Sorter.py sort \
  _reports/named_sample_coverage/run_20260721_103325/input_panel \
  _reports/named_sample_coverage/run_20260721_103325/sort_output \
  --no-zip --workers 6

python3 tools/named_sample_coverage_panel.py audit \
  --expected _reports/named_sample_coverage/run_20260721_103325/named_sample_panel_expected.csv \
  --manifest _reports/named_sample_coverage/run_20260721_103325/sort_output/Aaron_Sorted_Sounds_manifest.csv \
  --output-dir _reports/named_sample_coverage/run_20260721_103325/audit
```

## Later Mixed-Loop Phase

For long mixed loops, the current sorter should only make a minor pass:

- detect likely mixed/compound role
- avoid false single-source leaves
- preserve diagnostics for secondary sources

The later project should add optional stem/segment analysis:

- separate drums, bass, vocals, and other stems
- run source voters on each stem
- summarize dominant and secondary sources
- classify loops by composition, not just whole-file average
