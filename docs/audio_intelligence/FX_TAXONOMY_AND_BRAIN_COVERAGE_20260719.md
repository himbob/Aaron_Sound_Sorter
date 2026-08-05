# FX Taxonomy And Brain Coverage - 2026-07-19

## Bottom Line

The project has enough FX labels to start, but the trusted FX evidence is not
deep enough yet. The current full brain exposes many FX folders, while the
locked curated FX tree and GUI catalog cover only a small subset. That means
the sorter can appear knowledgeable while still having weak trusted evidence
for many FX roles.

Do not replace the sorter taxonomy in one jump. Add a structured FX role
ontology, audit coverage, and seed the trainable memory brains from trusted
examples first.

## Research Anchors

- Universal Category System (UCS): professional sound-effects category system
  used by sound libraries and metadata tools. Best long-term reference for
  pro SFX category breadth and material/source naming.
- Freesound Broad Sound Taxonomy (BST): useful high-level split between Music,
  Instrument samples, Speech, Sound effects, and Soundscapes. This supports the
  project rule that voice and soundscapes should not be swallowed by generic FX.
- AudioSet/FSD: useful trainable event ontologies and dataset references, but
  less producer-specific than UCS. These are better as model-training
  vocabulary inputs than as direct output folders.

## Local Findings

Latest local audit context:

- Full folder brain: 99 FX public labels.
- Locked curated FX tree: 100 FX slots, but only 34 audio references.
- Locked curated FX coverage report: 86 of 99 full-brain FX labels had zero
  locked curated training audio.
- GUI taxonomy catalog: 14 FX public labels, but the current catalog-only FX
  labels do not overlap the 99 active full-brain FX labels. The GUI is exposing
  a tiny future-category subset, not the active FX brain taxonomy.
- New voter-memory lanes have broad FX roles, but not enough role depth to
  replace static FX code yet.

This explains why FX remains confusing: the label vocabulary is wider than the
trusted evidence behind it.

## Recommended FX Role Layer

Use `config/fx_role_ontology_v1.json` as the first brain-facing map:

- `fx_transition_motion`: risers, builds, downlifters, drops, whooshes, sweeps,
  reverses.
- `fx_impact_punctuation`: impacts, booms, slams, sub hits, crashes, breaks.
- `fx_designed_electronic`: blips, beeps, lasers, alarms, sirens, radio,
  electrical, glitches, stutters.
- `fx_machine_transport`: engines, motors, servos, mechanical loops.
- `fx_foley_material_event`: doors, footsteps, keys, coins, glass, wood, metal,
  paper, cloth, splashes.
- `fx_human_creature_event`: non-musical human/body/creature/animal FX.
- `fx_texture_soundscape`: ambience, drones, granular textures, water, weather,
  hiss, static, vinyl noise.

This is intentionally role-first. The goal is not to force exact leaves. The
goal is to train memory brains to recognize what an FX sound is doing before
the folder arbiter picks a final place.

## Training Policy

Do not mass-resort all FX training data blindly.

Use this order:

1. Audit coverage with `commands/quality/RUN_FX_TAXONOMY_AUDIT.command`.
2. Pick role gaps with little or no locked curated audio.
3. Build small trusted panels: 5-12 listened examples per role.
4. Seed `stage4_voter_memory_brain.json`, `stage4_physics_memory_brain.json`,
   `stage4_shape_memory_brain.json`, and user-memory folder brain with explicit
   labels only.
5. Re-run acceptance, FX smoke, instrument smoke, and percussion smoke.
6. Only then consider deeper taxonomy changes or a larger curated rebuild.

## What Not To Do

- Do not let FX become the fallback for weird audio.
- Do not use input filenames or source folders as runtime evidence.
- Do not shrink the category library to make tests pass.
- Do not train broad acceptance prefixes as if they were exact labels.
- Do not use AudioSet/FSD labels as producer-folder truth without a mapping and
  human review.

## Commands

Run the coverage audit:

```bash
cd /path/to/Aaron_Sound_Sorter
./commands/quality/RUN_FX_TAXONOMY_AUDIT.command
```

The report is written under:

```text
_reports/fx_taxonomy_audit/run_YYYYMMDD_HHMMSS/
```

Key files:

- `fx_role_coverage.csv`
- `fx_label_coverage.csv`
- `README_FX_TAXONOMY_AUDIT.txt`

## Next Brain-First Implementation Step

Use the audit report to generate a reviewed FX memory seed panel. The next code
should train memory brains from explicit trusted rows, not add more static FX
rescues.
