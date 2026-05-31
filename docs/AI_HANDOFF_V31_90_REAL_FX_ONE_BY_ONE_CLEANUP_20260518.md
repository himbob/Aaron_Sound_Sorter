# AI Handoff v31.90: Real FX One-by-One Cleanup

## Controlling architecture

Follow `docs/REQUIRED_Aaron_SoundSorter_v3177_Architecture_Review.md`.

The important rule is unchanged: final folder placement belongs to `FamilyClaimArbiter` and public path resolution belongs to `PlacementResolver`. Brain lanes, shape, roles, and parent eligibility must produce claims or evidence, not scattered final decisions.

## What was wrong after v31.89

The latest full FX result still had these failure classes:

1. Sax and brass loops were no longer mostly going to Human/Voice, but a few were still routed to review or brittle one-shot leaves.
2. Baby-brain brass/woodwind recall was still too aggressive on generic pitched material. One synth lead was pulled into `Instruments/Brass and Woodwinds/Loops`.
3. Shape sanity could still pick a source-specific one-shot leaf for a measured loop/phrase when a broad Instrument Loops claim was safer.

## What changed in v31.90

### 1. Repeated tonal hit phrase broadening

`DecisionCoreV2._broad_instrument_loop_for_one_shot_leaf()` now treats short repeated tonal `hit_with_tail` material as broad Instrument Loop evidence when:

- the raw result is an Instrument one-shot leaf,
- the sound has multiple pitched events,
- sustain is high,
- percussive evidence is low,
- the raw leaf is not a voice/vocal/breath/mouth leaf.

This fixed real sax/brass material that had been narrowing to `Instruments/Brass/Trumpet/One Shots`.

### 2. Generic pitched broad loop claims can beat ambiguous FX review

`FamilyClaimArbiter` now allows `role_sanity_generic_pitched_broad_instrument_loop` to compete across from ambiguous FX leaves such as alarm/siren/glitch/coins/motor style false positives. This fixed the remaining sax loop that was going to `_TO_REVIEW/Measured Role Conflict` even though measured evidence and PhysicsVoter supported broad Instrument Loops.

### 3. Baby brass/woodwind recall is no longer allowed to steal stronger broad loop evidence

If a `baby_recall_brass_woodwind_claim` competes with a stronger broad Instrument Loops claim from shape, role, candidate true-bucket rescue, or top-family sanity, the broad Instrument Loops claim wins.

This fixed the synth lead that was being stolen by the baby brass/woodwind lane.

## Real validation performed here

Validation used the real uploaded brains from `Archive.zip` and the real `FX_Aaron2.zip` samples.

A full 130-file FX validation was run as one file at a time using the real sorter code and real brain files. It was not a bulk sort.

Output summary:

```text
Processed: 130
Needs review: 0

Top folders:
     96  Instruments
     29  Drums
      5  FX
```

The compact one-by-one result table is included in this bundle:

`docs/V31_90_REAL_FX_ONE_BY_ONE_SUMMARY_20260518.csv`

The sorter summary is included here:

`docs/V31_90_REAL_FX_ONE_BY_ONE_SORTER_SUMMARY_20260518.txt`

## Specific real sample outcomes checked

- `QUp_DzU_DrumLp_03_100bpm.wav` -> `Drums/Drum Loops/Loops`
- `RHSH_Brass_Saxophones_01_keyEm_103bpm.wav` -> `Instruments/Instrument Loops/Loops`
- `SC_[[Challenger]]_Melody_Multi_Sample_2_130BPM_Amin.wav` -> `Instruments/Instrument Loops/Loops`
- `01_WCS_No_Safety_BPM92_D#min__Bells.wav` -> `Instruments/Instrument Loops/Loops`
- `02_CHAMPAPI_vol_6_Pad_F#m_122bpm.wav` -> `Instruments/Instrument Loops/Loops`
- `03.strings_77bpm_Ebm.wav` -> `Instruments/Instrument Loops/Loops`
- `AV5_5_94bpm_Hit 2.wav` -> `Instruments/Instrument Loops/Loops`
- `HipHopTapes_28_Saxophone_D#m_90bpm.wav` -> `Instruments/Instrument Loops/Loops`
- `SCY097_01_Sax_Loop_KeyGm_90bpm_02.wav` -> `Instruments/Instrument Loops/Loops`
- `GS_Synth_Gangsta_Lead_G#min_97bpm.wav` -> `Instruments/Instrument Loops/Loops`
- `Riser Short Effect.wav` -> `FX/Structural and Transitional FX/Risers and Builds/Short Riser/Long FX`
- `DOJO_CGNB_Female_Vocal_Shot_01_D.wav` -> `FX/Human and Voice FX`
- `AAJV_Kick_Happiness.wav` -> `Drums/Kick Drums/Generic Kick/One Shots`

## Remaining caution

The current patch deliberately prefers broad `Instruments/Instrument Loops/Loops` for many pitched/mixed musical loops instead of over-narrowing. That is safer than Sax/Voice/FX/Bass steals, but it means the next quality pass can improve depth inside Instruments after the parent-family errors stay stable.

Vocal phrase loops currently often land in broad Instrument Loops while clear vocal one-shots/shouts land in Human/Voice FX. Do not “fix” this by making Human/Voice broad claims stronger without adding strict true-voice evidence, or sax/reed false positives will return.

## Next recommended work

1. Run the installed v31.90 bundle against Aaron's full FX pack on his Mac.
2. Review only hard-family mistakes first:
   - real drums outside Drums,
   - real sax/brass/keys/strings/synth loops outside Instruments,
   - concrete risers/impacts outside FX,
   - true vocal shots outside Human/Voice.
3. Avoid optimizing leaf depth until parent-family errors are rare.
4. If improving vocal phrase loops, add a true-voice claim that requires direct/body voice evidence plus a real Human/Voice candidate. Do not rely on `vocal_phrase` shape alone.
