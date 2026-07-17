# AI Handoff: Archive Failure Repair, 2026-05-16

## Non-negotiable rule

Do not ship or describe a sorter bundle as fixed unless all of these have run or the response explicitly says it is only a candidate patch:

1. `RUN_NO_SOURCE_NAME_SORTING_AUDIT.command`
2. `RUN_ALL_PYTESTS_CHUNKED.command`
3. The focused real-audio / golden audit relevant to the change, especially `RUN_FX_AARON2_ONE_BY_ONE_GOLDEN_AUDIT.command` for FX_Aaron2 regressions.

Synthetic tests are required, but synthetic tests alone are not acceptance.

## What this patch addresses

The uploaded chunked pytest report showed failures in real fixture tests even while decision-core unit tests were green. That exposed the false-green failure mode: testing isolated decision functions while the full CLI path still failed.

The repaired `decision_core_v2.py` keeps the true-bucket rescues but avoids the broad over-corrections that broke:

- short percussion hits becoming FX/Instruments/loops
- short sub-heavy kick hits becoming Instruments/Bass one-shots
- vocal one-shots becoming animal/foley leaves or review
- wet sax becoming Human/Voice or Drums
- drum loops being demoted to Instruments/Bass or generic Instrument Loops

## Test status from assistant-side sandbox

The assistant ran:

- no-source-name audit: PASS, 21 scanned production files
- focused decision-core regression set: PASS
- selected real CLI spot checks:
  - `03_bass_Emn_178bpm.wav -> Instruments/Bass/Bass Loops`
  - `04.bass_92bpm_Em.wav -> Instruments/Bass/Bass Loops`
  - `SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav -> Instruments/Instrument Loops/Loops`

The assistant could not run Aaron's full Mac-only audio fixture set because those fixture folders are not present in the sandbox. Aaron's Mac remains the acceptance environment for the full chunked test and full FX audit.
