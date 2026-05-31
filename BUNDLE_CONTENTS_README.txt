Aaron Sound Sorter bundle
Created: Sat May 30 09:05:00 PDT 2026
Mode: AI_HANDOFF_WITH_BRAINS
Project root source: /Volumes/T9/testbed/Aaron_Sound_Sorter
Includes active root brain JSONs: yes
Includes locked smoke acceptance audio fixtures: 1
Locked smoke expected cases: 32
Locked smoke audio fixtures included: 35
Locked smoke missing fixtures: 0
Cleaned before bundling: yes

Required first read:
- AI_READ_THIS_FIRST.md

Default purpose:
- AI handoff bundle for code review, QA tooling, and architecture work.
- Includes code, tests, commands, tools, docs, active root brain JSONs, and small QA fixtures.
- A receiving AI must be able to run ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE.command without Aaron separately adding fixtures.

Excluded by design:
- training folders
- large sample folders
- generated reports
- real sort outputs
- virtual environments
- git metadata
- large archives
- general audio files outside tests/acceptance/locked_smoke_v1/samples
