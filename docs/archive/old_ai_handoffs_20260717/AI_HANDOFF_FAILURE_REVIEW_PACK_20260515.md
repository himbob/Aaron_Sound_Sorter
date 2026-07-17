# AI Handoff: Failure Review Pack Builder

Date: 2026-05-15

## Purpose

Aaron asked for a script that groups failure cases with category folders and symlinks so he can inspect/listen locally. The same script can make a small copied-audio mini pack when the AI needs actual WAVs.

## Files

```text
tools/build_failure_review_pack.py
commands/quality/BUILD_FAILURE_REVIEW_PACK.command
README_FAILURE_REVIEW_PACK_20260515.md
```

## Why symlinks

The sample library is large. Symlinks let Aaron inspect failures locally without copying audio. Use copied-audio mode only for small test packs.

## Default command

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/BUILD_FAILURE_REVIEW_PACK.command
```

## Uploadable mini-pack

```bash
COPY_AUDIO=1 MAX_AUDIO_FILES=20 ./commands/quality/BUILD_FAILURE_REVIEW_PACK.command
```

## Important

Symlink zips do not give another machine access to the audio unless the same absolute paths exist. For AI-side local testing, ask Aaron for the copied-audio mini pack.
