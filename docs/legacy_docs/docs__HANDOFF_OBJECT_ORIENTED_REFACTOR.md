# Aaron Sound Sorter Object-Oriented Refactor Handoff

## What changed in this bundle

This is an architecture-safe first split. It does not rewrite the classifier math. It creates stable object-oriented seams around the existing sorter so the next changes can move code out of the legacy file without breaking the working CLI.

## New structure

```text
Aaron_Sound_Sorter.py                  compatibility launcher
src/aaron_sound_sorter/
  __init__.py
  core.py                              Sample, AudioAnalysis, Candidate, SortDecision
  audio.py                             AudioAnalyzer DSP service
  brain.py                             KnowledgeBase reference brain service
  judge.py                             SortingCommittee final decision service
  archive.py                           AudioIngestor ZIP/folder staging service
  reports.py                           ReportWriter manifest/report service
  cli.py                               CLI entry point shim
  legacy.py                            original proven sorter code, kept intact for compatibility
tests/
  test_audio_analyzer.py
  test_brain_and_committee.py
  test_legacy_cli_contract.py
commands/RUN_PYTEST.command
```

## Why this is the safe first move

The original file mixed domain models, WAV/AIFF reading, feature extraction, reference brain loading, judge logic, output writing, training helpers, mining helpers, CLI parsing, and self-tests in one script.

This bundle separates the public architecture without changing the math path:

1. `Sample` owns file identity and analysis state.
2. `AudioAnalyzer` owns DSP extraction.
3. `KnowledgeBase` owns brain loading and nearest competitor search.
4. `SortingCommittee` owns the final decision interface.
5. `ReportWriter` owns manifest/report writing.
6. `AudioIngestor` owns ZIP/folder staging.
7. `legacy.py` keeps the current working implementation until each function group is moved safely.

## Testing change

Tests now live in `tests/` as pytest tests. The bundled tests cover:

- importing the new package architecture
- analyzing a generated WAV through `Sample` plus `AudioAnalyzer`
- querying a tiny in-memory `KnowledgeBase`
- judging through `SortingCommittee`
- confirming the old launcher still prints CLI help

## What is not done yet

The legacy file is still large. That is intentional for this pass. Moving thousands of lines at once would create unnecessary risk.

The next safe moves are:

1. Move pure dataclasses from `legacy.py` into `core.py`, then import them from there.
2. Move all audio loading and DSP functions from `legacy.py` into `audio.py`.
3. Move brain JSON read/write and nearest-distance scoring into `brain.py`.
4. Move judge helper functions into `judge.py` as small Gate classes.
5. Replace the old `--self-test` implementation with pytest invocation or delete it after equivalent pytest coverage exists.
6. Keep the CLI behavior stable after each move.

## Commands

Run all tests:

```bash
cd /path/to/Aaron_Sound_Sorter_OO_Refactor
python3 -m pytest -q
```

Or double-click / run:

```bash
commands/RUN_PYTEST.command
```

Normal sorter compatibility command still works:

```bash
python3 Aaron_Sound_Sorter.py "/path/to/samples.zip" "/path/to/output_folder"
```

## Architectural rule for the next AI

Do not start by rewriting `judge_label` or training behavior. Keep using the OO facade and move one responsibility at a time. Every move must pass pytest before the next move.
