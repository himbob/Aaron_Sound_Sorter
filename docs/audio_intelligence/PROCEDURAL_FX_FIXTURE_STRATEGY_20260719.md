# Procedural FX Fixture Strategy

This note tracks the next test/training direction for FX behavior. The goal is
to make FX physics and shape voters trainable and testable without relying only
on whichever private ZIP happens to be available during a debugging session.

## Why This Exists

FX has been the most dangerous broad family because it can steal drums,
instruments, vocals, sax, pads, and synthetic loops when the lower voters are
too vague. Random sample-pack smoke tests are still useful, but they are not a
complete safety harness because they are slow, private, and hard to reproduce.

Procedural fixtures give us deterministic audio shapes for low-level voter
coverage:

- riser/build: upward chirp plus rising noise envelope
- downlifter/drop: downward chirp plus low bloom and decay
- whoosh/sweep: broadband shaped-noise motion
- impact/tail: fast transient plus low/noisy decay body
- beep/alarm: gated tonal alert pulses
- glitch/stutter: chopped digital tone/noise bursts

These fixtures do not replace real audio. They are calibration anchors that
prove a panel can still hear the behavior it claims to measure.

## Research Notes

The implementation follows common procedural-audio patterns:

- `librosa.chirp` documents generated sine sweeps for rising/falling frequency
  motion: https://librosa.org/doc/latest/generated/librosa.chirp.html
- SciPy `signal.chirp` documents linear/quadratic/log sweeps useful for FX
  transition fixtures: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html
- Web Audio API examples use oscillators, gain envelopes, and filters as the
  core building blocks for browser/game synthesis:
  https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
- Procedural SFX tools commonly expose impacts, lasers, whooshes, movement,
  weather, and machine-like generators as first-class sound-design primitives:
  https://www.procedural-audio.com/

## Current Code

The shared fixture module is:

`tests/synthetic_audio_fixtures.py`

The first locked panel tests are:

`tests/test_procedural_fx_fixture_panels.py`

The fixtures are intentionally source-name blind. Test names and filenames are
stable identifiers only; sorter production code must never use those names as
classification evidence.

## How To Extend

Add one fixture at a time, then test the lowest voter layer first:

1. Add a procedural waveform function in `tests/synthetic_audio_fixtures.py`.
2. Extract the fingerprint with `make_fingerprint_safe`.
3. Convert to `SharedAudioFacts`.
4. Assert the intended physics panel wakes up.
5. Only after the low-level panel is trustworthy, add a full-sort smoke case.

Good next fixtures:

- reverse cymbal / reverse swell
- granular riser
- sub impact versus kick one-shot
- static bed versus radio/electrical burst
- fire crackle versus vinyl crackle
- water splash versus rain bed
- animal-like chirp versus flute/synth chirp

## Training Policy

Procedural audio can seed panel coverage and smoke-test shape brains, but it
should not dominate final product training. Human-approved real samples remain
the highest-trust evidence. Procedural fixtures should be marked as synthetic in
training manifests so future learned brains can weight them as calibration
examples, not as producer-library truth.
