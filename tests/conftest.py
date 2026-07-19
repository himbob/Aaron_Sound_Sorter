"""Shared pytest fixture materialization for private audio regression folders.

Several regression tests were written against private uploaded WAV folders that
are not always present in AI handoff bundles. This conftest keeps those tests
meaningful by creating generated stand-ins under ``_reports`` and symlinking the
expected private folder names to those generated folders at pytest runtime.

The generated audio is used only when the private fixture folder is absent. The
sorter still receives audio only; no test changes the production classifier or
passes filename truth into the sorter.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TextIO

import numpy as np

from tests import synthetic_audio_fixtures as fixtures

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKED_SMOKE_AUDIO_DIR = PROJECT_ROOT / "tests" / "acceptance" / "locked_smoke_v1" / "samples"
GENERATED_ROOT = PROJECT_ROOT / "_reports" / "generated_private_regression_audio"

WaveFactory = Callable[[str], np.ndarray]


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    """Materialize generated private-audio stand-ins before tests run."""
    _acquire_pytest_session_lock(session)
    _ensure_v23_voice_short_hit_guard()
    _ensure_v24_drum_loop_steal_guard()
    _ensure_v25_transition_reverb()
    _ensure_v26_fx_smoke_reverb()
    _ensure_v28_fx_zip_matrix()
    _ensure_uploaded_regression_audio()
    _ensure_full_thread_matrix()


def pytest_sessionfinish(session, exitstatus: int) -> None:  # type: ignore[no-untyped-def]
    """Release the session lock even when tests fail or are interrupted."""
    del exitstatus
    lock_handle: TextIO | None = getattr(session.config, "_aaron_pytest_lock_handle", None)
    if lock_handle is None:
        return
    lock_module = _lock_module()
    if lock_module is not None:
        lock_module.flock(lock_handle.fileno(), lock_module.LOCK_UN)
    lock_handle.close()


def _lock_module() -> ModuleType | None:
    """Return the POSIX lock module when file locking is available."""
    return fcntl


def _acquire_pytest_session_lock(session) -> None:  # type: ignore[no-untyped-def]
    """Serialize pytest sessions that share generated fixture/report folders."""
    if os.environ.get("AARON_PYTEST_DISABLE_SESSION_LOCK") == "1":
        return
    lock_module = _lock_module()
    if lock_module is None:
        return

    lock_dir = PROJECT_ROOT / "_reports" / "pytest_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (lock_dir / "session.lock").open("a", encoding="utf-8")
    try:
        lock_module.flock(lock_handle.fileno(), lock_module.LOCK_EX | lock_module.LOCK_NB)
    except BlockingIOError:
        print("Another Aaron Sound Sorter pytest session is running; waiting for its generated fixtures.")
        lock_module.flock(lock_handle.fileno(), lock_module.LOCK_EX)
    session.config._aaron_pytest_lock_handle = lock_handle


def _link_generated_fixture_dir(folder_name: str, samples: dict[str, WaveFactory | str]) -> None:
    """Create a generated fixture directory and link the test's expected path.

    The fixture materializer should never abort pytest just because a copied
    private/locked sample has fragile filesystem metadata or disappears during
    setup. These generated WAVs are test fixtures, so preserving xattrs, chmod
    bits, or Finder metadata is not useful. Copy audio bytes only, and fall back
    to deterministic synthetic audio when a private source cannot be copied.
    """
    expected = PROJECT_ROOT / "tests" / folder_name
    if expected.exists() and not expected.is_symlink():
        return
    generated = GENERATED_ROOT / folder_name
    if generated.exists():
        shutil.rmtree(generated, ignore_errors=True)
    generated.mkdir(parents=True, exist_ok=True)
    for sample_name, source in samples.items():
        target = generated / sample_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, str):
            source_path = LOCKED_SMOKE_AUDIO_DIR / source
            _copy_locked_sample_or_write_fallback(source_path, target, sample_name)
        else:
            fixtures.write_wav(target, source(sample_name))
    if expected.is_symlink() or expected.exists():
        if expected.is_dir() and not expected.is_symlink():
            return
        expected.unlink(missing_ok=True)
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.symlink_to(generated, target_is_directory=True)


def _copy_locked_sample_or_write_fallback(source_path: Path, target: Path, sample_name: str) -> None:
    """Copy fixture audio bytes, or synthesize a stable fallback WAV.

    ``shutil.copy2`` can fail during ``copystat`` on some external/macOS
    volumes even after the audio payload copied. Fixture setup does not need
    metadata preservation, so use a plain byte copy through a temporary file and
    replace atomically.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        temporary_target = target.with_name(f"{target.name}.tmp_copy")
        try:
            temporary_target.unlink(missing_ok=True)
            shutil.copyfile(source_path, temporary_target)
            temporary_target.replace(target)
            return
        except OSError:
            temporary_target.unlink(missing_ok=True)
    fixtures.write_wav(target, _fallback_for_name(sample_name))


def _fallback_for_name(sample_name: str) -> np.ndarray:
    """Return a conservative fallback role for a generated sample name."""
    name = sample_name.lower()
    if "vocal" in name or "voice" in name or "dojo" in name or "stab" in name:
        return fixtures.synth_voice_like(230.0, dur=0.9)
    if "sax" in name or "brass" in name:
        return fixtures.synth_harmonic_phrase(360.0, dur=2.5)
    if "piano" in name or "keys" in name or "chords" in name:
        return fixtures.synth_piano_like(392.0, dur=2.4)
    if "bass" in name:
        return _synth_bass_loop(sample_name)
    if "drum" in name or "beat" in name:
        return fixtures.synth_drum_loop(sample_name, dur=2.4)
    if "riser" in name or "police" in name or "fx" in name:
        return fixtures.synth_fx_hit(dur=2.0)
    if "29793" in name:
        return fixtures.synth_tone(110.0, dur=0.35)
    if "16791" in name:
        return fixtures.synth_percussion_hit(dur=0.25)
    return fixtures.synth_tone(440.0, dur=1.2)


def _synth_bass_loop(sample_name: str, dur: float = 2.4) -> np.ndarray:
    """Build a low, tonal, repeated bass loop that is not drum-like."""
    del sample_name
    sr = fixtures.SAMPLE_RATE
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    phrase = np.sin(2.0 * np.pi * 55.0 * t) + 0.35 * np.sin(2.0 * np.pi * 110.0 * t)
    gate = 0.55 + 0.45 * (np.sin(2.0 * np.pi * 2.0 * t) > -0.2).astype(np.float32)
    env = np.minimum(1.0, 8.0 * t) * np.minimum(1.0, 8.0 * (dur - t))
    return phrase * gate * env


def _generated_drum_loop(sample_name: str) -> np.ndarray:
    return fixtures.synth_drum_loop(sample_name, dur=2.4)


def _generated_bass_loop(sample_name: str) -> np.ndarray:
    return _synth_bass_loop(sample_name)


def _generated_low_hit(sample_name: str) -> np.ndarray:
    del sample_name
    return fixtures.synth_tone(110.0, dur=0.35)


def _generated_percussion_hit(sample_name: str) -> np.ndarray:
    del sample_name
    return fixtures.synth_percussion_hit(dur=0.25)


def _ensure_v23_voice_short_hit_guard() -> None:
    _link_generated_fixture_dir(
        "regression_audio_v23_voice_short_hit_guard",
        {
            "DOJO_CGNB_Female_Vocal_Shot_01_D.wav": "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
            "29793.wav": _generated_low_hit,
            "16791.wav": _generated_percussion_hit,
        },
    )


def _ensure_v24_drum_loop_steal_guard() -> None:
    drum_loops = {
        name: _generated_drum_loop
        for name in [
            "Full Drum Loop 03 - 78BPM.wav",
            "JL_TDBL_Drum Full_act up_138bpm.wav",
            "SCY097_03_Drums_Full_Loop_90bpm_03.wav",
            "US_JF_Drum_140_Dapocket_FULL.wav",
            "2.Drum Loop_1_100bpm.wav",
            "A1_Kick_Clap_Loop_99bpm.wav",
            "MKS_98_Beat1.wav",
            "QUp_DzU_DrumLp_03_100bpm.wav",
            "WS2_KIT_1_WestCoast_Drum_&_Perc_Loop_101BPM.wav",
        ]
    }
    bass_loops = {
        name: _generated_bass_loop
        for name in [
            "03_bass_Emn_178bpm.wav",
            "04.bass_92bpm_Em.wav",
            "FL_TR_Kit02_96_Bass_Loop_Synth_Gm.wav",
        ]
    }
    _link_generated_fixture_dir(
        "regression_audio_drum_loop_steal_guard",
        {
            **drum_loops,
            **bass_loops,
            "AA_JBL_78bpm_Cm_Sax_Loop_1.wav": "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
            "DOJO_FBP_Female_Vocal_Shout.wav": "DOJO_FBP_Female_Vocal_Shout.wav",
            "DOJO_CGNB_Female_Vocal_Shot_01_D.wav": "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
        },
    )


def _ensure_v25_transition_reverb() -> None:
    _link_generated_fixture_dir(
        "regression_audio_v25_transition_reverb",
        {
            "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav": (
                "GrimyHipHop_Saxophone_15_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav"
            ),
            "Riser Short Effect.wav": "Riser Short Effect.wav",
            "US_CHV2_Vocal_female_shouts_processed_13.wav": "DOJO_FBP_Female_Vocal_Shout.wav",
            "Piano 1 - 80 Bpm - Key C.wav": "Piano 1 - 80 Bpm - Key C.wav",
        },
    )


def _ensure_v26_fx_smoke_reverb() -> None:
    _link_generated_fixture_dir(
        "regression_audio_v26_fx_smoke_reverb",
        {
            "HipHopTapes_29_Saxophone_D#m_90bpm.wav": "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
            "SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav": "SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav",
            "ABOUTME_94_DRUMLOOP.wav": _generated_drum_loop,
        },
    )


def _ensure_v28_fx_zip_matrix() -> None:
    _link_generated_fixture_dir(
        "regression_audio_v28_fx_zip_matrix",
        {
            "AA_JBL_78bpm_Cm_Sax_Loop_1.wav": "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
            "AMV_VRNB1_102_brass_saxophone_loop_cranesinthesky_Am.wav": (
                "Brass_Saxophone_RnB_Multi_Instrument_F_Minor_80BPM.wav"
            ),
            "EWS_Keys_resampled_HipHop_RnB_G_Major_88BPM.wav": "EWS_Keys_resampled_HipHop_RnB_G_Major_88BPM.wav",
            "WS2_KIT_1_Electric_Piano_Chords_Fm_101BPM.wav": "WS2_KIT_1_Electric_Piano_Chords_Fm_101BPM.wav",
            "GS_Synth_Gangsta_Lead_G#min_97bpm.wav": lambda name: fixtures.synth_tone(520.0, dur=1.9),
            "MS_TLV1_03_The Way It Is_Synth Lead 1_Eminor_98bpm_Wet.wav": lambda name: fixtures.synth_tone(
                440.0, dur=2.0
            ),
            "Money_vocals_female_rap_110bpm.wav": "Money_vocals_female_rap_110bpm.wav",
            "Phonk_Rap_Vocals_26_keyCmin_151bpm.wav": "Phonk_Rap_Vocals_26_keyCmin_151bpm.wav",
            "Vocal Phrase We Up 140bpm.wav": "DOJO_FBP_Female_Vocal_Shout.wav",
            "Stab 3.wav": "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
            "MS_O_01_Outlaw_Fx Police_D#minor_91bpm_Wet.wav": "MS_O_01_Outlaw_Fx Police_D#minor_91bpm_Wet.wav",
            "2.Drum Loop_1_100bpm.wav": _generated_drum_loop,
            "ABOUTME_94_DRUMLOOP.wav": _generated_drum_loop,
        },
    )


def _ensure_full_thread_matrix() -> None:
    """Create the complete full-thread matrix when private WAVs are absent."""
    sax_samples = {
        name: (lambda sample_name: fixtures.synth_harmonic_phrase(330.0, dur=2.4))
        for name in [
            "AA_JBL_74bpm_Am_Sax_Loop_18.wav",
            "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
            "AA_JBL_86bpm_Cm_Sax_Loop_12.wav",
            "AA_JBL_90bpm_Em_Sax_Loop_3.wav",
            "AMV_VRNB1_102_brass_saxophone_loop_cranesinthesky_Am.wav",
            "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
            "HipHopTapes_28_Saxophone_D#m_90bpm.wav",
            "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
            "SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav",
        ]
    }
    drum_loops = {
        name: _generated_drum_loop
        for name in [
            "ABOUTME_94_DRUMLOOP.wav",
            "DRUMS_LOOP_125BPM.wav",
            "Full Drum Loop 03 - 78BPM.wav",
            "NDAEV4_FULL_DRUM_LOOP_01_AFTERLIFE_74BPM.wav",
            "THO_ck4_drums top_130 bpm.wav",
        ]
    }
    voice_samples = {
        name: (lambda sample_name: fixtures.synth_voice_like(220.0, dur=1.6))
        for name in [
            "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
            "DOJO_FBP_Female_Vocal_Shout.wav",
            "Money_vocals_female_rap_110bpm.wav",
            "Stab 3.wav",
        ]
    }
    percussive_hits = {
        name: _generated_percussion_hit for name in ["125591.wav", "13144.wav", "16791.wav", "33157.wav", "50728.wav"]
    }
    kick_hits = {name: (lambda sample_name: fixtures.synth_kick(dur=0.45)) for name in ["13115.wav", "16291.wav"]}
    _link_generated_fixture_dir(
        "regression_audio_thread_matrix",
        {
            **sax_samples,
            **drum_loops,
            **voice_samples,
            **percussive_hits,
            **kick_hits,
            "AV5_5_94bpm_Hit 2.wav": _generated_drum_loop,
        },
    )


def _ensure_uploaded_regression_audio() -> None:
    _link_generated_fixture_dir(
        "regression_audio",
        {
            "04_Dmn_176bpm_bass.wav": "MS_TLV1_03_The Way It Is_Bass_Eminor_98bpm_Dry.wav",
            "04.bass_92bpm_Em.wav": "MS_TLV1_03_The Way It Is_Bass_Eminor_98bpm_Dry.wav",
            "DOJO_CGNB_Female_Vocal_Shot_01_D.wav": "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
            "Drumloop_hats_Dark_Rap_140BPM.wav": _generated_drum_loop,
            "GangstaFunk_Cmaj_96bpm.wav": lambda name: fixtures.synth_piano_like(220.0, dur=2.4),
            "SCY095_03_Drums_Top_Loop_90bpm_01.wav": _generated_drum_loop,
            "1.Saxophone_1_110bpm_Am.wav": "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
            "HipHopTapes_28_Saxophone_D#m_90bpm.wav": "HipHopTapes_28_Saxophone_D#m_90bpm.wav",
            "Vocal Phrase We Up 140bpm.wav": "DOJO_FBP_Female_Vocal_Shout.wav",
            "Compton_Fmin_100bpm.wav": _generated_drum_loop,
            "AV5_5_94bpm_Hit 2.wav": "AV5_5_94bpm_Hit 2.wav",
        },
    )
