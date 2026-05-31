#!/usr/bin/env python3
"""
aaron_training_physics_review_plan.py

Physics-only training cleanup PREVIEW tool.

Reads a Stage 4 train_feature_manifest.csv and creates a review tree containing
ONLY proposed changes as symlinks.

Workflow:
  1. Run this preview tool.
  2. Listen in the generated review tree.
  3. Delete symlinks for proposals you reject.
  4. Leave symlinks for proposals you accept.
  5. Later run a separate apply tool.

No files are moved, deleted, or renamed by this preview tool.
No filename words are used for decisions. Filenames are only used for display/logs.

v0.4.48 changes:
  - Loop proposals use independent hit/transient evidence when audio is readable.
  - Long-tail single-hit sounds are protected as one-shots across drums, instruments, and FX.
  - Removal/quarantine proposals are OFF by default and much stricter.
  - FX never moves to _LOOPS; FX repeated/long material moves to _LONG_FX.
  - Drum repeated material now separates pure/single-source loops from mixed drum loops.
  - Independent-hit detection now requires renewed hits to fall to a deep valley first.
    This protects metallic/cymbal amplitude beating from being treated as repeated hits.
  - Pure repeated claps/kicks/cymbals/percussion go to that category's _LOOPS; mixed drum material goes to Drums/Drum Loops.
  - Existing loop folders are searched both ways: single-hit loops move back to category _ONE_SHOTS when the category is known.
  - Proposal destinations are validated so top-level Drums/_ONE_SHOTS, Instruments/_ONE_SHOTS, or FX/_ONE_SHOTS are never created.
  - Removal outliers ignore structure features such as duration/event count, because
    those should become move proposals, not remove proposals.
  - Plan folders are unique even when repeated tests run inside the same second.
  - Test command scripts respect PROJECT_ROOT so temp installs can be tested safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import wave
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None

# These are for category/timbre outlier removal review only. Do not include
# pure structure fields here. Structure errors should be proposed as moves.
OUTLIER_FEATURE_COLS = [
    # Deliberately exclude duration/event/tail/decay/crest here. Those are
    # structure or envelope-shape questions, and they caused normal long crash
    # cymbals to look like removal candidates. Removal review should be reserved
    # for timbre/category weirdness, not normal long tails.
    "sub_bass_ratio_lt_150hz",
    "bass_ratio_150_500hz",
    "mid_ratio_500_2000hz",
    "high_total_ratio_gt_2000hz",
    "zcr_mean",
    "spectral_flatness_mean",
    "spectral_entropy_mean",
    "centroid_slope_norm",
    "stereo_width",
    "pitch_confidence",
]

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au", ".mp3", ".m4a"}


@dataclass
class HitAnalysis:
    audio_read: bool
    independent_hit_count: int
    raw_peak_count: int
    hit_span_ratio: float
    first_hit_ratio: float
    dominant_peak_ratio: float
    strong_peak_ratio: float
    energy_75_ratio: float
    energy_90_ratio: float
    decay_tail_ratio: float
    front_loaded: bool
    single_dominant_hit: bool
    reason: str
    drum_hit_family_counts: str = ""
    dominant_hit_family: str = ""
    hit_family_diversity: int = 0
    drum_mixed: bool = False


def safe_link_name(index: int, source_name: str) -> str:
    suffix = Path(source_name).suffix
    stem = Path(source_name).stem
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._") or "sample"
    return f"proposal_{index:06d}__{stem[:120]}{suffix}"


def f(row: dict[str, str], col: str, default: float = 0.0) -> float:
    try:
        val = row.get(col, "")
        if val is None or val == "":
            return default
        x = float(val)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def s(row: dict[str, str], col: str) -> str:
    return str(row.get(col, "") or "")


def rel_to_training(source_path: str, training_root: Path) -> Path | None:
    # Fast path first. Most Stage 4 reports store absolute paths containing this
    # marker, and resolving thousands of missing absolute parents is slow.
    marker = "training/locked_curated_v1/"
    text = str(source_path).replace("\\", "/")
    if marker in text:
        return Path(text.split(marker, 1)[1])

    p = Path(source_path)
    try:
        # Do not resolve the final path through a symlink. We need the training-tree
        # entry path, not the target audio file on T9.
        root_resolved = training_root.resolve(strict=False)
        parent_resolved = p.parent.resolve(strict=False)
        rel_parent = parent_resolved.relative_to(root_resolved)
        return rel_parent / p.name
    except Exception:
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def locate_reports_dir(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    for c in [run_dir / "rebuild_and_real_preview" / "reports", run_dir / "reports", run_dir]:
        if (c / "train_feature_manifest.csv").exists():
            return c
    raise SystemExit(f"Could not find train_feature_manifest.csv under: {run_dir}")


def make_unique_plan_dir(out_root: Path, run_name: str, stamp: str) -> Path:
    """Return a unique plan folder even if two tests/previews run in the same second."""
    base = out_root / f"plan_{run_name}_{stamp}"
    for attempt in range(1000):
        candidate = base if attempt == 0 else out_root / f"plan_{run_name}_{stamp}_{attempt:03d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise SystemExit(f"Could not create unique plan folder under: {out_root}")


def replace_marker(parent: Path, old_markers: Iterable[str], new_marker: str) -> Path:
    parts = list(parent.parts)
    old = set(old_markers)
    for i, part in enumerate(parts):
        if part in old:
            parts[i] = new_marker
            return Path(*parts)
    return parent / new_marker


STRUCTURE_MARKERS = {"_ONE_SHOTS", "_ONE_SHOT", "_LOOPS", "_LOOP", "_LONG_FX", "_LONG_RUNNING"}
TOP_LEVEL_NAMES = {"Drums", "Instruments", "FX", "Textures", "World Instruments"}
TOP_LEVEL_STRUCTURE_MARKERS = {"Drums", "Instruments", "FX", "Textures"}
DRUM_LOOP_DEST = Path("Drums") / "Drum Loops"


def path_from_label(label: str) -> Path | None:
    parts = [p.strip() for p in str(label or "").replace("\\", "/").split("/") if p.strip()]
    if not parts:
        return None
    # Normalize legacy World Instruments into Instruments.
    if parts[0] == "World Instruments":
        parts = ["Instruments", "World and Special Instruments"] + parts[1:]
    return Path(*parts)


def strip_structure_markers(path: Path) -> Path:
    parts = [p for p in path.parts if p not in STRUCTURE_MARKERS]
    return Path(*parts) if parts else Path()


def category_base_for_row(row: dict[str, str], rel: Path) -> Path:
    """Return the musical category path, without _ONE_SHOTS/_LOOPS/_LONG_FX.

    Prefer the manifest expected_label because global loop folders such as
    Drums/Drum Loops do not retain the original one-shot subcategory. Fall back
    to the current training parent with structure markers stripped.
    """
    label_path = path_from_label(s(row, "expected_label"))
    if label_path is not None and len(label_path.parts) >= 2:
        return strip_structure_markers(label_path)
    return strip_structure_markers(rel.parent)


def category_structure_dest(row: dict[str, str], rel: Path, marker: str) -> Path | None:
    base = category_base_for_row(row, rel)
    if len(base.parts) < 2:
        return None
    if base.parts[0] in TOP_LEVEL_NAMES and len(base.parts) == 1:
        return None
    return base / marker / rel.name


def drum_loop_dest(rel: Path) -> Path:
    return DRUM_LOOP_DEST / rel.name


def invalid_structure_destination(proposed_rel: Path) -> bool:
    parts = proposed_rel.parts
    if len(parts) >= 2 and parts[0] in TOP_LEVEL_STRUCTURE_MARKERS and parts[1] in STRUCTURE_MARKERS:
        return True
    return bool("_LOOPS" in parts and parts and parts[0] == "FX")


def strip_loopish_folder(parent: Path) -> Path | None:
    parts = list(parent.parts)
    out: list[str] = []
    removed = False
    for part in parts:
        low = part.lower().strip()
        if part in {"_LOOPS", "_LOOP", "_LONG_FX", "_LONG_RUNNING"}:
            out.append("_ONE_SHOTS")
            removed = True
        elif (
            low in {"loops", "loop", "drum loops", "instrument loops"}
            or low.endswith(" loops")
            or low.endswith(" loop")
        ):
            if not removed:
                out.append("_ONE_SHOTS")
                removed = True
        else:
            out.append(part)
    if not removed:
        out.append("_ONE_SHOTS")
    candidate = Path(*out)
    if invalid_structure_destination(candidate / "dummy.wav"):
        return None
    return candidate


def _read_audio_mono(path: Path, max_seconds: float = 30.0) -> tuple[np.ndarray | None, int, str]:
    """Read enough audio for hit-count analysis. Return mono float32, sr, status."""
    if not path.exists() and not path.is_symlink():
        return None, 0, "missing"
    try:
        if sf is not None:
            info = sf.info(str(path))
            sr = int(info.samplerate or 0)
            if sr <= 0:
                return None, 0, "bad_samplerate"
            frames = int(min(info.frames or 0, max_seconds * sr))
            if frames <= 0:
                return None, sr, "empty"
            data, sr2 = sf.read(str(path), frames=frames, always_2d=True, dtype="float32")
            sr = int(sr2)
            mono = np.mean(np.asarray(data, dtype=np.float32), axis=1)
            return mono, sr, "ok"
    except Exception:
        pass
    try:
        if path.suffix.lower() != ".wav":
            return None, 0, "no_soundfile_for_nonwav"
        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            channels = int(wf.getnchannels())
            sampw = int(wf.getsampwidth())
            frames = min(int(wf.getnframes()), int(max_seconds * sr))
            raw = wf.readframes(frames)
        if sampw == 2:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sampw == 4:
            arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif sampw == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        else:
            return None, sr, f"unsupported_wav_width_{sampw}"
        usable = (arr.size // channels) * channels
        arr = arr[:usable].reshape(-1, channels)
        mono = np.mean(arr, axis=1).astype(np.float32)
        return mono, sr, "ok"
    except Exception as exc:
        return None, 0, f"read_error:{str(exc)[:80]}"


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n <= 1 or x.size == 0:
        return x
    kernel = np.ones(n, dtype=np.float32) / float(n)
    return np.convolve(x, kernel, mode="same")


def _classify_drum_hit_family(y: np.ndarray, sr: int, time_sec: float) -> str:
    """Coarse physics label for a hit window.

    This is not final classification. It only asks whether a repeated drum pattern
    is made from one hit type or from mixed drum hit types. It deliberately stays
    broad so high, tiny hand claps are not over-corrected as cymbals.
    """
    center = int(max(0.0, time_sec) * sr)
    start = max(0, center - int(0.035 * sr))
    end = min(len(y), center + int(0.180 * sr))
    x = y[start:end]
    if x.size < 128:
        return "unknown"
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-8:
        return "unknown"
    x = x / peak
    win = np.hanning(x.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * win)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    total = float(np.sum(spec)) + 1e-12

    def br(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spec[mask]) / total) if np.any(mask) else 0.0

    sub = br(20, 120)
    low = br(20, 250)
    mid = br(250, 2500)
    high = br(2500, sr / 2)
    mag = np.sqrt(spec)
    flat = float(np.exp(np.mean(np.log(mag))) / (np.mean(mag) + 1e-12))
    centroid = float(np.sum(freqs * spec) / total)

    if low >= 0.58 and sub >= 0.20:
        return "kick_low"
    if high >= 0.50 and centroid >= 3800 and flat >= 0.16:
        # This covers hats, cymbals, and very bright tiny claps. The destination
        # decision still uses the current training category, so this does not
        # turn handclaps into cymbals.
        return "bright_metal_or_clap"
    if high >= 0.32 and mid >= 0.16 and flat >= 0.16:
        return "clap_snare_bright"
    if mid >= 0.42:
        return "mid_percussion"
    if high >= 0.38:
        return "bright_percussion"
    return "other"


def _summarize_drum_hit_mix(
    y: np.ndarray, sr: int, starts: np.ndarray, strong: Sequence[int]
) -> tuple[str, str, int, bool]:
    families: list[str] = []
    for i in strong:
        try:
            families.append(_classify_drum_hit_family(y, sr, float(starts[i] / sr)))
        except Exception:
            families.append("unknown")
    counts: dict[str, int] = defaultdict(int)
    for fam in families:
        counts[fam] += 1
    if not counts:
        return "", "", 0, False
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    dominant, dominant_count = ordered[0]
    useful = [
        (k, v)
        for k, v in ordered
        if k not in {"unknown", "other"} and v >= max(2, int(math.ceil(0.30 * max(1, len(families)))))
    ]
    diversity = len(useful)
    # Mixed means two or more substantial named hit families. Ignore "other" as
    # a secondary family here because broad windows around a kick/cymbal tail can
    # produce stray "other" hits. A 70/20 split should stay pure; a 50/50 kick+hat
    # or kick+clap split is mixed.
    mixed = bool(diversity >= 2 and dominant_count <= int(math.floor(0.70 * max(1, len(families)))))
    return ";".join(f"{k}:{v}" for k, v in ordered), dominant, diversity, mixed


def analyze_independent_hits(path: Path) -> HitAnalysis:
    """Count independent hit events, not decay shimmer.

    This is intentionally amplitude-envelope based. It protects long cymbal/crash
    one-shots: a cymbal can produce many spectral flux peaks while decaying, but it
    is still one hit unless there are separated renewed amplitude attacks.
    """
    y, sr, status = _read_audio_mono(path)
    if y is None or sr <= 0 or y.size < 64:
        return HitAnalysis(False, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, status)
    y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = y - float(np.mean(y))
    peak_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if peak_abs <= 1e-8:
        return HitAnalysis(True, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, "silent")
    y = y / peak_abs

    frame = max(128, int(0.020 * sr))
    hop = max(64, int(0.010 * sr))
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    starts = np.arange(0, max(1, y.size - frame + 1), hop)
    env = np.array([float(np.sqrt(np.mean(y[i : i + frame] ** 2))) for i in starts], dtype=np.float32)
    if env.size == 0 or float(np.max(env)) <= 1e-8:
        return HitAnalysis(True, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, "silent_env")
    env = env / float(np.max(env))
    env_s = _moving_average(env, 3)

    # Local maxima above a low threshold. We then require prominence/rise to reject
    # shimmer and decay ripples after a single crash.
    local_maxima: list[int] = []
    for i in range(1, env_s.size - 1):
        if env_s[i] >= env_s[i - 1] and env_s[i] > env_s[i + 1] and env_s[i] >= 0.08:
            local_maxima.append(i)
    if not local_maxima:
        i = int(np.argmax(env_s))
        local_maxima = [i]

    # Enforce minimum distance and keep highest peak in each neighborhood.
    min_sep = max(3, int(0.090 / (hop / sr)))
    selected: list[int] = []
    for i in sorted(local_maxima, key=lambda k: float(env_s[k]), reverse=True):
        if all(abs(i - j) >= min_sep for j in selected):
            selected.append(i)
    selected.sort()

    if not selected:
        return HitAnalysis(True, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, "no_peaks")

    max_peak = max(float(env_s[i]) for i in selected)
    strong: list[int] = []
    for i in selected:
        lo = max(0, i - int(0.120 / (hop / sr)))
        prev_valley = float(np.min(env_s[lo : i + 1])) if i > lo else 0.0
        amp = float(env_s[i])
        prominence = amp - prev_valley
        # A renewed hit needs both enough absolute level and enough local rise.
        if amp >= max(0.16, 0.30 * max_peak) and prominence >= max(0.06, 0.18 * amp):
            strong.append(i)

    if not strong:
        strong = [int(np.argmax(env_s))]

    duration = float(y.size / sr)
    [float(starts[i] / sr) for i in strong]

    # Convert raw strong maxima into independent hits.  A real repeated hit needs
    # a renewed attack from a deep local valley.  Metallic one-shots, especially
    # cymbals and crashes, often have periodic amplitude beating in the decay tail;
    # those peaks do not count unless the envelope fell low enough before them.
    independent: list[int] = []
    independent_debug: list[str] = []
    for pos, i in enumerate(strong):
        lo = max(0, i - int(0.160 / (hop / sr)))
        prev_valley = float(np.min(env_s[lo : i + 1])) if i > lo else 0.0
        amp = float(env_s[i])
        valley_ratio = prev_valley / (amp + 1e-9)
        time_ratio = float((starts[i] / sr) / max(duration, 1e-9))
        # Always allow the first early impact to establish a one-shot.  Later
        # events need a low pre-hit valley.  This is the key difference between
        # handclaps/kicks in a loop and a cymbal tail that wobbles while decaying.
        is_first_impact = bool(pos == 0 and time_ratio <= 0.12)
        is_renewed_hit = bool(pos > 0 and valley_ratio <= 0.38 and amp >= 0.16)
        if is_first_impact or is_renewed_hit:
            independent.append(i)
        independent_debug.append(f"{starts[i] / sr:.3f}:{amp:.2f}/{valley_ratio:.2f}")

    if not independent and strong:
        independent = [strong[0]]

    times = [float(starts[i] / sr) for i in independent]
    span = 0.0 if len(times) <= 1 or duration <= 0 else (max(times) - min(times)) / duration
    first_hit_ratio = (times[0] / duration) if duration > 0 and times else 0.0
    total_peak_energy = sum(float(env_s[i]) for i in independent) + 1e-9
    dominant_peak_ratio = max(float(env_s[i]) for i in independent) / total_peak_energy if independent else 0.0
    strong_peak_ratio = len(independent) / max(1, len(selected))

    # Energy time balance: if most of the envelope energy arrives early, later
    # spectral/onset ticks are probably cymbal shimmer/decay, not independent hits.
    # Use squared envelope for cumulative energy so a long quiet tail does not
    # outweigh the initial strike.
    t = np.arange(env_s.size, dtype=np.float32) * (hop / sr)
    temporal_centroid = float(np.sum(t * env_s) / (np.sum(env_s) + 1e-9) / max(duration, 1e-9))
    env_energy = np.maximum(env_s, 0.0) ** 2
    cum_energy = np.cumsum(env_energy)
    total_energy = float(cum_energy[-1]) + 1e-9

    def energy_ratio_at(q: float) -> float:
        idx = int(np.searchsorted(cum_energy, total_energy * q))
        idx = max(0, min(idx, len(starts) - 1))
        return float((starts[idx] / sr) / max(duration, 1e-9))

    energy_75_ratio = energy_ratio_at(0.75)
    energy_90_ratio = energy_ratio_at(0.90)
    tail_start = int(env_s.size * 0.55)
    head_end = max(1, int(env_s.size * 0.20))
    head_energy = float(np.mean(env_s[:head_end]))
    tail_energy = float(np.mean(env_s[tail_start:])) if tail_start < env_s.size else 0.0
    decay_tail_ratio = float(tail_energy / (head_energy + 1e-9))

    # A long cymbal/crash/impact can have many raw maxima.  It is still one-shot
    # when those maxima do not renew from deep valleys.  Do not classify modulation
    # or beating in a sustained metallic tail as loop structure.
    front_loaded = bool(
        first_hit_ratio <= 0.12 and len(independent) <= 1 and energy_75_ratio <= 0.55 and decay_tail_ratio <= 0.55
    )
    single_decay_hit = bool(
        len(independent) <= 1 and first_hit_ratio <= 0.12 and energy_90_ratio <= 0.75 and tail_energy <= 0.30
    )
    single_dominant = bool(len(independent) <= 1 or single_decay_hit or front_loaded)

    effective_hit_count = 1 if single_dominant else len(independent)
    mix_indices = independent if len(independent) >= 2 else strong
    drum_counts, dominant_family, family_diversity, drum_mixed = _summarize_drum_hit_mix(y, sr, starts, mix_indices)
    independent_debug_text = ";".join(independent_debug[:18])
    return HitAnalysis(
        True,
        int(effective_hit_count),
        int(len(selected)),
        float(span),
        float(first_hit_ratio),
        float(dominant_peak_ratio),
        float(strong_peak_ratio),
        float(energy_75_ratio),
        float(energy_90_ratio),
        float(decay_tail_ratio),
        front_loaded,
        single_dominant,
        f"effective_hits={effective_hit_count} independent_hits={len(independent)} raw_strong_hits={len(strong)} raw_peaks={len(selected)} span={span:.3f} first={first_hit_ratio:.3f} dom={dominant_peak_ratio:.3f} front={front_loaded} e75={energy_75_ratio:.3f} e90={energy_90_ratio:.3f} decay_tail={decay_tail_ratio:.3f} temporal={temporal_centroid:.3f} tail_env={tail_energy:.3f} drum_mix={drum_counts} dominant={dominant_family} mixed={drum_mixed} independent_debug={independent_debug_text}",
        drum_counts,
        dominant_family,
        int(family_diversity),
        bool(drum_mixed),
    )


def hit_analysis_for_row(
    row: dict[str, str], training_root: Path, rel: Path | None, enabled: bool, cache: dict[str, HitAnalysis]
) -> HitAnalysis:
    if rel is None or not enabled:
        return HitAnalysis(False, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, "audio_confirm_disabled")
    key = str(rel)
    if key not in cache:
        cache[key] = analyze_independent_hits(training_root / rel)
    return cache[key]


def legacy_report_repeated_transient_loop_like(row: dict[str, str]) -> bool:
    """Old v0.4.38-style report-only loop trigger, for protection logging only."""
    dur = f(row, "duration_sec")
    events = f(row, "primary_event_count_est")
    span = f(row, "onset_span_ratio")
    reg = f(row, "onset_interval_regularity")
    rate = f(row, "event_rate_hz")
    temporal = f(row, "temporal_centroid_ratio")
    tags = s(row, "physics_tags")
    if dur >= 1.35 and events >= 6 and span >= 0.45 and (reg >= 0.38 or rate >= 1.10):
        return True
    if dur >= 2.0 and events >= 4 and span >= 0.55 and "loop_pulse_candidate" in tags:
        return True
    return bool(dur >= 4.0 and events >= 8 and span >= 0.5 and temporal >= 0.28)


def single_dominant_hit_decay(row: dict[str, str]) -> bool:
    """Report-manifest fallback: one strong early hit with decay/shimmer is not a loop.

    This protects long crash cymbals, impacts, and other one-shots when the
    training audio cannot be opened for direct independent-hit analysis.
    It intentionally requires front-loaded energy and low tail energy, so a
    real pattern spread across the file can still move to loop/long review.
    """
    dur = f(row, "duration_sec")
    events = f(row, "primary_event_count_est")
    f(row, "onset_span_ratio")
    temporal = f(row, "temporal_centroid_ratio")
    tail = f(row, "tail_energy_ratio")
    s(row, "physics_tags")
    if dur <= 0:
        return False
    return bool(
        dur >= 0.8 and events >= 1 and temporal <= 0.24 and tail <= 0.25
        # A crash can have report-level onset span near 0.9 because shimmer peaks
        # occur throughout the decay. Low tail energy + early temporal centroid is
        # enough to protect it when direct audio hit analysis is unavailable.
    )


def report_repeated_transient_loop_like(row: dict[str, str]) -> bool:
    dur = f(row, "duration_sec")
    events = f(row, "primary_event_count_est")
    span = f(row, "onset_span_ratio")
    reg = f(row, "onset_interval_regularity")
    rate = f(row, "event_rate_hz")
    temporal = f(row, "temporal_centroid_ratio")
    f(row, "tail_energy_ratio")
    tags = s(row, "physics_tags")

    # Protect single-hit decay shapes from report-only event overcount.
    if single_dominant_hit_decay(row):
        return False
    if dur >= 1.35 and events >= 6 and span >= 0.45 and (reg >= 0.38 or rate >= 1.10):
        return True
    if dur >= 2.0 and events >= 4 and span >= 0.55 and "loop_pulse_candidate" in tags:
        return True
    return bool(dur >= 4.0 and events >= 8 and span >= 0.5 and temporal >= 0.28)


def repeated_transient_loop_like(row: dict[str, str], hit: HitAnalysis) -> bool:
    if hit.audio_read:
        if hit.single_dominant_hit:
            return False
        if hit.independent_hit_count >= 3 and hit.hit_span_ratio >= 0.42:
            return True
        return bool(hit.independent_hit_count >= 6 and hit.hit_span_ratio >= 0.3)
    return report_repeated_transient_loop_like(row)


def phrase_like_instrument(row: dict[str, str], hit: HitAnalysis) -> bool:
    if s(row, "expected_top") != "Instruments":
        return False
    if hit.audio_read and hit.single_dominant_hit:
        return False
    return (
        f(row, "duration_sec") >= 2.0
        and f(row, "primary_event_count_est") >= 4
        and f(row, "onset_span_ratio") >= 0.45
        and f(row, "pitch_confidence") >= 0.20
        and f(row, "temporal_centroid_ratio") >= 0.22
    )


def long_fx_or_texture_like(row: dict[str, str], hit: HitAnalysis) -> bool:
    top = s(row, "expected_top")
    label = s(row, "expected_label").lower()
    if top not in {"FX", "Textures"} and "texture" not in label and "ambience" not in label:
        return False
    if hit.audio_read and hit.single_dominant_hit:
        return False
    dur = f(row, "duration_sec")
    events = f(row, "primary_event_count_est")
    span = f(row, "onset_span_ratio")
    tail = f(row, "tail_energy_ratio")
    temporal = f(row, "temporal_centroid_ratio")
    flat = f(row, "spectral_flatness_mean")
    entropy = f(row, "spectral_entropy_mean")
    tags = s(row, "physics_tags")
    if dur >= 4.0 and (span >= 0.45 or tail >= 0.38 or temporal >= 0.30):
        return True
    return bool(dur >= 6.0 and (events >= 2 or flat >= 0.3 or entropy >= 0.7 or "events_spread_across_file" in tags))


def one_shot_like_inside_loop(row: dict[str, str], hit: HitAnalysis) -> bool:
    dur = f(row, "duration_sec")
    events = f(row, "primary_event_count_est")
    span = f(row, "onset_span_ratio")
    rate = f(row, "event_rate_hz")
    temporal = f(row, "temporal_centroid_ratio")
    tags = s(row, "physics_tags")
    if hit.audio_read:
        return hit.single_dominant_hit and dur <= 12.0
    return (
        events <= 2
        and span <= 0.25
        and rate <= 1.0
        and (temporal <= 0.30 or "single_event_or_sustain" in tags)
        and dur <= 8.0
    )


def reason_metrics(row: dict[str, str], hit: HitAnalysis | None = None) -> str:
    base = (
        f"dur={f(row, 'duration_sec'):.3f} events={f(row, 'primary_event_count_est'):.1f} "
        f"rate={f(row, 'event_rate_hz'):.3f} span={f(row, 'onset_span_ratio'):.3f} "
        f"regularity={f(row, 'onset_interval_regularity'):.3f} tail={f(row, 'tail_energy_ratio'):.3f} "
        f"temporal={f(row, 'temporal_centroid_ratio'):.3f} pitch={f(row, 'pitch_confidence'):.3f}"
    )
    if hit is not None:
        base += " | " + hit.reason
    return base


def structure_top(row: dict[str, str]) -> str:
    return s(row, "expected_top")


def is_fx_or_texture_row(row: dict[str, str]) -> bool:
    top = structure_top(row)
    label = s(row, "expected_label").lower()
    return top in {"FX", "Textures"} or "texture" in label or "ambience" in label


def hit_ctor(
    audio_read: bool,
    independent_hit_count: int,
    raw_peak_count: int,
    hit_span_ratio: float,
    first_hit_ratio: float,
    dominant_peak_ratio: float,
    strong_peak_ratio: float,
    energy_75_ratio: float,
    energy_90_ratio: float,
    decay_tail_ratio: float,
    front_loaded: bool,
    single_dominant_hit: bool,
    reason: str,
) -> HitAnalysis:
    return HitAnalysis(
        audio_read,
        independent_hit_count,
        raw_peak_count,
        hit_span_ratio,
        first_hit_ratio,
        dominant_peak_ratio,
        strong_peak_ratio,
        energy_75_ratio,
        energy_90_ratio,
        decay_tail_ratio,
        front_loaded,
        single_dominant_hit,
        reason,
    )


def propose_move(row: dict[str, str], rel: Path, hit: HitAnalysis | None = None) -> tuple[str, Path, str] | None:
    expected_structure = s(row, "expected_structure")
    top = structure_top(row)
    fx_or_texture = is_fx_or_texture_row(row)
    if hit is None:
        hit = hit_ctor(False, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, "no_audio_analysis_supplied")

    proposed: tuple[str, Path, str] | None = None

    # One-shot folders: only move when we have repeated independent events or
    # a real long/evolving FX/texture shape. Duration alone is not evidence.
    if expected_structure == "one_shot":
        if hit.audio_read and hit.single_dominant_hit:
            return None

        if top == "Drums":
            if repeated_transient_loop_like(row, hit):
                if hit.audio_read and not hit.drum_mixed:
                    dest = category_structure_dest(row, rel, "_LOOPS")
                    if dest is not None:
                        proposed = (
                            "move_drum_repeated_single_source_to_category_loop_structure",
                            dest,
                            "multiple_independent_drum_hits_but_one_dominant_hit_family_keep_under_current_drum_category_loops",
                        )
                else:
                    proposed = (
                        "move_mixed_drum_repeated_material_to_drum_loops",
                        drum_loop_dest(rel),
                        "multiple_independent_drum_hits_with_mixed_hit_families_go_to_global_drum_loops",
                    )
        elif fx_or_texture:
            if repeated_transient_loop_like(row, hit) or long_fx_or_texture_like(row, hit):
                dest = category_structure_dest(row, rel, "_LONG_FX")
                if dest is not None:
                    proposed = (
                        "move_fx_to_long_fx_structure",
                        dest,
                        "fx_or_texture_repeated_or_long_structure_use_long_fx_never_loops",
                    )
        else:
            if repeated_transient_loop_like(row, hit) or phrase_like_instrument(row, hit):
                dest = category_structure_dest(row, rel, "_LOOPS")
                if dest is not None:
                    proposed = (
                        "move_to_loop_structure",
                        dest,
                        "multiple_independent_hits_or_phrase_structure",
                    )

    # Loop folders: search the other way too. If a loop-folder training item is
    # really just one independent hit with a tail, propose moving it back to one-shots.
    elif expected_structure == "loop" and one_shot_like_inside_loop(row, hit):
        dest = category_structure_dest(row, rel, "_ONE_SHOTS")
        if dest is not None:
            proposed = (
                "move_loop_back_to_one_shot_structure",
                dest,
                "loop_folder_audio_looks_single_independent_hit",
            )

    # FX should not use _LOOPS as an active training structure. Repeated/evolving FX
    # goes to _LONG_FX. This preserves the user's simpler FX structure policy.
    elif expected_structure == "loop" and fx_or_texture:
        dest = category_structure_dest(row, rel, "_LONG_FX")
        if dest is not None:
            proposed = (
                "move_fx_loop_to_long_fx_structure",
                dest,
                "fx_or_texture_loop_folder_renamed_to_long_fx_never_loops",
            )

    if proposed is None:
        return None
    action, proposed_rel, reason = proposed
    if invalid_structure_destination(proposed_rel):
        return None
    return action, proposed_rel, reason


def median(values: Sequence[float]) -> float:
    vals = sorted(x for x in values if math.isfinite(x))
    return statistics.median(vals) if vals else 0.0


def build_outlier_scores(
    rows: list[dict[str, str]], min_label_count: int, threshold: float
) -> dict[int, tuple[float, str, float, str]]:
    groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        label = s(row, "expected_label")
        if label:
            groups[label].append((idx, row))
    global_floors: dict[str, float] = {}
    for col in OUTLIER_FEATURE_COLS:
        vals = [f(r, col) for r in rows]
        med = median(vals)
        mad = median([abs(x - med) for x in vals])
        global_floors[col] = max(0.05, mad * 0.10)
    scores: dict[int, tuple[float, str, float, str]] = {}
    for _label, pairs in groups.items():
        if len(pairs) < min_label_count:
            continue
        meds: dict[str, float] = {}
        denoms: dict[str, float] = {}
        for col in OUTLIER_FEATURE_COLS:
            vals = [f(r, col) for _, r in pairs]
            med = median(vals)
            mad = median([abs(x - med) for x in vals])
            meds[col] = med
            denoms[col] = max(global_floors[col], mad * 1.4826, 1e-6)
        for idx, row in pairs:
            ranked: list[tuple[float, str]] = []
            for col in OUTLIER_FEATURE_COLS:
                ranked.append((abs(f(row, col) - meds[col]) / denoms[col], col))
            ranked.sort(reverse=True)
            worst_score, worst_feature = ranked[0]
            second_score, second_feature = ranked[1] if len(ranked) > 1 else (0.0, "")
            # Two-dimensional confirmation avoids quarantining a valid but unusual
            # sample just because one feature is extreme.
            if worst_score >= threshold and second_score >= max(7.0, threshold * 0.60):
                scores[idx] = (worst_score, worst_feature, second_score, second_feature)
    return scores


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def ensure_note(folder: Path, text: str) -> None:
    note = folder / "_DESTINATION_NOTE.txt"
    if not note.exists():
        note.write_text(text, encoding="utf-8")


def make_symlink(link_path: Path, target_path: Path) -> Path:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    final = link_path
    if final.exists() or final.is_symlink():
        stem, suffix = final.stem, final.suffix
        for i in range(2, 9999):
            alt = final.with_name(f"{stem}__dup{i:03d}{suffix}")
            if not alt.exists() and not alt.is_symlink():
                final = alt
                break
    os.symlink(str(target_path), str(final))
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a physics-only symlink review plan for training cleanup.")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--training-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--max-total-proposals", type=int, default=2500)
    ap.add_argument("--max-proposals-per-label", type=int, default=60)
    ap.add_argument("--outlier-threshold", type=float, default=12.0)
    ap.add_argument("--outlier-min-label-count", type=int, default=20)
    ap.add_argument("--audio-confirm-structure", action="store_true", default=True)
    ap.add_argument("--no-audio-confirm-structure", dest="audio_confirm_structure", action="store_false")
    ap.add_argument("--include-removal-proposals", dest="include_outlier_removals", action="store_true", default=False)
    ap.add_argument("--no-outlier-removals", dest="include_outlier_removals", action="store_false")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    project_root = args.project_root.expanduser().resolve()
    training_root = args.training_root.expanduser().resolve()
    reports_dir = locate_reports_dir(args.run_dir)
    rows = read_rows(reports_dir / "train_feature_manifest.csv")

    run_name = args.run_dir.expanduser().resolve().name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or (project_root / "reports" / "stage4_training_physics_review_plans")
    plan_dir = make_unique_plan_dir(out_root, run_name, stamp)
    moves_root = plan_dir / "PROPOSED_MOVES"
    removals_root = plan_dir / "PROPOSED_REMOVALS_TO_QUARANTINE" / "TO_BE_REMOVED_REVIEW"

    (plan_dir / ".AARON_TRAINING_REVIEW_PLAN.json").write_text(
        json.dumps(
            {
                "tool": "aaron_training_physics_review_plan",
                "version": "0.4.48",
                "created": stamp,
                "project_root": str(project_root),
                "training_root": str(training_root),
                "source_reports_dir": str(reports_dir),
                "audio_confirm_structure": bool(args.audio_confirm_structure),
                "include_outlier_removals": bool(args.include_outlier_removals),
                "meaning": "Delete symlinks you reject. Remaining symlinks become actions when apply is run.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    outlier_scores = (
        build_outlier_scores(rows, args.outlier_min_label_count, args.outlier_threshold)
        if args.include_outlier_removals
        else {}
    )
    hit_cache: dict[str, HitAnalysis] = {}
    move_rows: list[dict[str, str]] = []
    removal_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    protected_rows: list[dict[str, str]] = []
    used_originals = set()
    per_label_count: dict[str, int] = defaultdict(int)
    total = 0

    for _idx, row in enumerate(rows):
        if total >= args.max_total_proposals:
            break
        rel = rel_to_training(s(row, "source_path"), training_root)
        if rel is None:
            continue
        label = s(row, "expected_label")
        if per_label_count[label] >= args.max_proposals_per_label:
            continue
        hit = hit_analysis_for_row(row, training_root, rel, args.audio_confirm_structure, hit_cache)
        # Log cases that old report-only math would have moved but audio confirms as single hit.
        if (
            s(row, "expected_structure") == "one_shot"
            and legacy_report_repeated_transient_loop_like(row)
            and hit.audio_read
            and hit.single_dominant_hit
        ):
            protected_rows.append(
                {
                    "original_training_relpath": str(rel),
                    "expected_top": s(row, "expected_top"),
                    "expected_label": s(row, "expected_label"),
                    "expected_structure": s(row, "expected_structure"),
                    "protection_reason": "single_dominant_hit_protected_from_loop_move",
                    "metrics": reason_metrics(row, hit),
                    "physics_tags": s(row, "physics_tags"),
                    "physics_summary": s(row, "physics_summary"),
                }
            )
        proposal = propose_move(row, rel, hit)
        if not proposal:
            continue
        action, proposed_rel, reason = proposal
        if proposed_rel == rel or str(rel) in used_originals:
            continue
        used_originals.add(str(rel))
        per_label_count[label] += 1
        total += 1
        original = training_root / rel
        review_folder = moves_root / proposed_rel.parent
        link_path = make_symlink(review_folder / safe_link_name(total, rel.name), original)
        ensure_note(
            review_folder,
            (
                "PROPOSED MOVE DESTINATION\n\n"
                f"Training entries represented by symlinks left in this folder will be moved to:\n"
                f"  training/locked_curated_v1/{proposed_rel.parent}\n\n"
                "Delete a symlink if you do NOT want that original training entry moved.\n"
                "The apply tool moves the training entry itself. If it is a symlink, it moves the symlink, not the target audio file.\n"
            ),
        )
        rec = {
            "proposal_index": str(total),
            "action": action,
            "review_symlink": str(link_path.relative_to(plan_dir)),
            "original_training_relpath": str(rel),
            "proposed_training_relpath": str(proposed_rel),
            "expected_top": s(row, "expected_top"),
            "expected_label": s(row, "expected_label"),
            "expected_structure": s(row, "expected_structure"),
            "physics_reason": reason,
            "metrics": reason_metrics(row, hit),
            "physics_tags": s(row, "physics_tags"),
            "physics_summary": s(row, "physics_summary"),
        }
        move_rows.append(rec)
        all_rows.append(rec)

    for idx, row in enumerate(rows):
        if total >= args.max_total_proposals:
            break
        if idx not in outlier_scores:
            continue
        rel = rel_to_training(s(row, "source_path"), training_root)
        if rel is None or str(rel) in used_originals:
            continue
        hit = hit_analysis_for_row(row, training_root, rel, args.audio_confirm_structure, hit_cache)
        # Removal proposals must be audio-confirmed. If the tool cannot read the
        # audio, it has no business proposing removal from locked training.
        if not hit.audio_read:
            continue
        if hit.single_dominant_hit:
            continue
        if propose_move(row, rel, hit):
            continue
        label = s(row, "expected_label")
        if per_label_count[label] >= args.max_proposals_per_label:
            continue
        score, worst, second_score, second = outlier_scores[idx]
        used_originals.add(str(rel))
        per_label_count[label] += 1
        total += 1
        original = training_root / rel
        review_folder = removals_root / rel.parent
        link_path = make_symlink(review_folder / safe_link_name(total, rel.name), original)
        ensure_note(
            review_folder,
            (
                "PROPOSED REMOVE FROM TRAINING\n\n"
                "Symlinks left in this folder will be moved out of locked training into a quarantine folder on apply.\n"
                "Delete a symlink if you want to KEEP that original training entry where it is.\n"
                "The apply tool does not permanently delete audio by default.\n"
            ),
        )
        rec = {
            "proposal_index": str(total),
            "action": "quarantine_training_outlier",
            "review_symlink": str(link_path.relative_to(plan_dir)),
            "original_training_relpath": str(rel),
            "proposed_training_relpath": "",
            "expected_top": s(row, "expected_top"),
            "expected_label": s(row, "expected_label"),
            "expected_structure": s(row, "expected_structure"),
            "physics_reason": f"crazy_category_outlier score={score:.2f} worst_feature={worst} second={second_score:.2f} second_feature={second}",
            "metrics": reason_metrics(row, hit),
            "physics_tags": s(row, "physics_tags"),
            "physics_summary": s(row, "physics_summary"),
        }
        removal_rows.append(rec)
        all_rows.append(rec)

    fields = [
        "proposal_index",
        "action",
        "review_symlink",
        "original_training_relpath",
        "proposed_training_relpath",
        "expected_top",
        "expected_label",
        "expected_structure",
        "physics_reason",
        "metrics",
        "physics_tags",
        "physics_summary",
    ]
    write_csv(plan_dir / "ALL_PROPOSALS.csv", all_rows, fields)
    write_csv(plan_dir / "MOVE_PROPOSALS.csv", move_rows, fields)
    write_csv(plan_dir / "QUARANTINE_OUTLIER_PROPOSALS.csv", removal_rows, fields)
    write_csv(
        plan_dir / "PROTECTED_SINGLE_HIT_NOT_MOVED.csv",
        protected_rows,
        [
            "original_training_relpath",
            "expected_top",
            "expected_label",
            "expected_structure",
            "protection_reason",
            "metrics",
            "physics_tags",
            "physics_summary",
        ],
    )

    original_counts: dict[str, int] = defaultdict(int)
    proposed_remove_from_parent: dict[str, int] = defaultdict(int)
    for row in rows:
        rel = rel_to_training(s(row, "source_path"), training_root)
        if rel is not None:
            original_counts[str(rel.parent)] += 1
    for rec in all_rows:
        parent = str(Path(rec["original_training_relpath"]).parent)
        proposed_remove_from_parent[parent] += 1
    empty_rows = []
    for parent, count in sorted(original_counts.items()):
        remaining = count - proposed_remove_from_parent.get(parent, 0)
        if remaining <= 2 and proposed_remove_from_parent.get(parent, 0) > 0:
            empty_rows.append(
                {
                    "training_folder_relpath": parent,
                    "original_count": str(count),
                    "proposed_removed_or_moved_count": str(proposed_remove_from_parent.get(parent, 0)),
                    "remaining_if_all_accepted": str(remaining),
                    "recommendation": "REVIEW_FOLDER_MAY_EMPTY",
                }
            )
    write_csv(
        plan_dir / "FOLDERS_THAT_MAY_EMPTY_IF_ACCEPTED.csv",
        empty_rows,
        [
            "training_folder_relpath",
            "original_count",
            "proposed_removed_or_moved_count",
            "remaining_if_all_accepted",
            "recommendation",
        ],
    )

    summary = (
        "Aaron Training Physics Review Plan\n"
        "==================================\n\n"
        f"Tool version: 0.4.48\n"
        f"Plan folder: {plan_dir}\n"
        f"Training root: {training_root}\n"
        f"Source reports: {reports_dir}\n"
        f"Audio-confirm structure: {args.audio_confirm_structure}\n"
        f"Removal proposals enabled: {args.include_outlier_removals}\n\n"
        f"Total proposals: {len(all_rows)}\n"
        f"Move proposals: {len(move_rows)}\n"
        f"Quarantine/removal-review proposals: {len(removal_rows)}\n"
        f"Single-hit rows protected from old loop logic: {len(protected_rows)}\n"
        f"Folders that may empty if all accepted: {len(empty_rows)}\n\n"
        "How to review:\n"
        "1. Open PROPOSED_MOVES and, only if enabled, PROPOSED_REMOVALS_TO_QUARANTINE.\n"
        "2. Listen to the symlinks.\n"
        "3. Delete symlinks for proposals you reject.\n"
        "4. Leave symlinks for proposals you accept.\n"
        "5. Run the apply tool later. It only acts on symlinks that remain.\n\n"
        "Safety:\n"
        "- Decisions are physics-only. Filenames are not used to choose moves/removals.\n"
        "- Preview does not move/delete training files.\n"
        "- Apply moves training symlinks/files, not symlink targets.\n"
        "- Removals are off by default and quarantined when explicitly enabled.\n"
        "- Long single-hit cymbals/crashes/impacts are protected from loop moves.\n- Pure repeated drum-category patterns go to category _LOOPS; mixed drum patterns go to Drums/Drum Loops.\n"
    )
    (plan_dir / "README_REVIEW_THIS_FOLDER.txt").write_text(summary, encoding="utf-8")
    (plan_dir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary)
    if args.open:
        os.system(f'open "{plan_dir}" >/dev/null 2>&1 || true')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
