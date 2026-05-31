#!/usr/bin/env python3
"""
Aaron Sound Sorter Stage 4 v0.5.7 COMMITTEE_PHYSICS_GAP_LOCKS

Clean Phase 3 replacement path.

Latest user-facing version label: v0.5.7 COMMITTEE_PHYSICS_GAP_LOCKS.

This is the single-file Stage 4 implementation. Compatibility stubs may call this file, but runtime logic lives here.

It uses the old project only as design reference:
  - build a feature fingerprint
  - train a scratch multi-prototype reference brain
  - evaluate holdout files
  - write diagnostics

It does not:
  - use old judge/router logic
  - invent labels outside the trained brain
  - use producer filenames as classification evidence
  - collapse trusted folder labels into generic buckets

Default:
  - Drums, Instruments, FX, and Textures
  - source rows from a trusted folder tree or approved sorted output
  - preserve the real folder paths exactly as learned labels
  - train from all available readable rows by default
  - choose the internal model per learned folder from measured spread
  - evaluate without passing the expected structure into prediction
  - one run folder
  - local reports only by default; product runners create separate logs-only upload ZIPs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Keep compatibility symbols from the original single-file namespace available
# to the split legacy modules that still intentionally use ``from .core import *``.
# These imports are not classifier evidence and do not affect routing behavior.
_CORE_NAMESPACE_COMPATIBILITY_EXPORTS = (
    argparse,
    csv,
    hashlib,
    json,
    math,
    re,
    shutil,
    signal,
    subprocess,
    sys,
    tempfile,
    time,
    warnings,
    zipfile,
    Counter,
    defaultdict,
    field,
    datetime,
)

# Keep numerical behavior stable and avoid oversubscription hangs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np  # noqa: E402

try:
    import soundfile as _sf  # type: ignore
except Exception:
    _sf = None

DEFAULT_ROOT = "/Volumes/T9/music_production/samples/Sorted samples"
DEFAULT_PROJECT_DIR = str(Path(__file__).resolve().parents[2])

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}
TARGET_SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 40
N_MFCC = 13
MAX_ANALYSIS_SECONDS = 12.0
FINGERPRINT_TIMEOUT_SECONDS = 30.0
BASE_FP_SIZE = 50

FEATURE_NAMES = [
    *[f"mfcc_mu_{i + 1}" for i in range(N_MFCC)],
    *[f"mfcc_std_{i + 1}" for i in range(N_MFCC)],
    "log_rolloff85_hz",
    "log_crest",
    "spectral_flux_mean",
    "spectral_flatness_mean",
    "spectral_entropy_mean",
    "log_decay_ratio",
    "stereo_width",
    "mid_side_ratio",
    "centroid_slope_norm",
    "log_transient_count",
    "sub_bass_ratio_lt_150hz",
    "bass_ratio_150_500hz",
    "mid_ratio_500_2000hz",
    "presence_ratio_2000_8000hz",
    "air_ratio_gt_8000hz",
    "zcr_mean",
    "temporal_centroid_ratio",
    "onset_interval_regularity",
    "attack_rise_time_norm",
    "onset_span_ratio",
    "event_rate_hz",
    "tail_energy_ratio",
    "spectral_flux_variance",
    "pitch_confidence",
]

# v0.5.5 committee physics agreement.  Clean-brain version: the sorter requires
# newly rebuilt v0.5 brains using this full 100-feature layout.  The first
# 50 slots remain at stable indexes for internal helper readability, but old
# 50-feature brains are intentionally not supported.  These are measured
# descriptors only: no filenames, no category-name rescues.
EXTRA_PHYSICS_FEATURE_NAMES = [
    # harmonic / pitch identity
    "f0_median_hz",
    "f0_voiced_ratio",
    "f0_stability_cents",
    "f0_slope_cents_per_sec",
    "harmonic_to_noise_ratio",
    "harmonic_peak_count",
    "harmonic_energy_ratio",
    "inharmonicity",
    "fundamental_dominance_ratio",
    "overtone_slope",
    # attack/body/tail split
    "attack_pitch_confidence",
    "body_pitch_confidence",
    "tail_pitch_confidence",
    "attack_flatness",
    "body_flatness",
    "tail_flatness",
    "attack_entropy",
    "body_entropy",
    "tail_entropy",
    "attack_zcr",
    "body_zcr",
    "tail_zcr",
    "attack_high_ratio",
    "body_high_ratio",
    "tail_high_ratio",
    "attack_low_ratio",
    "body_low_ratio",
    "tail_low_ratio",
    # noise burst / decay shape
    "noise_burst_duration_ms",
    "high_band_decay_slope",
    "spectral_centroid_decay_slope",
    "noise_tail_decay_slope",
    "attack_noise_ratio",
    "body_noise_ratio",
    "tail_noise_ratio",
    # low-end source identity
    "low_peak_frequency_hz",
    "low_peak_bandwidth_hz",
    "sub_attack_time_ms",
    "sub_decay_time_ms",
    "sub_to_click_offset_ms",
    "kick_pitch_drop_cents",
    "sub_sustain_ratio",
    # spectral peak / formant-like shape
    "spectral_peak_count",
    "top1_peak_frequency_hz",
    "top2_peak_frequency_hz",
    "top3_peak_frequency_hz",
    "peak_bandwidth_mean_hz",
    "spectral_peak_stability",
    "formant_like_peak_spacing",
    "spectral_envelope_slope",
]

LOOP_LONG_SEGMENT_FEATURE_NAMES = [
    # v20260512 loop/long population descriptors. These are measured summaries
    # of event slices and non-event regions, not routing rules. They let the
    # brain learn whether long/loop material is mostly percussive, mostly
    # tonal, or mixed without using filenames or folder names.
    "loop_pitched_event_ratio",
    "loop_percussive_event_ratio",
    "loop_noisy_event_ratio",
    "loop_event_timbre_diversity",
    "loop_sustained_tonal_frame_ratio",
    "loop_drumlike_frame_ratio",
    "loop_tonal_to_percussive_balance",
    "loop_mean_event_pitch_confidence",
    "loop_mean_event_noise_ratio",
    "loop_mean_event_low_ratio",
    "loop_mean_event_high_ratio",
    "loop_non_event_tonal_ratio",
]

FEATURE_NAMES.extend(EXTRA_PHYSICS_FEATURE_NAMES)
FEATURE_NAMES.extend(LOOP_LONG_SEGMENT_FEATURE_NAMES)
FP_SIZE = len(FEATURE_NAMES)

# First Phase 3 brain is intentionally core-only.
DEFAULT_ALLOWED_TOP = {"Drums", "Instruments", "FX", "Textures"}
DEFAULT_FOLDER_BRAIN = "stage4_folder_brain.json"


# v0.4.34 gold-candidate curation and physics-audit policy.
# The category universe remains available, but a label only becomes an active
# teacher when it has enough clean, source-diverse, structure-correct examples.
# Questionable rows are copied/reported for review; they do not train the brain.
MIN_ACTIVE_TRAIN_PER_LABEL = 1
MIN_AUTO_PLACE_TRAIN_PER_LABEL = 3
MIN_PROVISIONAL_TRAIN_PER_LABEL = 1
DEFAULT_SOURCE_CAP_PER_LABEL = 100
DEFAULT_MIN_SOURCE_GROUPS_PER_LABEL = 1
DEFAULT_GOLD_WORKSPACE_KEEP_LIMIT = 999999
DEFAULT_REAL_PREVIEW_OFFICIAL_COPY_PER_LABEL = 3
DEFAULT_REAL_PREVIEW_RAW_COPY_PER_LABEL = 3
DEFAULT_REAL_PREVIEW_BROAD_COPY_PER_LABEL = 3
DEFAULT_REAL_PREVIEW_RISKY_COPY_PER_LABEL = 8
DEFAULT_REAL_PREVIEW_RISKY_MIN_SIMILARITY = 0.35
DEFAULT_REAL_PREVIEW_BROAD_FAMILY_MIN_GAP = 0.0
DEFAULT_REAL_PREVIEW_FORCE_GUESS_COPY_PER_LABEL = 0
DEFAULT_BROAD_FAMILY_CONFLICT_OVERRIDE_GAP = 1.0

# No public training labels are disabled in code.
# Aaron edits/removes bad folders manually from the locked training tree.
DISABLED_TRAINING_PUBLIC_LABEL_PATTERNS: list[str] = []

# Pure loop labels must remain pure. Beats or multi-source loops go to mixed
# loop labels, or quarantine if the evidence is not safe.
PURE_DRUM_LOOP_WORDS = {
    "kick",
    "snare",
    "clap",
    "snap",
    "hat",
    "hi hat",
    "hihat",
    "tom",
    "cymbal",
    "crash",
    "ride",
    "bongo",
    "conga",
    "shaker",
    "rim",
    "stick",
    "sidestick",
    "side stick",
    "cowbell",
}
PURE_INSTRUMENT_LOOP_WORDS = {
    "guitar",
    "bass",
    "808",
    "keys",
    "piano",
    "organ",
    "synth",
    "vocal",
    "voice",
    "strings",
    "string",
    "brass",
    "woodwind",
    "sax",
    "flute",
    "horn",
}


# Pure-brain reliability gate. Labels with tiny training pools are allowed to
# exist, but they need stronger learned evidence before auto-placement.
TINY_LABEL_COUNT = 5
SMALL_LABEL_COUNT = 12
MEDIUM_LABEL_COUNT = 24
# v0.4.34 safe test mode: prefer review over weak auto-placement. Trash/weak/
# dirty labels are still blocked before brain build.
TINY_LABEL_MIN_SIMILARITY = 0.40
TINY_LABEL_MIN_MARGIN = 1.30
SMALL_LABEL_MIN_SIMILARITY = 0.46
SMALL_LABEL_MIN_MARGIN = 0.95
MEDIUM_LABEL_MIN_SIMILARITY = 0.54
MEDIUM_LABEL_MIN_MARGIN = 0.95
PROVISIONAL_TINY_MIN_SIMILARITY = 0.66
PROVISIONAL_TINY_MIN_MARGIN = 2.00
PROVISIONAL_SINGLE_SOURCE_MIN_SIMILARITY = 0.58
PROVISIONAL_SINGLE_SOURCE_MIN_MARGIN = 1.75

# Support-aware folder balancing. This is a general learned-folder correction,
# not a category or filename rule. It gives underrepresented training folders a
# small capped ranking boost so huge folders do not swallow close edge cases.
# Existing low-support auto-place gates still apply after prediction.
SUPPORT_BALANCE_ENABLED_DEFAULT = True
SUPPORT_BALANCE_THRESHOLD_RATIO = 0.55
SUPPORT_BALANCE_MAX_DISTANCE_BONUS = 0.28
SUPPORT_BALANCE_TINY_COUNT_FULL_BONUS = 3
SUPPORT_BALANCE_TINY_MULTIPLIER = 0.50

# Model tournament / ensemble chooser. These heads are all learned from the
# same folder tree and audio features. They do not inspect filenames or use
# category-specific rescue logic. The goal is not to punt to review, but to
# make a better forced pick when the nearest-prototype winner is only a weak
# raw distance win against a related rival.
MODEL_ENSEMBLE_ENABLED_DEFAULT = True
MODEL_ENSEMBLE_MAX_LABELS = 16
MODEL_ENSEMBLE_MIN_SWITCH_ADVANTAGE = 0.70
MODEL_ENSEMBLE_MAX_SCORE_PENALTY_RATIO = 1.20
MODEL_ENSEMBLE_CURRENT_RANK_WEIGHT = 0.40
MODEL_ENSEMBLE_NEUTRAL_RANK_WEIGHT = 0.18
MODEL_ENSEMBLE_SPREAD_RANK_WEIGHT = 0.18
MODEL_ENSEMBLE_ANCHOR_RANK_WEIGHT = 0.08
MODEL_ENSEMBLE_LINEAR_RANK_WEIGHT = 0.16
MODEL_ENSEMBLE_FRONTEND_ROUTER_RANK_WEIGHT = 0.18
# v0.6.2: the model tournament is a same-family re-ranker by default.
# Cross-family rescue belongs to explicit review/safety policy, not to a rank
# ensemble that can be fooled by one generic attack/brightness/anchor head.
MODEL_ENSEMBLE_ALLOW_CROSS_FAMILY_SWITCH_DEFAULT = False
LINEAR_RIDGE_HEAD_ENABLED_DEFAULT = True
LINEAR_RIDGE_L2 = 1.0
LINEAR_RIDGE_FEATURE_CLIP = 12.0
LINEAR_RIDGE_MAX_SAMPLE_WEIGHT = 8.0
LINEAR_RIDGE_MAX_ABS_WEIGHT = 10000.0

# v0.4.81 numerically safe front-end router brain.  This is a second learned model head,
# trained differently from the prototype/exemplar brain.  It learns broad
# routing targets from the trusted folder tree: top family, structure lane,
# and learned parent neighborhood.  Prediction uses it as a soft tie-breaker
# and broad guard, not as a category hard-code and not as a filename rescue.
FRONTEND_ROUTER_BRAIN_ENABLED_DEFAULT = True
FRONTEND_ROUTER_MAX_LABELS = 16
FRONTEND_ROUTER_TOP_PENALTY = 0.18
FRONTEND_ROUTER_STRUCTURE_PENALTY = 0.20
FRONTEND_ROUTER_PARENT_PENALTY = 0.24
FRONTEND_ROUTER_RANK_WEIGHT = 0.18
ADAPTIVE_DEPTH_PLACEMENT_ENABLED_DEFAULT = True
ADAPTIVE_DEPTH_MAX_SIMILARITY = 0.70
ADAPTIVE_DEPTH_MAX_MARGIN = 1.25

# v0.5.5 Committee Physics Agreement Membership Gate + Dynamic Structure Gate + Dynamic Rival Second Pass.
# Facts are learned from the folder tree, not hard-coded per instrument.
# Fact scoring shifts the primary ranking (current_score) so the first guess
# already uses measured audio evidence. Contrast scoring further separates
# close rivals using only the features that best split them.
FACT_SCORING_ENABLED_DEFAULT = True
FACT_SCORE_WEIGHT = 0.22
RIVAL_CONTRAST_ENABLED_DEFAULT = True
RIVAL_CONTRAST_WEIGHT = 0.18
RIVAL_CONTRAST_MAX_RIVALS = 6
RIVAL_CONTRAST_MIN_EFFECT_SIZE = 0.28
RIVAL_CONTRAST_MIN_RELIABILITY = 0.45
FACT_MIN_ELIGIBLE_COUNT = 4
FACT_STRONG_MIN_COUNT = 50
FACT_GOOD_MIN_COUNT = 20
FACT_OK_MIN_COUNT = 10
FACT_TENTATIVE_MIN_COUNT = 4

# v0.5.0 learned-folder membership gate.  This converts the category fact
# profiles from advisory score nudges into a product safety gate.  A nearest
# folder may not auto-place unless the sample fits that folder's reliable
# measured physics.  These thresholds are label-name agnostic and use only
# robust learned fact profiles from the training folder tree.
MEMBERSHIP_GATE_ENABLED_DEFAULT = True
MEMBERSHIP_MIN_PROFILE_COUNT = 3
MEMBERSHIP_MIN_RELIABILITY = 0.55
MEMBERSHIP_SEVERE_IQR_MULTIPLIER = 1.50
MEMBERSHIP_SEVERE_MAD_MULTIPLIER = 4.00
MEMBERSHIP_SEVERE_RANGE_FRACTION = 0.35
MEMBERSHIP_BLOCK_SEVERE_COUNT = 3
MEMBERSHIP_BLOCK_WEIGHTED_SEVERITY = 2.20
MEMBERSHIP_WARN_SEVERE_COUNT = 2
MEMBERSHIP_ALT_MAX_SCORE_GAP = 1.40
MEMBERSHIP_ALT_MIN_SEVERITY_IMPROVEMENT = 1.00
MEMBERSHIP_TOP_CONFLICT_FORCE_REVIEW = True

# v0.5.5 committee physics agreement.  The brain proposes candidates;
# physics helpers vote on whether each candidate actually fits the learned
# folder atlas.  The deepest physically approved candidate can beat the raw
# nearest centroid.  These weights are generic helper weights, not folder-name
# or category-name rules.
COMMITTEE_AGREEMENT_ENABLED_DEFAULT = True
COMMITTEE_MAX_CANDIDATES = 8
COMMITTEE_SWITCH_MIN_ADVANTAGE = 0.18
COMMITTEE_REVIEW_MIN_BEST_SCORE = 0.12
COMMITTEE_RAW_MAX_SCORE_GAP = 2.25
COMMITTEE_WEIGHT_PROTOTYPE = 0.16
COMMITTEE_WEIGHT_MEMBERSHIP = 0.30
COMMITTEE_WEIGHT_FACT_PROFILE = 0.16
COMMITTEE_WEIGHT_ROUTER = 0.12
COMMITTEE_WEIGHT_RIVAL_CONTRAST = 0.10
COMMITTEE_WEIGHT_SUPPORT = 0.0
COMMITTEE_WEIGHT_DEPTH = 0.03
COMMITTEE_WEIGHT_STRUCTURE = 0.08

# v0.4.82 cleanup: honest math-pool recall, no old lab upload ZIP by default, full-brain/small-sort workflow. v0.4.78 model tournament ensemble micro-tested. v0.4.77 decisive balanced sorter micro-tested. v0.4.76 broad imbalanced rival guard micro-tested. v0.4.75 sibling-imbalance guard micro-tested. v0.4.74 feature-complete micro-tested. v0.4.73 dynamic borderline for small labels. v0.4.72 calibrated small-folder generalization. v0.4.71 small-folder recall and consensus contract. v0.4.70 small-folder direct anchor contract. v0.4.69 small-folder recall contract. v0.4.68 contradiction-filtered balanced effective training.  The trusted folder tree remains truth,
# but the competitive brain should not let giant folders get hundreds of
# extra nearest-neighbor lottery tickets.  Raw rows are still counted and
# reported; the math brain uses this deterministic, capped effective pool.
BALANCED_EFFECTIVE_TRAINING_ENABLED_DEFAULT = True
EFFECTIVE_TRAIN_TARGET_PER_LABEL = 64
EFFECTIVE_TRAIN_MAX_PER_LABEL = 96
EFFECTIVE_TRAIN_HUGE_LABEL_THRESHOLD = 300
EFFECTIVE_TRAIN_HUGE_MAX_PER_LABEL = 128
MAX_ANCHORS_PER_LABEL = 32
MAX_EXEMPLARS_PER_LABEL_CAP = 16
CONTRADICTION_FILTER_EFFECTIVE_TRAINING_ENABLED_DEFAULT = True
EFFECTIVE_TRAIN_CENTRAL_FRACTION = 0.65
EFFECTIVE_TRAIN_DIVERSE_FRACTION = 0.35
EFFECTIVE_TRAIN_EDGE_FRACTION = 0.0


# Feature weights by named layout. These are deliberately simple, not category-specific magic.
# They make transient/envelope/spectral structure matter more than raw MFCC color.
FEATURE_WEIGHTS = np.array(
    [0.80] * 13
    + [0.70] * 13
    + [
        1.05,  # rolloff
        1.20,  # crest
        1.15,  # flux
        1.10,  # flatness
        1.10,  # entropy
        1.20,  # decay
        0.80,  # width
        0.70,  # mid-side
        1.05,  # centroid slope
        1.20,  # transient count
        1.65,  # sub-bass ratio, explicit kick/sub evidence
        1.45,  # bass ratio, explicit bass/body evidence
        1.20,  # mid ratio
        1.45,  # presence ratio, snap/attack/instrument bite
        1.65,  # air ratio, hats/cymbals/noise/flutters
        1.35,  # ZCR mean, noise-vs-tone crossing behavior
        1.45,  # temporal centroid, front/back/even energy placement
        1.40,  # onset interval regularity, loop pulse vs clustered FX bursts
        1.35,  # attack rise time, fast hits vs slow swells/risers
        1.55,  # onset span ratio, distinguishes clustered flams from real loops
        1.15,  # event rate hz, dense pulse versus isolated hit
        1.20,  # tail energy ratio, long sustain/tail versus dry hit
        1.25,  # spectral flux variance, repeated hit contrast vs smooth sustain
        1.55,  # pitch confidence, tonal/bass/instrument evidence vs noisy hits
    ],
    dtype=np.float32,
)
# Append v0.5.1 weights for the expanded source-identity physics.
# Harmonic/pitch and attack/body/tail features should influence identity more
# than structure. They remain generic measured descriptors, not category rules.
EXTRA_PHYSICS_FEATURE_WEIGHTS = np.asarray(
    [
        1.75,
        1.85,
        1.65,
        1.25,
        1.85,
        1.35,
        1.75,
        1.45,
        1.55,
        1.25,
        1.55,
        1.75,
        1.75,
        1.25,
        1.35,
        1.35,
        1.25,
        1.35,
        1.35,
        1.20,
        1.25,
        1.25,
        1.35,
        1.35,
        1.35,
        1.25,
        1.25,
        1.25,
        1.45,
        1.30,
        1.25,
        1.25,
        1.35,
        1.40,
        1.40,
        1.50,
        1.40,
        1.25,
        1.35,
        1.25,
        1.35,
        1.35,
        1.35,
        1.30,
        1.25,
        1.25,
        1.25,
        1.30,
        1.30,
        1.25,
    ],
    dtype=np.float32,
)

# These loop/long population descriptors are meant to matter enough to teach
# voters about mixed musical content, but they are not category-specific magic.
LOOP_LONG_SEGMENT_FEATURE_WEIGHTS = np.asarray(
    [
        1.55,  # loop_pitched_event_ratio
        1.55,  # loop_percussive_event_ratio
        1.15,  # loop_noisy_event_ratio
        1.30,  # loop_event_timbre_diversity
        1.70,  # loop_sustained_tonal_frame_ratio
        1.55,  # loop_drumlike_frame_ratio
        1.70,  # loop_tonal_to_percussive_balance
        1.45,  # loop_mean_event_pitch_confidence
        1.20,  # loop_mean_event_noise_ratio
        1.15,  # loop_mean_event_low_ratio
        1.15,  # loop_mean_event_high_ratio
        1.60,  # loop_non_event_tonal_ratio
    ],
    dtype=np.float32,
)

if len(FEATURE_WEIGHTS) == BASE_FP_SIZE and FP_SIZE > BASE_FP_SIZE:
    FEATURE_WEIGHTS = np.concatenate(
        [
            FEATURE_WEIGHTS,
            EXTRA_PHYSICS_FEATURE_WEIGHTS,
            LOOP_LONG_SEGMENT_FEATURE_WEIGHTS,
        ]
    )
assert len(FEATURE_NAMES) == FP_SIZE
assert len(FEATURE_WEIGHTS) == FP_SIZE

# Structure competition should listen mostly to timing/envelope behavior, not raw tone color.
# This is still learned brain math. It does not inspect filenames or force duration-only loops.
STRUCTURE_FEATURE_WEIGHTS = np.array(
    [0.35] * 13
    + [0.30] * 13
    + [
        0.55,  # rolloff
        1.70,  # crest
        1.90,  # flux
        0.80,  # flatness
        0.75,  # entropy
        1.75,  # decay
        0.35,  # width
        0.30,  # mid-side
        1.15,  # centroid slope
        2.30,  # transient count
        0.40,  # sub-bass ratio
        0.40,  # bass ratio
        0.35,  # mid ratio
        0.45,  # presence ratio
        0.45,  # air ratio
        1.15,  # ZCR mean
        2.05,  # temporal centroid
        2.45,  # onset interval regularity
        2.10,  # attack rise time
        3.00,  # onset span ratio; loop events must be spread, not clustered
        2.20,  # event rate hz; repeated pulse density
        1.90,  # tail energy ratio; tail/sustain not enough for loop proof
        1.25,  # spectral flux variance; loops pulse, pads remain smooth
        0.40,  # pitch confidence; weak structure cue, strong label cue
    ],
    dtype=np.float32,
)
EXTRA_PHYSICS_STRUCTURE_FEATURE_WEIGHTS = np.asarray(
    [
        0.45,
        0.55,
        0.55,
        0.45,
        0.45,
        0.35,
        0.45,
        0.35,
        0.35,
        0.35,
        0.75,
        0.65,
        0.65,
        0.55,
        0.55,
        0.55,
        0.55,
        0.55,
        0.55,
        0.70,
        0.70,
        0.70,
        0.65,
        0.60,
        0.60,
        0.50,
        0.50,
        0.50,
        1.25,
        0.90,
        0.90,
        0.80,
        0.85,
        0.85,
        0.85,
        0.50,
        0.50,
        0.80,
        0.90,
        0.70,
        0.70,
        0.75,
        0.55,
        0.45,
        0.45,
        0.45,
        0.45,
        0.50,
        0.50,
        0.50,
    ],
    dtype=np.float32,
)

LOOP_LONG_SEGMENT_STRUCTURE_FEATURE_WEIGHTS = np.asarray(
    [
        1.40,  # loop_pitched_event_ratio
        1.55,  # loop_percussive_event_ratio
        0.95,  # loop_noisy_event_ratio
        1.20,  # loop_event_timbre_diversity
        1.60,  # loop_sustained_tonal_frame_ratio
        1.60,  # loop_drumlike_frame_ratio
        1.70,  # loop_tonal_to_percussive_balance
        1.20,  # loop_mean_event_pitch_confidence
        0.95,  # loop_mean_event_noise_ratio
        0.90,  # loop_mean_event_low_ratio
        0.90,  # loop_mean_event_high_ratio
        1.50,  # loop_non_event_tonal_ratio
    ],
    dtype=np.float32,
)

if len(STRUCTURE_FEATURE_WEIGHTS) == BASE_FP_SIZE and FP_SIZE > BASE_FP_SIZE:
    STRUCTURE_FEATURE_WEIGHTS = np.concatenate(
        [
            STRUCTURE_FEATURE_WEIGHTS,
            EXTRA_PHYSICS_STRUCTURE_FEATURE_WEIGHTS,
            LOOP_LONG_SEGMENT_STRUCTURE_FEATURE_WEIGHTS,
        ]
    )
assert len(STRUCTURE_FEATURE_WEIGHTS) == FP_SIZE


@dataclass
class FeatureRow:
    path: str
    group_key: str
    label: str
    top: str
    structure: str
    duration_sec: float
    fingerprint: list[float]
    read_status: str = "ok"
    eval_reused_training_source: bool = False
    eval_selection_method: str = ""
    sampling_pool_size: int = 0
    source_pack: str = ""
    structure_remap_reason: str = ""
    structure_conflict_warning: str = ""
    training_reject_reason: str = ""
    training_active_status: str = ""
    label_source_group_count: int = 0
    label_clean_available: int = 0
    balance_group: str = ""


STRUCTURE_MARKERS = {
    "_one_shots",
    "_one_shot",
    "one_shots",
    "one shot",
    "one shots",
    "_loops",
    "_loop",
    "loops",
    "loop",
    "_long_running",
    "long_running",
    "long running",
    "_long_fx",
    "long_fx",
    "long fx",
    "long fxs",
}

# Training is folder-truth driven.  The brain learns the public label from
# whatever folders Aaron gives it.  The only fixed grammar is:
#   - top-level family folders, such as Drums, Instruments, Textures, and FX
#   - optional structure marker leaves, such as _ONE_SHOTS, _LOOPS, and _LONG_FX
# The code must not invent missing category names like "Mixed Drum Loop" just
# because a folder is shallow.  If Aaron gives Drums/Drum Loops/*.wav, the
# learned public label is Drums/Drum Loops and the structure lane is Loops.

# Training is now locked and folder-truth driven.
# No hard-coded filename fragments are allowed to remove or reroute teachers.
# Bad examples should be removed by Aaron from the training folder, not hidden in code.


_MEL_CACHE: dict[int, np.ndarray] = {}


PHYSICS_REPORT_FIELDS = [
    "physics_tags",
    "physics_summary",
    "primary_event_count_est",
    "crest_est",
    "decay_ratio_est",
    "sub_bass_ratio_lt_150hz",
    "bass_ratio_150_500hz",
    "mid_ratio_500_2000hz",
    "presence_ratio_2000_8000hz",
    "air_ratio_gt_8000hz",
    "low_total_ratio_lt_500hz",
    "high_total_ratio_gt_2000hz",
    "zcr_mean",
    "spectral_flatness_mean",
    "spectral_entropy_mean",
    "centroid_slope_norm",
    "stereo_width",
    "mid_side_ratio",
    "temporal_centroid_ratio",
    "onset_interval_regularity",
    "attack_rise_time_norm",
    "onset_span_ratio",
    "event_rate_hz",
    "tail_energy_ratio",
    "spectral_flux_variance",
    "pitch_confidence",
]


class FingerprintTimeout(Exception):
    pass


# v0.5.0 dynamic rival-contrast second pass.  This is deliberately not a
# filename or category-name rescue.  It can only compare labels already learned
# from the folder tree, and it only uses measured feature facts derived from the
# training rows.
RIVAL_SECOND_PASS_ENABLED_DEFAULT = True
RIVAL_SECOND_PASS_MAX_CANDIDATES = 8
RIVAL_SECOND_PASS_MAX_SCORE_GAP = 0.55
RIVAL_SECOND_PASS_MIN_CHALLENGER_SCORE = 0.34
RIVAL_SECOND_PASS_MIN_SCORE_EDGE = 0.20
RIVAL_SECOND_PASS_MAX_FACT_SCORE_DEFICIT = 0.18


APPLE_JUNK_NAMES = {".ds_store", "thumbs.db"}
TINY_DURATION_SECONDS_DEFAULT = 0.05


REAL_PREVIEW_EXCLUDE_PARTS = {
    "sorted samples",
    "_manifests",
    "_training_data",
    "training data",
    "golden_training_set",
    "phase3_pure_brain_lab_run",
    "training_preview",
    "training_curated_keep_preview",
    "training_curated_reject_preview",
    "sorted_preview",
    "real_sorted_preview",
    "real_raw_brain_prediction_preview",
    "real_broad_family_review_preview",
    "listen_review_pack",
    "__macosx",
}

REAL_PREVIEW_EXCLUDE_FILE_FRAGMENTS = {
    "training1",
    "training2",
    "training3",
    "training4",
    "combined_training",
    "one_shot_percussive_sounds",
    "sound_info_analysis",
    "fx_aaron2",
    "synthetic_v",
    "synthetic_",
    "phase3_pure_brain",
    "upload_back",
}
