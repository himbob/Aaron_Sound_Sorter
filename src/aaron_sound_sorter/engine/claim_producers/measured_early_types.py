# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Typed state objects for measured early adjudication.

The measured early-adjudication policy is intentionally order-sensitive.  These
small dataclasses keep the extracted guard classes readable without changing the
legacy branch thresholds or candidate ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim

VOICE_FRAGMENTS = (
    "human and voice",
    "voice",
    "vocal",
    "vox",
    "choir",
    "spoken",
    "breath",
    "crowd",
)
NONVOICE_FRAGMENTS = (
    "siren",
    "alarm",
    "beep",
    "blip",
    "chime",
    "bell",
    "whoosh",
    "swoosh",
    "swish",
    "sweep",
    "seagull",
    "seaguls",
    "bird",
    "animal",
    "clank",
    "metallic",
    "hybrid designed",
    "designed noise",
    "boom",
    "impact",
    "slam",
    "hit",
    "sub hit",
    "short impact",
    "trumpet",
    "rhodes",
    "keys",
    "guitar",
    "percussion",
    "conga",
    "tabla",
)
DRUM_LOOP_FRAGMENTS = (
    "drum loop",
    "drum loops",
    "break",
    "breaks",
    "kick",
    "percussion",
    "conga",
    "bongo",
    "tabla",
    "triangle",
    "wood block",
    "wood blocks",
    "metallic percussion",
)
TONAL_FX_FRAGMENTS = (
    "bell",
    "bells",
    "chime",
    "chimes",
    "riser",
    "build",
    "sweep",
    "whoosh",
    "hybrid designed",
    "designed tonal",
    "metallic",
)
DRUM_HIT_FRAGMENTS = (
    "kick",
    "tom",
    "snare",
    "clap",
    "rim",
    "hat",
    "cymbal",
    "percussion",
    "conga",
    "bongo",
    "tabla",
)
DRUM_HIT_EXCLUDES = ("drum loop", "drum loops", "loops")
CONCRETE_FX_FRAGMENTS = (
    "whoosh",
    "swoosh",
    "swish",
    "sweep",
    "seagull",
    "seaguls",
    "bird",
    "animal",
    "clank",
    "metallic",
    "bell",
    "chime",
    "hybrid designed",
    "riser",
    "build",
    "transition",
)
ABSTRACT_TONE_FX_FRAGMENTS = ("beep", "siren", "alarm", "blip")
VOICE_ROLE_NAMES = {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
PROTECTED_HIT_ROLES = {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}


@dataclass(frozen=True)
class EarlyAdjudicationContext:
    """Shared facts used by all measured early-adjudication guards."""

    raw: ConsensusClaim
    eligibility: EligibilityDecision
    measured_role: str
    shape: str
    shape_conf: float
    facts: SharedAudioFacts | None
    raw_path: str
    role: str
    raw_score: float
    direct_voice_strength: float
    full_voice_strength: float
    bass_loop_strength: float


@dataclass(frozen=True)
class VoiceCandidateEvidence:
    """Candidate and measured evidence for Human/Voice conflict checks."""

    best_voice: float | None
    best_voice_path: str
    best_voice_role_strength: float
    best_voice_candidate_role_strength: float
    best_nonvoice: tuple[float, str] | None
    best_drum_loop: tuple[float, str] | None
    best_inst_any: float | None
    best_fx_any: float | None
    measured_voice_strength: float
    direct_voice_strength: float
    best_competitor: float | None
    voice_candidate_is_actual_voice_path: bool
    voice_candidate_has_role_support: bool
    measured_voice_claim_strength: float
    voice_has_positive_audio_evidence: bool
    positive_voice_claim: bool
    direct_one_shot_voice_claim: bool
