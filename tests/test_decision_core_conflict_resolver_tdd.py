"""TDD regressions for confirmed role-vs-candidate conflicts.

Every test in this file was written to fail before the conflict-resolver patch.
The rule for future AI work is red/green: add a failing synthetic regression,
prove it fails on the current code, then make the smallest architecture-safe fix.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def raw(
    path: str, top: str, *, status: str = "strong_consensus", candidates: list[dict] | None = None
) -> ConsensusDecision:
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status=status,
        reason="synthetic raw decision",
        combined_rank_score=13.0,
        shared_candidates=candidates or [],
    )


def eligibility(role: str, top: tuple[str, ...], broad: str, *, confidence: float = 0.80) -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=top,
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def candidate(path: str, score: float, *, top: str | None = None, role: str = "", role_strength: float = 0.0) -> dict:
    folder_top = top or path.split("/", 1)[0]
    signature = {
        "bass_loop": 0.0,
        "bright_drum_loop": 0.0,
        "low_rhythmic_drum_loop": 0.0,
        "percussive_drum_loop": 0.0,
        "percussive_one_shot": 0.0,
        "pitched_music_loop": 0.0,
        "pitched_music_phrase": 0.0,
        "vocal_music_phrase": 0.0,
        "voiced_one_shot": 0.0,
    }
    if role:
        signature[role] = role_strength
    return {
        "folder_path": path,
        "label": path,
        "top_family": folder_top,
        "combined_rank_score": score,
        "candidate_role_signature": signature,
    }


def test_guitar_strum_conflict_reviews_instead_of_forcing_tom() -> None:
    """Guitar/strum/chord one-shot must not be forced into Drums/Toms."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Drums/Toms/Generic Tom/One Shots",
            "Drums",
            candidates=[
                candidate(
                    "Instruments/Guitar/Electric Guitar/One Shots", 10, role="percussive_one_shot", role_strength=0.68
                ),
                candidate("Drums/Toms/Generic Tom/One Shots", 13, role="percussive_one_shot", role_strength=0.98),
                candidate(
                    "Instruments/Guitar/Acoustic Guitar/One Shots", 18, role="percussive_one_shot", role_strength=1.0
                ),
            ],
        ),
        eligibility(
            "protected_percussive_one_shot",
            ("Drums", "_TO_REVIEW"),
            "Drums/Percussion/Generic Percussion/One Shots",
        ),
    )
    assert final.final_top == "_TO_REVIEW"
    assert "pitched instrument one-shot conflict" in final.reason


def test_true_tom_with_drum_candidate_margin_stays_drums() -> None:
    """The conflict resolver must not damage obvious drum one-shots."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Drums/Toms/Generic Tom/One Shots",
            "Drums",
            candidates=[
                candidate("Drums/Toms/Generic Tom/One Shots", 5, role="percussive_one_shot", role_strength=0.99),
                candidate(
                    "Drums/Percussion/Generic Percussion/One Shots", 7, role="percussive_one_shot", role_strength=0.95
                ),
                candidate(
                    "Instruments/Guitar/Acoustic Guitar/One Shots", 21, role="percussive_one_shot", role_strength=0.72
                ),
            ],
        ),
        eligibility(
            "protected_percussive_one_shot",
            ("Drums", "_TO_REVIEW"),
            "Drums/Percussion/Generic Percussion/One Shots",
        ),
    )
    assert final.folder_path == "Drums/Toms/Generic Tom/One Shots"


def test_vocal_loop_conflict_reviews_instead_of_forcing_drum_loops() -> None:
    """Vocal loop material must not be forced into Drum Loops by repeated events."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Drums/Drum Loops/Loops",
            "Drums",
            candidates=[
                candidate("FX/Human and Voice FX/Crowd/Long FX", 15, role="vocal_music_phrase", role_strength=0.82),
                candidate(
                    "FX/Human and Voice FX/Spoken Voice/Long FX", 18, role="pitched_music_phrase", role_strength=0.86
                ),
                candidate("Drums/Drum Loops/Loops", 24, role="percussive_drum_loop", role_strength=0.71),
            ],
        ),
        eligibility(
            "drum_loop",
            ("Drums", "_TO_REVIEW"),
            "Drums/Drum Loops/Loops",
        ),
    )
    assert final.final_top == "_TO_REVIEW"
    assert "vocal loop conflict" in final.reason


def test_fx_motion_candidate_conflict_reviews_instead_of_percussion_one_shot() -> None:
    """Whoosh/swoosh motion FX should not be forced into generic percussion."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Drums/Percussion/Generic Percussion/One Shots",
            "Drums",
            candidates=[
                candidate("Drums/Rims and Sticks/Rimshot/One Shots", 9, role="percussive_one_shot", role_strength=0.98),
                candidate(
                    "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/One Shots",
                    12,
                    role="percussive_one_shot",
                    role_strength=0.76,
                ),
                candidate("FX/Designed Noise FX/Blip/One Shots", 16, role="percussive_one_shot", role_strength=0.56),
            ],
        ),
        eligibility(
            "percussive_one_shot",
            ("Drums", "FX", "_TO_REVIEW"),
            "Drums/Percussion/One Shots",
        ),
    )
    assert final.final_top == "_TO_REVIEW"
    assert "fx motion conflict" in final.reason


def test_non_vocal_chime_conflict_reviews_instead_of_human_voice_fx() -> None:
    """Wind-chime/beep-like candidates should not be broadened to Human/Voice FX."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            candidates=[
                candidate("FX/Designed Noise FX/Siren/One Shots", 7, role="voiced_one_shot", role_strength=0.62),
                candidate("FX/Designed Noise FX/Beep/One Shots", 12, role="voiced_one_shot", role_strength=0.89),
                candidate("Instruments/Brass/Trumpet/One Shots", 16, role="voiced_one_shot", role_strength=0.95),
                candidate(
                    "FX/Human and Voice FX/Spoken Voice/Long FX", 40, role="vocal_music_phrase", role_strength=0.10
                ),
            ],
        ),
        eligibility(
            "vocal_phrase",
            ("FX", "Instruments", "_TO_REVIEW"),
            "FX/Human and Voice FX",
        ),
    )
    assert final.final_top == "_TO_REVIEW"
    assert "voice false-positive conflict" in final.reason
