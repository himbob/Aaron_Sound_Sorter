# SOURCE-NAME BLINDNESS INVARIANT:
# Profile candidate producers may inspect voter output and measured audio facts.
# They must never inspect source filenames or source folders.
"""Drum-family profile candidate claims."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_helpers import (
    _feature_number_from_facts,
    _has_drum_loop_structure_support,
    _shape_metric_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_category_guess, claim_from_folder_path


class ProfileDrumClaimMixin:
    """Produce kick and drum-loop claims from real BrainVoter candidates."""

    def kick_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_score: float,
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim | None:
        """Return a kick claim from real BrainVoter kick evidence."""
        role_supports_kick = role_name in {
            "percussive_one_shot",
            "protected_percussive_one_shot",
            "low_kick_like_hit",
        }
        shape_supports_kick = shape_name in {"hit_with_tail", "single_hit"} and shape_confidence >= 0.70
        if not (role_supports_kick and shape_supports_kick):
            return None
        if role_name != "low_kick_like_hit" and not self._facts_support_low_kick_hit(facts):
            return None
        best_guess = self.best_brain_guess(
            brain_result,
            fragments=("kick", "kick drums", "sub kick", "short kick"),
            max_rank=3,
        )
        if best_guess is None:
            return None
        return claim_from_category_guess(
            guess=best_guess,
            folder_path="Drums/Kick Drums/Generic Kick/One Shots",
            source="profile_candidate_kick_claim",
            reason=(
                "BrainVoter had a top kick candidate and measured audio supported "
                f"single-event percussion: role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                f"raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=min(raw_score + 1.0, float(best_guess.rank) + 1.0),
            can_override=True,
            strength=max(0.92, min(0.98, 1.01 - 0.03 * max(0, best_guess.rank - 1))),
        )

    @staticmethod
    def _facts_support_low_kick_hit(facts: SharedAudioFacts | None) -> bool:
        """Return True when measured facts describe a low, single hit.

        A generic percussive one-shot can be a clap, snare, rim, or click.  The
        kick profile is allowed to rescue only when the audio evidence is
        physically low-heavy, not merely short and percussive.
        """
        low_pitched_hit = _feature_number_from_facts(facts, "low_pitched_hit_raw")
        low_total = _feature_number_from_facts(facts, "low_total")
        event_count = max(
            _feature_number_from_facts(facts, "event_count_estimate"),
            _feature_number_from_facts(facts, "onset_count"),
        )
        duration = _feature_number_from_facts(facts, "duration_sec")
        return bool(
            low_pitched_hit >= 0.72
            and low_total >= 0.60
            and (event_count <= 2.0 or event_count == 0.0)
            and (duration <= 1.25 or duration == 0.0)
        )

    def snare_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_score: float,
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim | None:
        """Return a snare claim from real BrainVoter snare evidence.

        This is intentionally narrower than generic clap/snap recall.  A short
        mid-body hit with measurable tail can be a snare even when a baby lane
        calls it clap or rim.  The claim still requires an actual Drums/Snares
        candidate from a voter; no source filename text is inspected.
        """
        role_supports_snare = role_name in {
            "percussive_one_shot",
            "protected_percussive_one_shot",
        }
        shape_supports_snare = shape_name in {"single_hit", "hit_with_tail"} and shape_confidence >= 0.70
        if not (role_supports_snare and shape_supports_snare):
            return None
        if not self._facts_support_snare_hit(facts):
            return None
        best_guess = self.best_brain_guess(
            brain_result,
            fragments=("snares", "snare", "clap snare"),
            max_rank=8,
            include_top={"Drums"},
        )
        if best_guess is not None:
            return claim_from_category_guess(
                guess=best_guess,
                folder_path="Drums/Snares/Generic Snare/One Shots",
                source="profile_candidate_snare_claim",
                reason=(
                    "BrainVoter had a nearby snare candidate and measured audio "
                    "supported a short mid-body snare-like hit: "
                    f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                    f"raw={raw.folder_path}"
                ),
                shared=raw.shared_candidates,
                raw_candidate_score=min(raw_score + 1.0, float(best_guess.rank) + 1.0),
                can_override=True,
                strength=max(0.96, min(0.98, 1.01 - 0.02 * max(0, best_guess.rank - 1))),
            )

        full_brain_snare = self._snare_guess_from_full_brain_facts(facts)
        if full_brain_snare is None:
            return None
        snare_rank, snare_path = full_brain_snare
        return claim_from_folder_path(
            folder_path="Drums/Snares/Generic Snare/One Shots",
            source="profile_candidate_snare_claim",
            reason=(
                "Full BrainVoter had a nearby snare candidate and measured audio "
                "supported a short mid-body snare-like hit: "
                f"role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                f"full_brain_snare={snare_path}; raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=min(raw_score + 1.0, float(snare_rank) + 1.0),
            brain_rank=snare_rank,
            physics_rank=None,
            shared_winner=snare_path,
            can_override=True,
            strength=max(0.965, min(0.985, 1.01 - 0.01 * max(0, snare_rank - 1))),
            is_real_candidate=True,
        )

    @staticmethod
    def _snare_guess_from_full_brain_facts(facts: SharedAudioFacts | None) -> tuple[int, str] | None:
        """Return a nearby full-brain snare candidate from shared facts.

        The product ensemble can over-weight baby recall lanes.  For this
        narrow rescue we only use the full brain's real ranked candidates, and
        only after measured snare-hit physics has already passed.
        """
        if facts is None or not isinstance(facts.evidence, dict):
            return None
        result = facts.evidence.get("full_brain_vote_result")
        if not isinstance(result, dict):
            return None
        guesses = result.get("top_guesses")
        if not isinstance(guesses, list):
            return None
        best: tuple[int, str] | None = None
        for index, guess in enumerate(guesses[:8], start=1):
            if not isinstance(guess, dict):
                continue
            path = str(guess.get("folder_path") or guess.get("label") or "").replace("\\", "/")
            normalized = path.lower()
            if not normalized.startswith("drums/"):
                continue
            if "snare" not in normalized:
                continue
            try:
                rank = int(guess.get("rank") or index)
            except Exception:
                rank = index
            candidate = (rank, path or "Drums/Snares/Generic Snare/One Shots")
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best

    @staticmethod
    def _facts_support_snare_hit(facts: SharedAudioFacts | None) -> bool:
        """Return True for a short mid-band drum hit with tail/body.

        Claps in this codebase often have stronger high-event energy and very
        little tail.  This guard keeps clap recall from stealing compact snares
        while leaving clean clap hits alone.
        """
        duration = _feature_number_from_facts(facts, "duration_sec")
        high_event = _shape_metric_from_facts(facts, "high_event_ratio")
        mid_event = _shape_metric_from_facts(facts, "mid_event_ratio")
        low_event = _shape_metric_from_facts(facts, "low_event_ratio")
        tail = _shape_metric_from_facts(facts, "tail_ratio")
        drumlike = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        percussive = _feature_number_from_facts(facts, "percussive_one_shot")
        return bool(
            0.0 < duration <= 0.45
            and mid_event >= 0.45
            and high_event <= 0.32
            and low_event <= 0.35
            and tail >= 0.08
            and (drumlike >= 0.60 or percussive >= 0.55)
        )

    def drum_loop_claim(
        self,
        *,
        raw: ConsensusClaim,
        brain_result: VoterResult,
        role_name: str,
        shape_name: str,
        shape_confidence: float,
        raw_score: float,
    ) -> ConsensusClaim | None:
        """Return a drum-loop claim from real BrainVoter loop evidence."""
        role_supports_drum_loop = role_name == "drum_loop" or _has_drum_loop_structure_support(
            role_name,
            role_name,
            shape_name,
            shape_confidence,
        )
        if not role_supports_drum_loop:
            return None
        best_guess = self.best_brain_guess(
            brain_result,
            fragments=("drum loop", "drum loops", "break", "breaks"),
            max_rank=4,
        )
        if best_guess is None:
            return None
        return claim_from_category_guess(
            guess=best_guess,
            folder_path="Drums/Drum Loops/Loops",
            source="profile_candidate_drum_loop_claim",
            reason=(
                "BrainVoter had a nearby drum-loop profile candidate and measured audio "
                f"supported drum-loop structure: role={role_name}, shape={shape_name}:{shape_confidence:.2f}; "
                f"raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=min(raw_score + 1.0, float(best_guess.rank) + 1.0),
            can_override=True,
            strength=max(0.90, min(0.96, 1.00 - 0.025 * max(0, best_guess.rank - 1))),
        )
