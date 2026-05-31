# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Initial keep-raw guards for smoke-stability broadening."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _path_has_any,
    _role_strength_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class SmokeInitialGuardMixin:
    """Keep useful raw buckets before broad smoke-stability rescues run."""

    def _should_keep_existing_stable_bucket(
        self,
        raw: ConsensusClaim,
        raw_path: str,
        role: str,
        measured_role: str,
        shape: str,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when smoke-stability should not alter the raw winner."""
        if raw.final_top == "Instruments" and _path_has_any(raw_path, ("bass", "808", "sub bass", "synth bass")):
            return True
        if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops", "percussion")):
            return True
        if raw.final_top == "FX" and _path_has_any(raw_path, ("human and voice", "voice", "vocal")):
            direct_pitched_strength = max(
                _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
                _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
            )
            measured_vocal_strength = max(
                _role_strength_from_facts(facts, "vocal_music_phrase"),
                _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
            )
            return bool(
                role not in {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}
                and measured_role
                not in {
                    "pitched_music_loop",
                    "pitched_music_phrase",
                    "pitched_reed_or_instrument_loop",
                    "pitched_reed_or_instrument_phrase",
                }
                and shape not in {"pitched_phrase", "bass_phrase", "sustained_pad"}
                and not (direct_pitched_strength >= 0.84 and measured_vocal_strength < 0.55)
            )
        return False
