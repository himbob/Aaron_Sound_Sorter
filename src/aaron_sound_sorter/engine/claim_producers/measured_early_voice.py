# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Human/Voice evidence mixin for measured early adjudication."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.measured_early_voice_builder import (
    VoiceEvidenceBuilderMixin,
)
from aaron_sound_sorter.engine.claim_producers.measured_early_voice_claims import VoiceClaimGuardMixin


class VoiceEvidenceMixin(VoiceEvidenceBuilderMixin, VoiceClaimGuardMixin):
    """Compose Human/Voice evidence building and rescue guards."""
