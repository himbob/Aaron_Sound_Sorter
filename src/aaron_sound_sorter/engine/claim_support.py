# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Shared claim construction helpers for decision claim producers.

This public mixin composes smaller helper mixins.  Keeping this import point
stable lets existing claim producers inherit ``ClaimSupportMixin`` while the
implementation stays readable and testable.
"""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_broadening import ClaimBroadeningMixin
from aaron_sound_sorter.engine.claim_candidate_lookup import ClaimCandidateLookupMixin
from aaron_sound_sorter.engine.claim_review_builders import ClaimReviewBuilderMixin


class ClaimSupportMixin(
    ClaimCandidateLookupMixin,
    ClaimReviewBuilderMixin,
    ClaimBroadeningMixin,
):
    """Reusable lookup, redirect, review, and broadening helpers."""
