# v31.138 marker: measured musical-loop depth restoration installed
# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Central mapping from winning claims to output folders."""

from __future__ import annotations

from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class PlacementResolver:
    """Resolve a winning claim into the one final output folder path.

    The resolver owns broad synthetic bucket names.  Real candidate claims may
    carry a folder path from the brain/taxonomy candidate list; those paths are
    internal category labels, not source filenames.
    """

    FAMILY_FOLDER_MAP: dict[tuple[str, str], str] = {
        ("Drums", "Drum Loops"): "Drums/Drum Loops/Loops",
        ("Drums", "Kick One Shot"): "Drums/Kick Drums/Generic Kick/One Shots",
        ("Drums", "Percussion One Shot"): "Drums/Percussion/Generic Percussion/One Shots",
        ("Instruments", "Bass Loops"): "Instruments/Bass/Bass Loops",
        ("Instruments", "Instrument Loops"): "Instruments/Instrument Loops/Loops",
        ("Instruments", "Brass Woodwinds"): "Instruments/Brass and Woodwinds/Loops",
        ("Instruments", "Voice"): "Instruments/Voice/Phrase/One Shots",
        ("FX", "Human and Voice FX"): "FX/Human and Voice FX",
        ("FX", "Risers and Builds"): "FX/Structural and Transitional FX/Risers and Builds",
        ("FX", "Drops and Downlifters"): "FX/Structural and Transitional FX/Drops and Downlifters",
        ("FX", "Hybrid Designed FX"): "FX/Hybrid Designed FX",
        ("Textures", "Hybrid Textures"): "Textures/Hybrid Textures",
        ("_TO_REVIEW", "Measured Role Conflict"): "_TO_REVIEW/Measured Role Conflict",
    }

    @staticmethod
    def _should_preserve_specific_internal_path(claim: ConsensusClaim) -> bool:
        """Return True for specific taxonomy paths carried by measured claims.

        Broad claims must stay broad.  Only real candidate claims preserve their
        internal folder path, and that is handled before this method is called.
        This keeps measured Brass/Woodwinds rescues mapped through the central
        resolver table instead of leaking Saxophone-specific paths from
        eligibility code.
        """
        if not claim.folder_path:
            return False
        path = claim.folder_path.lower().replace("\\", "/")
        if claim.source in {
            "final_measured_sax_loop_invariant",
            "final_measured_musical_loop_sax_depth_invariant",
            "final_measured_voice_invariant",
            "final_measured_transition_fx_invariant",
            "raw_contract_true_voice_instrument_over_human_voice_fx",
            "true_voice_instrument_rehome",
        }:
            return True
        return bool(claim.source == "final_measured_branch_loop_broad_bucket" and "sax" in path)

    def resolve(self, claim: ConsensusClaim) -> str:
        """Return the folder path for a winning claim."""
        if claim.is_review and claim.folder_path:
            return claim.folder_path.strip("/")
        if claim.is_real_candidate and claim.folder_path:
            return claim.folder_path.strip("/")
        if self._should_preserve_specific_internal_path(claim):
            return claim.folder_path.strip("/")
        mapped = self.FAMILY_FOLDER_MAP.get((claim.family, claim.sub_family))
        if mapped:
            return mapped
        if claim.folder_path:
            return claim.folder_path.strip("/")
        if claim.label:
            return claim.label.strip("/")
        if claim.family == "_TO_REVIEW":
            return "_TO_REVIEW/Measured Role Conflict"
        return "/".join(part for part in [claim.family, claim.sub_family] if part).strip("/")
