"""Winner preselection for shared Brain/Physics consensus rows."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.consensus_support import role_signature_from_row
from aaron_sound_sorter.voters.scoring_tools import detected_parent_role_name, role_strength


class ConsensusWinnerPreselector:
    """Choose the raw shared winner before consensus sanity claims are built."""

    def choose(self, shared: list[dict[str, Any]], facts: SharedAudioFacts) -> dict[str, Any]:
        """Return the raw shared row without role/shape preselection.

        The shared-candidate table is already sorted by BrainVoter/PhysicsVoter
        rank agreement.  Earlier builds used measured roles here to replace the
        raw winner before the arbiter saw it.  That recreated the old rule-tree
        failure where ``percussive_one_shot`` could turn bass, boom, or sub-hit
        evidence into Drums.

        Role/shape facts now stay diagnostic until FamilyClaimArbiter compares
        explicit claims.
        """
        del facts
        return shared[0] if shared else {}

    @staticmethod
    def prefer_dynamic_broad_role_candidate(shared: list[dict[str, Any]], facts: SharedAudioFacts) -> dict[str, Any]:
        """Choose hierarchy depth using measured-role context."""
        if not shared:
            return {}
        leaf_winner = shared[0]
        best_top = str(leaf_winner.get("top_family", ""))
        candidates = [row for row in shared if str(row.get("top_family", "")) == best_top]
        if not candidates:
            return leaf_winner

        roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
        has_measured_role = detected_parent_role_name(roles if isinstance(roles, dict) else {}) != "unknown"
        best_score = float(leaf_winner.get("combined_rank_score", 9999.0))

        if not has_measured_role and _depth(leaf_winner) <= 3:
            specific = [
                row
                for row in candidates
                if _depth(row) > _depth(leaf_winner)
                and float(row.get("combined_rank_score", 9999.0)) <= best_score + 2.0
                and int(row.get("brain_rank", 9999)) <= 5
                and int(row.get("physics_rank", 9999)) <= 5
            ]
            if specific:
                specific.sort(
                    key=lambda row: (
                        float(row.get("combined_rank_score", 9999.0)),
                        int(row.get("physics_rank", 9999)),
                        int(row.get("brain_rank", 9999)),
                        str(row.get("label", "")),
                    )
                )
                return specific[0]

        if not has_measured_role:
            return leaf_winner

        broad = [
            row
            for row in candidates
            if _depth(row) < _depth(leaf_winner)
            and int(row.get("physics_rank", 9999)) <= 10
            and int(row.get("brain_rank", 9999)) <= 20
            and float(row.get("combined_rank_score", 9999.0)) <= best_score + 8.0
        ]
        if not broad:
            return leaf_winner
        broad.sort(
            key=lambda row: (
                _depth(row),
                int(row.get("physics_rank", 9999)),
                float(row.get("combined_rank_score", 9999.0)),
                str(row.get("label", "")),
            )
        )
        return broad[0]

    @staticmethod
    def prefer_profile_role_candidate(
        shared: list[dict[str, Any]],
        current_winner: dict[str, Any],
        facts: SharedAudioFacts,
    ) -> dict[str, Any]:
        """Prefer a close same-family candidate with stronger measured-role fit."""
        if not shared or not current_winner:
            return current_winner
        roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(roles, dict):
            return current_winner
        parent_role = detected_parent_role_name(roles)
        parent_strength = role_strength(roles, parent_role) if parent_role != "unknown" else 0.0
        if parent_strength < 0.75:
            return current_winner

        current_signature = role_signature_from_row(current_winner)
        current_role_fit = role_strength(current_signature, parent_role)
        winner_rank = float(current_winner.get("combined_rank_score", 9999.0))
        winner_top = str(current_winner.get("top_family", ""))
        candidates: list[dict[str, Any]] = []
        for row in shared:
            if str(row.get("top_family", "")) != winner_top:
                continue
            if str(row.get("label", "")) == str(current_winner.get("label", "")):
                continue
            if float(row.get("combined_rank_score", 9999.0)) > winner_rank + 3.0:
                continue
            if int(row.get("brain_rank", 9999)) > 8 or int(row.get("physics_rank", 9999)) > 12:
                continue
            row_fit = role_strength(role_signature_from_row(row), parent_role)
            if row_fit < 0.55:
                continue
            if row_fit <= current_role_fit + 0.12:
                continue
            candidates.append(row)
        if not candidates:
            return current_winner
        candidates.sort(
            key=lambda row: (
                -role_strength(role_signature_from_row(row), parent_role),
                float(row.get("combined_rank_score", 9999.0)),
                int(row.get("physics_rank", 9999)),
                int(row.get("brain_rank", 9999)),
                str(row.get("label", "")),
            )
        )
        return candidates[0]

    @staticmethod
    def prefer_cross_family_role_candidate(
        shared: list[dict[str, Any]],
        current_winner: dict[str, Any],
        facts: SharedAudioFacts,
    ) -> dict[str, Any]:
        """Prefer a better same-role family candidate when the raw winner is weak."""
        if not shared or not current_winner:
            return current_winner
        roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(roles, dict):
            return current_winner
        parent_role = detected_parent_role_name(roles)
        parent_strength = role_strength(roles, parent_role) if parent_role != "unknown" else 0.0
        if parent_role != "percussive_one_shot" or parent_strength < 0.75:
            return current_winner
        role_evidence = roles.get("evidence", {}) if isinstance(roles.get("evidence", {}), dict) else {}
        try:
            low_pitched_hit = float(role_evidence.get("low_pitched_hit_raw", 0.0) or 0.0)
        except Exception:
            low_pitched_hit = 0.0
        if str(current_winner.get("top_family", "")) == "Drums":
            return current_winner
        current_fit = role_strength(role_signature_from_row(current_winner), parent_role)
        candidates: list[dict[str, Any]] = []
        for row in shared:
            if str(row.get("top_family", "")) != "Drums":
                continue
            combined = float(row.get("combined_rank_score", 9999.0))
            brain_rank = int(row.get("brain_rank", 9999))
            physics_rank = int(row.get("physics_rank", 9999))
            if low_pitched_hit >= 0.80:
                if combined > 35.0 or min(brain_rank, physics_rank) > 12 or max(brain_rank, physics_rank) > 28:
                    continue
            elif combined > 20.0:
                continue
            if low_pitched_hit < 0.80 and (min(brain_rank, physics_rank) > 4 or max(brain_rank, physics_rank) > 20):
                continue
            fit = role_strength(role_signature_from_row(row), parent_role)
            required_fit = 0.90 if low_pitched_hit >= 0.80 else max(0.78, current_fit + 0.12)
            if fit < required_fit:
                continue
            candidates.append(row)
        if not candidates:
            return current_winner
        candidates.sort(
            key=lambda row: (
                -role_strength(role_signature_from_row(row), parent_role),
                float(row.get("combined_rank_score", 9999.0)),
                int(row.get("physics_rank", 9999)),
                int(row.get("brain_rank", 9999)),
                str(row.get("label", "")),
            )
        )
        return candidates[0]


def _depth(row: dict[str, Any]) -> int:
    """Return candidate label depth."""
    return len([part for part in str(row.get("label", "")).split("/") if part])
