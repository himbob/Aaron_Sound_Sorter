# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Third-level category physics layer.

This layer turns the complete source-name-blind category panel catalog into
candidate score shaping.  It does not hard-route files.  It reads only measured
panel scores already stored on SharedAudioFacts and the candidate destination
folder path.  Candidate folder paths are output labels, not source evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aaron_sound_sorter.domain.physics_category_panels import ALL_CATEGORY_SPECS, CategoryPanelSpec
from aaron_sound_sorter.voters.physics_layer_types import PhysicsLayerDecision
from aaron_sound_sorter.voters.physics_layer_utils import (
    candidate_is_broad_drum,
    candidate_is_broad_fx,
    candidate_is_broad_instrument,
    clamp01,
    drum_candidate_matches_branch,
    fx_candidate_matches_branch,
    instrument_candidate_matches_branch,
    normalized_path,
    safe_float,
)


@dataclass(frozen=True)
class PhysicsCategoryMatch:
    """Candidate-level category panel match evidence."""

    candidate_path: str
    score_key: str
    panel_score: float
    role: str
    exact: bool
    branch_compatible: bool
    broad_compatible: bool


class PhysicsCategoryLayer:
    """Third physics layer: top -> branch -> measured leaf/category panels."""

    def __init__(self) -> None:
        self.specs_by_path = {normalized_path(spec.category_path): spec for spec in ALL_CATEGORY_SPECS}
        self.specs_by_top = self._group_by_top()

    def _group_by_top(self) -> dict[str, list[CategoryPanelSpec]]:
        grouped: dict[str, list[CategoryPanelSpec]] = {}
        for spec in ALL_CATEGORY_SPECS:
            grouped.setdefault(spec.top_family, []).append(spec)
        return grouped

    def apply(
        self, folder_path: str, score: float, decision: PhysicsLayerDecision
    ) -> tuple[float, list[str], dict[str, Any]]:
        """Shape one candidate score using exact leaf/category panel evidence."""
        folder = normalized_path(folder_path)
        match = self.match_candidate(folder, decision)
        selected = self.selected_panels(decision)
        evidence: dict[str, Any] = {
            "physics_category_layer_source": "category_panel_catalog",
            "physics_category_layer_top_family": decision.top_family,
            "physics_category_layer_branch": decision.branch,
            **selected,
        }
        if match is None:
            evidence.update(
                {
                    "physics_category_candidate_path": folder,
                    "physics_category_match_status": "no_matching_category_panel",
                }
            )
            return score, [], evidence

        if not self.has_category_score(decision, match.score_key):
            evidence.update(
                {
                    "physics_category_candidate_path": match.candidate_path,
                    "physics_category_candidate_score_key": match.score_key,
                    "physics_category_candidate_panel_score": round(float(match.panel_score), 6),
                    "physics_category_candidate_role": match.role,
                    "physics_category_exact_match": bool(match.exact),
                    "physics_category_branch_compatible": bool(match.branch_compatible),
                    "physics_category_broad_compatible": bool(match.broad_compatible),
                    "physics_category_match_status": "no_measured_category_panel_score",
                    "physics_category_score_before": round(float(score), 6),
                    "physics_category_score_after": round(float(score), 6),
                    "physics_category_adjustment_reasons": [],
                }
            )
            return score, [], evidence

        candidate_top = folder.split("/", 1)[0] if folder else ""
        if (
            decision.top_family in {"Drums", "Instruments", "FX"}
            and candidate_top != decision.top_family
            and decision.top_confidence >= 0.66
        ):
            evidence.update(
                {
                    "physics_category_candidate_path": match.candidate_path,
                    "physics_category_candidate_score_key": match.score_key,
                    "physics_category_candidate_panel_score": round(float(match.panel_score), 6),
                    "physics_category_candidate_role": match.role,
                    "physics_category_exact_match": bool(match.exact),
                    "physics_category_branch_compatible": bool(match.branch_compatible),
                    "physics_category_broad_compatible": bool(match.broad_compatible),
                    "physics_category_match_status": "skipped_for_top_family_mismatch",
                    "physics_category_score_before": round(float(score), 6),
                    "physics_category_score_after": round(float(score), 6),
                    "physics_category_adjustment_reasons": [],
                }
            )
            return score, [], evidence

        adjusted = float(score)
        reasons: list[str] = []
        panel = match.panel_score
        evidence.update(
            {
                "physics_category_candidate_path": match.candidate_path,
                "physics_category_candidate_score_key": match.score_key,
                "physics_category_candidate_panel_score": round(float(panel), 6),
                "physics_category_candidate_role": match.role,
                "physics_category_exact_match": bool(match.exact),
                "physics_category_branch_compatible": bool(match.branch_compatible),
                "physics_category_broad_compatible": bool(match.broad_compatible),
            }
        )

        if match.exact:
            if panel >= 0.68:
                target = max(0.08, 0.60 - 0.42 * panel)
                adjusted = min(adjusted, target)
                reasons.append(f"category_leaf_exact_target:{target:.3f}")
            elif panel >= 0.50:
                target = max(0.16, 0.76 - 0.32 * panel)
                adjusted = min(adjusted, target)
                reasons.append(f"category_leaf_exact_soft_target:{target:.3f}")
            elif panel <= 0.24 and self.top_is_decisive(decision):
                penalty = 0.12
                adjusted += penalty
                reasons.append(f"category_leaf_weak_exact:+{penalty:.2f}")
        elif match.branch_compatible:
            branch_best_score = safe_float(selected.get("physics_category_branch_selected_score", 0.0), 0.0)
            branch_best_key = str(selected.get("physics_category_branch_selected_key", ""))
            if branch_best_score >= 0.66 and branch_best_key and match.score_key != branch_best_key:
                margin = branch_best_score - panel
                if margin >= 0.18:
                    penalty = min(0.28, 0.08 + 0.55 * margin)
                    adjusted += penalty
                    reasons.append(f"category_leaf_branch_neighbor_mismatch:+{penalty:.2f}")
                elif panel >= 0.44:
                    target = max(0.22, 0.82 - 0.24 * panel)
                    adjusted = min(adjusted, target)
                    reasons.append(f"category_leaf_branch_neighbor_target:{target:.3f}")
            elif panel >= 0.55:
                target = max(0.20, 0.80 - 0.28 * panel)
                adjusted = min(adjusted, target)
                reasons.append(f"category_leaf_branch_compatible_target:{target:.3f}")
        elif match.broad_compatible and panel >= 0.44:
            target = max(0.30, 0.86 - 0.18 * panel)
            adjusted = min(adjusted, target)
            reasons.append(f"category_leaf_broad_compatible_target:{target:.3f}")
        elif self.top_is_decisive(decision):
            top_best_score = safe_float(selected.get("physics_category_top_selected_score", 0.0), 0.0)
            if top_best_score >= 0.70:
                penalty = 0.10 if panel >= 0.36 else 0.16
                adjusted += penalty
                reasons.append(f"category_leaf_incompatible_with_top_best:+{penalty:.2f}")

        evidence["physics_category_score_before"] = round(float(score), 6)
        evidence["physics_category_score_after"] = round(float(adjusted), 6)
        evidence["physics_category_adjustment_reasons"] = list(reasons)
        return max(0.0, adjusted), reasons, evidence

    def match_candidate(self, folder: str, decision: PhysicsLayerDecision) -> PhysicsCategoryMatch | None:
        """Return the exact or compatible category panel for a candidate."""
        spec = self.specs_by_path.get(folder)
        if spec is not None:
            return PhysicsCategoryMatch(
                candidate_path=folder,
                score_key=spec.score_key,
                panel_score=self.category_score(decision, spec.score_key),
                role=spec.role,
                exact=True,
                branch_compatible=self.spec_matches_branch(spec, decision.branch),
                broad_compatible=self.spec_is_broad_for_branch(spec, decision.branch),
            )
        # Most current Stage 4 labels are exact paths.  Keep a conservative
        # fallback for future aliases where the candidate is below a known spec.
        for candidate_spec in self.specs_by_top.get(decision.top_family, []):
            path = normalized_path(candidate_spec.category_path)
            if folder.startswith(path + "/") or path.startswith(folder + "/"):
                return PhysicsCategoryMatch(
                    candidate_path=folder,
                    score_key=candidate_spec.score_key,
                    panel_score=self.category_score(decision, candidate_spec.score_key),
                    role=candidate_spec.role,
                    exact=False,
                    branch_compatible=self.spec_matches_branch(candidate_spec, decision.branch),
                    broad_compatible=self.spec_is_broad_for_branch(candidate_spec, decision.branch),
                )
        return None

    def selected_panels(self, decision: PhysicsLayerDecision) -> dict[str, Any]:
        """Return top and branch selected leaf panels for diagnostics."""
        top_specs = self.specs_by_top.get(decision.top_family, [])
        top_ranked = self.rank_specs(top_specs, decision)
        branch_ranked = self.rank_specs(
            [spec for spec in top_specs if self.spec_matches_branch(spec, decision.branch)],
            decision,
        )
        data: dict[str, Any] = {}
        if top_ranked:
            spec, value = top_ranked[0]
            data.update(
                {
                    "physics_category_top_selected_path": spec.category_path,
                    "physics_category_top_selected_key": spec.score_key,
                    "physics_category_top_selected_score": round(float(value), 6),
                    "physics_category_top_top5": [
                        {
                            "path": ranked_spec.category_path,
                            "score_key": ranked_spec.score_key,
                            "score": round(float(score), 6),
                        }
                        for ranked_spec, score in top_ranked[:5]
                    ],
                }
            )
        else:
            data.update(
                {
                    "physics_category_top_selected_path": "",
                    "physics_category_top_selected_key": "",
                    "physics_category_top_selected_score": 0.0,
                    "physics_category_top_top5": [],
                }
            )
        if branch_ranked:
            spec, value = branch_ranked[0]
            data.update(
                {
                    "physics_category_branch_selected_path": spec.category_path,
                    "physics_category_branch_selected_key": spec.score_key,
                    "physics_category_branch_selected_score": round(float(value), 6),
                    "physics_category_branch_top5": [
                        {
                            "path": ranked_spec.category_path,
                            "score_key": ranked_spec.score_key,
                            "score": round(float(score), 6),
                        }
                        for ranked_spec, score in branch_ranked[:5]
                    ],
                }
            )
        else:
            data.update(
                {
                    "physics_category_branch_selected_path": "",
                    "physics_category_branch_selected_key": "",
                    "physics_category_branch_selected_score": 0.0,
                    "physics_category_branch_top5": [],
                }
            )
        return data

    def rank_specs(
        self,
        specs: list[CategoryPanelSpec],
        decision: PhysicsLayerDecision,
    ) -> list[tuple[CategoryPanelSpec, float]]:
        """Rank specs by measured panel score, highest first."""
        ranked = [(spec, self.category_score(decision, spec.score_key)) for spec in specs]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def category_score(self, decision: PhysicsLayerDecision, score_key: str) -> float:
        """Read one category panel score from decision evidence."""
        flat = decision.evidence.get("physics_category_panel_flat", {})
        if isinstance(flat, dict):
            return clamp01(safe_float(flat.get(score_key, 0.0), 0.0))
        return 0.0

    def has_category_score(self, decision: PhysicsLayerDecision, score_key: str) -> bool:
        """Return True only when the measured category panel was actually computed."""
        flat = decision.evidence.get("physics_category_panel_flat", {})
        return isinstance(flat, dict) and score_key in flat

    def top_is_decisive(self, decision: PhysicsLayerDecision) -> bool:
        """Return True when top/branch evidence is strong enough to shape leaves."""
        return bool(
            decision.top_family in {"Drums", "Instruments", "FX"}
            and decision.top_confidence >= 0.66
            and decision.branch_confidence >= 0.46
        )

    def spec_matches_branch(self, spec: CategoryPanelSpec, branch: str) -> bool:
        """Return whether a category spec belongs to the measured branch."""
        folder = normalized_path(spec.category_path)
        if spec.top_family == "Drums":
            return drum_candidate_matches_branch(folder, branch)
        if spec.top_family == "Instruments":
            return instrument_candidate_matches_branch(folder, branch)
        if spec.top_family == "FX":
            return fx_candidate_matches_branch(folder, branch)
        return False

    def spec_is_broad_for_branch(self, spec: CategoryPanelSpec, branch: str) -> bool:
        """Return whether a spec is a broad safe bucket for the measured branch."""
        folder = normalized_path(spec.category_path)
        if spec.top_family == "Drums":
            return candidate_is_broad_drum(folder)
        if spec.top_family == "Instruments":
            return candidate_is_broad_instrument(folder)
        if spec.top_family == "FX":
            return candidate_is_broad_fx(folder)
        return False
