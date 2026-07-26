"""Read-only mapping from PANNs AudioSet events to Aaron broad families."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry

CORROBORATING_EVENT_FLOOR = 0.05
CORROBORATING_EVENT_THRESHOLD = 0.25


@dataclass(frozen=True)
class PannsEventScore:
    """One raw PANNs AudioSet event score."""

    label: str
    score: float


@dataclass(frozen=True)
class PannsEventMapping:
    """Broad Aaron support and contradiction contract for one event."""

    event_label: str
    supports: tuple[str, ...]
    contradicts: tuple[str, ...]
    minimum_support_score: float
    neutral: bool = False
    ambiguous: bool = False


@dataclass(frozen=True)
class PannsMappedEvidence:
    """Visible broad-family evidence for one candidate taxonomy label."""

    candidate_label: str
    support_score: float
    contradiction_score: float
    supporting_events: tuple[PannsEventScore, ...]
    contradicting_events: tuple[PannsEventScore, ...]
    neutral_events: tuple[PannsEventScore, ...]


class PannsMappingRegistry:
    """Validate and apply a versioned PANNs-to-taxonomy mapping.

    Args:
        mappings: AudioSet event mappings keyed by exact event label.
        mapping_version: Mapping version written to foundation reports.

    Side Effects:
        None.
    """

    def __init__(self, mappings: dict[str, PannsEventMapping], *, mapping_version: str) -> None:
        self.mappings = dict(mappings)
        self.mapping_version = str(mapping_version)

    @classmethod
    def load(cls, path: Path, taxonomy: TaxonomyRegistry) -> PannsMappingRegistry:
        """Load and validate mappings against canonical category prefixes."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if bool(payload.get("production_ownership_enabled", False)):
            raise ValueError("PANNs detailed production ownership must remain disabled")
        raw_events = payload.get("events", {})
        if not isinstance(raw_events, dict):
            raise ValueError("PANNs mapping events must be an object")
        mappings: dict[str, PannsEventMapping] = {}
        for event_label, raw_mapping in raw_events.items():
            if not isinstance(raw_mapping, dict):
                raise ValueError(f"PANNs mapping must be an object: {event_label}")
            mapping = PannsEventMapping(
                event_label=str(event_label),
                supports=_string_tuple(raw_mapping.get("supports", [])),
                contradicts=_string_tuple(raw_mapping.get("contradicts", [])),
                minimum_support_score=float(raw_mapping.get("minimum_support_score", 0.35)),
                neutral=bool(raw_mapping.get("neutral", False)),
                ambiguous=bool(raw_mapping.get("ambiguous", False)),
            )
            _validate_mapping_prefixes(mapping, taxonomy)
            mappings[mapping.event_label] = mapping
        return cls(mappings, mapping_version=str(payload.get("mapping_version", "")))

    def evaluate(
        self,
        events: tuple[PannsEventScore, ...],
        candidate_label: str,
    ) -> PannsMappedEvidence:
        """Map raw broad events to support or contradiction for a candidate."""
        supporting: list[PannsEventScore] = []
        contradicting: list[PannsEventScore] = []
        weak_supporting: list[PannsEventScore] = []
        weak_contradicting: list[PannsEventScore] = []
        neutral: list[PannsEventScore] = []
        for event in events:
            mapping = self.mappings.get(event.label)
            if mapping is None:
                continue
            if mapping.neutral or mapping.ambiguous:
                neutral.append(event)
            elif _matches_any_prefix(candidate_label, mapping.supports):
                if event.score >= mapping.minimum_support_score:
                    supporting.append(event)
                elif event.score >= CORROBORATING_EVENT_FLOOR:
                    weak_supporting.append(event)
            elif _matches_any_prefix(candidate_label, mapping.contradicts):
                if event.score >= mapping.minimum_support_score:
                    contradicting.append(event)
                elif event.score >= CORROBORATING_EVENT_FLOOR:
                    weak_contradicting.append(event)
            else:
                neutral.append(event)
        support_score = max((event.score for event in supporting), default=0.0)
        contradiction_score = max((event.score for event in contradicting), default=0.0)
        corroborating_support = _corroborating_score(weak_supporting)
        corroborating_contradiction = _corroborating_score(weak_contradicting)
        if len(weak_supporting) >= 2 and corroborating_support >= CORROBORATING_EVENT_THRESHOLD:
            supporting.extend(weak_supporting)
            support_score = max(support_score, corroborating_support)
        if len(weak_contradicting) >= 2 and corroborating_contradiction >= CORROBORATING_EVENT_THRESHOLD:
            contradicting.extend(weak_contradicting)
            contradiction_score = max(contradiction_score, corroborating_contradiction)
        return PannsMappedEvidence(
            candidate_label=candidate_label,
            support_score=support_score,
            contradiction_score=contradiction_score,
            supporting_events=tuple(sorted(supporting, key=lambda row: (-row.score, row.label))),
            contradicting_events=tuple(sorted(contradicting, key=lambda row: (-row.score, row.label))),
            neutral_events=tuple(sorted(neutral, key=lambda row: (-row.score, row.label))),
        )


def _validate_mapping_prefixes(mapping: PannsEventMapping, taxonomy: TaxonomyRegistry) -> None:
    for prefix in (*mapping.supports, *mapping.contradicts):
        if not any(path == prefix or path.startswith(f"{prefix}/") for path in taxonomy.categories):
            raise ValueError(
                f"PANNs mapping prefix is absent from canonical taxonomy: {mapping.event_label} -> {prefix}"
            )
    if not 0.0 <= mapping.minimum_support_score <= 1.0:
        raise ValueError(f"PANNs support threshold must be in [0, 1]: {mapping.event_label}")


def _matches_any_prefix(candidate_label: str, prefixes: tuple[str, ...]) -> bool:
    return any(candidate_label == prefix or candidate_label.startswith(f"{prefix}/") for prefix in prefixes)


def _corroborating_score(events: list[PannsEventScore]) -> float:
    """Combine several weak, related event scores without treating them as labels."""
    remaining_probability = 1.0
    for event in events:
        remaining_probability *= 1.0 - max(0.0, min(1.0, event.score))
    return 1.0 - remaining_probability


def _string_tuple(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise ValueError("PANNs mapping prefixes must be a list")
    return tuple(dict.fromkeys(str(value).strip() for value in payload if str(value).strip()))
