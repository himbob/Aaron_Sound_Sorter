from __future__ import annotations

from aaron_sound_sorter.neural_audio.prompt_brain import PromptCategoryScore
from tools.benchmark_clap_prompt_ranking import empty_metrics, rank_prompt_scores, update_metrics


def _score(path: str, positive: float, negative: float) -> PromptCategoryScore:
    return PromptCategoryScore(
        path, positive, negative, positive - negative, "support", positive, "confusion", negative
    )


def test_prompt_ranking_confusion_weight_is_explicit() -> None:
    scores = (
        _score("Instruments/Voice/Phrase/One Shots", 0.8, 0.7),
        _score("Instruments/Woodwinds/Saxophone/Alto/One Shots", 0.7, 0.1),
    )

    positive_only = rank_prompt_scores(scores, negative_weight=0.0)
    confusion_aware = rank_prompt_scores(scores, negative_weight=1.0)

    assert positive_only[0].path.startswith("Instruments/Voice")
    assert confusion_aware[0].path.startswith("Instruments/Woodwinds")


def test_prompt_metrics_separate_identity_from_structure() -> None:
    ranked = (
        _score("Instruments/Voice/Phrase/Loops", 0.8, 0.1),
        _score("Instruments/Voice/Phrase/One Shots", 0.7, 0.1),
    )
    metrics = empty_metrics()

    update_metrics(metrics, ranked, "Instruments/Voice/Phrase/One Shots")

    assert metrics["exact_top1"] == 0
    assert metrics["exact_top3"] == 1
    assert metrics["identity_top1"] == 1
