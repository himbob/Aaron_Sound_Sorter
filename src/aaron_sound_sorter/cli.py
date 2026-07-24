"""Command-line parsing for Aaron Sound Sorter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from aaron_sound_sorter.domain.models import SortRequest

DEFAULT_BRAIN = "stage4_folder_brain.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CommandLineParser:
    """Parse CLI arguments without running sort logic."""

    def build(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(prog="Aaron_Sound_Sorter.py")
        subcommands = parser.add_subparsers(dest="command", required=True)
        self.add_sort_command(subcommands)
        self.add_brain_lab_command(subcommands)
        self.add_self_test_command(subcommands)
        return parser

    def parse(self, argv: list[str] | None) -> argparse.Namespace:
        """Parse CLI arguments."""
        return self.build().parse_args(argv)

    @staticmethod
    def add_sort_command(subcommands: argparse._SubParsersAction) -> None:
        """Add the product sort command."""
        sort_parser = subcommands.add_parser(
            "sort", help="Sort a ZIP, folder, or audio file using the committee sorter."
        )
        sort_parser.add_argument("input_path", help="Input ZIP, folder, or audio file.")
        sort_parser.add_argument("output_dir", help="Output folder to create.")
        sort_parser.add_argument(
            "--brain", default=DEFAULT_BRAIN, help="Full brain JSON path. Default: stage4_folder_brain.json"
        )
        sort_parser.add_argument(
            "--baby-brain",
            default="",
            help="Legacy optional balanced recall baby brain. Used as spread baby when explicit spread path is omitted.",
        )
        sort_parser.add_argument(
            "--core-baby-brain",
            default="",
            help="Precision baby brain trained on central/clean anchors. Default: stage4_folder_brain_core_baby.json if present.",
        )
        sort_parser.add_argument(
            "--spread-baby-brain",
            default="",
            help="Balanced recall baby brain trained on diverse clean anchors. Default: stage4_folder_brain_spread_baby.json if present, then --baby-brain, then stage4_folder_brain_baby.json.",
        )
        sort_parser.add_argument(
            "--outlier-baby-brain",
            default="",
            help="Outlier/edge baby brain trained on plausible edge anchors. Default: stage4_folder_brain_outlier_baby.json if present.",
        )
        sort_parser.add_argument(
            "--harmonic-core-baby-brain",
            default="",
            help="Optional harmonic-core-trained precision baby brain. Diagnostic by default.",
        )
        sort_parser.add_argument(
            "--harmonic-spread-baby-brain",
            default="",
            help="Optional harmonic-core-trained spread baby brain. Diagnostic by default.",
        )
        sort_parser.add_argument(
            "--harmonic-outlier-baby-brain",
            default="",
            help="Optional harmonic-core-trained outlier baby brain. Diagnostic by default.",
        )
        sort_parser.add_argument(
            "--use-baby-brains-in-sort",
            action="store_true",
            default=True,
            help="Legacy explicit opt-in. Raw baby recall lanes are now part of the product BrainEnsembleVoter by default when baby brain files are present.",
        )
        sort_parser.add_argument(
            "--disable-baby-brain-ensemble",
            action="store_true",
            help="Diagnostic fallback: use only the full brain as the product BrainVoter result.",
        )
        sort_parser.add_argument(
            "--use-harmonic-brains-in-sort",
            action="store_true",
            help="Let harmonic-core-trained baby lanes affect the final product brain vote. Requires harmonic baby brain files.",
        )
        sort_parser.add_argument(
            "--candidate-count", type=int, default=100, help="Number of category candidates each voter returns."
        )
        sort_parser.add_argument(
            "--workers",
            type=int,
            default=0,
            help="Parallel per-file sort workers. Default: AARON_SORT_WORKERS or auto, capped at 6.",
        )
        sort_parser.add_argument(
            "--analysis-cache",
            action="store_true",
            help=(
                "Reuse persistent measured audio analysis between runs. "
                "Stores fingerprints/diagnostics only, never final folders."
            ),
        )
        sort_parser.add_argument(
            "--analysis-cache-dir",
            default="",
            help="Directory for --analysis-cache. Default: AARON_ANALYSIS_CACHE_DIR or _reports/analysis_cache/cli_v1.",
        )
        sort_parser.add_argument(
            "--no-analysis-cache",
            action="store_true",
            help="Disable persistent measured-analysis cache even if the environment enables it.",
        )
        sort_parser.add_argument("--no-zip", action="store_true", help="Do not create Aaron_Sorted_Sounds.zip.")
        sort_parser.add_argument(
            "--no-neural",
            action="store_true",
            help="Disable configured source-name-blind neural authority for this run.",
        )
        sort_parser.set_defaults(command_kind="sort")

    @staticmethod
    def add_brain_lab_command(subcommands: argparse._SubParsersAction) -> None:
        """Add a diagnostic command that reports voter lanes without sorting."""
        lab_parser = subcommands.add_parser(
            "brain-lab", help="Print raw voter lane evidence for one audio file without sorting or gating."
        )
        lab_parser.add_argument("input_path", help="Audio file to inspect.")
        lab_parser.add_argument(
            "--brain", default=DEFAULT_BRAIN, help="Full brain JSON path. Default: stage4_folder_brain.json"
        )
        lab_parser.add_argument("--baby-brain", default="", help="Legacy optional balanced recall baby brain.")
        lab_parser.add_argument(
            "--core-baby-brain", default="", help="Precision baby brain trained on central/clean anchors."
        )
        lab_parser.add_argument(
            "--spread-baby-brain", default="", help="Balanced recall baby brain trained on diverse clean anchors."
        )
        lab_parser.add_argument(
            "--outlier-baby-brain", default="", help="Outlier/edge baby brain trained on plausible edge anchors."
        )
        lab_parser.add_argument(
            "--harmonic-brain", default="", help="Full brain trained on harmonic-core fingerprints."
        )
        lab_parser.add_argument(
            "--harmonic-core-baby-brain", default="", help="Harmonic-core-trained precision baby brain."
        )
        lab_parser.add_argument(
            "--harmonic-spread-baby-brain", default="", help="Harmonic-core-trained spread baby brain."
        )
        lab_parser.add_argument(
            "--harmonic-outlier-baby-brain", default="", help="Harmonic-core-trained outlier baby brain."
        )
        lab_parser.add_argument("--top-n", type=int, default=10, help="Number of guesses to print per lane.")
        lab_parser.add_argument("--json-out", default="", help="Optional path for a JSON diagnostic report.")
        lab_parser.set_defaults(command_kind="brain-lab")

    @staticmethod
    def add_self_test_command(subcommands: argparse._SubParsersAction) -> None:
        """Add a tiny app-level smoke test command."""
        self_test = subcommands.add_parser("self-test", help="Run a small app wiring self-test.")
        self_test.set_defaults(command_kind="self-test")


def _optional_path(value: str) -> Path | None:
    value = str(value or "").strip()
    return Path(value).expanduser() if value else None


def _sort_workers_from_args(args: argparse.Namespace) -> int:
    """Return a conservative worker count for per-file classification."""
    raw = int(getattr(args, "workers", 0) or 0)
    if raw > 0:
        return max(1, raw)
    env_value = str(os.environ.get("AARON_SORT_WORKERS", "")).strip()
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            return 1
    cpu_count = os.cpu_count() or 2
    return max(1, min(6, cpu_count - 1))


def _analysis_cache_enabled_from_args(args: argparse.Namespace) -> bool:
    """Return whether persistent measured-analysis cache is enabled."""
    if bool(getattr(args, "no_analysis_cache", False)):
        return False
    if bool(getattr(args, "analysis_cache", False)):
        return True
    if str(getattr(args, "analysis_cache_dir", "") or "").strip():
        return True
    raw = str(os.environ.get("AARON_ANALYSIS_CACHE", "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(str(os.environ.get("AARON_ANALYSIS_CACHE_DIR", "")).strip())


def _analysis_cache_dir_from_args(args: argparse.Namespace) -> Path:
    """Return persistent measured-analysis cache directory for CLI runs."""
    configured = str(getattr(args, "analysis_cache_dir", "") or "").strip()
    if not configured:
        configured = str(os.environ.get("AARON_ANALYSIS_CACHE_DIR", "")).strip()
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / "_reports" / "analysis_cache" / "cli_v1"


def request_from_args(args: argparse.Namespace) -> SortRequest:
    """Build a SortRequest from parsed CLI args."""
    return SortRequest(
        input_path=Path(args.input_path).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        brain_path=Path(args.brain).expanduser(),
        baby_brain_path=_optional_path(getattr(args, "baby_brain", "")),
        core_baby_brain_path=_optional_path(getattr(args, "core_baby_brain", "")),
        spread_baby_brain_path=_optional_path(getattr(args, "spread_baby_brain", "")),
        outlier_baby_brain_path=_optional_path(getattr(args, "outlier_baby_brain", "")),
        harmonic_core_baby_brain_path=_optional_path(getattr(args, "harmonic_core_baby_brain", "")),
        harmonic_spread_baby_brain_path=_optional_path(getattr(args, "harmonic_spread_baby_brain", "")),
        harmonic_outlier_baby_brain_path=_optional_path(getattr(args, "harmonic_outlier_baby_brain", "")),
        use_baby_brains_in_sort=bool(getattr(args, "use_baby_brains_in_sort", True))
        and not bool(getattr(args, "disable_baby_brain_ensemble", False)),
        use_harmonic_brains_in_sort=bool(getattr(args, "use_harmonic_brains_in_sort", False)),
        write_zip=not bool(args.no_zip),
        candidate_count=int(args.candidate_count),
        sort_workers=_sort_workers_from_args(args),
        use_persistent_analysis_cache=_analysis_cache_enabled_from_args(args),
        analysis_cache_dir=_analysis_cache_dir_from_args(args),
        use_neural_runtime=not bool(getattr(args, "no_neural", False)),
        neural_project_root=PROJECT_ROOT,
    )
