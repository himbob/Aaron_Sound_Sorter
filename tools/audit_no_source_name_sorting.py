#!/usr/bin/env python3
"""Audit that production sorting code is blind to source names.

Hard invariant:
  Production sorting, voting, role detection, eligibility, consensus, and final
  placement must not use producer filenames, source folder names, ZIP member
  names, path tokens, or any other source-name text as evidence.

Allowed:
  - paths for I/O, staging, copying, symlinks, manifests, and display
  - folder paths as training labels only in training modules
  - diagnostic/report tools that run after placement
  - tests that select fixtures by filename

This audit scans only production sorter modules, not tools/tests.

2026-05-15 hotfix:
  - Skip macOS AppleDouble/resource-fork files such as ._foo.py.
  - Skip hidden/cache files and non-text binary files before AST parsing.
  - Never crash on null bytes. Real source issues are reported as findings.
"""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

BANNED_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bsource_hint(s)?\b",
        r"\bsource[_-]?backed\b",
        r"\bname_hint(s)?\b",
        r"\bpath_hint(s)?\b",
        r"\bfile(name)?[_-]?hint(s)?\b",
        r"\bfolder[_-]?hint(s)?\b",
        r"\btrust_input_names\b",
        r"\bproducer[_-]?name\b",
        r"\bproducer[_-]?file\b",
        r"\boriginal_producer\b",
        r"\bsource_context_for_input_path\b",
        r"\bsource_hint_tokens_for_path\b",
        r"\bchoose_filename_text_for_source_hints\b",
        r"\bsource_backed_review_rescue\b",
        r"\bfilename_words?\b",
        r"\bfolder_words?\b",
        r"\bpath_words?\b",
        r"\btokens_for_path\b",
    ]
)

DEFAULT_TARGETS = [
    "Aaron_Sound_Sorter.py",
    "src/aaron_sound_sorter/engine",
    "src/aaron_sound_sorter/domain",
    "src/aaron_sound_sorter/voters",
    "src/aaron_sound_sorter/neural_audio",
    "src/aaron_sound_sorter/committee.py",
]

ALLOWED_IDENTIFIER_EXACT = {
    "input_path",
    "output_path",
    "source_path",
    "original_path",
    "folder_path",
    "new_relative_path",
    "relative_path",
    "rel_path",
    "path",
    "paths",
    "filename",
    "filenames",
    "file_name",
    "file_names",
    "dirname",
    "dirnames",
    "folder",
    "folders",
    "member_name",
    "member_names",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    kind: str
    text: str


def is_hidden_or_resource_fork(path: Path) -> bool:
    """Return True for macOS resource forks, hidden cache files, and pycache."""
    for part in path.parts:
        if part in {"__pycache__", "__MACOSX"}:
            return True
        if part.startswith("._"):
            return True
    name = path.name
    if name.startswith("._"):
        return True
    # Do not skip the project root dot folders here because DEFAULT_TARGETS
    # do not point inside them anyway. This protects recursive target scans.
    return False


def is_probably_binary(data: bytes) -> bool:
    """Conservative binary detector.

    Python source files must not contain null bytes. A real production source
    file with null bytes cannot be imported and should not be audited by AST.
    Resource-fork files are skipped earlier.
    """
    return b"\x00" in data


def relative_text(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_source_for_ast(path: Path) -> tuple[str | None, Finding | None]:
    try:
        data = path.read_bytes()
    except OSError as error:
        return None, Finding(path, 0, "read_error", str(error))

    if is_probably_binary(data):
        if is_hidden_or_resource_fork(path):
            return None, None
        # Real production .py with null bytes is broken source. Report it
        # cleanly instead of traceback.
        return None, Finding(path, 0, "binary_or_null_byte_source", "Python source contains null bytes")

    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        try:
            return data.decode("utf-8", errors="replace"), None
        except Exception as error:
            return None, Finding(path, 0, "decode_error", str(error))


def is_docstring_node(node: ast.AST, parent: ast.AST | None) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    value = getattr(node, "value", None)
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return False
    body = getattr(parent, "body", None)
    return bool(body and body[0] is node)


class SourceNameEvidenceVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []
        self._parents: list[ast.AST] = []

    def visit(self, node: ast.AST) -> None:  # type: ignore[override]
        parent = self._parents[-1] if self._parents else None
        if is_docstring_node(node, parent):
            return
        self._parents.append(node)
        try:
            super().visit(node)
        finally:
            self._parents.pop()

    def _check_text(self, text: str, line: int, kind: str) -> None:
        if text in ALLOWED_IDENTIFIER_EXACT:
            return
        for regex in BANNED_REGEXES:
            if regex.search(text):
                self.findings.append(Finding(self.path, line, kind, text))
                return

    def visit_Name(self, node: ast.Name) -> None:
        self._check_text(node.id, getattr(node, "lineno", 0), "identifier")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._check_text(node.attr, getattr(node, "lineno", 0), "attribute")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_text(node.name, getattr(node, "lineno", 0), "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_text(node.name, getattr(node, "lineno", 0), "async_function")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_text(node.arg, getattr(node, "lineno", 0), "argument")

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        self._check_text(node.value, getattr(node, "lineno", 0), "string")


def iter_python_files(project_root: Path, targets: Sequence[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for target in targets:
        path = project_root / target
        if not path.exists():
            continue

        if path.is_file():
            if path.suffix == ".py" and not is_hidden_or_resource_fork(path):
                real = path.resolve()
                if real not in seen:
                    seen.add(real)
                    yield path
            continue

        if path.is_dir():
            for file_path in sorted(path.rglob("*.py")):
                if is_hidden_or_resource_fork(file_path):
                    continue
                real = file_path.resolve()
                if real in seen:
                    continue
                seen.add(real)
                yield file_path


def audit_file(path: Path) -> list[Finding]:
    source, read_finding = read_source_for_ast(path)
    if read_finding:
        return [read_finding]
    if source is None:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except ValueError as error:
        return [Finding(path, 0, "parse_value_error", str(error))]
    except SyntaxError as error:
        return [Finding(path, int(error.lineno or 0), "syntax_error", str(error))]

    visitor = SourceNameEvidenceVisitor(path)
    visitor.visit(tree)
    return visitor.findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--target", action="append", default=[], help="Additional file/dir target to audit")
    parser.add_argument("--format", choices=["text", "csv"], default="text")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    targets = list(DEFAULT_TARGETS) + list(args.target)
    findings: list[Finding] = []
    scanned = 0

    for file_path in iter_python_files(project_root, targets):
        scanned += 1
        findings.extend(audit_file(file_path))

    if args.format == "csv":
        print("path,line,kind,text")
        for finding in findings:
            rel = relative_text(finding.path, project_root)
            safe_text = finding.text.replace('"', '""')
            print(f'"{rel}",{finding.line},"{finding.kind}","{safe_text}"')
    elif findings:
        print("NO-SOURCE-NAME SORTING AUDIT FAILED")
        for finding in findings:
            rel = relative_text(finding.path, project_root)
            print(f"{rel}:{finding.line}: {finding.kind}: {finding.text}")
    else:
        print("PASS: production sorting/voting/decision code contains no banned source-name evidence markers.")

    if args.verbose:
        print(f"Scanned Python files: {scanned}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
