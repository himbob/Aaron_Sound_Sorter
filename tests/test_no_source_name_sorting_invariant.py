"""Hard invariant: production sorting code must be blind to source names.

File names, folder names, ZIP member names, path tokens, producer labels, and
sample-pack labels may be used for diagnostics after placement. They must never
be used as evidence by voters, roles, eligibility, consensus, or final placement.
"""

from __future__ import annotations

from pathlib import Path

from tools.audit_no_source_name_sorting import DEFAULT_TARGETS, audit_file, iter_python_files


def test_production_sorting_code_has_no_source_name_evidence_markers() -> None:
    project_root = Path(__file__).resolve().parents[1]
    findings = []
    for file_path in iter_python_files(project_root, DEFAULT_TARGETS):
        findings.extend(audit_file(file_path))
    assert findings == []
