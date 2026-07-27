# Aaron Sound Sorter developer Makefile.
#
# Normal human workflow:
#   make bootstrap
#   make test
#   make gui
#
# AI coding workflow:
#   make ai-preflight
#   make ai-fix
#   make ai-check
#   make ai-bundle-check BUNDLE_DIR=/path/to/bundle_root
#
# Policy:
#   - Full legacy-repo quality reports are allowed to be red while debt exists.
#   - AI-touched files must pass the changed-files gate before a bundle is handed off.
#   - Generated caches and macOS metadata must never be packaged.

SHELL := /bin/bash
PROJECT_ROOT := $(CURDIR)
PYTHON ?= python3
VENV_DIR ?= .venv_phase4
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_PYTHON) -m pip
PYTHON_BYTECODE_CACHE ?= $(PROJECT_ROOT)/_reports/python_pycache
export PYTHONPYCACHEPREFIX := $(PYTHON_BYTECODE_CACHE)
PIP_INSTALL_FLAGS ?=
PYTEST_ARGS ?= -q
QUALITY_REPORT_ARGS ?=
BUNDLE_DIR ?=
BUNDLE_NAME ?= Aaron_Sound_Sorter_AI_patch
BUNDLE_OUTPUT_DIR ?= _reports/bundles
AI_BASE ?= HEAD

.PHONY: help bootstrap init gui venv upgrade-pip install install-all install-runtime install-quality install-dev \
        ensure-quality \
        doctor check-tools audit-source-names audit-public-assets sanitize-public-assets pycompile test test-one test-coverage test-coverage-html \
        lint format format-check type-check docstyle quality qa ci quality-report quality-report-full \
        quality-report-coverage quality-gate-report quality-baseline quality-strict \
        ai-preflight ai-changed-files ai-fix ai-check ai-test ai-bundle-check \
        bundle bundle-dry-run patch-bundle patch-bundle-dry-run clean-generated clean clean-dry-run clean-for-bundle clean-heavy-local \
        clean-heavy-local-dry-run organize-root-commands show-junk

help:
	@echo "Aaron Sound Sorter developer commands"
	@echo ""
	@echo "Setup:"
	@echo "  make bootstrap                 Create/update $(VENV_DIR) and install all runtime + dev + quality deps."
	@echo "  make init                      Alias for bootstrap."
	@echo "  make install                   Alias for install-dev."
	@echo "  make install-all               Alias for install-dev."
	@echo "  make install-runtime           Install runtime deps from requirements.txt."
	@echo "  make install-quality           Install pytest/coverage/ruff/mypy/pydocstyle deps."
	@echo "  make install-dev               Install full dev stack from requirements-dev.txt."
	@echo "  make ensure-quality            Fast import-only check that the quality deps are already installed."
	@echo "  make doctor                    Verify required developer tools from the venv."
	@echo "  make gui                       Launch the sorter GUI."
	@echo ""
	@echo "AI changed-files quality gate:"
	@echo "  make ai-preflight              Verify tools, show changed files, and fail on generated junk."
	@echo "  make ai-changed-files          Print files changed since AI_BASE=$(AI_BASE)."
	@echo "  make ai-fix                    Ruff-format and safe-fix changed Python files only."
	@echo "  make ai-check                  Strict gate for changed Python files plus source-name audit."
	@echo "  make ai-test TEST=path::node   Run one focused pytest target through the venv."
	@echo "  make ai-bundle-check BUNDLE_DIR=/path  Fail if bundle contains caches, pyc files, reports, or macOS junk."
	@echo "  make bundle                    Build code ZIP + balanced upload-safe end-user pack parts with commands/bundle/bundle.sh."
	@echo "  make bundle-dry-run            Preview code/runtime selections and multipart upload settings without writing ZIPs."
	@echo "  make patch-bundle BUNDLE_NAME=name   Build the legacy changed-files AI patch bundle."
	@echo "  make patch-bundle-dry-run             Preview the legacy changed-files patch bundle."
	@echo ""
	@echo "Whole-repo tests and quality:"
	@echo "  make audit-source-names        Run the source-name evidence audit."
	@echo "  make pycompile                 Compile src, tests, tools, and root runner files."
	@echo "  make test                      Run pytest. Optional: PYTEST_ARGS='tests/test_x.py -q'."
	@echo "  make test-one TEST=path::node  Run one pytest target."
	@echo "  make test-coverage             Run pytest with terminal and XML coverage."
	@echo "  make test-coverage-html        Run pytest with HTML coverage under _reports/quality/htmlcov."
	@echo "  make lint                      Run Ruff lint checks on the whole repo. Expected red until legacy debt is paid."
	@echo "  make format                    Apply Ruff formatting to src, tests, and tools. Do not run casually on legacy code."
	@echo "  make format-check              Check whole-repo formatting. Expected red until legacy debt is paid."
	@echo "  make type-check                Run Mypy on src/aaron_sound_sorter. Expected red until legacy debt is paid."
	@echo "  make docstyle                  Run pydocstyle on src/aaron_sound_sorter. Expected red until legacy debt is paid."
	@echo "  make quality                   Whole-repo strict quality. Future target, not the normal AI gate yet."
	@echo "  make qa                        Alias for quality."
	@echo "  make ci                        Fresh dependency install plus quality and coverage gates."
	@echo ""
	@echo "Reports:"
	@echo "  make quality-report            Write _reports/quality/pro_quality_report_*.md without pretending legacy debt is clean."
	@echo "  make quality-report-full       Include pytest in the quality report."
	@echo "  make quality-report-coverage   Include pytest coverage in the quality report."
	@echo "  make quality-gate-report       Write report and fail when a required check fails."
	@echo "  make quality-baseline          Alias for quality-report."
	@echo "  make quality-strict            Alias for quality."
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-generated           Remove Python caches, pytest/mypy/ruff caches, coverage files, .DS_Store, and ._* files."
	@echo "  make clean                     Run cleanup_project.py standard cleanup when available, then clean-generated."
	@echo "  make clean-dry-run             Show what cleanup_project.py would remove."
	@echo "  make clean-for-bundle          Remove generated junk before AI handoff bundles."
	@echo "  make show-junk                 Print generated/junk candidates without deleting."
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  VENV_DIR=$(VENV_DIR)"
	@echo "  AI_BASE=$(AI_BASE)"
	@echo "  PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX)"
	@echo "  PIP_INSTALL_FLAGS=$(PIP_INSTALL_FLAGS)"
	@echo "  BUNDLE_NAME=$(BUNDLE_NAME)"
	@echo "  BUNDLE_OUTPUT_DIR=$(BUNDLE_OUTPUT_DIR)"

bootstrap: install-dev doctor

init: bootstrap

install: install-dev

install-all: install-dev

gui:
	./commands/gui/RUN_SORTER_GUI.command

venv:
	@test -x "$(VENV_PYTHON)" || "$(PYTHON)" -m venv "$(VENV_DIR)"

upgrade-pip: venv
	"$(VENV_PYTHON)" -m pip install --upgrade pip setuptools wheel

install-runtime: upgrade-pip
	$(VENV_PIP) install $(PIP_INSTALL_FLAGS) -r requirements.txt

install-quality: upgrade-pip
	$(VENV_PIP) install $(PIP_INSTALL_FLAGS) -r requirements-quality.txt

install-dev: upgrade-pip
	$(VENV_PIP) install $(PIP_INSTALL_FLAGS) -r requirements-dev.txt

ensure-quality: venv
	"$(VENV_PYTHON)" tools/check_dev_environment.py --project-root "$(PROJECT_ROOT)" --quick --quiet

check-tools: doctor

doctor:
	"$(VENV_PYTHON)" tools/check_dev_environment.py --project-root "$(PROJECT_ROOT)"

audit-source-names:
	"$(VENV_PYTHON)" tools/audit_no_source_name_sorting.py --project-root "$(PROJECT_ROOT)"

audit-public-assets:
	"$(VENV_PYTHON)" tools/audit_public_runtime_assets.py --project-root "$(PROJECT_ROOT)"

sanitize-public-assets:
	"$(VENV_PYTHON)" tools/audit_public_runtime_assets.py --project-root "$(PROJECT_ROOT)" --apply

pycompile:
	"$(VENV_PYTHON)" -m compileall -q src tests tools Aaron_Sound_Sorter.py

test: ensure-quality
	"$(VENV_PYTHON)" -m pytest $(PYTEST_ARGS)

test-one: ensure-quality
	@test -n "$(TEST)" || (echo "Usage: make test-one TEST='tests/path.py::test_name'" && exit 2)
	"$(VENV_PYTHON)" -m pytest -q "$(TEST)"

test-coverage: ensure-quality
	mkdir -p _reports/quality
	"$(VENV_PYTHON)" -m pytest --cov=src/aaron_sound_sorter --cov-report=term-missing --cov-report=xml:_reports/quality/coverage.xml $(PYTEST_ARGS)

test-coverage-html: ensure-quality
	mkdir -p _reports/quality
	"$(VENV_PYTHON)" -m pytest --cov=src/aaron_sound_sorter --cov-report=term-missing --cov-report=html:_reports/quality/htmlcov $(PYTEST_ARGS)

lint: ensure-quality
	"$(VENV_PYTHON)" -m ruff check src tests tools

format: ensure-quality
	"$(VENV_PYTHON)" -m ruff format src tests tools

format-check: ensure-quality
	"$(VENV_PYTHON)" -m ruff format --check src tests tools

type-check: ensure-quality
	"$(VENV_PYTHON)" -m mypy src/aaron_sound_sorter

docstyle: ensure-quality
	"$(VENV_PYTHON)" -m pydocstyle src/aaron_sound_sorter

quality: ensure-quality audit-source-names pycompile lint format-check type-check docstyle test

qa: quality

ci: bootstrap quality test-coverage

quality-report: clean-generated ensure-quality
	"$(VENV_PYTHON)" tools/pro_quality_report.py --project-root "$(PROJECT_ROOT)" --report-only $(QUALITY_REPORT_ARGS)

quality-report-full: clean-generated ensure-quality
	"$(VENV_PYTHON)" tools/pro_quality_report.py --project-root "$(PROJECT_ROOT)" --with-pytest --report-only $(QUALITY_REPORT_ARGS)

quality-report-coverage: clean-generated ensure-quality
	"$(VENV_PYTHON)" tools/pro_quality_report.py --project-root "$(PROJECT_ROOT)" --with-pytest --with-coverage --report-only $(QUALITY_REPORT_ARGS)

quality-gate-report: clean-generated ensure-quality
	"$(VENV_PYTHON)" tools/pro_quality_report.py --project-root "$(PROJECT_ROOT)" $(QUALITY_REPORT_ARGS)

quality-baseline: quality-report

quality-strict: quality

ai-preflight: clean-generated ensure-quality
	"$(VENV_PYTHON)" tools/ai_quality_gate.py --project-root "$(PROJECT_ROOT)" --base "$(AI_BASE)" preflight

ai-changed-files:
	"$(VENV_PYTHON)" tools/ai_quality_gate.py --project-root "$(PROJECT_ROOT)" --base "$(AI_BASE)" list

ai-fix: clean-generated ensure-quality
	"$(VENV_PYTHON)" tools/ai_quality_gate.py --project-root "$(PROJECT_ROOT)" --base "$(AI_BASE)" fix

ai-check: clean-generated ensure-quality
	"$(VENV_PYTHON)" tools/ai_quality_gate.py --project-root "$(PROJECT_ROOT)" --base "$(AI_BASE)" check
	"$(VENV_PYTHON)" tools/audit_public_runtime_assets.py --project-root "$(PROJECT_ROOT)"

ai-test: test-one

ai-bundle-check:
	@test -n "$(BUNDLE_DIR)" || (echo "Usage: make ai-bundle-check BUNDLE_DIR=/path/to/bundle_root" && exit 2)
	"$(VENV_PYTHON)" tools/ai_quality_gate.py --project-root "$(PROJECT_ROOT)" bundle-check --bundle-dir "$(BUNDLE_DIR)"

bundle:
	PROJECT_ROOT="$(PROJECT_ROOT)" \
	PYTHON_BIN="$(PYTHON)" \
	BUNDLE_OUTPUT_DIR="$(BUNDLE_OUTPUT_DIR)" \
	./commands/bundle/bundle.sh

bundle-dry-run:
	PROJECT_ROOT="$(PROJECT_ROOT)" \
	PYTHON_BIN="$(PYTHON)" \
	BUNDLE_OUTPUT_DIR="$(BUNDLE_OUTPUT_DIR)" \
	DRY_RUN=1 \
	./commands/bundle/bundle.sh

patch-bundle: clean-for-bundle ai-check
	PROJECT_ROOT="$(PROJECT_ROOT)" \
	PYTHON_BIN="$(VENV_PYTHON)" \
	AI_BASE="$(AI_BASE)" \
	BUNDLE_NAME="$(BUNDLE_NAME)" \
	BUNDLE_OUTPUT_DIR="$(BUNDLE_OUTPUT_DIR)" \
	./commands/build/BUILD_CLEAN_PATCH_BUNDLE.command

patch-bundle-dry-run: clean-for-bundle
	PROJECT_ROOT="$(PROJECT_ROOT)" \
	PYTHON_BIN="$(VENV_PYTHON)" \
	AI_BASE="$(AI_BASE)" \
	BUNDLE_NAME="$(BUNDLE_NAME)" \
	BUNDLE_OUTPUT_DIR="$(BUNDLE_OUTPUT_DIR)" \
	./commands/build/BUILD_CLEAN_PATCH_BUNDLE.command --dry-run

clean-generated:
	find . -name '._*' -type f -delete
	find . -name '.DS_Store' -type f -delete
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
	find . -name '.mypy_cache' -type d -prune -exec rm -rf {} +
	find . -name '.ruff_cache' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -type f -delete
	rm -rf "$(PYTHON_BYTECODE_CACHE)"
	rm -f .coverage

clean: clean-generated
	@if test -f tools/cleanup_project.py; then "$(VENV_PYTHON)" tools/cleanup_project.py --project-root "$(PROJECT_ROOT)" --mode standard --apply; fi
	rm -rf "$(PYTHON_BYTECODE_CACHE)"

clean-dry-run:
	@if test -f tools/cleanup_project.py; then "$(VENV_PYTHON)" tools/cleanup_project.py --project-root "$(PROJECT_ROOT)" --mode standard; else echo "tools/cleanup_project.py not present"; fi

clean-for-bundle: clean-generated
	@if test -f tools/cleanup_project.py; then "$(VENV_PYTHON)" tools/cleanup_project.py --project-root "$(PROJECT_ROOT)" --mode bundle --apply; fi
	rm -rf "$(PYTHON_BYTECODE_CACHE)"

show-junk:
	find . \( -name '._*' -o -name '.DS_Store' -o -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' -o -name '*.pyc' -o -name '.coverage' \) -print

clean-heavy-local:
	@if test -f tools/cleanup_project.py; then "$(VENV_PYTHON)" tools/cleanup_project.py --project-root "$(PROJECT_ROOT)" --mode heavy-local --apply; fi

clean-heavy-local-dry-run:
	@if test -f tools/cleanup_project.py; then "$(VENV_PYTHON)" tools/cleanup_project.py --project-root "$(PROJECT_ROOT)" --mode heavy-local; fi

organize-root-commands:
	./commands/maintenance/REORGANIZE_ROOT_COMMANDS.command
