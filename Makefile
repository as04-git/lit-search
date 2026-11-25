.PHONY: help clean setup install-cli test test-integration lint format types complexity security deps-audit docstrings dead-code ascii check
.DEFAULT_GOAL := help

VENV ?= .venv
RUN = uv run
EXTRAS ?=
PYTEST_ARGS ?=
TARGET ?= .
FIX ?=

-include .env
export $(shell [ -f .env ] && grep -v '^\s*#' .env | grep -v '^\s*$$' | sed 's/=.*//')

help:
	@echo "make clean"
	@echo "       clean project removing unnecessary files"
	@echo "make setup [EXTRAS=rerank]"
	@echo "       create .venv with uv (dev tools included); EXTRAS=rerank adds the torch re-ranker"
	@echo "make install-cli"
	@echo "       link lit-search onto PATH (~/.local/bin)"
	@echo "make lint [TARGET='path'] [FIX=1]"
	@echo "       check for lint issues with ruff; pass FIX=1 to auto-fix (optional: specify target path)"
	@echo "make format [TARGET='path'] [FIX=1]"
	@echo "       check code formatting with ruff; pass FIX=1 to auto-fix (optional: specify target path)"
	@echo "make types [TARGET='path']"
	@echo "       run static type checks with mypy (optional: specify target path)"
	@echo "       note: type errors are NOT auto-fixed"
	@echo "make test [PYTEST_ARGS='args']"
	@echo "       run tests with coverage report (optional: pass additional pytest arguments)"
	@echo "       examples: make test PYTEST_ARGS='-k test_name'"
	@echo "                 make test PYTEST_ARGS='tests/unit/test_query.py::TestClass -v'"
	@echo "make test-integration [PYTEST_ARGS='args']"
	@echo "       run integration/smoke tests that hit real external APIs"
	@echo "make complexity"
	@echo "       check cyclomatic complexity with xenon (fails if any block exceeds grade C)"
	@echo "make security"
	@echo "       run static security analysis with bandit"
	@echo "make deps-audit"
	@echo "       check dependencies for known vulnerabilities with pip-audit"
	@echo "make docstrings"
	@echo "       check docstring coverage with interrogate (fails if below 95%)"
	@echo "make dead-code"
	@echo "       detect unused code with vulture"
	@echo "make ascii"
	@echo "       reject non-ASCII characters in lit_search/ and tests/ (e.g. em dash, arrows, emoji)"
	@echo "make check"
	@echo "       run all quality checks: lint, format, types, complexity,"
	@echo "       security, deps-audit, docstrings, dead-code, ascii and test"

setup:
	@command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	@uv sync $(if $(EXTRAS),--extra $(EXTRAS))

install-cli: ## Link lit-search onto PATH (~/.local/bin); credentials: lit-search config / doctor
	@mkdir -p $(HOME)/.local/bin
	@ln -sf $(CURDIR)/$(VENV)/bin/lit-search $(HOME)/.local/bin/lit-search
	@echo "linked $(HOME)/.local/bin/lit-search; next: lit-search config import <keys file>; lit-search doctor"

clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage reports htmlcov .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

test:
	@$(RUN) pytest --durations=3 -v --cov=$(PWD)/lit_search --cov-report=term-missing $(PYTEST_ARGS)

test-integration:
	@$(RUN) pytest -v -m integration $(PYTEST_ARGS)

lint:
	@$(RUN) ruff check $(if $(FIX),--fix) $(TARGET)

format:
	@$(RUN) ruff format $(if $(FIX),,--check) $(TARGET)

types:
	@if [ "$(TARGET)" = "." ]; then \
		MYPYPATH=typings $(RUN) mypy lit_search tests/unit; \
	else \
		MYPYPATH=typings $(RUN) mypy $(TARGET); \
	fi

complexity:
	@$(RUN) xenon --max-absolute C --max-average B lit_search/

security:
	@$(RUN) bandit -r lit_search/ -c pyproject.toml

# PYSEC-2022-42969 (py): no fix available upstream; py is a transitive dep of
# interrogate and the vulnerable code path (py.path.svn) is never exercised here.
deps-audit:
	@$(RUN) pip-audit --ignore-vuln PYSEC-2022-42969

docstrings:
	@$(RUN) interrogate lit_search/

dead-code:
	@$(RUN) vulture lit_search/

ascii:
	@$(RUN) python tools/check_ascii.py

check:
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory format
	@$(MAKE) --no-print-directory types
	@$(MAKE) --no-print-directory complexity
	@$(MAKE) --no-print-directory security
	@$(MAKE) --no-print-directory deps-audit
	@$(MAKE) --no-print-directory docstrings
	@$(MAKE) --no-print-directory dead-code
	@$(MAKE) --no-print-directory ascii
	@$(MAKE) --no-print-directory test
