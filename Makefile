.PHONY: setup lint format typecheck test test-all check clean

# First-time setup
setup:
	uv sync --dev
	uv run pre-commit install

# Lint (auto-fix)
lint:
	uv run ruff check --fix src/ tests/

# Format
format:
	uv run ruff format src/ tests/

# Type check
typecheck:
	uv run mypy src/

# Fast tests (no slow/integration)
test:
	uv run pytest -m "not slow and not integration"

# All tests including slow and integration
test-all:
	uv run pytest

# Full validation (what pre-commit runs)
check: lint format typecheck test

# Clean caches
clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
