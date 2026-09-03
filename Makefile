.PHONY: install test lint format format-check typecheck check verify clean

install:
	uv sync --extra dev --extra modal

test:
	uv run --extra dev python -m pytest -q

lint:
	uv run --extra dev ruff check src tests scripts

format:
	uv run --extra dev ruff format src tests scripts

format-check:
	uv run --extra dev ruff format --check src tests scripts

typecheck:
	uv run --extra modal mypy src scripts

check: lint format-check typecheck test

verify: check

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist *.egg-info
	find . -path './.git' -prune -o -path './.venv' -prune -o -name '__pycache__' -type d -exec rm -rf {} +
