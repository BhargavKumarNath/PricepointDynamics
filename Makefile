.PHONY: setup test lint format typecheck benchmark benchmark-phase3 marts

setup:
	uv sync --all-extras

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src/pricepoint

benchmark:
	uv run python scripts/benchmark_phase2.py

benchmark-phase3:
	uv run python scripts/benchmark_phase3.py

marts:
	uv run python scripts/materialize_marts.py
