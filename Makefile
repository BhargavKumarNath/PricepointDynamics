.PHONY: setup test lint format typecheck

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
