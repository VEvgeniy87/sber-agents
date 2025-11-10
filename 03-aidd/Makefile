.PHONY: install run dev test clean

install:
	uv sync

run:
	uv run python -m src.bot

dev:
	uv run python -m src.bot

test:
	uv run python -m pytest tests/ -v

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache