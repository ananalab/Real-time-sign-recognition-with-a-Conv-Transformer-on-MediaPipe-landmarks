.PHONY: install lint format test figures web clean

PY := uv run python

install:
	uv venv --python 3.12 --allow-existing
	uv pip install -e ".[data,dev]"
	uv run playwright install chromium

lint:
	uv run ruff check src tests scripts app
	uv run ruff format --check src tests scripts app

format:
	uv run ruff format src tests scripts app
	uv run ruff check --fix src tests scripts app

test:
	uv run pytest

figures:
	$(PY) scripts/make_figures.py

web:
	cd app/web && python3 -m http.server 8080

clean:
	rm -rf .pytest_cache .ruff_cache
