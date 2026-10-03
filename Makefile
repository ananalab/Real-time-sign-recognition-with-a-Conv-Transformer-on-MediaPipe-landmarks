.PHONY: install lint format test analysis figures paper web clean

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

analysis:
	$(PY) scripts/stats_tests.py
	$(PY) scripts/analyse_lsfb_gap.py

figures:
	$(PY) scripts/make_figures.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp paper/main.pdf report.pdf

web:
	cd app/web && python3 -m http.server 8080

clean:
	rm -rf .pytest_cache .ruff_cache
