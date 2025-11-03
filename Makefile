PY := python3
PIP := $(PY) -m pip
PYTHONPATH := $(PWD)

.PHONY: help install test eval lint format type clean

help:
	@echo "Targets:"
	@echo "  install  - Install package + dev deps"
	@echo "  test     - Run unit tests"
	@echo "  eval     - Run retrieval eval script"
	@echo "  lint     - Ruff check + mypy type-check"
	@echo "  format   - Ruff format"
	@echo "  type     - mypy only"
	@echo "  clean    - Remove caches and build artifacts"

install:
	$(PIP) install -e .[dev]

test:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m unittest -v

eval:
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/eval_retrieval.py

lint:
	ruff check sdk scripts tests
	mypy sdk scripts

format:
	ruff format sdk scripts tests

type:
	mypy sdk scripts

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache build dist
