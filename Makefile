.PHONY: all install lint format test smoke-test train export-onnx benchmark clean help

CONDA_ENV_PY = /home/bishal-shrestha/miniconda3/envs/nlp_core/bin/python
PYTHON ?= $(shell if [ -f $(CONDA_ENV_PY) ]; then echo $(CONDA_ENV_PY); else which python3; fi)
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

help:
	@echo "RouteOps MLOps Gateway Build & Automation"
	@echo "=========================================="
	@echo "install      : Install Python dependencies from requirements.txt"
	@echo "lint         : Run flake8, isort check, black check, and mypy"
	@echo "format       : Format codebase using black and isort"
	@echo "test         : Run pytest test suite with coverage report"
	@echo "smoke-test   : Run RTX 2050 4GB VRAM safety verification test"
	@echo "train        : Fine-tune multilingual routing model under 3.0GB VRAM"
	@echo "export-onnx  : Export model to ONNX & dynamic Int8 quantization"
	@echo "benchmark    : Run latency (P95) and throughput benchmarking"
	@echo "clean        : Remove caches, build artifacts, and temporary files"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src tests --max-line-length=100 --extend-ignore=E203,W503 || true
	$(PYTHON) -m isort --check-only --profile black src tests || true
	$(PYTHON) -m black --check src tests || true
	$(PYTHON) -m mypy src --ignore-missing-imports || true

format:
	$(PYTHON) -m isort --profile black src tests
	$(PYTHON) -m black src tests

test:
	$(PYTEST) tests -v --cov=src --cov-report=term-missing

smoke-test:
	$(PYTHON) -m src.smoke_test_4gb

train:
	$(PYTHON) -m src.train --config configs/config.yaml

export-onnx:
	$(PYTHON) -m src.export_onnx --config configs/config.yaml

benchmark:
	$(PYTHON) -m tests.test_smoke --benchmark

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf models/checkpoints
