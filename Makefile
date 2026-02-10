# Makefile for QSAR AutoML Project
# =================================

.PHONY: help install install-dev test test-fast test-cov test-scenarios test-integration lint format clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install package dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test           - Run all tests"
	@echo "  make test-fast      - Run tests without slow tests"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-scenarios - Run scenario tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with black and isort"
	@echo "  make clean          - Clean up generated files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-test.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-cov:
	pytest tests/ -v --cov=automl_qsar --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-scenarios:
	pytest tests/scenarios/ -v

test-integration:
	pytest tests/integration/ -v

test-data-curation:
	pytest test_data_curation.py -v

# Code quality
lint:
	flake8 automl_qsar/ tests/ --max-line-length=100

format:
	black automl_qsar/ tests/
	isort automl_qsar/ tests/

# Cleanup
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
