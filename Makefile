# Variables
SRC_DIR=src
PYTHON=python3
PIP=$(PYTHON) -m pip
PYTEST=$(PYTHON) -m pytest

# Ensure dependencies are installed
dependencies:
	@echo "Checking and installing dependencies..."
	$(PIP) install flake8 black isort

# Test dependencies
test-dependencies:
	@echo "Checking and installing test dependencies..."
	$(PIP) install pytest pytest-cov pytest-xdist

.PHONY: all lint format clean check dependencies test-dependencies test test-unit test-integration test-all test-cov clean-tests

all: lint check

lint: dependencies
	@echo "Running lint checks..."
	$(PYTHON) -m flake8 --ignore=E501 $(SRC_DIR)

format: dependencies
	@echo "Formatting code..."
	$(PYTHON) -m black $(SRC_DIR)
	$(PYTHON) -m isort $(SRC_DIR)

clean: lint format

check: dependencies
	@echo "Checking code format..."
	$(PYTHON) -m black --check $(SRC_DIR)


# Testing commands
test-all: test-dependencies
	@echo "Running all tests..."
	$(PYTEST)

test-unit: test-dependencies
	@echo "Running unit tests..."
	$(PYTEST) -m "unit"

test-integration: test-dependencies
	@echo "Running integration tests..."
	$(PYTEST) -m "integration"

test-cov: test-dependencies
	@echo "Running tests with coverage..."
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=term --cov-report=html

# Default test command
test: test-all

# Clean pytest cache
clean-tests:
	@echo "Cleaning pytest cache..."
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

# Full clean command
clean-all: clean clean-tests
	@echo "Cleaning all generated files..."