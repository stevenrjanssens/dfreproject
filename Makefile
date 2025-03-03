# Variables
SRC_DIR=src
PYTHON=python3
PIP=$(PYTHON) -m pip

# Ensure dependencies are installed
dependencies:
	@echo "Checking and installing dependencies..."
	$(PIP) install flake8 black isort

.PHONY: all lint format clean check dependencies

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
	$(PYTHON) -m isort --check-only $(SRC_DIR)