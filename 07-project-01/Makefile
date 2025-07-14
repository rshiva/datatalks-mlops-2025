
# Makefile for the Personality Prediction Project

.PHONY: install format lint train predict all

# =============================================================================
# SETUP
# =============================================================================

install:
	@echo "Installing dependencies from requirements.txt..."
	pip install -r requirements.txt

# =============================================================================
# CODE QUALITY
# =============================================================================

format:
	@echo "Formatting code with black..."
	python -m black .

lint:
	@echo "Linting code with ruff..."
	python -m ruff . --fix

# =============================================================================
# PIPELINES
# =============================================================================

train:
	@echo "Running the training pipeline..."
	python pipelines/train.py

predict:
	@echo "Running the batch prediction pipeline..."
	python pipelines/batch_predict.py

# =============================================================================
# ALL-IN-ONE
# =============================================================================

all: format lint train
	@echo "Ran all steps: format, lint, train."


test:
	@echo "Running tests..."
	python -m pytest tests/

mage:
	@echo "Starting Mage UI..."
	cd mage_project && python -m mage start .
