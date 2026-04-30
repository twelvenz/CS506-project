PYTHON = venv/bin/python
PIP    = venv/bin/pip

# Windows compatibility
ifeq ($(OS),Windows_NT)
	PYTHON = venv/Scripts/python
	PIP    = venv/Scripts/pip
endif

.PHONY: all install data predict validate test clean help

## Default target: run the full pipeline
all: install data predict validate

## Install dependencies into a virtual environment
install:
	@echo "Creating virtual environment..."
	python -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Done. Activate with: source venv/bin/activate"

## Stage 2: Build training table from raw Billboard + Spotify data
data:
	@echo "Building training table..."
	$(PYTHON) build_training_table.py
	@echo "Done: training_table.csv"

## Stage 3: Run scoring model and generate SB LXI predictions
predict:
	@echo "Running scoring model..."
	$(PYTHON) sb_prediction_model.py
	@echo "Done: sb_lxi_scores.csv, sb_lxi_summary.txt"

## Retroactive validation (2020-2025 folds)
validate:
	@echo "Running retroactive validation..."
	$(PYTHON) sb_scoring_validation.py
	@echo "Done: sb_validation_results.csv, sb_validation_summary.png"

## Run EDA scripts (optional -- produces charts only)
eda:
	@echo "Running Billboard EDA..."
	$(PYTHON) billboard_EDA.py
	@echo "Running Spotify EDA..."
	$(PYTHON) spotify_EDA.py

## Run the test suite
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v

## Remove all generated output files
clean:
	@echo "Cleaning generated files..."
	rm -f training_table.csv
	rm -f sb_lxi_scores.csv sb_lxi_summary.txt
	rm -f sb_validation_results.csv
	rm -f model_cv_results.csv model_feature_importance.csv
	rm -f *.png
	rm -f join_diagnostics.csv artist_billboard_profiles.csv
	rm -f prediction_candidates_2026.csv
	@echo "Done."

## Show available targets
help:
	@echo ""
	@echo "  make install   -- create venv and install dependencies"
	@echo "  make data      -- build training_table.csv"
	@echo "  make predict   -- run scoring model -> sb_lxi_scores.csv"
	@echo "  make validate  -- run retroactive validation"
	@echo "  make eda       -- run EDA scripts (optional)"
	@echo "  make test      -- run test suite"
	@echo "  make all       -- install + data + predict + validate"
	@echo "  make clean     -- remove all generated files"
	@echo ""
