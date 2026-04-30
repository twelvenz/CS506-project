# Super Bowl halftime scoring / EDA pipeline
# Override if you use a venv: make PYTHON=./venv/bin/python run-predict

PYTHON ?= python3

.PHONY: help install build \
	eda-billboard eda-spotify \
	build-table run-predict run-validate

help:
	@echo "Targets:"
	@echo "  make install       - pip install pandas numpy matplotlib seaborn"
	@echo "  make eda-billboard - python billboard_EDA.py -> complete_training_table.csv"
	@echo "  make eda-spotify   - python spotify_EDA.py (plots + enriched CSV)"
	@echo "  make build-table   - python build_training_table.py (see script; outputs commented)"
	@echo "  make run-predict   - python sb_prediction_model.py -> sb_lxi_summary.txt"
	@echo "  make run-validate  - python sb_scoring_validation.py (needs training_table.csv)"
	@echo "Set PYTHON=./venv/bin/python to use your project venv."

install:
	$(PYTHON) -m pip install pandas numpy matplotlib seaborn

build: install

eda-billboard:
	$(PYTHON) billboard_EDA.py

eda-spotify:
	$(PYTHON) spotify_EDA.py

build-table:
	$(PYTHON) build_training_table.py

run-predict:
	$(PYTHON) sb_prediction_model.py

run-validate:
	$(PYTHON) sb_scoring_validation.py
