# CS506 Super Bowl ranking — install and run helpers.
# From repo root: make install && make run

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install run run-eda run-viz build

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) xgboost_model.py

run-eda:
	$(PYTHON) eda_dataset.py

run-viz:
	$(PYTHON) xgboost_visuals.py

build: install
