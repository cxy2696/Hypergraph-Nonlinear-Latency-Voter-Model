PYTHON ?= python

.PHONY: help install test reproduce quick diagnose longrun clean

help:
	@echo "make install    install runtime dependencies"
	@echo "make test       run the test suite (47 tests)"
	@echo "make reproduce  full experiments, tables and figures (~30 s)"
	@echo "make quick      quick smoke reproduction (5 replicates)"
	@echo "make diagnose   structural report for the shipped dataset"
	@echo "make longrun    supplementary long-horizon edge-sweep check"
	@echo "make clean      remove caches (keeps results/ and figures/)"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

reproduce:
	$(PYTHON) -m hnl.cli reproduce --root .

quick:
	$(PYTHON) -m hnl.cli reproduce --root . --quick

diagnose:
	$(PYTHON) -m hnl.cli diagnose --csv sentimentdataset.csv --scope full \
		--out results/diagnostics.md

longrun:
	$(PYTHON) scripts/edge_sweep_longrun.py --out results/edge_sweep_longrun.csv

clean:
	rm -rf __pycache__ .pytest_cache hnl/__pycache__ tests/__pycache__ \
		scripts/__pycache__ *.egg-info build dist
