PYTHON ?= python

.PHONY: all figures tables pdf release clean

all: figures tables pdf

figures:
	$(PYTHON) make_figures.py

tables:
	$(PYTHON) make_tables.py

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

release: pdf
	cp main.pdf HNL_Voter_Model_v2.pdf

clean:
	latexmk -C
	rm -f main.pdf
