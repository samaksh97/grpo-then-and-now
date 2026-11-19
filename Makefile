# Common tasks. `make setup` once, then `make test` before committing.
PY ?= .venv/bin/python

.PHONY: setup figures handouts poster notebooks test test-fast decks zip

setup:            ## create .venv and install pinned dependencies plus test tooling
	python3 -m venv .venv
	$(PY) -m pip install -r requirements-dev.txt

FIGSET ?= figures/GRPO Figure Set-png.zip

figures:          ## re-import an exported figure set (make figures FIGSET="path/to/export.zip"), redraw 00a/00b
	$(PY) figures/import_figure_set.py "$(FIGSET)"
	$(PY) figures/generate_primer_diagrams.py
	$(PY) tools/embed_figures.py

handouts:         ## PDF handouts of the decks into slides/handouts (needs Chrome, Chromium or Edge)
	python3 slides/build_handouts.py

poster:           ## the 61 x 85 cm poster PDF and the neurips.cc PNGs into poster/ (needs Chrome, Chromium or Edge)
	$(PY) poster/build_poster.py

notebooks:        ## execute every notebook in place so saved outputs are current
	$(PY) tools/run_notebooks.py

test-fast:        ## unit tests only
	$(PY) -m pytest -q tests/test_objectives.py

test:             ## unit tests and a headless run of every notebook
	$(PY) -m pytest -q

decks:            ## self-contained decks (KaTeX inlined) into build/slides
	$(PY) slides/build_standalone.py build/slides

zip:              ## submission ZIP (needs an up-to-date concept_summary.pdf)
	./make_submission_zip.sh
