# Tutorial notebooks: run them into executed copies (see README.md)
#
#   make          run every notebook that is newer than its executed copy, JOBS notebooks at a time
#   make clean    delete the executed copies and clear the notebooks in place
#
# PYTHON must be the Python of the environment with quadriga-lib and jupyter, e.g. the active conda environment.
# JOBS sets the number of notebooks run in parallel (JOBS=1: one after another).

PYTHON ?= python3
JOBS   ?= 4
MAKEFLAGS += --jobs=$(JOBS)

NOTEBOOKS := $(wildcard *.ipynb)
EXECUTED  := $(addprefix executed/,$(NOTEBOOKS))

.PHONY: all run clean
all run: $(EXECUTED)

# Each executed copy depends only on its notebook; delete it (or run make clean) to force a new run
executed/%.ipynb: %.ipynb
	$(PYTHON) -m qnb.run $< $@

clean:
	rm -rf executed
	$(PYTHON) -m qnb.clear $(NOTEBOOKS)
