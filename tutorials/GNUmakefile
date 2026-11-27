# Tutorial notebooks: run them into executed copies (see README.md)
#
#   make                  run every notebook that is newer than its executed copy, JOBS notebooks at a time
#   make python           only the Python notebooks (*.ipynb)
#   make matlab           only the MATLAB variants (matlab/*.ipynb), run with the Octave kernel
#   make executed/<stem>.ipynb, make executed/matlab/<stem>.ipynb   run one notebook
#   make check-matlab     run the code of the MATLAB variants in MATLAB, keeping no output
#   make clean            delete the executed copies and clear the notebooks in place
#
# PYTHON must be the Python of the environment with quadriga-lib, jupyter and octave_kernel, e.g. the active conda
# environment. JOBS sets the number of notebooks run in parallel (JOBS=1: one after another).
# OCTAVE_PATH and MATLABPATH name the folder with +quadriga_lib; by default the repository root, where the build
# installs the MEX files. Set them to another folder (e.g. /usr/share/quadriga-lib) to use an installed library.

PYTHON      ?= python3
JOBS        ?= 4
MATLAB_BIN  ?= matlab
OCTAVE_PATH ?= $(abspath ..)
MATLABPATH  ?= $(abspath ..)
export OCTAVE_PATH MATLABPATH
MAKEFLAGS += --jobs=$(JOBS) --keep-going  # a failing notebook does not stop the others

NOTEBOOKS := $(wildcard *.ipynb)
MATLAB_NOTEBOOKS := $(wildcard matlab/*.ipynb)
EXECUTED  := $(addprefix executed/,$(NOTEBOOKS))
MATLAB_EXECUTED := $(addprefix executed/,$(MATLAB_NOTEBOOKS))
MATLAB_CHECKS := $(patsubst matlab/%.ipynb,executed/check-matlab/%/done,$(MATLAB_NOTEBOOKS))

.PHONY: all run python matlab check-matlab clean
all run: python matlab
python: $(EXECUTED)
matlab: $(MATLAB_EXECUTED)
check-matlab: $(MATLAB_CHECKS)

# Each executed copy depends only on its notebook; delete it (or run make clean) to force a new run
executed/%.ipynb: %.ipynb
	$(PYTHON) -m qnb.run $< $@

# One folder per notebook; the script is called by name from matlab/, so that it runs there, as the notebook
# does in its kernel (run() would change into the folder of the script). A run that hangs (exit 124 after
# 3 minutes) can be repeated with the same command.
executed/check-matlab/%/done: matlab/%.ipynb
	@mkdir -p $(@D)
	$(PYTHON) -m qnb.to_m $< $(@D)/tutorial_check.m
	cd matlab && timeout 180 $(MATLAB_BIN) -batch "addpath('$(abspath $(@D))'); tutorial_check"
	@touch $@

clean:
	rm -rf executed
	$(PYTHON) -m qnb.clear $(NOTEBOOKS) $(MATLAB_NOTEBOOKS)
