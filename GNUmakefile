# This Makefile is for Linux / GCC environments

# bash with pipefail, so a recipe that pipes into tee (the deb builds) fails when the command before the pipe fails
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

# Set Armadillo sources
arma_internal = OFF
static_lib = ON
shared_lib = OFF
octave = ON
matlab = ON
python = ON
avx2 = ON
cuda = OFF

# GCC version override. Empty for system default. Note: NVCC max is 13, MATLAB max is 12
gcc_version = 12

CMAKE_BUILD_DIR = build_linux
ASAN_BUILD_DIR = build_linux_asan

# Build tool: Ninja if installed, else Make (override with USE_NINJA=0/1). Switching tools needs "make clean".
USE_NINJA ?= $(if $(shell command -v ninja),1,0)
ifeq ($(USE_NINJA),1)
  CMAKE_GENERATOR_ARG = -G Ninja
else
  BUILD_TOOL_ARGS = -- --no-print-directory
endif

# Location where pip-installable packages live
PYTHON_SITE_PACKAGES := $(shell python3 -c "import sysconfig, pathlib, json; print(sysconfig.get_paths()['purelib'])")

# The compiled extension that CMake puts into ./lib
PYTHON_SHARED_OBJ := $(wildcard lib/quadriga_lib.cpython*linux-gnu.so)

# Autodetect Octave
OCTAVE_VERSION := $(shell mkoctfile -v 2>/dev/null)

# Get Quadriga-Lib version
QUADRIGA_VERSION := $(shell grep -oP '(?<=#define QUADRIGA_LIB_VERSION_STR ")[^"]+' include/quadriga_lib.hpp)

# Matlab
MATLAB_BIN = matlab

ifneq ($(gcc_version),)
  CMAKE_CXX_ARG = -D CMAKE_C_COMPILER=gcc-$(gcc_version) -D CMAKE_CXX_COMPILER=g++-$(gcc_version)
endif

all:
	cmake -B $(CMAKE_BUILD_DIR) $(CMAKE_GENERATOR_ARG) -D CMAKE_INSTALL_PREFIX=. $(CMAKE_CXX_ARG) \
		-D ENABLE_MATLAB=$(matlab) \
		-D ENABLE_OCTAVE=$(octave) \
		-D ENABLE_MEX_DOC=$(python) \
		-D ENABLE_PYTHON=$(python) \
		-D ENABLE_TESTS=OFF \
		-D ARMA_EXT=$(arma_internal) \
		-D ENABLE_AVX2=$(avx2) \
		-D ENABLE_CUDA=$(cuda) \
		-D ENABLE_STATIC_LIB=$(static_lib) \
		-D ENABLE_SHARED_LIB=$(shared_lib)
	cmake --build $(CMAKE_BUILD_DIR) --parallel $(shell nproc) $(BUILD_TOOL_ARGS)
	cmake --install $(CMAKE_BUILD_DIR)

python:
	cmake -B $(CMAKE_BUILD_DIR) $(CMAKE_GENERATOR_ARG) -D CMAKE_INSTALL_PREFIX=. $(CMAKE_CXX_ARG) \
		-D ENABLE_MATLAB=OFF \
		-D ENABLE_OCTAVE=OFF \
		-D ENABLE_MEX_DOC=OFF \
		-D ENABLE_PYTHON=ON \
		-D ENABLE_TESTS=OFF \
		-D ARMA_EXT=$(arma_internal) \
		-D ENABLE_AVX2=$(avx2) \
		-D ENABLE_CUDA=OFF \
		-D ENABLE_STATIC_LIB=ON \
		-D ENABLE_SHARED_LIB=OFF
	cmake --build $(CMAKE_BUILD_DIR) --parallel $(shell nproc) $(BUILD_TOOL_ARGS)
	cmake --install $(CMAKE_BUILD_DIR)

python_install: python
	@if [ -z "$(PYTHON_SHARED_OBJ)" ]; then \
		echo "Error: quadriga_lib python package not found in lib/"; \
		exit 1; \
	fi
	@echo "Installing $(notdir $(PYTHON_SHARED_OBJ)) into $(PYTHON_SITE_PACKAGES)"
	cp  $(PYTHON_SHARED_OBJ)  $(PYTHON_SITE_PACKAGES)/

# Tests: builds and installs as "make" does, then builds Catch2 and the test binary and runs the tests
test:   all moxunit-lib
	cmake -B $(CMAKE_BUILD_DIR) -D ENABLE_TESTS=ON
	cmake --build $(CMAKE_BUILD_DIR) --parallel $(shell nproc)
	OMP_WAIT_POLICY=$${OMP_WAIT_POLICY:-passive} time $(CMAKE_BUILD_DIR)/test_bin
ifeq ($(octave),ON)
ifneq ($(OCTAVE_VERSION),)
	time octave --eval "cd tests; quadriga_lib_mex_tests;"
endif
endif
ifeq ($(matlab),ON)
	time $(MATLAB_BIN) -batch "run('tests/quadriga_lib_mex_tests.m');"
endif
ifeq ($(python),ON)
	if ls lib/quadriga_lib.cpython*linux-gnu.so >/dev/null 2>&1; then \
		PYTHONPATH=$(CURDIR)/lib python3 -m pytest tests/python_tests -x -s; \
	fi
endif

python_test: 
	PYTHONPATH=$(CURDIR)/lib python3 -m pytest tests/python_tests -x -s

# C++ tests under AddressSanitizer and UndefinedBehaviorSanitizer, in their own build folder. Stops at the first report.
# TEST_ARGS is passed to test_bin (e.g. TEST_ARGS='"qd_arrayant*"'). ASAN_OPTIONS and UBSAN_OPTIONS from the
# environment replace the defaults.
test-asan:
	cmake -B $(ASAN_BUILD_DIR) $(CMAKE_GENERATOR_ARG) $(CMAKE_CXX_ARG) \
		-D ENABLE_SANITIZERS=ON \
		-D ENABLE_TESTS=ON \
		-D ENABLE_MATLAB=OFF \
		-D ENABLE_OCTAVE=OFF \
		-D ENABLE_MEX_DOC=OFF \
		-D ENABLE_PYTHON=OFF \
		-D ARMA_EXT=$(arma_internal) \
		-D ENABLE_AVX2=$(avx2) \
		-D ENABLE_CUDA=OFF \
		-D ENABLE_STATIC_LIB=ON \
		-D ENABLE_SHARED_LIB=OFF
	cmake --build $(ASAN_BUILD_DIR) --target test_bin --parallel $(shell nproc) $(BUILD_TOOL_ARGS)
	OMP_WAIT_POLICY=$${OMP_WAIT_POLICY:-passive} \
		ASAN_OPTIONS=$${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1} \
		UBSAN_OPTIONS=$${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1} \
		time $(ASAN_BUILD_DIR)/test_bin $(TEST_ARGS)
.PHONY: test-asan

# Documentation: docs_builder/README.md and website_builder/README.md
docs:
	python3 docs_builder/build.py docs
docs-check:
	python3 docs_builder/build.py docs --check
website:
	@test -f website_builder/build.py || { echo "make website: website_builder/ is not part of the source package; build the website from a git clone"; exit 1; }
	python3 website_builder/build.py website
.PHONY: docs docs-check website

# Tutorial notebooks: tutorials/README.md; runs them into tutorials/executed/, in parallel (JOBS=4 by default)
tutorials:
	$(MAKE) -C tutorials
.PHONY: tutorials

moxunit-lib:
	- rm -rf external/MOxUnit-master
	unzip external/MOxUnit.zip -d external/

# Build source package from committed release
package:
	mkdir -p release
	git archive --format=zip --prefix=quadriga_lib_$(QUADRIGA_VERSION)/ -o release/quadriga_lib_$(QUADRIGA_VERSION).zip HEAD

# Build dirty package for testing - do not publish these!
package_dirty: clean
	mkdir -p release
	- rm release/quadriga_lib_$(QUADRIGA_VERSION).zip
	zip -r release/quadriga_lib_$(QUADRIGA_VERSION).zip . -x '.git/*' -x 'release/*' -x 'website/*'

# Python targets:
#
# PyPI packaging requires conda env "quadriga-pip": 
#   sudo apt install qemu-user-binfmt-hwe binfmt-support
#   conda create --name quadriga-pip -c conda-forge python=3.14 numpy cmake compilers make pytest pip
#   conda activate quadriga-pip
#   pip install scikit-build-core build twine cibuildwheel
#
# Publishing:
#   twine upload --repository testpypi --skip-existing dist/*.tar.gz wheelhouse/*.whl
#   twine upload dist/*.tar.gz wheelhouse/*.whl

pip-sdist:
	python3 -m build --sdist
	pip install dist/quadriga_lib-*.tar.gz
	python3 -c "import quadriga_lib; print(quadriga_lib.components())"
	python3 -m pytest tests/python_tests -x -s
	pip uninstall -y quadriga-lib

pip-x86:
	@ls wheelhouse/quadriga_lib-$(QUADRIGA_VERSION)-*x86_64.whl 1>/dev/null 2>&1 || \
		cibuildwheel --platform linux 2>&1 | tee build_pip_x86.log

CIBW_AARCH64_COMMON = CIBW_ARCHS="aarch64" CIBW_ENVIRONMENT='CMAKE_GENERATOR="Unix Makefiles" CMAKE_BUILD_PARALLEL_LEVEL="32"'

pip-aarch64-cp310:
	@ls wheelhouse/quadriga_lib-$(QUADRIGA_VERSION)-cp310-*aarch64.whl 1>/dev/null 2>&1 || \
		$(CIBW_AARCH64_COMMON) CIBW_BUILD="cp310-manylinux_aarch64" cibuildwheel --platform linux 2>&1 | tee build_pip_aarch64_cp310.log

pip-aarch64-cp311:
	@ls wheelhouse/quadriga_lib-$(QUADRIGA_VERSION)-cp311-*aarch64.whl 1>/dev/null 2>&1 || \
		$(CIBW_AARCH64_COMMON) CIBW_BUILD="cp311-manylinux_aarch64" cibuildwheel --platform linux 2>&1 | tee build_pip_aarch64_cp311.log

pip-aarch64-cp312:
	@ls wheelhouse/quadriga_lib-$(QUADRIGA_VERSION)-cp312-*aarch64.whl 1>/dev/null 2>&1 || \
		$(CIBW_AARCH64_COMMON) CIBW_BUILD="cp312-manylinux_aarch64" cibuildwheel --platform linux 2>&1 | tee build_pip_aarch64_cp312.log

pip-aarch64-cp313:
	@ls wheelhouse/quadriga_lib-$(QUADRIGA_VERSION)-cp313-*aarch64.whl 1>/dev/null 2>&1 || \
		$(CIBW_AARCH64_COMMON) CIBW_BUILD="cp313-manylinux_aarch64" cibuildwheel --platform linux 2>&1 | tee build_pip_aarch64_cp313.log

pip-aarch64-cp314:
	@ls wheelhouse/quadriga_lib-$(QUADRIGA_VERSION)-cp314-*aarch64.whl 1>/dev/null 2>&1 || \
		$(CIBW_AARCH64_COMMON) CIBW_BUILD="cp314-manylinux_aarch64" cibuildwheel --platform linux 2>&1 | tee build_pip_aarch64_cp314.log

pip: pip-sdist 
	$(MAKE) -j pip-x86 pip-aarch64-cp310 pip-aarch64-cp311 pip-aarch64-cp312 pip-aarch64-cp313 pip-aarch64-cp314

# Ubuntu targets:
DEB_JAMMY := quadriga-lib_$(QUADRIGA_VERSION)-1+jammy_amd64.deb
release/$(DEB_JAMMY):
	mkdir -p release
	docker build --build-arg UBUNTU_VERSION=22.04 --progress=plain -f Dockerfile.ubuntu -t quadriga-deb-jammy . 2>&1 | tee build_jammy.log
	docker run --rm -v /tmp/quadriga_docker_out:/out quadriga-deb-jammy
	cp /tmp/quadriga_docker_out/$(DEB_JAMMY) release/

DEB_NOBLE := quadriga-lib_$(QUADRIGA_VERSION)-1+noble_amd64.deb
release/$(DEB_NOBLE):
	mkdir -p release
	docker build --build-arg UBUNTU_VERSION=24.04 --progress=plain -f Dockerfile.ubuntu -t quadriga-deb-noble . 2>&1 | tee build_noble.log
	docker run --rm -v /tmp/quadriga_docker_out:/out quadriga-deb-noble
	cp /tmp/quadriga_docker_out/$(DEB_NOBLE) release/

DEB_RECOLUTE := quadriga-lib_$(QUADRIGA_VERSION)-1+resolute_amd64.deb
release/$(DEB_RECOLUTE):
	mkdir -p release
	docker build --build-arg UBUNTU_VERSION=26.04 --progress=plain -f Dockerfile.ubuntu -t quadriga-deb-resolute . 2>&1 | tee build_resolute.log
	docker run --rm -v /tmp/quadriga_docker_out:/out quadriga-deb-resolute
	cp /tmp/quadriga_docker_out/$(DEB_RECOLUTE) release/

deb-jammy:  release/$(DEB_JAMMY)
deb-noble:  release/$(DEB_NOBLE)
deb-resolute: release/$(DEB_RECOLUTE)

deb: 
	$(MAKE) -j deb-jammy deb-noble deb-resolute

# Release
release:
	test -z "$$(git status --porcelain --untracked-files=no; git status --porcelain -- docs)" || { echo "uncommitted changes: commit before a release"; exit 1; }
	python3 docs_builder/build.py docs --check
	$(MAKE) package
	$(MAKE) deb
	$(MAKE) pip
.PHONY: release

clean:
	- rm -rf $(CMAKE_BUILD_DIR)
	- rm -rf $(ASAN_BUILD_DIR)
	- rm +quadriga_lib/*.mex
	- rm +quadriga_lib/*.mexa64
	- rm release/*.deb
	- rm release/quadriga_lib-$(QUADRIGA_VERSION).zip
	- rm wheelhouse/*x86_64.whl
	- rm wheelhouse/*aarch64.whl
	- rm lib/*.a
	- rm lib/*.so
	- rm *.obj
	- rm tests/test_bin
	- rm tests/test.exe
	- rm -rf tests/python_tests/__pycache__
	- rm -rf .pytest_cache
	- rm -rf tests/.pytest_cache
	- rm *.hdf5
	- rm *.egg-info
	- rm build*.log	
	- rm -rf dist

tidy:   clean
	- rm -rf build*
	- rm version.tmp
	- rm -rf +quadriga_lib
	- rm -rf release
	- rm -rf lib
	- rm -rf wheelhouse 
	- rm -rf external/MOxUnit-master
	- rm -rf website
	- $(MAKE) -C tutorials clean
	- find docs_builder website_builder tutorials \( -name __pycache__ -o -name .ipynb_checkpoints \) -type d -prune -exec rm -rf {} +
	