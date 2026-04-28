# This Makefile is for Linux / GCC environments

# Set Armadillo and HDF5 sources
# Internal HDF5 deactivates Python and Octave
hdf5_internal = OFF
arma_internal = OFF
static_lib = ON
shared_lib = OFF
octave = ON
matlab = OFF
python = ON
avx2 = OFF
cuda = OFF

CMAKE_BUILD_DIR = build_linux

# Location where pip-installable packages live
PYTHON_SITE_PACKAGES := $(shell python3 -c "import sysconfig, pathlib, json; print(sysconfig.get_paths()['purelib'])")

# The compiled extension that CMake puts into ./lib
PYTHON_SHARED_OBJ := $(wildcard lib/quadriga_lib.cpython*linux-gnu.so)

# Autodetect Octave
OCTAVE_VERSION := $(shell mkoctfile -v 2>/dev/null)

# Get Quadriga-Lib version
QUADRIGA_VERSION := $(shell grep -oP '(?<=#define QUADRIGA_LIB_VERSION_STR ")[^"]+' include/quadriga_lib.hpp)

all:
	cmake -B $(CMAKE_BUILD_DIR) -D CMAKE_INSTALL_PREFIX=. \
		-D ENABLE_MATLAB=$(matlab) \
		-D ENABLE_OCTAVE=$(octave) \
		-D ENABLE_MEX_DOC=$(python) \
		-D ENABLE_PYTHON=$(python) \
		-D ENABLE_TESTS=OFF \
		-D ARMA_EXT=$(arma_internal) \
		-D HDF5_STATIC=$(hdf5_internal) \
		-D ENABLE_AVX2=$(avx2) \
		-D ENABLE_CUDA=$(cuda) \
		-D ENABLE_STATIC_LIB=$(static_lib) \
		-D ENABLE_SHARED_LIB=$(shared_lib)
	cmake --build $(CMAKE_BUILD_DIR) --parallel -- --no-print-directory
	cmake --install $(CMAKE_BUILD_DIR)

python:
	cmake -B $(CMAKE_BUILD_DIR) -D CMAKE_INSTALL_PREFIX=. \
		-D ENABLE_MATLAB=OFF \
		-D ENABLE_OCTAVE=OFF \
		-D ENABLE_MEX_DOC=OFF \
		-D ENABLE_PYTHON=ON \
		-D ENABLE_TESTS=OFF \
		-D ARMA_EXT=$(arma_internal) \
		-D HDF5_STATIC=OFF \
		-D ENABLE_AVX2=$(avx2) \
		-D ENABLE_CUDA=OFF \
		-D ENABLE_STATIC_LIB=ON \
		-D ENABLE_SHARED_LIB=OFF
	cmake --build $(CMAKE_BUILD_DIR) --parallel -- --no-print-directory
	cmake --install $(CMAKE_BUILD_DIR)

python_install: python
	@if [ -z "$(PYTHON_SHARED_OBJ)" ]; then \
		echo "Error: quadriga_lib python package not found in lib/"; \
		exit 1; \
	fi
	@echo "Installing $(notdir $(PYTHON_SHARED_OBJ)) into $(PYTHON_SITE_PACKAGES)"
	cp  $(PYTHON_SHARED_OBJ)  $(PYTHON_SITE_PACKAGES)/

# Tests
test:   moxunit-lib
	cmake -B $(CMAKE_BUILD_DIR) -D ENABLE_TESTS=ON
	cmake --build $(CMAKE_BUILD_DIR) --parallel
	$(CMAKE_BUILD_DIR)/test_bin
ifeq ($(octave),ON)
ifneq ($(OCTAVE_VERSION),)
	octave --eval "cd tests; quadriga_lib_mex_tests;"
endif
endif
ifneq ($(PYTHON_SHARED_OBJ),)
	python3 -m pytest tests/python_tests -x -s
endif

python_test: 
	python3 -m pytest tests/python_tests -x -s

# Documentation
documentation:
	sed 's/__VERSION__/$(QUADRIGA_VERSION)/g' tools/download_md/download.md.in > tools/download_md/download.md

	python3 tools/extract_version.py tools/html_parts/mex_api.html.part "MALAB / Octave API Documentation for Quadriga-Lib"
	python3 tools/extract_version.py tools/html_parts/python_api.html.part "Python API Documentation for Quadriga-Lib"
	python3 tools/extract_version.py tools/html_parts/cpp_api.html.part "C++ API Documentation for Quadriga-Lib"

	python3 tools/extract_html.py -o html_docu/index.html -p tools/html_parts/index.html.part
	python3 tools/extract_html.py -o html_docu/mex_api.html -p tools/html_parts/mex_api.html.part -d api_mex/ -l
	python3 tools/extract_html.py -o html_docu/cpp_api.html -p tools/html_parts/cpp_api.html.part -d src/ 
	python3 tools/extract_html.py -o html_docu/python_api.html -p tools/html_parts/python_api.html.part -d api_python/ 
	python3 tools/extract_html.py -o html_docu/formats.html -p tools/html_parts/formats.html.part -d tools/data_formats_md/
	python3 tools/extract_html.py -o html_docu/faq.html -p tools/html_parts/faq.html.part -d tools/questions_md/ -c
	python3 tools/extract_html.py -o html_docu/download.html -p tools/html_parts/download.html.part -d tools/download_md/ -c

	python3 tools/extract_md.py -o html_docu/quadriga_lib_api_cpp.md -a cpp

moxunit-lib:
	- rm -rf external/MOxUnit-master
	unzip external/MOxUnit.zip -d external/

package:
	mkdir -p release
	git archive --format=zip --prefix=quadriga_lib_$(QUADRIGA_VERSION)/ -o release/quadriga_lib_$(QUADRIGA_VERSION).zip HEAD

# Python targets:
#
# PyPI packaging requires conda env "quadriga-pip": 
#   sudo apt install qemu-user-static binfmt-support
#   conda create --name quadriga-pip -c conda-forge python=3.13 numpy cmake compilers make pytest
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

DEB_PLUCKY := quadriga-lib_$(QUADRIGA_VERSION)-1+plucky_amd64.deb
release/$(DEB_PLUCKY):
	mkdir -p release
	docker build --build-arg UBUNTU_VERSION=25.10 --progress=plain -f Dockerfile.ubuntu -t quadriga-deb-plucky . 2>&1 | tee build_plucky.log
	docker run --rm -v /tmp/quadriga_docker_out:/out quadriga-deb-plucky
	cp /tmp/quadriga_docker_out/$(DEB_PLUCKY) release/

deb-jammy:  release/$(DEB_JAMMY)
deb-noble:  release/$(DEB_NOBLE)
deb-plucky: release/$(DEB_PLUCKY)

deb: 
	$(MAKE) -j deb-jammy deb-noble deb-plucky

# Release
release: documentation package
	$(MAKE) deb
	$(MAKE) pip

clean:
	- rm -rf $(CMAKE_BUILD_DIR)
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
	- rm -rf external/hdf5-prebuilt
	- rm -rf external/Catch2-prebuilt
	