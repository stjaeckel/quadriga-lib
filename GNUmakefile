# This Makefile is for Linux / GCC environments

# Set Armadillo and HDF5 sources
# Internal HDF5 deactivates Python and Octave
hdf5_internal = OFF
arma_internal = ON
octave = ON
matlab = ON
avx2 = ON

CMAKE_BUILD_DIR = build_linux

# Location where pip-installable packages live
PYTHON_SITE_PACKAGES := $(shell python3 -c "import sysconfig, pathlib, json; print(sysconfig.get_paths()['purelib'])")

# The compiled extension that CMake puts into ./lib
PYTHON_SHARED_OBJ    := $(wildcard lib/quadriga_lib.cpython*linux-gnu.so)

# Autodetect Octave
OCTAVE_VERSION := $(shell mkoctfile -v 2>/dev/null)

all:
	cmake -B $(CMAKE_BUILD_DIR) -D CMAKE_INSTALL_PREFIX=. -D ENABLE_MATLAB=$(matlab) -D ENABLE_OCTAVE=$(octave) -D ENABLE_MEX_DOC=ON -D ENABLE_PYTHON=ON -D ARMA_EXT=$(arma_internal) -D HDF5_STATIC=$(hdf5_internal) -D ENABLE_AVX2=$(avx2) 
	cmake --build $(CMAKE_BUILD_DIR) --parallel
	cmake --install $(CMAKE_BUILD_DIR)

cpp:
	cmake -B $(CMAKE_BUILD_DIR) -D CMAKE_INSTALL_PREFIX=. -D ENABLE_MATLAB=OFF -D ENABLE_OCTAVE=OFF -D ENABLE_MEX_DOC=OFF -D ENABLE_PYTHON=OFF -D ARMA_EXT=$(arma_internal) -D HDF5_STATIC=$(hdf5_internal) -D ENABLE_AVX2=$(avx2) 
	cmake --build $(CMAKE_BUILD_DIR) --parallel
	cmake --install $(CMAKE_BUILD_DIR)

bin:   cpp
	cmake -B $(CMAKE_BUILD_DIR) -D ENABLE_BIN=ON -D ARMA_EXT=$(arma_internal) -D HDF5_STATIC=$(hdf5_internal) -D ENABLE_AVX2=$(avx2) 
	cmake --build $(CMAKE_BUILD_DIR) --parallel

python:
	cmake -B $(CMAKE_BUILD_DIR) -D CMAKE_INSTALL_PREFIX=. -D ENABLE_MATLAB=OFF -D ENABLE_OCTAVE=OFF -D ENABLE_MEX_DOC=OFF -D ENABLE_PYTHON=ON -D ARMA_EXT=$(arma_internal) -D ENABLE_AVX2=$(avx2) 
	cmake --build $(CMAKE_BUILD_DIR) --parallel
	cmake --install $(CMAKE_BUILD_DIR)

python_install: python
	@if [ -z "$(PYTHON_SHARED_OBJ)" ]; then \
		echo "Error: quadriga_lib python package not found in lib/"; \
		exit 1; \
	fi
	@echo "Installing $(notdir $(PYTHON_SHARED_OBJ)) into $(PYTHON_SITE_PACKAGES)"
	cp  $(PYTHON_SHARED_OBJ)  $(PYTHON_SITE_PACKAGES)/

# Tests
test:   all   moxunit-lib
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

cpp_test:   cpp
	cmake -B $(CMAKE_BUILD_DIR) -D ENABLE_TESTS=ON
	cmake --build $(CMAKE_BUILD_DIR) --parallel
	$(CMAKE_BUILD_DIR)/test_bin

python_test: 
	pytest tests/python_tests -x -s

# Documentation
documentation:   bin
	python3 tools/extract_version.py tools/html_parts/mex_api.html.part "MALAB / Octave API Documentation for Quadriga-Lib"
	python3 tools/extract_version.py tools/html_parts/python_api.html.part "Python API Documentation for Quadriga-Lib"
	python3 tools/extract_version.py tools/html_parts/cpp_api.html.part "C++ API Documentation for Quadriga-Lib"

	python3 tools/extract_html.py -o html_docu/index.html -p tools/html_parts/index.html.part
	python3 tools/extract_html.py -o html_docu/mex_api.html -p tools/html_parts/mex_api.html.part -d api_mex/ 
	python3 tools/extract_html.py -o html_docu/cpp_api.html -p tools/html_parts/cpp_api.html.part -d src/ 
	python3 tools/extract_html.py -o html_docu/python_api.html -p tools/html_parts/python_api.html.part -d api_python/ 
	python3 tools/extract_html.py -o html_docu/formats.html -p tools/html_parts/formats.html.part -d tools/data_formats_md/
	python3 tools/extract_html.py -o html_docu/faq.html -p tools/html_parts/faq.html.part -d tools/questions_md/ -c
	python3 tools/extract_html.py -o html_docu/download.html -p tools/html_parts/download.html.part -d tools/download_md/ -c

moxunit-lib:
	- rm -rf external/MOxUnit-master
	unzip external/MOxUnit.zip -d external/

catch2_version = 3.8.1
catch2-lib:
	- rm -rf external/build
	- rm -rf external/Catch2-$(catch2_version)
	- rm -rf external/Catch2-$(catch2_version)-Linux
	unzip external/Catch2-$(catch2_version).zip -d external/
	mkdir external/build
	cmake -S external/Catch2-$(catch2_version) -B external/build
	( cd external/build && make -j8 && make package )
	tar -xzf external/build/Catch2-$(catch2_version)-Linux.tar.gz -C external/
	- rm -rf external/build
	- rm -rf external/Catch2-$(catch2_version)

armadillo_version = 14.2.2
armadillo-lib:
	- rm -rf external/armadillo-$(armadillo_version)
	unzip external/armadillo-$(armadillo_version).zip -d external/

pybind11_version = 3.0.0
pybind11-lib:
	- rm -rf external/pybind11-$(pybind11_version)
	unzip external/pybind11-$(pybind11_version).zip -d external/

clean:
	- rm -rf external/build
	- rm -rf +quadriga_lib
	- rm -rf release
	- rm -rf lib
	- rm *.obj
	- rm tests/test_bin
	- rm tests/test_cmake
	- rm tests/test_static_bin
	- rm tests/test.exe
	- rm -rf $(CMAKE_BUILD_DIR)
	- rm -rf tests/python_tests/__pycache__
	- rm -rf .pytest_cache
	- rm -rf tests/.pytest_cache
	- rm *.hdf5
	- rm -rf external/MOxUnit-master
	- rm include/quadriga_lib_config.hpp

tidy:   clean
	- rm -rf build*

release:  all  bin
	- mkdir release
	tar czf release/quadrigalib-v$(shell $(CMAKE_BUILD_DIR)/version)-Ubuntu-$(shell lsb_release -r -s)-amd64.tar.gz \
		+quadriga_lib/*.mex +quadriga_lib/*.mexa64 +quadriga_lib/*.m include lib/*.a

package:  cpp  bin
	- mkdir release
	- rm -rf release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)
	- rm release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version).zip
	mkdir release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)
	mkdir release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/external
	cp external/armadillo-*.zip release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/external/
	cp external/pugixml-*.zip release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/external/
	cp external/Catch2-*.zip release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/external/
	cp external/hdf5-*.zip release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/external/
	cp external/pybind11-*.zip release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/external/
	cp external/MOxUnit.zip release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/external/
	cp -R include release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp -R api_mex release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp -R src release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp -R tests release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp -R tools release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp -R html_docu release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp -R api_python release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp CMakeLists.txt release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp GNUmakefile release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp LICENSE release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp Makefile release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	cp README.md release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/
	- rm -rf release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/tests/afl
	( cd release && zip -r quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version).zip quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)/ )
	- rm -rf release/quadriga_lib-$(shell $(CMAKE_BUILD_DIR)/version)
