# This Makefile is for Linux / GCC environments

# Steps for compiling Quadriga-Lib (Linux):
# - Get required tools and libraries: make, cmake, g++
# - Set MATLAB path below (or leave empty)
# - Run "make"

# Set path to your MATLAB installation (optional):
# If left empty, the build script trys to autodetect the MATLAB path. 
# If none is found, MATLAB targets are not compiled.
MATLAB_PATH = #/usr/local/MATLAB/R2023a

# Leave empty for using system library (recommended)
# Static linking the HDF5 library may cause Octave to crash
# Ubuntu 22.04 Armadillo system library does not work with MEX due to a bug
hdf5_version      = # 1.14.2
armadillo_version = 14.2.2

# External libraries
# External libraries are located in the 'external' folder. Set the version numbers here.
# You need to compile the HDF5 and Catch2 libraries (e.g. using 'make hdf5lib' or 'make catch2lib' )
pugixml_version   = 1.13
catch2_version    = 3.4.0
pybind11_version  = 2.12.0

# The following sections should not require editing.

# Compilers
CC   = g++
MEX  = $(MATLAB_PATH)/bin/mex
OCT  = mkoctfile

# Headers and Libraries
PUGIXML_H   = external/pugixml-$(pugixml_version)/src
CATCH2      = external/Catch2-$(catch2_version)-Linux
PYBIND11_H  = external/pybind11-$(pybind11_version)/include

# Compiler flags
CCFLAGS     = -std=c++17 -O3 -fPIC -fopenmp -Wall -Wconversion -Wpedantic -Wextra

# Selecting armadillo version
ifeq ($(armadillo_version),) # Use system
	ARMA_H = /usr/include
else # Use package
	ARMA_H = external/armadillo-$(armadillo_version)/include
	ARMA_CM = -DARMA_EXT=ON
endif

# Linking options for HDF5 library
ifeq ($(hdf5_version),) # Dynamic linking
	HDF5_H      = /usr/include/hdf5/serial/
	HDF5_STATIC = 
	HDF5_OBJ    = 
	HDF5_DYN    = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5
else # Static linking
	HDF5_H      = external/hdf5-$(hdf5_version)-Linux/include
	HDF5_STATIC = build/libhdf5.a
	HDF5_OBJ    = build/H5*.o
	HDF5_DYN    = 
	HDF5_CM     = -DHDF5_STATIC=ON
endif

# List of API functions for MATLAB / Octave
api_mex = $(wildcard api_mex/*.cpp)

# Check if MATLAB is installed and derive MATLAB API file names
ifeq ($(MATLAB_PATH),)
	MATLAB_PATH := $(shell readlink -f $(shell readlink -f $(shell which matlab)) | sed 's/\/bin\/matlab//')
endif
ifneq ($(MATLAB_PATH),)
	MATLAB_TARGETS = $(api_mex:api_mex/%.cpp=+quadriga_lib/%.mexa64)
endif

# Check if Octave is installed and derive Octave API file names
OCTAVE_VERSION := $(shell mkoctfile -v 2>/dev/null)
ifneq ($(OCTAVE_VERSION),)
	OCTAVE_TARGETS = $(api_mex:api_mex/%.cpp=+quadriga_lib/%.mex)
endif

# Autodetect the Python include path
api_python = $(wildcard api_python/*.cpp)

PYTHON_H = $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
ifeq ($(wildcard $(PYTHON_H)/Python.h),)
    PYTHON_H =
else
	PYTHON_EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)
	PYTHON_TARGET = lib/quadriga_lib$(PYTHON_EXTENSION_SUFFIX)
endif

# Compilation targets
.PHONY: dirs
all:
	@$(MAKE) dirs
	@$(MAKE) lib/libquadriga.a   $(PYTHON_TARGET)   $(OCTAVE_TARGETS)   $(MATLAB_TARGETS)   tests/test_bin

cpp:
	@$(MAKE) dirs
	@$(MAKE) lib/libquadriga.a

python: 
	@$(MAKE) dirs
	@$(MAKE) $(PYTHON_TARGET)

mex_matlab: cpp
	@$(MAKE) dirs
	@$(MAKE) $(MATLAB_TARGETS)

mex_octave: cpp
	@$(MAKE) dirs
	@$(MAKE) $(OCTAVE_TARGETS)

dirs:
	mkdir -p build
	mkdir -p lib
	mkdir -p +quadriga_lib
	mkdir -p release

lib/quadriga_lib$(PYTHON_EXTENSION_SUFFIX):  api_python/python_main.cpp   lib/libquadriga.a   $(api_python)   api_python/python_arma_adapter.hpp
	$(CC) -shared $(CCFLAGS) $< lib/libquadriga.a -o $@ -I include -I $(PYBIND11_H) -I $(PYTHON_H) -I $(ARMA_H) -lgomp -ldl $(HDF5_DYN)

# Use cmake to compile
cmake:
	cmake -B build_linux -D CMAKE_INSTALL_PREFIX=. $(ARMA_CM) $(HDF5_CM)
	cmake --build build_linux -j32 
	cmake --install build_linux

# Tests
tests 		= $(wildcard tests/catch2_tests/*.cpp)

test:   lib/libquadriga.a   tests/test_bin
	tests/test_bin
ifneq ($(OCTAVE_VERSION),)
	octave --eval "cd tests; quadriga_lib_mex_tests;"
endif
ifneq ($(PYTHON_TARGET),)
	pytest tests/python_tests -x -s
endif

test_catch2:   tests/test_bin
	tests/test_bin

test_cmake:  tests/quadriga_lib_catch2_tests.cpp   cmake
ifeq ($(hdf5_version),) # Dynamic linking of HDF5
	$(CC) -std=c++17 $< lib/libquadriga.a -o tests/test_cmake -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -lCatch2 -lgomp -ldl $(HDF5_DYN)
else # Static linking
	$(CC) -std=c++17 $< lib/libquadriga.a lib/libhdf5.a -o tests/test_cmake -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -lCatch2 -lgomp -ldl
endif
	tests/test_cmake
ifneq ($(OCTAVE_VERSION),)
	octave --eval "cd tests; quadriga_lib_mex_tests;"
endif
ifneq ($(PYTHON_TARGET),)
	pytest tests/python_tests -x -s
endif

tests/test_bin:   tests/quadriga_lib_catch2_tests.cpp   lib/libquadriga.a   $(tests)
ifeq ($(hdf5_version),) # Dynamic linking of HDF5
	$(CC) -std=c++17 $< lib/libquadriga.a -o $@ -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -lCatch2 -lgomp -ldl $(HDF5_DYN)
else # Static linking
	$(CC) -std=c++17 $< lib/libquadriga.a lib/libhdf5.a -o $@ -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -lCatch2 -lgomp -ldl
endif

# C++ object files
build/%.o:   src/%.cpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I include -I src -I $(PUGIXML_H) -I $(HDF5_H) -I $(ARMA_H) 

# C++ object files with AVX acceleration
build/%_avx2.o:   src/%_avx2.cpp
	$(CC) -mavx2 -mfma $(CCFLAGS) -c $< -o $@ -I src -I $(ARMA_H)

# Archive file for static linking
src_cpp = $(wildcard src/*.cpp)

build/libhdf5.a:
	cp external/hdf5-$(hdf5_version)-Linux/lib/libhdf5.a build/
	( cd build/ && ar x libhdf5.a && cd .. )

lib/libquadriga.a:   $(src_cpp:src/%.cpp=build/%.o)   $(HDF5_STATIC)
			ar rcs $@ $^ $(HDF5_OBJ)

# MEX MATLAB interface
+quadriga_lib/%.mexa64:   api_mex/%.cpp   lib/libquadriga.a
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -lgomp $(HDF5_DYN)

# MEX Ocate interface
+quadriga_lib/%.mex:   api_mex/%.cpp   lib/libquadriga.a
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) -s 

# Documentation of MEX files
mex_docu:   $(api_mex:api_mex/%.cpp=+quadriga_lib/%.m)

+quadriga_lib/%.m:   api_mex/%.cpp
	rm -f $@
	python3 tools/extract_matlab_comments.py $< $@

documentation:   dirs   mex_docu
	mkdir -p +quadriga_lib
	python3 tools/extract_html.py html_docu/index.html tools/html_parts/index.html.part
	python3 tools/extract_html.py html_docu/mex_api.html tools/html_parts/mex_api.html.part api_mex/ 
	python3 tools/extract_html.py html_docu/cpp_api.html tools/html_parts/cpp_api.html.part src/ 
	python3 tools/extract_html.py html_docu/python_api.html tools/html_parts/python_api.html.part api_python/ 
	python3 tools/extract_html.py html_docu/formats.html tools/html_parts/index.html.part
	python3 tools/extract_html.py html_docu/faq.html tools/html_parts/index.html.part
	python3 tools/extract_html.py html_docu/contact.html tools/html_parts/index.html.part
	python3 tools/extract_html.py html_docu/download.html tools/html_parts/index.html.part

# External libraries
external:   armadillo-lib   pybind11-lib   pugixml-lib   hdf5-lib   catch2-lib   moxunit-lib   

armadillo-lib:
	- rm -rf external/armadillo-$(armadillo_version)
	unzip external/armadillo-$(armadillo_version).zip -d external/

pybind11-lib:
	- rm -rf external/pybind11-$(pybind11_version)
	unzip external/pybind11-$(pybind11_version).zip -d external/

pugixml-lib:
	- rm -rf external/pugixml-$(pugixml_version)
	unzip external/pugixml-$(pugixml_version).zip -d external/

hdf5-lib:
ifneq ($(hdf5_version),)
	- rm -rf external/hdf5-$(hdf5_version)
	- rm -rf external/hdf5-$(hdf5_version)-Linux
	unzip external/hdf5-$(hdf5_version).zip -d external/
	( cd external/hdf5-$(hdf5_version) && ./configure CFLAGS="-O3 -march=native -fPIC" \
	   --enable-tools=no --enable-tests=no --enable-shared=no --enable-hl=no --enable-optimization=high --with-zlib=no --enable-threadsafe=no)
	( cd external/hdf5-$(hdf5_version) && make -j8 && make install )
	mv external/hdf5-$(hdf5_version)/hdf5 external/hdf5-$(hdf5_version)-Linux
	- rm -rf external/hdf5-$(hdf5_version)
endif

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

moxunit-lib:
	- rm -rf external/MOxUnit-master
	unzip external/MOxUnit.zip -d external/

clean:
	- rm -rf external/build
	- rm -rf external/Catch2-$(catch2_version)
	- rm -rf external/hdf5-$(hdf5_version)
	- rm -rf +quadriga_lib
	- rm -rf release
	- rm -rf lib
	- rm *.obj
	- rm tests/test_bin
	- rm tests/test_cmake
	- rm tests/test_static_bin
	- rm tests/test.exe
	- rm -rf build
	- rm -rf build*
	- rm -rf tests/python_tests/__pycache__
	- rm -rf .pytest_cache
	- rm -rf tests/.pytest_cache
	- rm *.hdf5

tidy: clean 
	- rm -rf external/Catch2-$(catch2_version)-Linux
	- rm -rf external/hdf5-$(hdf5_version)-Linux
	- rm -rf external/armadillo-$(armadillo_version)
	- rm -rf external/pugixml-$(pugixml_version)
	- rm -rf external/pybind11-$(pybind11_version)
	- rm -rf external/MOxUnit-master
	
build/quadriga-lib-version:   src/bin/version.cpp   lib/libquadriga.a
	$(CC) -std=c++17 $^ lib/libquadriga.a -o $@ -I src -I include -I $(ARMA_H)

build/quadriga-lib-arma-version:   src/bin/arma_version.cpp
	$(CC) -std=c++17 $^ lib/libquadriga.a -o $@ -I src -I include -I $(ARMA_H)

release:  all   build/quadriga-lib-version
	- mkdir release
	tar czf release/quadrigalib-v$(shell build/quadriga-lib-version)-Ubuntu-$(shell lsb_release -r -s)-amd64.tar.gz \
		+quadriga_lib/*.mex +quadriga_lib/*.mexa64 +quadriga_lib/*.m include lib/*.a

package:  cpp   build/quadriga-lib-version
	- mkdir release
	- rm -rf release/quadriga_lib-$(shell build/quadriga-lib-version)
	- rm release/quadriga_lib-$(shell build/quadriga-lib-version).zip
	mkdir release/quadriga_lib-$(shell build/quadriga-lib-version)
	mkdir release/quadriga_lib-$(shell build/quadriga-lib-version)/external
	cp external/armadillo-$(armadillo_version).zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/pugixml-$(pugixml_version).zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/Catch2-$(catch2_version).zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/hdf5-*.zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/pybind11-$(pybind11_version).zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/MOxUnit.zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp -R include release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R api_mex release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R src release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R tests release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R tools release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R html_docu release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R api_python release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp CMakeLists.txt release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp GNUmakefile release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp LICENSE release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp Makefile release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp README.md release/quadriga_lib-$(shell build/quadriga-lib-version)/
	- rm -rf release/quadriga_lib-$(shell build/quadriga-lib-version)/tests/afl
	( cd release && zip -r quadriga_lib-$(shell build/quadriga-lib-version).zip quadriga_lib-$(shell build/quadriga-lib-version)/ )
	- rm -rf release/quadriga_lib-$(shell build/quadriga-lib-version)
