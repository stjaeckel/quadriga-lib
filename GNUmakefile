# This Makefile is for Linux / GCC environments

# Compilers
CC    = g++
MEX   = /usr/local/MATLAB/R2021a/bin/mex
OCT   = mkoctfile

all:        mex_octave  mex_matlab 

# External libraries
ARMA_H      = external/armadillo-12.6.3/include
PUGIXML_H   = external/pugixml-1.13/src
CATCH2      = external/Catch2-3.3.2-Linux
HDF5_H      = /usr/include/hdf5/serial  # external/hdf5-1.14.2-Linux/include
HDF5_LIB    = /usr/lib/x86_64-linux-gnu/hdf5/serial  # external/hdf5-1.14.2-Linux/lib

# Configurations
CCFLAGS     = -std=c++17 -fPIC -O3 -fopenmp# -Wall #-Werror #-Wl,--gc-sections -Wall -Wextra -Wpedantic

# Sourcees
src     	= $(wildcard src/*.cpp)
mex         = $(wildcard mex/*.cpp)
tests 		= $(wildcard tests/catch2_tests/*.cpp)

.PHONY: 	clean all tidy mex_matlab mex_octave
.SECONDARY: $(src:src/%.cpp=build/%.o)

mex_matlab: $(mex:mex/%.cpp=+quadriga_lib/%.mexa64)
mex_octave: $(mex:mex/%.cpp=+quadriga_lib/%.mex)

test:   tests/test_bin
	tests/test_bin

tests/test_bin:   tests/quadriga_lib_catch2_tests.cpp   lib/quadriga_lib.a   $(tests)
	$(CC) $(CCFLAGS) $< lib/quadriga_lib.a -o $@ -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -L $(HDF5_LIB) -lCatch2 -lhdf5

# Individual Library files
build/qd_arrayant.o:   src/qd_arrayant.cpp   include/quadriga_lib.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_arrayant_qdant.o:   src/qd_arrayant_qdant.cpp   src/qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(PUGIXML_H)

build/qd_arrayant_interpolate.o:   src/qd_arrayant_interpolate.cpp   src/qd_arrayant_interpolate.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_channel.o:   src/qd_channel.cpp   include/quadriga_lib.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(HDF5_H)

build/quadriga_tools.o:   src/quadriga_tools.cpp   include/quadriga_tools.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) 

build/quadriga_lib.o:   src/quadriga_lib.cpp   include/quadriga_lib.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

#build/%.o:   src/%.cpp
#	$(CC) $(CCFLAGS) -c $< -o $@ $(ARMA_LIB) -I src -I include

# Archive file for static linking
lib/quadriga_lib.a:   build/quadriga_lib.o  build/qd_arrayant.o  build/qd_channel.o   \
                      build/quadriga_tools.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o
	ar rcs $@ $^

# MEX MATLAB interface
+quadriga_lib/test_hdf5.mexa64:   mex/test_hdf5.cpp   lib/quadriga_lib.a
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -L$(HDF5_LIB) -lhdf5 -lgomp

+quadriga_lib/%.mexa64:   mex/%.cpp   lib/quadriga_lib.a
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -lgomp

# MEX Ocate interface
+quadriga_lib/test_hdf5.mex:   mex/test_hdf5.cpp   lib/quadriga_lib.a
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) -L$(HDF5_LIB) -lhdf5 -s

+quadriga_lib/%.mex:   mex/%.cpp   lib/quadriga_lib.a
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) -s

# Maintainance section
clean:
	- rm build/*
	- rm +quadriga_lib/*.manifest
	- rm +quadriga_lib/*.exp
	- rm +quadriga_lib/*.lib
		- rm *.obj

tidy: clean 
	- rm +quadriga_lib/*.mex*
	- rm lib/*