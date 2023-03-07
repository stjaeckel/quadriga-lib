# This Makefile is for Linux / GCC environments

# Compilers
CC    = g++
MEX   = /usr/local/MATLAB/R2021a/bin/mex
OCT   = mkoctfile

all:        mex_octave   mex_matlab

# External library headers
ARMA_H      = external/armadillo-11.4.2/include
PUGIXML_H   = external/pugixml-1.13/src

# Configurations
CCFLAGS     = -std=c++17 -fPIC -O3 -fopenmp #-Wl,--gc-sections -Wall -Wextra -Wpedantic
LIBS        = -lgomp #-lblas

# Sourcees
src     	= $(wildcard src/*.cpp)
mex         = $(wildcard mex/*.cpp)

.PHONY: 	clean all tidy mex_matlab mex_octave
.SECONDARY: $(src:src/%.cpp=build/%.o)

mex_matlab: $(mex:mex/%.cpp=+quadriga_lib/%.mexa64)
mex_octave: $(mex:mex/%.cpp=+quadriga_lib/%.mex)

# Individual Library files
build/qd_arrayant_qdant.o:   src/qd_arrayant_qdant.cpp   src/qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I $(ARMA_H) -I $(PUGIXML_H) -I src -I include $(LIBS)

build/%.o:   src/%.cpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I $(ARMA_H) -I src -I include $(LIBS)

build/quadriga_lib_combined.o:   build/quadriga_lib.o   build/quadriga_tools.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o
	ld -r -o $@ $^

# MEX interface files
+quadriga_lib/%.mexa64:   mex/%.cpp
	$(MEX) -outdir +quadriga_lib $^ -I$(ARMA_H) -Isrc -Iinclude $(LIBS)

+quadriga_lib/%.mex:   mex/%.cpp
	$(OCT) --mex -o $@ $^ -I$(ARMA_H) -Isrc -Iinclude $(LIBS) -s

# List of additional dependencies
build/quadriga_lib.o:   include/quadriga_lib.hpp
build/quadriga_tools.o:   include/quadriga_tools.hpp
build/qd_arrayant_interpolate.o:   src/qd_arrayant_interpolate.hpp

# List of Octave-MEX files and their dependencies
+quadriga_lib/arrayant_interpolate.mex:   build/quadriga_lib_combined.o
+quadriga_lib/arrayant_qdant_read.mex:   build/quadriga_lib_combined.o
+quadriga_lib/calc_rotation_matrix.mex:   build/quadriga_tools.o
+quadriga_lib/cart2geo.mex:   build/quadriga_tools.o
+quadriga_lib/geo2cart.mex:   build/quadriga_tools.o
+quadriga_lib/version.mex:   build/quadriga_lib_combined.o

# List of MATLAB-MEX files and their dependencies
+quadriga_lib/arrayant_interpolate.mexa64:   build/quadriga_lib_combined.o
+quadriga_lib/arrayant_qdant_read.mexa64:   build/quadriga_lib_combined.o
+quadriga_lib/calc_rotation_matrix.mexa64:   build/quadriga_tools.o
+quadriga_lib/cart2geo.mexa64:   build/quadriga_tools.o
+quadriga_lib/geo2cart.mexa64:   build/quadriga_tools.o
+quadriga_lib/version.mexa64:   build/quadriga_lib_combined.o

clean:
	- rm build/*
	- rm +quadriga_lib/*.manifest
	- rm +quadriga_lib/*.exp
	- rm +quadriga_lib/*.lib

tidy: clean 
	- rm +quadriga_lib/*.mex*