# This Makefile is for Linux / GCC environments

# Compilers
CC    = g++
MEX   = /usr/local/MATLAB/R2021a/bin/mex
OCT   = mkoctfile

all:        mex_octave   mex_matlab

# External headers
ARMA_H      = external/armadillo-11.4.2/include
PUGIXML_H   = external/pugixml-1.13/src

# External pre-compiled libraries
CATCH2_LIB  = -I external/Catch2-3.3.2-Linux/include -L external/Catch2-3.3.2-Linux/lib -lCatch2
ARMA_LIB    = -I external/armadillo-11.4.2-Linux/include -L external/armadillo-11.4.2-Linux/lib -larmadillo


# Configurations
CCFLAGS     = -std=c++17 -fPIC -O3 -fopenmp -Wall #-Werror #-Wl,--gc-sections -Wall -Wextra -Wpedantic
LIBS        = -lgomp #-lblas

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
	$(CC) $(CCFLAGS) $< lib/quadriga_lib.a -o $@ -I include $(ARMA_LIB) $(CATCH2_LIB)

# Individual Library files
build/qd_arrayant_qdant.o:   src/qd_arrayant_qdant.cpp   src/qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ $(ARMA_LIB) -I $(PUGIXML_H) -I src -I include $(LIBS)

build/%.o:   src/%.cpp
	$(CC) $(CCFLAGS) -c $< -o $@ $(ARMA_LIB) -I src -I include $(LIBS)

lib/quadriga_lib.a:   build/quadriga_lib.o   build/quadriga_tools.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o
	ar rcs $@ $^

# MEX interface files
+quadriga_lib/%.mexa64:   mex/%.cpp   lib/quadriga_lib.a
	$(MEX) -outdir +quadriga_lib $^ -I$(ARMA_H) -Isrc -Iinclude $(LIBS)

+quadriga_lib/%.mex:   mex/%.cpp   lib/quadriga_lib.a
	$(OCT) --mex -o $@ $^ -I$(ARMA_H) -Isrc -Iinclude $(LIBS) -s

# List of additional dependencies
build/quadriga_lib.o:   include/quadriga_lib.hpp
build/quadriga_tools.o:   include/quadriga_tools.hpp
build/qd_arrayant_interpolate.o:   src/qd_arrayant_interpolate.hpp

clean:
	- rm build/*
	- rm +quadriga_lib/*.manifest
	- rm +quadriga_lib/*.exp
	- rm +quadriga_lib/*.lib
		- rm *.obj

tidy: clean 
	- rm +quadriga_lib/*.mex*
	- rm lib/*