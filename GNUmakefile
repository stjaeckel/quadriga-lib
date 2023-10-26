# This Makefile is for Linux / GCC environments

# Compilers
CC    = g++
MEX   = /usr/local/MATLAB/R2021a/bin/mex
OCT   = mkoctfile

all:        dirs   mex_octave  mex_matlab   mex_docu

# External libraries
hdf5version    = 1.14.2
catch2version  = 3.4.0

ARMA_H      = external/armadillo-12.6.3/include
PUGIXML_H   = external/pugixml-1.13/src
CATCH2      = external/Catch2-$(catch2version)-Linux
HDF5_H      = external/hdf5-$(hdf5version)-Linux/include
HDF5_LIB    = external/hdf5-$(hdf5version)-Linux/lib

# Configurations
CCFLAGS     = -std=c++17 -O3 -fPIC -Wall -Wextra -Wpedantic -Wconversion #   -Werror  #   # -Wall#-Wl,--gc-sections -Wall -Wextra -Wpedantic -Wconversion

# Sourcees
src     	= $(wildcard src/*.cpp)
mex         = $(wildcard mex/*.cpp)
tests 		= $(wildcard tests/catch2_tests/*.cpp)

.PHONY: 	clean all tidy mex_matlab mex_octave
.SECONDARY: $(src:src/%.cpp=build/%.o)

dirs:
	- mkdir build
	- mkdir lib
	- mkdir +quadriga_lib

mex_matlab:  $(mex:mex/%.cpp=+quadriga_lib/%.mexa64)
mex_octave:  $(mex:mex/%.cpp=+quadriga_lib/%.mex)
mex_docu:    $(mex:mex/%.cpp=+quadriga_lib/%.m)

test:   tests/test_bin   #mex_octave
#	octave --eval "cd tests; quadriga_lib_mex_tests;"
	tests/test_bin
	
tests/test_bin:   tests/quadriga_lib_catch2_tests.cpp   lib/quadriga_lib.a   $(tests)
	$(CC) -std=c++17 $< lib/quadriga_lib.a -o $@ -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -lCatch2 -lgomp -ldl

# Individual quadriga-lib objects
build/qd_arrayant.o:   src/qd_arrayant.cpp   include/quadriga_arrayant.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_arrayant_qdant.o:   src/qd_arrayant_qdant.cpp   src/qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(PUGIXML_H)

build/qd_arrayant_interpolate.o:   src/qd_arrayant_interpolate.cpp   src/qd_arrayant_interpolate.hpp
	$(CC) $(CCFLAGS) -fopenmp -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_channel.o:   src/qd_channel.cpp   include/quadriga_channel.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(HDF5_H)

build/quadriga_tools.o:   src/quadriga_tools.cpp   include/quadriga_tools.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) 

build/quadriga_lib.o:   src/quadriga_lib.cpp   include/quadriga_lib.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

# Archive file for static linking
lib/quadriga_lib.a:   build/quadriga_lib.o  build/qd_arrayant.o  build/qd_channel.o   \
                      build/quadriga_tools.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o
	cp $(HDF5_LIB)/libhdf5.a build/
	( cd build/ && ar x libhdf5.a && cd .. )
	ar rcs $@ $^ build/H5*.o

# MEX MATLAB interface
+quadriga_lib/%.mexa64:   mex/%.cpp   lib/quadriga_lib.a
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -lgomp

# MEX Ocate interface
+quadriga_lib/%.mex:   mex/%.cpp   lib/quadriga_lib.a
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) -s

# Documentation of MEX files
+quadriga_lib/%.m:   mex/%.cpp
	rm -f $@
	python3 tools/extract_matlab_comments.py $< $@

# Maintainance section
hdf5lib:
	- rm -rf external/hdf5-$(hdf5version)
	- rm -rf external/hdf5-$(hdf5version)-Linux
	unzip external/hdf5-$(hdf5version).zip -d external/
	( cd external/hdf5-$(hdf5version) && ./configure CFLAGS="-O3 -march=native -fPIC" \
	   --enable-tools=no --enable-tests=no --enable-shared=no --enable-hl=no --enable-optimization=high --with-zlib=no)
	( cd external/hdf5-$(hdf5version) && make -j8 && make install )
	mv external/hdf5-$(hdf5version)/hdf5 external/hdf5-$(hdf5version)-Linux
	- rm -rf external/hdf5-$(hdf5version)

catch2lib:
	- rm -rf external/build
	- rm -rf external/Catch2-$(catch2version)
	- rm -rf external/Catch2-$(catch2version)-Linux
	unzip external/Catch2-$(catch2version).zip -d external/
	mkdir external/build
	cmake -S external/Catch2-$(catch2version) -B external/build
	( cd external/build && make -j8 && make package )
	tar -xzf external/build/Catch2-$(catch2version)-Linux.tar.gz -C external/
	- rm -rf external/build
	- rm -rf external/Catch2-$(catch2version)

clean:
	- rm quadrigalib*.tar.gz
	- rm build/*
	- rm lib/*
	- rm +quadriga_lib/*.mex*
	- rm +quadriga_lib/*.manifest
	- rm +quadriga_lib/*.exp
	- rm +quadriga_lib/*.lib
	- rm *.obj
	- rm tests/test_bin

tidy: clean 
	- rm -rf external/Catch2-$(catch2version)-Linux
	- rm -rf external/hdf5-$(hdf5version)-Linux

build/quadriga-lib-version:   src/version.cpp   lib/quadriga_lib.a
	$(CC) -std=c++17 $^ -o $@ -I src -I include -I $(ARMA_H)

release:  all   build/quadriga-lib-version
	- mkdir release
	tar czf release/quadrigalib-v$(shell build/quadriga-lib-version)-Ubuntu-$(shell lsb_release -r -s)-amd64.tar.gz \
		+quadriga_lib/*.mex +quadriga_lib/*.mexa64 +quadriga_lib/*.m include lib/*.a