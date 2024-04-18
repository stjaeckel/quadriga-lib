# This Makefile is for Linux / GCC environments

# Steps for compiling Quadriga-Lib (Linux):
# - Get required tools and libraries: make, cmake, g++
# - Compile HDF5 library by "make hdf5lib"
# - Set MATLAB path below (or leave empty)
# - Run "make"

# Set path to your MATLAB installation (optional):
# Leave this empty if you don't want to use MATLAB (you can still use Octave).
MATLAB_PATH = /usr/local/MATLAB/R2021a

# External libraries
# External libraries are located in the 'external' folder. Set the version numbers here.
# You need to compile the HDF5 and Catch2 libraries (e.g. using 'make hdf5lib' or 'make catch2lib' )
armadillo_version = 12.6.3
pugixml_version   = 1.13
catch2_version    = 3.4.0

# Leave empty for using system library (recommended)
# Static linking the HDF5 library may cause Octave to crash
hdf5_version      = # 1.14.2

# The following sections should not require editing.

# Conditional compilation of MATLAB targets (check if MATLAB path is set)
ifeq ($(MATLAB_PATH),)
	MATLAB_TARGETS =
else
	MATLAB_TARGETS = mex_matlab
endif

# Check if Octave is installed by trying to run mkoctfile
OCTAVE_VERSION := $(shell mkoctfile -v 2>/dev/null)

# Conditional compilation of Octave targets
ifeq ($(OCTAVE_VERSION),)
	OCTAVE_TARGETS =
else
	OCTAVE_TARGETS = mex_octave
endif

# Compilers
CC   = g++
MEX  = $(MATLAB_PATH)/bin/mex
OCT  = mkoctfile

# Headers and Libraries
ARMA_H      = external/armadillo-$(armadillo_version)/include
PUGIXML_H   = external/pugixml-$(pugixml_version)/src
CATCH2      = external/Catch2-$(catch2_version)-Linux

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
endif

# Compiler flags
CCFLAGS     = -std=c++17 -O3 -fPIC -Wall -Wextra -Wpedantic -Wconversion #-g

# Compilations targets
.PHONY: dirs
all:        
	@$(MAKE) dirs
	@$(MAKE) lib/quadriga_lib.a   $(MATLAB_TARGETS)   $(OCTAVE_TARGETS)

src     	= $(wildcard src/*.cpp)
mex         = $(wildcard mex/*.cpp)
tests 		= $(wildcard tests/catch2_tests/*.cpp)

dirs:
	mkdir -p build
	mkdir -p lib
	mkdir -p +quadriga_lib

mex_matlab:  $(mex:mex/%.cpp=+quadriga_lib/%.mexa64)
mex_octave:  $(mex:mex/%.cpp=+quadriga_lib/%.mex)
mex_docu:    $(mex:mex/%.cpp=+quadriga_lib/%.m)

test:   tests/test_bin   #mex_octave
	octave --eval "cd tests; quadriga_lib_mex_tests;"
	tests/test_bin
	
tests/test_bin:   tests/quadriga_lib_catch2_tests.cpp   lib/quadriga_lib.a   $(tests)
	$(CC) -std=c++17 $< lib/quadriga_lib.a -o $@ -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -lCatch2 -lgomp -ldl $(HDF5_DYN)

# Individual quadriga-lib objects
build/qd_arrayant.o:   src/qd_arrayant.cpp   include/quadriga_arrayant.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_arrayant_qdant.o:   src/qd_arrayant_qdant.cpp   src/qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(PUGIXML_H)

build/qd_arrayant_interpolate.o:   src/qd_arrayant_interpolate.cpp   src/qd_arrayant_interpolate.hpp
	$(CC) -fopenmp $(CCFLAGS)  -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_channel.o:   src/qd_channel.cpp   include/quadriga_channel.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(HDF5_H)

build/quadriga_lib.o:   src/quadriga_lib.cpp   include/quadriga_lib.hpp
	$(CC) -mavx2 -mfma $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/quadriga_tools.o:   src/quadriga_tools.cpp   include/quadriga_tools.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) 

build/calc_diffraction_gain.o:   src/calc_diffraction_gain.cpp   include/quadriga_tools.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) 

build/ray_triangle_intersect.o:   src/ray_triangle_intersect.cpp   include/quadriga_tools.hpp
	$(CC) -mavx2 -mfma -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/ray_mesh_interact.o:   src/ray_mesh_interact.cpp   include/quadriga_tools.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

# Archive file for static linking
build/libhdf5.a:
	cp external/hdf5-$(hdf5_version)-Linux/lib/libhdf5.a build/
	( cd build/ && ar x libhdf5.a && cd .. )

lib/quadriga_lib.a:   $(HDF5_STATIC)   build/quadriga_lib.o  build/qd_arrayant.o  build/qd_channel.o   \
                      build/quadriga_tools.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   \
					  build/calc_diffraction_gain.o   build/ray_triangle_intersect.o   build/ray_mesh_interact.o
		ar rcs $@ $^ $(HDF5_OBJ)

# MEX MATLAB interface
+quadriga_lib/arrayant_calc_directivity.mexa64:   build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_combine_pattern.mexa64:    build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_generate.mexa64:           build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_interpolate.mexa64:        build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_qdant_read.mexa64:         build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_qdant_write.mexa64:        build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_rotate_pattern.mexa64:     build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/calc_diffraction_gain.mexa64:       build/calc_diffraction_gain.o   build/quadriga_tools.o   build/ray_triangle_intersect.o   build/ray_mesh_interact.o
+quadriga_lib/calc_rotation_matrix.mexa64:        build/quadriga_tools.o
+quadriga_lib/cart2geo.mexa64:                    build/quadriga_tools.o
+quadriga_lib/generate_diffraction_paths.mexa64:  build/quadriga_tools.o
+quadriga_lib/geo2cart.mexa64:                    build/quadriga_tools.o
+quadriga_lib/get_channels_planar.mexa64:         build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/get_channels_spherical.mexa64:      build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/hdf5_create_file.mexa64:            build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_channel.mexa64:           build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_dset.mexa64:              build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_dset_names.mexa64:        build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_layout.mexa64:            build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_reshape_layout.mexa64:         build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_write_channel.mexa64:          build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_write_dset.mexa64:             build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_version.mexa64:                build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/icosphere.mexa64:                   build/quadriga_tools.o
+quadriga_lib/interp.mexa64:                      build/quadriga_tools.o
+quadriga_lib/obj_file_read.mexa64:               build/quadriga_tools.o
+quadriga_lib/ray_mesh_interact.mexa64:           build/ray_mesh_interact.o
+quadriga_lib/ray_triangle_intersect.mexa64:      build/ray_triangle_intersect.o
+quadriga_lib/subdivide_triangles.mexa64:         build/quadriga_tools.o
+quadriga_lib/triangle_mesh_aabb.mexa64:          build/quadriga_tools.o
+quadriga_lib/triangle_mesh_segmentation.mexa64:  build/quadriga_tools.o
+quadriga_lib/version.mexa64:                     build/quadriga_lib.o

+quadriga_lib/%.mexa64:   mex/%.cpp
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -lgomp $(HDF5_DYN)

# MEX Ocate interface
+quadriga_lib/arrayant_calc_directivity.mex:   build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_combine_pattern.mex:    build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_generate.mex:           build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_interpolate.mex:        build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_qdant_read.mex:         build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_qdant_write.mex:        build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_rotate_pattern.mex:     build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/calc_diffraction_gain.mex:       build/calc_diffraction_gain.o   build/quadriga_tools.o   build/ray_triangle_intersect.o   build/ray_mesh_interact.o
+quadriga_lib/calc_rotation_matrix.mex:        build/quadriga_tools.o
+quadriga_lib/cart2geo.mex:                    build/quadriga_tools.o
+quadriga_lib/generate_diffraction_paths.mex:  build/quadriga_tools.o
+quadriga_lib/geo2cart.mex:                    build/quadriga_tools.o
+quadriga_lib/get_channels_planar.mex:         build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/get_channels_spherical.mex:      build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/hdf5_create_file.mex:            build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_channel.mex:           build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_dset.mex:              build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_dset_names.mex:        build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_layout.mex:            build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_reshape_layout.mex:         build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_write_channel.mex:          build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_write_dset.mex:             build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/hdf5_version.mex:                build/qd_channel.o   $(HDF5_STATIC)
+quadriga_lib/icosphere.mex:                   build/quadriga_tools.o
+quadriga_lib/interp.mex:                      build/quadriga_tools.o
+quadriga_lib/obj_file_read.mex:               build/quadriga_tools.o
+quadriga_lib/ray_mesh_interact.mex:           build/ray_mesh_interact.o
+quadriga_lib/ray_triangle_intersect.mex:      build/ray_triangle_intersect.o
+quadriga_lib/subdivide_triangles.mex:         build/quadriga_tools.o
+quadriga_lib/triangle_mesh_aabb.mex:          build/quadriga_tools.o
+quadriga_lib/triangle_mesh_segmentation.mex:  build/quadriga_tools.o
+quadriga_lib/version.mex:                     build/quadriga_lib.o

+quadriga_lib/%.mex:   mex/%.cpp
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) -s 

# Documentation of MEX files
+quadriga_lib/%.m:   mex/%.cpp
	rm -f $@
	python3 tools/extract_matlab_comments.py $< $@

documentation:   dirs   mex_docu
	mkdir -p +quadriga_lib
	python3 tools/extract_html.py html_docu/index.html tools/html_parts/index.html.part
	python3 tools/extract_html.py html_docu/mex_api.html tools/html_parts/mex_api.html.part mex/ 
	python3 tools/extract_html.py html_docu/cpp_api.html tools/html_parts/cpp_api.html.part src/ 

# Maintainance section
hdf5lib:
	- rm -rf external/hdf5-$(hdf5_version)
	- rm -rf external/hdf5-$(hdf5_version)-Linux
	unzip external/hdf5-$(hdf5_version).zip -d external/
	( cd external/hdf5-$(hdf5_version) && ./configure CFLAGS="-O3 -march=native -fPIC" \
	   --enable-tools=no --enable-tests=no --enable-shared=no --enable-hl=no --enable-optimization=high --with-zlib=no --enable-threadsafe=no)
	( cd external/hdf5-$(hdf5_version) && make -j8 && make install )
	mv external/hdf5-$(hdf5_version)/hdf5 external/hdf5-$(hdf5_version)-Linux
	- rm -rf external/hdf5-$(hdf5_version)

catch2lib:
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

clean:
	- rm build/*
	- rm lib/*
	- rm +quadriga_lib/*.mex*
	- rm +quadriga_lib/*.manifest
	- rm +quadriga_lib/*.exp
	- rm +quadriga_lib/*.lib
	- rm *.obj
	- rm tests/test_bin
	- rm tests/test.exe
	- rm -rf build

tidy: clean 
	- rm -rf external/Catch2-$(catch2_version)-Linux
	- rm -rf external/hdf5-$(hdf5_version)-Linux
	- rm +quadriga_lib/*.m
	- rm -rf +quadriga_lib
	- rm -rf lib

build/quadriga-lib-version:   src/version.cpp   lib/quadriga_lib.a
	$(CC) -std=c++17 $^ -o $@ -I src -I include -I $(ARMA_H)

release:  all   build/quadriga-lib-version
	- mkdir release
	tar czf release/quadrigalib-v$(shell build/quadriga-lib-version)-Ubuntu-$(shell lsb_release -r -s)-amd64.tar.gz \
		+quadriga_lib/*.mex +quadriga_lib/*.mexa64 +quadriga_lib/*.m include lib/*.a