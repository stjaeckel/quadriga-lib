# This Makefile is for Linux / GCC environments

# Steps for compiling Quadriga-Lib (Linux):
# - Get required tools and libraries: make, cmake, g++
# - Compile HDF5 library by "make hdf5lib"
# - Set MATLAB path below (or leave empty)
# - Run "make"

# Set path to your MATLAB installation (optional):
# If left empty, the build script trys to autodetect the MATLAB path. 
# If none is found, MATLAB targets are not compiled.
MATLAB_PATH = #/usr/local/MATLAB/R2023a

# Leave empty for using system library (recommended)
# Static linking the HDF5 library may cause Octave to crash
hdf5_version      = # 1.14.2

# External libraries
# External libraries are located in the 'external' folder. Set the version numbers here.
# You need to compile the HDF5 and Catch2 libraries (e.g. using 'make hdf5lib' or 'make catch2lib' )
armadillo_version = 14.2.2
pugixml_version   = 1.13
catch2_version    = 3.4.0
pybind11_version  = 2.12.0

# The following sections should not require editing.

# Autodetect MATLAB path
ifeq ($(MATLAB_PATH),)
	MATLAB_PATH := $(shell readlink -f $(shell readlink -f $(shell which matlab)) | sed 's/\/bin\/matlab//')
endif

# Conditional compilation of MATLAB targets 
ifneq ($(MATLAB_PATH),)
ifeq ($(CUDA_PATH),)
	MATLAB_TARGETS = mex_matlab
else
	MATLAB_TARGETS = mex_matlab   mex_matlab_cuda
endif
endif

# Check if Octave is installed by trying to run mkoctfile
OCTAVE_VERSION := $(shell mkoctfile -v 2>/dev/null)

# Conditional compilation of Octave targets
ifneq ($(OCTAVE_VERSION),)
ifeq ($(CUDA_PATH),)
	OCTAVE_TARGETS = mex_octave
else
	OCTAVE_TARGETS = mex_octave   mex_octave_cuda
endif
endif

# Autodetect the Python include path
PYTHON_H = $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
ifeq ($(wildcard $(PYTHON_H)/Python.h),)
    PYTHON_H =
else
	PYTHON_EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)
	PYTHON_TARGET = lib/quadriga_lib$(PYTHON_EXTENSION_SUFFIX)
endif

# Compilers
CC   = g++
MEX  = $(MATLAB_PATH)/bin/mex
OCT  = mkoctfile

# Headers and Libraries
ARMA_H      = external/armadillo-$(armadillo_version)/include
PUGIXML_H   = external/pugixml-$(pugixml_version)/src
CATCH2      = external/Catch2-$(catch2_version)-Linux
PYBIND11_H  = external/pybind11-$(pybind11_version)/include

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

# Compiler flags (remove -Wextra to suppress unused variable warnings)
CCFLAGS     = -std=c++17 -O3 -fPIC -Wall -Wconversion -Wpedantic #-Wextra  

# Compilations targets
.PHONY: dirs
all:        
	@$(MAKE) dirs
	@$(MAKE) lib/quadriga_lib.a   $(CUDA_A)   $(MATLAB_TARGETS)   $(OCTAVE_TARGETS)   $(PYTHON_TARGET)

src     	= $(wildcard src/*.cpp)
api_mex     = $(wildcard api_mex/*.cpp)
api_python  = $(wildcard api_python/*.cpp)
tests 		= $(wildcard tests/catch2_tests/*.cpp)

dirs:
	mkdir -p build
	mkdir -p lib
	mkdir -p +quadriga_lib
	mkdir -p release

mex_matlab:      $(api_mex:api_mex/%.cpp=+quadriga_lib/%.mexa64)
mex_octave:      $(api_mex:api_mex/%.cpp=+quadriga_lib/%.mex)
mex_docu:        $(api_mex:api_mex/%.cpp=+quadriga_lib/%.m)

test:   all   tests/test_bin
	tests/test_bin
ifneq ($(OCTAVE_VERSION),)
	octave --eval "cd tests; quadriga_lib_mex_tests;"
endif
ifneq ($(PYTHON_TARGET),)
	pytest tests/python_tests -x -s
endif

test_catch2:    lib/quadriga_lib.a   tests/test_bin
	tests/test_bin

tests/test_bin:   tests/quadriga_lib_catch2_tests.cpp   lib/quadriga_lib.a   $(tests)
	$(CC) -std=c++17 $< lib/quadriga_lib.a -o $@ -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -lCatch2 -lgomp -ldl $(HDF5_DYN)

tests/test_cuda_bin:   tests/quadriga_lib_catch2_cuda_tests.cpp   lib/quadriga_lib.a   $(CUDA_A)   $(tests)
	$(CC) -std=c++17 $< lib/quadriga_lib.a $(CUDA_A) -o $@ -Iinclude -I$(ARMA_H) -I$(CATCH2)/include -L$(CATCH2)/lib -lCatch2 -lgomp -ldl $(HDF5_DYN) $(NV_LIB)

# Individual quadriga-lib objects
build/calc_diffraction_gain.o:   src/calc_diffraction_gain.cpp   include/quadriga_tools.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) 

build/qd_arrayant.o:   src/qd_arrayant.cpp   include/quadriga_arrayant.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_arrayant_qdant.o:   src/qd_arrayant_qdant.cpp   src/qd_arrayant_functions.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(PUGIXML_H)

build/qd_arrayant_interpolate.o:   src/qd_arrayant_interpolate.cpp   src/qd_arrayant_functions.hpp
	$(CC) -fopenmp $(CCFLAGS)  -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_arrayant_generate.o:   src/qd_arrayant_generate.cpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_arrayant_chan_spherical.o:   src/qd_arrayant_chan_spherical.cpp
	$(CC) -fopenmp $(CCFLAGS)  -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_arrayant_chan_planar.o:   src/qd_arrayant_chan_planar.cpp
	$(CC) -fopenmp $(CCFLAGS)  -c $< -o $@ -I src -I include -I $(ARMA_H)

build/baseband_freq_response.o:   src/baseband_freq_response.cpp   include/quadriga_channel.hpp
	$(CC) -mavx2 -mfma -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_channel.o:   src/qd_channel.cpp   include/quadriga_channel.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(HDF5_H)

build/quadriga_lib.o:   src/quadriga_lib.cpp   include/quadriga_lib.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/quadriga_tools.o:   src/quadriga_tools.cpp   include/quadriga_tools.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) 

build/ray_mesh_interact.o:   src/ray_mesh_interact.cpp   include/quadriga_tools.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/ray_point_intersect.o:   src/ray_point_intersect.cpp   include/quadriga_tools.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/ray_triangle_intersect.o:   src/ray_triangle_intersect.cpp   include/quadriga_tools.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

# AVX2 library files
build/quadriga_lib_test_avx.o:   src/quadriga_lib_test_avx.cpp   src/quadriga_lib_test_avx.hpp
	$(CC) -mavx2 -mfma $(CCFLAGS) -c $< -o $@ -I src 

build/ray_triangle_intersect_avx2.o:   src/ray_triangle_intersect_avx2.cpp   src/ray_triangle_intersect_avx2.hpp
	$(CC) -mavx2 -mfma -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I $(ARMA_H)

build/ray_point_intersect_avx2.o:   src/ray_point_intersect_avx2.cpp   src/ray_point_intersect_avx2.hpp
	$(CC) -mavx2 -mfma -fopenmp $(CCFLAGS) -c $< -o $@ -I src

build/baseband_freq_response_avx2.o:   src/baseband_freq_response_avx2.cpp   src/baseband_freq_response_avx2.hpp
	$(CC) -mavx2 -mfma -fopenmp $(CCFLAGS) -c $< -o $@ -I src

# Archive file for static linking
build/libhdf5.a:
	cp external/hdf5-$(hdf5_version)-Linux/lib/libhdf5.a build/
	( cd build/ && ar x libhdf5.a && cd .. )

lib/quadriga_lib.a:   $(HDF5_STATIC)   build/quadriga_lib.o  build/quadriga_lib_test_avx.o  build/qd_arrayant.o  build/qd_channel.o   \
                      build/quadriga_tools.o   build/qd_arrayant_generate.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   \
					  build/qd_arrayant_chan_spherical.o   build/qd_arrayant_chan_planar.o   \
					  build/calc_diffraction_gain.o   build/ray_triangle_intersect.o  build/ray_triangle_intersect_avx2.o   build/ray_mesh_interact.o \
					  build/ray_point_intersect.o   build/baseband_freq_response.o  build/ray_point_intersect_avx2.o   \
					  build/baseband_freq_response_avx2.o
		ar rcs $@ $^ $(HDF5_OBJ)

build/%_link.o:   build/%.o
	$(NVCC) -dlink $< -o $@

# Python interface
python: 
	@$(MAKE) dirs
	@$(MAKE) $(PYTHON_TARGET)

lib/quadriga_lib$(PYTHON_EXTENSION_SUFFIX):  api_python/python_main.cpp   lib/quadriga_lib.a $(api_python)
	$(CC) -shared $(CCFLAGS) $< lib/quadriga_lib.a -o $@ -I include -I $(PYBIND11_H) -I $(PYTHON_H) -I $(ARMA_H) -lgomp -ldl $(HDF5_DYN)

# Dependencies
dep_quadriga_tools = build/quadriga_tools.o   build/ray_triangle_intersect.o   build/ray_triangle_intersect_avx2.o
dep_arrayant = build/qd_arrayant.o   build/qd_arrayant_generate.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   \
				build/qd_arrayant_chan_spherical.o   build/qd_arrayant_chan_planar.o   $(dep_quadriga_tools)

# MEX MATLAB interface
+quadriga_lib/arrayant_calc_directivity.mexa64:   $(dep_arrayant)
+quadriga_lib/arrayant_combine_pattern.mexa64:    $(dep_arrayant)
+quadriga_lib/arrayant_generate.mexa64:           $(dep_arrayant)
+quadriga_lib/arrayant_interpolate.mexa64:        $(dep_arrayant)
+quadriga_lib/arrayant_qdant_read.mexa64:         $(dep_arrayant)
+quadriga_lib/arrayant_qdant_write.mexa64:        $(dep_arrayant)
+quadriga_lib/arrayant_rotate_pattern.mexa64:     $(dep_arrayant)
+quadriga_lib/baseband_freq_response.mexa64:      build/baseband_freq_response.o   build/baseband_freq_response_avx2.o
+quadriga_lib/calc_diffraction_gain.mexa64:       build/calc_diffraction_gain.o   $(dep_quadriga_tools)   build/ray_mesh_interact.o
+quadriga_lib/calc_rotation_matrix.mexa64:        $(dep_quadriga_tools)
+quadriga_lib/cart2geo.mexa64:                    $(dep_quadriga_tools)
+quadriga_lib/channel_export_obj_file.mexa64:     build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/generate_diffraction_paths.mexa64:  $(dep_quadriga_tools)
+quadriga_lib/geo2cart.mexa64:                    $(dep_quadriga_tools)
+quadriga_lib/get_channels_planar.mexa64:         $(dep_arrayant)
+quadriga_lib/get_channels_spherical.mexa64:      $(dep_arrayant)
+quadriga_lib/hdf5_create_file.mexa64:            build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_channel.mexa64:           build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_dset.mexa64:              build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_dset_names.mexa64:        build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_layout.mexa64:            build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_reshape_layout.mexa64:         build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_write_channel.mexa64:          build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_write_dset.mexa64:             build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_version.mexa64:                build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/icosphere.mexa64:                   $(dep_quadriga_tools)
+quadriga_lib/interp.mexa64:                      $(dep_quadriga_tools)
+quadriga_lib/obj_file_read.mexa64:               $(dep_quadriga_tools)
+quadriga_lib/point_cloud_aabb.mexa64:            $(dep_quadriga_tools)
+quadriga_lib/point_cloud_segmentation.mexa64:    $(dep_quadriga_tools)
+quadriga_lib/ray_mesh_interact.mexa64:           build/ray_mesh_interact.o
+quadriga_lib/ray_point_intersect.mexa64:         build/ray_point_intersect.o   build/ray_point_intersect_avx2.o   $(dep_quadriga_tools)
+quadriga_lib/ray_triangle_intersect.mexa64:      build/ray_triangle_intersect.o   build/ray_triangle_intersect_avx2.o
+quadriga_lib/subdivide_triangles.mexa64:         $(dep_quadriga_tools)
+quadriga_lib/triangle_mesh_aabb.mexa64:          $(dep_quadriga_tools)
+quadriga_lib/triangle_mesh_segmentation.mexa64:  $(dep_quadriga_tools)
+quadriga_lib/version.mexa64:                     build/quadriga_lib.o   build/quadriga_lib_test_avx.o

+quadriga_lib/%.mexa64:   api_mex/%.cpp
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -lgomp $(HDF5_DYN)

+quadriga_lib/%.mexa64:   mex_cuda/%.cpp
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -lgomp $(HDF5_DYN) $(NV_LIB)

# MEX Ocate interface
+quadriga_lib/arrayant_calc_directivity.mex:   $(dep_arrayant)
+quadriga_lib/arrayant_combine_pattern.mex:    $(dep_arrayant)
+quadriga_lib/arrayant_generate.mex:           $(dep_arrayant)
+quadriga_lib/arrayant_interpolate.mex:        $(dep_arrayant)
+quadriga_lib/arrayant_qdant_read.mex:         $(dep_arrayant)
+quadriga_lib/arrayant_qdant_write.mex:        $(dep_arrayant)
+quadriga_lib/arrayant_rotate_pattern.mex:     $(dep_arrayant)
+quadriga_lib/baseband_freq_response.mex:      build/baseband_freq_response.o   build/baseband_freq_response_avx2.o
+quadriga_lib/calc_diffraction_gain.mex:       build/calc_diffraction_gain.o   $(dep_quadriga_tools)   build/ray_mesh_interact.o
+quadriga_lib/calc_rotation_matrix.mex:        $(dep_quadriga_tools)
+quadriga_lib/cart2geo.mex:                    $(dep_quadriga_tools)
+quadriga_lib/channel_export_obj_file.mex:     build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/generate_diffraction_paths.mex:  $(dep_quadriga_tools)
+quadriga_lib/geo2cart.mex:                    $(dep_quadriga_tools)
+quadriga_lib/get_channels_planar.mex:         $(dep_arrayant)
+quadriga_lib/get_channels_spherical.mex:      $(dep_arrayant)
+quadriga_lib/hdf5_create_file.mex:            build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_channel.mex:           build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_dset.mex:              build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_dset_names.mex:        build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_read_layout.mex:            build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_reshape_layout.mex:         build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_write_channel.mex:          build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_write_dset.mex:             build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/hdf5_version.mex:                build/qd_channel.o   $(dep_quadriga_tools)   $(HDF5_STATIC)
+quadriga_lib/icosphere.mex:                   $(dep_quadriga_tools)
+quadriga_lib/interp.mex:                      $(dep_quadriga_tools)
+quadriga_lib/obj_file_read.mex:               $(dep_quadriga_tools)
+quadriga_lib/point_cloud_aabb.mex:            $(dep_quadriga_tools)
+quadriga_lib/point_cloud_segmentation.mex:    $(dep_quadriga_tools)
+quadriga_lib/ray_mesh_interact.mex:           build/ray_mesh_interact.o
+quadriga_lib/ray_point_intersect.mex:         build/ray_point_intersect.o   build/ray_point_intersect_avx2.o   $(dep_quadriga_tools)
+quadriga_lib/ray_triangle_intersect.mex:      build/ray_triangle_intersect.o   build/ray_triangle_intersect_avx2.o
+quadriga_lib/subdivide_triangles.mex:         $(dep_quadriga_tools)
+quadriga_lib/triangle_mesh_aabb.mex:          $(dep_quadriga_tools)
+quadriga_lib/triangle_mesh_segmentation.mex:  $(dep_quadriga_tools)
+quadriga_lib/version.mex:                     build/quadriga_lib.o   build/quadriga_lib_test_avx.o

+quadriga_lib/%.mex:   api_mex/%.cpp
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) -s 

+quadriga_lib/%.mex:   mex_cuda/%.cpp
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) $(NV_LIB) -s

# Documentation of MEX files
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
	- rm +quadriga_lib/*.manifest
	- rm +quadriga_lib/*.exp
	- rm +quadriga_lib/*.lib
	- rm *.obj
	- rm tests/test_bin
	- rm tests/test_cuda_bin
	- rm -rf build
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
	- rm -rf +quadriga_lib
	- rm -rf release
	- rm -rf lib
	- rm tests/test.exe
	
build/quadriga-lib-version:   src/version.cpp   lib/quadriga_lib.a
	$(CC) -std=c++17 $^ -o $@ -I src -I include -I $(ARMA_H)

release:  all   build/quadriga-lib-version
	- mkdir release
	tar czf release/quadrigalib-v$(shell build/quadriga-lib-version)-Ubuntu-$(shell lsb_release -r -s)-amd64.tar.gz \
		+quadriga_lib/*.mex +quadriga_lib/*.mexa64 +quadriga_lib/*.m include lib/*.a

package:  all   build/quadriga-lib-version
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
	cp GNUmakefile release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp LICENSE release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp Makefile release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp README.md release/quadriga_lib-$(shell build/quadriga-lib-version)/
	- rm -rf release/quadriga_lib-$(shell build/quadriga-lib-version)/tests/afl
	( cd release && zip -r quadriga_lib-$(shell build/quadriga-lib-version).zip quadriga_lib-$(shell build/quadriga-lib-version)/ )
	- rm -rf release/quadriga_lib-$(shell build/quadriga-lib-version)
