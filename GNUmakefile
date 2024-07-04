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

# Set path to your CUDA installation
# Leave this empty if you don't want to use GPU acceleration
# You can get CUDA from: https://developer.nvidia.com/cuda-toolkit
#  !!! THERE ARE CURRENTLY NO CUDA EXENSIONS AVAILABLE  !!!
CUDA_PATH = #/usr/local/cuda-12.4

# If needed, adjust the NVIDIA compute capability (50 should run on most modern GPUs). 
# Adjusting the value to match your GPUs capability may improve performance and load times, but is not required.
# Minimum supported capability for CUDA-11 is 35 and for CUDA-12 it is 50.
# For more info: https://developer.nvidia.com/cuda-gpus
COMPUTE_CAPABILITY = 50

# Leave empty for using system library (recommended)
# Static linking the HDF5 library may cause Octave to crash
hdf5_version      = # 1.14.2

# External libraries
# External libraries are located in the 'external' folder. Set the version numbers here.
# You need to compile the HDF5 and Catch2 libraries (e.g. using 'make hdf5lib' or 'make catch2lib' )
armadillo_version = 12.8.3
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

# Conditional compilation of CUDA targets (check if CUDA path is set)
ifneq ($(CUDA_PATH),)
	CUDA_A     = lib/quadriga_cuda.a
	CUDA_TEST  = tests/test_cuda_bin
	NVCC       = $(CUDA_PATH)/bin/nvcc
	NV_LIB     = -L$(CUDA_PATH)/lib64 -lcudart
	NVCCFLAGS  = --std c++17 -ccbin=$(CC) --gpu-architecture=compute_$(COMPUTE_CAPABILITY) \
					--gpu-code=compute_$(COMPUTE_CAPABILITY) -Wno-deprecated-gpu-targets \
					-Xcompiler '-fPIC -fopenmp' -I$(CUDA_PATH)/include -lineinfo
endif

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

# Compiler flags
CCFLAGS     = -std=c++17 -O3 -fPIC -Wall -Wextra -Wpedantic -Wconversion #-g

# Compilations targets
.PHONY: dirs
all:        
	@$(MAKE) dirs
	@$(MAKE) lib/quadriga_lib.a   $(CUDA_A)   $(MATLAB_TARGETS)   $(OCTAVE_TARGETS)   $(PYTHON_TARGET)

src     	= $(wildcard src/*.cpp)
mex         = $(wildcard mex/*.cpp)
mex_cuda    = $(wildcard mex_cuda/*.cpp)
cpython     = $(wildcard cpython/*.cpp)
tests 		= $(wildcard tests/catch2_tests/*.cpp)

dirs:
	mkdir -p build
	mkdir -p lib
	mkdir -p +quadriga_lib
	mkdir -p release

mex_matlab:      $(mex:mex/%.cpp=+quadriga_lib/%.mexa64)
mex_matlab_cuda: $(mex_cuda:mex_cuda/%.cpp=+quadriga_lib/%.mexa64)
mex_octave:      $(mex:mex/%.cpp=+quadriga_lib/%.mex)
mex_octave_cuda: $(mex_cuda:mex_cuda/%.cpp=+quadriga_lib/%.mex)
mex_docu:        $(mex:mex/%.cpp=+quadriga_lib/%.m)

test:   all   tests/test_bin   $(CUDA_TEST)
	tests/test_bin
ifneq ($(OCTAVE_VERSION),)
	octave --eval "cd tests; quadriga_lib_mex_tests;"
endif
ifneq ($(CUDA_PATH),)
	tests/test_cuda_bin
endif
ifneq ($(PYTHON_TARGET),)
	pytest tests/python_tests
endif

tests/test_bin:   tests/quadriga_lib_catch2_tests.cpp   lib/quadriga_lib.a   $(tests)
	$(CC) -std=c++17 $< lib/quadriga_lib.a -o $@ -I include -I $(ARMA_H) -I $(CATCH2)/include -L $(CATCH2)/lib -lCatch2 -lgomp -ldl $(HDF5_DYN)

tests/test_cuda_bin:   tests/quadriga_lib_catch2_cuda_tests.cpp   lib/quadriga_lib.a   $(CUDA_A)   $(tests)
	$(CC) -std=c++17 $< lib/quadriga_lib.a $(CUDA_A) -o $@ -Iinclude -I$(ARMA_H) -I$(CATCH2)/include -L$(CATCH2)/lib -lCatch2 -lgomp -ldl $(HDF5_DYN) $(NV_LIB)

# Individual quadriga-lib objects
build/calc_diffraction_gain.o:   src/calc_diffraction_gain.cpp   include/quadriga_tools.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) 

build/qd_arrayant.o:   src/qd_arrayant.cpp   include/quadriga_arrayant.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_arrayant_qdant.o:   src/qd_arrayant_qdant.cpp   src/qd_arrayant_qdant.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(PUGIXML_H)

build/qd_arrayant_interpolate.o:   src/qd_arrayant_interpolate.cpp   src/qd_arrayant_interpolate.hpp
	$(CC) -fopenmp $(CCFLAGS)  -c $< -o $@ -I src -I include -I $(ARMA_H)

build/baseband_freq_response.o:   src/baseband_freq_response.cpp   include/quadriga_channel.hpp
	$(CC) -mavx2 -mfma -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/qd_channel.o:   src/qd_channel.cpp   include/quadriga_channel.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) -I $(HDF5_H)

build/quadriga_lib.o:   src/quadriga_lib.cpp   include/quadriga_lib.hpp
	$(CC) -mavx2 -mfma $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/quadriga_tools.o:   src/quadriga_tools.cpp   include/quadriga_tools.hpp
	$(CC) $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H) 

build/ray_mesh_interact.o:   src/ray_mesh_interact.cpp   include/quadriga_tools.hpp
	$(CC) -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/ray_point_intersect.o:   src/ray_point_intersect.cpp   include/quadriga_tools.hpp
	$(CC) -mavx2 -mfma -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/ray_triangle_intersect.o:   src/ray_triangle_intersect.cpp   include/quadriga_tools.hpp
	$(CC) -mavx2 -mfma -fopenmp $(CCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

build/get_CUDA_compute_capability.o:   src/get_CUDA_compute_capability.cu   include/quadriga_CUDA_tools.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I src -I include -I $(ARMA_H)

# Archive file for static linking
build/libhdf5.a:
	cp external/hdf5-$(hdf5_version)-Linux/lib/libhdf5.a build/
	( cd build/ && ar x libhdf5.a && cd .. )

lib/quadriga_lib.a:   $(HDF5_STATIC)   build/quadriga_lib.o  build/qd_arrayant.o  build/qd_channel.o   \
                      build/quadriga_tools.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   \
					  build/calc_diffraction_gain.o   build/ray_triangle_intersect.o   build/ray_mesh_interact.o \
					  build/ray_point_intersect.o   build/baseband_freq_response.o
		ar rcs $@ $^ $(HDF5_OBJ)

build/%_link.o:   build/%.o
	$(NVCC) -dlink $< -o $@

lib/quadriga_cuda.a:   build/get_CUDA_compute_capability.o   build/get_CUDA_compute_capability_link.o
	ar rcs $@ $^

# Python interface
python: 
	@$(MAKE) dirs
	@$(MAKE) $(PYTHON_TARGET)

lib/quadriga_lib$(PYTHON_EXTENSION_SUFFIX):  cpython/python_main.cpp   lib/quadriga_lib.a $(cpython)
	$(CC) -shared $(CCFLAGS) $< lib/quadriga_lib.a -o $@ -I include -I $(PYBIND11_H) -I $(PYTHON_H) -I $(ARMA_H) -lgomp -ldl $(HDF5_DYN)

# MEX MATLAB interface
+quadriga_lib/arrayant_calc_directivity.mexa64:   build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_combine_pattern.mexa64:    build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_generate.mexa64:           build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_interpolate.mexa64:        build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_qdant_read.mexa64:         build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_qdant_write.mexa64:        build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_rotate_pattern.mexa64:     build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/baseband_freq_response.mexa64:      build/baseband_freq_response.o
+quadriga_lib/calc_diffraction_gain.mexa64:       build/calc_diffraction_gain.o   build/quadriga_tools.o   build/ray_triangle_intersect.o   build/ray_mesh_interact.o
+quadriga_lib/calc_rotation_matrix.mexa64:        build/quadriga_tools.o
+quadriga_lib/cart2geo.mexa64:                    build/quadriga_tools.o
+quadriga_lib/generate_diffraction_paths.mexa64:  build/quadriga_tools.o
+quadriga_lib/geo2cart.mexa64:                    build/quadriga_tools.o
+quadriga_lib/get_channels_planar.mexa64:         build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/get_channels_spherical.mexa64:      build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/get_CUDA_compute_capability.mexa64: build/get_CUDA_compute_capability.o
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
+quadriga_lib/point_cloud_aabb.mexa64:            build/quadriga_tools.o
+quadriga_lib/point_cloud_segmentation.mexa64:    build/quadriga_tools.o
+quadriga_lib/ray_mesh_interact.mexa64:           build/ray_mesh_interact.o
+quadriga_lib/ray_point_intersect.mexa64:         build/ray_point_intersect.o   build/quadriga_tools.o
+quadriga_lib/ray_triangle_intersect.mexa64:      build/ray_triangle_intersect.o
+quadriga_lib/subdivide_triangles.mexa64:         build/quadriga_tools.o
+quadriga_lib/triangle_mesh_aabb.mexa64:          build/quadriga_tools.o
+quadriga_lib/triangle_mesh_segmentation.mexa64:  build/quadriga_tools.o
+quadriga_lib/version.mexa64:                     build/quadriga_lib.o

+quadriga_lib/%.mexa64:   mex/%.cpp
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -lgomp $(HDF5_DYN)

+quadriga_lib/%.mexa64:   mex_cuda/%.cpp
	$(MEX) CXXFLAGS="$(CCFLAGS)" -outdir +quadriga_lib $^ -Isrc -Iinclude -I$(ARMA_H) -lgomp $(HDF5_DYN) $(NV_LIB)

# MEX Ocate interface
+quadriga_lib/arrayant_calc_directivity.mex:   build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_combine_pattern.mex:    build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_generate.mex:           build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_interpolate.mex:        build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_qdant_read.mex:         build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_qdant_write.mex:        build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/arrayant_rotate_pattern.mex:     build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/baseband_freq_response.mex:      build/baseband_freq_response.o
+quadriga_lib/calc_diffraction_gain.mex:       build/calc_diffraction_gain.o   build/quadriga_tools.o   build/ray_triangle_intersect.o   build/ray_mesh_interact.o
+quadriga_lib/calc_rotation_matrix.mex:        build/quadriga_tools.o
+quadriga_lib/cart2geo.mex:                    build/quadriga_tools.o
+quadriga_lib/generate_diffraction_paths.mex:  build/quadriga_tools.o
+quadriga_lib/geo2cart.mex:                    build/quadriga_tools.o
+quadriga_lib/get_channels_planar.mex:         build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/get_channels_spherical.mex:      build/qd_arrayant.o   build/qd_arrayant_interpolate.o   build/qd_arrayant_qdant.o   build/quadriga_tools.o
+quadriga_lib/get_CUDA_compute_capability.mex: build/get_CUDA_compute_capability.o
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
+quadriga_lib/point_cloud_aabb.mex:            build/quadriga_tools.o
+quadriga_lib/point_cloud_segmentation.mex:    build/quadriga_tools.o
+quadriga_lib/ray_mesh_interact.mex:           build/ray_mesh_interact.o
+quadriga_lib/ray_point_intersect.mex:         build/ray_point_intersect.o   build/quadriga_tools.o
+quadriga_lib/ray_triangle_intersect.mex:      build/ray_triangle_intersect.o
+quadriga_lib/subdivide_triangles.mex:         build/quadriga_tools.o
+quadriga_lib/triangle_mesh_aabb.mex:          build/quadriga_tools.o
+quadriga_lib/triangle_mesh_segmentation.mex:  build/quadriga_tools.o
+quadriga_lib/version.mex:                     build/quadriga_lib.o

+quadriga_lib/%.mex:   mex/%.cpp
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) -s 

+quadriga_lib/%.mex:   mex_cuda/%.cpp
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -Isrc -Iinclude -I$(ARMA_H) $(NV_LIB) -s

# Documentation of MEX files
+quadriga_lib/%.m:   mex/%.cpp
	rm -f $@
	python3 tools/extract_matlab_comments.py $< $@

documentation:   dirs   mex_docu
	mkdir -p +quadriga_lib
	python3 tools/extract_html.py html_docu/index.html tools/html_parts/index.html.part
	python3 tools/extract_html.py html_docu/mex_api.html tools/html_parts/mex_api.html.part mex/ 
	python3 tools/extract_html.py html_docu/cpp_api.html tools/html_parts/cpp_api.html.part src/ 
	python3 tools/extract_html.py html_docu/python_api.html tools/html_parts/python_api.html.part cpython/ 
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
	
build/quadriga-lib-version:   src/version.cpp   lib/quadriga_lib.a
	$(CC) -std=c++17 $^ -o $@ -I src -I include -I $(ARMA_H)

release:  all   build/quadriga-lib-version
	- mkdir release
	tar czf release/quadrigalib-v$(shell build/quadriga-lib-version)-Ubuntu-$(shell lsb_release -r -s)-amd64.tar.gz \
		+quadriga_lib/*.mex +quadriga_lib/*.mexa64 +quadriga_lib/*.m include lib/*.a

package:  all  build/quadriga-lib-version
	- mkdir release
	- rm -rf release/quadriga_lib-$(shell build/quadriga-lib-version)
	- rm release/quadriga_lib-$(shell build/quadriga-lib-version).zip
	mkdir release/quadriga_lib-$(shell build/quadriga-lib-version)
	mkdir release/quadriga_lib-$(shell build/quadriga-lib-version)/external
	cp external/armadillo-$(armadillo_version).zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/pugixml-$(pugixml_version).zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/Catch2-$(catch2_version).zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/hdf5-*.zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp external/MOxUnit.zip release/quadriga_lib-$(shell build/quadriga-lib-version)/external/
	cp -R include release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R mex release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R mex_cuda release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R src release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R tests release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R tools release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R html_docu release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp -R cpython release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp GNUmakefile release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp LICENSE release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp Makefile release/quadriga_lib-$(shell build/quadriga-lib-version)/
	cp README.md release/quadriga_lib-$(shell build/quadriga-lib-version)/
	( cd release && zip -r quadriga_lib-$(shell build/quadriga-lib-version).zip quadriga_lib-$(shell build/quadriga-lib-version)/ )
	- rm -rf release/quadriga_lib-$(shell build/quadriga-lib-version)
