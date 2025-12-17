# This Makefile is for Windows / MSVC environments

# Set Armadillo and HDF5 sources
hdf5_internal = ON
arma_internal = ON
avx2 = ON

CMAKE_BUILD_DIR = build_windows

all:   
	cmake -B $(CMAKE_BUILD_DIR) -D CMAKE_INSTALL_PREFIX=. -D ARMA_EXT=$(arma_internal) -D HDF5_STATIC=$(hdf5_internal) -D ENABLE_AVX2=$(avx2) 
	cmake --build $(CMAKE_BUILD_DIR) --config Release --parallel
	cmake --install $(CMAKE_BUILD_DIR)

test:   all   moxunit-lib
	cmake -B $(CMAKE_BUILD_DIR) -D ENABLE_TESTS=ON
	cmake --build $(CMAKE_BUILD_DIR) --config Release --parallel
	$(CMAKE_BUILD_DIR)\Release\test_bin.exe
	matlab -batch "run('tests\quadriga_lib_mex_tests.m');"

moxunit-lib:
	- rmdir /s /q external\MOxUnit-master
	tar -xf external/MOxUnit.zip
	move MOxUnit-master external

clean:
	- rmdir /s /q $(CMAKE_BUILD_DIR)
	- rmdir /s /q "+quadriga_lib"
	- rmdir /s /q lib
	- rmdir /s /q external\MOxUnit-master
