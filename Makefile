# This Makefile is for Windows / MSVC environments

# Components to build
matlab = ON
python = ON
avx2 = ON

CMAKE_BUILD_DIR = build_windows

# Options for prebuilding the HDF5 and Catch2 libraries to speed up build time
hdf5_version   = 1.14.6
HDF5_PREBUILT  = external\hdf5-prebuilt
HDF5_SRC_DIR   = hdf5-hdf5_$(hdf5_version)
HDF5_LIB_CHECK = $(HDF5_PREBUILT)\lib\libhdf5.lib

catch2_version   = 3.8.1
CATCH2_PREBUILT  = external\Catch2-prebuilt
CATCH2_SRC_DIR   = Catch2-$(catch2_version)
CATCH2_LIB_CHECK = $(CATCH2_PREBUILT)\lib\Catch2.lib

all:
	cmake -B $(CMAKE_BUILD_DIR) -D CMAKE_INSTALL_PREFIX=%CD% -D ENABLE_AVX2=$(avx2) -D ENABLE_MATLAB=$(matlab) -D ENABLE_PYTHON=$(python) -D ENABLE_TESTS=OFF
	cmake --build $(CMAKE_BUILD_DIR) --config Release --parallel
	cmake --install $(CMAKE_BUILD_DIR) --config Release

# Requires pytest and numpy:
#	pip install pytest numpy
test: all moxunit-lib
	cmake -B $(CMAKE_BUILD_DIR) -D ENABLE_TESTS=ON
	cmake --build $(CMAKE_BUILD_DIR) --config Release --target test_bin
	$(CMAKE_BUILD_DIR)\Release\test_bin.exe
!IF "$(matlab)" == "ON"
	matlab -batch "run('tests\quadriga_lib_mex_tests.m');"
!ENDIF
!IF "$(python)" == "ON"
	set PYTHONPATH=%CD%\lib
	python -m pytest tests/python_tests -x -s
!ENDIF

# Prebuild libraries
hdf5: $(HDF5_LIB_CHECK)

$(HDF5_LIB_CHECK):
	tar -xf external\hdf5-$(hdf5_version).zip -C external
	cmake -B external\hdf5-build -S external\$(HDF5_SRC_DIR) -DCMAKE_INSTALL_PREFIX=%CD%\$(HDF5_PREBUILT) -DBUILD_SHARED_LIBS=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF -DBUILD_TESTING=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_HL_LIB=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	cmake --build external\hdf5-build --config Release
	cmake --install external\hdf5-build --config Release
	- rmdir /s /q external\$(HDF5_SRC_DIR)
	- rmdir /s /q external\hdf5-build

catch2: $(CATCH2_LIB_CHECK)

$(CATCH2_LIB_CHECK):
	tar -xf external\Catch2-$(catch2_version).zip -C external
	cmake -B external\catch2-build -S external\$(CATCH2_SRC_DIR) -DCMAKE_INSTALL_PREFIX=%CD%\$(CATCH2_PREBUILT) -DBUILD_TESTING=OFF
	cmake --build external\catch2-build --config Release
	cmake --install external\catch2-build --config Release
	- rmdir /s /q external\$(CATCH2_SRC_DIR)
	- rmdir /s /q external\catch2-build

moxunit-lib:
	- rmdir /s /q external\MOxUnit-master
	tar -xf external/MOxUnit.zip
	move MOxUnit-master external

clean:
	- rmdir /s /q $(CMAKE_BUILD_DIR)
	- rmdir /s /q "+quadriga_lib"
	- rmdir /s /q lib

tidy: clean
	- rmdir /s /q external\MOxUnit-master
	- rmdir /s /q $(HDF5_PREBUILT)
	- rmdir /s /q $(CATCH2_PREBUILT)
