# This Makefile is for Windows / MSVC environments

# External libraries
# External libraries are located in the 'external' folder. Set the version numbers here.
# You need to compile the HDF5 and Catch2 libraries (e.g. using 'make hdf5lib' or 'make catch2lib' )
armadillo_version = 14.2.2
hdf5_version      = 1.14.2
catch2_version    = 3.4.0

# nmake cheat sheet:
#	$@    Current target's full name (path, base name, extension)
#	$*    Current target's path and base name minus fimaaeaeaele extension.
#	$**   All dependents of the current target
#	$(@B) Current target's base name (no path, no extension)
#	$(@F) Current target's base name + estension (no path)

# Compilers
CC      = cl
CCFLAGS = /EHsc /std:c++17 /Zc:__cplusplus /nologo /MD

CMAKE_BUILD_DIR = build_windows

# Header files
ARMA_H      = external/armadillo-$(armadillo_version)/include
CATCH2      = external\Catch2-$(catch2_version)-win64
HDF5        = external\hdf5-$(hdf5_version)-win64

all:   quadriga-lib

quadriga-lib:
	cmake -B $(CMAKE_BUILD_DIR) -D HDF5_PATH=external\hdf5-$(hdf5_version)-win64 -D CMAKE_INSTALL_PREFIX=.
	cmake --build $(CMAKE_BUILD_DIR) --config Release -- /m
	cmake --install $(CMAKE_BUILD_DIR)

test:   tests\test.exe
	tests\test.exe
	matlab -batch "run('tests\quadriga_lib_mex_tests.m');"

tests\test.exe:   tests\quadriga_lib_catch2_tests.cpp   lib\quadriga.lib   lib\libhdf5.lib
	$(CC) $(CCFLAGS) /Fetests\test.exe $** /Iinclude /I$(ARMA_H) /I$(CATCH2)\include /link $(CATCH2)\lib\Catch2.lib shlwapi.lib
	del quadriga_lib_catch2_tests.obj

# External libraries
external:   armadillo-lib   hdf5-lib   catch2-lib   moxunit-lib

armadillo-lib:
	- rmdir /s /q external\armadillo-$(armadillo_version)
	tar -xf external/armadillo-$(armadillo_version).zip
	move armadillo-$(armadillo_version) external

hdf5-lib:
	- rmdir /s /q external\build
	- rmdir /s /q external\hdf5-$(hdf5_version)
	- rmdir /s /q external\hdf5-$(hdf5_version)-win64
	tar -xf external/hdf5-$(hdf5_version).zip
	move hdf5-$(hdf5_version) external
	mkdir external\build
	cmake -S external\hdf5-$(hdf5_version) -B external\build -D CMAKE_INSTALL_PREFIX=external\hdf5-$(hdf5_version)-win64 -D BUILD_SHARED_LIBS=OFF -D HDF5_ENABLE_Z_LIB_SUPPORT=OFF -D BUILD_TESTING=OFF -D DHDF5_BUILD_TOOLS=OFF -D DHDF5_BUILD_EXAMPLES=OFF -D DHDF5_BUILD_HL_LIB=OFF
	cmake --build external\build --config Release --target install
	rmdir /s /q external\build
	rmdir /s /q external\hdf5-$(hdf5_version)

catch2-lib:
	- rmdir /s /q external\build
	- rmdir /s /q external\Catch2-$(catch2_version)
	- rmdir /s /q external\Catch2-$(catch2_version)-win64
	tar -xf external/Catch2-$(catch2_version).zip
	move Catch2-$(catch2_version) external
	mkdir external\build
	cmake -S external\Catch2-$(catch2_version) -B external\build
	cmake --build external\build --config Release --target package
	move external\build\_CPack_Packages\win64\NSIS\Catch2-$(catch2_version)-win64 external
	rmdir /s /q external\build
	rmdir /s /q external\Catch2-$(catch2_version)

moxunit-lib:
	- rmdir /s /q external\MOxUnit-master
	tar -xf external/MOxUnit.zip
	move MOxUnit-master external

clean:
	- rmdir /s /q external\build
	- rmdir /s /q external\Catch2-$(catch2_version)
	- rmdir /s /q external\hdf5-$(hdf5_version)
	- rmdir /s /q $(CMAKE_BUILD_DIR)
	- rmdir /s /q "+quadriga_lib"
	- rmdir /s /q lib
	- del tests\test.exe
	
tidy:   clean
	- rmdir /s /q external\Catch2-$(catch2_version)-win64
	- rmdir /s /q external\hdf5-$(hdf5_version)-win64
	- rmdir /s /q external\MOxUnit-master
