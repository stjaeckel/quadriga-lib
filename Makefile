# This Makefile is for Windows / MSVC environments

# Set Armadillo and HDF5 sources
hdf5_internal = ON
arma_internal = ON
avx2 = ON

CMAKE_BUILD_DIR = build_windows

# Binary distribution for Windows
DIST_DIR = release\quadriga_lib-win64
VERSION = 0.10.3
ARMA_VERSION = 14.2.2

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

binaries: all
	- mkdir release
	@echo Creating binary distribution...
	- rmdir /s /q $(DIST_DIR)
	mkdir $(DIST_DIR)
	mkdir $(DIST_DIR)\+quadriga_lib
	mkdir $(DIST_DIR)\lib
	mkdir $(DIST_DIR)\include
	@REM Copy MEX files
	copy "+quadriga_lib\*.mexw64" "$(DIST_DIR)\+quadriga_lib\"
	copy "+quadriga_lib\*.m" "$(DIST_DIR)\+quadriga_lib\"
	@REM Copy static libraries
	copy "lib\*.lib" "$(DIST_DIR)\lib\"
	@REM Copy public headers
	xcopy /E /I "include" "$(DIST_DIR)\include"
	xcopy /E /I "$(CMAKE_BUILD_DIR)\armadillo-$(ARMA_VERSION)\include" "$(DIST_DIR)\include"
	@REM Copy license and documentation
	copy LICENSE "$(DIST_DIR)\" 2>nul || echo No LICENSE file
	copy README.md "$(DIST_DIR)\" 2>nul || echo No README file
	@REM Optional: Copy HTML documentation
	if exist html_docu xcopy /E /I "html_docu" "$(DIST_DIR)\html_docu"
	@echo Binary distribution created in $(DIST_DIR)/
	@echo Creating ZIP archive...
	powershell -Command "Compress-Archive -Path '$(DIST_DIR)\*' -DestinationPath 'release\quadriga_lib_v$(VERSION)_win64.zip' -Force"
	@echo Created quadriga_lib-win64-$(VERSION).zip
	- rmdir /s /q $(DIST_DIR)

