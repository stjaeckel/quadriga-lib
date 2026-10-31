# This Makefile is for Windows / MSVC environments
# Build steps: all, test, pip-win, release-win

# Components to build
matlab = ON
python = ON
avx2 = ON

# Set to 1 to use Ninja (faster), 0 to use Visual Studio generator
USE_NINJA = 1
CMAKE_BUILD_DIR = build_windows

# Armadillo version bundled in external\ (headers copied by release-win)
armadillo_version = 14.2.2

# Read Quadriga-Lib version number from include\quadriga_lib.hpp
!IF [cmd /c tools\get_version.cmd] == 0
!INCLUDE version.tmp
!ENDIF
DIST_DIR = release\quadriga_lib_$(QUADRIGA_VERSION)_win64

# Select generator (Ninja vs Visual Studio)
!IF "$(USE_NINJA)" == "1"
GENERATOR = -G Ninja
TEST_BIN  = $(CMAKE_BUILD_DIR)\test_bin.exe
!ELSE
GENERATOR =
TEST_BIN  = $(CMAKE_BUILD_DIR)\Release\test_bin.exe
!ENDIF

all:
	if not exist $(CMAKE_BUILD_DIR)\build.ninja cmake -B $(CMAKE_BUILD_DIR) $(GENERATOR) -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=%CD% -D ENABLE_AVX2=$(avx2) -D ENABLE_MATLAB=$(matlab) -D ENABLE_PYTHON=$(python) -D ENABLE_TESTS=ON
	cmake --build $(CMAKE_BUILD_DIR) --config Release --parallel
	cmake --install $(CMAKE_BUILD_DIR) --config Release

# Requires pytest and numpy:
#	pip install pytest numpy
test: all moxunit-lib
#	cmake -B $(CMAKE_BUILD_DIR) $(GENERATOR) -D CMAKE_BUILD_TYPE=Release -D ENABLE_TESTS=ON
#	cmake --build $(CMAKE_BUILD_DIR) --config Release --target test_bin
	$(TEST_BIN)
!IF "$(matlab)" == "ON"
	matlab -batch "run('tests\quadriga_lib_mex_tests.m');"
!ENDIF
!IF "$(python)" == "ON"
	set PYTHONPATH=%CD%\lib
	python -m pytest tests/python_tests -x -s
!ENDIF

moxunit-lib:
	- rmdir /s /q external\MOxUnit-master
	tar -xf external/MOxUnit.zip
	move MOxUnit-master external

# pip targets:
#
#   pip install cibuildwheel build
#   Python 3.10–3.13 installed from python.org (py launcher must see them)

wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp310-cp310-win_amd64.whl:
	python -m cibuildwheel --only cp310-win_amd64

wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp311-cp311-win_amd64.whl:
	python -m cibuildwheel --only cp311-win_amd64

wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp312-cp312-win_amd64.whl:
	python -m cibuildwheel --only cp312-win_amd64

wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp313-cp313-win_amd64.whl:
	python -m cibuildwheel --only cp313-win_amd64

wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp314-cp314-win_amd64.whl:
	python -m cibuildwheel --only cp314-win_amd64

pip-win-cp310: wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp310-cp310-win_amd64.whl
pip-win-cp311: wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp311-cp311-win_amd64.whl
pip-win-cp312: wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp312-cp312-win_amd64.whl
pip-win-cp313: wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp313-cp313-win_amd64.whl
pip-win-cp314: wheelhouse\quadriga_lib-$(QUADRIGA_VERSION)-cp314-cp314-win_amd64.whl

pip-win: pip-win-cp310 pip-win-cp311 pip-win-cp312 pip-win-cp313 pip-win-cp314

# Release
# Make these first: all pip-win
release-win: 
	- rmdir /s /q $(DIST_DIR)
	cmake --install $(CMAKE_BUILD_DIR) --config Release --prefix $(DIST_DIR)
	mkdir $(DIST_DIR)\wheels
	copy wheelhouse\*win_amd64.whl $(DIST_DIR)\wheels\ 
	copy README_win.md $(DIST_DIR)\README.md
	copy LICENSE "$(DIST_DIR)\" 2>nul || echo No LICENSE file
	xcopy /E /I "include" "$(DIST_DIR)\include"
	xcopy /E /I "$(CMAKE_BUILD_DIR)\armadillo-$(armadillo_version)\include" "$(DIST_DIR)\include"
	if exist html_docu xcopy /Y /I "html_docu\*" "release\quadriga_lib_0.11.5_win64\html_docu\"
	powershell -Command "Compress-Archive -Path $(DIST_DIR) -DestinationPath $(DIST_DIR).zip -Force"
	- rmdir /s /q $(DIST_DIR)

clean:
	- rmdir /s /q $(CMAKE_BUILD_DIR)
	- del version.tmp
	- del wheelhouse\*-win_amd64.whl
	- del $(DIST_DIR).zip
	- del "+quadriga_lib\*.mexw64"
	- del lib\*.lib
	- del lib\*.pyd

tidy: clean
	- rmdir /s /q external\MOxUnit-master
	- rmdir /s /q "+quadriga_lib"
	- rmdir /s /q lib
