# This Makefile is for Windows / MSVC environments
# Build steps: all, test, pip-win, release-win

# Components to build
matlab = ON
python = ON
avx2 = ON
tests = OFF

# Set to 1 to use Ninja (faster), 0 to use Visual Studio generator
USE_NINJA = 1
CMAKE_BUILD_DIR = build_windows

# Armadillo version bundled in external\ (headers copied by release-win)
armadillo_version = 15.6.2

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
	cmake -B $(CMAKE_BUILD_DIR) $(GENERATOR) -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=%CD% -D ENABLE_AVX2=$(avx2) -D ENABLE_MATLAB=$(matlab) -D ENABLE_PYTHON=$(python) -D ENABLE_TESTS=$(tests)
	cmake --build $(CMAKE_BUILD_DIR) --config Release --parallel
	cmake --install $(CMAKE_BUILD_DIR) --config Release

# Requires pytest and numpy:
#	pip install pytest numpy
# Builds and installs as "all" does, with Catch2 and the test binary in the same build, then runs the C++, MATLAB and
# Tutorial notebooks (tutorials\README.md): runs every notebook that is newer than its executed copy into
# tutorials\executed\, in the active Python environment, which needs Jupyter, numpy, matplotlib and, for the MATLAB
# variants, the MATLAB kernel (pip install jupyter-matlab-proxy). The MATLAB variants run with the installed MATLAB
# instead of Octave. The library comes from this folder, so build it first with "nmake".
tutorials: FORCE
	set PYTHONPATH=$(MAKEDIR)\lib
	set MATLABPATH=$(MAKEDIR)
	set MWI_USE_EXISTING_LICENSE=True
	cd tutorials && python -m qnb.runall --matlab-kernel jupyter_matlab_kernel

# Website (website_builder\README.md): needs pandoc and the Python packages listed there
website: FORCE
	python website_builder\build.py website

# Always out of date, so that targets named like a folder (tutorials, website) run
FORCE:

# Python tests
test: moxunit-lib
	$(MAKE) /nologo CMAKE_BUILD_DIR=$(CMAKE_BUILD_DIR) USE_NINJA=$(USE_NINJA) matlab=$(matlab) python=$(python) avx2=$(avx2) tests=ON all
	$(TEST_BIN)
!IF "$(matlab)" == "ON"
	matlab -batch "run('tests\quadriga_lib_mex_tests.m');"
!ENDIF
!IF "$(python)" == "ON"
	set PYTHONPATH=$(MAKEDIR)\lib
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
	python docs_builder\build.py docs --check
	- rmdir /s /q $(DIST_DIR)
	cmake --install $(CMAKE_BUILD_DIR) --config Release --prefix $(DIST_DIR)
	mkdir $(DIST_DIR)\wheels
	copy wheelhouse\*win_amd64.whl $(DIST_DIR)\wheels\ 
	copy README_win.md $(DIST_DIR)\README.md
	copy LICENSE "$(DIST_DIR)\" 2>nul || echo No LICENSE file
	xcopy /E /I "include" "$(DIST_DIR)\include"
	xcopy /E /I "$(CMAKE_BUILD_DIR)\armadillo-$(armadillo_version)\include" "$(DIST_DIR)\include"
	xcopy /E /I "docs" "$(DIST_DIR)\docs"
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
