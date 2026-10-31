/*!SECTION
Installation from Source
SECTION!*/

/*!SECTION_DESC

<br><br>

**Linux**<br>
- Requirements: a C++17 compiler (GCC 11 or newer) and CMake 3.18 or newer; optional MATLAB, Octave with `octave-dev`, and Python 3 with `numpy` and `pytest`. On Ubuntu:
```
sudo apt install build-essential cmake git python3-dev python3-numpy python3-pytest octave-dev
```
- Get the code and build it. `make` builds the library and all enabled interfaces and installs them into the repository folder:
```
git clone https://github.com/stjaeckel/quadriga-lib
cd quadriga-lib
make
```
- Without MATLAB, or with the system compiler instead of GCC 12: `make gcc_version= matlab=OFF`
- MATLAB / Octave: `addpath('/path/to/quadriga-lib')`
- Python: `export PYTHONPATH=/path/to/quadriga-lib/lib`, or `make python_install` to copy the module into the active environment.

<br>

**Windows**<br>
- Install the Build Tools for Visual Studio 2022 and CMake; optionally MATLAB and Python with `numpy` and `pytest`.
- Open the "x64 Native Tools Command Prompt", go to the repository folder and run `nmake`. Octave is not supported on Windows.

<br>

**More options**<br>
- Build options (AVX2, CUDA, shared library), make targets, a conda setup, the test suites and the MATLAB compiler versions are described in the <a target="_blank" rel="noopener noreferrer" href="https://github.com/stjaeckel/quadriga-lib#building-from-source">README on GitHub</a>.

SECTION_DESC!*/
