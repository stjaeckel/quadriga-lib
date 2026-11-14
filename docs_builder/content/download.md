+++
title = "Get started"
description = "%%GAP: meta description of the Get started page%%"
+++

%%GAP: text converted from the old page; check it against the new layout (headings for the index)%%

## Download {#download}

### Latest Version (Source Code)

[quadriga_lib_{{ version }}.zip]({{ download_base_url }}/quadriga_lib_{{ version }}.zip)

### Windows Binary Package

[quadriga_lib_{{ version }}_win64.zip]({{ download_base_url }}/quadriga_lib_{{ version }}_win64.zip)

### Ubuntu 22.04 (Jammy)

[quadriga-lib_{{ version }}-1+jammy_amd64.deb]({{ download_base_url }}/quadriga-lib_{{ version }}-1+jammy_amd64.deb)

### Ubuntu 24.04 (Noble)

[quadriga-lib_{{ version }}-1+noble_amd64.deb]({{ download_base_url }}/quadriga-lib_{{ version }}-1+noble_amd64.deb)

### Ubuntu 26.04 (Resolute)

[quadriga-lib_{{ version }}-1+resolute_amd64.deb]({{ download_base_url }}/quadriga-lib_{{ version }}-1+resolute_amd64.deb)

### Python

`pip install quadriga-lib`

### [Older versions]({{ releases_url }})

## Installation from Binaries {#installation-from-binaries}

### Python

- Install from PyPI:

```bash
pip install quadriga-lib
```

- Prebuilt wheels exist for Linux x86_64, Linux aarch64 and Windows x86_64 (Python 3.10–3.14). On other platforms, pip tries to build from source, which needs CMake and a C++17 compiler.

### Ubuntu

- Download the package for your Ubuntu release (see above) and install it with `dpkg` (replace `<version>` and `<distro>`):

```bash
sudo dpkg -i quadriga-lib_<version>-1+<distro>_amd64.deb
```

- MATLAB / Octave: add the library to the path with `addpath('/usr/share/quadriga-lib')` in the command window or in your `startup.m` file.

### Windows

- Extract the ZIP file to a folder of your choice, e.g. `C:\quadriga-lib`. It contains the MEX files, the static library, the headers and Python wheels.
- MATLAB: add the library to the path with `addpath('C:\quadriga-lib')` in the command window or in your `startup.m` file.
- Python (optional): install the wheel that matches your Python version. Replace `cp312` with your version (`cp310` to `cp314`); `python --version` shows it.

```text
pip install wheels\quadriga_lib-{{ version }}-cp312-cp312-win_amd64.whl
```

### C++

- The binary packages contain the public headers and the static library (`libquadriga.a` on Linux, `quadriga.lib` on Windows), together with the [Armadillo](https://arma.sourceforge.net) headers. Link the static library and enable OpenMP (`-fopenmp` with GCC).

## Installation from Source {#installation-from-source}

### Linux

- Requirements: a C++17 compiler (GCC 11 or newer) and CMake 3.18 or newer; optional MATLAB, Octave with `octave-dev`, and Python 3 with `numpy` and `pytest`. On Ubuntu:

```bash
sudo apt install build-essential cmake git python3-dev python3-numpy python3-pytest octave-dev
```

- Get the code and build it. `make` builds the library and all enabled interfaces and installs them into the repository folder:

```bash
git clone https://github.com/stjaeckel/quadriga-lib
cd quadriga-lib
make
```

- Without MATLAB, or with the system compiler instead of GCC 12: `make gcc_version= matlab=OFF`
- MATLAB / Octave: `addpath('/path/to/quadriga-lib')`
- Python: `export PYTHONPATH=/path/to/quadriga-lib/lib`, or `make python_install` to copy the module into the active environment.

### Windows

- Install the Build Tools for Visual Studio 2022 and CMake; optionally MATLAB and Python with `numpy` and `pytest`.
- Open the "x64 Native Tools Command Prompt", go to the repository folder and run `nmake`. Octave is not supported on Windows.

### More options

- Build options (AVX2, CUDA, shared library), make targets, a conda setup, the test suites and the MATLAB compiler versions are described in the [README on GitHub](https://github.com/stjaeckel/quadriga-lib#building-from-source).

## Quick Start {#quick-start}

All three examples create a 2 × 2 3GPP antenna array at 3.7 GHz and print the directivity of the first element (about 9.8 dBi). More examples are in the API references.

### Python

```python
import quadriga_lib

ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9)
print(quadriga_lib.version())
print(quadriga_lib.arrayant.calc_directivity(ant, 0))
```

### MATLAB / Octave

```matlab
ant = quadriga_lib.arrayant_generate('3GPP', 1, 3.7e9, [], [], [], 2, 2);
disp(quadriga_lib.version())
disp(quadriga_lib.arrayant_calc_directivity(ant, 1))
```

### C++

```cpp
#include <quadriga_lib.hpp>
#include <iostream>

int main()
{
    auto ant = quadriga_lib::generate_arrayant_3GPP<double>(2, 2, 3.7e9);
    std::cout << quadriga_lib::quadriga_lib_version() << "\n";
    arma::vec directivity_dBi;
    quadriga_lib::arrayant_calc_directivity(ant, directivity_dBi);
    std::cout << directivity_dBi(0) << " dBi\n";
}
```

Compile in the source folder after `make`:

```bash
g++ -std=c++17 -O2 -fopenmp -I include -I external/armadillo-{{ armadillo_version }}/include example.cpp lib/libquadriga.a -o example
```
