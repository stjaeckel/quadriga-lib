# Quadriga-Lib

[![PyPI](https://img.shields.io/pypi/v/quadriga-lib)](https://pypi.org/project/quadriga-lib/)
[![Python](https://img.shields.io/pypi/pyversions/quadriga-lib)](https://pypi.org/project/quadriga-lib/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-quadriga--lib.org-informational)](http://quadriga-lib.org)

C++ library with MATLAB, Octave and Python interfaces for radio channel modelling and simulation.

Quadriga-Lib generates channel impulse responses for system-level studies of mobile radio networks. It also provides tools for array antennas, channel data storage and processing, and site-specific (ray-tracing) simulations. Documentation and API reference: **[quadriga-lib.org](http://quadriga-lib.org)**.

## Relation to QuaDRiGa

[QuaDRiGa](https://quadriga-channel-model.de) is a geometry-based stochastic channel model written in MATLAB. Quadriga-Lib is its C++ successor by the same lead developer: the model components are being rewritten in C++ for speed, with thin interfaces to MATLAB, Octave and Python. It is not a drop-in replacement for the MATLAB toolbox; functions, names and file formats follow their own API. See the [API documentation](http://quadriga-lib.org) for what is available.

[QuaDRiGa-NG](https://github.com/stjaeckel/QuaDRiGa-NG) is a fork of the MATLAB QuaDRiGa toolbox that uses Quadriga-Lib in its backend.

## Features

- **Stochastic channel model (GBSM)**: large-scale and small-scale fading parameters, path loss, LOS probability, O2I loss, and scenario parameters for 3GPP TR 38.901 (configuration files in `data/config/`)
- **IEEE 802.11 (Wi-Fi) indoor channel models**: TGn channel types A to F with the TGac, TGax and TGah extensions (MU-MIMO, floor penetration loss, time-evolving channels)
- **Spatially consistent random fields**: sum-of-sinusoids generators for correlated random variables
- **Array antennas**: predefined models (omni, dipole, ULA, 3GPP panels, multibeam), pattern interpolation, rotation, coupling, directivity and beamwidth, QDANT file format
- **Channel coefficients**: antenna patterns applied to path data (spherical and planar wave, multi-frequency, IRS), baseband frequency response, delay quantization, delay and angular spread, K-factor, XPR
- **Channel data storage**: HDF5 channel files for large datasets (read, write, reshape)
- **Site-specific simulation**: ray–triangle and ray–point intersection, mesh tools, OBJ and Mitsuba XML export, reading of QRT ray-tracing results
- **Acoustics (experimental)**: room-acoustic simulation with the ray tracer, loudspeaker models and acoustic material tables; under active development
- **Fast core**: C++17 with OpenMP, optional AVX2 and CUDA acceleration

## Platforms

| Platform        | C++ | MATLAB | Octave | Python |
| :-------------- | :-: | :----: | :----: | :----: |
| Linux (x86_64)  | yes | yes    | yes    | yes    |
| Windows (x64)   | yes | yes    | –      | yes    |

Prebuilt Python wheels also exist for Linux aarch64.

## Installation

Binary packages are built for each release. The [download page](http://quadriga-lib.org/download.html) always offers the latest release; the `main` branch on GitHub may be ahead of it (see [Changes on main](#changes-on-main)).

| Package                     | How to install                                                                 |
| :-------------------------- | :----------------------------------------------------------------------------- |
| Python (Linux, Windows)     | `pip install quadriga-lib`                                                     |
| Ubuntu (`.deb`)             | download from the [download page](http://quadriga-lib.org/download.html), then `sudo dpkg -i quadriga-lib_<version>+<distro>_amd64.deb`; MATLAB path: `/usr/share/quadriga-lib` |
| Windows (`.zip`)            | download and extract; contains the MEX files, static library, headers and Python wheels |
| Source                      | see [Building from source](#building-from-source)                              |

For MATLAB or Octave, add the folder that contains `+quadriga_lib` to the path, e.g. `addpath('/usr/share/quadriga-lib')`.

## Quick start

All three examples create a 2 × 2 3GPP antenna array at 3.7 GHz and print the directivity of the first element (about 9.8 dBi).

**Python**

```python
import quadriga_lib

ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9)
print(quadriga_lib.version())
print(quadriga_lib.arrayant.calc_directivity(ant, 0))
```

**MATLAB / Octave**

```matlab
ant = quadriga_lib.arrayant_generate('3GPP', 1, 3.7e9, [], [], [], 2, 2);
disp(quadriga_lib.version())
disp(quadriga_lib.arrayant_calc_directivity(ant, 1))
```

**C++**

```cpp
#include <quadriga_lib.hpp>
#include <iostream>

int main()
{
    auto ant = quadriga_lib::generate_arrayant_3GPP<double>(2, 2, 3.7e9);
    std::cout << quadriga_lib::quadriga_lib_version() << "\n";
    std::cout << ant.calc_directivity_dBi(0) << " dBi\n";
}
```

```bash
g++ -std=c++17 -O2 -fopenmp -I include -I external/armadillo-14.2.2/include example.cpp lib/libquadriga.a -o example
```

More examples are in the usage sections of the [API documentation](http://quadriga-lib.org).

## Building from source

### Requirements

- C++17 compiler (GCC 11 or newer, or MSVC 2022) and CMake 3.18 or newer
- Optional: MATLAB (with a supported compiler, see [MATLAB and GCC](#matlab-and-gcc)), Octave with `octave-dev`, Python 3 with `numpy` and `pytest`
- Bundled in `external/` (no install needed): Armadillo, pybind11, Catch2, MOxUnit, pugixml
- Optional, tests only: HDF5 C library 1.12 or newer, for an interoperability test of the HDF5 channel files

On Ubuntu:

```bash
sudo apt install build-essential cmake git python3-dev python3-numpy python3-pytest octave-dev
```

### Linux

```bash
git clone https://github.com/stjaeckel/quadriga-lib
cd quadriga-lib
make
```

`make` configures CMake in `build_linux/`, builds, and installs into the repository folder:

| Output                              | Location           |
| :---------------------------------- | :----------------- |
| Static library `libquadriga.a`      | `lib/`             |
| Python module `quadriga_lib*.so`    | `lib/`             |
| MATLAB / Octave MEX files and help  | `+quadriga_lib/`   |

Then use it from:

- **MATLAB / Octave**: `addpath('/path/to/quadriga-lib')`
- **Python**: `export PYTHONPATH=/path/to/quadriga-lib/lib`, or `make python_install` to copy the module into the active environment's `site-packages`
- **C++**: headers in `include/`, link `lib/libquadriga.a` with `-fopenmp` (see the [C++ example](#quick-start))

### Build options

Options are variables at the top of `GNUmakefile` and can be overridden on the command line:

| Variable      | Default | Meaning                                                        |
| :------------ | :------ | :------------------------------------------------------------- |
| `gcc_version` | `12`    | Uses `gcc-12`/`g++-12`; set empty (`gcc_version=`) for the system compiler |
| `matlab`      | `ON`    | MATLAB MEX interface                                           |
| `octave`      | `ON`    | Octave MEX interface                                           |
| `python`      | `ON`    | Python interface                                               |
| `avx2`        | `ON`    | AVX2 acceleration (turn off on CPUs without AVX2)              |
| `cuda`        | `OFF`   | CUDA GPU acceleration                                          |
| `shared_lib`  | `OFF`   | Also build a shared library                                    |
| `MATLAB_BIN`  | `matlab` | MATLAB command used by `make test`                            |

Example without MATLAB, with the system compiler:

```bash
make gcc_version= matlab=OFF
```

The same switches exist as CMake options (`ENABLE_MATLAB`, `ENABLE_OCTAVE`, `ENABLE_PYTHON`, `ENABLE_AVX2`, `ENABLE_CUDA`, `ENABLE_SHARED_LIB`, `ENABLE_TESTS`) if you prefer to call CMake directly.

### Make targets

| Target                | Does                                                                |
| :-------------------- | :------------------------------------------------------------------ |
| `make`                | Builds and installs the library and all enabled interfaces          |
| `make python`         | Builds only the static library and the Python module                |
| `make python_install` | Copies the Python module into the active environment                |
| `make test`           | Builds and runs the C++ tests, then the Octave, MATLAB and Python tests that are enabled |
| `make python_test`    | Runs only the Python tests                                          |
| `make documentation`  | Rebuilds the website in `html_docu/` from the source comments       |
| `make clean`          | Removes build files                                                 |

Without MATLAB, run the tests with `make test matlab=OFF`.

### Tests

| Suite           | Location                          | Run directly                                                  |
| :-------------- | :-------------------------------- | :------------------------------------------------------------ |
| C++ (Catch2)    | `tests/catch2_tests/`             | `build_linux/test_bin` (one case: `build_linux/test_bin "<test name>"`) |
| MATLAB / Octave | `tests/mex_tests/`                | `octave --eval "cd tests; quadriga_lib_mex_tests;"`           |
| Python (pytest) | `tests/python_tests/`             | `PYTHONPATH=lib python3 -m pytest tests/python_tests -x`      |

### Windows

1. Install the Build Tools for Visual Studio 2022 and CMake; optionally MATLAB and Python with `numpy` and `pytest`.
2. Open the "x64 Native Tools Command Prompt" and go to the repository folder.
3. Run `nmake` to build the library, the MEX files and the Python module; `nmake test` builds and runs the tests.

Octave is not supported on Windows.

### Conda

A conda environment provides compilers, CMake, Octave and Python in one place:

```bash
conda create --name quadriga-lib -c conda-forge python numpy pytest h5py compilers make cmake octave
conda activate quadriga-lib
make gcc_version= matlab=OFF
make python_install
```

### MATLAB and GCC

MATLAB ships its own, older C++ runtime (`libstdc++`). MEX files built with a newer GCC than your MATLAB release supports fail to load with a `GLIBCXX_... not found` error. The `GNUmakefile` therefore uses GCC 12 by default. If your MATLAB release needs a different version, install it (e.g. `sudo apt install g++-11`) and build with `make gcc_version=11`. The compilers supported by each MATLAB release are listed by MathWorks.

## Changes on main

The HDF5 channel files use file format version 3. Files of version 2 (written by quadriga-lib 0.12.4 and earlier) are rejected. Convert them once with:

```bash
python tools/hdf5_convert_v2.py old.hdf5 new.hdf5
```

This needs `numpy`, `h5py` and the `quadriga_lib` Python module, also for MATLAB and Octave users.

## Repository layout

| Folder          | Content                                                                         |
| :-------------- | :------------------------------------------------------------------------------ |
| `src/`          | C++ sources with their documentation comments; related API functions may share a file|
| `include/`      | Public C++ headers                                                              |
| `api_mex/`      | MATLAB / Octave MEX interface sources                                           |
| `api_python/`   | Python interface sources (pybind11)                                             |
| `+quadriga_lib/`| Built MEX files and MATLAB help files                                           |
| `lib/`          | Built static library and Python module                                          |
| `tests/`        | C++, MATLAB / Octave and Python tests, benchmarks                               |
| `data/`         | Scenario configuration files and material tables                                |
| `html_docu/`    | Website and API reference ([quadriga-lib.org](http://quadriga-lib.org)), built by `make documentation` |
| `html_builder/` | Scripts and page sources for the website                                        |
| `md/`           | API specifications and development notes                                        |
| `tools/`        | Helper scripts, e.g. the HDF5 file converter                                    |
| `external/`     | Bundled third-party libraries                                                   |
| `debian/`       | Debian packaging                                                                |

The rules for adding or changing an API function are in `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md`.

## Contributing and bug reports

- **Bugs:** open a [bug report](https://github.com/stjaeckel/quadriga-lib/issues/new?template=bug_report.yml) with the library version and a small, self-contained example that shows the problem.
- **Feature requests:** use the [feature request form](https://github.com/stjaeckel/quadriga-lib/issues/new?template=feature_request.yml) and describe the task you want to solve.
- **Questions:** ask in [Discussions](https://github.com/stjaeckel/quadriga-lib/discussions).
- **Security problems:** report them privately, see [SECURITY.md](SECURITY.md).

Please search the [existing issues](https://github.com/stjaeckel/quadriga-lib/issues?q=is%3Aissue) first. How issues are handled, and how to contribute code, is described in [CONTRIBUTING.md](CONTRIBUTING.md).

## How to cite

If you use Quadriga-Lib in published work, please cite the software and the version you used:

```bibtex
@software{quadriga_lib,
  author  = {Jaeckel, Stephan},
  title   = {Quadriga-Lib: C++/MATLAB/Python library for radio channel modelling and simulation},
  url     = {https://github.com/stjaeckel/quadriga-lib},
  version = {<version>},
  year    = {<year>}
}
```

The installed version is returned by `quadriga_lib.version()`.

## Support

If Quadriga-Lib is useful to you, you can support its development via [GitHub Sponsors](https://github.com/sponsors/stjaeckel) or the options on [quadriga-lib.org](http://quadriga-lib.org).

## License

Apache License 2.0, see [LICENSE](LICENSE). Quadriga-Lib can be used in open-source and proprietary software.
