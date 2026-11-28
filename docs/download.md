# Get started

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/download.html

| Section | Topics |
| --- | --- |
| [Download](#download) |  |
| [Windows](#windows) | [Windows package](#windows-package) · [Python](#python) · [conda](#conda) · [From source](#from-source) |
| [Ubuntu](#ubuntu) | [Ubuntu package](#ubuntu-package) · [Python in a virtual environment](#python-in-a-virtual-environment) · [From source](#from-source-2) |
| [Linux with conda](#linux-conda) | [conda](#conda-2) · [From source](#from-source-3) |
| [C++ and more options](#cpp) |  |
| [Quick Start](#quick-start) | [Python](#python-2) · [MATLAB / Octave](#matlab-octave) · [C++](#c) |

How to get Quadriga-Lib running: the downloads, then the installation for each platform, from the packages or built from the source code, and a first example in each language. Pick your platform in the menu; each section is complete on its own.

<a id="download"></a>
## Download

- Source code: [quadriga_lib_0.12.5.zip](http://quadriga-lib.org/download/quadriga_lib_0.12.5.zip)
- Windows package (MATLAB interface, library, headers, Python wheels): [quadriga_lib_0.12.5_win64.zip](http://quadriga-lib.org/download/quadriga_lib_0.12.5_win64.zip)
- Ubuntu packages (MATLAB and Octave interface, library, headers): [22.04 Jammy](http://quadriga-lib.org/download/quadriga-lib_0.12.5-1+jammy_amd64.deb) · [24.04 Noble](http://quadriga-lib.org/download/quadriga-lib_0.12.5-1+noble_amd64.deb) · [26.04 Resolute](http://quadriga-lib.org/download/quadriga-lib_0.12.5-1+resolute_amd64.deb)
- Python: `pip install quadriga-lib`
- [Older versions](http://quadriga-lib.org/releases/)

<a id="windows"></a>
## Windows

<a id="windows-package"></a>
### Windows package

- Extract the ZIP file from the [download section](#download) to a folder of your choice, e.g. `C:\quadriga-lib`. It contains the MATLAB interface (`+quadriga_lib`), the static library, the headers and Python wheels.
- MATLAB: add the library to the path with `addpath('C:\quadriga-lib')` in the command window or in your `startup.m` file. Octave is not supported on Windows.

<a id="python"></a>
### Python

- Install from PyPI; wheels exist for Python 3.10 to 3.14:

```bat
pip install quadriga-lib
```

- Without internet access, install the wheel of the Windows package that matches your Python version (`python --version` shows it): `pip install wheels\quadriga_lib-0.12.5-cp314-cp314-win_amd64.whl`.

<a id="conda"></a>
### conda

- With [Miniforge](https://conda-forge.org/download/), in the "Miniforge Prompt":

```bat
conda create -n quadriga --override-channels -c conda-forge python=3.14 pip numpy
conda activate quadriga
pip install quadriga-lib
```

<a id="from-source"></a>
### From source

- Install the Build Tools for Visual Studio 2022 from the [Visual Studio older downloads](https://visualstudio.microsoft.com/vs/older-downloads/) page (MATLAB does not support Visual Studio 2026 yet) with the workload "Desktop development with C++"; it includes CMake and Ninja. Install [Git](https://git-scm.com/downloads/win), or take the source code from the [download section](#download) instead of `git clone`. Optionally MATLAB.
- In the "x64 Native Tools Command Prompt", or in the Miniforge Prompt after `call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`:

```bat
git clone https://github.com/stjaeckel/quadriga-lib.git
cd quadriga-lib
nmake
```

- `nmake` builds the library, the MATLAB interface into `+quadriga_lib` and the Python module into `lib`. MATLAB: `addpath('C:\path\to\quadriga-lib')`. Python: `set PYTHONPATH=C:\path\to\quadriga-lib\lib`, or in a conda environment `conda env config vars set PYTHONPATH=%CD%\lib` once.

<a id="ubuntu"></a>
## Ubuntu

<a id="ubuntu-package"></a>
### Ubuntu package

- Download the package for your Ubuntu release from the [download section](#download) and install it (replace `<version>` and `<release>`):

```bash
sudo apt install ./quadriga-lib_<version>-1+<release>_amd64.deb
```

- MATLAB / Octave: add the library to the path with `addpath('/usr/share/quadriga-lib')` in the command window, or in `startup.m` (MATLAB) or `~/.octaverc` (Octave).

<a id="python-in-a-virtual-environment"></a>
### Python in a virtual environment

- Ubuntu does not allow pip to install into the system Python, so use a virtual environment and activate it in each new terminal:

```bash
sudo apt install python3-venv
python3 -m venv ~/quadriga-venv
. ~/quadriga-venv/bin/activate
pip install quadriga-lib
```

<a id="from-source-2"></a>
### From source

```bash
sudo apt install build-essential cmake git python3-dev octave-dev
git clone https://github.com/stjaeckel/quadriga-lib.git
cd quadriga-lib
make
make python_install
```

- `make` builds the library, the Octave interface, the Python module and, if MATLAB is on the `PATH`, the MATLAB interface (with GCC 12, `sudo apt install g++-12`). `make python_install` copies the module into the active virtual environment.
- MATLAB / Octave: `addpath('/path/to/quadriga-lib')`.

<a id="linux-conda"></a>
## Linux with conda

<a id="conda-2"></a>
### conda

- With [Miniforge](https://conda-forge.org/download/), also on other Linux distributions; `octave` is optional:

```bash
conda create -n quadriga --override-channels -c conda-forge python=3.14 pip numpy octave
conda activate quadriga
pip install quadriga-lib
```

- The Python module comes from PyPI. The Octave interface comes with the Ubuntu package, or is built from source.

<a id="from-source-3"></a>
### From source

```bash
conda install --override-channels -c conda-forge cxx-compiler cmake make git
git clone https://github.com/stjaeckel/quadriga-lib.git
cd quadriga-lib
make
make python_install
```

- In an active environment, `make` uses its compiler and Octave; `make python_install` copies the module into the environment. MATLAB / Octave: `addpath('/path/to/quadriga-lib')`.

<a id="cpp"></a>
## C++ and more options

- The binary packages contain the public headers and the static library (`libquadriga.a` on Linux, `quadriga.lib` on Windows), together with the [Armadillo](https://arma.sourceforge.net) headers; a build from source puts them into `include` and `lib`. Link the static library and enable OpenMP (`-fopenmp` with GCC).
- Build options (AVX2, CUDA, shared library), make targets, the test suites and the MATLAB compiler versions are described in the [README on GitHub](https://github.com/stjaeckel/quadriga-lib#building-from-source).

<a id="quick-start"></a>
## Quick Start

All three examples create a 2 × 2 3GPP antenna array at 3.7 GHz and print the directivity of the first element (about 9.8 dBi). More examples are in the API references.

<a id="python-2"></a>
### Python

```python
import quadriga_lib as qd

ant = qd.arrayant.generate_antenna('3gpp', panel=[2, 2], center_freq=3.7e9)
print(qd.version())
print(qd.arrayant.calc_directivity(ant, i_element=0))
```

<a id="matlab-octave"></a>
### MATLAB / Octave

```matlab
ant = quadriga_lib.generate_antenna('3GPP', 1, 3.7e9, [], [2, 2]);
disp(quadriga_lib.version())
disp(quadriga_lib.calc_directivity(ant, 1))
```

<a id="c"></a>
### C++

```cpp
#include <quadriga_lib.hpp>
#include <iostream>

int main()
{
    auto ant = quadriga_lib::generate_antenna<double>("3GPP", 1.0, 3.7e9, {}, {2, 2});
    std::cout << quadriga_lib::quadriga_lib_version() << "\n";
    arma::vec directivity_dBi = quadriga_lib::calc_directivity(ant);
    std::cout << directivity_dBi(0) << " dBi\n";
}
```

Compile in the source folder after `make`:

```bash
g++ -std=c++17 -O2 -fopenmp -I include -I external/armadillo-15.6.2/include example.cpp lib/libquadriga.a -o example
```
