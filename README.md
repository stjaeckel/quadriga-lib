# Quadriga-Lib
C++/MEX/Python Utility library for radio channel modelling and simulations

## Introduction

**Quadriga-Lib** is an open-source utility library for radio channel modelling and simulation. It generates realistic channel impulse responses for system-level studies of mobile radio networks and offers a rich toolset for antenna arrays, channel data handling, and site-specific simulations.

For a full documentation, see the [quadriga-lib.org](http://quadriga-lib.org).

## Installation

Installation of **`quadriga-lib`** is supported on Linux/Ubuntu and Windows. The library can be used in MATLAB, Octave, and Python.

### Linux / Ubuntu

* [**OPTIONAL**]: Install Matlab from (https://www.mathworks.com) - you need to obtain a license
* Install the required packages (octave-dev is only needed if you want to use Octave):
```
sudo apt install bzip2 gcc git make cmake g++ libhdf5-dev python3-dev python3-pytest python3-numpy octave-dev python3-pip python3-venv octave-dev
```
* Get Quadriga-Lib either from Github or [quadriga-lib.org](http://quadriga-lib.org)
```
git clone https://github.com/stjaeckel/quadriga-lib
cd quadriga-lib
```
* Compile Quadriga-Lib using `make` (cmake is used internally and can also be used directly)
* [**OPTIONAL**]: Build and run tests: `make test`
* [**OPTIONAL**]: Add the library to your MATLAB path by running `addpath('quadriga-lib')` in the MATLAB command window or by adding it to your `startup.m` file.
* [**OPTIONAL**]: If you want to use the Python API, add the library to your Python path by adding the library path to your `PYTHONPATH` environment variable. You can do this by adding the following line to your `.bashrc` or `.bash_profile` file:
```
export PYTHONPATH=$PYTHONPATH:/path/to/quadriga-lib/lib
```

### Linux / Ubuntu with Anaconda

* Install Miniconda from [Anaconda's official website](https://www.anaconda.com/products/distribution#download-section). If you already have Anaconda installed, you can skip this step.
```
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
* Setup a new Anaconda environment with the required packages (you can also use the default channel instead of `conda-forge`, but octave is only available in `conda-forge`):
```
conda deactivate
conda create --name quadriga-lib -c conda-forge python=3.13 numpy pandas jupyterlab seaborn pytest scipy jupyterlab_widgets ipywidgets traittypes jupyter compilers make cmake hdf5 octave
conda activate quadriga-lib
```
* Get Quadriga-Lib either from Github or [quadriga-lib.org](http://quadriga-lib.org)
```
git clone https://github.com/stjaeckel/quadriga-lib
cd quadriga-lib
```
* Compile Quadriga-Lib (C++, MATLAB, Octave, Python) using `make` (cmake is used internally and can also be used directly)
* [**OPTIONAL**]: Build and run tests: `make test`
* [**ALTERNATIVE**]: If you only want to use the Python API, use `make python` and `make python_test`.
* [**OPTIONAL**]: Add the library to your MATLAB path by running `addpath('quadriga-lib')` in the MATLAB command window or by adding it to your `startup.m` file.
* [**OPTIONAL**]: If you want to use the Python API, install the Python package by running: `make python_install`. This will install the `quadriga-lib` package into your Anaconda environment.


### Windows
* Octave and Python are not supported yet (only MATLAB is)
* Install Build Tools for Visual Studio 2022 from https://visualstudio.microsoft.com
* Install Matlab from (https://www.mathworks.com) - you need to obtain a licence
* From MATLAB Shell run "mex -setup -v" and select compiler MSVC Compiler
* If error: https://yingzhouli.com/posts/2020-06/mex-msvc.html
* Open "x64 Native Tools Command Prompt" from start menu
* Navigate to library path, e.g. `cd Z:\quadriga-lib`
* Run `nmake` to compile `quadriga-lib` and the MEX Matlab interface
* [**OPTIONAL**]: Compile and run the unit tests by running `nmake test`
* [**OPTIONAL**]: Add the library to your MATLAB path by running `addpath('quadriga-lib')` in the MATLAB command window or by adding it to your `startup.m` file.


## Distribution License

**`quadriga-lib`** can be used in both open-source and proprietary (closed-source) software.

**`quadriga-lib`** is licensed under the Apache License, Version 2.0 (the "License").
A copy of the License is included in the "LICENSE" file. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Software Structure
### Folders

| Folder | Content |
|:---|:---|
`+quadriga_lib` | Compiled mex files for usage in MATLAB and Octave
`api_mex`       | Source files for the MATLAB / Octave MEX interface
`api_python`    | Source files for the Python API
`build*`        | Folder(s) for temporary build files
`external`      | External tools used in the project
`html_docu`     | Documentation of the API functions (aka: [quadriga-lib.org](http://quadriga-lib.org))
`include`       | Public header files for the `quadriga-lib` library
`lib`           | Library files for static and dynamic linking, Python package
`references`    | Relevant external documents and papers
`release`       | Folder for source packages and compiled binary packages
`src`           | C++ source files and private header files
`tests`         | Test files
`tools`         | Other tools used in the project (e.g. the website generator)


