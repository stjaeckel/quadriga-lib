# quadriga-lib
Utility library for radio channel modelling and simulations

## Introduction

**`quadriga-lib`** is a support library for the handling of array antenna data. Array antenna models are usually required in Ray-Tracing applications for electromagnetig propagation modelling or in geometry-based stochastic channel models such as QuaDRiGa (https://quadriga-channel-model.de). Such simulation tools are used to determine the performance of new digital-radio technologies in either stochastic or deterministic propagation environments. 

An antenna model describes the radiated power, phase and polarization as a function of the angle from which the antenna is seen. In addition, modern radio-communication systems use multi-element antennas for MIMO communications (so-called array antennas). The array antenna is then defined by the set of all directional responses of its individual elements, also known as radiation pattern.

This library implements an interface to both, Matlab and Octave as well as to c++. It is currently in an early stage of development. Only the array antenna interpolation in polar-spheric basis is implemented. However, this is the most computing intense operation since it is performed millions of times in a typical radio simulation. Using this library, e.g. in QuaDRiGa can significantly improve the performance.

## Software Structure
### Folders

| Folder | Content |
|:---|:---|
`+quadriga_lib` | Compiled mex files for usage in MATLAB and Octave
`api_mex`       | Source files for the MATLAB / Octave MEX interface
`api_python`    | Source files for the Python API
`build`         | Folder for temporary build files
`external`      | External tools used in the project
`html_docu`     | Documentation of the API functions
`include`       | Public header files for the `quadriga-lib` library
`lib`           | Library files for static linking
`references`    | Relevant external documents and papers
`release`       | Folder for source packages and compiled binary packages
`src`           | C++ source files and private header files
`tests`         | Test files
`tools`         | Other tools used in the project

## Installation

Precompiled versions for Linux (MATLAB, Octave) and Windows (MATLAB only) are already included in the `release` folder. To use them e.g. in QuaDRiGa-NG (https://github.com/stjaeckel/QuaDRiGa-NG), simply add the arrayant_lib folder to your MATLAB/Octave path.

### Linux / Ubuntu

* [**OPTIONAL**]: Install Matlab from (https://www.mathworks.com) - you need to obtain a license
* Install dependencies (Python3 and Octave are optional): 
```
sudo apt install bzip2 gcc git make cmake g++ libhdf5-dev python3-dev python3-pytest python3-numpy octave-dev
```
* Clone quadiga-lib from GitHub: `git clone https://github.com/stjaeckel/quadriga-lib`
* Change to quadriga-lib folder: `cd quadriga-lib`
* Build external modules: `make external`
* Build Quadriga-Lib: `make -j16`
* [**OPTIONAL**]: Generate documentation: `make documentation`
* [**OPTIONAL**]: Run tests: `make test`



### Windows
* Octave is not supported (only MATLAB is)
* Install Build Tools for Visual Studio 2022 from https://visualstudio.microsoft.com
* Install Matlab from (https://www.mathworks.com) - you need to obtain a licence
* From MATLAB Shell run "mex -setup -v" and select compiler MSVC Compiler
* If error: https://yingzhouli.com/posts/2020-06/mex-msvc.html
* Open "x64 Native Tools Command Prompt" from start menu
* Navigate to library path, e.g. `cd Z:\quadriga-lib`
* Build external modules: `nmake external`
* Run `nmake` to compile `quadriga-lib` and the MEX Matlab interface
* [**OPTIONAL**]: Compile and run the unit tests by running `nmake test`


## Distribution License

**`quadriga-lib`** can be used in both open-source and proprietary (closed-source) software.

**`quadriga-lib`** is licensed under the Apache License, Version 2.0 (the "License").
A copy of the License is included in the "LICENSE" file. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
