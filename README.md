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
`build` | Folder for temporary build files
`external` | External tools used in the project
`include` | Public header files for the `quadriga-lib` library
`lib` | Library files for static linking
`mex` | Source files for the MEX interface
`release` | Precompiled versions of `quadriga-lib`
`src` | C++ source files and private header files
`tests` | Test files
`tools` | Other tools used in the project

## Compilation

Precompiled versions for Linux (MATLAB, Octave) and Windows (MATLAB only) are already included in the `release` folder. To use them e.g. in QuaDRiGa-NG (https://github.com/stjaeckel/QuaDRiGa-NG), simply add the arrayant_lib folder to your MATLAB/Octave path.

### Linux / Ubuntu

* [**OPTIONAL**]: Install Octave 8.0 (or higher) from https://octave.org or use the system package installer
* [**OPTIONAL**]: Install Matlab from (https://www.mathworks.com) - you need to obtain a license
* Open the `GNUmakefile` (e.g. by `gedit GNUmakefile`)
* Set the compiler paths for your system (`CC`, `MEX` and `OCT` variables)
* Remove the unwanted targets in the all section (e.g. if only octave is needed, remove `mex_matlab`)
* Compile the HDF5 library by running `make hdf5lib`
* Run `make` to compile `quadriga-lib` and the MEX Matlab / Octave interface
* [**OPTIONAL**]: Compile the Catch2 unit testing library by running `make catch2lib`
* [**OPTIONAL**]: Compile and run the unit tests by running `make test`


### Windows
* Octave is not supported (only MATLAB is)
* Install Build Tools for Visual Studio 2022 from https://visualstudio.microsoft.com
* Install Matlab from (https://www.mathworks.com) - you need to obtain a licence
* From MATLAB Shell run "mex -setup -v" and select compiler MSVC Compiler
* If error: https://yingzhouli.com/posts/2020-06/mex-msvc.html
* Open "x64 Native Tools Command Prompt" from start menu
* Navigate to library path, e.g. `cd Z:\quadriga-lib`
* Open the `Makefile` (e.g. by `notepad Makefile`) set the correct MATLAB mex path 
* Compile the HDF5 library by running `nmake hdf5lib`
* Run `nmake` to compile `quadriga-lib` and the MEX Matlab interface
* [**OPTIONAL**]: To compile the Catch2 library, you need NSIS (https://nsis.sourceforge.io)
* [**OPTIONAL**]: Compile the Catch2 unit testing library by running `nmake catch2lib`
* [**OPTIONAL**]: Compile and run the unit tests by running `nmake test`


### Pugixml Library
* pugixml (https://pugixml.org) is used to read/write QDANT-Files
* It can be used in "header-only" mode
* Extract the "`pugixml-1.13.tar.gz`" into external
* Navigate to "`external/pugixml-1.13/src`" and open the file "`pugiconfig.hpp`"
* Uncomment the line "`#define PUGIXML_HEADER_ONLY`"
* Make sure the correct path is set in the "GNUMakefile" (Linux) and/or "Makefile" (Windows)

## Distribution License

**`quadriga-lib`** can be used in both open-source and proprietary (closed-source) software.

**`quadriga-lib`** is licensed under the Apache License, Version 2.0 (the "License").
A copy of the License is included in the "LICENSE" file. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.