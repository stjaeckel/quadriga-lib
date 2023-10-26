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
`mex` | Source files for the MEX interface
`src` | C++ source files and private header files
`tests` | Test files
`tools` | Other tools used in the project

## Compilation

Precompiled versions for Linux (MATLAB, Octave) and Windows (MATLAB only) are already included in the `+quadriga_lib` folder. To use them e.g. in QuaDRiGa-NG (https://github.com/stjaeckel/QuaDRiGa-NG), simply add the arrayant_lib folder to your MATLAB/Octave path.

### Linux / Ubuntu
* You need to install "`libhdf5-dev`" from the system package manager
* Optional: Install Octave from https://octave.org or use the system package installer
* Optional: Install Matlab from (https://www.mathworks.com) - you need to obtain a license
* Open the `GNUmakefile` (e.g. by `gedit GNUmakefile`)
* Set the compiler paths for your system (`CC`, `MEX` and `OCT` variables)
* Remove the unwanted targets in the all section (e.g. if only octave is needed, remove `mex_matlab`)
* Run `make`


### Windows
* Octave is not supported (only MATLAB is)
* Install Build Tools for Visual Studio 2022 from https://visualstudio.microsoft.com
* Install Matlab from (https://www.mathworks.com) - you need to obtain a licence
* From MATLAB Shell run "mex -setup -v" and select compiler MSVC Compiler
* If error: https://yingzhouli.com/posts/2020-06/mex-msvc.html
* Open "x64 Native Tools Command Prompt" from start menu
* Navigate to library path, e.g. `cd Z:\quadriga-lib`
* Open the `Makefile` (e.g. by `notepad Makefile`) set the correct MATLAB mex path 
* Run `nmake`

### Armadillo Library
* Armadillo is a high quality linear algebra library used by `quadriga-lib`
* Armadillo is a "header-only" library and does not need to be compiled
* Extract the "include"-folder from the armadillo zip-file from "https://arma.sourceforge.net" into external
* Make sure the correct path is set in the "GNUMakefile" (Linux) and/or "Makefile" (Windows)
* Compiling for Linux (optional):
```
cd external
mkdir build && mkdir armadillo-12.6.3-Linux
cmake -S armadillo-12.6.3 -B build/ -D CMAKE_INSTALL_PREFIX=armadillo-12.6.3-Linux -D BUILD_SHARED_LIBS=OFF
cd build && make && make install
cd .. && rm -rf build
```
* For Windows, you need the Build Tools for Visual Studio (see above)
* Open "x64 Native Tools Command Prompt" and run the following commands (optional)
```
cd external
mkdir build 
mkdir armadillo-12.6.3-win64
cmake -S armadillo-12.6.3 -B build -D CMAKE_INSTALL_PREFIX=armadillo-12.6.3-win64
cmake --build build --config Release --target install
```
### Pugixml Library
* pugixml (https://pugixml.org) is used to read/write QDANT-Files
* It can be used in "header-only" mode
* Extract the "`pugixml-1.13.tar.gz`" into external
* Navigate to "`external/pugixml-1.13/src`" and open the file "`pugiconfig.hpp`"
* Uncomment the line "`#define PUGIXML_HEADER_ONLY`"
* Make sure the correct path is set in the "GNUMakefile" (Linux) and/or "Makefile" (Windows)

### HDF5 Library
* Hierarchical Data Format version 5 (HDF5), is an open source file format that supports large, complex, heterogeneous data. It is used to store channel coefficients and metadata.
* If you intend to use Octave, you will need at least Octave 8. Older versions cause a crash with HDF5
* For Windows, you need to compile the library using Build Tools for Visual Studio (see above)
* Extract the hdf5 tar.gz-file from "https://arma.sourceforge.net" into external
* Open "x64 Native Tools Command Prompt" and run the following commands
```
cd external
mkdir build 
mkdir hdf5-1.14.2-win64
cmake -S hdf5-1.14.2 -B build -D CMAKE_INSTALL_PREFIX=hdf5-1.14.2-win64 -D BUILD_SHARED_LIBS=OFF -D HDF5_ENABLE_Z_LIB_SUPPORT=OFF
cmake --build build --config Release --target install
```
* Compiling for Linux:
```
cd external
mkdir build && mkdir hdf5-1.14.2-Linux
cmake -S hdf5-1.14.2 -B build/ -D CMAKE_INSTALL_PREFIX=hdf5-1.14.2-Linux -D BUILD_SHARED_LIBS=OFF -D HDF5_ENABLE_Z_LIB_SUPPORT=OFF -D BUILD_TESTING=OFF -D CMAKE_C_FLAGS=-O3
cd build && make -j32 && make install
cd .. && rm -rf build
```
* Make sure the correct path is set in the "GNUMakefile" (Linux) and/or "Makefile" (Windows)

### Catch2 (v3) Unit Test Framework
* This is only needed if you want to run the tests
* `quadriga-lib` uses the Catch2 v3 framework. This is not supported yet by all system distributions (e.g. ubuntu22.04).
* Follow these steps to compile Catch2 locally
* Extract the zip-file from "https://github.com/catchorg/Catch2" to external
* Compiling for Linux:
```
cd external
mkdir build && mkdir Catch2-3.3.2-Linux
cmake -S Catch2-3.3.2 -B build
cd build && make && make package
tar -xzf Catch2-*-Linux.tar.gz -C ../
cd .. && rm -rf build
```
* You should now be able to run `make test` from the `quadriga-lib` base Folder
* For Windows, you need the Build Tools for Visual Studio (see above) and NSIS (https://nsis.sourceforge.io)
* Open "x64 Native Tools Command Prompt" and run the following commands
```
cd external
mkdir build
cmake -S Catch2-3.3.2 -B build
cmake --build build --config Release --target package
```
* This creates an exe-File inside of the `build`-Folder that can be installed to the "Catch2-3.3.2-win64" folder in `external`
* Alternatively, you can extract the exe-File
* After this, you can delete the `build` and `Catch2-3.3.2` folders
* You should now be able to run `nmake test` from the `quadriga-lib` base Folder


## Distribution License

**`quadriga-lib`** can be used in both open-source and proprietary (closed-source) software.

**`quadriga-lib`** is licensed under the Apache License, Version 2.0 (the "License").
A copy of the License is included in the "LICENSE" file. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.