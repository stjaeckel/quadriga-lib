/*!SECTION
Installation from Binaries
SECTION!*/

/*!SECTION_DESC

<br><br>

**Python**<br>
- Install from PyPI:
```
pip install quadriga-lib
```
- Prebuilt wheels exist for Linux x86_64, Linux aarch64 and Windows x86_64 (Python 3.10–3.14). On other platforms, pip tries to build from source, which needs CMake and a C++17 compiler.

<br>

**Ubuntu**<br>
- Download the package for your Ubuntu release (see above) and install it with `dpkg` (replace `<version>` and `<distro>`):
```
sudo dpkg -i quadriga-lib_<version>-1+<distro>_amd64.deb
```
- MATLAB / Octave: add the library to the path with `addpath('/usr/share/quadriga-lib')` in the command window or in your `startup.m` file.

<br>

**Windows**<br>
- Extract the ZIP file to a folder of your choice, e.g. `C:\quadriga-lib`. It contains the MEX files, the static library, the headers and Python wheels.
- MATLAB: add the library to the path with `addpath('C:\quadriga-lib')` in the command window or in your `startup.m` file.
- Python (optional): install the wheel that matches your Python version. Replace `cp312` with your version (`cp310` to `cp314`); `python --version` shows it.
```
pip install wheels\quadriga_lib-<version>-cp312-cp312-win_amd64.whl
```

<br>

**C++**<br>
- The binary packages contain the public headers and the static library (`libquadriga.a` on Linux, `quadriga.lib` on Windows), together with the <a target="_blank" rel="noopener noreferrer" href="https://arma.sourceforge.net">Armadillo</a> headers. Link the static library and enable OpenMP (`-fopenmp` with GCC).

SECTION_DESC!*/
