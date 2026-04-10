/*!SECTION
Installation from Binaries
SECTION!*/

/*!SECTION_DESC

<br><br>

**Ubuntu**<br>
- Download the appropriate binary package for your distribution (see above)
- Install the package using `dpkg` (replace `<version>` and `<distro>` with the appropriate values):
```
sudo dpkg -i quadriga-lib-<version>+<distro>_amd64.deb
```
- [**OPTIONAL**]: Add the library to your MATLAB path by running `addpath('/usr/share/quadriga-lib')` in the MATLAB command window or by adding it to your `startup.m` file.

<br>

**Windows**<br>
- Extract the downloaded ZIP file to a location of your choice, e.g. `C:\quadriga-lib`
- [**OPTIONAL**]: Add the library to your MATLAB path by running `addpath('C:\quadriga-lib')` in the MATLAB command window or by adding it to your `startup.m` file.
- [**OPTIONAL**]: Install the Python wheel matching your Python version (3.10–3.14) (Replace `cp312` with your Python version (`cp310`, `cp311`, `cp312`, `cp313`, or `cp314`). To check your Python version run `python --version`)
```
pip install wheels\quadriga_lib-<version>-cp312-cp312-win_amd64.whl
```

<br>

**Python**<br>
- Available via PyPI, install using pip:
```
pip install quadriga-lib
```
- Prebuilt wheels are available for Linux x86_64, Linux aarch64, and Windows x86_64 (Python 3.10–3.14). On other platforms, pip will attempt to build from source (requires CMake and a C++17 compiler; may not work on all platforms).

<br>

**C++ Integration**<br>
- The binary packages include public header files (in `include/`) and static libraries (in `lib/`). Link against `quadriga.lib` (or `libquadriga.a` on Linux) and `libhdf5` in your build system of choice. The bundled <a target="_blank" rel="noopener noreferrer" href="https://arma.sourceforge.net">Armadillo</a> headers are included.

SECTION_DESC!*/
