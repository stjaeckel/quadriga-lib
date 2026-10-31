/*!SECTION
About the Library
SECTION!*/

/*!SECTION_DESC
- **How is Quadriga-Lib related to QuaDRiGa?**<br>
  <a target="_blank" rel="noopener noreferrer" href="https://quadriga-channel-model.de">QuaDRiGa</a> is a geometry-based stochastic channel model written in MATLAB. Quadriga-Lib is its C++ successor by the same lead developer: the model components are being rewritten in C++ for speed, with thin interfaces to MATLAB, Octave and Python. It is not a drop-in replacement for the MATLAB toolbox; functions, names and file formats follow their own API. <a target="_blank" rel="noopener noreferrer" href="https://github.com/stjaeckel/QuaDRiGa-NG">QuaDRiGa-NG</a> is a fork of the MATLAB QuaDRiGa toolbox that uses Quadriga-Lib in its backend.

- **How do I cite Quadriga-Lib?**<br>
  If you use Quadriga-Lib in published work, please cite the software and the version you used. The installed version is returned by `quadriga_lib.version()` in Python, `quadriga_lib.version` in MATLAB / Octave and `quadriga_lib::quadriga_lib_version()` in C++. Publications on the models are listed on the <a href="publications.html">Publications</a> page.
```
@software{quadriga_lib,
  author  = {Jaeckel, Stephan},
  title   = {Quadriga-Lib: C++/MATLAB/Python library for radio channel modelling and simulation},
  url     = {https://github.com/stjaeckel/quadriga-lib},
  version = {<version>},
  year    = {<year>}
}
```

- **Is the API stable?**<br>
  Quadriga-Lib is under active development. Breaking changes are avoided where possible, but they do happen between releases. For reproducible results, use a release version and note it, or check out a release tag of the <a target="_blank" rel="noopener noreferrer" href="https://github.com/stjaeckel/quadriga-lib">GitHub repository</a>. The `main` branch may be ahead of the latest release. The API documentation on this site is generated from the source code of the latest release.

- **Which compilers are supported?**<br>
  Quadriga-Lib needs C++17. It is built and tested with GCC 11 or newer (GCC 12 by default on Linux) and with MSVC 2022 on Windows.

- **MATLAB reports a "GLIBCXX ... not found" error when loading the MEX files. What can I do?**<br>
  MATLAB ships its own, older C++ runtime (`libstdc++`). MEX files built with a newer GCC than your MATLAB release supports fail to load with this error. The build uses GCC 12 by default. If your MATLAB release needs a different version, install it (e.g. `sudo apt install g++-11`) and build with `make gcc_version=11`. The compilers supported by each MATLAB release are listed by MathWorks.

- **Can Quadriga-Lib be used together with other C++ libraries?**<br>
  Yes. Quadriga-Lib uses <a target="_blank" rel="noopener noreferrer" href="https://arma.sourceforge.net">Armadillo</a> for its matrices and cubes. To exchange data with other libraries, create Armadillo objects that use auxiliary memory, use their STL-style iterators, or get a pointer to the matrix memory with `.memptr()`.
SECTION_DESC!*/
