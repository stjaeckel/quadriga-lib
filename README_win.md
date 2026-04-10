# Quadriga-Lib — Windows Binary Package

**Quadriga-Lib** is an open-source utility library for radio channel modelling and simulation. It generates realistic channel impulse responses for system-level studies of mobile radio networks and provides a rich toolset for antenna arrays, channel data handling, and site-specific ray-tracing simulations.

Full documentation is available at [quadriga-lib.org](http://quadriga-lib.org).

---

## Package Contents

| Folder / File    | Content                                                          |
| :--------------- | :--------------------------------------------------------------- |
| `include/`       | Public C++ header files (quadriga-lib and Armadillo)             |
| `lib/`           | Static library (`quadriga.lib`) and HDF5 library (`libhdf5.lib`) |
| `+quadriga_lib/` | Compiled MEX files and `.m` documentation for MATLAB             |
| `wheels/`        | Prebuilt Python wheels for Windows (cp310–cp314)                 |
| `html_docu/`     | Full offline API documentation                                   |
| `LICENSE`        | Apache License, Version 2.0                                      |

---

## MATLAB Setup

1. Open MATLAB.
2. Add the package folder to your MATLAB path:
   ```matlab
   addpath('C:\path\to\quadriga_lib-VERSION-win_amd64')
   savepath
   ```
   Replace `C:\path\to\quadriga_lib-VERSION-win_amd64` with the actual path where you extracted this package.
3. Verify the installation:
   ```matlab
   quadriga_lib.version()
   ```

To make the path persistent across MATLAB sessions, add the `addpath` call to your `startup.m` file.

---

## Python Setup

Prebuilt wheels are included in the `wheels\` folder for Python 3.10–3.14. Install the wheel matching your Python version:

```
pip install wheels\quadriga_lib-VERSION-cp312-cp312-win_amd64.whl
```

Replace `cp312` with your Python version (`cp310`, `cp311`, `cp312`, `cp313`, or `cp314`). To check your Python version run `python --version`.

Verify the installation:
```python
import quadriga_lib
print(quadriga_lib.components())
```

Alternatively, install directly from PyPI (requires internet access):
```
pip install quadriga-lib
```

---

## C++ Integration

The `include\` folder contains all public headers including the bundled Armadillo headers. Link against `lib\quadriga.lib` and `lib\libhdf5.lib` in your build system of choice.

---

## Requirements

- Windows 10 or later (x86_64)
- MATLAB R2021a or later (for MEX files)
- Python 3.10–3.14 (for Python wheels)
- Visual Studio 2019 or later runtime (for C++ integration)

---

## License

Apache License, Version 2.0. See the included `LICENSE` file or [github.com/stjaeckel/quadriga-lib](https://github.com/stjaeckel/quadriga-lib/blob/main/LICENSE).