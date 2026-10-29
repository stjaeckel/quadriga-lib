# Quadriga-Lib

C++/Python library for radio channel modelling and simulations.

## Installation

```
pip install quadriga-lib
```

Prebuilt wheels are available for Linux (x86_64 and aarch64, Python 3.9–3.13).
On other platforms, pip will build from source (requires CMake and a C++17 compiler).

## Channel files of quadriga-lib 0.12.4 and earlier

The HDF5 channel files use file format version 3; files of version 2 (written by quadriga-lib 0.12.4 and
earlier) are rejected. Convert them once with the script [hdf5_convert_v2.py](https://github.com/stjaeckel/quadriga-lib/blob/main/tools/hdf5_convert_v2.py)
of the quadriga-lib sources:

```
pip install quadriga-lib h5py
python hdf5_convert_v2.py old.hdf5 new.hdf5
```

## Quick Start

```python
import quadriga_lib
print(quadriga_lib.components())
```

## Documentation

Full documentation is available at [quadriga-lib.org](http://quadriga-lib.org).
The Python API reference is at [quadriga-lib.org/python_api.html](http://quadriga-lib.org/python_api.html).

## Building from Source

If no prebuilt wheel is available for your platform:

```
sudo apt install cmake g++
pip install quadriga-lib
```

## Running Tests

```
pip install quadriga-lib[test]
pytest tests/python_tests -x -s
```

## License

Apache License, Version 2.0. See [LICENSE](https://github.com/stjaeckel/quadriga-lib/blob/main/LICENSE).
