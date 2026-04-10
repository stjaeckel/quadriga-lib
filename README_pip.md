# Quadriga-Lib

C++/Python library for radio channel modelling and simulations.

## Installation

```
pip install quadriga-lib
```

Prebuilt wheels are available for Linux (x86_64 and aarch64, Python 3.9–3.13).
On other platforms, pip will build from source (requires CMake and a C++17 compiler).

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
