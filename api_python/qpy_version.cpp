// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python.hpp"
#include "quadriga_lib.hpp"



/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# version
Returns the quadriga-lib version number

## Usage:
```
version = quadriga_lib.version();
```

## Caveat:
- If Quadriga-Lib was compiled with AVX2 support and the CPU supports intrinsic AVX2 instructions,
  an suffix `_AVX2` is added after the version number
MD!*/

std::string version()
{
    std::string quadriga_lib_version = quadriga_lib::quadriga_lib_version();

    if (quadriga_lib::quadriga_lib_has_AVX2())
        quadriga_lib_version += "_AVX2";

    return quadriga_lib_version;
}

/*!MD
# components
Returns the version numbers of all quadriga-lib sub-components

## Usage:
```
components = quadriga_lib.components()
```
MD!*/

std::string components()
{
    std::ostringstream oss;
    oss << "quadriga-lib = " << quadriga_lib::quadriga_lib_version() << std::endl;
    oss << "Armadillo    = " << quadriga_lib::quadriga_lib_armadillo_version() << std::endl;
    std::string py_version_full = Py_GetVersion();
    std::string py_version_short = py_version_full.substr(0, py_version_full.find(' '));
    oss << "Python       = " << py_version_short << std::endl;

    oss << "pybind11     = " << PYBIND11_VERSION_MAJOR << "." << PYBIND11_VERSION_MINOR << "." << PYBIND11_VERSION_PATCH << std::endl;

    try
    {
        py::module_ np = py::module_::import("numpy");
        std::string np_version = py::str(np.attr("__version__"));
        oss << "NumPy        = " << np_version << std::endl;
    }
    catch (const py::error_already_set &e)
    {
        oss << "NumPy        = not available (" << e.what() << ")" << std::endl;
    }

    return oss.str();
}
