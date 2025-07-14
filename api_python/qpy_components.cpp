// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# COMPONENTS
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
    oss << "HDF5         = " << quadriga_lib::get_HDF5_version() << std::endl;
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