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
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_DSET
Read a single unstructured dataset from an HDF5 file

## Description:
Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition
to structured datasets, the library facilitates the inclusion of extra datasets of various types
and shapes. This feature is particularly beneficial for integrating descriptive data or analysis
results. The function `quadriga_lib.channel.hdf5_read_dset` retrieves a single unstructured dataset.
Theo output type of the function is defined by the datatype in the file. An empty matrix is returned
if the dataset does not exist in the file.

## Usage:
```
from quadriga_lib import channel
dset = channel.hdf5_read_dset( fn, ix, iy, iz, iw, name )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`ix`**<br>
  Storage index for x-dimension, Default = 0

- **`iy`**<br>
  Storage index for y-dimension, Default = 0

- **`iz`**<br>
  Storage index for z-dimension, Default = 0

- **`iw`**<br>
  Storage index for w-dimension, Default = 0

- **`name`**<br>
  Name of the dataset; String

## Output Argument:
- **`dset`**<br>
  Output data. Type and size is defined by the dataspace in the file

## Caveat:
- Only datasets that are present in the HDF file are returned in the dictionary.
MD!*/

py::array hdf5_read_dset(const std::string &fn,
                         unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                         const std::string &name)
{
    // Read dataset
    std::any dset = quadriga_lib::hdf5_read_dset(fn, name, ix, iy, iz, iw);
    int type_id = quadriga_lib::any_type_id(&dset);

    // Convert data to python type
    switch (type_id)
    {
    case 10:
        return py::float_(std::any_cast<float>(dset));
    case 11:
        return py::float_(std::any_cast<double>(dset));
    case 12:
        return py::int_(std::any_cast<unsigned long long int>(dset));
    case 13:
        return py::int_(std::any_cast<long long int>(dset));
    case 14:
        return py::int_(std::any_cast<unsigned int>(dset));
    case 15:
        return py::int_(std::any_cast<int>(dset));
    case 20:
        return qd_python_copy2numpy(std::any_cast<arma::Mat<float>>(dset));
    case 21:
        return qd_python_copy2numpy(std::any_cast<arma::Mat<double>>(dset));
    case 22:
        return qd_python_copy2numpy(std::any_cast<arma::Mat<unsigned long long>>(dset));
    case 23:
        return qd_python_copy2numpy(std::any_cast<arma::Mat<long long>>(dset));
    case 24:
        return qd_python_copy2numpy(std::any_cast<arma::Mat<unsigned>>(dset));
    case 25:
        return qd_python_copy2numpy(std::any_cast<arma::Mat<int>>(dset));
    case 30:
        return qd_python_copy2numpy(std::any_cast<arma::Cube<float>>(dset));
    case 31:
        return qd_python_copy2numpy(std::any_cast<arma::Cube<double>>(dset));
    case 32:
        return qd_python_copy2numpy(std::any_cast<arma::Cube<unsigned long long>>(dset));
    case 33:
        return qd_python_copy2numpy(std::any_cast<arma::Cube<long long>>(dset));
    case 34:
        return qd_python_copy2numpy(std::any_cast<arma::Cube<unsigned>>(dset));
    case 35:
        return qd_python_copy2numpy(std::any_cast<arma::Cube<int>>(dset));
    case 40:
        return qd_python_copy2numpy(std::any_cast<arma::Col<float>>(dset));
    case 41:
        return qd_python_copy2numpy(std::any_cast<arma::Col<double>>(dset));
    case 42:
        return qd_python_copy2numpy(std::any_cast<arma::Col<unsigned long long>>(dset));
    case 43:
        return qd_python_copy2numpy(std::any_cast<arma::Col<long long>>(dset));
    case 44:
        return qd_python_copy2numpy(std::any_cast<arma::Col<unsigned>>(dset));
    case 45:
        return qd_python_copy2numpy(std::any_cast<arma::Col<int>>(dset));
    case -2:
        throw std::runtime_error("Dataset '" + name + "' does not exist in file.");
    default:
        throw std::runtime_error("Dataset '" + name + "' has an unsupported type.");
    }

    // Default return type, should never be called
    return py::int_(0);
}
