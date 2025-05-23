// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "quadriga_python_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_DSET_NAMES
Read the names of unstructured data fields from an HDF5 file

## Description:
Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition
to structured datasets, the library facilitates the inclusion of extra datasets of various types
and shapes. This feature is particularly beneficial for integrating descriptive data or analysis
results. Users can add any number of such unstructured datasets, each identified by a unique
dataset name. The function `quadriga_lib.hdf5_read_dset_names` retrieves the names of all these
datasets, returning them as a list of strings.

## Usage:

```
names = quadriga_lib.hdf5_read_dset_names( fn, ix, iy, iz, iw );
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

## Output Argument:
- **`names`**<br>
  List of names of all these at the given location in the files; Cell array of strings
MD!*/

py::list hdf5_read_dset_names(const std::string fn,
                              unsigned ix, unsigned iy, unsigned iz, unsigned iw)
{
    std::vector<std::string> par_names;
    quadriga_lib::hdf5_read_dset_names(fn, &par_names, ix, iy, iz, iw);

    py::list list;
    for (const auto &item : par_names)
        list.append(item);

    return list;
}