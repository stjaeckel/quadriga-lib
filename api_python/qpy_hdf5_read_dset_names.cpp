// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_read_dset_names
Read the names of unstructured data fields from an HDF5 file

## Description:
Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition
to structured datasets, the library facilitates the inclusion of extra datasets of various types
and shapes. This feature is particularly beneficial for integrating descriptive data or analysis
results. Users can add any number of such unstructured datasets, each identified by a unique
dataset name. The function `quadriga_lib.channel.hdf5_read_dset_names` retrieves the names of all 
these datasets, returning them as a list of strings.

## Usage:
```
from quadriga_lib import channel
names = channel.hdf5_read_dset_names( fn, ix, iy, iz, iw );
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