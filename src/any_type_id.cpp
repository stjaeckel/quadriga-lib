// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "qd_helpers_channel.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# any_type_id
Get the type ID and the shape of the value in a `std::any` object

- Inspects a `std::any` object and returns an integer type identifier for its contents
- Optionally retrieves dimensions (rows, columns, slices) for Armadillo matrix/cube/vector types; for `std::string`, `dims[0]` is the string length, `dims[1]`/`dims[2]` are zero
- To access the data, cast the object with `std::any_cast` to the type given by the ID

## Declaration:
```
int quadriga_lib::any_type_id(
    const std::any *data,
    unsigned long long *dims = nullptr);
```

## Input:
- **`data`** — Pointer to the `std::any` object to inspect

## Output:
- **`dims`** *(optional)* — Array of 3 values filled with `[rows, cols, slices]` of the contained Armadillo object

## Returns:
- Integer type ID of the contained value:<br><br>
  | ID  | Type                      | ID  | Type                   | ID  | Type                      |
  | --- | ------------------------- | --- | ---------------------- | --- | ------------------------- |
  | -2  | `no value`                | -1  | `unsupported type`     | 9   | `std::string`             |
  | 10  | `float`                   | 11  | `double`               | 12  | `unsigned long long int`  |
  | 13  | `long long int`           | 14  | `unsigned int`         | 15  | `int`                     |
  | 20  | `arma::Mat<float>`        | 21  | `arma::Mat<double>`    | 22  | `arma::Mat<arma::uword>`  |
  | 23  | `arma::Mat<arma::sword>`  | 24  | `arma::Mat<unsigned>`  | 25  | `arma::Mat<int>`          |
  | 30  | `arma::Cube<float>`       | 31  | `arma::Cube<double>`   | 32  | `arma::Cube<arma::uword>` |
  | 33  | `arma::Cube<arma::sword>` | 34  | `arma::Cube<unsigned>` | 35  | `arma::Cube<int>`         |
  | 40  | `arma::Col<float>`        | 41  | `arma::Col<double>`    | 42  | `arma::Col<arma::uword>`  |
  | 43  | `arma::Col<arma::sword>`  | 44  | `arma::Col<unsigned>`  | 45  | `arma::Col<int>`          |
  | 50  | `arma::Row<float>`        | 51  | `arma::Row<double>`    | 52  | `arma::Row<arma::uword>`  |
  | 53  | `arma::Row<arma::sword>`  | 54  | `arma::Row<unsigned>`  | 55  | `arma::Row<int>`          |

## See also:
- [[hdf5_read_dset]] (uses `any_type_id` to read dataset from HDF5 file)
- [[hdf5_write_dset]] (HDF5 dataset writer)
MD!*/

int quadriga_lib::any_type_id(const std::any *par_data, unsigned long long *dims)
{
    return qd_any_type_id(par_data, dims);
}
