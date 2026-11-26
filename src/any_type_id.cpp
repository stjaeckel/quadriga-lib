// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "qd_helpers_channel.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# any_type_id
Get the type ID and the shape of the value in a `std::any` object

- Tells which of the supported types a `std::any` holds, e.g. an unstructured dataset of [[hdf5_read_dset]] or an
  entry of `par_data` of a [[channel]], so that `std::any_cast` to that type gives the value
- The ID of a numeric value is $10 m + k$ with the container $m$ and the element type $k$:<br><br>
  | $m$ | Container | $k$ | Element type |
  | --- | --------- | --- | ------------ |
  | 1 | scalar | 0 | `float` |
  | 2 | `arma::Mat` | 1 | `double` |
  | 3 | `arma::Cube` | 2 | `unsigned long long` (`arma::uword`) |
  | 4 | `arma::Col` | 3 | `long long` (`arma::sword`) |
  | 5 | `arma::Row` | 4 | `unsigned` |
  | | | 5 | `int` |

  e.g. 21 is `arma::Mat<double>`; 9 is `std::string`, -1 a type not listed, -2 no value

## Declaration:
```
int quadriga_lib::any_type_id(
    const std::any *data,
    unsigned long long *dims = nullptr);
```

## Input:
- **`data`** — Object to inspect; `nullptr` gives -2

## Output:
- **`dims`** *(optional)* — Shape `[rows, cols, slices]`: `[1, 1, 1]` for a scalar or a type not listed,
  `[n_elem, 1, 1]` for `arma::Col` and `arma::Row`, `[length, 0, 0]` for a string; not changed for -2; 3 values

## Returns:
- Type ID

## See also:
- [[hdf5_read_dset]] (returns such objects)
- [[hdf5_write_dset]] (stores them)
MD!*/

int quadriga_lib::any_type_id(const std::any *par_data, unsigned long long *dims)
{
    return qd_any_type_id(par_data, dims);
}
