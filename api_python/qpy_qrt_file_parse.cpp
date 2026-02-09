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
# qrt_file_parse
Read metadata from a QRT file

## Usage:
```
from quadriga_lib import channel

# Separate outputs
no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version = channel.qrt_file_parse( fn )

# Output as tuple
data = channel.qrt_file_parse( fn )
```

## Input Argument:
- **`fn`**<br>
  Filename of the QRT file, string

## Output Arguments:
- **`no_cir`**<br>
  Number of channel snapshots per origin point

- **`no_orig`**<br>
  Number of origin points (e.g., TXs)

- **`no_dest`**<br>
  Number of destinations (RX)

- **`no_freq`**<br>
  Number of frequencies

- **`cir_offset`**<br>
  CIR offset for each destination

- **`orig_names`**<br>
  Names of the origin points (TXs), list of strings

- **`dest_names`**<br>
  Names of the destination points (RXs), list of strings

- **`version`**<br>
  QRT file version
MD!*/

py::tuple qrt_file_parse(const std::string &fn)
{
    arma::uword no_cir, no_orig, no_dest, no_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;

    quadriga_lib::qrt_file_parse(fn, &no_cir, &no_orig, &no_dest, &no_freq, &cir_offset, &orig_names, &dest_names, &version);

    auto cir_offset_p = qd_python_copy2numpy(cir_offset);
    auto orig_names_p = qd_python_copy2python(orig_names);
    auto dest_names_p = qd_python_copy2python(dest_names);

    return py::make_tuple(no_cir, no_orig, no_dest, no_freq, cir_offset_p, orig_names_p, dest_names_p, version);
}