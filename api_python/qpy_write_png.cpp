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
# write_png
Write data to a PNG file

## Description:
- Converts input data into a color-coded PNG file for visualization
- Support optional selection of a colormap, as well a minimum and maximum value limits
- Uses the <a href="https://github.com/lvandeve/lodepng">LodePNG</a> library for PNG writing

## Declaration:
```
import quadriga_lib

quadriga_lib.tools.write_png( fn, data, colormap, min_val, max_val, log_transform )
```

## Arguments:
- **`fn`**<br>
  Filename of the PNG file, string, required

- **`data`**<br>
  Data matrix, required, size `[N, M]` 

- **`colormap`** (optional)<br>
  Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
  'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', optional, default = 'jet'

- **`min_val`** (optional)<br>
  Minimum value. Values below this value will have be encoded with the color of the smallest value.
  If `NAN` is provided (default), the lowest values is determined from the data.

- **`max_val`** (optional)<br>
  Maximum value. Values above this value will have be encoded with the color of the largest value.
  If `NAN` is provided (default), the largest values is determined from the data.

- `**log_transform**` (optional)<br>
  If enabled, the `data` values are transformed to the log-domain (`10*log10(data)`) before processing.
  Default: false (disabled)
MD!*/

void write_png(const std::string &fn,           // Output file name
               const py::array_t<double> &data, // Data matrix
               const std::string &colormap,     // Colormap
               double min_val,                  // Minimum value, when passing NAN, minimum in data is used
               double max_val,                  // Maximum value, when passing NAN, maximum data is used
               bool log_transform)              // Transform data to log-domain (10*log10(data))
{
    auto data_arma = qd_python_numpy2arma_Mat(data, true);
    quadriga_lib::write_png(data_arma, fn, colormap, min_val, max_val, log_transform);
}