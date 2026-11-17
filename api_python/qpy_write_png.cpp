// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

#include "python_quadriga_adapter.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# write_png
Write data to a PNG file

- Converts input data into a color-coded PNG file for visualization
- Support optional selection of a colormap, as well a minimum and maximum value limits
- Uses the <a href="https://github.com/lvandeve/lodepng">LodePNG</a> library for PNG writing
- The auto-detected range ignores `NaN` and `+-inf`; `inf` and `-inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value) maps every value to the first colour
- Raises `ValueError` for empty `data`, an infinite `min_val` or `max_val`, and for `min_val > max_val` (both given)

## Usage:
```
quadriga_lib.tools.write_png( fn, data, colormap, min_val, max_val, log_transform )
```

## Arguments:
- **`fn`**<br>
  Filename of the PNG file, required

- **`data`**<br>
  Data matrix, required, size `[N, M]` 

- **`colormap`** (optional)<br>
  Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
  'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', optional, default = 'jet'

- **`min_val`** (optional)<br>
  Lower clipping bound; values below get the first colour. NaN (default): data minimum.

- **`max_val`** (optional)<br>
  Upper clipping bound; values above get the last colour. NaN (default): data maximum.

- `**log_transform**` (optional)<br>
  If enabled, the `data` values are transformed to the log-domain (`10*log10(data)`) before processing.
  Default: false (disabled)
MD!*/

void write_png(const py::handle &fn,            // Output file name
               const py::array_t<double> &data, // Data matrix
               const std::string &colormap,     // Colormap
               double min_val,                  // Lower clipping bound, NaN = data minimum
               double max_val,                  // Upper clipping bound, NaN = data maximum
               py::handle log_transform)        // Transform data to log-domain (10*log10(data))
{
    const std::string fn_s = qd_python_path(fn, "fn");

    const bool log_transform_b = qd_python_flag(log_transform, false, "log_transform");

    auto data_arma = qd_python_numpy2arma_Mat(data, true);
    quadriga_lib::write_png(data_arma, fn_s, colormap, min_val, max_val, log_transform_b);
}