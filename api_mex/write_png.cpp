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

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

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
quadriga_lib.write_png( fn, data, colormap, min_val, max_val, log_transform )
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string fn = qd_mex_get_string(prhs[0]);

    arma::mat data;
    if (mxIsDouble(prhs[1]))
        data = qd_mex_reinterpret_Mat<double>(prhs[1]); // No copy
    else
        data = qd_mex_typecast_Mat<double>(prhs[1]);

    std::string colormap = (nrhs < 3) ? "jet" : qd_mex_get_string(prhs[2], "jet");
    double min_val = (nrhs < 4) ? NAN : qd_mex_get_scalar<double>(prhs[3], "min_val", NAN);
    double max_val = (nrhs < 5) ? NAN : qd_mex_get_scalar<double>(prhs[4], "max_val", NAN);
    bool log_transform = (nrhs < 6) ? false : qd_mex_get_scalar<bool>(prhs[5], "log_transform", false);

    CALL_QD(quadriga_lib::write_png(data, fn, colormap, min_val, max_val, log_transform));

    // Dummy output
    if (nlhs == 1)
    {
        double tmp = 1.0;
        plhs[0] = qd_mex_copy2matlab(&tmp);
    }
}