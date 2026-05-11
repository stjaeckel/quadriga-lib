// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_EXPORT_OBJ_FILE
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives a finer mesh
- Supports multi-frequency arrayant models: pass a struct array and select the entry to export
  via the `freq` argument

## Usage:
```
quadriga_lib.arrayant_export_obj_file( fn, arrayant, directivity_range, colormap,  object_radius, ...
   icosphere_n_div, i_element, i_freq );
```

## Inputs:
- **`fn`** — Output OBJ filename; must not be empty; filename must end in `.obj`
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in
  [[arrayant_generate]]; a struct array may contain a frequency-dependent model
- **`directivity_range`** *(optional)* — Dynamic range of the visualized directivity pattern in dB; default: 30
- **`colormap`** *(optional)* — Colormap name; default: jet; Available: jet, parula, winter, 
  hot, turbo, copper, spring, cool, gray, autumn, summer
- **`object_radius`** *(optional)* — Radius of the exported object; default: 1
- **`icosphere_n_div`** *(optional)* — Icosphere subdivision count; higher = finer mesh; see [[icosphere]]; default: 4
- **`i_element`** *(optional)* — Element indices to export; 1-based; uint64; empty = export all elements
- **`i_freq`** *(optional)* — Frequency index to export from a multi-frequency arrayant struct
  array; 1-based; uint64; default: 1; must satisfy `1 <= freq <= n_freq`

## See also:
- [[icosphere]] (icosphere primitive used internally)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 2 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    if (!mxIsStruct(prhs[1]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant' must be a struct.");

    // Parse scalar / string inputs
    std::string fn = qd_mex_get_string(prhs[0]);
    double directivity_range = (nrhs < 3) ? 30.0 : qd_mex_get_scalar<double>(prhs[2], "directivity_range", 30.0);
    std::string colormap = (nrhs < 4) ? "jet" : qd_mex_get_string(prhs[3], "jet");
    double object_radius = (nrhs < 5) ? 1.0 : qd_mex_get_scalar<double>(prhs[4], "object_radius", 1.0);
    arma::uword icosphere_n_div = (nrhs < 6) ? 4 : qd_mex_get_scalar<arma::uword>(prhs[5], "icosphere_n_div", 4);

    // i_element: 1-based input, convert to 0-based
    arma::uvec element_ind = (nrhs < 7) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[6], "i_element");
    if (!element_ind.empty())
    {
        if (arma::any(element_ind == 0))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'i_element' cannot be 0 (1-based index).");
        element_ind -= 1;
    }

    // freq: 1-based frequency index for multi-frequency models
    size_t n_freq = (size_t)mxGetNumberOfElements(prhs[1]);
    size_t freq = (nrhs < 8) ? 1 : qd_mex_get_scalar<size_t>(prhs[7], "freq", 1);
    if (freq < 1 || freq > n_freq)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'freq' is out of bound.");

    // Load arrayant (single entry for the requested frequency)
    auto ant = quadriga_lib::arrayant<double>();
    if (n_freq > 1)
    {
        auto ant_multi = qd_mex_struct2arrayant_multi(prhs[1], true);
        ant = ant_multi[freq - 1];
    }
    else
        ant = qd_mex_struct2arrayant(prhs[1], true);

    CALL_QD(ant.export_obj_file(fn, directivity_range, colormap, object_radius, icosphere_n_div, element_ind));

    double out = 1.0;
    if (nlhs == 1)
        plhs[0] = qd_mex_copy2matlab(&out);
}
