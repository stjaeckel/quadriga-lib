// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# EXPORT_PATTERN_OBJ
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- Writes the power pattern of each element as a colored 3D surface, one object per element, placed at its position in
  `element_pos`
- The distance of the surface from the element is linear in the level in dB: `object_radius` at the peak over all
  exported elements, 0 at `directivity_range_dB` below it. The color shows the same level

## Usage:
```
quadriga_lib.export_pattern_obj( fn, arrayant, directivity_range_dB, colormap, object_radius, ...
    icosphere_n_div, i_element, i_freq );
```

## Inputs:
- **`fn`** — Output OBJ file name, must end in `.obj`; the colormap is written to a `.mtl` file of the same name;
  existing files are overwritten
- **`arrayant`** — Arrayant struct ([[generate_antenna]]), or 1xN struct array for a multi-frequency array; the exported
  elements must have finite field values and must not be zero in all directions
- **`directivity_range_dB`** — Range of the plot below the peak in dB, finite and not 0; the sign is ignored; default:
  30
- **`colormap`** — Colormap name: `jet`, `parula`, `winter`, `hot`, `turbo`, `copper`, `spring`, `cool`, `gray`,
  `autumn` or `summer`; default: `jet`
- **`object_radius`** — Radius of the surface at the peak in m, positive and finite; default: 1
- **`icosphere_n_div`** — Subdivisions n of the [[icosphere]] that carries the surface; higher = finer mesh with 20 n²
  faces; default: 4
- **`i_element`** — Element indices to export, 1-based; default: all elements
- **`i_freq`** — Entry of a multi-frequency array to export, 1-based, at most `numel(arrayant)`; default: 1

## See also:
- [[calc_directivity]], [[calc_beamwidth]] (the same pattern as numbers)
- [[channel_export_obj_file]] (exports propagation paths to an OBJ file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs; // No outputs

    if (nrhs < 2 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    qd_mex_check_struct(prhs[1], "arrayant");

    std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    double directivity_range_dB = (nrhs < 3) ? 30.0 : qd_mex_get_scalar<double>(prhs[2], "directivity_range_dB", 30.0);
    std::string colormap = (nrhs < 4) ? "jet" : qd_mex_get_string(prhs[3], "jet", "colormap");
    double object_radius = (nrhs < 5) ? 1.0 : qd_mex_get_scalar<double>(prhs[4], "object_radius", 1.0);
    arma::uword icosphere_n_div = (nrhs < 6) ? 4 : qd_mex_get_scalar<arma::uword>(prhs[5], "icosphere_n_div", 4);

    const arma::uvec element_ind = (nrhs < 7) ? arma::uvec() : qd_mex_get_index_Col(prhs[6], "i_element"); // 1-based -> 0-based

    size_t n_freq = (size_t)mxGetNumberOfElements(prhs[1]);
    const size_t i_freq = (nrhs < 8) ? 0 : qd_mex_get_index<size_t>(prhs[7], "i_freq"); // 1-based -> 0-based
    if (i_freq >= n_freq)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'i_freq' is out of range.");

    // Load arrayant (single entry for the requested frequency)
    auto ant = quadriga_lib::arrayant<double>();
    if (n_freq > 1)
    {
        auto ant_multi = qd_mex_struct2arrayant_multi(prhs[1], true, false, "arrayant");
        ant = ant_multi[i_freq];
    }
    else
        ant = qd_mex_struct2arrayant(prhs[1], true, false, "arrayant");

    CALL_QD(quadriga_lib::export_pattern_obj(fn, ant, directivity_range_dB, colormap, object_radius, icosphere_n_div, element_ind));
}