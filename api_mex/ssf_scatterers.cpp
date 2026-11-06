// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include <cstring>
#include "quadriga_lib.hpp"
#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# SSF_SCATTERERS
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

- Places the scatterers that the mobility `drift` needs and returns the validity scales of [[ssf_paths]]
- NLOS rays with power > 0 at some frequency: FBS on the departure ray and LBS on the arrival ray at equal distances
  from their end, so that the polyline Tx, FBS, LBS, Rx has the length d3D + c·delay (d3D the Tx-Rx distance) and the
  angles are exact
- Where no such pair exists with both distances at least `d_min`, both scatterers are placed at `d_min` along the ray's
  own directions and `fallback` is 1: the polyline is longer than the ray, `delay` keeps its value. With `d_min` = 2 m
  this applies to up to about 13 % of the rays (most in the TR methods without absolute TOA, where the first cluster
  has zero excess delay) and lengthens them by a few ns
- LOS ray and NLOS rays with zero power at all frequencies: the midpoint of the direct line; GR ray: the specular point
  on z = 0, which requires both ends at z >= 0
- `rays_out` is a copy of `rays` (all fields; unchanged fields keep their class) with new `fbs_pos`, `lbs_pos` and
  `fallback`; `rays` is not changed, also when it already holds scatterers
- `b_med`, the scale of `drift`: the lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS
  rays with power > 0, fallback rays included; NaN without NLOS power. `b_eff`, the scale of `phase`:
  `sqrt(sum P / sum(P / b²))` over the rays with power `P > 0`, with `b` the end-to-scatterer distance (NLOS), d3D
  (LOS) or the GR path length (GR)
- One link per call; loop over the elements of a struct array: `rays(l) = quadriga_lib.ssf_scatterers( rays(l) )`

## Usage:
```
[ rays_out, b_med, b_eff ] = quadriga_lib.ssf_scatterers( rays, d_min );
```

## Inputs:
- **`rays`** — Ray struct of one link, see [[ssf_rays]]; `[1, 1]` struct
- **`d_min`** *(optional)* — Minimum distance of the scatterers to both ends in [m], finite and > 0; default: 2

## Outputs:
- **`rays_out`** — Ray struct with `fbs_pos`, `lbs_pos` `[3, n_ray]` and `fallback` `[n_ray, 1]` (uint32)
- **`b_med`**, **`b_eff`** — Validity scales in [m], rows Tx and Rx, column `k` for frequency `k`; `[2, n_freq]`

## See also:
- [[ssf_rays]] (ray data of a set of links)
- [[ssf_paths]] (mobility `drift` needs the scatterers)
- [[gbsm_single_link_channel]] (reports `b_med`, `b_eff` and the displacement ratios)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'rays' must be a 1x1 struct.");

    // The input is borrowed (read-only); the solver writes into the three new fields of the output
    auto rays = qd_mex_struct2gbsm_rays(prhs[0]);
    const double d_min = (nrhs < 2) ? 2.0 : qd_mex_get_scalar<double>(prhs[1], "d_min", 2.0);
    const arma::uword n_ray = rays.n_ray(), n_freq = rays.n_freq();

    // Output struct with all fields: the solver outputs allocated, the other fields copied (empty where missing)
    plhs[0] = qd_mex_make_struct(qd_mex_gbsm_rays_fields, 1);
    for (const auto &f : qd_mex_gbsm_rays_fields)
    {
        mxArray *a;
        if (f == "fbs_pos")
            a = qd_mex_init_output(&rays.fbs_pos, 3, n_ray);
        else if (f == "lbs_pos")
            a = qd_mex_init_output(&rays.lbs_pos, 3, n_ray);
        else if (f == "fallback")
            a = qd_mex_init_output(&rays.fallback, n_ray);
        else // A plain copy of the data: mxDuplicateArray of the fields leaks memory in Octave
        {
            const mxArray *in = mxGetField(prhs[0], 0, f.c_str());
            if (in == nullptr)
                a = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
            else if (mxIsChar(in))
                a = mxCreateString(qd_mex_get_string(in).c_str());
            else // An empty cell or struct (non-empty ones are rejected when read) is copied as an empty double
            {
                a = mxIsLogical(in) ? mxCreateLogicalArray(mxGetNumberOfDimensions(in), mxGetDimensions(in))
                                    : mxCreateNumericArray(mxGetNumberOfDimensions(in), mxGetDimensions(in),
                                                           mxIsNumeric(in) ? mxGetClassID(in) : mxDOUBLE_CLASS, mxREAL);
                if (mxGetNumberOfElements(in) > 0)
                    std::memcpy(mxGetData(a), mxGetData(in), mxGetNumberOfElements(in) * mxGetElementSize(in));
            }
        }
        mxSetField(plhs[0], 0, f.c_str(), a);
    }

    CALL_QD(rays.solve_scatterers(d_min));

    // Validity scales per frequency
    arma::mat b_med, b_eff;
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&b_med, 2, n_freq);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&b_eff, 2, n_freq);
    for (arma::uword k = 0; k < n_freq; ++k)
    {
        if (nlhs > 1)
            CALL_QD(b_med.col(k) = rays.b_med(k));
        if (nlhs > 2)
            CALL_QD(b_eff.col(k) = rays.b_eff(k));
    }
}
