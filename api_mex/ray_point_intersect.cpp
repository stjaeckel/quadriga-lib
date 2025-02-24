// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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
#include "mex_helper_functions.cpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# RAY_POINT_INTERSECT
Calculates the intersection of ray beams with points in three dimensions

## Description:
Unlike traditional ray tracing, where rays do not have a physical size, beam tracing models rays as 
beams with volume. Beams are defined by triangles whose vertices diverge as the beam extends. This 
approach is used to simulate a kind of divergence or spread in the beam, reminiscent of how radio 
waves spreads as they travel from a point source. The volumetric nature of the beams allows for more 
realistic modeling of energy distribution. As beams widen, the energy they carry can be distributed 
across their cross-sectional area, affecting the intensity of the interaction with surfaces.
Unlike traditional ray tracing where intersections are line-to-geometry tests, beam tracing requires 
volumetric intersection tests.

Ray beams are determined by an origin point, three vectors pointing from the origin to the three 
vertices of a triangle that defines the shape of the tube and the three direction of the rays at 
the vertices.

## Usage:

```
[ hit_count, ray_ind ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, ...
    max_no_hit, sub_cloud_index, target_size );
```

## Input Arguments:
- **`orig`**<br>
  Ray origins in 3D Cartesian coordinates; Size: `[ no_ray, 3 ]`

- **`trivec`**<br>
  The 3 vectors pointing from the center point of the ray at the ray `origin` to the vertices of 
  a triangular propagation tube (the beam), the values are in the order 
  `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[ no_ray, 9 ]`

- **`tridir`**<br>
  The directions of the vertex-rays. Size: `[ n_ray, 9 ]`, Values must be given in Cartesian 
  coordinates in the order  `[ d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z  ]`; The vector does
  not need to be normalized.

- **`points`**<br>
  Points in 3D-Cartesian space; Size: `[ n_points_in, 3 ]`

- **`max_no_hit`** (optional)<br>
  Max. number of hits in the output `ray_ind`, default = 32

- **`sub_cloud_index`** (optional)<br>
  Start indices of the sub-clouds in 0-based notation. Type: uint32; Vector of length `[ n_sub_cloud ]` 
  If this optional input is not given, the sub-could index is calculated automatically. Passing a
  value of `uint32(0)` will disable the sub-cloud calculation. 

- **`target_size`** (optional)<br>
  Target value for the sub-cloud size, only evaluated if 'sub_cloud_index' is not given.

## Output Arguments:

- **`hit_count`**<br>
  Number of rays that hit a point, unit32, Length: `[ n_points ]`

- **`ray_ind`**<br>
  Ray indices that hit the points, 1-based, 0 = no hit, Size `[ n_points, max_no_hit ]`

MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - orig                Ray origin points in GCS, Size [ n_ray, 3 ]
    //  1 - trivec              Vectors pointing from the origin to the vertices of the propagation tube, Size [ n_ray, 9 ]
    //  2 - tridir              Directions of the vertex-rays; Cartesian format; Size [ n_ray, 9 ]
    //  3 - points              Points in 3D Space, Size: [ n_points, 3 ]
    //  4 - max_no_hit          Max. number of hits in the output "ray_ind", default = 32
    //  5 - sub_cloud_index     Sub-cloud index, 0-based, Optional, Length: [ n_sub ]
    //  6 - target_size         Target value for the sub-cloud size, only evaluated if 'sub_cloud_index' is not given

    // Outputs:
    //  0 - hit_count           Number of rays that hit a point, unit32, Length: [ n_points ]
    //  1 - ray_ind             Ray indices that hit the points, 1-based, 0 = no hit, Size [ n_points, max_no_hit ]

    if (nrhs < 4 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:ray_point_intersect:IO_error", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:ray_point_intersect:IO_error", "Too many output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:ray_point_intersect:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 1; i < 4; ++i)
        if (nrhs > i)
            if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
                mexErrMsgIdAndTxt("quadriga_lib:ray_point_intersect:IO_error", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    arma::fmat orig_single, trivec_single, tridir_single, points_single;
    arma::mat orig_double, trivec_double, tridir_double, points_double;

    if (use_single)
    {
        orig_single = qd_mex_reinterpret_Mat<float>(prhs[0]);
        trivec_single = qd_mex_reinterpret_Mat<float>(prhs[1]);
        tridir_single = qd_mex_reinterpret_Mat<float>(prhs[2]);
        points_single = qd_mex_reinterpret_Mat<float>(prhs[3]);
    }
    else
    {
        orig_double = qd_mex_reinterpret_Mat<double>(prhs[0]);
        trivec_double = qd_mex_reinterpret_Mat<double>(prhs[1]);
        tridir_double = qd_mex_reinterpret_Mat<double>(prhs[2]);
        points_double = qd_mex_reinterpret_Mat<double>(prhs[3]);
    }

    arma::uword n_points = use_single ? points_single.n_rows : points_double.n_rows;
    arma::uword max_no_hit = (nrhs < 5) ? 32 : qd_mex_get_scalar<arma::uword>(prhs[4], "max_no_hit", 32);
    arma::uword target_size = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "target_size", 0);

    arma::u32_vec sub_cloud_index;
    if (nrhs > 5 && !mxIsEmpty(prhs[5]))
    {
        if (!mxIsUint32(prhs[5]))
            mexErrMsgIdAndTxt("quadriga_lib:ray_point_intersect:IO_error", "Input 'sub_cloud_index' must be provided as 'uint32'.");

        sub_cloud_index = qd_mex_reinterpret_Col<unsigned>(prhs[5]);
    }

    // Create the sub-cloud index
    arma::fmat points_single_indexed;
    arma::mat points_double_indexed;
    arma::u32_vec reverse_index;

    if (target_size == 0)
    {
        target_size = 12 * (size_t)std::ceil(std::sqrt((double)n_points));
        target_size = (target_size < 1024) ? 0 : target_size;
    }

    if (sub_cloud_index.n_elem != 0)
        target_size = 0;

    if (target_size != 0 && use_single)
        quadriga_lib::point_cloud_segmentation(&points_single, &points_single_indexed, &sub_cloud_index, target_size, 8, nullptr, &reverse_index);
    else if (target_size != 0)
        quadriga_lib::point_cloud_segmentation(&points_double, &points_double_indexed, &sub_cloud_index, target_size, 8, nullptr, &reverse_index);

    //std::cout << "target_size = " << target_size << ", Points = " << n_points << ", Points Indexed = " << points_double_indexed.n_rows << std::endl;

    // Call library function
    std::vector<arma::Col<unsigned>> index;
    arma::u32_vec hit_count_local;
    try
    {
        if (use_single && target_size != 0)
            index = quadriga_lib::ray_point_intersect(&points_single_indexed, &orig_single, &trivec_single, &tridir_single, &sub_cloud_index, &hit_count_local);
        else if (use_single && target_size == 0)
            index = quadriga_lib::ray_point_intersect(&points_single, &orig_single, &trivec_single, &tridir_single, &sub_cloud_index, &hit_count_local);
        else if (!use_single && target_size != 0)
            index = quadriga_lib::ray_point_intersect(&points_double_indexed, &orig_double, &trivec_double, &tridir_double, &sub_cloud_index, &hit_count_local);
        else if (!use_single && target_size == 0)
            index = quadriga_lib::ray_point_intersect(&points_double, &orig_double, &trivec_double, &tridir_double, &sub_cloud_index, &hit_count_local);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:ray_point_intersect:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:ray_point_intersect:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Generate hit counter
    arma::u32_vec hit_count;
    if (nlhs > 0)
    {
        plhs[0] = qd_mex_init_output(&hit_count, n_points);
        if (target_size == 0) // Copy data
            std::memcpy(hit_count.memptr(), hit_count_local.memptr(), n_points * sizeof(unsigned));
        else // Map to original order
        {
            unsigned *p_hit_count = hit_count.memptr();
            unsigned *p_hit_count_local = hit_count_local.memptr();
            unsigned *i = reverse_index.memptr();
            for (arma::uword i_point = 0; i_point < n_points; ++i_point)
                p_hit_count[i_point] = p_hit_count_local[i[i_point]];
        }
    }

    // Generate ray index
    arma::u32_mat ray_ind;
    if (nlhs > 1)
    {
        plhs[1] = qd_mex_init_output(&ray_ind, n_points, max_no_hit);
        unsigned *p_ray_ind = ray_ind.memptr();

        if (target_size == 0) // Copy data
        {
            for (arma::uword i_point = 0; i_point < n_points; ++i_point)
                for (arma::uword i_ray = 0; i_ray < hit_count_local.at(i_point); ++i_ray)
                    if (i_ray < max_no_hit)
                        p_ray_ind[i_ray * n_points + i_point] = index[i_point].at(i_ray);
        }
        else // Map to original order
        {
            unsigned *i = reverse_index.memptr();
            for (arma::uword i_point = 0; i_point < n_points; ++i_point)
            {
                arma::uword i_point_orig = i[i_point];
                for (arma::uword i_ray = 0; i_ray < hit_count_local.at(i_point_orig); ++i_ray)
                    if (i_ray < max_no_hit)
                        p_ray_ind[i_ray * n_points + i_point] = index[i_point_orig].at(i_ray);
            }
        }
    }
}
