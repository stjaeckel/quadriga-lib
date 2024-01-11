// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# RAY_MESH_INTERACT
Calculates interactions (reflection, transmission, refraction) of radio waves with objects

## Description:
- Radio waves that interact with a building, or other objects in the environment, will produce losses 
  that depend on the electrical properties of the materials and material structure.
- This function calculates the interactions of radio-waves with objects in a 3D environment.
- It considers a plane wave incident upon a planar interface between two homogeneous and isotropic 
  media of differing electric properties. The media extend sufficiently far from the interface such 
  that the effect of any other interface is negligible.
- Air to media transition is assumed if front side of a face is hit and FBS != SBS
- Media to air transition is assumed if back side of a face is hit and FBS != SBS
- Media to media transition is assumed if FBS = SBS with opposing face orientations
- Order of the vertices determines side (front or back) of a mesh element
- Overlapping geometry in the triangle mesh must be avoided, since materials are transparent to radio 
  waves.
- Implementation is done according to ITU-R P.2040-1. 
- Rays that do not interact with the environment (i.e. for which `fbs_ind = 0`) are omitted from 
  the output.

## Material properties:
Each material is defined by its electrical properties. Radio waves that interact with a building will 
produce losses that depend on the electrical properties of the building materials, the material 
structure and the frequency of the radio wave. The fundamental quantities of interest are the electrical 
permittivity (ϵ) and the conductivity (σ). The five parameters returned in `mtl_prop` then are:

- Real part of relative permittivity at f = 1 GHz (a)
- Frequency dependence of rel. permittivity (b) such that ϵ = a · f^b
- Conductivity at f = 1 GHz (c)
- Frequency dependence of conductivity (d) such that σ = c· f^d
- Fixed attenuation in dB applied to each transition

## Usage:

```
[ origN, destN, gainN, xprmatN, trivecN, tridirN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, normal_vecN] = ...
    quadriga_lib.ray_mesh_interact(  interaction_type, center_freq, orig, dest, fbs, sbs, mesh, mtl_prop, 
    fbs_ind, sbs_ind, trivec, tridir, orig_length );
```

## Input Arguments:
- **`interaction_type`**<br>
  Interaction type: (0) Reflection, (1) Transmission, (2) Refraction

- **`center_freq`**<br>
  Center frequency in [Hz]; Scalar value

- **`orig`**<br>
  Ray origins in 3D Cartesian coordinates; Size: `[ no_ray, 3 ]`

- **`dest`**<br>
  Ray destinations in 3D Cartesian coordinates; Size: `[ no_ray, 3 ]`

- **`fbs`**<br>
  First interaction point between the rays and the triangular mesh. Size: `[ no_ray, 3 ]`

- **`sbs`**<br>
  Second interaction point between the rays and the triangular mesh. Size: `[ no_ray, 3 ]`

- **`mesh`**<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3 points
  in 3D-space. Hence, a face has 9 values in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ];
  Size: `[ no_mesh, 9 ]`

- **`mtl_prop`**<br>
  Material properties of each mesh element (see above); Size: `[ no_mesh, 5 ]`

- **`fbs_ind`**<br>
  Index of the triangle that was hit by the ray at the FBS location; 1-based; Length: `[ no_ray ]`

- **`sbs_ind`**<br>
  Index of the triangle that was hit by the ray at the SBS location; 1-based; Length: `[ no_ray ]`

- **`trivec`** (optional)<br>
  The 3 vectors pointing from the center point of the ray at the `origin` to the vertices of a triangular 
  propagation tube, the values are in the order `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`;
  Size: `[ no_ray, 9 ]`

- **`tridir`** (optional)<br>
  The directions of the vertex-rays in geographic coordinates (azimuth and elevation angle in rad); 
  the values are in the order `[ v1az, v1el, v2az, v2el, v3az, v3el ]`; <br>Size: `[ no_ray, 6 ]`

- **`orig_length`** (optional)<br>
  Path length at origin point, default is 0, Size: [ n_ray ]

## Output Arguments:
- **`origN`**<br>
  New ray origins after the interaction with the medium, usually placed close to the FBS location. 
  A small offset of 0.001 m in the direction of travel after the interaction with the medium is added 
  to avoid getting stuct inside a mesh element. Size: `[ no_rayN, 3 ]`

- **`destN`**<br>
  New ray destinaion after the interaction with the medium, taking the change of direction into account;
  Size: `[ no_rayN, 3 ]`

- **`gainN`**<br>
  Gain (negative loss) caused by the interaction with the medium, averaged over both polarization 
  directions. This value includes the in-medium attenuation, but does not account for FSPL. Linear scale.
  Size: `[ no_rayN ]`

- **`xprmatN`**<br>
  Polarization transfer matrix; 
  Interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH);
  The values account for the following effects: (1) gain caused by the interaction with the medium, 
  (2) different reflection/transmission coefficients for transverse electric (TE) and transverse 
  magnetic (TM) polarisation, (3) orientation of the incidence plane, (4) in-medium attenuation. 
  FSPL is excluded.  Size: `[ no_rayN, 8 ]`

- **`trivecN`**<br>
  The 3 vectors pointing from the center point of the ray at `origN` to the vertices of a triangular 
  propagation tube.  Size: `[ no_rayN, 9 ]`

- **`tridirN`**<br>
  The directions of the vertex-rays after interaction with the medium in geographic coordinates 
  (azimuth and elevation angle in rad); Size: `[ no_rayN, 6 ]`

- **`orig_lengthN`**<br>
  Length of the ray from `orig` to `origN`. If `orig_length` is given as input, its value is added.
  Size: `[ no_rayN ]`

- **`fbs_angleN`**<br>
  Angle between incoming ray and FBS in [rad], Size `[ n_rayN ]`

- **`thicknessN`**<br>
  Material thickness in meters calculated from the difference between FBS and SBS, Size `[ n_rayN ]`

- **`edge_lengthN`**<br>
  Max. edge length of the ray tube triangle at the new origin. A value of infinity indicates that only
  a part of the ray tube hits the object. Size `[ n_rayN, 3 ]`

- **`normal_vecN`**<br>
  Normal vector of FBS and SBS, Size `[ n_rayN, 6 ]`
MD!*/


#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.cpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - interaction_type    Interaction type: (0) Reflection, (1) Transmission, (2) Refraction
    //  1 - center_freq    Center frequency in [Hz]
    //  2 - orig                Ray origin points in GCS, Size [ n_ray, 3 ]
    //  3 - dest                Ray destination points in GCS, Size [ n_ray, 3 ]
    //  4 - fbs                 First interaction points in GCS, Size [ n_ray, 3 ]
    //  5 - sbs                 Second interaction points in GCS, Size [ n_ray, 3 ]
    //  6 - mesh                Faces of the triangular mesh, Size: [ n_mesh, 9 ]
    //  7 - mtl_prop            Material properties, Size: [ n_mesh, 5 ]
    //  8 - fbs_ind             Index of first hit mesh element, 1-based, 0 = no hit, Size [ n_ray ]
    //  9 - sbs_ind             Index of second hit mesh element, 1-based, 0 = no hit, Size [ n_ray ]
    // 10 - trivec              (optional) Vectors pointing from the origin to the vertices of a triangular propagation tube, Size [n_ray, 9]
    // 11 - tridir              (optional) Directions of the vertex-rays in rad; Size [n_ray, 6]
    // 12 - orig_length         (optional) Path length at origin point, Size [ n_ray ]

    // Outputs:
    //  0 - origN               New ray origin points in GCS, Size [ n_rayN, 3 ]
    //  1 - destN               New ray destination points in GCS, Size [ n_rayN, 3 ]
    //  2 - gainN               Average interaction gain, Size [ n_rayN ]
    //  3 - xprmatN             Polarization transfer matrix, Size [n_rayN, 8]
    //  4 - trivecN             Vectors pointing from the new origin to the vertices of the triangular propagation tube, Size [ n_rayN, 9 ]
    //  5 - tridirN             The new directions of the vertex-rays, Size [ n_rayN, 6 ]
    //  6 - orig_lengthN        Path length at the new origin point, Size [ n_rayN ]
    //  7 - fbs_angleN          Angle between incoming ray and FBS in [rad], Size [ n_rayN ]
    //  8 - thicknessN          Material thickness in meters calculated from the difference between FBS and SBS, Size [ n_rayN ]
    //  9 - edge_lengthN        Max. edge length of the ray tube triangle at the new origin, Size [ n_rayN, 3 ]
    // 10 - normal_vecN         Normal vector of FBS and SBS, Size [ n_rayN, 6 ]

    if (nrhs < 10)
        mexErrMsgIdAndTxt("quadriga_lib:ray_mesh_interact:IO_error", "Need at least 10 input arguments.");

    if (nrhs > 13)
        mexErrMsgIdAndTxt("quadriga_lib:ray_mesh_interact:IO_error", "Can have at most 13 input arguments.");

    if (nlhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:ray_mesh_interact:IO_error", "Too many output arguments.");

    // Read scalar inputs
    int interaction_type = qd_mex_get_scalar<int>(prhs[0], "interaction_type", 0);
    double center_frequency = qd_mex_get_scalar<double>(prhs[1], "center_frequency", 1.0e9);

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[2]) || mxIsDouble(prhs[2]))
        use_single = mxIsSingle(prhs[2]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:ray_mesh_interact:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 3; i < 13; ++i)
        if (nrhs > i && i != 8 && i != 9)
            if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
                mexErrMsgIdAndTxt("quadriga_lib:ray_mesh_interact:IO_error", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    arma::fmat orig_single, dest_single, fbs_single, sbs_single, mesh_single, mtl_prop_single, trivec_single, tridir_single;
    arma::fvec orig_length_single;
    arma::mat orig_double, dest_double, fbs_double, sbs_double, mesh_double, mtl_prop_double, trivec_double, tridir_double;
    arma::vec orig_length_double;

    if (use_single)
    {
        orig_single = qd_mex_reinterpret_Mat<float>(prhs[2]);
        dest_single = qd_mex_reinterpret_Mat<float>(prhs[3]);
        fbs_single = qd_mex_reinterpret_Mat<float>(prhs[4]);
        sbs_single = qd_mex_reinterpret_Mat<float>(prhs[5]);
        mesh_single = qd_mex_reinterpret_Mat<float>(prhs[6]);
        mtl_prop_single = qd_mex_reinterpret_Mat<float>(prhs[7]);
    }
    else
    {
        orig_double = qd_mex_reinterpret_Mat<double>(prhs[2]);
        dest_double = qd_mex_reinterpret_Mat<double>(prhs[3]);
        fbs_double = qd_mex_reinterpret_Mat<double>(prhs[4]);
        sbs_double = qd_mex_reinterpret_Mat<double>(prhs[5]);
        mesh_double = qd_mex_reinterpret_Mat<double>(prhs[6]);
        mtl_prop_double = qd_mex_reinterpret_Mat<double>(prhs[7]);
    }

    arma::u32_vec fbs_ind = qd_mex_typecast_Col<unsigned>(prhs[8], "fbs_ind");
    arma::u32_vec sbs_ind = qd_mex_typecast_Col<unsigned>(prhs[9], "sbs_ind");

    if (nrhs > 10 && use_single)
        trivec_single = qd_mex_reinterpret_Mat<float>(prhs[10]);
    else if (nrhs > 10)
        trivec_double = qd_mex_reinterpret_Mat<double>(prhs[10]);

    if (nrhs > 11 && use_single)
        tridir_single = qd_mex_reinterpret_Mat<float>(prhs[11]);
    else if (nrhs > 11)
        tridir_double = qd_mex_reinterpret_Mat<double>(prhs[11]);

    bool use_ray_tube = (!trivec_single.is_empty() & !tridir_single.is_empty()) |
                        (!trivec_double.is_empty() & !tridir_double.is_empty());

    if (nrhs > 12 && use_single)
        orig_length_single = qd_mex_reinterpret_Col<float>(prhs[12]);
    else if (nrhs > 12)
        orig_length_double = qd_mex_reinterpret_Col<double>(prhs[12]);

    // Get number of output rays
    auto N_rayN = 0ULL;
    for (auto p = fbs_ind.begin(); p < fbs_ind.end(); ++p)
        N_rayN += (*p == 0) ? 0ULL : 1ULL;

    // Initialize output containers
    arma::fmat origN_single, destN_single, xprmatN_single, trivecN_single, tridirN_single, normal_vecN_single;
    arma::fvec gainN_single, orig_lengthN_single, fbs_angleN_single, thicknessN_single, edge_lengthN_single;

    arma::mat origN_double, destN_double, xprmatN_double, trivecN_double, tridirN_double, normal_vecN_double;
    arma::vec gainN_double, orig_lengthN_double, fbs_angleN_double, thicknessN_double, edge_lengthN_double;

    // Get pointers
    arma::fmat *p_origN_single = &origN_single;
    arma::fmat *p_destN_single = &destN_single;
    arma::fmat *p_xprmatN_single = &xprmatN_single;
    arma::fmat *p_trivecN_single = &trivecN_single;
    arma::fmat *p_tridirN_single = &tridirN_single;
    arma::fmat *p_normal_vecN_single = &normal_vecN_single;
    arma::fvec *p_gainN_single = &gainN_single;
    arma::fvec *p_orig_lengthN_single = &orig_lengthN_single;
    arma::fvec *p_fbs_angleN_single = &fbs_angleN_single;
    arma::fvec *p_thicknessN_single = &thicknessN_single;
    arma::fvec *p_edge_lengthN_single = &edge_lengthN_single;

    arma::mat *p_origN_double = &origN_double;
    arma::mat *p_destN_double = &destN_double;
    arma::mat *p_xprmatN_double = &xprmatN_double;
    arma::mat *p_trivecN_double = &trivecN_double;
    arma::mat *p_tridirN_double = &tridirN_double;
    arma::mat *p_normal_vecN_double = &normal_vecN_double;
    arma::vec *p_gainN_double = &gainN_double;
    arma::vec *p_orig_lengthN_double = &orig_lengthN_double;
    arma::vec *p_fbs_angleN_double = &fbs_angleN_double;
    arma::vec *p_thicknessN_double = &thicknessN_double;
    arma::vec *p_edge_lengthN_double = &edge_lengthN_double;

    // Allocate memory
    if (nlhs > 0 && use_single)
        plhs[0] = qd_mex_init_output(p_origN_single, N_rayN, 3);
    else if (nlhs > 0) // double
        plhs[0] = qd_mex_init_output(p_origN_double, N_rayN, 3);
    else
        p_origN_single = nullptr, p_origN_double = nullptr;

    if (nlhs > 1 && use_single)
        plhs[1] = qd_mex_init_output(p_destN_single, N_rayN, 3);
    else if (nlhs > 1) // double
        plhs[1] = qd_mex_init_output(p_destN_double, N_rayN, 3);
    else
        p_destN_single = nullptr, p_destN_double = nullptr;

    if (nlhs > 2 && use_single)
        plhs[2] = qd_mex_init_output(p_gainN_single, N_rayN);
    else if (nlhs > 2) // double
        plhs[2] = qd_mex_init_output(p_gainN_double, N_rayN);
    else
        p_gainN_single = nullptr, p_gainN_double = nullptr;

    if (nlhs > 3 && use_single)
        plhs[3] = qd_mex_init_output(p_xprmatN_single, N_rayN, 8);
    else if (nlhs > 3) // double
        plhs[3] = qd_mex_init_output(p_xprmatN_double, N_rayN, 8);
    else
        p_xprmatN_single = nullptr, p_xprmatN_double = nullptr;

    if (nlhs > 4 && use_single)
    {
        if (use_ray_tube)
            plhs[4] = qd_mex_init_output(p_trivecN_single, N_rayN, 9);
        else
            plhs[4] = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
    }
    else if (nlhs > 4) // double
    {
        if (use_ray_tube)
            plhs[4] = qd_mex_init_output(p_trivecN_double, N_rayN, 9);
        else
            plhs[4] = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
    }
    else
        p_trivecN_single = nullptr, p_trivecN_double = nullptr;

    if (nlhs > 5 && use_single)
    {
        if (use_ray_tube)
            plhs[5] = qd_mex_init_output(p_tridirN_single, N_rayN, 6);
        else
            plhs[5] = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
    }
    else if (nlhs > 5) // double
    {
        if (use_ray_tube)
            plhs[5] = qd_mex_init_output(p_tridirN_double, N_rayN, 6);
        else
            plhs[5] = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
    }
    else
        p_tridirN_single = nullptr, p_tridirN_double = nullptr;

    if (nlhs > 6 && use_single)
        plhs[6] = qd_mex_init_output(p_orig_lengthN_single, N_rayN);
    else if (nlhs > 6) // double
        plhs[6] = qd_mex_init_output(p_orig_lengthN_double, N_rayN);
    else
        p_orig_lengthN_single = nullptr, p_orig_lengthN_double = nullptr;

    if (nlhs > 7 && use_single)
        plhs[7] = qd_mex_init_output(p_fbs_angleN_single, N_rayN);
    else if (nlhs > 7) // double
        plhs[7] = qd_mex_init_output(p_fbs_angleN_double, N_rayN);
    else
        p_fbs_angleN_single = nullptr, p_fbs_angleN_double = nullptr;

    if (nlhs > 8 && use_single)
        plhs[8] = qd_mex_init_output(p_thicknessN_single, N_rayN);
    else if (nlhs > 8) // double
        plhs[8] = qd_mex_init_output(p_thicknessN_double, N_rayN);
    else
        p_thicknessN_single = nullptr, p_thicknessN_double = nullptr;

    if (nlhs > 9 && use_single)
        plhs[9] = qd_mex_init_output(p_edge_lengthN_single, N_rayN);
    else if (nlhs > 9) // double
        plhs[9] = qd_mex_init_output(p_edge_lengthN_double, N_rayN);
    else
        p_edge_lengthN_single = nullptr, p_edge_lengthN_double = nullptr;

    if (nlhs > 10 && use_single)
        plhs[10] = qd_mex_init_output(p_normal_vecN_single, N_rayN, 6);
    else if (nlhs > 10) // double
        plhs[10] = qd_mex_init_output(p_normal_vecN_double, N_rayN, 6);
    else
        p_normal_vecN_single = nullptr, p_normal_vecN_double = nullptr;

    // Call library function
    try
    {
        if (use_single)
        {
            quadriga_lib::ray_mesh_interact(interaction_type, (float)center_frequency, &orig_single, &dest_single, &fbs_single, &sbs_single,
                                            &mesh_single, &mtl_prop_single, &fbs_ind, &sbs_ind, &trivec_single, &tridir_single, &orig_length_single,
                                            p_origN_single, p_destN_single, p_gainN_single, p_xprmatN_single, p_trivecN_single,
                                            p_tridirN_single, p_orig_lengthN_single, p_fbs_angleN_single, p_thicknessN_single,
                                            p_edge_lengthN_single, p_normal_vecN_single);
        }
        else // double
        {
            quadriga_lib::ray_mesh_interact(interaction_type, center_frequency, &orig_double, &dest_double, &fbs_double, &sbs_double,
                                            &mesh_double, &mtl_prop_double, &fbs_ind, &sbs_ind, &trivec_double, &tridir_double, &orig_length_double,
                                            p_origN_double, p_destN_double, p_gainN_double, p_xprmatN_double, p_trivecN_double,
                                            p_tridirN_double, p_orig_lengthN_double, p_fbs_angleN_double, p_thicknessN_double,
                                            p_edge_lengthN_double, p_normal_vecN_double);
        }
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:ray_mesh_interact:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:ray_mesh_interact:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}
