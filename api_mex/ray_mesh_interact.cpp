// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# RAY_MESH_INTERACT
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`)
- Face side determined by vertex order; CCW winding = front, CW = back (right-hand rule);
  front-side hit with FBS≠SBS → air-to-media; back-side hit with FBS≠SBS → media-to-air;
  FBS=SBS with opposing normals → media-to-media
- Rays with `fbs_ind = 0` (no interaction) are omitted from output, so `n_rayN ≤ n_ray`
- Output direction encoding (spherical/Cartesian) matches input `tridir` format
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves)
- Types 3–4 (scalar) use TE-only reflection with no total internal reflection, suitable for
  acoustic simulation with impedance-mapped material parameters (ε derived from Z)

## Usage:
```
[ origN, destN, gainN, xprmatN, trivecN, tridirN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, ...
    normal_vecN, out_typeN ] = quadriga_lib.ray_mesh_interact( interaction_type, center_frequency, ...
    orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, trivec, tridir, orig_length );
```

## Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission
- **`center_frequency`** — Center frequency
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`fbs`**, **`sbs`** — First/second interaction points in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see `obj_file_read`; `[n_mesh, 9]`
- **`mtl_prop`** — Material properties; see `obj_file_read`; `[n_mesh, 9]`
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); uint32; `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin;
   order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`; `[n_ray, 9]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical
  `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`orig_length`** *(optional)* — Accumulated path length at origin; default: 0; `[n_ray]`

## Outputs:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear, includes in-medium attenuation, excludes FSPL);
  averaged over TE/TM polarizations for types 0–2, TE-only for types 3–4; `[n_rayN]`
- **`xprmatN`** — For types 0–2: polarization transfer matrix, interleaved complex
  `[ReVV ImVV ReVH ImVH ReHV ImHV ReHH ImHH]`; for types 3–4 (scalar):
  `[Re Im 0 0 0 0 0 0]` where Re+jIm is the scalar pressure coefficient; includes
  interaction gain, TE/TM coefficients, incidence plane orientation, in-medium
  attenuation (excludes FSPL); `[n_rayN, 8]`
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input);
  empty if `trivec`/`tridir` not provided
- **`orig_lengthN`** — Path length from `orig` to `origN`, added to input `orig_length` if given; `[n_rayN]`
- **`fbs_angleN`** — Incidence angle at FBS; `[n_rayN]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (Inf if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code (int32); `[n_rayN]`<br><br>
   | Code  | Description                                         |
   | ----- | --------------------------------------------------- |
   |   1   | Single hit, outside→inside                          |
   |   2   | Single hit, inside→outside                          |
   |   3   | Single hit, inside→outside, total reflection        |
   |   4   | Media-to-media, M2 hit first                        |
   |   5   | Media-to-media, M1 hit first                        |
   |   6   | Media-to-media, M1 hit first, total reflection      |
   |   7   | Overlapping faces, outside→inside                   |
   |   8   | Overlapping faces, inside→outside                   |
   |   9   | Overlapping faces, inside→outside, total reflection |
   |  10   | Edge hit, outside→inside→outside                    |
   |  11   | Edge hit, inside→outside→inside                     |
   |  12   | Edge hit, inside→outside→inside, total reflection   |
   |  13   | Edge hit, outside→inside                            |
   |  14   | Edge hit, inside→outside                            |
   |  15   | Edge hit, inside→outside, total reflection          |
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 10 || nrhs > 13)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 12)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const auto interaction_type = qd_mex_get_scalar<int>(prhs[0], "interaction_type", 0);
    const auto center_frequency = qd_mex_get_scalar<double>(prhs[1], "center_frequency", 0.0);
    const auto orig = qd_mex_get_Mat<double>(prhs[2]);
    const auto dest = qd_mex_get_Mat<double>(prhs[3]);
    const auto fbs = qd_mex_get_Mat<double>(prhs[4]);
    const auto sbs = qd_mex_get_Mat<double>(prhs[5]);
    const auto mesh = qd_mex_get_Mat<double>(prhs[6]);
    const auto mtl_prop = qd_mex_get_Mat<double>(prhs[7]);
    const auto fbs_ind = qd_mex_get_Col<unsigned>(prhs[8]);
    const auto sbs_ind = qd_mex_get_Col<unsigned>(prhs[9]);
    const auto trivec = (nrhs < 11) ? arma::mat() : qd_mex_get_Mat<double>(prhs[10]);
    const auto tridir = (nrhs < 12) ? arma::mat() : qd_mex_get_Mat<double>(prhs[11]);
    const auto orig_length = (nrhs < 13) ? arma::vec() : qd_mex_get_Col<double>(prhs[12]);

    // Get number of output rays (fbs_ind != 0)
    arma::uword n_rayN = 0;
    const unsigned *p_fbs = fbs_ind.memptr();
    for (arma::uword i_ray = 0; i_ray < fbs_ind.n_elem; ++i_ray)
        n_rayN += p_fbs[i_ray] ? 1 : 0;

    // Wrap optional input pointers
    const arma::mat *p_trivec = trivec.empty() ? nullptr : &trivec;
    const arma::mat *p_tridir = tridir.empty() ? nullptr : &tridir;
    const arma::vec *p_orig_length = orig_length.empty() ? nullptr : &orig_length;

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN;
    arma::s32_vec out_typeN;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&origN, n_rayN, 3);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&destN, n_rayN, 3);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&gainN, n_rayN);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&xprmatN, n_rayN, 8);
    if (nlhs > 4)
        plhs[4] = p_trivec ? qd_mex_init_output(&trivecN, n_rayN, 9) : mxCreateDoubleMatrix(0, 0, mxREAL);
    if (nlhs > 5)
        plhs[5] = p_tridir ? qd_mex_init_output(&tridirN, n_rayN, tridir.n_cols) : mxCreateDoubleMatrix(0, 0, mxREAL);
    if (nlhs > 6)
        plhs[6] = qd_mex_init_output(&orig_lengthN, n_rayN);
    if (nlhs > 7)
        plhs[7] = qd_mex_init_output(&fbs_angleN, n_rayN);
    if (nlhs > 8)
        plhs[8] = qd_mex_init_output(&thicknessN, n_rayN);
    if (nlhs > 9)
        plhs[9] = qd_mex_init_output(&edge_lengthN, n_rayN);
    if (nlhs > 10)
        plhs[10] = qd_mex_init_output(&normal_vecN, n_rayN, 6);
    if (nlhs > 11)
        plhs[11] = qd_mex_init_output(&out_typeN, n_rayN);

    // Wrap optional output pointers based on requested outputs
    arma::mat *p_origN = (nlhs > 0) ? &origN : nullptr;
    arma::mat *p_destN = (nlhs > 1) ? &destN : nullptr;
    arma::vec *p_gainN = (nlhs > 2) ? &gainN : nullptr;
    arma::mat *p_xprmatN = (nlhs > 3) ? &xprmatN : nullptr;
    arma::mat *p_trivecN = (nlhs > 4) ? &trivecN : nullptr;
    arma::mat *p_tridirN = (nlhs > 5) ? &tridirN : nullptr;
    arma::vec *p_orig_lengthN = (nlhs > 6) ? &orig_lengthN : nullptr;
    arma::vec *p_fbs_angleN = (nlhs > 7) ? &fbs_angleN : nullptr;
    arma::vec *p_thicknessN = (nlhs > 8) ? &thicknessN : nullptr;
    arma::vec *p_edge_lengthN = (nlhs > 9) ? &edge_lengthN : nullptr;
    arma::mat *p_normal_vecN = (nlhs > 10) ? &normal_vecN : nullptr;
    arma::s32_vec *p_out_typeN = (nlhs > 11) ? &out_typeN : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::ray_mesh_interact<double>(
        interaction_type, center_frequency,
        &orig, &dest, &fbs, &sbs, &mesh, &mtl_prop, &fbs_ind, &sbs_ind,
        p_trivec, p_tridir, p_orig_length,
        p_origN, p_destN, p_gainN, p_xprmatN,
        p_trivecN, p_tridirN, p_orig_lengthN,
        p_fbs_angleN, p_thicknessN, p_edge_lengthN, p_normal_vecN, p_out_typeN));
}