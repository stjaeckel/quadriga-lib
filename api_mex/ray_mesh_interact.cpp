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
- With `compact = true` (default), rays with `fbs_ind = 0` (no interaction) are dropped, so
  `n_rayN ≤ n_ray`; with `compact = false` all rays are kept and no-hit rays pass through unchanged
- Output direction encoding (spherical/Cartesian) matches input `tridir` format
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves)
- Types 3–5 (scalar) use a single TE-only coefficient, suitable for acoustic simulation with
  impedance-mapped material parameters (ε derived from Z); total internal reflection is handled as
  in the EM path
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Usage:
```
[ origN, destN, fbsN, sbsN, gainN, xprmatN, trivecN, tridirN, fbs_angleN, thicknessN, ...
    edge_lengthN, normal_vecN, out_typeN, path_dirN, ray_indN ] = ...
    quadriga_lib.ray_mesh_interact( interaction_type, center_frequency, orig, dest, mesh, ...
    mtl_ind, mtl_prop, fbs_ind, sbs_ind, trivec, tridir, compact );
```

## Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction,
  3 = scalar reflection, 4 = scalar transmission, 5 = scalar refraction
- **`center_frequency`** — Center frequency
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see [[obj_file_read]]; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `csv_ind` output of
  [[obj_file_read]]); `[n_mesh]`
- **`mtl_prop`** — Material properties as a struct; each field is one column (the `csv_prop` output
  of [[obj_file_read]]); each field holds a vector of length `n_mtl`
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); uint32; `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; order
  `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`; `[n_ray, 9]`; default: `[]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical
  `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian; default: `[]`
- **`compact`** *(optional)* — If true, rays with no interaction (`fbs_ind = 0`) are dropped so
  `n_rayN <= n_ray`; if false, all rays are kept (`n_rayN = n_ray`) and no-hit rays are returned as a
  transparent pass-through; logical; default: true

## Outputs:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`fbsN`**, **`sbsN`** — First/second interaction points in GCS; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear scale); averaged over TE/TM polarizations for types 0–2,
  single TE coefficient for types 3–5; `[n_rayN]`
- **`xprmatN`** — For types 0–2: polarization transfer matrix, interleaved complex, column-major
  `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`; includes interaction gain, TE/TM coefficients,
  incidence plane orientation; excludes in-medium attenuation and FSPL; `[8, n_rayN]`. For types
  3–5 (scalar): `[Re Im]` where Re+jIm is the scalar pressure coefficient; `[2, n_rayN]`
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if
  `trivec`/`tridir` not provided
- **`fbs_angleN`** — Incidence angle at FBS; `[n_rayN]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (Inf if partial hit);
  requires the `trivec` and `tridir` inputs, returns all zeros without them; `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded (uint32); `[n_rayN]`<br><br>
   |  Bit | Meaning                                                                 |
   | :--: | ----------------------------------------------------------------------- |
   |   0  | OK flag (0 = no valid interaction / undefined)                          |
   |   1  | Front-side flag (1 = front: o→i or M2 hit first; 0 = back: i→o or M1)   |
   |   2  | Co-located FBS/SBS flag (1 = single point, required for media-to-media) |
   |   3  | Same-direction flag (FBS and SBS normals point the same way)            |
   |   4  | Corner-hit flag (FBS/SBS faces not parallel)                            |
   |   5  | Total-reflection flag (also set when a transmission factor forced it)   |
   Reachable composite values (add 32 for the total-reflection variant):<br><br>
   | Code  |  TIR  | Description                                         |
   | :---: | :---: | --------------------------------------------------- |
   |   0   |   —   | No hit                                              |
   |   1   |  33   | Single hit, inside→outside (exit)                   |
   |   3   |  35   | Single hit, outside→inside (entry)                  |
   |   5   |  37   | Media-to-media, M1 (current, back) hit first        |
   |   7   |  39   | Media-to-media, M2 (next, front) hit first          |
   |  13   |  45   | Overlapping faces, inside-inside→outside            |
   |  15   |  47   | Overlapping faces, outside→inside-inside            |
   |  21   |  53   | Corner hit, inside→outside→inside                   |
   |  23   |  55   | Corner hit, outside→inside→outside                  |
   |  29   |  61   | Corner hit, inside-inside→outside                   |
   |  31   |  63   | Corner hit, outside→inside-inside                   |
- **`path_dirN`** — Refraction-correct continuation direction: mirror for types 0/3, Snell direction
  for types 1/2/4/5; for undeviated transmission (types 1/4) this is the refracted direction, which
  differs from the geometric continuation used for `origN`/`destN`; `[n_rayN, 3]`
- **`ray_indN`** — 1-based input ray index for each output ray (inverse of the compaction map,
  order-preserving); equals `1:n_ray` when `compact = false`; uint32; `[n_rayN]`

## See also:
- [[obj_file_read]] (for loading `mesh` and `mtl_prop` from OBJ file)
- [[icosphere]] (for generating beams)
- [[ray_triangle_intersect]] (for computing FBS and SBS positions)
- [[ray_point_intersect]] (for calculating beam interactions with sampling points)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 9 || nrhs > 12)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const auto interaction_type = qd_mex_get_scalar<int>(prhs[0], "interaction_type", 0);
    const auto center_frequency = qd_mex_get_scalar<double>(prhs[1], "center_frequency", 0.0);
    const auto orig = qd_mex_get_Mat<double>(prhs[2]);
    const auto dest = qd_mex_get_Mat<double>(prhs[3]);
    const auto mesh = qd_mex_get_Mat<double>(prhs[4]);
    const arma::uvec mtl_ind = qd_mex_get_Col<arma::uword>(prhs[5]);
    const auto mtl_prop = qd_mex_struct2map<double>(prhs[6]);
    const auto fbs_ind = qd_mex_get_Col<unsigned>(prhs[7]);
    const auto sbs_ind = qd_mex_get_Col<unsigned>(prhs[8]);
    const auto trivec = (nrhs < 10) ? arma::mat() : qd_mex_get_Mat<double>(prhs[9]);
    const auto tridir = (nrhs < 11) ? arma::mat() : qd_mex_get_Mat<double>(prhs[10]);
    const auto compact = (nrhs < 12) ? true : qd_mex_get_scalar<bool>(prhs[11], "compact", true);

    // Number of output rays: all rays when compact == false, else only rays that hit (fbs_ind != 0)
    arma::uword n_rayN = orig.n_rows;
    if (compact)
    {
        n_rayN = 0;
        const unsigned *p_fbs = fbs_ind.memptr();
        for (arma::uword i_ray = 0; i_ray < fbs_ind.n_elem; ++i_ray)
            n_rayN += p_fbs[i_ray] ? 1 : 0;
    }

    // Wrap optional input pointers
    const arma::mat *p_trivec = trivec.empty() ? nullptr : &trivec;
    const arma::mat *p_tridir = tridir.empty() ? nullptr : &tridir;
    const arma::uvec *p_mtl_ind = mtl_ind.is_empty() ? nullptr : &mtl_ind;
    const auto *p_mtl_prop = mtl_prop.empty() ? nullptr : &mtl_prop;

    // Output containers with known size (aliased to MATLAB memory via qd_mex_init_output)
    arma::mat origN, destN, fbsN, sbsN, xprmatN, trivecN, tridirN, normal_vecN, path_dirN;
    arma::vec gainN, fbs_angleN, thicknessN, edge_lengthN;
    std::vector<uint8_t> out_typeN;
    arma::u32_vec ray_indN;

    // xprmat: 8 rows for EM (types 0-2), 2 rows for scalar (types 3-5); one column per output ray
    const arma::uword n_xpr = (interaction_type <= 2) ? 8 : 2;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&origN, n_rayN, 3);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&destN, n_rayN, 3);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&fbsN, n_rayN, 3);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&sbsN, n_rayN, 3);
    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&gainN, n_rayN);
    if (nlhs > 5)
        plhs[5] = qd_mex_init_output(&xprmatN, n_xpr, n_rayN);
    if (nlhs > 6)
        plhs[6] = p_trivec ? qd_mex_init_output(&trivecN, n_rayN, 9) : mxCreateDoubleMatrix(0, 0, mxREAL);
    if (nlhs > 7)
        plhs[7] = p_tridir ? qd_mex_init_output(&tridirN, n_rayN, tridir.n_cols) : mxCreateDoubleMatrix(0, 0, mxREAL);
    if (nlhs > 8)
        plhs[8] = qd_mex_init_output(&fbs_angleN, n_rayN);
    if (nlhs > 9)
        plhs[9] = qd_mex_init_output(&thicknessN, n_rayN);
    if (nlhs > 10)
        plhs[10] = qd_mex_init_output(&edge_lengthN, n_rayN);
    if (nlhs > 11)
        plhs[11] = qd_mex_init_output(&normal_vecN, n_rayN, 6);
    if (nlhs > 13)
        plhs[13] = qd_mex_init_output(&path_dirN, n_rayN, 3);

    // Wrap optional output pointers based on requested outputs
    arma::mat *p_origN = (nlhs > 0) ? &origN : nullptr;
    arma::mat *p_destN = (nlhs > 1) ? &destN : nullptr;
    arma::mat *p_fbsN = (nlhs > 2) ? &fbsN : nullptr;
    arma::mat *p_sbsN = (nlhs > 3) ? &sbsN : nullptr;
    arma::vec *p_gainN = (nlhs > 4) ? &gainN : nullptr;
    arma::mat *p_xprmatN = (nlhs > 5) ? &xprmatN : nullptr;
    arma::mat *p_trivecN = (nlhs > 6) ? &trivecN : nullptr;
    arma::mat *p_tridirN = (nlhs > 7) ? &tridirN : nullptr;
    arma::vec *p_fbs_angleN = (nlhs > 8) ? &fbs_angleN : nullptr;
    arma::vec *p_thicknessN = (nlhs > 9) ? &thicknessN : nullptr;
    arma::vec *p_edge_lengthN = (nlhs > 10) ? &edge_lengthN : nullptr;
    arma::mat *p_normal_vecN = (nlhs > 11) ? &normal_vecN : nullptr;
    std::vector<uint8_t> *p_out_typeN = (nlhs > 12) ? &out_typeN : nullptr;
    arma::mat *p_path_dirN = (nlhs > 13) ? &path_dirN : nullptr;
    arma::u32_vec *p_ray_indN = (nlhs > 14) ? &ray_indN : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::ray_mesh_interact<double>(
        interaction_type, center_frequency,
        &orig, &dest, &mesh, p_mtl_ind, p_mtl_prop, &fbs_ind, &sbs_ind,
        p_trivec, p_tridir,
        p_origN, p_destN, p_fbsN, p_sbsN, p_gainN, p_xprmatN,
        p_trivecN, p_tridirN,
        p_fbs_angleN, p_thicknessN, p_edge_lengthN, p_normal_vecN, p_out_typeN,
        p_path_dirN, compact, p_ray_indN));

    // Copy post-processed outputs to MATLAB
    if (nlhs > 12) // out_typeN: bit-encoded uint8 -> uint32
    {
        arma::u32_vec out_type_u32;
        plhs[12] = qd_mex_init_output(&out_type_u32, out_typeN.size());
        for (arma::uword i = 0; i < (arma::uword)out_typeN.size(); ++i)
            out_type_u32[i] = (unsigned)out_typeN[i];
    }

    if (nlhs > 14) // ray_indN: 0-based (C++) -> 1-based (MATLAB)
    {
        ray_indN += 1;
        plhs[14] = qd_mex_copy2matlab(&ray_indN);
    }
}