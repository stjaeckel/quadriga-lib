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
# OBJ_FILE_READ
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

## Description:
- Parses a triangulated `.obj`; quads and n-gons are rejected
- Parses a triangulated `.obj`; quads and n-gons are rejected. Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`)
- Unmatched names throw when `csv_strict` is true; otherwise they map to row 1 of the table (the transparent fallback)
- With an empty `fn`, geometry and `.mtl` outputs are empty and only the table (`csv_names`,
  `csv_prop`) is populated; if `fn_csv` is also empty, the built-in default table is returned
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Usage:
```
[ mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, ...
    csv_ind, csv_names, csv_prop ] = quadriga_lib.obj_file_read( fn, fn_csv, csv_strict );
```

## Inputs:
- **`fn`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** *(optional)* — Path to an EM/acoustic material CSV; must contain a `name` column, and
  row 1 is the fallback material (should be transparent, e.g. air); empty uses the built-in default table
- **`csv_strict`** *(optional)* — If true, throw when a `usemtl` material is absent from the table;
  otherwise map to row 1; default: false

## Outputs:
- **`mesh`** — Triangle vertex coordinates `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}` per row; `[n_mesh, 9]`
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 1-based vertex indices into `vert_list` per triangle; uint64; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; uint64; `[n_mesh]`
- **`obj_names`** — Object names; cell array of strings; length `max(obj_ind)`
- **`mtl_ind`** — 1-based visual-material index per triangle; uint64; `[n_mesh]`
- **`mtl_names`** — Visual material names (raw `usemtl`); cell array of strings; length `no_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `[no_mtl, 17]`
- **`csv_ind`** — 1-based EM/acoustic-material index per triangle; uint64; `[n_mesh]`
- **`csv_names`** — Material names from the table; cell array of strings; length `n_csv`
- **`csv_prop`** — Material properties as a struct; each field is one CSV column (excluding `name`)
  holding a column vector of length `n_csv`

## See also:
- [[obj_file_write]] (for writing OBJ files)
- [[triangle_mesh_segmentation]] (used to calculate indexed mesh for faster processing)
- [[ray_mesh_interact]] (calculating interactions between rays and the triangular mesh)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = (nrhs < 1) ? "" : qd_mex_get_string(prhs[0], "");
    const std::string fn_csv = (nrhs < 2) ? "" : qd_mex_get_string(prhs[1], "");
    const bool csv_strict = (nrhs < 3) ? false : qd_mex_get_scalar<bool>(prhs[2], "csv_strict", false);

    // Output containers
    arma::mat mesh, vert_list, bsdf;
    arma::umat face_ind;
    arma::uvec obj_ind, mtl_ind, csv_ind;
    std::vector<std::string> obj_names, mtl_names, csv_names;
    std::unordered_map<std::string, std::vector<double>> csv_prop;

    // Wrap optional outputs (skip work the user did not request)
    arma::mat *p_mesh = (nlhs > 0) ? &mesh : nullptr;
    arma::mat *p_vert_list = (nlhs > 1) ? &vert_list : nullptr;
    arma::umat *p_face_ind = (nlhs > 2) ? &face_ind : nullptr;
    arma::uvec *p_obj_ind = (nlhs > 3) ? &obj_ind : nullptr;
    std::vector<std::string> *p_obj_names = (nlhs > 4) ? &obj_names : nullptr;
    arma::uvec *p_mtl_ind = (nlhs > 5) ? &mtl_ind : nullptr;
    std::vector<std::string> *p_mtl_names = (nlhs > 6) ? &mtl_names : nullptr;
    arma::mat *p_bsdf = (nlhs > 7) ? &bsdf : nullptr;
    arma::uvec *p_csv_ind = (nlhs > 8) ? &csv_ind : nullptr;
    std::vector<std::string> *p_csv_names = (nlhs > 9) ? &csv_names : nullptr;
    std::unordered_map<std::string, std::vector<double>> *p_csv_prop = (nlhs > 10) ? &csv_prop : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::obj_file_read<double>(fn, p_mesh, p_vert_list, p_face_ind, p_obj_ind, p_obj_names,
                                                p_mtl_ind, p_mtl_names, p_bsdf,
                                                fn_csv, p_csv_ind, p_csv_names, p_csv_prop, csv_strict));

    // Copy to MATLAB (convert 0-based C++ indices to 1-based)
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&mesh);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&vert_list);
    if (nlhs > 2)
    {
        face_ind += 1;
        plhs[2] = qd_mex_copy2matlab(&face_ind);
    }
    if (nlhs > 3)
    {
        obj_ind += 1;
        plhs[3] = qd_mex_copy2matlab(&obj_ind);
    }
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&obj_names);
    if (nlhs > 5)
    {
        mtl_ind += 1;
        plhs[5] = qd_mex_copy2matlab(&mtl_ind);
    }
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&mtl_names);
    if (nlhs > 7)
        plhs[7] = qd_mex_copy2matlab(&bsdf);
    if (nlhs > 8)
    {
        csv_ind += 1;
        plhs[8] = qd_mex_copy2matlab(&csv_ind);
    }
    if (nlhs > 9)
        plhs[9] = qd_mex_copy2matlab(&csv_names);
    if (nlhs > 10)
    {
        // Stable field order: standard EM columns first, then any extra CSV columns
        std::vector<std::string> order = {"a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef", "m", "resF", "resQ", "resS", "coiF", "coiQ", "coiA"};
        for (const auto &kv : csv_prop)
            if (std::find(order.begin(), order.end(), kv.first) == order.end())
                order.push_back(kv.first);
        plhs[10] = qd_mex_map2struct(csv_prop, order);
    }
}