// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# OBJ_FILE_WRITE
Write a Wavefront .obj file

- Supply geometry as either `mesh`, or as `vert_list` and `face_ind`; giving both, or neither, is  an error
- With `mesh`, `vert_list_out` and `face_ind_out` are derived from it, merging vertices of the same
  object that are closer than `threshold` (no merging across objects)
- With `vert_list` and `face_ind`, the geometry is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- Without `obj_ind` and `obj_names`, a single object named `object` is written
- Without `mtl_ind`, no `usemtl` tags and no `.mtl` file are written
- The `.mtl` file is named after the `.obj` and lists each used material; values default to a gray
  material when `bsdf` is omitted

## Usage:
```
[ vert_list_out, face_ind_out ] = quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, ...
    obj_names, mtl_names, vert_list, face_ind, bsdf, threshold );
```

## Inputs:
- **`fn`** — Path to the output `.obj` file; must end in `.obj`; if empty, no file is written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{X1,Y1,Z1,...,X3,Y3,Z3}` per row; `[n_mesh, 9]`; mutually exclusive with `vert_list` and `face_ind`
- **`obj_ind`** — 1-based object index per face; `[n_mesh]`; each object must form a contiguous block
- **`mtl_ind`** — 1-based material index per face; `[n_mesh]`; omit or pass `[]` for no materials
- **`obj_names`** — Object names; cell array of strings; length >= max(obj_ind); required if `obj_ind` is given
- **`mtl_names`** — Material names; cell array of strings; length >= max(mtl_ind); required if `mtl_ind` is given
- **`vert_list`** — Vertex positions; `[n_vert, 3]`; only valid with `face_ind`; written unchanged
- **`face_ind`** — 1-based vertex indices per face; `[n_mesh, 3]`; required with `vert_list`
- **`bsdf`** — Principled BSDF values for the `.mtl` file; `[n_mtl, 17]`; see [[obj_file_read]] for the column layout
- **`threshold`** — Vertex co-location distance for merging within an object; default: 0.001 (1 mm)

## Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `[n_vert, 3]`
- **`face_ind_out`** — 1-based face indices derived from `mesh`, or a copy of `face_ind`; `[n_mesh, 3]`

## See also:
- [[obj_file_read]] (for reading OBJ files and the BSDF column layout)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    std::string fn = qd_mex_get_string(prhs[0]);

    const arma::mat mesh = (nrhs < 2) ? arma::mat() : qd_mex_get_Mat<double>(prhs[1]);

    // Copy obj_ind / mtl_ind so we can convert MATLAB 1-based indices to C++ 0-based.
    // Empty input -> no objects / no materials (passed as nullptr below).
    arma::uvec obj_ind = (nrhs < 3) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[2], true);
    if (!obj_ind.is_empty())
        obj_ind -= 1;

    arma::uvec mtl_ind = (nrhs < 4) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[3], true);
    if (!mtl_ind.is_empty())
        mtl_ind -= 1;

    std::vector<std::string> obj_names = (nrhs < 5) ? std::vector<std::string>() : qd_mex_get_strings(prhs[4]);
    std::vector<std::string> mtl_names = (nrhs < 6) ? std::vector<std::string>() : qd_mex_get_strings(prhs[5]);
    const arma::mat vert_list = (nrhs < 7) ? arma::mat() : qd_mex_get_Mat<double>(prhs[6]);

    // Copy face_ind so we can convert MATLAB 1-based indices to C++ 0-based
    arma::umat face_ind = (nrhs < 8) ? arma::umat() : qd_mex_get_Mat<arma::uword>(prhs[7], true);
    if (!face_ind.is_empty())
        face_ind -= 1;

    const arma::mat bsdf = (nrhs < 9) ? arma::mat() : qd_mex_get_Mat<double>(prhs[8]);
    const double threshold = (nrhs < 10) ? 0.001 : qd_mex_get_scalar<double>(prhs[9], "threshold", 0.001);

    // Wrap optional inputs as nullptr when empty
    const arma::mat *p_mesh = mesh.is_empty() ? nullptr : &mesh;
    const arma::uvec *p_obj_ind = obj_ind.is_empty() ? nullptr : &obj_ind;
    const arma::uvec *p_mtl_ind = mtl_ind.is_empty() ? nullptr : &mtl_ind;
    const std::vector<std::string> *p_obj_names = obj_names.empty() ? nullptr : &obj_names;
    const std::vector<std::string> *p_mtl_names = mtl_names.empty() ? nullptr : &mtl_names;
    const arma::mat *p_vert_list = vert_list.is_empty() ? nullptr : &vert_list;
    const arma::umat *p_face_ind = face_ind.is_empty() ? nullptr : &face_ind;
    const arma::mat *p_bsdf = bsdf.is_empty() ? nullptr : &bsdf;

    // Outputs (size determined by the C++ call) - request only what the user asked for
    arma::mat vert_list_out;
    arma::umat face_ind_out;
    arma::mat *p_vert_list_out = (nlhs > 0) ? &vert_list_out : nullptr;
    arma::umat *p_face_ind_out = (nlhs > 1) ? &face_ind_out : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::obj_file_write<double>(fn, p_mesh, p_obj_ind, p_mtl_ind, p_obj_names, p_mtl_names,
                                                 p_vert_list_out, p_face_ind_out, p_vert_list, p_face_ind,
                                                 p_bsdf, threshold));

    // Copy to MATLAB (convert face indices back to 1-based)
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&vert_list_out);

    if (nlhs > 1)
    {
        face_ind_out += 1;
        plhs[1] = qd_mex_copy2matlab(&face_ind_out);
    }
}
