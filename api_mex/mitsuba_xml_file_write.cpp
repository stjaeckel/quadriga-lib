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
# MITSUBA_XML_FILE_WRITE
Write a triangular mesh to a Mitsuba 3 XML scene file

- Converts quadriga-lib mesh data structures to Mitsuba 3 XML format, loadable by NVIDIA Sionna RT for
  differentiable radio-propagation simulations
- Supports grouping faces into named objects with per-face material assignments
- Optionally maps material names to ITU-defined presets used by Sionna RT
- Creates a subdirectory `<stem>_meshes/` next to the XML file and writes one binary PLY file per object into it;
  both the XML and the mesh folder must be distributable together; the XML refers to each PLY file by its path
  relative to the XML file (`<stem>_meshes/<object>.ply`)
- Objects whose faces reference more than one material are automatically split into sub-objects (one per material)
  and renamed `<obj_name>_<mtl_name>`; the effective object count in the output may therefore exceed the length of `obj_names`

## Usage:
```
quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, ...
    mtl_names, bsdf, map_to_itu_materials );
```

## Inputs:
- **`fn`** — Output file path including `.xml` extension
- **`vert_list`** — Vertex coordinates (x, y, z); `[n_vert, 3]`
- **`face_ind`** — Triangle definitions as 1-based vertex indices; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; `[n_mesh]`
- **`mtl_ind`** — 1-based material index per triangle; every face must reference a material (0 is rejected); `[n_mesh]`
- **`obj_names`** — Object names; cell array of strings; length must equal `max(obj_ind)`
- **`mtl_names`** — Material names; cell array of strings; length must equal `max(mtl_ind)`
- **`bsdf`** — BSDF material parameters per material; ignored by Sionna RT, used only by Mitsuba renderer; see [[obj_file_read]] for field definitions; `[numel(mtl_names), 17]`; default: `[]` (none)
- **`map_to_itu_materials`** — If `true`, maps material names to ITU presets recognised by Sionna RT; default: `false`

## See also:
- [[obj_file_read]] (source for mesh data and BSDF field layout)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs; // No outputs

    if (nrhs < 7 || nrhs > 9)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const arma::mat vert_list = qd_mex_get_Mat<double>(prhs[1], false, "vert_list");

    // Face indices: 1-based (MATLAB) -> 0-based (C++), read as a vector and reshaped to [n_mesh, 3]
    arma::umat face_ind = qd_mex_get_index_Col(prhs[2], "face_ind");
    face_ind.reshape(mxGetM(prhs[2]), mxGetN(prhs[2]));

    // Object index: 1-based (MATLAB) -> 0-based (C++)
    const arma::uvec obj_ind = qd_mex_get_index_Col(prhs[3], "obj_ind");

    // mtl_ind is 1-based on both sides: no shift intended
    const arma::uvec mtl_ind = qd_mex_get_Col<arma::uword>(prhs[4], false, "mtl_ind");

    const std::vector<std::string> obj_names = qd_mex_get_strings(prhs[5], "obj_names");
    const std::vector<std::string> mtl_names = qd_mex_get_strings(prhs[6], "mtl_names");
    const arma::mat bsdf = (nrhs < 8) ? arma::mat() : qd_mex_get_Mat<double>(prhs[7], false, "bsdf");
    const bool map_to_itu_materials = (nrhs < 9) ? false : qd_mex_get_scalar<bool>(prhs[8], "map_to_itu_materials", false);

    CALL_QD(quadriga_lib::mitsuba_xml_file_write<double>(fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names,
                                                         mtl_names, bsdf, map_to_itu_materials));
}
