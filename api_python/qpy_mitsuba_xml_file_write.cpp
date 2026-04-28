// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# mitsuba_xml_file_write
Write a triangular mesh to a Mitsuba 3 XML scene file

- Converts quadriga-lib mesh data structures to Mitsuba 3 XML format, loadable by NVIDIA Sionna RT for
  differentiable radio-propagation simulations
- Supports grouping faces into named objects with per-face material assignments
- Optionally maps material names to ITU-defined presets used by Sionna RT
- Creates a subdirectory `<stem>_meshes/` next to the XML file and writes one binary PLY file per object into it;
  both the XML and the mesh folder must be distributable together
- Objects whose faces reference more than one material are automatically split into sub-objects (one per material)
  and renamed `<obj_name>_<mtl_name>`; the effective object count in the output may therefore exceed the length of `obj_names`

## Usage:
```
quadriga_lib.RTtools.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_id, mtl_id, obj_names, mtl_names, bsdf, map_to_itu )
```

## Input Arguments:
- **`fn`** — Output file path including `.xml` extension
- **`vert_list`** — Vertex coordinates (x, y, z); `(n_vert, 3)`
- **`face_ind`** — Triangle definitions as 0-based vertex indices; uint64; `(n_mesh, 3)`
- **`obj_ind`** — 1-based object index per triangle; length must match `obj_names`; uint64; `(n_mesh,)`
- **`mtl_ind`** — 1-based material index per triangle; length must match `mtl_names`; uint64; `(n_mesh,)`
- **`obj_names`** — Object names; list of strings; length must equal `max(obj_ind)`
- **`mtl_names`** — Material names; list of strings; length must equal `max(mtl_ind)`
- **`bsdf`** *(optional)* — BSDF material parameters per material; ignored by Sionna RT, used only by Mitsuba renderer; see [[obj_file_read]] for field definitions; `(mtl_names.size(), 17)`
- **`map_to_itu_materials`** *(optional)* — If `true`, maps material names to ITU presets recognised by Sionna RT

## See also:
- [[obj_file_read]] (source for mesh data and BSDF field layout)
MD!*/

void mitsuba_xml_file_write(const std::string &fn,
                            py::array_t<double> vert_list,
                            py::array_t<arma::uword> face_ind,
                            py::array_t<arma::uword> obj_ind,
                            py::array_t<arma::uword> mtl_ind,
                            py::list obj_names,
                            py::list mtl_names,
                            py::array_t<double> bsdf,
                            bool map_to_itu)
{
    const arma::mat vert_list_a = qd_python_numpy2arma_Mat(vert_list, true);
    const arma::umat face_ind_a = qd_python_numpy2arma_Mat(face_ind, true);
    const arma::uvec obj_ind_a = qd_python_numpy2arma_Col(obj_ind, true);
    const arma::uvec mtl_ind_a = qd_python_numpy2arma_Col(mtl_ind, true);
    const std::vector<std::string> obj_names_a = qd_python_list2vector_Strings(obj_names);
    const std::vector<std::string> mtl_names_a = qd_python_list2vector_Strings(mtl_names);
    const arma::mat bsdf_a = qd_python_numpy2arma_Mat(bsdf, true);

    quadriga_lib::mitsuba_xml_file_write(fn, vert_list_a, face_ind_a, obj_ind_a, mtl_ind_a, obj_names_a, mtl_names_a, bsdf_a, map_to_itu);
}
