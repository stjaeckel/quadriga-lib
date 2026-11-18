// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

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
  both the XML and the mesh folder must be distributable together; the XML refers to each PLY file by its path
  relative to the XML file (`<stem>_meshes/<object>.ply`)
- Objects whose faces reference more than one material are automatically split into sub-objects (one per material)
  and renamed `<obj_name>_<mtl_name>`; the effective object count in the output may therefore exceed the length of `obj_names`
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Usage:
```
quadriga_lib.RTtools.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, \
    mtl_names, bsdf, *, map_to_itu_materials )
```

## Input Arguments:
- **`fn`** — Output file path including `.xml` extension
- **`vert_list`** — Vertex coordinates (x, y, z); `(n_vert, 3)`
- **`face_ind`** — Triangle definitions as 0-based vertex indices; uint64; `(n_mesh, 3)`
- **`obj_ind`** — 0-based object index per triangle; length `obj_names` must equal `max(obj_ind) + 1`; uint64; `(n_mesh,)`
- **`mtl_ind`** — 1-based material index per triangle (0 = no material); length `mtl_names` must be ≥ `max(mtl_ind)`; uint64; `(n_mesh,)`
- **`obj_names`** — Object names; list of strings; length must equal `max(obj_ind) + 1`
- **`mtl_names`** — Material names; list of strings; length must be ≥ `max(mtl_ind)` (1-based, so material `w` → `mtl_names[w-1]`)
- **`bsdf`** *(optional)* — BSDF material parameters per material; ignored by Sionna RT, used only by Mitsuba renderer; see [[obj_file_read]] for field definitions; `(mtl_names.size(), 17)`; default: `None`
- **`map_to_itu_materials`** *(optional)* — If `True`, maps material names to ITU presets recognised by Sionna RT; default: `False`

## See also:
- [[obj_file_read]] (source for mesh data and BSDF field layout)
MD!*/

void mitsuba_xml_file_write(const py::handle &fn,
                            py::array_t<double> vert_list,
                            const py::handle &face_ind,
                            const py::handle &obj_ind,
                            const py::handle &mtl_ind,
                            py::list obj_names,
                            py::list mtl_names,
                            const py::handle &bsdf,
                            py::handle map_to_itu_materials)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    const bool map_to_itu_b = qd_python_flag(map_to_itu_materials, false, "map_to_itu_materials");

    const arma::mat vert_list_a = qd_python_numpy2arma_Mat(vert_list, true);
    const arma::umat face_ind_a = qd_python_numpy2arma_Mat<arma::uword>(face_ind, true, false, "face_ind");
    const arma::uvec obj_ind_a = qd_python_numpy2arma_Col<arma::uword>(obj_ind, true, false, "obj_ind");
    const arma::uvec mtl_ind_a = qd_python_numpy2arma_Col<arma::uword>(mtl_ind, true, false, "mtl_ind");
    const std::vector<std::string> obj_names_a = qd_python_list2vector_Strings(obj_names);
    const std::vector<std::string> mtl_names_a = qd_python_list2vector_Strings(mtl_names);
    const arma::mat bsdf_a = qd_python_numpy2arma_Mat<double>(bsdf, true);

    quadriga_lib::mitsuba_xml_file_write(fn_s, vert_list_a, face_ind_a, obj_ind_a, mtl_ind_a, obj_names_a, mtl_names_a, &bsdf_a, map_to_itu_b);
}
