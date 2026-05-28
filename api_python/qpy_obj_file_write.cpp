// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# obj_file_write
Write a Wavefront .obj file

- Supply geometry as either `mesh`, or as `vert_list` and `face_ind`; giving both, or neither, is an error
- With `mesh`, `vert_list_out` and `face_ind_out` are derived from it, merging vertices of the same
  object that are closer than `threshold` (no merging across objects)
- With `vert_list` and `face_ind`, the geometry is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- Without `obj_ind` and `obj_names`, a single object named `object` is written
- Without `mtl_ind` (or if all entries are 0), no `usemtl` tags and no `.mtl` file are written
- The `.mtl` file is named after the `.obj` and lists each used material; values default to a gray
  material when `bsdf` is omitted

## Usage:
```
vert_list_out, face_ind_out = quadriga_lib.RTtools.obj_file_write( fn, mesh, obj_ind, mtl_ind, \
    obj_names, mtl_names, vert_list, face_ind, bsdf, threshold )
```

## Inputs:
- **`fn`** — Path to the output `.obj` file; must end in `.obj`; if empty, no file is written (outputs are still computed); default: `""`
- **`mesh`** — Triangle coordinates `{X1,Y1,Z1,...,X3,Y3,Z3}` per row; `(n_mesh, 9)`; mutually exclusive with `vert_list` and `face_ind`; default: None
- **`obj_ind`** — 1-based object index per face; `(n_mesh,)`; each object must form a contiguous block; default: None
- **`mtl_ind`** — 1-based material index per face (0 = unassigned); `(n_mesh,)`; default: None
- **`obj_names`** — Object names; list of str; length >= max(obj_ind); required if `obj_ind` is given; default: None
- **`mtl_names`** — Material names; list of str; length >= max(mtl_ind); required if `mtl_ind` has nonzero entries; default: None
- **`vert_list`** — Vertex positions; `(n_vert, 3)`; only valid with `face_ind`; written unchanged; default: None
- **`face_ind`** — 0-based vertex indices per face; `(n_mesh, 3)`; required with `vert_list`; default: None
- **`bsdf`** — Principled BSDF values for the `.mtl` file; `(n_mtl, 17)`; see [[obj_file_read]] for the column layout; default: None
- **`threshold`** — Vertex co-location distance for merging within an object; default: 0.001 (1 mm)

## Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `(n_vert, 3)`
- **`face_ind_out`** — 0-based face indices derived from `mesh`, or a copy of `face_ind`; `(n_mesh, 3)`

## See also:
- [[obj_file_read]] (for reading OBJ files and the BSDF column layout)
- [[mitsuba_xml_file_write]] (for exporting to Mitsuba scene file format)
MD!*/

py::tuple obj_file_write(const std::string &fn,
                         py::handle mesh,
                         py::handle obj_ind,
                         py::handle mtl_ind,
                         py::handle obj_names,
                         py::handle mtl_names,
                         py::handle vert_list,
                         py::handle face_ind,
                         py::handle bsdf,
                         double threshold)
{
    // Convert inputs (empty Armadillo object when the handle is None)
    const auto mesh_a = qd_python_numpy2arma_Mat<double>(mesh, true);
    const auto obj_ind_a = qd_python_numpy2arma_Col<arma::uword>(obj_ind, true);
    const auto mtl_ind_a = qd_python_numpy2arma_Col<arma::uword>(mtl_ind, true);
    const auto vert_list_a = qd_python_numpy2arma_Mat<double>(vert_list, true);
    const auto face_ind_a = qd_python_numpy2arma_Mat<arma::uword>(face_ind, true);
    const auto bsdf_a = qd_python_numpy2arma_Mat<double>(bsdf, true);
    const auto obj_names_v = qd_python_list2vector_Strings(obj_names);
    const auto mtl_names_v = qd_python_list2vector_Strings(mtl_names);

    // Wrap optional inputs as nullptr when absent (C++ keys its mode off nullptr, not empty)
    const arma::mat *p_mesh = mesh_a.is_empty() ? nullptr : &mesh_a;
    const arma::uvec *p_obj_ind = obj_ind_a.is_empty() ? nullptr : &obj_ind_a;
    const arma::uvec *p_mtl_ind = mtl_ind_a.is_empty() ? nullptr : &mtl_ind_a;
    const std::vector<std::string> *p_obj_names = obj_names_v.empty() ? nullptr : &obj_names_v;
    const std::vector<std::string> *p_mtl_names = mtl_names_v.empty() ? nullptr : &mtl_names_v;
    const arma::mat *p_vert_list = vert_list_a.is_empty() ? nullptr : &vert_list_a;
    const arma::umat *p_face_ind = face_ind_a.is_empty() ? nullptr : &face_ind_a;
    const arma::mat *p_bsdf = bsdf_a.is_empty() ? nullptr : &bsdf_a;

    // Outputs (size determined by the C++ call) - always allocated and returned
    arma::mat vert_list_out;
    arma::umat face_ind_out;

    // Call library function
    quadriga_lib::obj_file_write<double>(fn, p_mesh, p_obj_ind, p_mtl_ind, p_obj_names, p_mtl_names,
                                         &vert_list_out, &face_ind_out, p_vert_list, p_face_ind,
                                         p_bsdf, threshold);

    // Copy to python (face indices stay 0-based)
    auto vert_list_out_py = qd_python_copy2numpy(&vert_list_out);
    auto face_ind_out_py = qd_python_copy2numpy<arma::uword, py::ssize_t>(&face_ind_out);

    return py::make_tuple(vert_list_out_py, face_ind_out_py);
}

// pybind11 declaration:
// m.def("obj_file_write", &obj_file_write,
//       py::arg("fn") = "",
//       py::arg("mesh") = py::none(),
//       py::arg("obj_ind") = py::none(),
//       py::arg("mtl_ind") = py::none(),
//       py::arg("obj_names") = py::none(),
//       py::arg("mtl_names") = py::none(),
//       py::arg("vert_list") = py::none(),
//       py::arg("face_ind") = py::none(),
//       py::arg("bsdf") = py::none(),
//       py::arg("threshold") = 0.001);
