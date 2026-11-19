// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

- Parses a triangulated `.obj`; quads and n-gons are rejected. Face references may be `v`, `v/vt`, `v//vn` or
  `v/vt/vn`; a negative (relative) index counts back from the last vertex defined before the face line (`-1` = that
  vertex). Index 0, or an index beyond the vertices defined before the face, raises a `RuntimeError`, for any requested outputs
- Lines may end in LF or CRLF (`.obj`, `.mtl` and material CSV)
- Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`)
- `usemtl (null)` (written by [[obj_file_write]]) marks the following faces as having no material: index 0 in
  `mtl_ind` and `csv_ind` (also with `csv_strict`); `(null)` is not listed in `mtl_names` or `bsdf`
- Unmatched names raise an error when `csv_strict = True`; otherwise they map to index 0 (no material)
- Geometry indices (`face_ind`, `obj_ind`) are 0-based; material indices (`mtl_ind`, `csv_ind`) are
  1-based, with 0 reserved for the outside / no-material state
- With an empty `fn`, geometry and `.mtl` outputs are empty and only the table (`csv_names`,
  `csv_prop`) is populated; if `fn_csv` is also empty, the built-in default table is returned
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Usage:
```
mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, csv_ind, csv_names, csv_prop = \
    quadriga_lib.RTtools.obj_file_read( fn, fn_csv, *, csv_strict )
```

## Inputs:
- **`fn`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** — Path to an EM/acoustic material CSV; must contain a `name` column. Unmatched faces map
  to index 0 (no material) unless `csv_strict` is set; empty uses the built-in default table; default: `""`
- **`csv_strict`** — If `True`, raise when a `usemtl` material is absent from the table; otherwise map to 
  index 0 (no material); default: `False`

## Outputs:
- **`mesh`** — Triangle vertex coordinates `[X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3]` per row; `(n_mesh, 9)`
- **`vert_list`** — All vertex positions in the file; `(n_vert, 3)`
- **`face_ind`** — 0-based vertex indices into `vert_list` per triangle; `(n_mesh, 3)`
- **`obj_ind`** — 0-based object index per triangle; `(n_mesh,)`
- **`obj_names`** — Object names; list of `str`; length `max(obj_ind) + 1`
- **`mtl_ind`** — 1-based visual-material index per triangle (0 = no material); `(n_mesh,)`
- **`mtl_names`** — Visual material names (raw `usemtl`); list of `str`; length `n_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `(n_mtl, 17)`
- **`csv_ind`** — 1-based EM/acoustic-material index per triangle (0 = no material); `(n_mesh,)`
- **`csv_names`** — Material names from the table; list of `str`; length `n_csv`
- **`csv_prop`** — Material properties as a `dict`; each key is one CSV column (excluding `name`)
  mapping to a 1D array of length `n_csv`

## See also:
- [[obj_file_write]] (for writing OBJ files)
- [[triangle_mesh_segmentation]] (used to calculate indexed mesh for faster processing)
MD!*/

py::tuple obj_file_read(const py::handle &fn,
                        const py::handle &fn_csv,
                        py::handle csv_strict)
{
    const std::string fn_s = qd_python_path(fn, "fn");
    const std::string fn_csv_s = qd_python_path(fn_csv, "fn_csv");

    const bool csv_strict_b = qd_python_flag(csv_strict, false, "csv_strict");

    arma::mat mesh, vert_list, bsdf;
    arma::umat face_ind;
    arma::uvec obj_ind, mtl_ind, csv_ind;
    std::vector<std::string> obj_names, mtl_names, csv_names;
    std::unordered_map<std::string, std::vector<double>> csv_prop;

    quadriga_lib::obj_file_read<double>(fn_s, &mesh, &vert_list, &face_ind, &obj_ind, &obj_names,
                                        &mtl_ind, &mtl_names, &bsdf,
                                        fn_csv_s, &csv_ind, &csv_names, &csv_prop, csv_strict_b);

    // Stable dict key order: standard EM columns first, then any extra CSV columns
    std::vector<std::string> order = {"a", "b", "c", "d", "e", "f", "g", "h", "thk",
                                      "att", "attB", "alpha", "alphaB", "fRef", "m",
                                      "resF", "resQ", "resS", "coiF", "coiQ", "tf", "tfB", "tfP", "tfF", "tfQ", "side"};

    return py::make_tuple(
        qd_python_copy2numpy(&mesh),
        qd_python_copy2numpy(&vert_list),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&face_ind),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&obj_ind),
        qd_python_copy2list(obj_names),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&mtl_ind),
        qd_python_copy2list(mtl_names),
        qd_python_copy2numpy(&bsdf),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&csv_ind),
        qd_python_copy2list(csv_names),
        qd_python_map2dict(csv_prop, order));
}

/*!MD
# obj_file_write
Write a Wavefront .obj file

- Supply geometry as either `mesh`, or as `vert_list` and `face_ind`; giving both, or neither, is an error
- With `mesh`, `vert_list_out` and `face_ind_out` are derived from it, merging vertices of the same object
  that are closer than `threshold` (no merging across objects)
- With `vert_list` and `face_ind`, the geometry is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- With `split_loose_parts`, each object is split into connected components ("separate by loose parts");
  faces sharing a vertex belong to the same part, and parts of a split object are named `name.001`, `name.002`, ...
- Without `mtl_ind`, no `usemtl` tags and no `.mtl` file are written
- A face with `mtl_ind = 0` that follows a face with a material in the same object (or part) is preceded by
  `usemtl (null)`, which [[obj_file_read]] reads back as 0; the `.mtl` gets no `(null)` entry. Face order and objects
  are unchanged; a material named `(null)` that is used by a face is rejected with a `ValueError` (unused entries of
  `mtl_names` may carry the name)
- The `.mtl` file is named after the `.obj` and lists each used material; values default to a gray material when `bsdf` is omitted
- Duplicate entries in `mtl_names` are merged into one `.mtl` entry if their `bsdf` rows are identical (or if `bsdf`
  is omitted); duplicates with differing `bsdf` rows are written as `name.001`, `name.002`, ...
- If `csv_names` is given, the EM/acoustic material table is written to a companion `.csv` (named after the
  `.obj`); columns follow a fixed canonical order then any extra `csv_prop` fields (alphabetical);
  `csv_write_defaults` also emits canonical columns absent from `csv_prop`, filled with their defaults
  (`a`, `e`, `fRef` = 1, `side` = 3.29, else 0)
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Usage:
```
vert_list_out, face_ind_out = quadriga_lib.RTtools.obj_file_write( fn, mesh, obj_ind, mtl_ind, \
    obj_names, mtl_names, *, vert_list, face_ind, bsdf, threshold, csv_ind, csv_names, csv_prop, \
    csv_write_defaults, split_loose_parts )
```

## Inputs:
- **`fn`** — Path to the output `.obj` file; must end in `.obj`; if empty, no file is written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{X1,Y1,Z1,...,X3,Y3,Z3}` per row; `(n_mesh, 9)`; mutually exclusive with `vert_list` and `face_ind`; default: None
- **`obj_ind`** — 0-based object index per face; `(n_mesh,)`; each object must form a contiguous block; default: None
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `mtl_ind`/`csv_ind` output of [[obj_file_read]]); `(n_mesh,)`; omit (None) for no materials; default: None
- **`obj_names`** — Object names; list of str; length > max(obj_ind); required if `obj_ind` is given; default: None
- **`mtl_names`** — Material names; list of str; length > max(mtl_ind); required if `mtl_ind` is given; default: None
- **`vert_list`** — Vertex positions; `(n_vert, 3)`; only valid with `face_ind`; written unchanged; default: None
- **`face_ind`** — 0-based vertex indices per face; `(n_mesh, 3)`; required with `vert_list`; default: None
- **`bsdf`** — Principled BSDF values for the `.mtl` file; `(n_mtl, 17)`; see [[obj_file_read]] for the column layout; default: None
- **`threshold`** — Vertex co-location distance for merging within an object; default: 0.001 (1 mm)
- **`csv_ind`** — 1-based EM/acoustic-material index per face (0 = no material); `(n_mesh,)`; validated if given; default: None
- **`csv_names`** — EM/acoustic material names (the full table); list of str; required to write the `.csv`; default: None
- **`csv_prop`** — Material properties as a `dict`; each key is one CSV column mapping to a 1D array of length `len(csv_names)`; default: None
- **`csv_write_defaults`** — If True, also write canonical columns absent from `csv_prop` using their defaults; default: False
- **`split_loose_parts`** — If True, split each object into connected components; default: False

## Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `(n_vert, 3)`
- **`face_ind_out`** — 0-based face indices derived from `mesh`, or a copy of `face_ind`; `(n_mesh, 3)`

## See also:
- [[obj_file_read]] (for reading OBJ files and the BSDF column layout)
- [[mitsuba_xml_file_write]] (for exporting to Mitsuba scene file format)
MD!*/

py::tuple obj_file_write(const py::handle &fn,
                         py::handle mesh,
                         py::handle obj_ind,
                         py::handle mtl_ind,
                         py::handle obj_names,
                         py::handle mtl_names,
                         py::handle vert_list,
                         py::handle face_ind,
                         py::handle bsdf,
                         double threshold,
                         py::handle csv_ind,
                         py::handle csv_names,
                         py::handle csv_prop,
                         py::handle csv_write_defaults,
                         py::handle split_loose_parts)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    const bool csv_write_defaults_b = qd_python_flag(csv_write_defaults, false, "csv_write_defaults");
    const bool split_loose_parts_b = qd_python_flag(split_loose_parts, false, "split_loose_parts");

    // Convert inputs (empty Armadillo object when the handle is None)
    const auto mesh_a = qd_python_numpy2arma_Mat<double>(mesh, true);
    const auto obj_ind_a = qd_python_numpy2arma_Col<arma::uword>(obj_ind, true, false, "obj_ind");
    const auto mtl_ind_a = qd_python_numpy2arma_Col<arma::uword>(mtl_ind, true, false, "mtl_ind");
    const auto vert_list_a = qd_python_numpy2arma_Mat<double>(vert_list, true);
    const auto face_ind_a = qd_python_numpy2arma_Mat<arma::uword>(face_ind, true, false, "face_ind");
    const auto bsdf_a = qd_python_numpy2arma_Mat<double>(bsdf, true);
    const auto obj_names_v = qd_python_list2vector_Strings(obj_names);
    const auto mtl_names_v = qd_python_list2vector_Strings(mtl_names);
    const auto csv_ind_a = qd_python_numpy2arma_Col<arma::uword>(csv_ind, true, false, "csv_ind"); // 1-based (0 = no material), not shifted
    const auto csv_names_v = qd_python_list2vector_Strings(csv_names);

    std::unordered_map<std::string, std::vector<double>> csv_prop_m;
    if (!csv_prop.is_none())
        csv_prop_m = qd_python_dict2map<double>(csv_prop);

    arma::mat vert_list_out;
    arma::umat face_ind_out;

    quadriga_lib::obj_file_write<double>(fn_s, mesh_a, obj_ind_a, mtl_ind_a, obj_names_v, mtl_names_v,
                                         &vert_list_out, &face_ind_out, vert_list_a, face_ind_a,
                                         bsdf_a, threshold, csv_ind_a, csv_names_v, csv_prop_m,
                                         csv_write_defaults_b, split_loose_parts_b);

    // Face indices stay 0-based
    auto vert_list_out_py = qd_python_copy2numpy(&vert_list_out);
    auto face_ind_out_py = qd_python_copy2numpy<arma::uword, py::ssize_t>(&face_ind_out);

    return py::make_tuple(vert_list_out_py, face_ind_out_py);
}
