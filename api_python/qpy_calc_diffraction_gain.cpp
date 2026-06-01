// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# calc_diffraction_gain
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided
  into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients,
  generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [[triangle_mesh_segmentation]]) accelerates computation by skipping
  triangles whose bounding box does not intersect the TX-RX path
- For a detailed description of the material model see
  <a href="http://quadriga-lib.org/formats.html">Data Formats</a> section

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, \
    center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode )

# Unpacked outputs
gain, coord = quadriga_lib.RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, \
    center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode )
```

## Inputs:
- **`orig`** — TX positions; `(n_pos, 3)`
- **`dest`** — RX positions; `(n_pos, 3)`
- **`mesh`** — Triangle vertices, each row `[X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3]`; `(n_mesh, 9)`
- **`mtl_ind`** — 0-based material index per face (the `csv_ind` output of [[obj_file_read]]); `(n_mesh,)`
- **`mtl_prop`** — Material properties as a `dict`; each key is one column (the `csv_prop` output of [[obj_file_read]]) mapping to a 1D array of length `n_mtl`
- **`center_frequency`** — Center frequency
- **`lod`** — Level of detail (0–6), controls `n_path` and `n_seg`; see [[generate_diffraction_paths]]; default: 2
- **`verbose`** — Verbosity level; default: 0
- **`sub_mesh_index`** — 0-based sub-mesh index for acceleration; see [[triangle_mesh_segmentation]]; `(n_mesh,)` or `None`; default: `None`
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable; default: 0
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels; default: 0
- **`scalar_mode`** — If `True`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging; default: `False`

## Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear scale; `(n_pos,)`
- **`coord`** — Diffracted path coordinates excluding endpoints; `(3, n_seg-1, n_pos)`

## See also:
- [[generate_diffraction_paths]] (controls path/segment count via `lod`)
- [[triangle_mesh_segmentation]] (generates `sub_mesh_index`)
- [[obj_file_read]] (defines the material format)
MD!*/

py::tuple calc_diffraction_gain(const py::array_t<double> &orig,
                                const py::array_t<double> &dest,
                                const py::array_t<double> &mesh,
                                const py::handle &mtl_ind,
                                const py::handle &mtl_prop,
                                double center_freq,
                                int lod,
                                int verbose,
                                const py::handle &sub_mesh_index,
                                int use_kernel,
                                int gpu_id,
                                bool scalar_mode)
{
    const auto orig_a = qd_python_numpy2arma_Mat<double>(orig, true);
    const auto dest_a = qd_python_numpy2arma_Mat<double>(dest, true);
    const auto mesh_a = qd_python_numpy2arma_Mat<double>(mesh, true);
    const auto mtl_ind_a = qd_python_numpy2arma_Col<arma::uword>(mtl_ind, true);
    const auto mtl_prop_map = qd_python_dict2map<double>(mtl_prop);
    const auto sub_mesh_index_a = qd_python_numpy2arma_Col<unsigned>(sub_mesh_index, true);

    arma::uword n_pos = orig_a.n_rows;
    arma::uword n_seg = 0;
    if (lod == 1 || lod == 2)
        n_seg = 2;
    else if (lod == 3)
        n_seg = 3;
    else if (lod == 4)
        n_seg = 4;
    else if (lod == 5 || lod == 6)
        n_seg = 1;

    // Pre-allocate outputs in Python memory and map Armadillo wrappers to them
    arma::vec gain;
    arma::cube coord;
    auto gain_p = qd_python_init_output(n_pos, &gain);
    auto coord_p = qd_python_init_output(3, n_seg, n_pos, &coord);

    // Resolve optional pointer
    const arma::u32_vec *p_sub_mesh_index = sub_mesh_index_a.empty() ? nullptr : &sub_mesh_index_a;

    quadriga_lib::calc_diffraction_gain<double>(&orig_a, &dest_a, &mesh_a, &mtl_ind_a, &mtl_prop_map,
                                                center_freq, lod, &gain, &coord, verbose,
                                                p_sub_mesh_index, use_kernel, gpu_id, scalar_mode);

    return py::make_tuple(gain_p, coord_p);
}

// pybind11 declaration:
// m.def("calc_diffraction_gain", &calc_diffraction_gain,
//       py::arg("orig"),
//       py::arg("dest"),
//       py::arg("mesh"),
//       py::arg("mtl_ind"),
//       py::arg("mtl_prop"),
//       py::arg("center_frequency"),
//       py::arg("lod") = 2,
//       py::arg("verbose") = 0,
//       py::arg("sub_mesh_index") = py::none(),
//       py::arg("use_kernel") = 0,
//       py::arg("gpu_id") = 0,
//       py::arg("scalar_mode") = false);