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

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_prop, center_frequency,
    lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode )

# Unpacked outputs
gain, coord = RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_prop, center_frequency,
    lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode )
```

## Inputs:
- **`orig`** — TX positions; `(n_pos, 3)`
- **`dest`** — RX positions; `(n_pos, 3)`
- **`mesh`** — Triangle vertices, each row [X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3]; `(n_mesh, 9)`
- **`mtl_prop`** — Material properties; see [[obj_file_read]]; `(n_mesh, 9)`
- **`center_frequency`** — Center frequency
- **`lod`** *(optional)* — Level of detail (0–6), controls `n_path` and `n_seg`; see [[generate_diffraction_paths]]
- **`verbose`** *(optional)* — Verbosity level
- **`sub_mesh_index`** *(optional)* — 0-based sub-mesh index for acceleration; see [[triangle_mesh_segmentation]]; `(n_mesh,)`
- **`use_kernel`** *(optional)* — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable
- **`gpu_id`** *(optional)* — CUDA device ID; ignored for non-CUDA kernels
- **`scalar_mode`** *(optional)* — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging. Default `false` (EM mode). 

## Outputs:
- **`gain`** *(optional)* — Diffraction gain per TX-RX pair, linear scale; `(n_pos,)`
- **`coord`** *(optional)* — Diffracted path coordinates excluding endpoints; `(3, n_seg-1, n_pos)`

## See also:
- [[generate_diffraction_paths]] (controls path/segment count via `lod`)
- [[triangle_mesh_segmentation]] (generates `sub_mesh_index`)
- [[obj_file_read]] (defines `mtl_prop` format)
MD!*/

py::tuple calc_diffraction_gain(const py::array_t<double> &orig,
                                const py::array_t<double> &dest,
                                const py::array_t<double> &mesh,
                                const py::array_t<double> &mtl_prop,
                                double center_freq,
                                int lod,
                                int verbose,
                                const py::array_t<unsigned> &sub_mesh_index,
                                int use_kernel,
                                int gpu_id, 
                                bool scalar_mode)
{
    const auto orig_arma = qd_python_numpy2arma_Mat(orig, true);
    const auto dest_arma = qd_python_numpy2arma_Mat(dest, true);
    const auto mesh_arma = qd_python_numpy2arma_Mat(mesh, true);
    const auto mtl_prop_arma = qd_python_numpy2arma_Mat(mtl_prop, true);
    const auto sub_mesh_index_arma = qd_python_numpy2arma_Col(sub_mesh_index, true);

    arma::uword n_pos = orig_arma.n_rows;
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

    // Resolve optional pointers
    const arma::u32_vec *p_sub_mesh_index = sub_mesh_index_arma.empty() ? nullptr : &sub_mesh_index_arma;

    quadriga_lib::calc_diffraction_gain<double>(&orig_arma, &dest_arma, &mesh_arma, &mtl_prop_arma,
                                                center_freq, lod, &gain, &coord, verbose,
                                                p_sub_mesh_index, use_kernel, gpu_id, scalar_mode);

    return py::make_tuple(gain_p, coord_p);
}