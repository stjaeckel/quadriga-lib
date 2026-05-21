// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# channel_export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g. for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` caps the total number of exported paths
- Takes raw channel data fields directly; no channel dict is required
- The per-snapshot fields `no_interact`, `interact_coord` and `coeff` each accept either a `list`
  (one entry per snapshot, ragged) or a single padded `ndarray` (MATLAB-style nD layout)

## Usage:
```
quadriga_lib.channel.channel_export_obj_file( fn, max_no_paths, gain_max, gain_min, colormap, radius_max, radius_min, 
    n_edges, rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff, coeff_re, coeff_im, i_snap )
```

## Inputs:
- **`fn`** — Output `.obj` file path; str
- **`max_no_paths`** — Max paths to export; 0 includes all paths above `gain_min`; default: 0
- **`gain_max`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5
- **`rx_pos`** — Receiver positions; ndarray `(3, n_snap)` or `(3, 1)`
- **`tx_pos`** — Transmitter positions; ndarray `(3, n_snap)` or `(3, 1)`
- **`no_interact`** — Number of interaction points of each path with the environment; uint32;
  either a `list` of `n_snap` arrays of shape `(n_path,)`, or an ndarray `(n_path, n_snap)`
- **`interact_coord`** — Interaction coordinates; either a `list` of `n_snap` arrays of shape
  `(3, sum(no_interact))`, or an ndarray `(3, max(sum(no_interact)), n_snap)`
- **`center_freq`** — Center frequency in Hz; ndarray `(n_snap,)` or scalar
- **`coeff`** — Channel coefficients, complex valued; either a `list` of `n_snap` arrays of shape
  `(n_rx, n_tx, n_path)`, or an ndarray `(n_rx, n_tx, n_path, n_snap)`; provide this or
  `coeff_re`/`coeff_im`, not both; default: None
- **`coeff_re`** — Real part of the channel coefficients; same format as `coeff` but real-valued;
  must be paired with `coeff_im`; default: None
- **`coeff_im`** — Imaginary part of the channel coefficients; same format as `coeff_re`;
  must be paired with `coeff_re`; default: None
- **`i_snap`** — Snapshot indices to include; 0-based; `None` or empty exports all; default: None

## Outputs:
- None; the OBJ file is written directly to disk
MD!*/

void channel_export_obj_file(const std::string &fn,
                             size_t max_no_paths,
                             double gain_max,
                             double gain_min,
                             const std::string &colormap,
                             double radius_max,
                             double radius_min,
                             size_t n_edges,
                             py::handle &rx_pos,
                             py::handle &tx_pos,
                             py::handle no_interact,
                             py::handle interact_coord,
                             py::handle &center_freq,
                             py::handle coeff,
                             py::handle coeff_re,
                             py::handle coeff_im,
                             py::handle i_snap)
{
    // Construct channel object from input data
    auto c = quadriga_lib::channel<double>();

    c.rx_pos = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    c.tx_pos = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    c.center_frequency = qd_python_numpy2arma_Col<double>(center_freq, true);

    // no_interact: list of 1D arrays (ragged) OR a 2D ndarray (n_path, n_snap)
    if (py::isinstance<py::list>(no_interact))
        c.no_interact = qd_python_list2vector_Col<unsigned>(py::reinterpret_borrow<py::list>(no_interact));
    else
    {
        const auto no_interact_a = qd_python_numpy2arma_Mat<unsigned>(no_interact, true);
        c.no_interact.reserve(no_interact_a.n_cols);
        for (arma::uword j = 0; j < no_interact_a.n_cols; ++j)
            c.no_interact.push_back(no_interact_a.col(j));
    }

    // interact_coord: list of 2D arrays (ragged) OR a 3D ndarray (3, X, n_snap)
    if (py::isinstance<py::list>(interact_coord))
        c.interact_coord = qd_python_list2vector_Mat<double>(py::reinterpret_borrow<py::list>(interact_coord));
    else
    {
        const auto interact_coord_a = qd_python_numpy2arma_Cube<double>(interact_coord, true);
        c.interact_coord.reserve(interact_coord_a.n_slices);
        for (arma::uword j = 0; j < interact_coord_a.n_slices; ++j)
            c.interact_coord.push_back(interact_coord_a.slice(j));
    }

    bool have_coeff = !coeff.is_none();
    bool have_cre = !coeff_re.is_none();
    bool have_cim = !coeff_im.is_none();

    if (have_coeff && (have_cre || have_cim))
        throw std::invalid_argument("Cannot provide both 'coeff' and 'coeff_re'/'coeff_im'.");
    if (have_cre != have_cim)
        throw std::invalid_argument("'coeff_re' and 'coeff_im' must both be provided.");
    if (!have_coeff && !have_cre)
        throw std::invalid_argument("Must provide either 'coeff' or both 'coeff_re' and 'coeff_im'.");

    if (have_coeff) // complex input: list of 3D arrays OR a 4D complex ndarray
    {
        if (py::isinstance<py::list>(coeff))
            qd_python_list2vector_Cube_Cplx<double>(py::reinterpret_borrow<py::list>(coeff), c.coeff_re, c.coeff_im);
        else
            qd_python_numpy2arma_vecCube_Cplx<double>(coeff, c.coeff_re, c.coeff_im);
    }
    else // split real input: each is a list of 3D arrays OR a 4D real ndarray
    {
        if (py::isinstance<py::list>(coeff_re))
            c.coeff_re = qd_python_list2vector_Cube<double>(py::reinterpret_borrow<py::list>(coeff_re));
        else
            c.coeff_re = qd_python_numpy2arma_vecCube<double>(coeff_re, true);

        if (py::isinstance<py::list>(coeff_im))
            c.coeff_im = qd_python_list2vector_Cube<double>(py::reinterpret_borrow<py::list>(coeff_im));
        else
            c.coeff_im = qd_python_numpy2arma_vecCube<double>(coeff_im, true);
    }

    // All three per-snapshot fields must have the same number of snapshots as the
    // coefficients, otherwise the trim loop below indexes the vectors out of bounds.
    const size_t n_snap = c.coeff_re.size();
    if (c.no_interact.size() != n_snap || c.interact_coord.size() != n_snap)
        throw std::invalid_argument("Number of snapshots in interact_coord must match coefficients.");

    for (size_t i = 0; i < n_snap; ++i)
    {
        c.delay.push_back(arma::cube(c.coeff_re[i].n_rows, c.coeff_re[i].n_cols, c.coeff_re[i].n_slices));
        unsigned sum_no_int = arma::sum(c.no_interact[i]);
        c.interact_coord[i] = arma::resize(c.interact_coord[i], 3, sum_no_int);
    }

    // Optional snapshot selection (0-based, empty = all)
    arma::uvec i_snap_a = qd_python_numpy2arma_Col<arma::uword>(i_snap, true);

    // Call library function
    c.write_paths_to_obj_file(fn, max_no_paths, gain_max, gain_min, colormap, i_snap_a, radius_max, radius_min, n_edges);
}