// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm.draw
Spatially correlated normalized random values of one generator set at a set of links

- Evaluates the phases of [[gbsm.init_phases]] at the links, one column per generator; columns, distributions and
  decorrelation distances: see the set table in the section description. The value of a link depends neither on the
  other links of the call nor on the other columns; in single mobility it does not depend on `tx_pos`
- Dual mobility: the Tx is a second device on the same maps, its side of each generator uses the mirror generator.
  Every column keeps its distribution; swapping `tx_pos` and `rx_pos` exchanges the columns of each mirror pair and
  keeps the self-mirrored columns, exactly. The ray delays (block 14 of the shared `"ray"` set) change 2 to 8 times
  faster with a Tx move than plane waves at the Tx would (one decorrelation distance, from the cluster ASA)
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are uncorrelated at equal
  horizontal positions, exactly within a block of `n_coeff` consecutive IDs, otherwise up to a random residual of
  standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation
- A set never changes size. An empty set (`NumClusters = 1`) gives `(n_link, 0)` and accepts any empty `phase`
- The flags take a bool or `numpy.bool_`; `None` selects the default, any other type raises `TypeError`

## Usage:
```
rnd = quadriga_lib.gbsm.draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, center_freq, \
    dual_mobility, horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`, cluster DS and ASA
- **`set_type`** — Set type of [[gbsm.init_phases]]; str
- **`phase`** — Phases from [[gbsm.init_phases]], finite; `(n_coeff, n_gen, 2)`
- **`tx_pos`** — Tx positions in m; used in dual mobility, always validated; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`, `n_link = 0` allowed
- **`set_id`** — Set of [[gbsm.init_phases]]: `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`;
  default: -1
- **`preset`** — Preset of [[gbsm.init_phases]]; another preset with the same `n_coeff` is not detected; default:
  `"Comb256"`
- **`floor_id`** — Floor index of the UT per link, 0-based (`n_fl - 1` for the UT heights of TR 38.901 Table 7.2-1, 0
  for outdoor UTs); int or float array; `None` or empty = floor 0; 2D mode only; `(n_link,)`; default: None
- **`center_freq`** — Carrier frequencies in Hz (messages name it `center_frequency`); required by the shared `"ray"`
  set, whose block 14 uses their mean cluster delay spread: pass the same list as to [[ssf_rays]] (another list gives
  other `quadriga` rays); ignored and not checked for all other sets; `(n_freq,)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move; default: False
- **`horizontal`** — `True`: 2D mode (TR 38.901); `False`: 3D mode (legacy QuaDRiGa); default: True

## Output:
- **`rnd`** — Random values; `(n_link, n_gen)`

## See also:
- [[lsp_generate]] (the large-scale parameters from the `"lsf"` set)
- [[path_loss]] (column 10 of the `"lsf"` set as `rnd_he`)
- [[ssf_clusters]] (the clusters from the `"cluster"` sets)
- [[ssf_rays]] (the rays from the `"ray"` sets)
MD!*/

py::array_t<double> gbsm_draw(py::handle scenpar,
                              const std::string &set_type,
                              const py::array_t<double> &phase,
                              const py::array_t<double> &tx_pos,
                              const py::array_t<double> &rx_pos,
                              long long set_id,
                              const std::string &preset,
                              py::handle floor_id,
                              py::handle center_freq,
                              py::handle dual_mobility,
                              py::handle horizontal)
{
    // Read inputs
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const auto phase_a = qd_python_numpy2arma_Cube<double>(phase, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto floor_id_in = qd_python_numpy2arma_Col<double>(floor_id, true);
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const bool dual_mobility_b = qd_python_flag(dual_mobility, false, "dual_mobility");
    const bool horizontal_b = qd_python_flag(horizontal, true, "horizontal");

    const arma::uvec floor_id_a = qd_python_uint_entries(floor_id_in, "floor_id");

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id_a.is_empty() ? nullptr : &floor_id_a;
    const arma::vec *p_center_freq = center_freq.is_none() ? nullptr : &center_freq_a;

    // Output allocation in numpy memory: (n_link, n_gen), n_gen = columns of 'phase' (checked by the library)
    arma::mat rnd;
    auto rnd_py = qd_python_init_output(rx_pos_a.n_rows, phase_a.n_cols, &rnd);

    // Call library function
    quadriga_lib::gbsm_draw(scenpar_a, set_type, phase_a, tx_pos_a, rx_pos_a, rnd, (arma::sword)set_id, preset,
                            p_floor_id, p_center_freq, dual_mobility_b, horizontal_b);

    return rnd_py;
}

// pybind11 declaration:
// m.def("draw", &gbsm_draw,
//       py::arg("scenpar"),
//       py::arg("set_type"),
//       py::arg("phase").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("set_id") = -1,
//       py::arg("preset") = "Comb256",
//       py::arg("floor_id") = py::none(),
//       py::arg("center_freq") = py::none(),
//       py::arg("dual_mobility") = false,
//       py::arg("horizontal") = true);
