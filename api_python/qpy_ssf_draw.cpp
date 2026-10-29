// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_draw
Spatially correlated normalized random values of one small-scale fading generator set at a set of links

- Evaluates a set of [[ssf_init_phases]] at the links with the rules of [[lsf_draw]] for single and dual mobility,
  floors, 2D/3D mode and precision; every generator uses the decorrelation distance `SC_lambda` (TR 38.901 Table
  7.6.3.1-2; 0 = spatial consistency off, used as 0.1 m)
- Columns, `N = NumClusters - 1`; column `b·N + n` belongs to block `b` and NLOS cluster `n`:
  - shared set (`set_id = -1`, `9·N` columns): block 0 cluster delay, Uniform(0,1); blocks 1, 2 sign of AOD, AOA;
    3, 4 sign of ZOD, ZOA; 5, 6 angle offset of AOD, AOA; 7, 8 angle offset of ZOD, ZOA; Normal(0,1)
  - per-frequency set (`set_id = k`, `N` columns): cluster shadowing, Normal(0,1)
- Dual mobility: swapping `tx_pos` and `rx_pos` exchanges the blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 and keeps
  the delay and shadowing blocks, exactly (D2D reciprocity of S. Jaeckel et al., "A Spatially Consistent Geometric
  D2D Small-Scale Fading Model for Multiple Frequencies", 2019)
- A column depends neither on the other columns nor on the other links; later versions append blocks and keep the
  values of the existing columns
- `preset`, `horizontal` and the set type must match [[ssf_init_phases]]. Zero links give `(0, n_gen)`;
  `NumClusters = 1` gives `(n_link, 0)`

## Usage:
```
rnd = quadriga_lib.gbsm.ssf_draw( scenpar, phase, tx_pos, rx_pos, set_id, floor_id, dual_mobility, preset, \
    horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters as in [[ssf_init_phases]]; `NumClusters` and `SC_lambda` are used
- **`phase`** — Phases from [[ssf_init_phases]], finite; `(n_coeff, n_gen, 2)`
- **`tx_pos`** — Tx positions in m, one row for all links; used in dual mobility; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`set_id`** — `-1` = shared cluster set; `k >= 0` = per-frequency set; int; default: -1
- **`floor_id`** — Floor index per link as in [[lsf_draw]], 2D mode only; int or float array of non-negative
  integers; `None` or empty = floor 0; `(n_link,)`; default: None
- **`dual_mobility`** — `True`: the Tx is a second moving device; bool; default: False
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; default: `"Comb256"`
- **`horizontal`** — `True`: 2D correlation (TR 38.901); `False`: 3D (legacy QuaDRiGa); bool; default: True

## Output:
- **`rnd`** — Random values; `(n_link, n_gen)`

## See also:
- [[ssf_init_phases]] (draws the phases)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_draw]] (the large-scale values; mobility, floor and precision rules)
MD!*/

py::array_t<double> ssf_draw(py::handle scenpar,
                             const py::array_t<double> &phase,
                             const py::array_t<double> &tx_pos,
                             const py::array_t<double> &rx_pos,
                             long long set_id,
                             py::handle floor_id,
                             py::handle dual_mobility,
                             const std::string &preset,
                             py::handle horizontal)
{
    // Read inputs; the entry 'SubpathMethod' of the scenario dict is not used
    std::string subpath_method;
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");
    const auto phase_a = qd_python_numpy2arma_Cube<double>(phase, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto floor_id_in = qd_python_numpy2arma_Col<double>(floor_id, true);
    const bool dual_mobility_b = qd_python_flag(dual_mobility, false, "dual_mobility");
    const bool horizontal_b = qd_python_flag(horizontal, true, "horizontal");

    const arma::uvec floor_id_a = qd_python_uint_entries(floor_id_in, "floor_id");

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id_a.is_empty() ? nullptr : &floor_id_a;

    // Call library function; the number of columns depends on 'scenpar', so 'rnd' is copied afterwards
    arma::mat rnd;
    quadriga_lib::ssf_draw(scenpar_a, phase_a, tx_pos_a, rx_pos_a, rnd, (arma::sword)set_id, p_floor_id,
                           dual_mobility_b, preset, horizontal_b);

    return qd_python_copy2numpy(&rnd);
}

// pybind11 declaration:
// m.def("ssf_draw", &ssf_draw,
//       py::arg("scenpar"),
//       py::arg("phase").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("set_id") = -1,
//       py::arg("floor_id") = py::none(),
//       py::arg("dual_mobility") = false,
//       py::arg("preset") = "Comb256",
//       py::arg("horizontal") = true);
