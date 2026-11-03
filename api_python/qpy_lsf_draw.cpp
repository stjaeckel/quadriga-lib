// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# lsf_draw
Spatially correlated normalized random values of the large-scale generators at a set of links

- Evaluates the phases of [[lsf_init_phases]] at the links, one column per generator (see the section description). The
  value of a link does not depend on the other links of the call; in single mobility it does not depend on `tx_pos`
- Dual mobility: the Tx is a second device on the same maps. Every column keeps its distribution; ASD and ASA, ESD and
  ESA of a link are correlated by the ACF at the Tx-Rx distance (equal at distance 0). Swapping `tx_pos` and `rx_pos`
  swaps the columns 3 and 4, 5 and 6 and leaves the others unchanged, exactly
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are uncorrelated at equal
  horizontal positions, exactly within a block of `n_coeff` consecutive IDs, otherwise up to a random residual of
  standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation (AVX2 kernel of [[sos.val]])

## Usage:
```
rnd = quadriga_lib.gbsm.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, preset, horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances are used
- **`phase`** — Phases from [[lsf_init_phases]], finite; `(n_coeff, 11, 2)`
- **`tx_pos`** — Tx positions in m; used in dual mobility, always validated; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`floor_id`** — Floor index of the UT per link, starting at 0; int or float array of non-negative integers below 2^53
  (`n_fl - 1` for the UT heights of TR 38.901 Table 7.2-1, 0 for outdoor UTs); `None` or empty = floor 0; 2D mode only;
  `(n_link,)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move; default: False
- **`preset`** — Sum-of-sinusoids preset of [[lsf_init_phases]], see [[sos.preset]]; a different preset with the same
  number of sinusoids is not detected; default: `"Comb256"`
- **`horizontal`** — Mode of [[lsf_init_phases]]; default: True

## Output:
- **`rnd`** — Random values, columns as in the section description; `(n_link, 11)`

## See also:
- [[lsf_init_phases]] (draws the phases)
- [[path_loss]] (column 10 as `rnd_he`)
- [[lsp_generate]] (the large-scale parameters from `rnd`)
MD!*/

py::array_t<double> lsf_draw(py::handle scenpar,
                             const py::array_t<double> &phase,
                             const py::array_t<double> &tx_pos,
                             const py::array_t<double> &rx_pos,
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

    // Output allocation
    arma::mat rnd;
    auto rnd_py = qd_python_init_output(rx_pos_a.n_rows, (arma::uword)11, &rnd);

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id_a.is_empty() ? nullptr : &floor_id_a;

    // Call library function
    quadriga_lib::lsf_draw(scenpar_a, phase_a, tx_pos_a, rx_pos_a, rnd, p_floor_id, dual_mobility_b, preset,
                           horizontal_b);

    return rnd_py;
}

// pybind11 declaration:
// m.def("lsf_draw", &lsf_draw,
//       py::arg("scenpar"),
//       py::arg("phase").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("floor_id") = py::none(),
//       py::arg("dual_mobility") = false,
//       py::arg("preset") = "Comb256",
//       py::arg("horizontal") = true);
