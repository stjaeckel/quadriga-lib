// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# lsf_init_phases
Draw the random phases of the large-scale generators of one BS site and scenario

- Draws the phases of the 11 large-scale generators (see the section description); [[lsf_draw]] evaluates them at the
  links. The caller stores `phase` to evaluate the same fields again along tracks or on maps
- Without bias removal, slice 0 (Rx side) equals [[sos.init_phases]] with 11 generators and the `n_coeff` of `preset`;
  slice 1 (Tx side) is its mirror, with and without bias removal
- Bias removal (`remove_bias = True` and at least 3 rows in `rx_pos`): [[sos.remove_bias]] sets the sample mean of each
  Normal field (columns 0 to 8) over the positions to 0; in dual mobility over `rx_pos` and `tx_pos` together. It
  removes variance unless the positions spread over many decorrelation distances
- In 2D mode the bias removal treats all positions as one floor: for a drop with UTs on several floors pass the
  positions of one floor or set `remove_bias = False`
- TR 38.901 clause 7.4.4 literally (exponential ACF, no bias removal): `preset = "Exp256"`, `remove_bias = False`. The
  default `"Comb256"` has the same decorrelation distance without the kink of the exponential ACF at zero distance, so
  the values evolve smoothly along a track

## Usage:
```
phase = quadriga_lib.gbsm.lsf_init_phases( scenpar, seed, tx_pos, rx_pos, dual_mobility, remove_bias, \
    preset, horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances are used
- **`seed`** — Seed of the phases; int from -1 to 2^63-1; -1 = system random device, new phases on each call; default:
  -1
- **`tx_pos`** — Tx positions in m; required for the bias removal in dual mobility, otherwise not used; `(1, 3)` or
  `(n_link, 3)`; default: None
- **`rx_pos`** — Rx positions of the drop in m for the bias removal; `None` or empty or fewer than 3 rows: no bias
  removal; `(n_link, 3)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move; default: False
- **`remove_bias`** — `True`: remove the sample mean of the Normal fields over the positions; default: True
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; sets `n_coeff`; default: `"Comb256"`
- **`horizontal`** — `True`: 2D mode (TR 38.901); `False`: 3D mode (legacy QuaDRiGa); default: True

## Output:
- **`phase`** — Phases in rad, in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side; `(n_coeff, 11, 2)`

## See also:
- [[lsf_draw]] (evaluates the phases at the links)
- [[ssf_init_phases]] (the small-scale sets, seeds derived from this seed)
- [[sos.remove_bias]] (the bias removal of one generator)
MD!*/

py::array_t<double> lsf_init_phases(py::handle scenpar,
                                    long long seed,
                                    py::handle tx_pos,
                                    py::handle rx_pos,
                                    py::handle dual_mobility,
                                    py::handle remove_bias,
                                    const std::string &preset,
                                    py::handle horizontal)
{
    // Read inputs; the entry 'SubpathMethod' of the scenario dict is not used
    std::string subpath_method;
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");
    const bool dual_mobility_b = qd_python_flag(dual_mobility, false, "dual_mobility");
    const bool remove_bias_b = qd_python_flag(remove_bias, true, "remove_bias");
    const bool horizontal_b = qd_python_flag(horizontal, true, "horizontal");
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);

    // Wrap optional pointers
    const arma::mat *p_tx_pos = tx_pos_a.is_empty() ? nullptr : &tx_pos_a;
    const arma::mat *p_rx_pos = rx_pos_a.is_empty() ? nullptr : &rx_pos_a;

    // Call library function; the size of 'phase' depends on the preset, so it is copied afterwards
    arma::cube phase;
    quadriga_lib::lsf_init_phases(scenpar_a, phase, (arma::sword)seed, p_tx_pos, p_rx_pos, dual_mobility_b,
                                  remove_bias_b, preset, horizontal_b);

    return qd_python_copy2numpy(&phase);
}

// pybind11 declaration:
// m.def("lsf_init_phases", &lsf_init_phases,
//       py::arg("scenpar"),
//       py::arg("seed") = -1,
//       py::arg("tx_pos") = py::none(),
//       py::arg("rx_pos") = py::none(),
//       py::arg("dual_mobility") = false,
//       py::arg("remove_bias") = true,
//       py::arg("preset") = "Comb256",
//       py::arg("horizontal") = true);
