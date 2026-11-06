// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"
#include "quadriga_gbsm_helpers.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm.init_phases
SOS phases of one random set of a BS site and scenario

- Draws the SOS phases of one random set (set table in [[gbsm.draw]]): the random phases of the sinusoids of its
  [sum-of-sinusoids (SOS)](python_api.html#sec-sum-of-sinusoids-random-number-generator) generators, as
  [[sos.init_phases]]; [[gbsm.draw]] evaluates them at the links. Store `phase` or the seed to evaluate the same fields
  again along tracks or on maps
- The SOS phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms (bias-removed `"lsf"`
  sets within about 2e-14 rad); the sets are independent; each scenario of a site needs its own seed
- `"lsf"` bias removal (`remove_bias`): [[sos.remove_bias]] sets the sample mean of columns 0 to 8 over the positions
  (Rx, and Tx in dual mobility; in 2D mode all on floor 0) to 0, so these means do not vary from drop to drop
- `NumClusters = 1`: all sets but `"lsf"` are empty
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal
- The flags take a bool or `numpy.bool_`; `None` selects the default, any other type raises `TypeError`

## Usage:
```
phase = quadriga_lib.gbsm.init_phases( scenpar, set_type, seed, set_id, preset, tx_pos, rx_pos, dual_mobility, \
    remove_bias, horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances, `NumClusters` and `NumSubPaths` are used
- **`set_type`** — `"lsf"`, `"cluster"` or `"ray"`; str
- **`seed`** — Seed of the site and scenario; int from -1 to 2^63-1; -1 = system random device, new SOS phases on each
  call; default: -1
- **`set_id`** — `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`; default: -1
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; sets `n_coeff`; default: `"Comb256"`
- **`tx_pos`**, **`rx_pos`** — Bias-removal positions in m as in [[gbsm.draw]]; `"lsf"` only, else `ValueError`;
  default: None
- **`dual_mobility`** — `"lsf"` with bias removal: Tx and Rx both move; as in [[gbsm.draw]]; default: False
- **`remove_bias`** — `"lsf"` only: remove the sample mean over the positions; needs `rx_pos` with at least 3 rows, else
  no effect; default: True
- **`horizontal`** — `"lsf"` with bias removal: `True` 2D, `False` 3D mode; as in [[gbsm.draw]]; default: True

## Output:
- **`phase`** — SOS phases in rad, in [-pi, pi), row = sinusoid, column = generator; slice 0 = Rx side, slice 1 = Tx
  side; `(n_coeff, n_gen, 2)`

## See also:
- [[gbsm.draw]] (evaluates the SOS phases at the links)
- [[sos.init_phases]], [[sos.val]] (SOS phases and evaluation of one generator), [[sos.remove_bias]] (its bias removal)
MD!*/

py::array_t<double> gbsm_init_phases(py::handle scenpar,
                                     const std::string &set_type,
                                     long long seed,
                                     long long set_id,
                                     const std::string &preset,
                                     py::handle tx_pos,
                                     py::handle rx_pos,
                                     py::handle dual_mobility,
                                     py::handle remove_bias,
                                     py::handle horizontal)
{
    // Read inputs
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const bool dual_mobility_b = qd_python_flag(dual_mobility, false, "dual_mobility");
    const bool remove_bias_b = qd_python_flag(remove_bias, true, "remove_bias");
    const bool horizontal_b = qd_python_flag(horizontal, true, "horizontal");

    // Wrap optional pointers
    const arma::mat *p_tx_pos = tx_pos_a.is_empty() ? nullptr : &tx_pos_a;
    const arma::mat *p_rx_pos = rx_pos_a.is_empty() ? nullptr : &rx_pos_a;

    // Output allocation in numpy memory: n_coeff of the preset, n_gen of the set from a completed copy of 'scenpar';
    // with an unknown preset or set type or an invalid 'scenpar' the call throws before any output is needed
    arma::cube phase;
    py::array_t<double> phase_py;
    const bool lsf = set_type == "lsf";
    const arma::uword n_coeff = qd_gbsm_n_coeff(preset);
    bool known = n_coeff > 0 && (lsf || set_type == "cluster" || set_type == "ray");
    auto sp = scenpar_a;
    try
    {
        if (known)
            quadriga_lib::scenpar_validate(sp);
    }
    catch (const std::exception &)
    {
        known = false;
    }
    if (known)
        phase_py = qd_python_init_output(n_coeff, qd_gbsm_n_gen(sp, set_type, (arma::sword)set_id), (arma::uword)2,
                                         &phase);

    // Call library function
    quadriga_lib::gbsm_init_phases(scenpar_a, set_type, phase, (arma::sword)seed, (arma::sword)set_id, preset, p_tx_pos,
                                   p_rx_pos, dual_mobility_b, remove_bias_b, horizontal_b);

    return phase_py;
}

// pybind11 declaration:
// m.def("init_phases", &gbsm_init_phases,
//       py::arg("scenpar"),
//       py::arg("set_type"),
//       py::arg("seed") = -1,
//       py::arg("set_id") = -1,
//       py::arg("preset") = "Comb256",
//       py::arg("tx_pos") = py::none(),
//       py::arg("rx_pos") = py::none(),
//       py::arg("dual_mobility") = false,
//       py::arg("remove_bias") = true,
//       py::arg("horizontal") = true);
