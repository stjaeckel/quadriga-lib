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

- The large-scale fading (LSF) of a site is described by 11 spatially correlated sum-of-sinusoids generators,
  one column each: 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 absolute TOA (Normal fields),
  9 ground permittivity, 10 UMa environment height hE (Uniform fields)
- Their decorrelation distances are the scenario parameters `DS_lambda`, `KF_lambda`, `SF_lambda`,
  `AS_D_lambda`, `AS_A_lambda`, `ES_D_lambda`, `ES_A_lambda`, `XPR_lambda`, `absTOA_lambda`, `SC_lambda` and
  again `SF_lambda` for hE; a decorrelation distance of 0 is used as 0.1 m. The generator of column 8 exists
  also when the absolute TOA is disabled
- This function draws the phases; [[lsf_draw]] evaluates them at the links. The caller stores `phase` and
  evaluates the same fields again along tracks or on maps
- `scenpar` is completed with the defaults of missing keys and validated as in [[scenpar_validate]]; its
  messages (e.g. for an unknown key or a negative decorrelation distance) pass through unchanged as
  `ValueError`. An invalid value of a key that is not used here raises as well
- Slice 0 (Rx side) holds the phases of the 11 generators. Without bias removal it equals
  `quadriga_lib.sos.init_phases( n_coeff, 11, seed )` of [[sos.init_phases]], with `n_coeff` the number of
  sinusoids of `preset`. The same seed gives the same phases
- Slice 1 (Tx side) is always the mirror of slice 0:
  `phase[:, :, 1] = phase[:, [0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10], 0]`, i.e. ASD and ASA and ESD and ESA are
  exchanged, all others mirror themselves. For dual mobility the departure spreads at the Tx are thus the
  arrival spreads of the Tx as a second device
- Bias removal (`remove_bias = True`, `rx_pos` given, `n_link > 2`): one realisation of a correlated field has
  a non-zero mean over a drop. For each Normal generator (columns 0 to 8) one common phase offset is applied
  with [[sos.remove_bias]], so that the sample mean of its field over the positions is exactly 0. Single
  mobility: over the Rx positions; `tx_pos` is not used. Dual mobility: over the Rx and Tx positions together
  (`2·n_link` positions), because the field of a generator is shared by both devices (the Tx side of ASD is
  the ASA field); `tx_pos` is then required. The Uniform generators 9 and 10 are never corrected
- The correction is applied to slice 0 and the Tx side is mirrored afterwards, so the mirror relation holds
  exactly with and without bias removal. Swapping `tx_pos` and `rx_pos` in dual mobility gives the same phases
  up to rounding (the sums over the positions are formed in another order)
- Consequences for [[lsf_draw]] at the same positions: in single mobility the link mean of columns 0 to 8 is 0;
  in dual mobility the link mean of DS, KF, SF, XPR and TOA is 0 when all links have the same Tx-Rx distance
  (the normalisation is then the same for all links), while ASD/ASA and ESD/ESA have a zero link mean only
  approximately (their two devices see different fields, each corrected over the union of the positions)
- The positions are used as in [[lsf_draw]]: projected on the x-y plane (z = 0) for `horizontal = True`, in
  3D for `horizontal = False`. `preset` and `horizontal` must be the same in [[lsf_draw]]
- All positions in `rx_pos` are treated as one field (floor 0): one phase correction per generator cannot zero
  the means of several floor fields at once. For a drop with UTs on several floors pass the positions of one
  floor (e.g. the links of floor 0) or set `remove_bias = False`; the fields of the other floors keep their
  sample mean
- The correction removes variance: the field is conditioned on a zero mean over the positions, which costs
  little only when the positions spread over many decorrelation distances (see [[sos.remove_bias]])
- With fewer than 3 positions, without `rx_pos` or with `remove_bias = False` no correction is made
- The flags `dual_mobility`, `remove_bias` and `horizontal` take a bool or `numpy.bool_`; `None` selects the
  default, any other type raises `TypeError`
- Validation order: `seed`, `preset`, `scenpar`, then the positions (3 columns, rows of `tx_pos` 1 or
  `n_link` when both are given, finite values), then the requirement of `tx_pos` for the bias removal in
  dual mobility

## Usage:
```
phase = quadriga_lib.gbsm.lsf_init_phases( scenpar, seed, tx_pos, rx_pos, dual_mobility, remove_bias, \
    preset, horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [[conf_file_read]] or [[scenpar_validate]] (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; only the 10 decorrelation distances are used
- **`seed`** — Seed of the random phases; int from -1 to 2^63-1; -1 = system random device, a new set of
  phases on each call; default: -1
- **`tx_pos`** — Tx positions in m; used only for the bias removal with dual mobility, required there; one row
  is used for all links; `(1, 3)` or `(n_link, 3)`; default: None
- **`rx_pos`** — Rx positions of the drop in m (one floor) for the bias removal; `None`, empty or fewer than
  3 rows = no bias removal; `(n_link, 3)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move, the Tx is a second device on the same maps;
  default: False
- **`remove_bias`** — `True`: remove the sample mean of the Normal fields over the positions; default: True
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; sets `n_coeff`; default: `"Comb256"`
- **`horizontal`** — `True`: 2D correlation in the x-y plane (TR 38.901 behaviour); `False`: 3D correlation
  (legacy QuaDRiGa behaviour); default: True

## Output:
- **`phase`** — Phases in rad; slice 0 = Rx side, slice 1 = Tx side (mirror of slice 0); all values in
  [-pi, pi); `(n_coeff, 11, 2)`

## See also:
- [[conf_file_read]] (the scenario parameters)
- [[lsf_draw]] (evaluates the phases at the links)
- [[sos.init_phases]] (the raw phases)
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
