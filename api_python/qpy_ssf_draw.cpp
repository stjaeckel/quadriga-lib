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

- Evaluates the generators of a set of [[ssf_init_phases]] at the links. With `N = NumClusters - 1`, column
  `c = b·N + n` belongs to block `b` and NLOS cluster `n` (0-based). Shared cluster set (`set_id = -1`, `9·N`
  columns): block 0 cluster delay (Uniform(0,1)), blocks 1, 2 sign of AOD, AOA, 3, 4 sign of ZOD, ZOA, 5, 6 angle
  offset of AOD, AOA, 7, 8 angle offset of ZOD, ZOA (Normal(0,1)). Per-frequency set (`set_id = k >= 0`, `N`
  columns): cluster shadowing (Normal(0,1))
- All generators use the decorrelation distance `SC_lambda` (TR 38.901 Table 7.6.3.1-2); `SC_lambda = 0`
  (spatial consistency off) is used as 0.1 m, which gives fields that decorrelate within 0.1 m
- `scenpar` is completed and validated as in [[ssf_init_phases]]; the messages of [[scenpar_validate]] pass
  through unchanged
- Single mobility (`dual_mobility = False`): column `c` is the field of generator `c` at the Rx position,
  `x_c = s_c(rx)` with `s_c(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )` and the phases of slice 0; the
  values do not depend on `tx_pos`
- Dual mobility: the Tx is a second device. Its side uses slice 1, i.e. the field of the mirror generator at the Tx
  position (departure and arrival blocks exchanged). Self-mirrored columns (delay, shadowing) share one map:
  `x_c = ( s_c(rx) + s_c(tx) ) / sqrt( 2 + 2·A(D) )`, with the Tx-Rx distance `D` and the ACF `A` of the preset,
  linearly interpolated. The pair columns are two maps: `x_c = ( s_c(rx) + s_m(tx) ) / sqrt(2)`
- Swap reciprocity in dual mobility (S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading
  Model for Multiple Frequencies", 2019, eqs. (3)-(5)): swapping `tx_pos` and `rx_pos` leaves the delay and the
  shadowing blocks unchanged and exchanges block 1 with 2, 3 with 4, 5 with 6 and 7 with 8, exactly (the same
  evaluations in swapped roles)
- Uniform block 0 of the shared set: `0.5 · erfc( -x / sqrt(2) )` of the Normal value
- `horizontal = True`: the positions are projected on the x-y plane (z = 0) and the values do not depend on z;
  `False`: 3D positions (legacy). `D` is measured in the same space
- Floors as in [[lsf_draw]] (same phase offsets per floor ID; TR 38.901 clause 7.6.3.3: SSF variables are
  uncorrelated across floors): pass the same `floor_id` as to [[lsf_draw]]; only in 2D mode
- The value of a link does not depend on the other links of the call, and a column does not depend on the phases of
  the other columns (append-only: appended blocks leave the existing columns unchanged bit for bit)
- Clusters by TR 38.901 clause 7.5 under spatial consistency: clause 7.6.3.1 keeps the signs and the random offsets
  of the clusters unchanged per drop; the caller evaluates them at the drop or segment-start positions and reuses
  those values along the track
- Precision: the sinusoids are evaluated by [[sos.val]] with the AVX2 kernel where available (sine and cosine in
  single precision, accumulation in double); the values deviate from a double-precision evaluation by less than
  `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset`, `horizontal` and the set type (`set_id < 0` or `>= 0`) must match [[ssf_init_phases]]
- `n_link = 0` (`rx_pos` of shape `(0, 3)`) returns `rnd` with shape `(0, n_gen)`; `NumClusters = 1` returns
  `(n_link, 0)` and accepts any empty `phase`
- The flags take a bool or `numpy.bool_`; `None` selects the default, any other type raises `TypeError`
- Validation order: positions (3 columns, rows of `tx_pos`, finite values), `set_id`, `preset`, `scenpar`, size of
  `phase`, finite `phase`, length of `floor_id`, `floor_id` with `horizontal = False`

## Usage:
```
rnd = quadriga_lib.gbsm.ssf_draw( scenpar, phase, tx_pos, rx_pos, set_id, floor_id, dual_mobility, preset, \
    horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [[conf_file_read]] or [[scenpar_validate]] (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; `NumClusters` and `SC_lambda` are used
- **`phase`** — Phases in rad from [[ssf_init_phases]]; slice 0 = Rx side, slice 1 = Tx side; must be finite;
  `(n_coeff, n_gen, 2)`, `n_coeff` of `preset`, `n_gen` of the set type
- **`tx_pos`** — Tx positions in m; one row is used for all links; not used in single mobility (but
  validated); `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`set_id`** — Set type as in [[ssf_init_phases]]: `-1` = shared cluster set, `k >= 0` = per-frequency set;
  int; default: -1
- **`floor_id`** — Floor index per link, starting at 0; non-negative integers below 2^53, given as int or float
  array; `None` or empty = all links on floor 0; in dual mobility the floor of both devices; only with
  `horizontal = True`; `(n_link,)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move, the Tx side is evaluated as a second device;
  default: False
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; must match [[ssf_init_phases]];
  default: `"Comb256"`
- **`horizontal`** — `True`: 2D correlation in the x-y plane (TR 38.901 behaviour); `False`: 3D correlation
  (legacy QuaDRiGa behaviour); must match [[ssf_init_phases]]; default: True

## Output:
- **`rnd`** — Random values; shared set: columns 0 to N-1 Uniform(0,1), the others Normal(0,1); per-frequency
  set: Normal(0,1); `(n_link, n_gen)`

## See also:
- [[ssf_init_phases]] (draws the phases)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_draw]] (the large-scale values, same floors and mobility rules)
- [[sos.val]] (evaluation of a sum-of-sinusoids generator)
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

    // Floor IDs are labels: read as double, checked, converted; integers from 2^53 on are not exact in a double
    arma::uvec floor_id_a(floor_id_in.n_elem);
    for (arma::uword i = 0; i < floor_id_in.n_elem; ++i)
    {
        const double v = floor_id_in[i];
        if (!(v >= 0.0 && v < 9007199254740992.0 && v == std::floor(v)))
            throw std::invalid_argument("Entries in 'floor_id' must be non-negative integers below 2^53.");
        floor_id_a[i] = (arma::uword)v;
    }

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
