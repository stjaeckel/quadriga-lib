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

- Evaluates the 11 large-scale fading generators of [[lsf_init_phases]] at the links: column 0 DS, 1 KF, 2 SF,
  3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 absolute TOA (Normal(0,1)), 9 ground permittivity, 10 UMa environment
  height hE (Uniform(0,1)); each generator with the sum-of-sinusoids frequencies of `preset` at its own
  decorrelation distance (a distance of 0 is used as 0.1 m)
- `scenpar` is completed and validated as in [[lsf_init_phases]]; the messages of [[scenpar_validate]] pass
  through unchanged
- Single mobility (`dual_mobility = False`): column g is the field of generator g at the Rx position,
  `x_g = s_g(rx)` with `s_g(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )` and the phases of
  slice 0; the values do not depend on `tx_pos`
- Dual mobility: the Tx is a second device. Its side uses slice 1, i.e. the field of the mirror generator
  `m(g)` at the Tx position (ASD and ASA, ESD and ESA exchanged). Self-mirrored generators (DS, KF, SF, XPR,
  TOA, columns 9 and 10) share one map: `x_g = ( s_g(rx) + s_g(tx) ) / sqrt( 2 + 2·A_g(D) )`, with the Tx-Rx
  distance `D` and the ACF `A_g` of the preset, linearly interpolated. The pairs are two maps:
  `x_g = ( s_g(rx) + s_m(tx) ) / sqrt(2)`. Columns 3 to 6 thus stay Normal(0,1), and ASD/ASA and ESD/ESA have
  the correlation `( A_g(D) + A_m(D) ) / 2` at the Tx-Rx distance, 1 at `D = 0` (then the two columns are
  equal)
- Swap reciprocity in dual mobility: swapping `tx_pos` and `rx_pos` leaves columns 0, 1, 2, 7, 8, 9, 10
  unchanged and swaps column 3 with 4 and column 5 with 6, exactly (the same evaluations in swapped roles), also
  for unequal decorrelation distances and with bias-removed phases, when both calls use the same `phase`
- Uniform columns 9 and 10: `0.5 · erfc( -x / sqrt(2) )` of the Normal value
- `horizontal = True`: the positions are projected on the x-y plane (z = 0) and the values do not depend on z;
  `False`: 3D positions. `D` is measured in the same space
- Floors: links with different floor IDs see different fields, links with equal IDs share one field. Floor `q`
  adds the offsets `2·pi · k · n / N + psi_b(n)` to the phases of all generators on both sides, with
  `N = n_coeff`, `k = q mod N`, `b = q div N`, `n = 0 ... N-1`, `psi_0 = 0` and `psi_b(n) = 2·pi · u_n` for
  `b >= 1`, where `u_n` are the uniform numbers of a `std::mt19937_64` seeded with `b` (the upper 53 bits of
  each raw output times 2^-53). Floor 0 and `floor_id = None` give the same values
- Fields of two floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3): exactly
  for two IDs in the same block of `N` IDs, with a random residual of standard deviation `1/sqrt(2N)` for IDs
  in different blocks (0.044 for 256 sinusoids). At other horizontal lags the cross-correlation of two floors
  is not zero (up to about 0.16 for 256 sinusoids), the same level as for random offsets. Mirror relations and
  swap reciprocity hold per floor
- `floor_id` is the caller's floor index of the UT, starting at 0: for the UT heights
  `hUT = 3·(n_fl - 1) + 1.5` of TR 38.901 Table 7.2-1 it is `n_fl - 1`; outdoor UTs use 0
- The value of a link does not depend on the other links of the call; `n_link = 0` returns an array of
  shape `(0, 11)`
- Precision: the sinusoids are evaluated by [[sos.val]] with the AVX2 kernel where available (sine and cosine
  in single precision, accumulation in double); the values deviate from a double-precision evaluation by less
  than `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset` and `horizontal` must be those of [[lsf_init_phases]]. A wrong preset with the same number of
  sinusoids (e.g. `Exp256` for `Comb256`) cannot be detected
- Validation order: `floor_id` (non-negative integers, checked by the wrapper), positions (3 columns, rows of
  `tx_pos`, finite values), `preset`, `scenpar`, size of `phase`, finite `phase`, length of `floor_id`

## Usage:
```
rnd = quadriga_lib.gbsm.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, preset, horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [[conf_file_read]] or [[scenpar_validate]] (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; only the 10 decorrelation distances are used
- **`phase`** — Phases in rad from [[lsf_init_phases]]; slice 0 = Rx side, slice 1 = Tx side; must be finite;
  `(n_coeff, 11, 2)`, `n_coeff` of `preset`
- **`tx_pos`** — Tx positions in m; one row is used for all links; not used in single mobility (but
  validated); `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`floor_id`** — Floor index per link, starting at 0; non-negative integers, given as int or float array;
  `None` or empty = all links on floor 0; `(n_link,)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move, the Tx side is evaluated as a second device;
  default: False
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; must match [[lsf_init_phases]];
  default: `"Comb256"`
- **`horizontal`** — `True`: 2D correlation in the x-y plane (TR 38.901 behaviour); `False`: 3D correlation
  (legacy QuaDRiGa behaviour); must match [[lsf_init_phases]]; default: True

## Output:
- **`rnd`** — Random values; columns 0 to 8 Normal(0,1), columns 9 and 10 Uniform(0,1); `(n_link, 11)`

## See also:
- [[lsf_init_phases]] (draws the phases)
- [[path_loss]] (column 10 as `rnd_he`)
- [[lsp_generate]] (the large-scale parameters from `rnd`)
- [[sos.val]] (evaluation of a sum-of-sinusoids generator)
MD!*/

py::array_t<double> lsf_draw(py::handle scenpar,
                             const py::array_t<double> &phase,
                             const py::array_t<double> &tx_pos,
                             const py::array_t<double> &rx_pos,
                             py::handle floor_id,
                             bool dual_mobility,
                             const std::string &preset,
                             bool horizontal)
{
    // Read inputs; the entry 'SubpathMethod' of the scenario dict is not used
    std::string subpath_method;
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");
    const auto phase_a = qd_python_numpy2arma_Cube<double>(phase, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto floor_id_in = qd_python_numpy2arma_Col<double>(floor_id, true);

    // Floor IDs are labels: read as double, checked, converted
    arma::uvec floor_id_a(floor_id_in.n_elem);
    for (arma::uword i = 0; i < floor_id_in.n_elem; ++i)
    {
        const double v = floor_id_in[i];
        if (!(v >= 0.0 && v < 18446744073709551616.0 && v == std::floor(v)))
            throw std::invalid_argument("Entries in 'floor_id' must be non-negative integers.");
        floor_id_a[i] = (arma::uword)v;
    }

    // Output allocation
    arma::mat rnd;
    auto rnd_py = qd_python_init_output(rx_pos_a.n_rows, (arma::uword)11, &rnd);

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id_a.is_empty() ? nullptr : &floor_id_a;

    // Call library function
    quadriga_lib::lsf_draw(scenpar_a, phase_a, tx_pos_a, rx_pos_a, rnd, p_floor_id, dual_mobility, preset,
                           horizontal);

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
