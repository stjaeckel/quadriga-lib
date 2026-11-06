// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# path_loss
Path loss and shadow-fading standard deviation of a set of links

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and carrier frequency;
  the TR 38.901 v19.5.0 conf files of the library reproduce Table 7.4.1-1 with them. The O2I penetration loss is not
  included (see [[o2i_loss]]). The models are evaluated as written for any positive frequency, also outside their
  validity range
- Notation: `f = center_freq/1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance, at least 0.1
  m; `hBS`, `hUT` the z coordinates of Tx and Rx; `lg = log10`; `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the
  shadow-fading std. Keys of `plpar` in parentheses: the required ones, after the semicolon the optional ones (`X = v`:
  default)
- Models (the key `model` of `plpar`):
  - `constant` (A): `A`
  - `logdist` (A, B, C; SF): `A·lg d3 + B + C·lg f`; `sf = SF`
  - `logdist_simple` (A, B): `A·lg d3 + B`
  - `winner_los` (A1, A2 different, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb = max(hBS, 1.5)`, `hm =
    max(hUT, 1.5)`, `Gj = Bj + Cj·lg f + Dj·lg hb + Ej·lg hm`, breakpoint `bp = 10^((G2 - G1)/(A1 - A2))`; `Aj·lg d2 +
    Gj + Fj·hm` and `sf = sigj` with `j = 1` for `d2 <= bp`, else `j = 2`
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as `winner_los`; `(Aj + Ahj·lg
    hb)·lg d2 + Bj + Cj·lg f + Dj·lg hb + Ej·lg hm + Fj·hm` with `j = 1` for `f < 1.5`, 2 for `f < 2`, else 3
  - `winner_pathloss` (A, B, C, X; SF): `A·lg d3 + B + C·lg(f/5) + X`; `sf = SF`
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2): `hb = max(hBS, 1.1·hE, 0.1)`,
    `hm = max(hUT, 1.1·hE, 0.1)`, `dBP = E·f·(hb - hE)·(hm - hE)`, `dBP3 = sqrt(dBP² + (hb - hm)²)`; `PL1(d3)`, `sf =
    sig1` for `d2 <= dBP`, else `PL1(dBP3) + A2·lg(d3/dBP3)`, `sf = sig2`
  - `tripple_slope`, spelled as in QuaDRiGa (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0,
    hE2 = 0, sig1..sig3): the `dual_slope` rule with the breakpoints `dBP1 = min(dBP1, dBP2)` and `dBP2` of `E1, hE1`
    and `E2, hE2`, the slopes `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax, Bx, Cx, sig): `hb =
    max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² +
    D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; with
    `LOSmax != 0` (which requires the `dual_slope` keys) at least the `dual_slope` loss; with `Ax`, `Bx` or `Cx` at
    least `Ax·lg d3 + Bx + Cx·lg f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for `sa <= 0`, else `A·lg
    d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by atmospheric gases over `sa` (1 to 100 GHz,
    constant beyond) if `usePLa != 0`
  - no model (`plpar` is `None`, `{}` or has no entry `model`): 0
- Where the model gives no `sf` (an `SF` or `sig` key not set, or a model without one): the SF std of [[lsp_generate]]
  at the geometric distance, so that passing `sf_sigma` to [[lsp_generate]] is neutral
- `hE_rand != 0` with `rnd_he` given: the random environment height of TR 38.901 Table 7.4.1-1 note 1, drawn from
  `rnd_he`, which makes it spatially consistent at the decorrelation distance `SF_lambda` (the TR draws it independently
  per link); the UT height is then clamped at `hE + 0.1` instead of `1.1·hE`. Along a track the loss beyond the
  breakpoint steps where hE changes; evaluate `rnd_he` once per track segment to avoid this
- Distances: the geometric distances of TR 38.901 eq. (7.4-1). With `d2d_in` (legacy QuaDRiGa), links with `d2d_in > 0`
  use the outdoor part of the distances instead, at least 10 m
- A required key that is missing is an error, naming `A1` for "A1 or A" and `E1` for "E1 or E"; every key that the model
  reads must have one finite value; the breakpoint factors `E`, `E1`, `E2` and `G2n` must be > 0

## Usage:
```
pl, sf_sigma = quadriga_lib.gbsm.path_loss( plpar, scenpar, center_freq, tx_pos, rx_pos, d2d_in, rnd_he )
```

## Inputs:
- **`plpar`** — Path-loss parameters; dict as returned by [[conf_file_read]]: the entry `model` (str) and one real
  scalar or 1D array per further parameter, named without "PL_"; keys not read are ignored
- **`scenpar`** — Scenario parameters; the SF std terms and `LSP_fc_min` are used
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0 (messages name it `center_frequency`); `(n_freq,)`
- **`tx_pos`** — Tx positions in m; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`d2d_in`** — Indoor 2D distance in m of the UT of each link (e.g. from [[o2i_loss]]), finite and >= 0; given =
  legacy outdoor distance; 0 = outdoor link; `None` or empty = geometric distance; `(n_link,)`; default: None
- **`rnd_he`** — Uniform(0,1) value per link for the random environment height, column 10 of [[gbsm.draw]] (`"lsf"`);
  `None` or empty = fixed `hE`; `(n_link,)`; default: None

## Outputs:
- **`pl`** — Path loss in dB; `(n_link, n_freq)`
- **`sf_sigma`** — Shadow-fading standard deviation in dB; `(n_link, n_freq)`

## See also:
- [[lsp_generate]] (takes `sf_sigma`)
- [[o2i_loss]] (penetration loss and indoor distance)
MD!*/

py::tuple path_loss(py::handle plpar,
                    py::handle scenpar,
                    const py::array_t<double> &center_freq,
                    const py::array_t<double> &tx_pos,
                    const py::array_t<double> &rx_pos,
                    py::handle d2d_in,
                    py::handle rnd_he)
{
    // Read inputs; the entry 'model' of 'plpar' is the model
    std::string pl_model;
    std::vector<std::string> pl_order;
    const auto plpar_a = qd_python_dict2plpar(plpar, pl_model, pl_order, "plpar");
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto d2d_in_a = qd_python_numpy2arma_Col<double>(d2d_in, true);
    const auto rnd_he_a = qd_python_numpy2arma_Col<double>(rnd_he, true);

    // Output allocation
    arma::mat pl, sf_sigma;
    auto pl_py = qd_python_init_output(rx_pos_a.n_rows, center_freq_a.n_elem, &pl);
    auto sf_sigma_py = qd_python_init_output(rx_pos_a.n_rows, center_freq_a.n_elem, &sf_sigma);

    // Wrap optional pointers
    const arma::vec *p_d2d_in = d2d_in_a.is_empty() ? nullptr : &d2d_in_a;
    const arma::vec *p_rnd_he = rnd_he_a.is_empty() ? nullptr : &rnd_he_a;

    // Call library function
    quadriga_lib::path_loss(pl_model, plpar_a, scenpar_a, center_freq_a, tx_pos_a, rx_pos_a, pl, &sf_sigma, p_d2d_in,
                            p_rnd_he);

    return py::make_tuple(pl_py, sf_sigma_py);
}

// pybind11 declaration:
// m.def("path_loss", &path_loss,
//       py::arg("plpar"),
//       py::arg("scenpar"),
//       py::arg("center_freq").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("d2d_in") = py::none(),
//       py::arg("rnd_he") = py::none());
