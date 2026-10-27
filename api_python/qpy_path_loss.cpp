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

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and carrier
  frequency. `plpar` is the dict of [[conf_file_read]]: its entry `model` is the model (value of `PL_model`), the
  other entries are the `PL_` keys without the prefix; the TR 38.901 v19.5.0 conf files in `data/config`
  reproduce Table 7.4.1-1 with them
- Notation: `f = fc / 1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance in m; `hBS`,
  `hUT` the z coordinates of Tx and Rx (no ground model); `[X]` the value of key `X`, or its default if the key is
  missing (`X = v` in the lists below); `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the shadow-fading std
- Models, required keys and defaults:
  - `constant` (A): `pl = A`
  - `logdist` (A, B, C; SF optional): `pl = A·lg(d3) + B + C·lg(f)`; `sf = [SF]` if present
  - `logdist_simple` (A, B): `pl = A·lg(d3) + B`
  - `winner_los` (A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb = max(hBS, 1.5)`,
    `hm = max(hUT, 1.5)` of each link; `G = B1 + C1·lg f + D1·lg hb + E1·lg hm`,
    `H = B2 + C2·lg f + D2·lg hb + E2·lg hm`, breakpoint `bp = 10^((H - G)/(A1 - A2))`;
    `d2 <= bp`: `pl = A1·lg d2 + G + F1·hm`, `sf = sig1`; else `pl = A2·lg d2 + H + F2·hm`, `sf = sig2`
    (continuous at `bp` for `F1 = F2`); `A1` and `A2` must differ
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as `winner_los`;
    `L_j = (A_j + Ah_j·lg hb)·lg d2 + B_j + C_j·lg f + D_j·lg hb + E_j·lg hm + F_j·hm`; `pl = L_1` for `f < 1.5`,
    `L_2` for `1.5 <= f < 2`, `L_3` for `f >= 2`
  - `winner_pathloss` (A, B, C, X; SF optional): `pl = A·lg d3 + B + C·lg(f/5) + X`; `sf = [SF]` if present
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2 optional):
    `hb = max(hBS, 1.1·hE, 0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`; `dBP = E·f·(hb - hE)·(hm - hE)`;
    `d2 <= dBP`: `pl = PL1(d3)`, `sf = [sig1]`; `d2 > dBP`: `pl = PL1(dBP3) + A2·lg(d3/dBP3)`, `sf = [sig2]`
    (each only if present), with the 3D breakpoint distance `dBP3 = sqrt(dBP² + (hb - hm)²)`. This is Table 7.4.1-1
    for UMa, UMi, InH and InF; for RMa and SMa LOS the TR uses the 2D breakpoint distance in PL2, which differs by
    at most 0.034 dB over the TR ranges
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0, sig1..sig3
    optional): `hb = max(hBS, 1.1·hE1, 1.1·hE2, 0.1)`, `hm` the same; `dBPj = Ej·f·(hb - hEj)·(hm - hEj)`,
    `dBP1 = min(dBP1, dBP2)`, `dBPj3 = sqrt(dBPj² + (hb - hm)²)`; `LBP1 = PL1(dBP13)`,
    `LBP2 = LBP1 + A2·lg(dBP23/dBP13)`; `d2 <= dBP1`: `PL1(d3)`, `sf = [sig1]`; `dBP1 < d2 <= dBP2`:
    `LBP1 + A2·lg(d3/dBP13)`, `sf = [sig2]`; `d2 > dBP2`: `LBP2 + A3·lg(d3/dBP23)`, `sf = [sig3]`
  - `nlos` (An, Bn; the `dual_slope` keys if `LOSmax != 0`; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0,
    G2n = 1, LOSmax = 1, sig optional; Ax, Bx, Cx = 0): `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`;
    `PLn = An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² + D3n·hb + En·lg hm +
    E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; `pl = PLn`;
    if `[LOSmax] != 0`: `pl = max(pl, PL_LOS)` with the `dual_slope` loss of the same parameters (incl. the
    random hE); if `Ax`, `Bx` or `Cx` is present: `pl = max(pl, Ax·lg d3 + Bx + Cx·lg f)` (e.g. the InF-SL term of
    InF-DL); `sf = [sig]` if present
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS / d3, 1)` with the absolute Tx height;
    `pl = +Inf` if `sa <= 0`; else `pl = A·lg d3 + B + C·lg f + D·lg(asin(sa)) + PLa` with `PLa = Z(f) / sa` if
    `usePLa != 0`, else 0; `Z(f)` is the zenith attenuation by atmospheric gases (93 values from 1 to 100 GHz,
    linear interpolation, constant beyond both ends)
  - no model (`plpar` is `None`, `{}` or has no entry `model`): `pl = 0`
- New keys of the TR 38.901 v19 files: `hE_rand` (non-zero: random environment height of UMa), `LOSmax` (0: no
  maximum with the LOS term in `nlos`), `Ax`, `Bx`, `Cx` (additional term in the maximum of `nlos`)
- Random environment height (TR 38.901 Table 7.4.1-1, note 1), in `dual_slope` and the LOS term of `nlos` when
  `[hE_rand] != 0` and `rnd_he` is given, per link with `u = rnd_he(l)`, `d = d2`, raw `hUT`:
  `g = 0` for `d <= 18`, else `1.25·(d/100)³·exp(-d/150)`; `C' = 0` for `hUT < 13`, `((hUT - 13)/10)^1.5` for
  `hUT <= 23`, else 1; `C = C'·g`, `p1 = 1/(1 + C)`; `n` = size of the set {12, 15, ..., hUT - 1.5};
  `hE = 1` if `C = 0`, `n = 0`, `u < p1` or `p1` rounds to 1 (C below 1.1e-16, e.g. beyond about 7.5 km at
  `hUT = 20`), else `hE = 12 + 3·min(floor((u - p1)/(1 - p1)·n), n - 1)`.
  The UT height then gets no 1.1·hE clamp (`hm = max(hUT, 0.1)`), since `hUT - hE >= 1.5` holds for the TR set.
  Without `rnd_he` the fixed `[hE]` is used. `rnd_he` is column 10 of [[lsf_draw]], a spatially consistent
  uniform value: along a track hE switches between its discrete values where `u` crosses a threshold, and the
  loss beyond the breakpoint steps there; evaluate `rnd_he` once per track segment to avoid this
- Distances: `d3`, `d2` are the geometric distances (TR 38.901 eq. 7.4-1: for an indoor UT `d3D-out + d3D-in`).
  If `d2d_in` is given (legacy mode), a link with `d2d_in(l) > 0` uses the outdoor distance instead:
  `din = min(d2d_in, d2)`, `d3in = d3·din/d2` (`d3` if `d2 = 0`), limited to `d3`; if `d3 - d3in < 10`:
  `d3in = d3 - 10` (minimum outdoor distance 10 m), then `max(d3in, 0)`; `p = d3in/d3` (0 if `d3 = 0`);
  `d3 = d3 - d3in`, `d2 = d2·(1 - p)`. A link with `d2d_in(l) = 0` is an outdoor link and keeps the geometric
  distances (no 10 m rule). Then `d3 = max(d3, 0.1)`, `d2 = max(d2, 0.1)`. The O2I penetration loss is not added;
  add the output of [[o2i_loss]]
- Default shadow-fading std, where a model does not set it: the SF std of the LSP formula of [[lsp_generate]]
  with the geometric distance, `sf = max(0, max(0, SF_sigma + SF_delta·lg(SF_omega + f')) + SF_kappa·lg(d) +
  SF_tau·lg(h) + SF_beta·lg(atan(h/d)))` with `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`,
  `f' = max(f, LSP_fc_min)`; for scenario files without such terms this is `SF_sigma`. Passing `sf_sigma` to
  [[lsp_generate]] is then neutral. `SF_sigma`, `SF_delta`, `SF_kappa`, `SF_tau`, `SF_beta` must be finite, and
  a frequency term with `SF_delta != 0` requires a finite `SF_omega` and `SF_omega + f' > 0`
- Validation: positions, `fc`, the lengths of `d2d_in` and `rnd_he`, their values, the model name, the
  parameters of the model in the order the model reads them (a required key that is missing raises
  `Path-loss model '<model>' requires the parameter '<key>'.`, naming `A1` for "A1 or A" and `E1` for "E1 or E";
  every key that the model reads must have exactly one value, which must be finite; keys not read are ignored),
  the breakpoint factors `E`, `E1`, `E2` and `G2n` of `nlos` (> 0) and `A1 != A2` of `winner_los`, then `scenpar`
  ([[scenpar_validate]], its messages pass through), then the SF coefficients and frequency term; all errors are
  raised as `ValueError`
- The models are evaluated as written for any positive frequency, also outside their validity range

## Usage:
```
pl, sf_sigma = quadriga_lib.gbsm.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d2d_in, rnd_he )
```

## Inputs:
- **`plpar`** — Path-loss parameters; dict as returned by [[conf_file_read]]: the entry `model` (str) and one
  real scalar or 1D array with one value per further parameter, named without "PL_"; `None`, `{}` or no
  entry `model` = no model, zero loss
- **`scenpar`** — Scenario parameters (SF std terms, `LSP_fc_min`); dict as returned by [[conf_file_read]] or
  [[scenpar_validate]] (the entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have
  their default value; `None` = all defaults
- **`fc`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; one row is used for all links; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`d2d_in`** — Indoor 2D distance in m of the UT of each link (e.g. from [[o2i_loss]]), finite and >= 0;
  given = legacy mode with the outdoor distance; 0 = outdoor link; `None` or empty = geometric distance
  (TR 38.901); `(n_link,)`; default: None
- **`rnd_he`** — Uniform(0,1) value per link for the random environment height, column 10 of [[lsf_draw]], in
  [0, 1]; `None` or empty = fixed `hE`; `(n_link,)`; default: None

## Outputs:
- **`pl`** — Path loss in dB; `(n_link, n_freq)`
- **`sf_sigma`** — Shadow-fading standard deviation in dB, >= 0 for the default; `(n_link, n_freq)`

## See also:
- [[conf_file_read]] (the model and its parameters)
- [[lsf_draw]] (the random values, column 10 for `rnd_he`)
- [[lsp_generate]] (takes `sf_sigma`)
- [[o2i_loss]] (penetration loss and indoor distance)
MD!*/

py::tuple path_loss(py::handle plpar,
                    py::handle scenpar,
                    const py::array_t<double> &fc,
                    const py::array_t<double> &tx_pos,
                    const py::array_t<double> &rx_pos,
                    py::handle d2d_in,
                    py::handle rnd_he)
{
    // Read inputs; the entry 'model' of 'plpar' is the model, 'SubpathMethod' of 'scenpar' is not used
    std::string pl_model, subpath_method;
    std::vector<std::string> pl_order;
    const auto plpar_a = qd_python_dict2plpar(plpar, pl_model, pl_order, "plpar");
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");
    const auto fc_a = qd_python_numpy2arma_Col<double>(fc, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto d2d_in_a = qd_python_numpy2arma_Col<double>(d2d_in, true);
    const auto rnd_he_a = qd_python_numpy2arma_Col<double>(rnd_he, true);

    // Output allocation
    arma::mat pl, sf_sigma;
    auto pl_py = qd_python_init_output(rx_pos_a.n_rows, fc_a.n_elem, &pl);
    auto sf_sigma_py = qd_python_init_output(rx_pos_a.n_rows, fc_a.n_elem, &sf_sigma);

    // Wrap optional pointers
    const arma::vec *p_d2d_in = d2d_in_a.is_empty() ? nullptr : &d2d_in_a;
    const arma::vec *p_rnd_he = rnd_he_a.is_empty() ? nullptr : &rnd_he_a;

    // Call library function
    quadriga_lib::path_loss(pl_model, plpar_a, scenpar_a, fc_a, tx_pos_a, rx_pos_a, pl, &sf_sigma, p_d2d_in,
                            p_rnd_he);

    return py::make_tuple(pl_py, sf_sigma_py);
}

// pybind11 declaration:
// m.def("path_loss", &path_loss,
//       py::arg("plpar"),
//       py::arg("scenpar"),
//       py::arg("fc").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("d2d_in") = py::none(),
//       py::arg("rnd_he") = py::none());
