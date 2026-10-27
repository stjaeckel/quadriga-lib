// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# PATH_LOSS
Path loss and shadow-fading standard deviation of a set of links

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and
  carrier frequency. `plpar` is the struct of [[conf_file_read]]: its field `model` is the model
  (value of `PL_model`), the other fields are the `PL_` keys without the prefix; the TR 38.901
  v19.5.0 conf files in `data/config` reproduce Table 7.4.1-1 with them
- Notation: `f = fc / 1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance
  in m; `hBS`, `hUT` the z coordinates of Tx and Rx (no ground model); `[X]` the value of field `X`,
  or its default if the field is missing (`X = v` in the lists below);
  `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the shadow-fading std
- Models, required fields and defaults:
  - `constant` (A): `pl = A`
  - `logdist` (A, B, C; SF optional): `pl = A·lg(d3) + B + C·lg(f)`; `sf = [SF]` if present
  - `logdist_simple` (A, B): `pl = A·lg(d3) + B`
  - `winner_los` (A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2):
    `hb = max(hBS, 1.5)`, `hm = max(hUT, 1.5)` of each link;
    `G = B1 + C1·lg f + D1·lg hb + E1·lg hm`, `H = B2 + C2·lg f + D2·lg hb + E2·lg hm`,
    breakpoint `bp = 10^((H - G)/(A1 - A2))`;
    `d2 <= bp`: `pl = A1·lg d2 + G + F1·hm`, `sf = sig1`; else `pl = A2·lg d2 + H + F2·hm`,
    `sf = sig2` (continuous at `bp` for `F1 = F2`); `A1` and `A2` must differ
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as
    `winner_los`;
    `L_j = (A_j + Ah_j·lg hb)·lg d2 + B_j + C_j·lg f + D_j·lg hb + E_j·lg hm + F_j·hm`;
    `pl = L_1` for `f < 1.5`, `L_2` for `1.5 <= f < 2`, `L_3` for `f >= 2`
  - `winner_pathloss` (A, B, C, X; SF optional): `pl = A·lg d3 + B + C·lg(f/5) + X`; `sf = [SF]` if
    present
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2 optional):
    `hb = max(hBS, 1.1·hE, 0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`; `dBP = E·f·(hb - hE)·(hm - hE)`;
    `d2 <= dBP`: `pl = PL1(d3)`, `sf = [sig1]`; `d2 > dBP`: `pl = PL1(dBP3) + A2·lg(d3/dBP3)`,
    `sf = [sig2]` (each only if present), with the 3D breakpoint distance
    `dBP3 = sqrt(dBP² + (hb - hm)²)`. This is Table 7.4.1-1 for UMa, UMi, InH and InF; for RMa
    and SMa LOS the TR uses the 2D breakpoint distance in PL2, which differs by at most 0.034 dB
    over the TR ranges
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0,
    sig1..sig3 optional): `hb = max(hBS, 1.1·hE1, 1.1·hE2, 0.1)`, `hm` the same;
    `dBPj = Ej·f·(hb - hEj)·(hm - hEj)`, `dBP1 = min(dBP1, dBP2)`,
    `dBPj3 = sqrt(dBPj² + (hb - hm)²)`;
    `LBP1 = PL1(dBP13)`, `LBP2 = LBP1 + A2·lg(dBP23/dBP13)`; `d2 <= dBP1`: `PL1(d3)`, `sf = [sig1]`;
    `dBP1 < d2 <= dBP2`: `LBP1 + A2·lg(d3/dBP13)`, `sf = [sig2]`; `d2 > dBP2`:
    `LBP2 + A3·lg(d3/dBP23)`, `sf = [sig3]`
  - `nlos` (An, Bn; the `dual_slope` fields if `LOSmax != 0`; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n,
    E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, sig optional; Ax, Bx, Cx = 0): `hb = max(hBS, 0.1)`,
    `hm = max(hUT, 0.1)`; `PLn = An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb +
    D2n·lg(hb)/hb² + D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm +
    Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; `pl = PLn`; if `[LOSmax] != 0`: `pl = max(pl, PL_LOS)`
    with the `dual_slope` loss of the same parameters (incl. the random hE); if `Ax`, `Bx` or `Cx`
    is present: `pl = max(pl, Ax·lg d3 + Bx + Cx·lg f)` (e.g. the InF-SL term of InF-DL);
    `sf = [sig]` if present
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS / d3, 1)` with the
    absolute Tx height; `pl = +Inf` if `sa <= 0`; else `pl = A·lg d3 + B + C·lg f + D·lg(asin(sa)) +
    PLa` with `PLa = Z(f) / sa` if `usePLa != 0`, else 0; `Z(f)` is the zenith attenuation by
    atmospheric gases (93 values from 1 to 100 GHz, linear interpolation, constant beyond both ends)
  - no model (`plpar` empty or without the field `model`): `pl = 0`
- New fields of the TR 38.901 v19 files: `hE_rand` (non-zero: random environment height of UMa),
  `LOSmax` (0: no maximum with the LOS term in `nlos`), `Ax`, `Bx`, `Cx` (additional term in the
  maximum of `nlos`)
- Random environment height (TR 38.901 Table 7.4.1-1, note 1), in `dual_slope` and the LOS term of
  `nlos` when `[hE_rand] != 0` and `rnd_he` is given, per link with `u = rnd_he(l)`, `d = d2`, raw
  `hUT`: `g = 0` for `d <= 18`, else `1.25·(d/100)³·exp(-d/150)`; `C' = 0` for `hUT < 13`,
  `((hUT - 13)/10)^1.5` for `hUT <= 23`, else 1; `C = C'·g`, `p1 = 1/(1 + C)`; `n` = size of the
  set {12, 15, ..., hUT - 1.5}; `hE = 1` if `C = 0`, `n = 0`, `u < p1` or `p1` rounds to 1 (C
  below 1.1e-16, e.g. beyond about 7.5 km at `hUT = 20`), else
  `hE = 12 + 3·min(floor((u - p1)/(1 - p1)·n), n - 1)`. The UT height then gets no 1.1·hE clamp
  (`hm = max(hUT, 0.1)`), since `hUT - hE >= 1.5` holds for the TR set. Without `rnd_he` the fixed
  `[hE]` is used. `rnd_he` is column 11 of [[lsf_draw]], a spatially consistent uniform value:
  along a track hE switches between its discrete values where `u` crosses a threshold, and the loss
  beyond the breakpoint steps there; evaluate `rnd_he` once per track segment to avoid this
- Distances: `d3`, `d2` are the geometric distances (TR 38.901 eq. 7.4-1: for an indoor UT
  `d3D-out + d3D-in`). If `d2d_in` is given (legacy mode), a link with `d2d_in(l) > 0` uses the
  outdoor distance instead: `din = min(d2d_in, d2)`, `d3in = d3·din/d2` (`d3` if `d2 = 0`), limited
  to `d3`; if `d3 - d3in < 10`: `d3in = d3 - 10` (minimum outdoor distance 10 m), then
  `max(d3in, 0)`; `p = d3in/d3` (0 if `d3 = 0`); `d3 = d3 - d3in`, `d2 = d2·(1 - p)`. A link with
  `d2d_in(l) = 0` is an outdoor link and keeps the geometric distances (no 10 m rule). Then
  `d3 = max(d3, 0.1)`, `d2 = max(d2, 0.1)`. The O2I penetration loss is not added; add the output
  of [[o2i_loss]]
- Default shadow-fading std, where a model does not set it: the SF std of the LSP formula of
  [[lsp_generate]] with the geometric distance, `sf = max(0, max(0, SF_sigma +
  SF_delta·lg(SF_omega + f')) + SF_kappa·lg(d) + SF_tau·lg(h) + SF_beta·lg(atan(h/d)))` with
  `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `f' = max(f, LSP_fc_min)`; for scenario files without
  such terms this is `SF_sigma`. Passing `sf_sigma` to [[lsp_generate]] is then neutral. With
  `sf_sigma` requested, a frequency term with `SF_delta != 0` requires `SF_omega + f' > 0`
- Validation: positions, `fc`, the lengths of `d2d_in` and `rnd_he`, their values, the model name,
  the parameters of the model in the order the model reads them (a required field that is missing
  raises `Path-loss model '<model>' requires the parameter '<key>'.`, naming `A1` for "A1 or A" and
  `E1` for "E1 or E"; every field that the model reads must have exactly one value; fields not read
  are ignored), the breakpoint factors `E`, `E1`, `E2` (> 0) and `A1 != A2` of `winner_los`, then
  `scenpar` (as in [[scenpar_validate]], its messages pass through)
- The models are evaluated as written for any positive frequency, also outside their validity range

## Usage:
```
[ pl, sf_sigma ] = quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d2d_in, rnd_he );
```

## Inputs:
- **`plpar`** — Path-loss parameters; struct as returned by [[conf_file_read]]: the field `model`
  (string) and one numeric field per parameter, named without "PL_", one value each; `[]` or no
  field `model` = no model, zero loss
- **`scenpar`** — Scenario parameters (SF std terms, `LSP_fc_min`); struct as returned by
  [[conf_file_read]] or [[scenpar_validate]] (the field `SubpathMethod` is ignored), or any subset
  of its fields; missing fields have their default value; `[]` = all defaults
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m] of the UT of each link (e.g. from
  [[o2i_loss]]), finite and >= 0; given = legacy mode with the outdoor distance; 0 = outdoor link;
  omitted or empty = geometric distance (TR 38.901); `[n_link]`
- **`rnd_he`** *(optional)* — Uniform(0,1) value per link for the random environment height,
  column 11 of [[lsf_draw]], in [0, 1]; omitted or empty = fixed `hE`; `[n_link]`

## Outputs:
- **`pl`** — Path loss in [dB]; `[n_link, n_freq]`
- **`sf_sigma`** — Shadow-fading standard deviation in [dB], >= 0 for the default;
  `[n_link, n_freq]`

## See also:
- [[conf_file_read]] (the model and its parameters)
- [[lsf_draw]] (the random values, column 11 for `rnd_he`)
- [[lsp_generate]] (takes `sf_sigma`)
- [[o2i_loss]] (penetration loss and indoor distance)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 5 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the field 'model' of 'plpar' is the model, 'SubpathMethod' of 'scenpar' is not used
    std::string pl_model, subpath_method;
    std::vector<std::string> pl_order;
    const auto plpar = qd_mex_struct2plpar(prhs[0], pl_model, pl_order, "plpar");
    const auto scenpar = qd_mex_struct2scenpar(prhs[1], subpath_method, "scenpar");
    const arma::vec fc = qd_mex_get_Col<double>(prhs[2]);
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[3]);
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[4]);
    const arma::vec d2d_in = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5]);
    const arma::vec rnd_he = (nrhs < 7) ? arma::vec() : qd_mex_get_Col<double>(prhs[6]);

    // Output allocation
    arma::mat pl, sf_sigma;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&pl, rx_pos.n_rows, fc.n_elem);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&sf_sigma, rx_pos.n_rows, fc.n_elem);

    // Wrap optional pointers
    arma::mat *p_sf_sigma = (nlhs > 1) ? &sf_sigma : nullptr;
    const arma::vec *p_d2d_in = d2d_in.empty() ? nullptr : &d2d_in;
    const arma::vec *p_rnd_he = rnd_he.empty() ? nullptr : &rnd_he;

    // Call library function
    CALL_QD(quadriga_lib::path_loss(pl_model, plpar, scenpar, fc, tx_pos, rx_pos, pl, p_sf_sigma, p_d2d_in, p_rnd_he));
}
