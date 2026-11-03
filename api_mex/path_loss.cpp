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
  carrier frequency; the TR 38.901 v19.5.0 conf files of the library reproduce Table 7.4.1-1 with
  them. The O2I penetration loss is not included (see [[o2i_loss]]). The models are evaluated as
  written for any positive frequency, also outside their validity range
- Notation: `f = fc/1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance, at
  least 0.1 m; `hBS`, `hUT` the z coordinates of Tx and Rx; `lg = log10`; `PL1(x) = A1·lg(x) + B +
  C·lg(f) + D·x`; `sf` the shadow-fading std. Fields of `plpar` in parentheses: the required ones,
  after the semicolon the optional ones (`X = v`: default)
- Models (the field `model` of `plpar`):
  - `constant` (A): `A`
  - `logdist` (A, B, C; SF): `A·lg d3 + B + C·lg f`; `sf = SF`
  - `logdist_simple` (A, B): `A·lg d3 + B`
  - `winner_los` (A1, A2 different, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb =
    max(hBS, 1.5)`, `hm = max(hUT, 1.5)`, `Gj = Bj + Cj·lg f + Dj·lg hb + Ej·lg hm`, breakpoint `bp
    = 10^((G2 - G1)/(A1 - A2))`; `Aj·lg d2 + Gj + Fj·hm` and `sf = sigj` with `j = 1` for `d2 <=
    bp`, else `j = 2`
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as
    `winner_los`; `(Aj + Ahj·lg hb)·lg d2 + Bj + Cj·lg f + Dj·lg hb + Ej·lg hm + Fj·hm` with `j = 1`
    for `f < 1.5`, 2 for `f < 2`, else 3
  - `winner_pathloss` (A, B, C, X; SF): `A·lg d3 + B + C·lg(f/5) + X`; `sf = SF`
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2): `hb =
    max(hBS, 1.1·hE, 0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`, `dBP = E·f·(hb - hE)·(hm - hE)`, `dBP3 =
    sqrt(dBP² + (hb - hm)²)`; `PL1(d3)`, `sf = sig1` for `d2 <= dBP`, else `PL1(dBP3) +
    A2·lg(d3/dBP3)`, `sf = sig2`
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0,
    sig1..sig3): the `dual_slope` rule with the breakpoints `dBP1 = min(dBP1, dBP2)` and `dBP2` of
    `E1, hE1` and `E2, hE2`, the slopes `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax,
    Bx, Cx, sig): `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb +
    D1n·lg(hb)/hb + D2n·lg(hb)/hb² + D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm +
    Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; with `LOSmax != 0` (which requires the `dual_slope`
    fields) at least the `dual_slope` loss; with `Ax`, `Bx` or `Cx` at least `Ax·lg d3 + Bx + Cx·lg
    f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for
    `sa <= 0`, else `A·lg d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by
    atmospheric gases over `sa` (1 to 100 GHz, constant beyond) if `usePLa != 0`
  - no model (`plpar` empty or without the field `model`): 0
- Where the model sets no `sf` (a missing `SF`, `sig` field or no such field): the SF std of
  [[lsp_generate]] at the geometric distance, so that passing `sf_sigma` to [[lsp_generate]] is
  neutral
- `hE_rand != 0` with `rnd_he` given: the random environment height of TR 38.901 Table 7.4.1-1 note
  1, drawn from `rnd_he`, which makes it spatially consistent at the decorrelation distance
  `SF_lambda` (the TR draws it independently per link); the UT height is then clamped at `hE + 0.1`
  instead of `1.1·hE`. Along a track the loss beyond the breakpoint steps where hE changes; evaluate
  `rnd_he` once per track segment to avoid this
- Distances: the geometric distances of TR 38.901 eq. (7.4-1). With `d2d_in` (legacy QuaDRiGa),
  links with `d2d_in > 0` use the outdoor part of the distances instead, at least 10 m
- A required field that is missing is an error, naming `A1` for "A1 or A" and `E1` for "E1 or E";
  every field that the model reads must have one finite value; the breakpoint factors `E`, `E1`,
  `E2` and `G2n` must be > 0

## Usage:
```
[ pl, sf_sigma ] = quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d2d_in, rnd_he );
```

## Inputs:
- **`plpar`** — Path-loss parameters; struct as returned by [[conf_file_read]]: the field `model`
  (string) and one numeric field per parameter, named without "PL_"; fields not read are ignored
- **`scenpar`** — Scenario parameters; the SF std terms and `LSP_fc_min` are used
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m] of the UT of each link (e.g. from
  [[o2i_loss]]), finite and >= 0; given = legacy outdoor distance; 0 = outdoor link; omitted or
  empty = geometric distance; `[n_link]`
- **`rnd_he`** *(optional)* — Uniform(0,1) value per link for the random environment height, column
  11 of [[lsf_draw]]; omitted or empty = fixed `hE`; `[n_link]`

## Outputs:
- **`pl`** — Path loss in [dB]; `[n_link, n_freq]`
- **`sf_sigma`** — Shadow-fading standard deviation in [dB]; `[n_link, n_freq]`

## See also:
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
