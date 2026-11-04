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
# LSP_GENERATE
Large-scale parameters of a set of links from normalized random values

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values
  of [[lsf_draw]], and the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB
  for KF, SF and XPR), with the field prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`,
  `XPR`), `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `a = atan(h/d)` and `f' = max(fc/1e9,
  LSP_fc_min)` in GHz:
  - `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`
  - `sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) +
    P_beta·lg(a))`
  - SF has the mean 0. DS adds `lg(DS_VS_a·DS_VS + DS_VS_b)` if `DS_VS > 0` (V/S of the InF hall, TR
    38.901 Table 7.5-6 Part-3). ESD adds
    `ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)` and is
    limited below by `ES_D_mu_min` (TR 38.901 Tables 7.5-7 to 7.5-12)
  - The multiplying coefficients must be finite; a frequency term with `P_gamma` or `P_delta` not 0
    requires `P_omega + f' > 0`
- `lsp = 10^(mu + sigma·x)` for DS and the angular spreads, `10^(0.1·(mu + sigma·x))` for KF, SF and
  XPR; `x` are the columns 1 to 8 of `rnd`, cross-correlated by a root of the LSP cross-correlation
  matrix of `scenpar`, the same for all frequencies (TR 38.901 clause 7.6.5). `rnd = zeros(n_link,
  11)` gives the median LSPs
- Absolute TOA offset (TR 38.901 clause 7.6.9), active if `absTOA_mu > -30`: `10^(absTOA_mu +
  absTOA_sigma·x)` with column 9 of `rnd`, bounded by `2·absTOA_Lmax/c` if `absTOA_Lmax > 0`
- The InF and InH NLOS files set `absTOA_Lmax`, and the InF files `DS_VS`, for the hall or office of
  their calibration scenario; for another hall set these fields in `scenpar`
- Ground permittivity: QuaDRiGa's fits for dry, medium and wet ground, interpolated linearly over
  column 10 of `rnd` (0 dry, 0.5 medium, 1 wet) with `fc` not limited by `LSP_fc_min`; they are not
  the materials of TR 38.901 Table 7.6.8-1. The imaginary part is >= 0: the complex value of TR
  38.901 eq. (7.6-40) is `gr_eps_re - j·gr_eps_im`
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e +
  EoD_offset_e_gamma·lg f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg
  f')·lg(max(EoD_offset_b, d)) + EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if
  `EoD_offset_b > 0`, plus `(atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi` if `hUT <
  EoD_offset_hcl`
- Outputs that are not requested are not computed

## Usage:
```
[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = ...
    quadriga_lib.lsp_generate( scenpar, fc, tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads );
```

## Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Random values from [[lsf_draw]], finite; column 10 in [0, 1] when 3 or more outputs
  are requested (the error message counts the columns from 0); column 11 is not used; `[n_link, 11]`
- **`sf_sigma`** *(optional)* — SF standard deviation in [dB] (e.g. from [[path_loss]]), finite
  and >= 0; replaces the value from `scenpar`, also in `lsp_sigma`; omitted or empty = the scenario
  value; `[n_link, n_freq]`
- **`cholesky`** — `true`: Cholesky root in the order of TR 38.901 clause 7.5 step 4 (SF, KF, DS,
  ASD, ASA, ESD, ESA; XPR last); `false`: symmetric square root (legacy QuaDRiGa); default: true
- **`clip_spreads`** — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg (TR 38.901 clause
  7.5 step 4); `lsp_mu` and `lsp_sigma` are not affected; default: true

## Outputs:
- **`lsp`** — Large-scale parameters; slices: 1 DS in [s], 2 KF, 3 SF, 4 ASD, 5 ASA, 6 ESD, 7 ESA in
  [deg], 8 XPR; `[n_link, n_freq, 8]`
- **`abs_toa`** — Absolute TOA offset in [s], 0 if not active; `[n_link, 1]`
- **`gr_eps_re`**, **`gr_eps_im`** — Relative permittivity of the ground, real part and imaginary
  part >= 0; `[n_link, n_freq]`
- **`eod_offset_deg`** — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** — Mean `mu` in the log domain; `[n_link, n_freq, 8]`
- **`lsp_sigma`** — Standard deviation `sigma` in the log domain, >= 0; `[n_link, n_freq, 8]`

## See also:
- [[lsf_draw]] (the random values `rnd`)
- [[path_loss]] (the SF standard deviation `sf_sigma`)
- [[ssf_clusters]] (the clusters from the LSPs)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 5 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the field 'SubpathMethod' of the scenario struct is not used
    std::string subpath_method;
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], subpath_method, "scenpar");
    const arma::vec fc = qd_mex_get_Col<double>(prhs[1]);
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[2]);
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[3]);
    const arma::mat rnd = qd_mex_get_Mat<double>(prhs[4]);
    const arma::mat sf_sigma = (nrhs < 6) ? arma::mat() : qd_mex_get_Mat<double>(prhs[5]);
    const bool cholesky = (nrhs < 7) ? true : qd_mex_get_scalar<bool>(prhs[6], "cholesky", true);
    const bool clip_spreads = (nrhs < 8) ? true : qd_mex_get_scalar<bool>(prhs[7], "clip_spreads", true);

    // Output allocation: the requested outputs only, the others are passed to C++ as nullptr
    const arma::uword n_link = rx_pos.n_rows, n_freq = fc.n_elem;
    arma::cube lsp, lsp_mu, lsp_sigma;
    arma::vec abs_toa;
    arma::mat gr_eps_re, gr_eps_im, eod_offset_deg;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&lsp, n_link, n_freq, 8);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&abs_toa, n_link);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&gr_eps_re, n_link, n_freq);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&gr_eps_im, n_link, n_freq);
    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&eod_offset_deg, n_link, n_freq);
    if (nlhs > 5)
        plhs[5] = qd_mex_init_output(&lsp_mu, n_link, n_freq, 8);
    if (nlhs > 6)
        plhs[6] = qd_mex_init_output(&lsp_sigma, n_link, n_freq, 8);

    // Wrap optional pointers
    const arma::mat *p_sf_sigma = sf_sigma.empty() ? nullptr : &sf_sigma;
    arma::vec *p_abs_toa = (nlhs > 1) ? &abs_toa : nullptr;
    arma::mat *p_gr_eps_re = (nlhs > 2) ? &gr_eps_re : nullptr;
    arma::mat *p_gr_eps_im = (nlhs > 3) ? &gr_eps_im : nullptr;
    arma::mat *p_eod_offset_deg = (nlhs > 4) ? &eod_offset_deg : nullptr;
    arma::cube *p_lsp_mu = (nlhs > 5) ? &lsp_mu : nullptr;
    arma::cube *p_lsp_sigma = (nlhs > 6) ? &lsp_sigma : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::lsp_generate(scenpar, fc, tx_pos, rx_pos, rnd, lsp, p_sf_sigma, p_abs_toa, p_gr_eps_re,
                                       p_gr_eps_im, p_eod_offset_deg, p_lsp_mu, p_lsp_sigma, cholesky, clip_spreads));
}
