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

- Generates the 8 large-scale parameters (LSPs) of TR 38.901 clause 7.5 step 4 per link and
  frequency: delay spread DS, Ricean K-factor KF, shadow fading SF, azimuth spreads of departure and
  arrival ASD, ASA, elevation spreads of departure and arrival ESD, ESA and cross-polarization ratio
  XPR. The random values `rnd` are the normalized, spatially correlated values of [[lsf_draw]]
  (columns 1 to 8 Normal(0,1) for the 8 LSPs, column 9 Normal(0,1) for the absolute TOA, column 10
  Uniform(0,1) for the ground permittivity; column 11 is not used here)
- `scenpar` is completed with the defaults of missing fields and validated as in
  [[scenpar_validate]]; its messages pass through unchanged. An invalid value of a field that is not
  used here throws as well
- Mean and standard deviation in the log domain, per link `l` and frequency `k`, with the field
  prefix `P` of the LSP (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), the 2D distance
  `d2D`, the Tx height `hBS = tx_pos(:, 3)` and the Rx height `hUT = rx_pos(:, 3)`:
  `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `a = atan(h / d)` (elevation in rad),
  `f' = max(fc / 1e9, LSP_fc_min)` (frequency in GHz, limited from below by `LSP_fc_min` for all LSP
  terms); `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`;
  `s = max(0, P_sigma + P_delta·lg(P_omega + f'))`;
  `sigma = max(0, s + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`. The standard deviation is
  limited at 0 after all terms. SF has no mean fields (mean 0). A frequency term with the
  coefficient 0 contributes 0; an active one (`P_gamma` or `P_delta` not 0) requires
  `P_omega + f' > 0` at every frequency
- DS with the V/S term of TR 38.901 Table 7.5-6 Part-3 (InF): if `DS_VS > 0`,
  `mu += lg(DS_VS_a·DS_VS + DS_VS_b)`. The InF files set `DS_VS` for the hall of their calibration
  scenario; a caller with another hall sets the field `DS_VS` (V/S in m) of `scenpar`
- ESD with the terms of TR 38.901 Tables 7.5-7 to 7.5-12: `mu += ES_D_mu_A·d/1000 + ES_D_mu_B·hUT +
  ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)`, then the floor `mu = max(mu, ES_D_mu_min)`
- Units of mean and std: DS in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF and XPR in dB
- `sf_sigma` (e.g. from [[path_loss]]) replaces the SF standard deviation per link and frequency,
  also in `lsp_sigma`; it must be finite and >= 0 (a negative value would flip the sign of the
  shadow fading)
- Cross-correlation: `x = Q·s` with `s = rnd(l, 1:8)` and a root `Q` of the LSP cross-correlation
  matrix `R` of `scenpar` (`Q·Q^T = R`). `cholesky = true` (TR 38.901 clause 7.5 step 4): Cholesky
  factorization in the TR order [SF, KF, DS, ASD, ASA, ESD, ESA], XPR appended last; the SF value is
  passed through (`x_SF = s_SF`). `cholesky = false`: the symmetric square root of `R` (principal
  root, legacy QuaDRiGa behaviour); it does not depend on the order of the parameters. Both raise an
  error if `R` is not positive definite
- The same `x` is used for all frequencies of a link (TR 38.901 clause 7.6.5)
- Linear values: `lsp = 10^(mu + sigma·x)` for DS, ASD, ASA, ESD, ESA; `10^(0.1·(mu + sigma·x))` for
  KF, SF, XPR. `clip_spreads = true` (TR 38.901 clause 7.5 step 4): ASD and ASA are limited to
  104 deg, ESD and ESA to 52 deg; `lsp_mu` and `lsp_sigma` are not affected
- `rnd = zeros(n_link, 11)` gives the median LSPs; `lsp_mu` and `lsp_sigma` are the mean and std
  without any randomness
- Absolute TOA offset (TR 38.901 clause 7.6.9): active if `absTOA_mu > -30`, then
  `abs_toa = min(10^(absTOA_mu + absTOA_sigma·rnd(l, 9)), 2·absTOA_Lmax / c)` with
  `c = 299792458 m/s`, without the bound if `absTOA_Lmax = 0`; 0 if not active. The InF and InH NLOS
  files set `absTOA_Lmax` to the largest dimension of the hall or office of their calibration
  scenario; a caller with another hall sets the field
- Ground permittivity with `U = rnd(l, 10)` and `f = fc / 1e9` (not limited by `LSP_fc_min`): dry
  ground `3 + j·0.003·f^1.34`, medium `30.4·f^-0.47 + j·0.18·f^1.05`, wet
  `31.3·f^-0.48 + j·0.63·f^0.77`; linear interpolation from dry (`U = 0`) to medium (`U = 0.5`) to
  wet (`U = 1`). The imaginary part is >= 0; TR 38.901 eq. (7.6-40) writes the complex permittivity
  with the opposite sign of the imaginary part, so the TR reflection formulas take the complex
  conjugate `gr_eps_re - j·gr_eps_im`
- Column 10 of `rnd` must be in [0, 1] when `gr_eps_re` or `gr_eps_im` is requested, i.e. with 3 or
  more outputs; it is not checked otherwise. The error message counts the columns from 0
  ("Column 9 of 'rnd' ...")
- EoD offset in degrees, positive = up (the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to
  7.5-12), with `f'`, `d`, `h` as above: `e = EoD_offset_e + EoD_offset_e_gamma·lg f'`; if
  `EoD_offset_b > 0`: `e += 10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) +
  EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)`; if `EoD_offset_hcl > 0` and
  `hUT < EoD_offset_hcl`: `e += (atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi`
- Outputs that are not requested are not computed
- Validation order: positions, `fc`, `rnd`, `sf_sigma`, `scenpar`, positive definiteness, column 10
  of `rnd`, the frequency terms

## Usage:
```
[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = ...
    quadriga_lib.lsp_generate( scenpar, fc, tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads );
```

## Inputs:
- **`scenpar`** — Scenario parameters; struct as returned by [[conf_file_read]] or
  [[scenpar_validate]] (the field `SubpathMethod` is ignored), or any subset of its fields; missing
  fields have their default value; `[]` = all defaults; the LSP cross-correlation matrix must be
  positive definite
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Normalized random values from [[lsf_draw]], finite; columns 1 to 8 for the 8 LSPs,
  column 9 for the absolute TOA, column 10 in [0, 1] for the ground permittivity (checked only with
  3 or more outputs), column 11 not used; `[n_link, 11]`
- **`sf_sigma`** *(optional)* — SF standard deviation in [dB], finite and >= 0, e.g. from
  [[path_loss]]; replaces the value from `scenpar`; omitted or empty = the scenario value;
  `[n_link, n_freq]`
- **`cholesky`** — `true`: Cholesky root in the TR order; `false`: symmetric square root (legacy);
  default: true
- **`clip_spreads`** — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg; default: true

## Outputs:
- **`lsp`** — Large-scale parameters; slices: 1 DS in [s], 2 KF (linear), 3 SF (linear power
  ratio), 4 ASD, 5 ASA, 6 ESD, 7 ESA in [deg], 8 XPR (linear); `[n_link, n_freq, 8]`
- **`abs_toa`** — Absolute TOA offset in [s], 0 if disabled; `[n_link, 1]`
- **`gr_eps_re`** — Relative permittivity of the ground, real part; `[n_link, n_freq]`
- **`gr_eps_im`** — Relative permittivity of the ground, imaginary part, >= 0; `[n_link, n_freq]`
- **`eod_offset_deg`** — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** — Mean in the log domain (lg for DS, ASD, ASA, ESD, ESA; dB for KF, SF, XPR);
  `[n_link, n_freq, 8]`
- **`lsp_sigma`** — Standard deviation in the log domain, >= 0; `[n_link, n_freq, 8]`

## See also:
- [[lsf_draw]] (the random values `rnd`)
- [[path_loss]] (the SF standard deviation `sf_sigma`)
- [[lsf_init_phases]] (the phases of the random generators)
- [[scenpar_validate]] (the scenario parameters and the cross-correlation matrix)
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
