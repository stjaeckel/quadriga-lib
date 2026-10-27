// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# lsp_generate
Large-scale parameters of a set of links from normalized random values

- Generates the 8 large-scale parameters (LSPs) of TR 38.901 clause 7.5 step 4 per link and frequency: delay
  spread DS, Ricean K-factor KF, shadow fading SF, azimuth spreads of departure and arrival ASD, ASA, elevation
  spreads of departure and arrival ESD, ESA and cross-polarization ratio XPR. The random values `rnd` are the
  normalized, spatially correlated values of [[lsf_draw]] (columns 0 to 7 Normal(0,1) for the 8 LSPs, column 8
  Normal(0,1) for the absolute TOA, column 9 Uniform(0,1) for the ground permittivity; column 10 is not used here)
- `scenpar` is completed with the defaults of missing keys and validated as in [[scenpar_validate]]; its messages pass
  through unchanged. An invalid value of a key that is not used here raises as well
- Mean and standard deviation in the log domain, per link `l` and frequency `k`, with the key prefix `P` of the
  LSP (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), the 2D distance `d2D`, the Tx height
  `hBS = tx_pos.z` and the Rx height `hUT = rx_pos.z`:
  `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `a = atan(h / d)` (elevation in rad),
  `f' = max(fc / 1e9, LSP_fc_min)` (frequency in GHz, limited from below by `LSP_fc_min` for all LSP terms);
  `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`;
  `s = max(0, P_sigma + P_delta·lg(P_omega + f'))`;
  `sigma = max(0, s + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`. The standard deviation is limited at 0 after
  all terms. SF has no mean keys (mean 0). A frequency term with the coefficient 0 contributes 0
- DS with the V/S term of TR 38.901 Table 7.5-6 Part-3 (InF): if `DS_VS > 0`,
  `mu += lg(DS_VS_a·DS_VS + DS_VS_b)`. The InF files set `DS_VS` for the hall of their calibration scenario; a
  caller with another hall sets `DS_VS` (V/S in m) in its copy of `scenpar`
- ESD with the terms of TR 38.901 Tables 7.5-7 to 7.5-12: `mu += ES_D_mu_A·d/1000 + ES_D_mu_B·hUT +
  ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)`, then the floor `mu = max(mu, ES_D_mu_min)`
- Units of mean and std: DS in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF and XPR in dB
- `sf_sigma` (e.g. from [[path_loss]]) replaces the SF standard deviation per link and frequency, also in
  `lsp_sigma`; it must be finite and >= 0 (a negative value would flip the sign of the shadow fading)
- Cross-correlation: `x = Q·s` with `s = rnd(l, 0..7)` and a root `Q` of the LSP cross-correlation matrix `R` of
  `scenpar` (`Q·Q^T = R`). `cholesky = true` (TR 38.901 clause 7.5 step 4): Cholesky factorization in the TR order
  [SF, KF, DS, ASD, ASA, ESD, ESA], XPR appended last; the SF value is passed through (`x_SF = s_SF`).
  `cholesky = false`: the symmetric square root of `R` (principal root, legacy QuaDRiGa behaviour); it does not
  depend on the order of the parameters. Both raise `ValueError` if `R` is not positive definite
- The same `x` is used for all frequencies of a link (TR 38.901 clause 7.6.5)
- Linear values: `lsp = 10^(mu + sigma·x)` for DS, ASD, ASA, ESD, ESA; `10^(0.1·(mu + sigma·x))` for KF, SF, XPR.
  `clip_spreads = true` (TR 38.901 clause 7.5 step 4): ASD and ASA are limited to 104 deg, ESD and ESA to 52 deg;
  `lsp_mu` and `lsp_sigma` are not affected
- `rnd = numpy.zeros((n_link, 11))` gives the median LSPs; `lsp_mu` and `lsp_sigma` are the mean and std without any
  randomness
- Absolute TOA offset (TR 38.901 clause 7.6.9): active if `absTOA_mu > -30`, then
  `abs_toa = min(10^(absTOA_mu + absTOA_sigma·rnd(l, 8)), 2·absTOA_Lmax / c)` with `c = 299792458 m/s`, without
  the bound if `absTOA_Lmax = 0`; 0 if not active. The InF and InH NLOS files set `absTOA_Lmax` to the largest
  dimension of the hall or office of their calibration scenario; a caller with another hall sets the key
- Ground permittivity with `U = rnd(l, 9)` and `f = fc / 1e9` (not limited by `LSP_fc_min`): dry ground
  `3 + j·0.003·f^1.34`, medium `30.4·f^-0.47 + j·0.18·f^1.05`, wet `31.3·f^-0.48 + j·0.63·f^0.77`; linear
  interpolation from dry (`U = 0`) to medium (`U = 0.5`) to wet (`U = 1`). The imaginary part is >= 0;
  TR 38.901 eq. (7.6-40) writes the complex permittivity with the opposite sign of the imaginary part, so the TR
  reflection formulas take the complex conjugate `gr_eps.conj()`. The permittivity is always computed, so
  column 9 of `rnd` must always be in [0, 1]
- EoD offset in degrees, positive = up (the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12), with
  `f'`, `d`, `h` as above: `e = EoD_offset_e + EoD_offset_e_gamma·lg f'`; if `EoD_offset_b > 0`:
  `e += 10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) + EoD_offset_c +
  EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)`; if `EoD_offset_hcl > 0` and `hUT < EoD_offset_hcl`:
  `e += (atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi`
- `n_link = 0` gives outputs with 0 rows
- Validation order: positions, `fc`, `rnd`, `sf_sigma`, `scenpar`, positive definiteness, column 9 of `rnd`;
  all errors are raised as `ValueError`
- `complex = True` returns the permittivity as one complex array `gr_eps = gr_eps_re + 1j·gr_eps_im`;
  `complex = False` returns the two real arrays `gr_eps_re`, `gr_eps_im` in its place (one more output)

## Usage:
```
# complex=True (default)
lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( scenpar, fc, \
    tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads, complex=True )

# complex=False
lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( ..., \
    complex=False )
```

## Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [[conf_file_read]] or [[scenpar_validate]] (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; the LSP cross-correlation matrix must be positive definite
- **`fc`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; one row is used for all links; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`rnd`** — Normalized random values from [[lsf_draw]], finite; columns 0 to 7 for the 8 LSPs, column 8 for
  the absolute TOA, column 9 in [0, 1] for the ground permittivity, column 10 not used; `(n_link, 11)`
- **`sf_sigma`** — SF standard deviation in dB, finite and >= 0, e.g. from [[path_loss]]; replaces the value
  from `scenpar`; `None` or empty = the scenario value; `(n_link, n_freq)`; default: None
- **`cholesky`** — `True`: Cholesky root in the TR order; `False`: symmetric square root (legacy);
  default: True
- **`clip_spreads`** — `True`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg; default: True
- **`complex`** — `True`: return the ground permittivity as one complex array `gr_eps`; `False`: as two real
  arrays `gr_eps_re`, `gr_eps_im`; default: True

## Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in s, 1 KF (linear), 2 SF (linear power ratio), 3 ASD,
  4 ASA, 5 ESD, 6 ESA in deg, 7 XPR (linear); `(n_link, n_freq, 8)`
- **`abs_toa`** — Absolute TOA offset in s, 0 if disabled; `(n_link,)`
- **`gr_eps`** — Relative permittivity of the ground (`complex=True`); imaginary part >= 0; complex128;
  `(n_link, n_freq)`
- **`gr_eps_re`** — Relative permittivity of the ground, real part (`complex=False`); `(n_link, n_freq)`
- **`gr_eps_im`** — Relative permittivity of the ground, imaginary part, >= 0 (`complex=False`);
  `(n_link, n_freq)`
- **`eod_offset_deg`** — EoD offset in deg, positive = up; `(n_link, n_freq)`
- **`lsp_mu`** — Mean in the log domain (lg for DS, ASD, ASA, ESD, ESA; dB for KF, SF, XPR);
  `(n_link, n_freq, 8)`
- **`lsp_sigma`** — Standard deviation in the log domain, >= 0; `(n_link, n_freq, 8)`

## See also:
- [[lsf_draw]] (the random values `rnd`)
- [[path_loss]] (the SF standard deviation `sf_sigma`)
- [[lsf_init_phases]] (the phases of the random generators)
- [[scenpar_validate]] (the scenario parameters and the cross-correlation matrix)
MD!*/

py::tuple lsp_generate(py::handle scenpar,
                       const py::array_t<double> &fc,
                       const py::array_t<double> &tx_pos,
                       const py::array_t<double> &rx_pos,
                       const py::array_t<double> &rnd,
                       py::handle sf_sigma,
                       bool cholesky,
                       bool clip_spreads,
                       bool complex)
{
    // Read inputs; the entry 'SubpathMethod' of the scenario dict is not used
    std::string subpath_method;
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");
    const auto fc_a = qd_python_numpy2arma_Col<double>(fc, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto rnd_a = qd_python_numpy2arma_Mat<double>(rnd, true);
    const auto sf_sigma_a = qd_python_numpy2arma_Mat<double>(sf_sigma, true);

    // Output allocation
    const arma::uword n_link = rx_pos_a.n_rows, n_freq = fc_a.n_elem;
    arma::cube lsp, lsp_mu, lsp_sigma;
    arma::vec abs_toa;
    arma::mat gr_eps_re, gr_eps_im, eod_offset_deg;

    auto lsp_py = qd_python_init_output(n_link, n_freq, (arma::uword)8, &lsp);
    auto abs_toa_py = qd_python_init_output(n_link, &abs_toa);
    auto eod_offset_deg_py = qd_python_init_output(n_link, n_freq, &eod_offset_deg);
    auto lsp_mu_py = qd_python_init_output(n_link, n_freq, (arma::uword)8, &lsp_mu);
    auto lsp_sigma_py = qd_python_init_output(n_link, n_freq, (arma::uword)8, &lsp_sigma);

    // The split permittivity is written zero-copy; the complex one is merged after the call
    py::array_t<double> gr_eps_re_py, gr_eps_im_py;
    if (!complex)
    {
        gr_eps_re_py = qd_python_init_output(n_link, n_freq, &gr_eps_re);
        gr_eps_im_py = qd_python_init_output(n_link, n_freq, &gr_eps_im);
    }

    // Wrap optional pointers
    const arma::mat *p_sf_sigma = sf_sigma_a.is_empty() ? nullptr : &sf_sigma_a;

    // Call library function
    quadriga_lib::lsp_generate(scenpar_a, fc_a, tx_pos_a, rx_pos_a, rnd_a, lsp, p_sf_sigma, &abs_toa, &gr_eps_re,
                               &gr_eps_im, &eod_offset_deg, &lsp_mu, &lsp_sigma, cholesky, clip_spreads);

    if (complex)
    {
        auto gr_eps_py = qd_python_copy2numpy<double, std::complex<double>>(&gr_eps_re, &gr_eps_im);
        return py::make_tuple(lsp_py, abs_toa_py, gr_eps_py, eod_offset_deg_py, lsp_mu_py, lsp_sigma_py);
    }
    return py::make_tuple(lsp_py, abs_toa_py, gr_eps_re_py, gr_eps_im_py, eod_offset_deg_py, lsp_mu_py,
                          lsp_sigma_py);
}

// pybind11 declaration:
// m.def("lsp_generate", &lsp_generate,
//       py::arg("scenpar"),
//       py::arg("fc").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("rnd").none(false),
//       py::arg("sf_sigma") = py::none(),
//       py::arg("cholesky") = true,
//       py::arg("clip_spreads") = true,
//       py::arg("complex") = true);
