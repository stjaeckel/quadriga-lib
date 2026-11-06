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

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values of [[gbsm.draw]]
  (`"lsf"`), and the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB for KF, SF and XPR),
  with the key prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), `d = max(d2D, 1e-4)`, `h = max(hBS,
  1e-4)`, `a = atan(h/d)` and `f' = max(center_freq/1e9, LSP_fc_min)` in GHz:
  - `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`
  - `sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`
  - SF has the mean 0. DS adds `lg(DS_VS_a·DS_VS + DS_VS_b)` if `DS_VS > 0` (V/S of the InF hall, TR 38.901 Table 7.5-6
    Part-3). ESD adds `ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)` and is
    limited below by `ES_D_mu_min` (TR 38.901 Tables 7.5-7 to 7.5-12)
  - The multiplying coefficients must be finite; a frequency term with `P_gamma` or `P_delta` not 0 requires `P_omega +
    f' > 0`
- `lsp = 10^(mu + sigma·x)` for DS and the angular spreads, `10^(0.1·(mu + sigma·x))` for KF, SF and XPR; `x` are the
  columns 0 to 7 of `rnd`, cross-correlated by a root of the LSP cross-correlation matrix of `scenpar`, the same for all
  frequencies (TR 38.901 clause 7.6.5). `rnd = numpy.zeros((n_link, 11))` gives the median LSPs
- Absolute TOA offset (TR 38.901 clause 7.6.9), active if `absTOA_mu > -30`: `10^(absTOA_mu + absTOA_sigma·x)` with
  column 8 of `rnd`, bounded by `2·absTOA_Lmax/c` if `absTOA_Lmax > 0`
- The InF and InH NLOS files set `absTOA_Lmax`, and the InF files `DS_VS`, for the hall or office of their calibration
  scenario; for another hall set these keys in `scenpar`
- Ground permittivity: QuaDRiGa's fits for dry, medium and wet ground, interpolated linearly over column 9 of `rnd` (0
  dry, 0.5 medium, 1 wet) with `center_freq` not limited by `LSP_fc_min`; they are not the materials of TR 38.901 Table
  7.6.8-1
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e + EoD_offset_e_gamma·lg
  f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) + EoD_offset_c +
  EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if `EoD_offset_b > 0`, plus `(atan((EoD_offset_hcl - h)/d) -
  atan((hUT - h)/d))·180/pi` if `hUT < EoD_offset_hcl`
- `complex = True` returns the permittivity as one complex array `gr_eps = gr_eps_re + 1j·gr_eps_im`; `complex = False`
  returns the two real arrays `gr_eps_re`, `gr_eps_im` in its place (one more output)

## Usage:
```
# complex=True (default)
lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( scenpar, center_freq, \
    tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads, complex=True )

# complex=False
lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( ..., \
    complex=False )
```

## Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`rnd`** — Random values from [[gbsm.draw]] (`"lsf"`), finite; column 9 in [0, 1] (the permittivity is always
  computed); column 10 is not used; `(n_link, 11)`
- **`sf_sigma`** — SF standard deviation in dB (e.g. from [[path_loss]]), finite and >= 0; replaces the value from
  `scenpar`, also in `lsp_sigma`; `None` or empty = the scenario value; `(n_link, n_freq)`; default: None
- **`cholesky`** — `True`: Cholesky root in the order of TR 38.901 clause 7.5 step 4 (SF, KF, DS, ASD, ASA, ESD, ESA;
  XPR last); `False`: symmetric square root (legacy QuaDRiGa); default: True
- **`clip_spreads`** — `True`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg (TR 38.901 clause 7.5 step 4);
  `lsp_mu` and `lsp_sigma` are not affected; default: True
- **`complex`** — See the description; default: True

## Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in s, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA in deg, 7 XPR; `(n_link,
  n_freq, 8)`
- **`abs_toa`** — Absolute TOA offset in s, 0 if not active; `(n_link,)`
- **`gr_eps`** — Relative permittivity of the ground (`complex=True`), imaginary part >= 0; the complex value of TR
  38.901 eq. (7.6-40) is `gr_eps.conj()`; complex128; `(n_link, n_freq)`
- **`gr_eps_re`**, **`gr_eps_im`** — Real and imaginary part instead of `gr_eps` (`complex=False`); `(n_link, n_freq)`
- **`eod_offset_deg`** — EoD offset in deg, positive = up; `(n_link, n_freq)`
- **`lsp_mu`** — Mean `mu` in the log domain; required by [[ssf_rays]]; `(n_link, n_freq, 8)`
- **`lsp_sigma`** — Standard deviation `sigma` in the log domain, >= 0; required by [[ssf_rays]]; `(n_link, n_freq, 8)`

## See also:
- [[gbsm.draw]] (the random values `rnd`)
- [[path_loss]] (the SF standard deviation `sf_sigma`)
- [[ssf_clusters]] (the clusters from the LSPs)
- [[ssf_rays]] (takes `lsp`, `lsp_mu` and `lsp_sigma`)
MD!*/

py::tuple lsp_generate(py::handle scenpar,
                       const py::array_t<double> &center_freq,
                       const py::array_t<double> &tx_pos,
                       const py::array_t<double> &rx_pos,
                       const py::array_t<double> &rnd,
                       py::handle sf_sigma,
                       py::handle cholesky,
                       py::handle clip_spreads,
                       py::handle complex)
{
    // Read inputs
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const bool cholesky_b = qd_python_flag(cholesky, true, "cholesky");
    const bool clip_spreads_b = qd_python_flag(clip_spreads, true, "clip_spreads");
    const bool complex_b = qd_python_flag(complex, true, "complex");
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto rnd_a = qd_python_numpy2arma_Mat<double>(rnd, true);
    const auto sf_sigma_a = qd_python_numpy2arma_Mat<double>(sf_sigma, true);

    // Output allocation
    const arma::uword n_link = rx_pos_a.n_rows, n_freq = center_freq_a.n_elem;
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
    if (!complex_b)
    {
        gr_eps_re_py = qd_python_init_output(n_link, n_freq, &gr_eps_re);
        gr_eps_im_py = qd_python_init_output(n_link, n_freq, &gr_eps_im);
    }

    // Wrap optional pointers
    const arma::mat *p_sf_sigma = sf_sigma_a.is_empty() ? nullptr : &sf_sigma_a;

    // Call library function
    quadriga_lib::lsp_generate(scenpar_a, center_freq_a, tx_pos_a, rx_pos_a, rnd_a, lsp, p_sf_sigma, &abs_toa, &gr_eps_re,
                               &gr_eps_im, &eod_offset_deg, &lsp_mu, &lsp_sigma, cholesky_b, clip_spreads_b);

    if (complex_b)
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
//       py::arg("center_freq").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("rnd").none(false),
//       py::arg("sf_sigma") = py::none(),
//       py::arg("cholesky") = true,
//       py::arg("clip_spreads") = true,
//       py::arg("complex") = true);
