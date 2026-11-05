// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace
{
    // Principal square root S of a symmetric positive definite 8x8 matrix A (S symmetric positive definite, S·S = A),
    // column-major double[64], by the coupled Newton-Schulz iteration on A / tr(A):
    //   Y = A / s, Z = I;  M = Z·Y, e = ||I - M||_F, T = (3·I - M) / 2, Y = Y·T, Z = T·Z  until e <= 1e-10
    //   (at most 100 iterations; the update computed with the final e is kept);  S = sqrt(s)·Y, symmetrized
    // All eigenvalues of A / tr(A) lie in (0, 1), which is the convergence condition of the iteration. For a smallest
    // eigenvalue at the rounding level it may be slightly negative in floating point: the iteration then stalls and
    // diverges, so the iterate Y with the smallest e is kept and the iteration stops at a non-finite e
    // - Returns false if S is not finite
    bool qd_sqrtm8(const double *A, double *S)
    {
        // c = a·b for 8x8 matrices, column-major double[64]
        auto mul8 = [](const double *a, const double *b, double *c)
        {
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i)
                {
                    double s = 0.0;
                    for (int k = 0; k < 8; ++k)
                        s += a[k * 8 + i] * b[j * 8 + k];
                    c[j * 8 + i] = s;
                }
        };

        double s = 0.0;
        for (int i = 0; i < 8; ++i)
            s += A[i * 9];

        double Y[64], Z[64], M[64], T[64], tmp[64], Y_best[64];
        for (int i = 0; i < 64; ++i)
            Y[i] = A[i] / s, Z[i] = (i % 9 == 0) ? 1.0 : 0.0;
        std::copy(Y, Y + 64, Y_best);

        double e_best = std::numeric_limits<double>::infinity();
        for (int it = 0; it < 100; ++it)
        {
            mul8(Z, Y, M);
            double e = 0.0;
            for (int i = 0; i < 64; ++i)
            {
                const double d = ((i % 9 == 0) ? 1.0 : 0.0) - M[i];
                e += d * d;
                T[i] = 0.5 * (((i % 9 == 0) ? 3.0 : 0.0) - M[i]);
            }
            e = std::sqrt(e);
            if (!std::isfinite(e))
                break;
            if (e < e_best)
                e_best = e, std::copy(Y, Y + 64, Y_best);
            mul8(Y, T, tmp);
            std::copy(tmp, tmp + 64, Y);
            mul8(T, Z, tmp);
            std::copy(tmp, tmp + 64, Z);
            if (e <= 1e-10)
            {
                std::copy(Y, Y + 64, Y_best);
                break;
            }
        }

        const double r = std::sqrt(s);
        bool finite = true;
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
            {
                S[j * 8 + i] = 0.5 * r * (Y_best[j * 8 + i] + Y_best[i * 8 + j]);
                finite = finite && std::isfinite(S[j * 8 + i]);
            }
        return finite;
    }
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# lsp_generate
Large-scale parameters of a set of links from normalized random values

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values of
  [[gbsm_draw]] (`"lsf"`), and on request the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB for KF, SF and
  XPR), with the key prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), `d = max(d2D, 1e-4)`,
  `h = max(hBS, 1e-4)`, `a = atan(h/d)` and `f' = max(center_frequency/1e9, LSP_fc_min)` in GHz:
  - `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`
  - `sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`
  - SF has the mean 0. DS adds `lg(DS_VS_a·DS_VS + DS_VS_b)` if `DS_VS > 0` (V/S of the InF hall, TR 38.901 Table
    7.5-6 Part-3). ESD adds `ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)`
    and is limited below by `ES_D_mu_min` (TR 38.901 Tables 7.5-7 to 7.5-12)
  - The multiplying coefficients must be finite; a frequency term with `P_gamma` or `P_delta` not 0 requires
    `P_omega + f' > 0`
- `lsp = 10^(mu + sigma·x)` for DS and the angular spreads, `10^(0.1·(mu + sigma·x))` for KF, SF and XPR; `x` are
  the columns 0 to 7 of `rnd`, cross-correlated by a root of the LSP cross-correlation matrix of `scenpar`, the same
  for all frequencies (TR 38.901 clause 7.6.5). `rnd = 0` gives the median LSPs
- Absolute TOA offset (TR 38.901 clause 7.6.9), active if `absTOA_mu > -30`: `10^(absTOA_mu + absTOA_sigma·x)`
  with column 8 of `rnd`, bounded by `2·absTOA_Lmax/c` if `absTOA_Lmax > 0`
- The InF and InH NLOS files set `absTOA_Lmax`, and the InF files `DS_VS`, for the hall or office of their
  calibration scenario; for another hall set these keys in `scenpar`
- Ground permittivity: QuaDRiGa's fits for dry, medium and wet ground, interpolated linearly over column 9 of `rnd`
  (0 dry, 0.5 medium, 1 wet) with `center_frequency` not limited by `LSP_fc_min`; they are not the materials of TR
  38.901 Table 7.6.8-1
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e +
  EoD_offset_e_gamma·lg f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) +
  EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if `EoD_offset_b > 0`, plus
  `(atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi` if `hUT < EoD_offset_hcl`

## Declaration:
```
void quadriga_lib::lsp_generate(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::vec &center_frequency,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::mat &rnd,
    arma::cube &lsp,
    const arma::mat *sf_sigma = nullptr,
    arma::vec *abs_toa = nullptr,
    arma::mat *gr_eps_re = nullptr,
    arma::mat *gr_eps_im = nullptr,
    arma::mat *eod_offset_deg = nullptr,
    arma::cube *lsp_mu = nullptr,
    arma::cube *lsp_sigma = nullptr,
    bool cholesky = true,
    bool clip_spreads = true);
```

## Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`center_frequency`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Random values from [[gbsm_draw]] (`"lsf"`), finite; column 9 in [0, 1] when the ground permittivity
  is requested; column 10 is not used; `[n_link, 11]`
- **`sf_sigma`** *(optional)* — SF standard deviation in [dB] (e.g. from [[path_loss]]), finite and >= 0; replaces
  the value from `scenpar`, also in `lsp_sigma`; `[n_link, n_freq]`
- **`cholesky`** *(optional)* — `true`: Cholesky root in the order of TR 38.901 clause 7.5 step 4 (SF, KF, DS, ASD,
  ASA, ESD, ESA; XPR last); `false`: symmetric square root (legacy QuaDRiGa)
- **`clip_spreads`** *(optional)* — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg (TR 38.901 clause 7.5
  step 4); `lsp_mu` and `lsp_sigma` are not affected

## Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in [s], 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA in [deg], 7 XPR;
  `[n_link, n_freq, 8]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s], 0 if not active; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Relative permittivity of the ground, real part and imaginary
  part >= 0; the complex value of TR 38.901 eq. (7.6-40) is `gr_eps_re - j·gr_eps_im`; `[n_link, n_freq]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** *(optional)* — Mean `mu` in the log domain; `[n_link, n_freq, 8]`
- **`lsp_sigma`** *(optional)* — Standard deviation `sigma` in the log domain, >= 0; `[n_link, n_freq, 8]`

## See also:
- [[gbsm_draw]] (the random values `rnd`)
- [[path_loss]] (the SF standard deviation `sf_sigma`)
- [[ssf_clusters]] (the clusters from the LSPs)
MD!*/

void quadriga_lib::lsp_generate(const std::unordered_map<std::string, double> &scenpar,
                                const arma::vec &center_frequency,
                                const arma::mat &tx_pos,
                                const arma::mat &rx_pos,
                                const arma::mat &rnd,
                                arma::cube &lsp,
                                const arma::mat *sf_sigma,
                                arma::vec *abs_toa,
                                arma::mat *gr_eps_re,
                                arma::mat *gr_eps_im,
                                arma::mat *eod_offset_deg,
                                arma::cube *lsp_mu,
                                arma::cube *lsp_sigma,
                                bool cholesky,
                                bool clip_spreads)
{
    // Validation, before any output is touched
    qd_gbsm_check_links(&tx_pos, &rx_pos);
    qd_gbsm_check_fc(center_frequency);
    const arma::uword n_link = rx_pos.n_rows, n_freq = center_frequency.n_elem;

    if (rnd.n_rows != n_link || rnd.n_cols != 11)
        throw std::invalid_argument("Input 'rnd' must have size [n_link, 11].");
    if (!rnd.is_finite())
        throw std::invalid_argument("Input 'rnd' must be finite.");
    if (sf_sigma != nullptr)
    {
        if (sf_sigma->n_rows != n_link || sf_sigma->n_cols != n_freq)
            throw std::invalid_argument("Input 'sf_sigma' must have size [n_link, n_freq].");
        if (!sf_sigma->is_finite() || arma::any(arma::vectorise(*sf_sigma) < 0.0))
            throw std::invalid_argument("Input 'sf_sigma' must be finite and >= 0.");
    }

    std::unordered_map<std::string, double> sp = scenpar;
    arma::mat R;
    if (!quadriga_lib::scenpar_validate(sp, &R, nullptr, true))
        throw std::invalid_argument("LSP cross-correlation matrix in 'scenpar' is not positive definite.");

    // Root Q of R in our LSP order: x = Q·s
    // - Default: Cholesky in the order of TR 38.901 clause 7.5 step 4, so the results follow the TR (the SF value is
    //   passed through). Option: the symmetric square root, QuaDRiGa's rule (MATLAB sqrtm), which does not depend on
    //   the order of the parameters
    // - A nearly singular R may pass scenpar_validate but fail here in the permuted order (its pivots round
    //   differently); it throws the same message
    double Q[64];
    if (cholesky)
    {
        // TR order [SF, KF, DS, ASD, ASA, ESD, ESA, XPR]: position a holds our LSP p[a]
        const int p[8] = {2, 1, 0, 3, 4, 5, 6, 7};
        double Rp[64], L[64];
        for (int b = 0; b < 8; ++b)
            for (int a = 0; a < 8; ++a)
                Rp[b * 8 + a] = R(p[a], p[b]);
        if (!qd_gbsm_chol8(Rp, L))
            throw std::invalid_argument("LSP cross-correlation matrix in 'scenpar' is not positive definite.");
        for (int b = 0; b < 8; ++b)
            for (int a = 0; a < 8; ++a)
                Q[p[b] * 8 + p[a]] = L[b * 8 + a];
    }
    else if (!qd_sqrtm8(R.memptr(), Q))
        throw std::invalid_argument("LSP cross-correlation matrix in 'scenpar' is not positive definite.");

    if (gr_eps_re != nullptr || gr_eps_im != nullptr)
    {
        const double *u = rnd.memptr() + 9 * n_link; // column 9
        for (arma::uword l = 0; l < n_link; ++l)
            if (!(u[l] >= 0.0 && u[l] <= 1.0))
                throw std::invalid_argument("Column 9 of 'rnd' must be in the range from 0 to 1.");
    }

    // Mean and std per link, frequency and LSP; checks the coefficients and the frequency terms
    qd_gbsm_lsp_stats stats(sp, center_frequency);

    // Coefficients of the requested EoD offset and absolute TOA: an infinite value can give Inf·0 = NaN
    auto finite = [&](const char *key)
    {
        if (!std::isfinite(sp.at(key)))
            throw std::invalid_argument(std::string("Scenario parameter '") + key + "' must be finite.");
    };
    if (eod_offset_deg != nullptr)
        for (const char *key : {"EoD_offset_e", "EoD_offset_e_gamma", "EoD_offset_a", "EoD_offset_a_gamma",
                                "EoD_offset_c", "EoD_offset_c_gamma", "EoD_offset_c_hUT"})
            finite(key);
    if (abs_toa != nullptr && sp.at("absTOA_mu") > -30.0)
        finite("absTOA_sigma");

    // Outputs
    auto size3 = [&](arma::cube *c)
    {
        if (c != nullptr && (c->n_rows != n_link || c->n_cols != n_freq || c->n_slices != 8))
            c->set_size(n_link, n_freq, 8);
    };
    size3(&lsp), size3(lsp_mu), size3(lsp_sigma);
    if (eod_offset_deg != nullptr && (eod_offset_deg->n_rows != n_link || eod_offset_deg->n_cols != n_freq))
        eod_offset_deg->set_size(n_link, n_freq);

    // Large-scale parameters and EoD offset, per link
    const bool in_db[8] = {false, true, true, false, false, false, false, true};
    const double clip[8] = {0.0, 0.0, 0.0, 104.0, 104.0, 52.0, 52.0, 0.0};
    const double e0 = sp.at("EoD_offset_e"), e_g = sp.at("EoD_offset_e_gamma"), a0 = sp.at("EoD_offset_a"),
                 a_g = sp.at("EoD_offset_a_gamma"), b = sp.at("EoD_offset_b"), c0 = sp.at("EoD_offset_c"),
                 c_g = sp.at("EoD_offset_c_gamma"), c_h = sp.at("EoD_offset_c_hUT"), hcl = sp.at("EoD_offset_hcl"),
                 fc_min = sp.at("LSP_fc_min");
    const bool tx_single = tx_pos.n_rows != n_link;
    for (arma::uword l = 0; l < n_link; ++l)
    {
        const arma::uword t = tx_single ? 0 : l;
        const double dx = rx_pos(l, 0) - tx_pos(t, 0), dy = rx_pos(l, 1) - tx_pos(t, 1);
        const double d2D = std::sqrt(dx * dx + dy * dy), hBS = tx_pos(t, 2), h_ut = rx_pos(l, 2);
        stats.set_link(d2D, hBS, h_ut);

        double x[8];
        for (int i = 0; i < 8; ++i)
        {
            double v = 0.0;
            for (int j = 0; j < 8; ++j)
                v += Q[j * 8 + i] * rnd(l, j);
            x[i] = v;
        }
        for (arma::uword k = 0; k < n_freq; ++k)
            for (arma::uword i = 0; i < 8; ++i)
            {
                double mu = 0.0, sigma = 0.0;
                stats.get(i, k, mu, sigma);
                if (i == 2 && sf_sigma != nullptr)
                    sigma = (*sf_sigma)(l, k);
                const double v = mu + sigma * x[i];
                double val = in_db[i] ? std::pow(10.0, 0.1 * v) : std::pow(10.0, v);
                if (clip_spreads && clip[i] > 0.0)
                    val = std::min(val, clip[i]);
                lsp(l, k, i) = val;
                if (lsp_mu != nullptr)
                    (*lsp_mu)(l, k, i) = mu;
                if (lsp_sigma != nullptr)
                    (*lsp_sigma)(l, k, i) = sigma;
            }

        // EoD offset in degrees, positive = up
        if (eod_offset_deg != nullptr)
        {
            const double d = std::max(d2D, 1e-4), h = std::max(hBS, 1e-4);
            const double geo = (hcl > 0.0 && h_ut < hcl)
                                   ? (std::atan((hcl - h) / d) - std::atan((h_ut - h) / d)) * 180.0 / arma::datum::pi
                                   : 0.0;
            for (arma::uword k = 0; k < n_freq; ++k)
            {
                const double lf = std::log10(std::max(center_frequency(k) / 1.0e9, fc_min));
                double e = e0 + e_g * lf;
                if (b > 0.0)
                    e += std::pow(10.0, (a0 + a_g * lf) * std::log10(std::max(b, d)) + c0 + c_g * lf + c_h * h_ut);
                (*eod_offset_deg)(l, k) = e + geo;
            }
        }
    }

    // Absolute TOA offset
    if (abs_toa != nullptr)
    {
        if (abs_toa->n_elem != n_link)
            abs_toa->set_size(n_link);
        const double toa_mu = sp.at("absTOA_mu"), toa_sigma = sp.at("absTOA_sigma"), l_max = sp.at("absTOA_Lmax");
        const double bound = (l_max > 0.0) ? 2.0 * l_max / 299792458.0 : std::numeric_limits<double>::infinity();
        for (arma::uword l = 0; l < n_link; ++l)
            (*abs_toa)(l) = (toa_mu > -30.0) ? std::min(std::pow(10.0, toa_mu + toa_sigma * rnd(l, 8)), bound) : 0.0;
    }

    // Ground permittivity: dry, medium and wet ground, linear interpolation over U
    // - QuaDRiGa's fits (gen_lsf_parameters), not the materials of TR 38.901 Table 7.6.8-1: very dry ground agrees,
    //   medium and wet ground differ (at 3.5 GHz medium 16.87 + j·0.67 here, 13.23 + j·1.39 for medium dry ground of
    //   the table; wet j·1.65 vs. j·3.93). A caller who needs the table values uses eqs. (7.6-40) to (7.6-42)
    if (gr_eps_re != nullptr || gr_eps_im != nullptr)
    {
        if (gr_eps_re != nullptr && (gr_eps_re->n_rows != n_link || gr_eps_re->n_cols != n_freq))
            gr_eps_re->set_size(n_link, n_freq);
        if (gr_eps_im != nullptr && (gr_eps_im->n_rows != n_link || gr_eps_im->n_cols != n_freq))
            gr_eps_im->set_size(n_link, n_freq);
        for (arma::uword k = 0; k < n_freq; ++k)
        {
            const double f = center_frequency(k) / 1.0e9;
            const double dry_re = 3.0, dry_im = 0.003 * std::pow(f, 1.34);
            const double med_re = 30.4 * std::pow(f, -0.47), med_im = 0.18 * std::pow(f, 1.05);
            const double wet_re = 31.3 * std::pow(f, -0.48), wet_im = 0.63 * std::pow(f, 0.77);
            for (arma::uword l = 0; l < n_link; ++l)
            {
                const double U = rnd(l, 9);
                double re, im;
                if (U <= 0.5)
                {
                    const double w = 2.0 * U;
                    re = (1.0 - w) * dry_re + w * med_re, im = (1.0 - w) * dry_im + w * med_im;
                }
                else
                {
                    const double w = 2.0 * (U - 0.5);
                    re = (1.0 - w) * med_re + w * wet_re, im = (1.0 - w) * med_im + w * wet_im;
                }
                if (gr_eps_re != nullptr)
                    (*gr_eps_re)(l, k) = re;
                if (gr_eps_im != nullptr)
                    (*gr_eps_im)(l, k) = im;
            }
        }
    }
}
