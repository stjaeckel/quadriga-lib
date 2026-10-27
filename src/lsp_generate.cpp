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
    // C = A·B for 8x8 matrices, column-major double[64]
    inline void qd_mul8(const double *A, const double *B, double *C)
    {
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
            {
                double s = 0.0;
                for (int k = 0; k < 8; ++k)
                    s += A[k * 8 + i] * B[j * 8 + k];
                C[j * 8 + i] = s;
            }
    }

    // Principal square root S of a symmetric positive definite 8x8 matrix A (S symmetric positive definite, S·S = A),
    // column-major double[64], by the coupled Newton-Schulz iteration on A / tr(A):
    //   Y = A / s, Z = I;  M = Z·Y, e = ||I - M||_F, T = (3·I - M) / 2, Y = Y·T, Z = T·Z  until e <= 1e-10
    //   (at most 100 iterations; the update computed with the final e is kept);  S = sqrt(s)·Y, symmetrized
    // All eigenvalues of A / tr(A) lie in (0, 1), which is the convergence condition of the iteration
    void qd_sqrtm8(const double *A, double *S)
    {
        double s = 0.0;
        for (int i = 0; i < 8; ++i)
            s += A[i * 9];

        double Y[64], Z[64], M[64], T[64], tmp[64];
        for (int i = 0; i < 64; ++i)
            Y[i] = A[i] / s, Z[i] = (i % 9 == 0) ? 1.0 : 0.0;

        for (int it = 0; it < 100; ++it)
        {
            qd_mul8(Z, Y, M);
            double e = 0.0;
            for (int i = 0; i < 64; ++i)
            {
                const double d = ((i % 9 == 0) ? 1.0 : 0.0) - M[i];
                e += d * d;
                T[i] = 0.5 * (((i % 9 == 0) ? 3.0 : 0.0) - M[i]);
            }
            e = std::sqrt(e);
            qd_mul8(Y, T, tmp);
            std::copy(tmp, tmp + 64, Y);
            qd_mul8(T, Z, tmp);
            std::copy(tmp, tmp + 64, Z);
            if (e <= 1e-10)
                break;
        }

        const double r = std::sqrt(s);
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
                S[j * 8 + i] = 0.5 * r * (Y[j * 8 + i] + Y[i * 8 + j]);
    }
}

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
- `scenpar` is completed with the defaults of missing keys and validated by [[scenpar_validate]]; its messages pass
  through unchanged. An invalid value of a key that is not used here throws as well
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
- `sf_sigma` (e.g. from the path-loss model) replaces the SF standard deviation per link and frequency, also in
  `lsp_sigma`; it must be finite and >= 0 (a negative value would flip the sign of the shadow fading)
- Cross-correlation: `x = Q·s` with `s = rnd(l, 0..7)` and a root `Q` of the LSP cross-correlation matrix `R` of
  `scenpar` (`Q·Q^T = R`). `cholesky = true` (TR 38.901 clause 7.5 step 4): Cholesky factorization in the TR order
  [SF, KF, DS, ASD, ASA, ESD, ESA], XPR appended last; the SF value is passed through (`x_SF = s_SF`).
  `cholesky = false`: the symmetric square root of `R` (principal root, legacy QuaDRiGa behaviour); it does not
  depend on the order of the parameters. Both throw if `R` is not positive definite
- The same `x` is used for all frequencies of a link (TR 38.901 clause 7.6.5)
- Linear values: `lsp = 10^(mu + sigma·x)` for DS, ASD, ASA, ESD, ESA; `10^(0.1·(mu + sigma·x))` for KF, SF, XPR.
  `clip_spreads = true` (TR 38.901 clause 7.5 step 4): ASD and ASA are limited to 104 deg, ESD and ESA to 52 deg;
  `lsp_mu` and `lsp_sigma` are not affected
- `rnd = zeros(n_link, 11)` gives the median LSPs; `lsp_mu` and `lsp_sigma` are the mean and std without any
  randomness
- Absolute TOA offset (TR 38.901 clause 7.6.9): active if `absTOA_mu > -30`, then
  `abs_toa = min(10^(absTOA_mu + absTOA_sigma·rnd(l, 8)), 2·absTOA_Lmax / c)` with `c = 299792458 m/s`, without
  the bound if `absTOA_Lmax = 0`; 0 if not active. The InF and InH NLOS files set `absTOA_Lmax` to the largest
  dimension of the hall or office of their calibration scenario; a caller with another hall sets the key
- Ground permittivity with `U = rnd(l, 9)` and `f = fc / 1e9` (not limited by `LSP_fc_min`): dry ground
  `3 + j·0.003·f^1.34`, medium `30.4·f^-0.47 + j·0.18·f^1.05`, wet `31.3·f^-0.48 + j·0.63·f^0.77`; linear
  interpolation from dry (`U = 0`) to medium (`U = 0.5`) to wet (`U = 1`). The imaginary part is >= 0;
  TR 38.901 eq. (7.6-40) writes the complex permittivity with the opposite sign of the imaginary part, so the TR
  reflection formulas take the complex conjugate `gr_eps_re - j·gr_eps_im`. Column 9 of `rnd` must be in [0, 1]
  when `gr_eps_re` or `gr_eps_im` is requested; it is not checked otherwise
- EoD offset in degrees, positive = up (the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12), with
  `f'`, `d`, `h` as above: `e = EoD_offset_e + EoD_offset_e_gamma·lg f'`; if `EoD_offset_b > 0`:
  `e += 10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) + EoD_offset_c +
  EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)`; if `EoD_offset_hcl > 0` and `hUT < EoD_offset_hcl`:
  `e += (atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi`
- Optional outputs that are not requested are not computed. `n_link = 0` gives outputs with 0 rows
- Validation order: positions, `fc`, `rnd`, `sf_sigma`, `scenpar`, positive definiteness, column 9 of `rnd`;
  all checks run before any output is written

## Declaration:
```
void quadriga_lib::lsp_generate(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::vec &fc,
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
- **`scenpar`** — Scenario parameters as returned by [[conf_file_read]] or [[scenpar_validate]]; missing keys
  have their default value; the LSP cross-correlation matrix must be positive definite
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Normalized random values from [[lsf_draw]], finite; columns 0 to 7 for the 8 LSPs, column 8 for the
  absolute TOA, column 9 in [0, 1] for the ground permittivity, column 10 not used; `[n_link, 11]`
- **`sf_sigma`** *(optional)* — SF standard deviation in [dB], finite and >= 0; replaces the value from
  `scenpar`; omitted = the scenario value; `[n_link, n_freq]`
- **`cholesky`** *(optional)* — `true`: Cholesky root in the TR order; `false`: symmetric square root (legacy)
- **`clip_spreads`** *(optional)* — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg

## Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in [s], 1 KF (linear), 2 SF (linear power ratio), 3 ASD,
  4 ASA, 5 ESD, 6 ESA in [deg], 7 XPR (linear); `[n_link, n_freq, 8]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s], 0 if disabled; `[n_link]`
- **`gr_eps_re`** *(optional)* — Relative permittivity of the ground, real part; `[n_link, n_freq]`
- **`gr_eps_im`** *(optional)* — Relative permittivity of the ground, imaginary part, >= 0; `[n_link, n_freq]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** *(optional)* — Mean in the log domain (lg for DS, ASD, ASA, ESD, ESA; dB for KF, SF, XPR);
  `[n_link, n_freq, 8]`
- **`lsp_sigma`** *(optional)* — Standard deviation in the log domain, >= 0; `[n_link, n_freq, 8]`

## See also:
- [[lsf_draw]] (the random values `rnd`)
- [[path_loss]] (the SF standard deviation `sf_sigma`)
- [[lsf_init_phases]] (the phases of the random generators)
- [[scenpar_validate]] (the scenario parameters and the cross-correlation matrix)
MD!*/

void quadriga_lib::lsp_generate(const std::unordered_map<std::string, double> &scenpar,
                                const arma::vec &fc,
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
    if (rx_pos.n_cols != 3)
        throw std::invalid_argument("Input 'rx_pos' must have 3 columns.");
    if (tx_pos.n_cols != 3)
        throw std::invalid_argument("Input 'tx_pos' must have 3 columns.");
    const arma::uword n_link = rx_pos.n_rows;
    if (tx_pos.n_rows != 1 && tx_pos.n_rows != n_link)
        throw std::invalid_argument("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'.");
    if (!rx_pos.is_finite())
        throw std::invalid_argument("Input 'rx_pos' must be finite.");
    if (!tx_pos.is_finite())
        throw std::invalid_argument("Input 'tx_pos' must be finite.");
    if (fc.n_elem == 0)
        throw std::invalid_argument("Input 'fc' cannot be empty.");
    if (!fc.is_finite() || arma::any(fc <= 0.0))
        throw std::invalid_argument("Input 'fc' must be finite and > 0.");
    const arma::uword n_freq = fc.n_elem;

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
    if (!quadriga_lib::scenpar_validate(sp, "legacy", &R, nullptr, true))
        throw std::invalid_argument("LSP cross-correlation matrix in 'scenpar' is not positive definite.");

    // Root Q of R in our LSP order: x = Q·s
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
    else
        qd_sqrtm8(R.memptr(), Q);

    if ((gr_eps_re != nullptr || gr_eps_im != nullptr) && (arma::any(rnd.col(9) < 0.0) || arma::any(rnd.col(9) > 1.0)))
        throw std::invalid_argument("Column 9 of 'rnd' must be in the range from 0 to 1.");

    // Mean and std per link, frequency and LSP
    arma::cube mu, sigma;
    qd_gbsm_lsp_mu_sigma(sp, fc, tx_pos, rx_pos, mu, sigma);
    if (sf_sigma != nullptr)
        sigma.slice(2) = *sf_sigma;

    // Large-scale parameters
    if (lsp.n_rows != n_link || lsp.n_cols != n_freq || lsp.n_slices != 8)
        lsp.set_size(n_link, n_freq, 8);
    const bool in_db[8] = {false, true, true, false, false, false, false, true};
    const double clip[8] = {0.0, 0.0, 0.0, 104.0, 104.0, 52.0, 52.0, 0.0};
    for (arma::uword l = 0; l < n_link; ++l)
    {
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
                const double v = mu(l, k, i) + sigma(l, k, i) * x[i];
                double val = in_db[i] ? std::pow(10.0, 0.1 * v) : std::pow(10.0, v);
                if (clip_spreads && clip[i] > 0.0)
                    val = std::min(val, clip[i]);
                lsp(l, k, i) = val;
            }
    }

    if (lsp_mu != nullptr)
    {
        if (lsp_mu->n_rows != n_link || lsp_mu->n_cols != n_freq || lsp_mu->n_slices != 8)
            lsp_mu->set_size(n_link, n_freq, 8);
        std::copy(mu.memptr(), mu.memptr() + mu.n_elem, lsp_mu->memptr());
    }
    if (lsp_sigma != nullptr)
    {
        if (lsp_sigma->n_rows != n_link || lsp_sigma->n_cols != n_freq || lsp_sigma->n_slices != 8)
            lsp_sigma->set_size(n_link, n_freq, 8);
        std::copy(sigma.memptr(), sigma.memptr() + sigma.n_elem, lsp_sigma->memptr());
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
    if (gr_eps_re != nullptr || gr_eps_im != nullptr)
    {
        if (gr_eps_re != nullptr && (gr_eps_re->n_rows != n_link || gr_eps_re->n_cols != n_freq))
            gr_eps_re->set_size(n_link, n_freq);
        if (gr_eps_im != nullptr && (gr_eps_im->n_rows != n_link || gr_eps_im->n_cols != n_freq))
            gr_eps_im->set_size(n_link, n_freq);
        for (arma::uword k = 0; k < n_freq; ++k)
        {
            const double f = fc(k) / 1.0e9;
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

    // EoD offset in degrees, positive = up
    if (eod_offset_deg != nullptr)
    {
        if (eod_offset_deg->n_rows != n_link || eod_offset_deg->n_cols != n_freq)
            eod_offset_deg->set_size(n_link, n_freq);
        const double e0 = sp.at("EoD_offset_e"), e_g = sp.at("EoD_offset_e_gamma"), a0 = sp.at("EoD_offset_a"),
                     a_g = sp.at("EoD_offset_a_gamma"), b = sp.at("EoD_offset_b"), c0 = sp.at("EoD_offset_c"),
                     c_g = sp.at("EoD_offset_c_gamma"), c_h = sp.at("EoD_offset_c_hUT"), hcl = sp.at("EoD_offset_hcl"),
                     fc_min = sp.at("LSP_fc_min");
        const bool tx_single = tx_pos.n_rows != n_link;
        for (arma::uword l = 0; l < n_link; ++l)
        {
            const arma::uword t = tx_single ? 0 : l;
            const double dx = rx_pos(l, 0) - tx_pos(t, 0), dy = rx_pos(l, 1) - tx_pos(t, 1);
            const double d = std::max(std::sqrt(dx * dx + dy * dy), 1e-4);
            const double h = std::max(tx_pos(t, 2), 1e-4);
            const double h_ut = rx_pos(l, 2);
            const double geo = (hcl > 0.0 && h_ut < hcl)
                                   ? (std::atan((hcl - h) / d) - std::atan((h_ut - h) / d)) * 180.0 / arma::datum::pi
                                   : 0.0;
            for (arma::uword k = 0; k < n_freq; ++k)
            {
                const double lf = std::log10(std::max(fc(k) / 1.0e9, fc_min));
                double e = e0 + e_g * lf;
                if (b > 0.0)
                    e += std::pow(10.0, (a0 + a_g * lf) * std::log10(std::max(b, d)) + c0 + c_g * lf + c_h * h_ut);
                (*eod_offset_deg)(l, k) = e + geo;
            }
        }
    }
}
