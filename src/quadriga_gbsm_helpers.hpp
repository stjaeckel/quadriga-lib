// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the source files of the geometry-based stochastic model (quadriga_gbsm.hpp). Internal header,
// not part of the public API and not installed. It holds only helpers that are used by more than one source file;
// all names start with 'qd_gbsm_'.
//
// Helpers and their users:
//   qd_gbsm_gen             generator description (decorrelation distance, distribution, mirror, bias removal)
//   qd_gbsm_lsf_generators  the 11 LSF generators from a completed scenpar      lsf_init_phases.cpp, lsf_draw.cpp
//   qd_gbsm_freq            frequency and ACF tables per distinct distance      lsf_init_phases.cpp, lsf_draw.cpp
//   qd_gbsm_positions       Rx / Tx positions for the evaluation (2D / 3D)      lsf_init_phases.cpp, lsf_draw.cpp
//   qd_gbsm_lsp_mu_sigma    LSP mean and std per link and frequency             lsp_generate.cpp, path_loss.cpp
//   qd_gbsm_chol8           Cholesky factorization of a symmetric 8x8 matrix    lsp_generate.cpp, qd_scenpar.cpp
//
// LSF generator index (column of the phase cube and of the random values of lsf_draw):
//
//   g   Generator                  Distribution   Decorrelation distance   Mirror m(g)   Bias removal
//   0   DS                         Normal         DS_lambda                0             yes
//   1   KF                         Normal         KF_lambda                1             yes
//   2   SF                         Normal         SF_lambda                2             yes
//   3   ASD                        Normal         AS_D_lambda              4             yes
//   4   ASA                        Normal         AS_A_lambda              3             yes
//   5   ESD                        Normal         ES_D_lambda              6             yes
//   6   ESA                        Normal         ES_A_lambda              5             yes
//   7   XPR                        Normal         XPR_lambda               7             yes
//   8   absolute TOA               Normal         absTOA_lambda            8             yes
//   9   ground permittivity        Uniform        SC_lambda                9             no
//  10   UMa environment height hE  Uniform        SF_lambda                10            no
//
// A decorrelation distance of 0 is used as 0.1 m (as in sos_randn).
//
// Phase cube of a phase set, [n_coeff, n_gen, 2]: slice 0 = Rx side, slice 1 = Tx side. The Tx side is the mirror of
// the Rx side, slice1.col(g) = slice0.col(m(g)): at the Tx position, generator g evaluates the field of its mirror
// generator m(g) (e.g. the ASD of a link seen from the Tx is the ASA field of the Tx as a second device).
//
// Units: positions in [m], rows {x, y, z}; frequencies of the sum-of-sinusoids tables in [1/m]; decorrelation
// distances in [m]; carrier frequencies in [Hz].
//
// LSP order of the mean / std cubes and of the cross-correlation matrix: 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD,
// 6 ESA, 7 XPR. Units of mean and std: DS in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF, XPR in dB.

#ifndef quadriga_gbsm_helpers_H
#define quadriga_gbsm_helpers_H

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// One generator of a phase set: decorrelation distance in [m] (0 already replaced by 0.1 m), distribution,
// mirror partner, bias removal on/off
struct qd_gbsm_gen
{
    double lambda;      // Decorrelation distance in [m], > 0
    bool uniform;       // false: Normal(0,1), true: Uniform(0,1)
    arma::uword mirror; // Index of the mirror generator (Tx side); equal to the own index for self-mirrored ones
    bool bias;          // Remove the sample mean in the bias removal
};

// The 11 LSF generators (index table) from a completed scenpar             users: lsf_init_phases.cpp, lsf_draw.cpp
// - 'sp' must hold all keys (output of scenpar_validate); a missing key throws std::out_of_range
static inline std::vector<qd_gbsm_gen> qd_gbsm_lsf_generators(const std::unordered_map<std::string, double> &sp)
{
    const char *keys[11] = {"DS_lambda", "KF_lambda", "SF_lambda", "AS_D_lambda", "AS_A_lambda", "ES_D_lambda",
                            "ES_A_lambda", "XPR_lambda", "absTOA_lambda", "SC_lambda", "SF_lambda"};
    const arma::uword mirror[11] = {0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10};

    std::vector<qd_gbsm_gen> gen(11);
    for (arma::uword g = 0; g < 11; ++g)
    {
        const double lambda = sp.at(keys[g]);
        gen[g].lambda = (lambda > 0.0) ? lambda : 0.1;
        gen[g].uniform = g >= 9;
        gen[g].mirror = mirror[g];
        gen[g].bias = g < 9;
    }
    return gen;
}

// Frequency tables, one per distinct decorrelation distance of 'gen' (in order of first appearance):
// sos_preset(preset, freq[t], &acf[t], &dist[t], lambda_t); tab[g] = table index of generator g; throws for an
// unknown preset with "Input 'preset' is not a known preset." (checked with qd_sos_preset_lookup); returns n_coeff
//                                                                          users: lsf_init_phases.cpp, lsf_draw.cpp
// - Distances are grouped by exact equality: equal input values give identical tables
static inline arma::uword qd_gbsm_freq(const std::string &preset, const std::vector<qd_gbsm_gen> &gen,
                                       std::vector<arma::mat> &freq, std::vector<arma::vec> &acf,
                                       std::vector<arma::vec> &dist, std::vector<arma::uword> &tab)
{
    int type = 0;
    arma::uword n_coeff = 0;
    if (!qd_sos_preset_lookup(preset, type, n_coeff))
        throw std::invalid_argument("Input 'preset' is not a known preset.");

    freq.clear(), acf.clear(), dist.clear();
    tab.assign(gen.size(), 0);
    std::vector<double> lambda;
    for (size_t g = 0; g < gen.size(); ++g)
    {
        size_t t = 0;
        while (t < lambda.size() && lambda[t] != gen[g].lambda)
            ++t;
        if (t == lambda.size())
        {
            lambda.push_back(gen[g].lambda);
            freq.emplace_back(), acf.emplace_back(), dist.emplace_back();
            quadriga_lib::sos_preset<double>(preset, freq[t], &acf[t], &dist[t], gen[g].lambda);
        }
        tab[g] = (arma::uword)t;
    }
    return n_coeff;
}

// Positions for the evaluation: R = rx_pos, T = tx_pos broadcast to n_link rows; z = 0 in 2D mode
//                                                                          users: lsf_init_phases.cpp, lsf_draw.cpp
// - 'rx_pos' [n_link, 3]; 'tx_pos' [1, 3] or [n_link, 3]; R and T [n_link, 3]
static inline void qd_gbsm_positions(const arma::mat &rx_pos, const arma::mat &tx_pos, bool horizontal,
                                     arma::mat &R, arma::mat &T)
{
    const arma::uword n_link = rx_pos.n_rows;
    R = rx_pos;
    if (tx_pos.n_rows == n_link)
        T = tx_pos;
    else
    {
        T.set_size(n_link, 3);
        for (arma::uword c = 0; c < 3; ++c)
            T.col(c).fill(tx_pos(0, c));
    }
    if (horizontal)
        R.col(2).zeros(), T.col(2).zeros();
}

// LSP mean and std of all links and frequencies, [n_link, n_freq, 8] each   users: lsp_generate.cpp, path_loss.cpp
// - 'sp' must hold all keys (output of scenpar_validate); the keys that do not exist for SF (SF_mu, SF_gamma,
//   SF_epsilon, SF_zeta, SF_alpha) count as 0, so the SF mean is 0
// - 'fc' in [Hz]; 'tx_pos' [1, 3] or [n_link, 3]; 'rx_pos' [n_link, 3]; the inputs are not validated here
// - Per link and frequency, with P the key prefix of the LSP (DS, KF, SF, AS_D, AS_A, ES_D, ES_A, XPR):
//     d = max(d2D, 1e-4), h = max(hBS, 1e-4), a = atan(h / d), f' = max(fc / 1e9, LSP_fc_min)
//     mu    = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)
//     s     = max(0, P_sigma + P_delta·lg(P_omega + f'))
//     sigma = max(0, s + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))
//   DS: mu += lg(DS_VS_a·DS_VS + DS_VS_b) if DS_VS > 0
//   ESD: mu += ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0), then
//        mu = max(mu, ES_D_mu_min); hUT is the z coordinate of the Rx
// - A frequency term with the coefficient 0 contributes 0, also where lg(P_omega + f') is not finite
static inline void qd_gbsm_lsp_mu_sigma(const std::unordered_map<std::string, double> &sp, const arma::vec &fc,
                                        const arma::mat &tx_pos, const arma::mat &rx_pos,
                                        arma::cube &mu, arma::cube &sigma)
{
    const arma::uword n_link = rx_pos.n_rows, n_freq = fc.n_elem;
    if (mu.n_rows != n_link || mu.n_cols != n_freq || mu.n_slices != 8)
        mu.set_size(n_link, n_freq, 8);
    if (sigma.n_rows != n_link || sigma.n_cols != n_freq || sigma.n_slices != 8)
        sigma.set_size(n_link, n_freq, 8);

    // Coefficients per LSP: 0 mu, 1 gamma, 2 omega, 3 epsilon, 4 zeta, 5 alpha, 6 sigma, 7 delta, 8 kappa, 9 tau,
    // 10 beta
    const char *prefix[8] = {"DS", "KF", "SF", "AS_D", "AS_A", "ES_D", "ES_A", "XPR"};
    const char *suffix[11] = {"_mu", "_gamma", "_omega", "_epsilon", "_zeta", "_alpha",
                              "_sigma", "_delta", "_kappa", "_tau", "_beta"};
    double c[8][11];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 11; ++j)
        {
            auto it = sp.find(std::string(prefix[i]) + suffix[j]);
            c[i][j] = (it == sp.end()) ? 0.0 : it->second;
        }

    const double fc_min = sp.at("LSP_fc_min");
    const double ds_vs = sp.at("DS_VS");
    const double ds_term = (ds_vs > 0.0) ? std::log10(sp.at("DS_VS_a") * ds_vs + sp.at("DS_VS_b")) : 0.0;
    const double esd_A = sp.at("ES_D_mu_A"), esd_B = sp.at("ES_D_mu_B"), esd_C = sp.at("ES_D_mu_C"),
                 esd_D = sp.at("ES_D_mu_D"), esd_min = sp.at("ES_D_mu_min");

    const bool tx_single = tx_pos.n_rows != n_link;
    for (arma::uword l = 0; l < n_link; ++l)
    {
        const arma::uword t = tx_single ? 0 : l;
        const double dx = rx_pos(l, 0) - tx_pos(t, 0), dy = rx_pos(l, 1) - tx_pos(t, 1);
        const double d = std::max(std::sqrt(dx * dx + dy * dy), 1e-4);
        const double h = std::max(tx_pos(t, 2), 1e-4);
        const double h_ut = rx_pos(l, 2);
        const double lg_d = std::log10(d), lg_h = std::log10(h), lg_a = std::log10(std::atan(h / d));

        for (arma::uword k = 0; k < n_freq; ++k)
        {
            const double f = std::max(fc(k) / 1.0e9, fc_min);
            for (arma::uword i = 0; i < 8; ++i)
            {
                const double *p = c[i];
                const double lg_f = std::log10(p[2] + f);

                double m = p[0] + p[3] * lg_d + p[4] * lg_h + p[5] * lg_a;
                if (p[1] != 0.0)
                    m += p[1] * lg_f;
                if (i == 0)
                    m += ds_term;
                else if (i == 5)
                {
                    m += esd_A * d / 1000.0 + esd_B * h_ut + esd_C * std::abs(h_ut - h) +
                         esd_D * std::max(h_ut - h, 0.0);
                    m = std::max(m, esd_min);
                }

                double s = p[6];
                if (p[7] != 0.0)
                    s += p[7] * lg_f;
                s = std::max(0.0, s);
                s = std::max(0.0, s + p[8] * lg_d + p[9] * lg_h + p[10] * lg_a);

                mu(l, k, i) = m;
                sigma(l, k, i) = s;
            }
        }
    }
}

// Cholesky factorization of a symmetric 8x8 matrix, column-major double[64]: A = L·L^T, L lower triangular with
// zeros above the diagonal; returns false at the first pivot that is not > 0 (NaN included), L is then incomplete
//                                                                          users: lsp_generate.cpp, qd_scenpar.cpp
// - Only the lower triangle of A is read; the pivots are formed in ascending order
static inline bool qd_gbsm_chol8(const double *A, double *L)
{
    for (int i = 0; i < 64; ++i)
        L[i] = 0.0;
    for (int j = 0; j < 8; ++j)
    {
        double d = A[j * 8 + j];
        for (int k = 0; k < j; ++k)
            d -= L[k * 8 + j] * L[k * 8 + j];
        if (!(d > 0.0))
            return false;
        d = std::sqrt(d);
        L[j * 8 + j] = d;
        for (int i = j + 1; i < 8; ++i)
        {
            double s = A[j * 8 + i];
            for (int k = 0; k < j; ++k)
                s -= L[k * 8 + i] * L[k * 8 + j];
            L[j * 8 + i] = s / d;
        }
    }
    return true;
}

#endif
