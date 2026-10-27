// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the source files of the geometry-based stochastic model (quadriga_gbsm.hpp). Internal header,
// not part of the public API and not installed. It holds only helpers that are used by more than one source file;
// all names start with 'qd_gbsm_'. The phase-set machinery of the spatially correlated generators is file-local in
// qd_gbsm_phases.cpp.
//
// Helpers and their users:
//   qd_gbsm_check_links   common validation of tx_pos / rx_pos     qd_gbsm_phases.cpp, path_loss.cpp, lsp_generate.cpp
//   qd_gbsm_check_fc      validation of the carrier frequencies    path_loss.cpp, lsp_generate.cpp, los_probability.cpp,
//                                                                  o2i_loss.cpp
//   qd_gbsm_lsp_stats     LSP mean and std per link and frequency  lsp_generate.cpp, path_loss.cpp
//   qd_gbsm_chol8         Cholesky factorization of an 8x8 matrix  lsp_generate.cpp, qd_scenpar.cpp
//
// Units: positions in [m], rows {x, y, z}; carrier frequencies in [Hz].
//
// LSP order of the mean / std values and of the cross-correlation matrix: 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD,
// 6 ESA, 7 XPR. Units of mean and std: DS in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF, XPR in dB.

#ifndef quadriga_gbsm_helpers_H
#define quadriga_gbsm_helpers_H

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Common validation of the link positions, in this order: 3 columns of rx_pos and tx_pos, rows of tx_pos 1 or
// n_link (when both are given), finite rx_pos and tx_pos; throws std::invalid_argument
//                                                    users: qd_gbsm_phases.cpp, path_loss.cpp, lsp_generate.cpp
// - nullptr skips the checks of that argument
static inline void qd_gbsm_check_links(const arma::mat *tx_pos, const arma::mat *rx_pos)
{
    if (rx_pos != nullptr && rx_pos->n_cols != 3)
        throw std::invalid_argument("Input 'rx_pos' must have 3 columns.");
    if (tx_pos != nullptr && tx_pos->n_cols != 3)
        throw std::invalid_argument("Input 'tx_pos' must have 3 columns.");
    if (rx_pos != nullptr && tx_pos != nullptr && tx_pos->n_rows != 1 && tx_pos->n_rows != rx_pos->n_rows)
        throw std::invalid_argument("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'.");
    if (rx_pos != nullptr && !rx_pos->is_finite())
        throw std::invalid_argument("Input 'rx_pos' must be finite.");
    if (tx_pos != nullptr && !tx_pos->is_finite())
        throw std::invalid_argument("Input 'tx_pos' must be finite.");
}

// Validation of the carrier frequencies in [Hz]: not empty, finite and > 0; throws std::invalid_argument
//                                       users: path_loss.cpp, lsp_generate.cpp, los_probability.cpp, o2i_loss.cpp
static inline void qd_gbsm_check_fc(const arma::vec &fc)
{
    if (fc.n_elem == 0)
        throw std::invalid_argument("Input 'fc' cannot be empty.");
    for (arma::uword k = 0; k < fc.n_elem; ++k)
        if (!(std::isfinite(fc[k]) && fc[k] > 0.0))
            throw std::invalid_argument("Input 'fc' must be finite and > 0.");
}

// LSP mean and std per link and frequency of a completed scenpar             users: lsp_generate.cpp, path_loss.cpp
// - Per link and frequency, with P the key prefix of the LSP (DS, KF, SF, AS_D, AS_A, ES_D, ES_A, XPR):
//     d = max(d2D, 1e-4), h = max(hBS, 1e-4), a = atan(h / d), f' = max(fc / 1e9, LSP_fc_min)
//     mu    = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)
//     s     = max(0, P_sigma + P_delta·lg(P_omega + f'))
//     sigma = max(0, s + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))
//   DS: mu += lg(DS_VS_a·DS_VS + DS_VS_b) if DS_VS > 0
//   ESD: mu += ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0), then
//        mu = max(mu, ES_D_mu_min)
// - The keys that do not exist for SF (SF_mu, SF_gamma, SF_epsilon, SF_zeta, SF_alpha) count as 0, so the SF mean
//   is 0. A frequency term with P_gamma = 0 (P_delta = 0) contributes 0 to mu (s)
// - The constructor reads the coefficients and evaluates the frequency terms at 'fc' for the LSPs in 'lsps'
//   (bit i = LSP i; the others are not evaluated and must not be queried). It throws std::invalid_argument if a
//   frequency term of such an LSP is active (P_gamma or P_delta not 0) but P_omega + f' is not > 0, which would
//   give a NaN mean and hide the NaN of the std in the limit at 0
// - 'sp' must hold all keys (output of scenpar_validate), a missing key throws std::out_of_range; 'fc' in [Hz],
//   finite and > 0 (not validated here)
// - set_link() takes the geometry of one link, get() returns the values of one LSP and frequency for it
class qd_gbsm_lsp_stats
{
public:
    qd_gbsm_lsp_stats(const std::unordered_map<std::string, double> &sp, const arma::vec &fc, unsigned lsps = 0xFFu)
        : gamma_f(8 * fc.n_elem, 0.0), s_f(8 * fc.n_elem, 0.0)
    {
        const char *prefix[8] = {"DS", "KF", "SF", "AS_D", "AS_A", "ES_D", "ES_A", "XPR"};
        auto key = [&](int i, const char *suffix)
        {
            auto it = sp.find(std::string(prefix[i]) + suffix);
            return (it == sp.end()) ? 0.0 : it->second;
        };

        const double fc_min = sp.at("LSP_fc_min");
        for (int i = 0; i < 8; ++i)
        {
            const double p_gamma = key(i, "_gamma"), p_omega = key(i, "_omega"), p_sigma = key(i, "_sigma"),
                         p_delta = key(i, "_delta");
            c[i][0] = key(i, "_mu"), c[i][1] = key(i, "_epsilon"), c[i][2] = key(i, "_zeta");
            c[i][3] = key(i, "_alpha"), c[i][4] = key(i, "_kappa"), c[i][5] = key(i, "_tau");
            c[i][6] = key(i, "_beta");
            if (((lsps >> i) & 1u) == 0u)
                continue;
            for (arma::uword k = 0; k < fc.n_elem; ++k)
            {
                const double arg = p_omega + std::max(fc[k] / 1.0e9, fc_min);
                if ((p_gamma != 0.0 || p_delta != 0.0) && !(arg > 0.0))
                    throw std::invalid_argument("Scenario parameter '" + std::string(prefix[i]) +
                                                "_omega' plus the frequency in GHz must be > 0 where '" + prefix[i] +
                                                "_gamma' or '" + prefix[i] + "_delta' is not 0.");
                double s = p_sigma;
                if (p_gamma != 0.0)
                    gamma_f[k * 8 + i] = p_gamma * std::log10(arg);
                if (p_delta != 0.0)
                    s += p_delta * std::log10(arg);
                s_f[k * 8 + i] = std::max(0.0, s);
            }
        }

        const double ds_vs = sp.at("DS_VS");
        ds_term = (ds_vs > 0.0) ? std::log10(sp.at("DS_VS_a") * ds_vs + sp.at("DS_VS_b")) : 0.0;
        esd[0] = sp.at("ES_D_mu_A"), esd[1] = sp.at("ES_D_mu_B"), esd[2] = sp.at("ES_D_mu_C");
        esd[3] = sp.at("ES_D_mu_D"), esd[4] = sp.at("ES_D_mu_min");
    }

    // Geometry of one link: 2D distance, Tx height hBS and Rx height hUT in [m]
    void set_link(double d2D, double hBS, double hUT)
    {
        d = std::max(d2D, 1e-4), h = std::max(hBS, 1e-4), h_ut = hUT;
        lg_d = std::log10(d), lg_h = std::log10(h), lg_a = std::log10(std::atan(h / d));
    }

    // Mean and std of LSP i at frequency index k for the link of the last set_link call
    void get(arma::uword i, arma::uword k, double &mu, double &sigma) const
    {
        const double *p = c[i];
        mu = p[0] + p[1] * lg_d + p[2] * lg_h + p[3] * lg_a;
        mu += gamma_f[k * 8 + i];
        if (i == 0)
            mu += ds_term;
        else if (i == 5)
        {
            mu += esd[0] * d / 1000.0 + esd[1] * h_ut + esd[2] * std::abs(h_ut - h) + esd[3] * std::max(h_ut - h, 0.0);
            mu = std::max(mu, esd[4]);
        }
        sigma = std::max(0.0, s_f[k * 8 + i] + p[4] * lg_d + p[5] * lg_h + p[6] * lg_a);
    }

private:
    std::vector<double> gamma_f, s_f; // P_gamma·lg(P_omega + f') and s per frequency, [n_freq · 8], index k · 8 + i
    double c[8][7];                   // P_mu, P_epsilon, P_zeta, P_alpha, P_kappa, P_tau, P_beta
    double ds_term = 0.0, esd[5];     // DS V/S term; ES_D_mu_A, _B, _C, _D, _min
    double d = 1.0, h = 1.0, h_ut = 0.0, lg_d = 0.0, lg_h = 0.0, lg_a = 0.0;
};

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
