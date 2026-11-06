// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the source files of the geometry-based stochastic model (quadriga_gbsm.hpp). Internal header,
// not part of the public API and not installed. It holds only helpers that are used by more than one source file;
// all names start with 'qd_gbsm_'. The phase-set machinery of the spatially correlated generators is file-local in
// qd_gbsm_phases.cpp.
//
// Helpers and their users:
//   qd_gbsm_check_links   common validation of tx_pos / rx_pos     qd_gbsm_phases.cpp, path_loss.cpp, lsp_generate.cpp,
//                                                                  ssf_clusters.cpp, ssf_rays.cpp
//   qd_gbsm_check_fc      validation of the carrier frequencies    path_loss.cpp, lsp_generate.cpp, los_probability.cpp,
//                                                                  o2i_loss.cpp, ssf_rays.cpp, qd_gbsm_phases.cpp
//   qd_gbsm_n_coeff       number of sinusoids of a preset          qd_gbsm_phases.cpp, MEX and Python wrappers of
//                                                                  gbsm_init_phases
//   qd_gbsm_n_gen         number of generators of a random set     qd_gbsm_phases.cpp, MEX and Python wrappers of
//                                                                  gbsm_init_phases
//   qd_gbsm_ray_layout    ray and generator counts of a scenario   qd_gbsm_n_gen, qd_gbsm_phases.cpp, ssf_rays.cpp,
//                                                                  MEX and Python wrappers of ssf_rays
//   qd_gbsm_cds           cluster delay spread c_DS at a frequency qd_gbsm_phases.cpp, ssf_rays.cpp
//   qd_gbsm_fresnel       reflection coefficients of the ground    ssf_rays.cpp, ssf_paths.cpp
//   qd_gbsm_dist          distance of two points                   qd_gbsm_rays.cpp, ssf_paths.cpp,
//                                                                  gbsm_rays_from_paths.cpp
//   qd_gbsm_direction     angles towards the next distinct point   ssf_paths.cpp, gbsm_rays_from_paths.cpp
//   qd_gbsm_specular_point  ground reflection point on z = 0       qd_gbsm_rays.cpp, ssf_paths.cpp
//   qd_gbsm_lsp_stats     LSP mean and std per link and frequency  lsp_generate.cpp, path_loss.cpp
//   qd_gbsm_chol8         Cholesky factorization of an 8x8 matrix  lsp_generate.cpp, qd_scenpar.cpp
//
// Units: positions in [m], rows {x, y, z}; carrier frequencies in [Hz].
//
// LSP order of the mean / std values and of the cross-correlation matrix: 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD,
// 6 ESA, 7 XPR. Units of mean and std: DS in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF, XPR in dB.

#ifndef quadriga_gbsm_helpers_H
#define quadriga_gbsm_helpers_H

#include "quadriga_lib_helper_functions.hpp"

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Common validation of the link positions, in this order: 3 columns of rx_pos and tx_pos, rows of tx_pos 1 or
// n_link (when both are given), finite rx_pos and tx_pos; throws std::invalid_argument
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
static inline void qd_gbsm_check_fc(const arma::vec &center_frequency)
{
    if (center_frequency.n_elem == 0)
        throw std::invalid_argument("Input 'center_frequency' cannot be empty.");
    for (arma::uword k = 0; k < center_frequency.n_elem; ++k)
        if (!(std::isfinite(center_frequency[k]) && center_frequency[k] > 0.0))
            throw std::invalid_argument("Input 'center_frequency' must be finite and > 0.");
}

// Number of sinusoids of a preset of sos_preset; 0 for a name that is not a preset, so that the caller throws its
// own message in its order of checks
static inline arma::uword qd_gbsm_n_coeff(const std::string &preset)
{
    int type = 0;
    arma::uword n_coeff = 0;
    return qd_sos_preset_lookup(preset, type, n_coeff) ? n_coeff : 0;
}

// Counts of the ray level of a completed scenpar (output of scenpar_validate; a missing key throws
// std::out_of_range): N = NumClusters - 1 NLOS clusters, M = NumSubPaths rays per cluster, G = 1 if GR_enabled > 0
// else 0, n_ray = 1 + G + N·M rays per link (LOS, GR, then ray m of cluster n at 1 + G + n·M + m) and
// n_gen = 14·N·M + N generators of a "ray" set
static inline void qd_gbsm_ray_layout(const std::unordered_map<std::string, double> &sp, arma::uword &N,
                                      arma::uword &M, arma::uword &G, arma::uword &n_ray, arma::uword &n_gen)
{
    N = (arma::uword)sp.at("NumClusters") - 1;
    M = (arma::uword)sp.at("NumSubPaths");
    G = (sp.at("GR_enabled") > 0.0) ? 1 : 0;
    n_ray = 1 + G + N * M;
    n_gen = 14 * N * M + N;
}

// Number of generators n_gen of a random set of a completed scenpar: "lsf" 11; "cluster" 9·N for set_id -1 and N
// for set_id >= 0; "ray" 14·N·M + N for any set_id; 0 for an unknown set type, so that the caller throws its own
// message in its order of checks
static inline arma::uword qd_gbsm_n_gen(const std::unordered_map<std::string, double> &sp,
                                        const std::string &set_type, arma::sword set_id)
{
    if (set_type == "lsf")
        return 11;
    arma::uword N, M, G, n_ray, n_gen;
    qd_gbsm_ray_layout(sp, N, M, G, n_ray, n_gen);
    if (set_type == "cluster")
        return (set_id < 0) ? 9 * N : N;
    return (set_type == "ray") ? n_gen : 0;
}

// Cluster delay spread c_DS in [s] of a completed scenpar at the frequency 'fc' in [Hz] (TR 38.901 Table 7.5-6):
// 3.91 ns if PerClusterDS = 0 (no value in the TR), else
// max(PerClusterDS_min, PerClusterDS + PerClusterDS_gamma·lg(max(fc / 1e9, PerClusterDS_fc_min))) ns
static inline double qd_gbsm_cds(const std::unordered_map<std::string, double> &sp, double fc)
{
    const double ds = sp.at("PerClusterDS");
    if (ds == 0.0)
        return 3.91e-9;
    const double f_ghz = std::max(fc / 1.0e9, sp.at("PerClusterDS_fc_min"));
    return 1.0e-9 * std::max(sp.at("PerClusterDS_min"), ds + sp.at("PerClusterDS_gamma") * std::log10(f_ghz));
}

// Reflection coefficients R_par and R_per of the ground at z = 0 (TR 38.901 eqs. (7.6-38), (7.6-39)) for the
// permittivity eps = eps_re - j·eps_im (eq. (7.6-40)), Tx and Rx heights h_t, h_r >= 0 and the 2D distance d2D in
// [m]: cos(theta) = -(h_t + h_r)/d_GR' and sin²(theta) = (d2D/d_GR')² with d_GR' = max(d_GR, 1e-5),
// d_GR = sqrt(d2D² + (h_t + h_r)²); a zero denominator (grazing incidence with eps = 1) gives that coefficient 0
static inline void qd_gbsm_fresnel(double eps_re, double eps_im, double h_t, double h_r, double d2D,
                                   std::complex<double> &R_par, std::complex<double> &R_per)
{
    const double d_gr = std::max(std::sqrt(d2D * d2D + (h_t + h_r) * (h_t + h_r)), 1e-5);
    const double ct = -(h_t + h_r) / d_gr, s2 = (d2D / d_gr) * (d2D / d_gr);
    const std::complex<double> eps(eps_re, -eps_im);
    const std::complex<double> q = std::sqrt(eps - s2), a = eps * ct, d_par = a - q, d_per = ct - q;
    R_par = (d_par == 0.0) ? std::complex<double>(0.0) : (a + q) / d_par;
    R_per = (d_per == 0.0) ? std::complex<double>(0.0) : (ct + q) / d_per;
}

// Distance of the points a and b in [m], [3] each
static inline double qd_gbsm_dist(const double *a, const double *b)
{
    return qd_calc_length(a[0], a[1], a[2], b[0], b[1], b[2]);
}

// Azimuth and elevation in [rad] from O towards the first of the points P that differs from O; 0 if none differs
static inline void qd_gbsm_direction(const double *O, std::initializer_list<const double *> P, double &az, double &el)
{
    az = 0.0, el = 0.0;
    for (const double *x : P)
        if (x[0] != O[0] || x[1] != O[1] || x[2] != O[2])
        {
            const double dx = x[0] - O[0], dy = x[1] - O[1], dz = x[2] - O[2];
            az = std::atan2(dy, dx), el = std::atan2(dz, std::sqrt(dx * dx + dy * dy));
            return;
        }
}

// Specular point G on z = 0 of the ground reflection between T and R (z >= 0): G = T + h_t / (h_t + h_r)·(R' - T)
// with R' = (R_x, R_y, -R_z); both ends on the ground: the midpoint
static inline void qd_gbsm_specular_point(const double *T, const double *R, double *G)
{
    const double h = T[2] + R[2], s = (h > 0.0) ? T[2] / h : 0.5;
    G[0] = T[0] + s * (R[0] - T[0]), G[1] = T[1] + s * (R[1] - T[1]), G[2] = 0.0;
}

// LSP mean and std per link and frequency of a completed scenpar
// - Per link and frequency, with P the key prefix of the LSP (DS, KF, SF, AS_D, AS_A, ES_D, ES_A, XPR):
//     d = max(d2D, 1e-4), h = max(hBS, 1e-4), a = atan(h / d), f' = max(center_frequency / 1e9, LSP_fc_min)
//     mu    = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)
//     s     = max(0, P_sigma + P_delta·lg(P_omega + f'))
//     sigma = max(0, s + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))
//   DS: mu += lg(DS_VS_a·DS_VS + DS_VS_b) if DS_VS > 0
//   ESD: mu += ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0), then
//        mu = max(mu, ES_D_mu_min)
// - The inner limit at 0 is QuaDRiGa's; the outer one is the library's: QuaDRiGa added the distance, height and
//   elevation terms after its limit, so the std could become negative, which flips the sign of the realization and
//   of its cross-correlations. The TR 38.901 files have no _kappa, _tau or _beta terms, so the outer limit never
//   acts for them; the default sf_sigma of path_loss is always >= 0, as lsp_generate requires
// - The keys that do not exist for SF (SF_mu, SF_gamma, SF_epsilon, SF_zeta, SF_alpha) count as 0, so the SF mean
//   is 0. A frequency term with P_gamma = 0 (P_delta = 0) contributes 0 to mu (s)
// - The constructor reads the coefficients and evaluates the frequency terms at 'center_frequency' for the LSPs in
//   'lsps' (bit i = LSP i; the others are not evaluated and must not be queried). It throws std::invalid_argument for
//   such an LSP if a multiplying coefficient is not finite (P_sigma, P_gamma, P_delta, P_epsilon, P_zeta,
//   P_alpha, P_kappa, P_tau, P_beta; ES_D_mu_A to ES_D_mu_D for ESD; P_omega of an active frequency term), since
//   Inf·0 gives NaN, or if a frequency term is active (P_gamma or P_delta not 0) but P_omega + f' is not > 0, which
//   would give a NaN mean and hide the NaN of the std in the limit at 0. P_mu and ES_D_mu_min are not checked
//   (an infinite value gives no NaN)
// - 'sp' must hold all keys (output of scenpar_validate), a missing key throws std::out_of_range;
//   'center_frequency' in [Hz], finite and > 0 (not validated here)
// - set_link() takes the geometry of one link, get() returns the values of one LSP and frequency for it
class qd_gbsm_lsp_stats
{
public:
    qd_gbsm_lsp_stats(const std::unordered_map<std::string, double> &sp, const arma::vec &center_frequency,
                      unsigned lsps = 0xFFu)
        : gamma_f(8 * center_frequency.n_elem, 0.0), s_f(8 * center_frequency.n_elem, 0.0)
    {
        const char *prefix[8] = {"DS", "KF", "SF", "AS_D", "AS_A", "ES_D", "ES_A", "XPR"};
        auto key = [&](int i, const char *suffix)
        {
            auto it = sp.find(std::string(prefix[i]) + suffix);
            return (it == sp.end()) ? 0.0 : it->second;
        };

        auto finite = [&](int i, const char *suffix)
        {
            if (!std::isfinite(key(i, suffix)))
                throw std::invalid_argument("Scenario parameter '" + std::string(prefix[i]) + suffix +
                                            "' must be finite.");
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
            for (const char *suffix : {"_sigma", "_gamma", "_delta", "_epsilon", "_zeta", "_alpha", "_kappa", "_tau",
                                       "_beta"})
                finite(i, suffix);
            if (i == 5)
                for (const char *suffix : {"_mu_A", "_mu_B", "_mu_C", "_mu_D"})
                    finite(i, suffix);
            const bool active = p_gamma != 0.0 || p_delta != 0.0;
            if (active)
                finite(i, "_omega");
            for (arma::uword k = 0; k < center_frequency.n_elem; ++k)
            {
                const double arg = p_omega + std::max(center_frequency[k] / 1.0e9, fc_min);
                if (active && !(arg > 0.0))
                    throw std::invalid_argument("Scenario parameter '" + std::string(prefix[i]) +
                                                "_omega' plus the frequency in GHz must be > 0 where " +
                                                (i == 2 ? "'SF_delta' is not 0." // SF has no key SF_gamma
                                                        : "'" + std::string(prefix[i]) + "_gamma' or '" + prefix[i] +
                                                              "_delta' is not 0."));
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
