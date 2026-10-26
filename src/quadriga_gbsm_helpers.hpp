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
// distances in [m].

#ifndef quadriga_gbsm_helpers_H
#define quadriga_gbsm_helpers_H

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <armadillo>
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

#endif
