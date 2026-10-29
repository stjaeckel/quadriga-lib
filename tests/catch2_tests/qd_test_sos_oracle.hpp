// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Oracles and helpers of the phase-set tests (test_gbsm_phases.cpp, test_ssf_phases.cpp), independent of the
// functions under test:
//  - qd_ref_sos: the sum of sinusoids sqrt(2/N) · sum_n cos( 2·pi · F(n,:)·x + phi(n) + off(n) ), typed here
//  - qd_unif: the Uniform map 0.5·erfc(-v/sqrt(2))
//  - qd_preset / qd_acf_at: frequencies and ACF table of sos_preset, the ACF interpolated linearly by sos_acfi
//  - qd_floor_offset: floor offsets 2·pi·k·n/N + psi_b(n), k = q mod N, b = q div N, psi_0 = 0, psi_b(n) = 2·pi·u_n
//    with u_n from the raw outputs x_n of std::mt19937_64 seeded with b: u_n = (x_n >> 11)·2^-53
//  - qd_grid, qd_corr, qd_equal: positions on a regular grid, Pearson correlation, bitwise equality with size

#ifndef QD_TEST_SOS_ORACLE_HPP
#define QD_TEST_SOS_ORACLE_HPP

#include "quadriga_sos.hpp"

#include <cmath>
#include <random>
#include <string>

static inline double qd_ref_sos(const arma::mat &F, const double *phi, double x, double y, double z,
                                const arma::vec *off = nullptr)
{
    const double pi = arma::datum::pi;
    double s = 0.0;
    for (arma::uword n = 0; n < F.n_rows; ++n)
        s += std::cos(2.0 * pi * (F(n, 0) * x + F(n, 1) * y + F(n, 2) * z) + phi[n] + (off ? (*off)(n) : 0.0));
    return std::sqrt(2.0 / (double)F.n_rows) * s;
}

static inline double qd_unif(double v) { return 0.5 * std::erfc(-v / std::sqrt(2.0)); }

// Frequencies and ACF table of a preset at decorrelation distance d
struct qd_preset_tab
{
    arma::mat F;
    arma::vec acf, dist;
};

static inline qd_preset_tab qd_preset(const std::string &preset, double d)
{
    qd_preset_tab t;
    quadriga_lib::sos_preset<double>(preset, t.F, &t.acf, &t.dist, d);
    return t;
}

static inline double qd_acf_at(const qd_preset_tab &t, double D)
{
    arma::vec v;
    quadriga_lib::sos_acfi<double>(t.acf, t.dist, arma::vec{D}, v);
    return v(0);
}

static inline arma::vec qd_floor_offset(arma::uword N, arma::uword q)
{
    const arma::uword k = q % N, b = q / N;
    arma::vec d(N);
    std::mt19937_64 gen(b);
    for (arma::uword n = 0; n < N; ++n)
    {
        d(n) = 2.0 * arma::datum::pi * (double)k * (double)n / (double)N;
        if (b > 0)
            d(n) += 2.0 * arma::datum::pi * ((double)(gen() >> 11) / 9007199254740992.0);
    }
    return d;
}

// Regular grid of nx · ny positions with the given spacing at height z, starting at (x0, y0)
static inline arma::mat qd_grid(arma::uword nx, arma::uword ny, double spacing, double z, double x0 = 0.0,
                                double y0 = 0.0)
{
    arma::mat p(nx * ny, 3);
    for (arma::uword i = 0; i < nx; ++i)
        for (arma::uword j = 0; j < ny; ++j)
            p(i * ny + j, 0) = x0 + spacing * (double)i, p(i * ny + j, 1) = y0 + spacing * (double)j, p(i * ny + j, 2) = z;
    return p;
}

static inline double qd_corr(const arma::vec &a, const arma::vec &b)
{
    double ma = arma::mean(a), mb = arma::mean(b), sab = 0.0, saa = 0.0, sbb = 0.0;
    for (arma::uword i = 0; i < a.n_elem; ++i)
        sab += (a(i) - ma) * (b(i) - mb), saa += (a(i) - ma) * (a(i) - ma), sbb += (b(i) - mb) * (b(i) - mb);
    return sab / std::sqrt(saa * sbb);
}

// Same size and bitwise equal values
static inline bool qd_equal(const arma::mat &a, const arma::mat &b)
{
    return arma::size(a) == arma::size(b) && arma::all(arma::vectorise(a) == arma::vectorise(b));
}

static inline bool qd_equal(const arma::cube &a, const arma::cube &b)
{
    return arma::size(a) == arma::size(b) && arma::all(arma::vectorise(a) == arma::vectorise(b));
}

#endif
