// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <cmath>
#include <omp.h>
#include <random>

// Reference: the same updates applied in chunks of 1000 (below the OpenMP threshold of xpr_update, so each chunk runs
// serially), i.e. one after the other in update-index order
static void xpr_update_chunked(arma::mat &X, const arma::mat &U, arma::vec *gain, bool normalize, bool apply_gain,
                               const arma::uvec &idx)
{
    const arma::uword n = idx.n_elem, chunk = 1000;
    arma::vec g_out(n);
    for (arma::uword s = 0; s < n; s += chunk)
    {
        arma::uword e = std::min(n, s + chunk) - 1;
        arma::uvec ii = idx.subvec(s, e);
        arma::mat uu = U.n_cols == 1 ? U : U.cols(s, e);
        arma::vec gg;
        if (apply_gain)
            gg = gain->subvec(s, e);
        quadriga_lib::xpr_update<double>(X, &uu, gain ? &gg : nullptr, false, normalize, apply_gain, &ii);
        if (gain && !apply_gain)
            g_out.subvec(s, e) = gg;
    }
    if (gain && !apply_gain)
        *gain = g_out;
}

TEST_CASE("xpr_update - duplicate ray_index entries give the single-thread result for any thread count")
{
    const int n_threads_before = omp_get_max_threads();

    // Repro: 200 000 updates, all to column 0, each a rotation by 1e-5 rad; the product is a rotation by 2 rad
    const arma::uword N = 200000;
    arma::uvec idx(N, arma::fill::zeros);
    arma::mat U(8, N, arma::fill::zeros);
    for (arma::uword i = 0; i < N; ++i)
        U(0, i) = std::cos(1e-5), U(2, i) = std::sin(1e-5), U(4, i) = -std::sin(1e-5), U(6, i) = std::cos(1e-5);

    arma::mat Xref(8, 4);
    quadriga_lib::xpr_update<double>(Xref, nullptr, nullptr, true);
    xpr_update_chunked(Xref, U, nullptr, false, false, idx);
    CHECK(std::abs(Xref(0, 0) - std::cos(2.0)) < 1e-9);
    CHECK(std::abs(Xref(2, 0) - std::sin(2.0)) < 1e-9);

    for (int nt : {1, 4, 12, 12})
    {
        omp_set_num_threads(nt);
        arma::mat X(8, 4);
        quadriga_lib::xpr_update<double>(X, nullptr, nullptr, true);
        quadriga_lib::xpr_update<double>(X, &U, nullptr, false, false, false, &idx);
        INFO("threads " << nt);
        CHECK(arma::approx_equal(X, Xref, "absdiff", 0.0));
    }

    // Random indices with duplicates and unique columns mixed, with normalization and gain output, and with a
    // broadcast update and gain application
    std::mt19937 g(12);
    std::uniform_int_distribution<arma::uword> C(0, 2999);
    std::uniform_real_distribution<double> D(-1.0, 1.0);
    const arma::uword M = 20000;
    arma::uvec ridx(M);
    for (arma::uword i = 0; i < M; ++i)
        ridx(i) = C(g);
    arma::mat RU(8, M), X0(8, 3000);
    for (arma::uword i = 0; i < RU.n_elem; ++i)
        RU(i) = D(g);
    for (arma::uword i = 0; i < X0.n_elem; ++i)
        X0(i) = D(g);
    arma::mat B = RU.col(0);
    arma::vec gin = arma::abs(RU.row(1).t()) + 0.5;

    arma::mat Xr1 = X0, Xr2 = X0;
    arma::vec gr1, gr2 = gin;
    xpr_update_chunked(Xr1, RU, &gr1, true, false, ridx);
    xpr_update_chunked(Xr2, B, &gr2, true, true, ridx);

    for (int nt : {1, 4, 12})
    {
        omp_set_num_threads(nt);
        INFO("threads " << nt);
        arma::mat X1 = X0, X2 = X0;
        arma::vec g1, g2 = gin;
        quadriga_lib::xpr_update<double>(X1, &RU, &g1, false, true, false, &ridx);
        quadriga_lib::xpr_update<double>(X2, &B, &g2, false, true, true, &ridx);
        CHECK(arma::approx_equal(X1, Xr1, "absdiff", 0.0));
        CHECK(arma::approx_equal(g1, gr1, "absdiff", 0.0));
        CHECK(arma::approx_equal(X2, Xr2, "absdiff", 0.0));
    }

    // Unique indices in scattered order (parallel loop): the same values as the serial reference
    arma::uvec uidx = arma::randperm<arma::uvec>(3000);
    arma::mat UU = RU.cols(0, 2999), Xu_ref = X0;
    xpr_update_chunked(Xu_ref, UU, nullptr, false, false, uidx);
    for (int nt : {1, 12})
    {
        omp_set_num_threads(nt);
        arma::mat Xu = X0;
        quadriga_lib::xpr_update<double>(Xu, &UU, nullptr, false, false, false, &uidx);
        INFO("threads " << nt);
        CHECK(arma::approx_equal(Xu, Xu_ref, "absdiff", 0.0));
    }

    omp_set_num_threads(n_threads_before);
}
