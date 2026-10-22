// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_math.hpp"

#include <cmath>
#include <random>

// Reference: the linear index scan (first grid value > query) with the same weights and the same 4-term sum as interp_2D
template <typename dtype>
static void ref_indices(const arma::Col<dtype> &s_in, const arma::Col<dtype> &s_out, arma::uvec &ip_v, arma::uvec &in_v,
                        arma::Col<dtype> &wp_v, arma::Col<dtype> &wn_v)
{
    const arma::uword n = s_in.n_elem, m = s_out.n_elem;
    arma::uvec ind = s_in.is_sorted() ? arma::regspace<arma::uvec>(0, n - 1) : arma::sort_index(s_in);
    arma::Col<dtype> grid = s_in.elem(ind), diff(n);
    diff(0) = dtype(1.0);
    for (arma::uword i = 1; i < n; ++i)
        diff(i) = dtype(1.0) / (grid(i) - grid(i - 1));
    ip_v.set_size(m), in_v.set_size(m), wp_v.set_size(m), wn_v.set_size(m);
    for (arma::uword i = 0; i < m; ++i)
    {
        arma::uword ip = 0, in = 0;
        dtype val = s_out(i), wp = dtype(1.0), wn = dtype(0.0);
        while (ip < n && grid(ip) <= val)
            ++ip;
        if (ip == n)
            in = --ip;
        else if (ip != 0)
        {
            in = ip--;
            wp = (grid(in) - val) * diff(in);
            wp = (wp > dtype(1.0)) ? dtype(1.0) : wp;
            wp = (wp < dtype(0.0)) ? dtype(0.0) : wp;
            wn = dtype(1.0) - wp;
        }
        ip_v(i) = ind(ip), in_v(i) = ind(in), wp_v(i) = wp, wn_v(i) = wn;
    }
}

template <typename dtype>
static arma::Mat<dtype> ref_interp_2D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                                      const arma::Col<dtype> &xo, const arma::Col<dtype> &yo)
{
    arma::uvec ixp, ixn, iyp, iyn;
    arma::Col<dtype> xp, xn, yp, yn;
    ref_indices(xi, xo, ixp, ixn, xp, xn);
    ref_indices(yi, yo, iyp, iyn, yp, yn);
    const arma::uword ny = input.n_rows;
    const dtype *p = input.memptr();
    arma::Mat<dtype> out(yo.n_elem, xo.n_elem);
    for (arma::uword ix = 0; ix < xo.n_elem; ++ix)
        for (arma::uword iy = 0; iy < yo.n_elem; ++iy)
        {
            dtype wA = xp(ix) * yp(iy), wB = xn(ix) * yp(iy), wC = xp(ix) * yn(iy), wD = xn(ix) * yn(iy);
            out(iy, ix) = wA * p[ixp(ix) * ny + iyp(iy)] + wB * p[ixn(ix) * ny + iyp(iy)] +
                          wC * p[ixp(ix) * ny + iyn(iy)] + wD * p[ixn(ix) * ny + iyn(iy)];
        }
    return out;
}

TEST_CASE("interp - binary index search gives the results of the linear scan")
{
    std::mt19937 g(21);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (arma::uword n : {1ULL, 2ULL, 3ULL, 7ULL, 16ULL, 17ULL, 33ULL, 100ULL, 1000ULL})
        for (int variant = 0; variant < 3; ++variant)
        {
            // variant 0: sorted grid; 1: unsorted; 2: sorted with duplicate values
            arma::vec xi(n);
            for (auto &v : xi)
                v = U(g);
            if (variant != 1)
                xi = arma::sort(xi);
            if (variant == 2 && n > 3)
                xi(1) = xi(0), xi(n - 1) = xi(n - 2), xi(n / 2) = xi(n / 2 + 1);
            arma::vec yi = arma::sort(arma::vec(arma::randu(5)));
            arma::vec d1 = arma::randn(n);
            arma::mat d2 = arma::randn(5, n);

            // Queries: random inside and outside the grid, and every grid value
            arma::vec xo(300 + n);
            for (arma::uword i = 0; i < 300; ++i)
                xo(i) = 1.3 * U(g);
            xo.tail(n) = xi;
            arma::vec yo = {-0.5, 0.0, 0.3, 0.99, 2.0};

            INFO("n = " << n << ", variant " << variant);
            arma::vec r1 = quadriga_lib::interp_1D<double>(d1, xi, xo);
            arma::mat ref1 = ref_interp_2D<double>(arma::mat(d1.t()), xi, arma::vec(1), xo, arma::vec(1));
            CHECK(arma::approx_equal(r1, arma::vec(ref1.row(0).t()), "absdiff", 0.0));

            arma::mat r2 = quadriga_lib::interp_2D<double>(d2, xi, yi, xo, yo);
            arma::mat ref2 = ref_interp_2D<double>(d2, xi, yi, xo, yo);
            CHECK(arma::approx_equal(r2, ref2, "absdiff", 0.0));

            arma::fvec xif = arma::conv_to<arma::fvec>::from(xi), xof = arma::conv_to<arma::fvec>::from(xo);
            arma::fvec d1f = arma::conv_to<arma::fvec>::from(d1);
            arma::fvec r1f = quadriga_lib::interp_1D<float>(d1f, xif, xof);
            arma::fmat ref1f = ref_interp_2D<float>(arma::fmat(d1f.t()), xif, arma::fvec(1), xof, arma::fvec(1));
            CHECK(arma::approx_equal(r1f, arma::fvec(ref1f.row(0).t()), "absdiff", 0.0f));
        }
}

TEST_CASE("interp - a NaN query gives NaN, outside the grid the nearest end value is returned")
{
    arma::vec xi = {0, 1, 2, 3}, in = {0, 10, 20, 30}, xo = {-1, 0.5, 3.5, NAN, 3.0, 0.0};
    arma::vec r = quadriga_lib::interp_1D<double>(in, xi, xo);
    CHECK(r(0) == 0.0);
    CHECK(r(1) == 5.0);
    CHECK(r(2) == 30.0);
    CHECK(std::isnan(r(3)));
    CHECK(r(4) == 30.0);
    CHECK(r(5) == 0.0);

    // Unsorted grid and float
    arma::fvec xif = {3, 1, 0, 2}, inf_ = {30, 10, 0, 20}, xof = {-1.0f, NAN, 2.5f};
    arma::fvec rf = quadriga_lib::interp_1D<float>(inf_, xif, xof);
    CHECK(rf(0) == 0.0f);
    CHECK(std::isnan(rf(1)));
    CHECK(rf(2) == 25.0f);

    // 2D: a NaN x query gives a NaN column, a NaN y query a NaN row, everything else is finite
    arma::vec yi = {0, 1};
    arma::mat d = {{0, 1, 2, 3}, {10, 11, 12, 13}};
    arma::vec xo2 = {0.5, NAN, 2.0}, yo2 = {NAN, 0.5, 2.0};
    arma::mat r2 = quadriga_lib::interp_2D<double>(d, xi, yi, xo2, yo2);
    for (arma::uword ix = 0; ix < 3; ++ix)
        for (arma::uword iy = 0; iy < 3; ++iy)
        {
            INFO("ix " << ix << ", iy " << iy);
            CHECK(std::isnan(r2(iy, ix)) == (ix == 1 || iy == 0));
        }
    CHECK(r2(1, 0) == 5.5);
    CHECK(r2(2, 2) == 12.0);

    // Multiple data sets (1D matrix form)
    arma::mat dm = arma::join_rows(in, 2.0 * in);
    arma::mat rm = quadriga_lib::interp_1D<double>(dm, xi, xo);
    CHECK(std::isnan(rm(3, 0)));
    CHECK(std::isnan(rm(3, 1)));
    CHECK(rm(1, 1) == 10.0);
}

TEST_CASE("interp - non-finite grid values in 'xi' or 'yi' throw")
{
    // A NaN in a grid that is_sorted() accepts gave NaN at some queries, an unsorted grid with NaN threw inside the
    // sort; the grids are now checked first
    auto message = [](auto fn) -> std::string
    {
        try
        {
            fn();
        }
        catch (const std::invalid_argument &e)
        {
            return e.what();
        }
        return "no error";
    };
    const arma::vec in = {0, 10, 20, 30}, xo = {0.5, 1.5, 2.5};
    for (double bad : {(double)NAN, (double)INFINITY, -(double)INFINITY})
    {
        INFO("bad " << bad);
        for (arma::uword pos = 0; pos < 4; ++pos)
        {
            arma::vec xi = {0, 1, 2, 3};
            xi(pos) = bad;
            CHECK(message([&]() { quadriga_lib::interp_1D<double>(in, xi, xo); }) == "Input 'xi' must be finite.");
            arma::fvec xif = arma::conv_to<arma::fvec>::from(xi), inf_ = arma::conv_to<arma::fvec>::from(in);
            CHECK(message([&]() { quadriga_lib::interp_1D<float>(inf_, xif, arma::fvec{0.5f}); }) == "Input 'xi' must be finite.");
        }
        arma::vec xi = {0, 1, 2, 3}, yi = {0, bad};
        arma::mat d = {{0, 1, 2, 3}, {10, 11, 12, 13}};
        CHECK(message([&]() { quadriga_lib::interp_2D<double>(d, xi, yi, xo, arma::vec{0.5}); }) == "Input 'yi' must be finite.");
    }
}
