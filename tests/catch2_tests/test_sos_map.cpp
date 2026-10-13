// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::sos_map.
//
// Pins down: the meshgrid ordering of the output, the handling of omitted grid vectors, one cube per
// generator, the Uniform switch, sizes, float and double, input validation.
// It leaves the values of the sum of sinusoids to test_sos_val.
// Oracle: scalar transcription of s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n ) at the grid point.
// Conventions: grid vectors have different lengths (nx = 3, ny = 2, nz = 4) and unequal spacing, so that
// a swapped axis changes the size and the values.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

// Diagnostic output. Comment out to silence the tables. The assertions run either way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

namespace
{
    const double pi = 3.141592653589793;

    // Uniform random matrix in [lo, hi), own generator so that the fixtures do not depend on the library
    arma::mat rnd(arma::uword n_rows, arma::uword n_cols, double lo, double hi, unsigned seed)
    {
        std::mt19937_64 rng(seed);
        arma::mat m(n_rows, n_cols);
        for (arma::uword i = 0; i < m.n_elem; ++i)
            m(i) = lo + (hi - lo) * ((double)(rng() >> 11) / 9007199254740992.0);
        return m;
    }

    // Oracle: value of generator g at {x, y, z}
    double ref_s(const arma::mat &freq, const arma::mat &phase, arma::uword g, double x, double y, double z)
    {
        double acc = 0.0, p[3] = {x, y, z};
        for (arma::uword n = 0; n < freq.n_rows; ++n)
        {
            double arg = 0.0;
            for (arma::uword c = 0; c < freq.n_cols; ++c)
                arg += freq(n, c) * p[c];
            acc += std::cos(2.0 * pi * arg + phase(n, g));
        }
        return std::sqrt(2.0 / (double)freq.n_rows) * acc;
    }
}

TEST_CASE("sos_map - cubes are in meshgrid order with one cube per generator")
{
    arma::mat freq = rnd(9, 3, -0.3, 0.3, 1), phase = rnd(9, 2, -pi, pi, 2);
    arma::vec x = {-3.0, 0.5, 7.0}, y = {2.0, -4.5}, z = {0.0, 1.0, 2.5, 10.0};

    std::vector<arma::cube> map;
    quadriga_lib::sos_map(freq, phase, x, map, &y, &z, false, 1);
    REQUIRE(map.size() == 2);

    double max_err = 0.0, max_val = 0.0;
    QD_DIAG("\n# diag: sos_map against the scalar oracle\n");
    QD_DIAG("gen,iy,ix,iz,val,ref,diff\n");
    for (arma::uword g = 0; g < 2; ++g)
    {
        REQUIRE(map[g].n_rows == 2);
        REQUIRE(map[g].n_cols == 3);
        REQUIRE(map[g].n_slices == 4);
        for (arma::uword iz = 0; iz < 4; ++iz)
            for (arma::uword ix = 0; ix < 3; ++ix)
                for (arma::uword iy = 0; iy < 2; ++iy)
                {
                    double ref = ref_s(freq, phase, g, x(ix), y(iy), z(iz)), v = map[g](iy, ix, iz);
                    if (iz == 1 && iy == 1)
                        QD_DIAG("%llu,%llu,%llu,%llu,%.9f,%.9f,%.2e\n", (unsigned long long)g, (unsigned long long)iy,
                                (unsigned long long)ix, (unsigned long long)iz, v, ref, v - ref);
                    max_err = std::max(max_err, std::abs(v - ref));
                    max_val = std::max(max_val, std::abs(v));
                }
    }
    CHECK(max_err < 1e-12);
    CHECK(max_val > 0.5);
    CHECK(arma::abs(map[0] - map[1]).max() > 0.1);
}

TEST_CASE("sos_map - omitted grid vectors are a single sample at zero")
{
    arma::mat freq = rnd(9, 3, -0.3, 0.3, 3), phase = rnd(9, 1, -pi, pi, 4);
    arma::vec x = {-3.0, 0.5, 7.0}, y = {2.0, -4.5}, z = {1.0, 2.5, 10.0, -1.0}, none;

    std::vector<arma::cube> map;
    quadriga_lib::sos_map(freq, phase, x, map, (arma::vec *)nullptr, (arma::vec *)nullptr, false, 1);
    REQUIRE(map.size() == 1);
    REQUIRE(map[0].n_rows == 1);
    REQUIRE(map[0].n_cols == 3);
    REQUIRE(map[0].n_slices == 1);
    for (arma::uword ix = 0; ix < 3; ++ix)
        CHECK(std::abs(map[0](0, ix, 0) - ref_s(freq, phase, 0, x(ix), 0.0, 0.0)) < 1e-12);
    CHECK(arma::abs(map[0]).max() > 0.1);

    // An empty vector means omitted
    std::vector<arma::cube> map_e;
    quadriga_lib::sos_map(freq, phase, x, map_e, &none, &none, false, 1);
    REQUIRE(map_e.size() == 1);
    CHECK(arma::approx_equal(map_e[0], map[0], "absdiff", 0.0));

    // x and y
    quadriga_lib::sos_map(freq, phase, x, map, &y, (arma::vec *)nullptr, false, 1);
    REQUIRE(map[0].n_rows == 2);
    REQUIRE(map[0].n_cols == 3);
    REQUIRE(map[0].n_slices == 1);
    CHECK(std::abs(map[0](1, 2, 0) - ref_s(freq, phase, 0, x(2), y(1), 0.0)) < 1e-12);

    // x and z without y
    quadriga_lib::sos_map(freq, phase, x, map, (arma::vec *)nullptr, &z, false, 1);
    REQUIRE(map[0].n_rows == 1);
    REQUIRE(map[0].n_cols == 3);
    REQUIRE(map[0].n_slices == 4);
    CHECK(std::abs(map[0](0, 1, 3) - ref_s(freq, phase, 0, x(1), 0.0, z(3))) < 1e-12);

    // Frequencies with one column
    arma::mat f1 = freq.col(0);
    quadriga_lib::sos_map(f1, phase, x, map, (arma::vec *)nullptr, (arma::vec *)nullptr, false, 1);
    REQUIRE(map[0].n_cols == 3);
    CHECK(std::abs(map[0](0, 2, 0) - ref_s(f1, phase, 0, x(2), 0.0, 0.0)) < 1e-12);
}

TEST_CASE("sos_map - Uniform output, kernels and float")
{
    arma::mat freq = rnd(30, 3, -0.3, 0.3, 5), phase = rnd(30, 3, -pi, pi, 6);
    arma::vec x = arma::linspace(-20.0, 20.0, 11), y = arma::linspace(-5.0, 9.0, 5);

    std::vector<arma::cube> nrm, uni, k0;
    quadriga_lib::sos_map(freq, phase, x, nrm, &y, (arma::vec *)nullptr, false, 1);
    quadriga_lib::sos_map(freq, phase, x, uni, &y, (arma::vec *)nullptr, true, 1);
    quadriga_lib::sos_map(freq, phase, x, k0, &y);
    REQUIRE(uni.size() == 3);
    REQUIRE(k0.size() == 3);
    REQUIRE(uni[2].n_elem == 55);
    for (arma::uword i = 0; i < 55; ++i)
        CHECK(std::abs(uni[2](i) - 0.5 * std::erfc(-nrm[2](i) / std::sqrt(2.0))) < 1e-14);
    CHECK(arma::abs(nrm[2]).max() > 0.5);

    // Automatic kernel: within the single-precision bound of the AVX2 kernel, 4e-7 · sqrt(2·30)
    CHECK(arma::abs(k0[1] - nrm[1]).max() < 4e-7 * std::sqrt(60.0));

    arma::fmat ff = arma::conv_to<arma::fmat>::from(freq), pf = arma::conv_to<arma::fmat>::from(phase);
    arma::fvec xf = arma::conv_to<arma::fvec>::from(x), yf = arma::conv_to<arma::fvec>::from(y);
    std::vector<arma::fcube> mf;
    quadriga_lib::sos_map(ff, pf, xf, mf, &yf);
    REQUIRE(mf.size() == 3);
    REQUIRE(mf[0].n_rows == 5);
    REQUIRE(mf[0].n_cols == 11);
    // Inputs rounded to float, arguments up to about 60 rad: see test_sos_val for the bound
    CHECK(arma::abs(arma::conv_to<arma::cube>::from(mf[1]) - nrm[1]).max() < 1e-3);
}

TEST_CASE("sos_map - a pre-sized output is written in place and a wrong size is replaced")
{
    arma::mat freq = rnd(9, 3, -0.3, 0.3, 7), phase = rnd(9, 2, -pi, pi, 8);
    arma::vec x = {-3.0, 0.5, 7.0}, y = {2.0, -4.5};

    std::vector<arma::cube> ref;
    quadriga_lib::sos_map(freq, phase, x, ref, &y, (arma::vec *)nullptr, false, 1);
    REQUIRE(ref.size() == 2);

    std::vector<arma::cube> sized(2);
    sized[0].set_size(2, 3, 1), sized[1].set_size(2, 3, 1);
    sized[0].fill(99.0), sized[1].fill(99.0);
    const double *p0 = sized[0].memptr(), *p1 = sized[1].memptr();
    quadriga_lib::sos_map(freq, phase, x, sized, &y, (arma::vec *)nullptr, false, 1);
    CHECK(sized[0].memptr() == p0);
    CHECK(sized[1].memptr() == p1);
    CHECK(arma::approx_equal(sized[0], ref[0], "absdiff", 0.0));
    CHECK(arma::approx_equal(sized[1], ref[1], "absdiff", 0.0));

    std::vector<arma::cube> wrong(5);
    wrong[0].set_size(4, 4, 4);
    wrong[0].fill(99.0);
    quadriga_lib::sos_map(freq, phase, x, wrong, &y, (arma::vec *)nullptr, false, 1);
    REQUIRE(wrong.size() == 2);
    CHECK(arma::approx_equal(wrong[0], ref[0], "absdiff", 0.0));
    CHECK(arma::approx_equal(wrong[1], ref[1], "absdiff", 0.0));
    CHECK(arma::abs(ref[0]).max() > 0.1);
}

TEST_CASE("sos_map - input validation")
{
    arma::mat freq = rnd(9, 3, -0.3, 0.3, 9), phase = rnd(9, 2, -pi, pi, 10);
    arma::vec x = {-3.0, 0.5, 7.0}, y = {2.0, -4.5}, z = {1.0};
    std::vector<arma::cube> map;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_map(freq, phase, x, map));
        CHECK_NOTHROW(quadriga_lib::sos_map(freq, phase, x, map, &y, &z, true, 1));
    }
    SECTION("freq")
    {
        arma::mat empty, four(9, 4, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_map(empty, phase, x, map), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_map(four, phase, x, map), std::invalid_argument);
    }
    SECTION("phase")
    {
        arma::mat empty, rows(8, 2, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_map(freq, empty, x, map), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_map(freq, rows, x, map), std::invalid_argument);
    }
    SECTION("x")
    {
        arma::vec empty;
        CHECK_THROWS_AS(quadriga_lib::sos_map(freq, phase, empty, map), std::invalid_argument);
    }
    SECTION("y and z need enough dimensions")
    {
        arma::mat f1 = freq.col(0), f2 = freq.cols(0, 1);
        CHECK_THROWS_AS(quadriga_lib::sos_map(f1, phase, x, map, &y), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_map(f2, phase, x, map, &y, &z), std::invalid_argument);
        CHECK_NOTHROW(quadriga_lib::sos_map(f2, phase, x, map, &y));
    }
    SECTION("use_kernel")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_map(freq, phase, x, map, &y, &z, false, 3), std::invalid_argument);
    }
    SECTION("a kernel that is not available")
    {
        // Runs in a build without AVX2 or on a CPU without it; with AVX2 the same call must work
        if (!quadriga_lib::quadriga_lib_has_AVX2())
            CHECK_THROWS_AS(quadriga_lib::sos_map(freq, phase, x, map, &y, &z, false, 2), std::invalid_argument);
        else
            CHECK_NOTHROW(quadriga_lib::sos_map(freq, phase, x, map, &y, &z, false, 2));
    }
    SECTION("a failed call leaves the output untouched")
    {
        std::vector<arma::cube> keep(1);
        keep[0].set_size(2, 2, 2);
        keep[0].fill(7.0);
        arma::vec empty;
        CHECK_THROWS_AS(quadriga_lib::sos_map(freq, phase, empty, keep), std::invalid_argument);
        REQUIRE(keep.size() == 1);
        CHECK(keep[0].n_slices == 2);
        CHECK(keep[0](1, 1, 1) == 7.0);
    }
}

TEST_CASE("sos_map - cubes that share one block of memory are filled in place")
{
    // The MEX and Python wrappers allocate the 4-D output as one block and hand the function cubes that
    // alias it. The function then evaluates into that block; the result must be the same as for separate
    // cubes, the cubes must keep their memory, and nothing around the block may be written.
    arma::mat freq = rnd(9, 3, -0.3, 0.3, 11), phase = rnd(9, 3, -pi, pi, 12);
    arma::vec x = {-3.0, 0.5, 7.0, 9.0}, y = {2.0, -4.5}, z = {1.0, 6.0, -2.0};
    const arma::uword n_pos = 4 * 2 * 3, n_gen = 3;

    std::vector<arma::cube> ref;
    quadriga_lib::sos_map(freq, phase, x, ref, &y, &z, false, 1);
    REQUIRE(ref.size() == n_gen);
    CHECK(arma::abs(ref[2]).max() > 0.3);

    // One guard value before and after the block
    arma::vec block(n_pos * n_gen + 2);
    block.fill(77.0);
    std::vector<arma::cube> map;
    map.reserve(n_gen); // A cube that aliases memory is copied when it is moved, so the vector must not grow
    for (arma::uword g = 0; g < n_gen; ++g)
        map.emplace_back(block.memptr() + 1 + g * n_pos, 2, 4, 3, false, true);
    REQUIRE(map[2].memptr() == block.memptr() + 1 + 2 * n_pos);

    for (bool uniform : {false, true})
    {
        std::vector<arma::cube> sep;
        quadriga_lib::sos_map(freq, phase, x, sep, &y, &z, uniform, 1);
        quadriga_lib::sos_map(freq, phase, x, map, &y, &z, uniform, 1);
        REQUIRE(map.size() == n_gen);
        for (arma::uword g = 0; g < n_gen; ++g)
        {
            INFO("uniform " << uniform << ", generator " << g);
            CHECK(map[g].memptr() == block.memptr() + 1 + g * n_pos);
            CHECK(arma::approx_equal(map[g], sep[g], "absdiff", 0.0));
        }
        CHECK(block(0) == 77.0);
        CHECK(block(n_pos * n_gen + 1) == 77.0);
    }

    // Float
    arma::fmat ff = arma::conv_to<arma::fmat>::from(freq), pf = arma::conv_to<arma::fmat>::from(phase);
    arma::fvec xf = arma::conv_to<arma::fvec>::from(x), yf = arma::conv_to<arma::fvec>::from(y), zf = arma::conv_to<arma::fvec>::from(z);
    arma::fvec fblock(n_pos * n_gen);
    std::vector<arma::fcube> fmap, fsep;
    fmap.reserve(n_gen);
    for (arma::uword g = 0; g < n_gen; ++g)
        fmap.emplace_back(fblock.memptr() + g * n_pos, 2, 4, 3, false, true);
    quadriga_lib::sos_map(ff, pf, xf, fsep, &yf, &zf, false, 1);
    quadriga_lib::sos_map(ff, pf, xf, fmap, &yf, &zf, false, 1);
    for (arma::uword g = 0; g < n_gen; ++g)
    {
        CHECK(fmap[g].memptr() == fblock.memptr() + g * n_pos);
        CHECK(arma::approx_equal(fmap[g], fsep[g], "absdiff", 0.0f));
    }
}
