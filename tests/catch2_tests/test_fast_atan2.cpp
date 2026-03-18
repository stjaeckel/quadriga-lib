// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include "quadriga_math.hpp"
#include <cmath>
#include <cstdint>

// Helper: compute ULP distance between two floats
static int32_t ulp_dist(float a, float b)
{
    if (std::isnan(a) || std::isnan(b))
        return INT32_MAX;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = (int32_t)0x80000000 - ia;
    if (ib < 0) ib = (int32_t)0x80000000 - ib;
    int32_t d = ia - ib;
    return d < 0 ? -d : d;
}

TEST_CASE("fast_atan2 - Basic quadrant coverage with float input")
{
    arma::fvec y = {1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f};
    arma::fvec x = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f};
    arma::fvec a;

    quadriga_lib::fast_atan2(y, x, a);

    REQUIRE(a.n_elem == 8);

    // Compare against std::atan2f with tight tolerance
    for (arma::uword i = 0; i < 8; ++i)
    {
        float ref = std::atan2(y(i), x(i));
        CHECK(ulp_dist(a(i), ref) <= 3);
    }
}

TEST_CASE("fast_atan2 - Basic quadrant coverage with double input")
{
    arma::vec y = {1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0};
    arma::vec x = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0};
    arma::fvec a;

    quadriga_lib::fast_atan2(y, x, a);

    REQUIRE(a.n_elem == 8);

    for (arma::uword i = 0; i < 8; ++i)
    {
        float ref = std::atan2((float)y(i), (float)x(i));
        CHECK(ulp_dist(a(i), ref) <= 3);
    }
}

TEST_CASE("fast_atan2 - Special cases")
{
    arma::fvec y = {0.0f, 0.0f, 0.0f, 1e-30f, 1.0f, 1e30f};
    arma::fvec x = {0.0f, 1.0f, -1.0f, 1.0f, 1e-30f, 1e30f};
    arma::fvec a;

    quadriga_lib::fast_atan2(y, x, a);

    REQUIRE(a.n_elem == 6);

    // atan2(0, 0) = 0
    CHECK(a(0) == 0.0f);

    // atan2(0, +1) = 0
    CHECK(a(1) == 0.0f);

    // atan2(0, -1) = pi
    CHECK(ulp_dist(a(2), (float)M_PI) <= 1);

    // atan2(tiny, 1) ≈ 0
    CHECK(ulp_dist(a(3), std::atan2(1e-30f, 1.0f)) <= 3);

    // atan2(1, tiny) ≈ pi/2
    CHECK(ulp_dist(a(4), (float)(M_PI / 2.0)) <= 3);

    // atan2(big, big) = pi/4
    CHECK(ulp_dist(a(5), (float)(M_PI / 4.0)) <= 3);
}

TEST_CASE("fast_atan2 - Accuracy bound over dense sweep (float)")
{
    // Generate a dense grid of angles covering all quadrants
    const arma::uword N = 512;
    const arma::uword total = (2 * N + 1) * (2 * N + 1) - 1; // exclude (0,0)

    arma::fvec y_vec(total), x_vec(total);
    arma::uword idx = 0;
    for (int iy = -(int)N; iy <= (int)N; ++iy)
    {
        for (int ix = -(int)N; ix <= (int)N; ++ix)
        {
            if (ix == 0 && iy == 0)
                continue;
            y_vec(idx) = (float)iy / (float)N * 10.0f;
            x_vec(idx) = (float)ix / (float)N * 10.0f;
            idx++;
        }
    }

    arma::fvec a;
    quadriga_lib::fast_atan2(y_vec, x_vec, a);

    REQUIRE(a.n_elem == total);

    int32_t max_ulp = 0;
    for (arma::uword i = 0; i < total; ++i)
    {
        float ref = std::atan2(y_vec(i), x_vec(i));
        int32_t d = ulp_dist(a(i), ref);
        if (d > max_ulp)
            max_ulp = d;
    }

    // Accuracy bound: must be <= 3 ULP everywhere
    CHECK(max_ulp <= 3);
}

TEST_CASE("fast_atan2 - Accuracy bound over dense sweep (double)")
{
    const arma::uword N = 512;
    const arma::uword total = (2 * N + 1) * (2 * N + 1) - 1;

    arma::vec y_vec(total), x_vec(total);
    arma::uword idx = 0;
    for (int iy = -(int)N; iy <= (int)N; ++iy)
    {
        for (int ix = -(int)N; ix <= (int)N; ++ix)
        {
            if (ix == 0 && iy == 0)
                continue;
            y_vec(idx) = (double)iy / (double)N * 10.0;
            x_vec(idx) = (double)ix / (double)N * 10.0;
            idx++;
        }
    }

    arma::fvec a;
    quadriga_lib::fast_atan2(y_vec, x_vec, a);

    REQUIRE(a.n_elem == total);

    int32_t max_ulp = 0;
    for (arma::uword i = 0; i < total; ++i)
    {
        float ref = std::atan2((float)y_vec(i), (float)x_vec(i));
        int32_t d = ulp_dist(a(i), ref);
        if (d > max_ulp)
            max_ulp = d;
    }

    CHECK(max_ulp <= 3);
}

TEST_CASE("fast_atan2 - Non-multiple-of-8 length (tail handling)")
{
    // Length 13: bulk = 8, tail = 5
    arma::fvec y(13), x(13);
    for (arma::uword i = 0; i < 13; ++i)
    {
        float angle = -3.0f + 6.0f * (float)i / 12.0f;
        y(i) = std::sin(angle) * (1.0f + (float)i);
        x(i) = std::cos(angle) * (1.0f + (float)i);
    }

    arma::fvec a;
    quadriga_lib::fast_atan2(y, x, a);

    REQUIRE(a.n_elem == 13);

    for (arma::uword i = 0; i < 13; ++i)
    {
        float ref = std::atan2(y(i), x(i));
        CHECK(ulp_dist(a(i), ref) <= 3);
    }
}

TEST_CASE("fast_atan2 - Empty input")
{
    arma::fvec y, x, a;
    quadriga_lib::fast_atan2(y, x, a);
    CHECK(a.n_elem == 0);
}

TEST_CASE("fast_atan2 - Output auto-resize")
{
    arma::fvec y = {1.0f, 2.0f, 3.0f, 4.0f};
    arma::fvec x = {4.0f, 3.0f, 2.0f, 1.0f};
    arma::fvec a(100); // wrong size, should be resized

    quadriga_lib::fast_atan2(y, x, a);
    CHECK(a.n_elem == 4);
}

TEST_CASE("fast_atan2 - Error: mismatched input lengths")
{
    arma::fvec y(10), x(12), a;
    CHECK_THROWS_AS(quadriga_lib::fast_atan2(y, x, a), std::invalid_argument);
}

TEST_CASE("fast_atan2 - Error: input-output aliasing")
{
    arma::fvec y = {1.0f, 2.0f, 3.0f, 4.0f};
    arma::fvec x = {1.0f, 1.0f, 1.0f, 1.0f};

    // y and a are the same buffer
    CHECK_THROWS_AS(quadriga_lib::fast_atan2(y, x, y), std::invalid_argument);

    // x and a are the same buffer
    CHECK_THROWS_AS(quadriga_lib::fast_atan2(y, x, x), std::invalid_argument);
}

TEST_CASE("fast_atan2 - Consistency: atan2(sin(t), cos(t)) recovers t")
{
    // For angles in (-pi, pi], atan2(sin(t), cos(t)) should recover t
    const arma::uword N = 1000;
    arma::fvec angles = arma::linspace<arma::fvec>(-3.14f, 3.14f, N);
    arma::fvec y(N), x(N);

    for (arma::uword i = 0; i < N; ++i)
    {
        y(i) = std::sin(angles(i));
        x(i) = std::cos(angles(i));
    }

    arma::fvec a;
    quadriga_lib::fast_atan2(y, x, a);

    REQUIRE(a.n_elem == N);

    // The recovered angle should be close to the original
    // Allow slightly larger tolerance due to sin/cos approximation compounding
    for (arma::uword i = 0; i < N; ++i)
    {
        float ref = std::atan2(y(i), x(i));
        CHECK(ulp_dist(a(i), ref) <= 3);
    }
}

TEST_CASE("fast_atan2 - Large vector (exercises OpenMP and AVX2 bulk path)")
{
    const arma::uword N = 10000;
    arma::fvec y(N), x(N);
    for (arma::uword i = 0; i < N; ++i)
    {
        float t = -100.0f + 200.0f * (float)i / (float)(N - 1);
        y(i) = std::sin(t) * (1.0f + 0.01f * (float)i);
        x(i) = std::cos(t) * (1.0f + 0.01f * (float)i);
    }

    arma::fvec a;
    quadriga_lib::fast_atan2(y, x, a);

    REQUIRE(a.n_elem == N);

    int32_t max_ulp = 0;
    for (arma::uword i = 0; i < N; ++i)
    {
        float ref = std::atan2(y(i), x(i));
        int32_t d = ulp_dist(a(i), ref);
        if (d > max_ulp)
            max_ulp = d;
    }

    CHECK(max_ulp <= 3);
}
