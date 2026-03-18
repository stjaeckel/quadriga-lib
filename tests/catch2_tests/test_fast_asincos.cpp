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

// Helper: compute max absolute error between fast result (float) and reference (double)
static double max_abs_error(const arma::fvec &fast, const arma::vec &ref)
{
    double max_err = 0.0;
    for (arma::uword i = 0; i < fast.n_elem; ++i)
    {
        double err = std::abs((double)fast(i) - ref(i));
        if (err > max_err)
            max_err = err;
    }
    return max_err;
}

TEST_CASE("fast_asin - Basic values (float input)")
{
    arma::fvec x(7);
    x(0) = -1.0f;
    x(1) = -0.75f;
    x(2) = -0.5f;
    x(3) = 0.0f;
    x(4) = 0.5f;
    x(5) = 0.75f;
    x(6) = 1.0f;

    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    REQUIRE(s.n_elem == 7);

    // Compare against std::asin (double precision reference)
    for (arma::uword i = 0; i < 7; ++i)
    {
        double ref = std::asin((double)x(i));
        CHECK(std::abs((double)s(i) - ref) < 3e-7);
    }
}

TEST_CASE("fast_asin - Basic values (double input)")
{
    arma::vec x(7);
    x(0) = -1.0;
    x(1) = -0.75;
    x(2) = -0.5;
    x(3) = 0.0;
    x(4) = 0.5;
    x(5) = 0.75;
    x(6) = 1.0;

    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    REQUIRE(s.n_elem == 7);

    for (arma::uword i = 0; i < 7; ++i)
    {
        double ref = std::asin(x(i));
        CHECK(std::abs((double)s(i) - ref) < 3e-7);
    }
}

TEST_CASE("fast_acos - Basic values (float input)")
{
    arma::fvec x(7);
    x(0) = -1.0f;
    x(1) = -0.75f;
    x(2) = -0.5f;
    x(3) = 0.0f;
    x(4) = 0.5f;
    x(5) = 0.75f;
    x(6) = 1.0f;

    arma::fvec c;
    quadriga_lib::fast_acos(x, c);

    REQUIRE(c.n_elem == 7);

    for (arma::uword i = 0; i < 7; ++i)
    {
        double ref = std::acos((double)x(i));
        CHECK(std::abs((double)c(i) - ref) < 3e-7);
    }
}

TEST_CASE("fast_acos - Basic values (double input)")
{
    arma::vec x(7);
    x(0) = -1.0;
    x(1) = -0.75;
    x(2) = -0.5;
    x(3) = 0.0;
    x(4) = 0.5;
    x(5) = 0.75;
    x(6) = 1.0;

    arma::fvec c;
    quadriga_lib::fast_acos(x, c);

    REQUIRE(c.n_elem == 7);

    for (arma::uword i = 0; i < 7; ++i)
    {
        double ref = std::acos(x(i));
        CHECK(std::abs((double)c(i) - ref) < 3e-7);
    }
}

TEST_CASE("fast_asin - Sweep accuracy (double input, error against float-cast reference)")
{
    // Use double input; the fast path converts to float internally.
    // Compare against asin((double)(float)x) to isolate polynomial error
    // from input quantization error.
    const arma::uword N = 100001;
    arma::vec x = arma::linspace<arma::vec>(-1.0, 1.0, N);

    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    REQUIRE(s.n_elem == N);

    // Reference: cast to float first (matching what the AVX2 path does), then compute in double
    arma::vec ref(N);
    for (arma::uword i = 0; i < N; ++i)
        ref(i) = std::asin((double)(float)x(i));

    double err = max_abs_error(s, ref);

    // Maximum error should be below 3e-7 (~2 ULP of single precision)
    CHECK(err < 3e-7);
}

TEST_CASE("fast_acos - Sweep accuracy (double input, error against float-cast reference)")
{
    const arma::uword N = 100001;
    arma::vec x = arma::linspace<arma::vec>(-1.0, 1.0, N);

    arma::fvec c;
    quadriga_lib::fast_acos(x, c);

    REQUIRE(c.n_elem == N);

    arma::vec ref(N);
    for (arma::uword i = 0; i < N; ++i)
        ref(i) = std::acos((double)(float)x(i));

    double err = max_abs_error(c, ref);

    CHECK(err < 3e-7);
}

TEST_CASE("fast_asin - Identity: sin(asin(x)) == x")
{
    const arma::uword N = 10001;
    arma::fvec x = arma::linspace<arma::fvec>(-1.0f, 1.0f, N);

    arma::fvec asin_x;
    quadriga_lib::fast_asin(x, asin_x);

    // Apply sin to asin result, check roundtrip
    arma::fvec s, c_dummy;
    quadriga_lib::fast_sincos(asin_x, &s, &c_dummy);

    for (arma::uword i = 0; i < N; ++i)
    {
        double roundtrip_err = std::abs((double)s(i) - (double)x(i));
        CHECK(roundtrip_err < 1e-5);
    }
}

TEST_CASE("fast_acos - Identity: cos(acos(x)) == x")
{
    const arma::uword N = 10001;
    arma::fvec x = arma::linspace<arma::fvec>(-1.0f, 1.0f, N);

    arma::fvec acos_x;
    quadriga_lib::fast_acos(x, acos_x);

    arma::fvec s_dummy, c;
    quadriga_lib::fast_sincos(acos_x, &s_dummy, &c);

    for (arma::uword i = 0; i < N; ++i)
    {
        double roundtrip_err = std::abs((double)c(i) - (double)x(i));
        CHECK(roundtrip_err < 1e-5);
    }
}

TEST_CASE("fast_asin + fast_acos - Pythagorean identity: asin(x) + acos(x) == pi/2")
{
    const arma::uword N = 10001;
    arma::vec x = arma::linspace<arma::vec>(-1.0, 1.0, N);

    arma::fvec s, c;
    quadriga_lib::fast_asin(x, s);
    quadriga_lib::fast_acos(x, c);

    const double pio2 = 1.5707963267948966;
    for (arma::uword i = 0; i < N; ++i)
    {
        double sum = (double)s(i) + (double)c(i);
        CHECK(std::abs(sum - pio2) < 5e-7);
    }
}

TEST_CASE("fast_asin - Edge cases: exact endpoints")
{
    arma::fvec x(3);
    x(0) = -1.0f;
    x(1) = 0.0f;
    x(2) = 1.0f;

    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    REQUIRE(s.n_elem == 3);

    const double pio2 = 1.5707963267948966;
    CHECK(std::abs((double)s(0) - (-pio2)) < 3e-7);  // asin(-1) = -pi/2
    CHECK(std::abs((double)s(1) - 0.0) < 1e-10);     // asin(0)  = 0
    CHECK(std::abs((double)s(2) - pio2) < 3e-7);      // asin(1)  = pi/2
}

TEST_CASE("fast_acos - Edge cases: exact endpoints")
{
    arma::fvec x(3);
    x(0) = -1.0f;
    x(1) = 0.0f;
    x(2) = 1.0f;

    arma::fvec c;
    quadriga_lib::fast_acos(x, c);

    REQUIRE(c.n_elem == 3);

    const double pi = 3.14159265358979323846;
    const double pio2 = 1.5707963267948966;
    CHECK(std::abs((double)c(0) - pi) < 3e-7);    // acos(-1) = pi
    CHECK(std::abs((double)c(1) - pio2) < 3e-7);  // acos(0)  = pi/2
    CHECK(std::abs((double)c(2) - 0.0) < 1e-10);  // acos(1)  = 0
}

TEST_CASE("fast_asin - Odd symmetry: asin(-x) == -asin(x)")
{
    const arma::uword N = 5001;
    arma::vec x_pos = arma::linspace<arma::vec>(0.0, 1.0, N);
    arma::vec x_neg = -x_pos;

    arma::fvec s_pos, s_neg;
    quadriga_lib::fast_asin(x_pos, s_pos);
    quadriga_lib::fast_asin(x_neg, s_neg);

    for (arma::uword i = 0; i < N; ++i)
        CHECK(std::abs((double)s_pos(i) + (double)s_neg(i)) < 1e-7);
}

TEST_CASE("fast_acos - Symmetry: acos(-x) == pi - acos(x)")
{
    const arma::uword N = 5001;
    arma::vec x_pos = arma::linspace<arma::vec>(0.0, 1.0, N);
    arma::vec x_neg = -x_pos;

    arma::fvec c_pos, c_neg;
    quadriga_lib::fast_acos(x_pos, c_pos);
    quadriga_lib::fast_acos(x_neg, c_neg);

    const double pi = 3.14159265358979323846;
    for (arma::uword i = 0; i < N; ++i)
        CHECK(std::abs((double)c_neg(i) - (pi - (double)c_pos(i))) < 5e-7);
}

TEST_CASE("fast_asin - Non-multiple-of-8 length (tail handling)")
{
    // 13 elements: 8 via AVX2 bulk + 5 via generic tail
    arma::vec x(13);
    for (arma::uword i = 0; i < 13; ++i)
        x(i) = -1.0 + 2.0 * (double)i / 12.0;

    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    REQUIRE(s.n_elem == 13);

    for (arma::uword i = 0; i < 13; ++i)
    {
        double ref = std::asin(x(i));
        CHECK(std::abs((double)s(i) - ref) < 3e-7);
    }
}

TEST_CASE("fast_acos - Non-multiple-of-8 length (tail handling)")
{
    arma::vec x(13);
    for (arma::uword i = 0; i < 13; ++i)
        x(i) = -1.0 + 2.0 * (double)i / 12.0;

    arma::fvec c;
    quadriga_lib::fast_acos(x, c);

    REQUIRE(c.n_elem == 13);

    for (arma::uword i = 0; i < 13; ++i)
    {
        double ref = std::acos(x(i));
        CHECK(std::abs((double)c(i) - ref) < 3e-7);
    }
}

TEST_CASE("fast_asin - Output resized automatically")
{
    arma::fvec x(16);
    x.fill(0.5f);

    arma::fvec s(3); // wrong size
    quadriga_lib::fast_asin(x, s);

    CHECK(s.n_elem == 16);
}

TEST_CASE("fast_acos - Output resized automatically")
{
    arma::fvec x(16);
    x.fill(0.5f);

    arma::fvec c(3); // wrong size
    quadriga_lib::fast_acos(x, c);

    CHECK(c.n_elem == 16);
}

TEST_CASE("fast_asin - Single element")
{
    arma::vec x(1);
    x(0) = 0.5;

    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    REQUIRE(s.n_elem == 1);
    CHECK(std::abs((double)s(0) - std::asin(0.5)) < 3e-7);
}

TEST_CASE("fast_acos - Single element")
{
    arma::vec x(1);
    x(0) = 0.5;

    arma::fvec c;
    quadriga_lib::fast_acos(x, c);

    REQUIRE(c.n_elem == 1);
    CHECK(std::abs((double)c(0) - std::acos(0.5)) < 3e-7);
}

TEST_CASE("fast_asin - Empty input")
{
    arma::fvec x;
    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    CHECK(s.n_elem == 0);
}

TEST_CASE("fast_acos - Empty input")
{
    arma::fvec x;
    arma::fvec c;
    quadriga_lib::fast_acos(x, c);

    CHECK(c.n_elem == 0);
}

TEST_CASE("fast_asin - Region boundary stress test around |x| = 0.5")
{
    // The polynomial switches regions at |x| = 0.5 — test values densely around this boundary
    const arma::uword N = 10001;
    arma::vec x = arma::linspace<arma::vec>(0.49, 0.51, N);

    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    arma::vec ref(N);
    for (arma::uword i = 0; i < N; ++i)
        ref(i) = std::asin((double)(float)x(i));

    double err = max_abs_error(s, ref);
    CHECK(err < 3e-7);
}

TEST_CASE("fast_acos - Region boundary stress test around |x| = 0.5")
{
    const arma::uword N = 10001;
    arma::vec x = arma::linspace<arma::vec>(0.49, 0.51, N);

    arma::fvec c;
    quadriga_lib::fast_acos(x, c);

    arma::vec ref(N);
    for (arma::uword i = 0; i < N; ++i)
        ref(i) = std::acos((double)(float)x(i));

    double err = max_abs_error(c, ref);
    CHECK(err < 3e-7);
}

TEST_CASE("fast_asin - Stress test near |x| = 1")
{
    // Near the endpoints, the half-angle identity kicks in — verify stability
    // Compare against float-cast reference to isolate polynomial error
    const arma::uword N = 10001;
    arma::vec x = arma::linspace<arma::vec>(0.99, 1.0, N);

    arma::fvec s;
    quadriga_lib::fast_asin(x, s);

    arma::vec ref(N);
    for (arma::uword i = 0; i < N; ++i)
        ref(i) = std::asin((double)(float)x(i));

    double err = max_abs_error(s, ref);
    CHECK(err < 3e-7);
}

TEST_CASE("fast_acos - Stress test near x = 1 and x = -1")
{
    const arma::uword N = 10001;

    // Near x = 1 (acos → 0)
    arma::vec x1 = arma::linspace<arma::vec>(0.99, 1.0, N);
    arma::fvec c1;
    quadriga_lib::fast_acos(x1, c1);

    arma::vec ref1(N);
    for (arma::uword i = 0; i < N; ++i)
        ref1(i) = std::acos((double)(float)x1(i));

    CHECK(max_abs_error(c1, ref1) < 3e-7);

    // Near x = -1 (acos → pi)
    arma::vec x2 = arma::linspace<arma::vec>(-1.0, -0.99, N);
    arma::fvec c2;
    quadriga_lib::fast_acos(x2, c2);

    arma::vec ref2(N);
    for (arma::uword i = 0; i < N; ++i)
        ref2(i) = std::acos((double)(float)x2(i));

    CHECK(max_abs_error(c2, ref2) < 3e-7);
}