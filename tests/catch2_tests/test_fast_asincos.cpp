// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of quadriga_lib::fast_asin and fast_acos: accuracy against std::asin / std::acos for float and
// double input, identities, endpoints and symmetries, the region boundary at |x| = 0.5 and the endpoints,
// lengths that are not a multiple of 8, output sizing, aliasing, and the GENERIC and AVX2 kernels.
// Double input is rounded to float by the function, so the sweeps compare with the exact value of the
// float-rounded input; 3e-7 is about 2 ULP of single precision.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_math.hpp"
#include "quadriga_lib.hpp" // quadriga_lib_has_AVX2
#include <cmath>

// Max absolute error of a float result against a double reference
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

    // Reference: cast to float first (as both kernels do), then compute in double
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
    // 13 elements: one full 8-wide block and a 5-element tail
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

TEST_CASE("fast_asin / fast_acos - Outputs that alias inputs")
{
    arma::fvec x = arma::linspace<arma::fvec>(-1.0f, 1.0f, 9), y;
    CHECK_NOTHROW(quadriga_lib::fast_asin(x, y));
    CHECK_NOTHROW(quadriga_lib::fast_acos(x, y));

    // The output is the input
    CHECK_THROWS_AS(quadriga_lib::fast_asin(x, x), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::fast_acos(x, x), std::invalid_argument);

    // The output uses part of the memory of the input (shifted by one element)
    arma::fvec buf = arma::linspace<arma::fvec>(-1.0f, 1.0f, 10);
    arma::fvec x_part(buf.memptr(), 9, false, true), y_part(buf.memptr() + 1, 9, false, true);
    CHECK_THROWS_AS(quadriga_lib::fast_asin(x_part, y_part), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::fast_acos(x_part, y_part), std::invalid_argument);
}

TEST_CASE("fast_asin / fast_acos - GENERIC and AVX2 meet the documented error, auto is AVX2 if available")
{
    // Both kernels stay within the documented error of the exact values of the float-rounded input; auto is AVX2
    // when available, else GENERIC
    const bool avx2 = quadriga_lib::quadriga_lib_has_AVX2();
    const arma::vec x = arma::linspace<arma::vec>(-1.0, 1.0, 1003);
    const arma::vec xr = arma::conv_to<arma::vec>::from(arma::conv_to<arma::fvec>::from(x));
    arma::fvec as_auto, ac_auto;
    quadriga_lib::fast_asin(x, as_auto);
    quadriga_lib::fast_acos(x, ac_auto);
    for (int k : {1, 2})
    {
        arma::fvec as, ac;
        if (k == 2 && !avx2)
        {
            CHECK_THROWS_AS(quadriga_lib::fast_asin(x, as, k), std::invalid_argument);
            CHECK_THROWS_AS(quadriga_lib::fast_acos(x, ac, k), std::invalid_argument);
            continue;
        }
        quadriga_lib::fast_asin(x, as, k);
        quadriga_lib::fast_acos(x, ac, k);
        CHECK(max_abs_error(as, arma::asin(xr)) < 3e-7);
        CHECK(max_abs_error(ac, arma::acos(xr)) < 3e-7);
        if (k == (avx2 ? 2 : 1))
        {
            CHECK(arma::approx_equal(as, as_auto, "absdiff", 0.0f));
            CHECK(arma::approx_equal(ac, ac_auto, "absdiff", 0.0f));
        }
    }
    arma::fvec a;
    CHECK_THROWS_AS(quadriga_lib::fast_asin(x, a, 3), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::fast_acos(x, a, -1), std::invalid_argument);
}
