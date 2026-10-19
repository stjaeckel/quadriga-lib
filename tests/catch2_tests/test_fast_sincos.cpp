// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>           // Catch::Approx (Catch2 v3)
#include "quadriga_math.hpp"                // declares quadriga_lib::sincos_approx
#include <armadillo>
#include <cmath>
#include <vector>

static inline float ref_sin(float x) { return std::sin(x); }
static inline float ref_cos(float x) { return std::cos(x); }

TEMPLATE_TEST_CASE("Fast Sine / Cosine computation", "[fast_sincos]", float, double)
{
    using dtype = TestType;

    // deterministic set of angles
    constexpr float pi = 3.14159265358979323846f;
    std::vector<float> base = {
        0.0f,  pi/6,  pi/4,  pi/3,  pi/2,  2*pi/3,  3*pi/4,  5*pi/6,  pi,
       -pi/6, -pi/4, -pi/2, -pi,    2*pi,  -2*pi,   7.0f,   -7.0f,
        10.0f*pi, -10.0f*pi, 123.456f, -321.987f
    };

    const size_t target_len = 1003; // not divisible by 8
    base.reserve(target_len);
    for (size_t i = base.size(); i < target_len; ++i) {
        float t = static_cast<float>(i) * 0.03125f - 16.0f;
        base.push_back(t * pi);
    }

    arma::Col<dtype> x(base.size());
    for (size_t i = 0; i < base.size(); ++i)
        x[i] = static_cast<dtype>(base[i]);

    arma::fvec s, c;
    quadriga_lib::fast_sincos(x, &s, &c);

    REQUIRE(s.n_elem == x.n_elem);
    REQUIRE(c.n_elem == x.n_elem);

    const float abs_tol    = 2e-6f;
    const float pythag_tol = 5e-6f;

    for (arma::uword i = 0; i < x.n_elem; ++i) {
        INFO("dtype=" << (std::is_same_v<dtype,float> ? "float" : "double")
             << " i=" << i << " x=" << static_cast<float>(x[i]));

        const float rs = ref_sin(static_cast<float>(x[i]));
        const float rc = ref_cos(static_cast<float>(x[i]));

        REQUIRE(s[i] == Catch::Approx(rs).margin(abs_tol));
        REQUIRE(c[i] == Catch::Approx(rc).margin(abs_tol));

        const float unit = s[i]*s[i] + c[i]*c[i];
        REQUIRE(unit == Catch::Approx(1.0f).margin(pythag_tol));
    }
}
TEMPLATE_TEST_CASE("Fast Sine / Cosine - large arguments", "[fast_sincos]", float, double)
{
    using dtype = TestType;

    // Large finite arguments: same accuracy as for small ones, |s| and |c| never exceed 1
    // 9 values: one full block of 8 and a tail of 1
    std::vector<double> xs = {1.0e7, 3.3e7, 1.0e8, 1.7e9, 3.0e9, -5.0e9, 1.0e10, -1.0e10, 2.0e9};
    arma::Col<dtype> x(xs.size());
    for (size_t i = 0; i < xs.size(); ++i)
        x[i] = (dtype)xs[i];

    arma::fvec s, c;
    quadriga_lib::fast_sincos(x, &s, &c);
    for (arma::uword i = 0; i < x.n_elem; ++i)
    {
        const double xd = (double)x[i]; // reference for the value as stored in dtype
        INFO("x = " << xd);
        CHECK(std::abs((double)s[i] - std::sin(xd)) < 1e-6);
        CHECK(std::abs((double)c[i] - std::cos(xd)) < 1e-6);
        CHECK(std::abs(s[i]) <= 1.0f);
        CHECK(std::abs(c[i]) <= 1.0f);
    }

    // Sine only and cosine only use the same kernels
    arma::fvec s1, c1;
    quadriga_lib::fast_sincos(x, &s1, nullptr);
    quadriga_lib::fast_sincos(x, nullptr, &c1);
    CHECK(arma::approx_equal(s1, s, "absdiff", 0.0f));
    CHECK(arma::approx_equal(c1, c, "absdiff", 0.0f));

    // Non-finite input gives NaN
    arma::Col<dtype> xn = {(dtype)INFINITY, -(dtype)INFINITY, (dtype)NAN, (dtype)1.0};
    quadriga_lib::fast_sincos(xn, &s, &c);
    for (arma::uword i = 0; i < 3; ++i)
    {
        CHECK(std::isnan(s[i]));
        CHECK(std::isnan(c[i]));
    }
    CHECK(std::abs(s[3] - std::sin(1.0f)) < 1e-6f);
}

TEMPLATE_TEST_CASE("Fast Sine / Cosine - sweep over [-1e10, 1e10] and lane independence", "[fast_sincos]", float, double)
{
    using dtype = TestType;

    // Log-spaced magnitudes from 1e-3 to 1e10, both signs, plus a uniform sweep over [-1e10, 1e10]
    const arma::uword n = 20001;
    arma::vec mag = arma::logspace<arma::vec>(-3.0, 10.0, n);
    arma::vec xd = arma::join_cols(mag, -mag, arma::linspace<arma::vec>(-1e10, 1e10, n));
    arma::Col<dtype> x = arma::conv_to<arma::Col<dtype>>::from(xd);

    arma::fvec s, c;
    quadriga_lib::fast_sincos(x, &s, &c);
    double es = 0.0, ec = 0.0;
    for (arma::uword i = 0; i < x.n_elem; ++i)
    {
        const double v = (double)x[i];
        es = std::max(es, std::abs((double)s[i] - std::sin(v)));
        ec = std::max(ec, std::abs((double)c[i] - std::cos(v)));
    }
    CHECK(es < 5e-7); // Any two kernels within 5e-7 of the reference agree to 1e-6
    CHECK(ec < 5e-7);

    // The result of a lane does not depend on the other lanes of its block
    arma::Col<dtype> small = {(dtype)0.1, (dtype)-2.5, (dtype)3.0, (dtype)100.0, (dtype)-7.0, (dtype)0.0, (dtype)1e4, (dtype)5e5};
    arma::fvec s0, c0;
    quadriga_lib::fast_sincos(small, &s0, &c0);
    for (arma::uword k = 0; k < 8; ++k)
    {
        arma::Col<dtype> mixed = small;
        mixed[k] = (dtype)3.0e9; // one large lane
        quadriga_lib::fast_sincos(mixed, &s, &c);
        for (arma::uword i = 0; i < 8; ++i)
            if (i != k)
            {
                CHECK(s[i] == s0[i]);
                CHECK(c[i] == c0[i]);
            }
    }
}
