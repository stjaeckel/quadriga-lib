// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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
#include <catch2/catch_approx.hpp>           // Catch::Approx (Catch2 v3)
#include "quadriga_tools.hpp"                // declares quadriga_lib::sincos_approx
#include <armadillo>
#include <cmath>
#include <vector>

static inline float ref_sin(float x) { return std::sin(x); }
static inline float ref_cos(float x) { return std::cos(x); }

TEST_CASE("Fast Sine / Cosine computation (float only)")
{
    // Build a deterministic set of angles, incl. non-multiple-of-8 length
    constexpr float pi = 3.14159265358979323846f;
    std::vector<float> base = {
        0.0f,  pi/6,  pi/4,  pi/3,  pi/2,  2*pi/3,  3*pi/4,  5*pi/6,  pi,
       -pi/6, -pi/4, -pi/2, -pi,    2*pi,  -2*pi,   7.0f,   -7.0f,
        10.0f*pi, -10.0f*pi, 123.456f, -321.987f
    };

    const size_t target_len = 1003; // not divisible by 8
    base.reserve(target_len);
    for (size_t i = base.size(); i < target_len; ++i) {
        float t = static_cast<float>(i) * 0.03125f - 16.0f; // [-16, ~15]
        base.push_back(t * pi);
    }

    arma::Col<float> x(base.size());
    for (size_t i = 0; i < base.size(); ++i) x[i] = base[i];

    arma::Col<float> s, c;
    quadriga_lib::fast_sincos(x, &s, &c);

    REQUIRE(s.n_elem == x.n_elem);
    REQUIRE(c.n_elem == x.n_elem);

    // Tolerances for single-precision
    const float abs_tol    = 2e-6f;
    const float pythag_tol = 5e-6f;

    for (arma::uword i = 0; i < x.n_elem; ++i) {
        INFO("i=" << i << " x=" << x[i]);
        const float rs = ref_sin(x[i]);
        const float rc = ref_cos(x[i]);

        REQUIRE(s[i] == Catch::Approx(rs).margin(abs_tol));
        REQUIRE(c[i] == Catch::Approx(rc).margin(abs_tol));

        // sin^2 + cos^2 ≈ 1 (looser tolerance)
        const float unit = s[i]*s[i] + c[i]*c[i];
        REQUIRE(unit == Catch::Approx(1.0f).margin(pythag_tol));
    }
}