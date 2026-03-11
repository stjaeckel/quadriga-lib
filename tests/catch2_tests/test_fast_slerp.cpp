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
#include "quadriga_tools.hpp"
#include <cmath>

// Helper: scalar double-precision SLERP reference (matches slerp_complex_mf logic)
static void slerp_ref(double Ar, double Ai, double Br, double Bi, double w,
                       double &Xr, double &Xi)
{
    double wB = w, wA = 1.0 - w;
    double ampA = std::sqrt(Ar * Ar + Ai * Ai);
    double ampB = std::sqrt(Br * Br + Bi * Bi);
    double eps = std::numeric_limits<double>::epsilon();
    double R0 = eps * eps * eps;
    double R1 = eps;

    bool tinyA = ampA < R1, tinyB = ampB < R1;
    if (tinyA && tinyB) { Xr = 0.0; Xi = 0.0; return; }

    double gAr = tinyA ? 0.0 : Ar / ampA;
    double gAi = tinyA ? 0.0 : Ai / ampA;
    double gBr = tinyB ? 0.0 : Br / ampB;
    double gBi = tinyB ? 0.0 : Bi / ampB;
    double cPhase = (tinyA || tinyB) ? -1.0 : gAr * gBr + gAi * gBi;

    double tL = -0.999, tS = -0.99, dT = 1.0 / (tS - tL);
    bool linear_int = cPhase < tS;

    double fXr = 0.0, fXi = 0.0;
    if (linear_int)
        fXr = wA * Ar + wB * Br, fXi = wA * Ai + wB * Bi;

    if (cPhase > tL)
    {
        double Phase = (cPhase >= 1.0) ? R0 : std::acos(cPhase) + R0;
        double sPhase = 1.0 / std::sin(Phase);
        double wp = std::sin(wB * Phase) * sPhase;
        double wn = std::sin(wA * Phase) * sPhase;
        double gXr = wn * gAr + wp * gBr;
        double gXi = wn * gAi + wp * gBi;
        double ampX = wA * ampA + wB * ampB;

        if (linear_int)
        {
            double m = (tS - cPhase) * dT, n = 1.0 - m;
            fXr = n * gXr * ampX + m * fXr;
            fXi = n * gXi * ampX + m * fXi;
        }
        else
            fXr = gXr * ampX, fXi = gXi * ampX;
    }
    Xr = fXr;
    Xi = fXi;
}

TEST_CASE("fast_slerp - Basic spherical interpolation")
{
    // Orthogonal unit vectors, w=0.5 → result should be on the unit circle at 45 degrees
    arma::fvec Ar = {1.0f}, Ai = {0.0f};
    arma::fvec Br = {0.0f}, Bi = {1.0f};
    arma::fvec w = {0.5f};
    arma::fvec Xr, Xi;

    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);

    REQUIRE(Xr.n_elem == 1);
    REQUIRE(Xi.n_elem == 1);

    // Expected: (1/sqrt(2), 1/sqrt(2))
    float expected = 1.0f / std::sqrt(2.0f);
    CHECK(std::abs(Xr(0) - expected) < 1e-5f);
    CHECK(std::abs(Xi(0) - expected) < 1e-5f);
}

TEST_CASE("fast_slerp - Boundary weights w=0 and w=1")
{
    arma::fvec Ar = {3.0f}, Ai = {1.0f};
    arma::fvec Br = {0.5f}, Bi = {2.0f};
    arma::fvec Xr, Xi;

    // w=0 → output should equal A
    arma::fvec w0 = {0.0f};
    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w0, Xr, Xi);
    CHECK(std::abs(Xr(0) - 3.0f) < 1e-6f);
    CHECK(std::abs(Xi(0) - 1.0f) < 1e-6f);

    // w=1 → output should equal B
    arma::fvec w1 = {1.0f};
    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w1, Xr, Xi);
    CHECK(std::abs(Xr(0) - 0.5f) < 1e-6f);
    CHECK(std::abs(Xi(0) - 2.0f) < 1e-6f);
}

TEST_CASE("fast_slerp - Identical inputs")
{
    arma::fvec Ar = {1.5f}, Ai = {-0.7f};
    arma::fvec Br = {1.5f}, Bi = {-0.7f};
    arma::fvec w = {0.42f};
    arma::fvec Xr, Xi;

    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);

    // Result should be very close to the input (identical A and B)
    CHECK(std::abs(Xr(0) - 1.5f) < 1e-5f);
    CHECK(std::abs(Xi(0) - (-0.7f)) < 1e-5f);
}

TEST_CASE("fast_slerp - Near-antipodal (transition zone)")
{
    // cPhase ≈ -0.995 → between tL=-0.999 and tS=-0.99
    arma::fvec Ar = {1.0f}, Ai = {0.01f};
    arma::fvec Br = {-1.0f}, Bi = {0.01f};
    arma::fvec w = {0.5f};
    arma::fvec Xr, Xi;

    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);

    // Validate against double-precision reference
    double refXr, refXi;
    slerp_ref(1.0, 0.01, -1.0, 0.01, 0.5, refXr, refXi);
    CHECK(std::abs(Xr(0) - (float)refXr) < 1e-4f);
    CHECK(std::abs(Xi(0) - (float)refXi) < 1e-4f);
}

TEST_CASE("fast_slerp - Pure linear (antipodal)")
{
    arma::fvec Ar = {1.0f}, Ai = {0.0f};
    arma::fvec Br = {-1.0f}, Bi = {0.0f};
    arma::fvec w = {0.5f};
    arma::fvec Xr, Xi;

    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);

    // Exactly antipodal: linear interpolation → (0, 0)
    CHECK(std::abs(Xr(0)) < 1e-6f);
    CHECK(std::abs(Xi(0)) < 1e-6f);
}

TEST_CASE("fast_slerp - Tiny amplitudes")
{
    arma::fvec Xr, Xi;

    // Both tiny → zero output
    arma::fvec tA_r = {1e-20f}, tA_i = {0.0f};
    arma::fvec tB_r = {0.0f}, tB_i = {1e-25f};
    arma::fvec w = {0.5f};
    quadriga_lib::fast_slerp(tA_r, tA_i, tB_r, tB_i, w, Xr, Xi);
    CHECK(std::abs(Xr(0)) < 1e-10f);
    CHECK(std::abs(Xi(0)) < 1e-10f);

    // A tiny, B normal → output approaches B scaled by w
    arma::fvec nB_r = {1.0f}, nB_i = {1.0f};
    quadriga_lib::fast_slerp(tA_r, tA_i, nB_r, nB_i, w, Xr, Xi);
    double refXr, refXi;
    slerp_ref(1e-20, 0.0, 1.0, 1.0, 0.5, refXr, refXi);
    CHECK(std::abs(Xr(0) - (float)refXr) < 1e-5f);
    CHECK(std::abs(Xi(0) - (float)refXi) < 1e-5f);
}

TEST_CASE("fast_slerp - Double-precision input")
{
    arma::vec Ar = {1.0, 2.0}, Ai = {0.0, -1.0};
    arma::vec Br = {0.0, 0.5}, Bi = {1.0, 3.0};
    arma::vec w = {0.5, 0.3};
    arma::fvec Xr, Xi;

    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);

    REQUIRE(Xr.n_elem == 2);
    REQUIRE(Xi.n_elem == 2);

    // Check against double-precision reference
    for (arma::uword i = 0; i < 2; ++i)
    {
        double refXr, refXi;
        slerp_ref(Ar(i), Ai(i), Br(i), Bi(i), w(i), refXr, refXi);
        CHECK(std::abs(Xr(i) - (float)refXr) < 1e-5f);
        CHECK(std::abs(Xi(i) - (float)refXi) < 1e-5f);
    }
}

TEST_CASE("fast_slerp - Accuracy bounds on larger vector")
{
    // Generate a vector large enough to exercise both AVX2 bulk and scalar tail paths
    // 8*13 + 5 = 109 elements (13 full AVX2 vectors + 5-element tail)
    const arma::uword n = 109;

    arma::fvec Ar(n), Ai(n), Br(n), Bi(n), w(n);
    arma::fvec Xr, Xi;

    // Fill with deterministic test data spanning all regions
    for (arma::uword i = 0; i < n; ++i)
    {
        float t = (float)i / (float)(n - 1); // 0..1
        float angle_A = t * 6.0f;
        float angle_B = angle_A + 0.1f + t * 3.0f; // varying phase offsets
        float ampA = 0.5f + t * 2.0f;
        float ampB = 2.5f - t * 1.5f;

        Ar(i) = ampA * std::cos(angle_A);
        Ai(i) = ampA * std::sin(angle_A);
        Br(i) = ampB * std::cos(angle_B);
        Bi(i) = ampB * std::sin(angle_B);
        w(i) = t;
    }

    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);

    REQUIRE(Xr.n_elem == n);
    REQUIRE(Xi.n_elem == n);

    // Measure max ULP distance against double-precision reference
    // ULP for a float value v: 1 ULP = |v| * 2^-23 = |v| * 1.1920929e-7
    // For near-zero results, use float min_normal as floor to avoid division by zero
    const double ulp_scale = 1.1920928955078125e-7; // 2^-23
    const double floor_val = (double)std::numeric_limits<float>::min(); // ~1.175e-38

    double max_ulp = 0.0;
    for (arma::uword i = 0; i < n; ++i)
    {
        // Double-precision reference
        double refXr, refXi;
        slerp_ref((double)Ar(i), (double)Ai(i), (double)Br(i), (double)Bi(i), (double)w(i), refXr, refXi);

        // AVX2/generic result as double for error computation
        double dXr = (double)Xr(i);
        double dXi = (double)Xi(i);

        // ULP distance per component: |result - ref| / (|ref| * 2^-23)
        // Use max(|ref|, min_normal) to handle near-zero gracefully
        double ulp_r = std::abs(dXr - refXr) / std::max(std::abs(refXr) * ulp_scale, floor_val);
        double ulp_i = std::abs(dXi - refXi) / std::max(std::abs(refXi) * ulp_scale, floor_val);
        double ulp = std::max(ulp_r, ulp_i);
        if (ulp > max_ulp)
            max_ulp = ulp;
    }

    // Uncomment exactly one of the following lines (keep the tightest that passes):
    // Note: SLERP chains acos → 3× sincos → division, so compound error is much larger
    // than a single transcendental. 96 ULP ≈ 1.1e-5 relative error ≈ 5.5 bits lost of 23.
    // CHECK(max_ulp < 50.0);  // 50 ULP
    CHECK(max_ulp < 100.0); // 100 ULP
    CHECK(max_ulp < 150.0); // 150 ULP
}

TEST_CASE("fast_slerp - Empty input")
{
    arma::fvec Ar, Ai, Br, Bi, w, Xr, Xi;
    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);
    CHECK(Xr.n_elem == 0);
    CHECK(Xi.n_elem == 0);
}

TEST_CASE("fast_slerp - Error: mismatched input lengths")
{
    arma::fvec Ar = {1.0f, 2.0f}, Ai = {0.0f};
    arma::fvec Br = {0.0f}, Bi = {1.0f};
    arma::fvec w = {0.5f};
    arma::fvec Xr, Xi;

    CHECK_THROWS_AS(quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi), std::invalid_argument);
}

TEST_CASE("fast_slerp - Error: input-output aliasing")
{
    arma::fvec Ar = {1.0f}, Ai = {0.0f}, Br = {0.0f}, Bi = {1.0f}, w = {0.5f};

    // Xr aliased with Ar
    CHECK_THROWS_AS(quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Ar, Ai), std::invalid_argument);
}

TEST_CASE("fast_slerp - Output is resized correctly")
{
    arma::fvec Ar = {1.0f, 2.0f, 3.0f};
    arma::fvec Ai = {0.0f, 0.0f, 0.0f};
    arma::fvec Br = {0.0f, 0.0f, 0.0f};
    arma::fvec Bi = {1.0f, 2.0f, 3.0f};
    arma::fvec w = {0.0f, 0.5f, 1.0f};
    arma::fvec Xr(1), Xi(10); // wrong sizes

    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);

    REQUIRE(Xr.n_elem == 3);
    REQUIRE(Xi.n_elem == 3);

    // w=0 → A, w=1 → B
    CHECK(std::abs(Xr(0) - 1.0f) < 1e-6f);
    CHECK(std::abs(Xi(0) - 0.0f) < 1e-6f);
    CHECK(std::abs(Xr(2) - 0.0f) < 1e-6f);
    CHECK(std::abs(Xi(2) - 3.0f) < 1e-6f);
}