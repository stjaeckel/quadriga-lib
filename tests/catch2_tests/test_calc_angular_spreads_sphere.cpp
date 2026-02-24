// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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

TEST_CASE("calc_angular_spreads_sphere - Single path returns zero spread")
{
    arma::mat az(1, 1);
    az(0, 0) = 0.5;
    arma::mat el(1, 1);
    el(0, 0) = 0.3;
    arma::mat pw(1, 1);
    pw(0, 0) = 1.0;

    arma::vec as, es;
    arma::mat orient;

    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient);

    REQUIRE(as.n_elem == 1);
    REQUIRE(es.n_elem == 1);
    CHECK(as(0) < 1e-10);
    CHECK(es(0) < 1e-10);

    // Orientation should point at the single path
    CHECK(std::abs(orient(2, 0) - 0.5) < 1e-10); // heading ~ azimuth
    CHECK(std::abs(orient(1, 0) - 0.3) < 1e-10); // tilt ~ elevation
}

TEST_CASE("calc_angular_spreads_sphere - Two symmetric azimuth paths")
{
    // Two paths at +/- 0.1 rad azimuth, equal power, zero elevation
    arma::mat az(1, 2);
    az(0, 0) = 0.1;
    az(0, 1) = -0.1;
    arma::mat el(1, 2);
    el(0, 0) = 0.0;
    el(0, 1) = 0.0;
    arma::mat pw(1, 2);
    pw(0, 0) = 1.0;
    pw(0, 1) = 1.0;

    arma::vec as, es;
    arma::mat orient;

    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient);

    // Azimuth spread should be 0.1 rad (RMS of +/-0.1 with equal weights)
    CHECK(std::abs(as(0) - 0.1) < 1e-6);
    // Elevation spread should be ~0
    CHECK(es(0) < 1e-6);
    // Heading should be ~0 (mean of symmetric paths)
    CHECK(std::abs(orient(2, 0)) < 1e-6);
}

TEST_CASE("calc_angular_spreads_sphere - Two symmetric elevation paths")
{
    // Two paths at 0 azimuth, +/- 0.2 rad elevation
    arma::mat az(1, 2);
    az(0, 0) = 0.0;
    az(0, 1) = 0.0;
    arma::mat el(1, 2);
    el(0, 0) = 0.2;
    el(0, 1) = -0.2;
    arma::mat pw(1, 2);
    pw(0, 0) = 1.0;
    pw(0, 1) = 1.0;

    arma::vec as, es;
    arma::mat orient, phi, theta;

    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient, &phi, &theta);

    REQUIRE(phi.n_rows == 1);
    REQUIRE(phi.n_cols == 2);

    // After rotation, the elevation spread should become azimuth spread (bank rotation)
    // or vice versa. The important thing is that AS >= ES (bank angle maximizes AS).
    CHECK(as(0) >= es(0) - 1e-6);
    // Total angular extent should be preserved
    double total = std::sqrt(as(0) * as(0) + es(0) * es(0));
    CHECK(std::abs(total - 0.2) < 1e-4);
}

TEST_CASE("calc_angular_spreads_sphere - Pole paths (near zenith)")
{
    // 4 paths near the north pole, spread in azimuth — naive method would give large AS
    arma::mat az(1, 4);
    az(0, 0) = 0.0;
    az(0, 1) = 1.5708;  // pi/2
    az(0, 2) = 3.14159; // pi
    az(0, 3) = -1.5708; // -pi/2
    arma::mat el(1, 4);
    el(0, 0) = 1.4;
    el(0, 1) = 1.4;
    el(0, 2) = 1.4;
    el(0, 3) = 1.4;
    arma::mat pw(1, 4);
    pw(0, 0) = 1.0;
    pw(0, 1) = 1.0;
    pw(0, 2) = 1.0;
    pw(0, 3) = 1.0;

    arma::vec as, es;
    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es);

    // After spherical wrapping, the spreads should reflect the small solid angle
    // The actual angular extent on the sphere is small (~0.17 rad from pole)
    CHECK(as(0) < 0.5);
    CHECK(es(0) < 0.5);
}

TEST_CASE("calc_angular_spreads_sphere - Power broadcasting (single pow row)")
{
    arma::mat az(2, 3);
    az(0, 0) = 0.1;
    az(0, 1) = -0.1;
    az(0, 2) = 0.0;
    az(1, 0) = 0.2;
    az(1, 1) = -0.2;
    az(1, 2) = 0.0;
    arma::mat el(2, 3);
    el.zeros();
    arma::mat pw(1, 3);
    pw(0, 0) = 1.0;
    pw(0, 1) = 1.0;
    pw(0, 2) = 1.0;

    arma::vec as, es;
    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es);

    REQUIRE(as.n_elem == 2);
    // Second angle set has wider spread
    CHECK(as(1) > as(0));
}

TEST_CASE("calc_angular_spreads_sphere - No bank angle calculation")
{
    arma::mat az(1, 2);
    az(0, 0) = 0.0;
    az(0, 1) = 0.0;
    arma::mat el(1, 2);
    el(0, 0) = 0.2;
    el(0, 1) = -0.2;
    arma::mat pw(1, 2);
    pw(0, 0) = 1.0;
    pw(0, 1) = 1.0;

    arma::vec as_bank, es_bank, as_nobank, es_nobank;
    arma::mat orient_bank, orient_nobank;

    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as_bank, &es_bank, &orient_bank, (arma::mat *)nullptr, (arma::mat *)nullptr, true);
    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as_nobank, &es_nobank, &orient_nobank, (arma::mat *)nullptr, (arma::mat *)nullptr, false);

    // Without bank angle, bank should be 0
    CHECK(std::abs(orient_nobank(0, 0)) < 1e-10);
    // With bank angle, AS should be >= AS without bank angle
    CHECK(as_bank(0) >= as_nobank(0) - 1e-6);
}

TEST_CASE("calc_angular_spreads_sphere - Quantization groups nearby paths")
{
    // Two paths at 0.0 and 0.01 rad — within 3 deg quantization, they should merge
    arma::mat az(1, 2);
    az(0, 0) = 0.0;
    az(0, 1) = 0.01;
    arma::mat el(1, 2);
    el.zeros();
    arma::mat pw(1, 2);
    pw(0, 0) = 1.0;
    pw(0, 1) = 1.0;

    arma::vec as_raw, as_quant;

    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as_raw, (arma::vec *)nullptr,
                                              (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr, true, 0.0);
    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as_quant, (arma::vec *)nullptr,
                                              (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr, true, 3.0);

    // Quantized spread should be <= raw spread (paths merged)
    CHECK(as_quant(0) <= as_raw(0) + 1e-8);
}

TEST_CASE("calc_angular_spreads_sphere - Wrap-around at +/- pi")
{
    // Two paths straddling the +/- pi boundary
    arma::mat az(1, 2);
    az(0, 0) = 3.0;
    az(0, 1) = -3.0;
    arma::mat el(1, 2);
    el.zeros();
    arma::mat pw(1, 2);
    pw(0, 0) = 1.0;
    pw(0, 1) = 1.0;

    arma::vec as, es;
    arma::mat orient;

    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient);

    // The angular gap is about 2*(pi - 3.0) ~ 0.283 rad, so spread ~ 0.14
    double gap = 2.0 * (3.14159265358979 - 3.0);
    CHECK(as(0) < gap);
    CHECK(as(0) > 0.0);
}

TEST_CASE("calc_angular_spreads_sphere - Float precision")
{
    arma::fmat az(1, 3);
    az(0, 0) = 0.1f;
    az(0, 1) = -0.1f;
    az(0, 2) = 0.0f;
    arma::fmat el(1, 3);
    el.zeros();
    arma::fmat pw(1, 3);
    pw(0, 0) = 1.0f;
    pw(0, 1) = 1.0f;
    pw(0, 2) = 2.0f;

    arma::fvec as, es;
    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es);

    REQUIRE(as.n_elem == 1);
    CHECK(as(0) > 0.0f);
    CHECK(es(0) < 0.01f);
}

TEST_CASE("calc_angular_spreads_sphere - Multiple angle sets (n_ang > 1)")
{
    arma::mat az(3, 4);
    arma::mat el(3, 4);
    arma::mat pw(3, 4);
    el.zeros();
    pw.ones();

    // Set 0: narrow spread
    az(0, 0) = 0.01;
    az(0, 1) = -0.01;
    az(0, 2) = 0.005;
    az(0, 3) = -0.005;
    // Set 1: medium spread
    az(1, 0) = 0.1;
    az(1, 1) = -0.1;
    az(1, 2) = 0.05;
    az(1, 3) = -0.05;
    // Set 2: wide spread
    az(2, 0) = 1.0;
    az(2, 1) = -1.0;
    az(2, 2) = 0.5;
    az(2, 3) = -0.5;

    arma::vec as, es;
    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es);

    REQUIRE(as.n_elem == 3);
    CHECK(as(0) < as(1));
    CHECK(as(1) < as(2));
}

TEST_CASE("calc_angular_spreads_sphere - Rotated angles output shape and consistency")
{
    arma::mat az(1, 5);
    az(0, 0) = 0.3;
    az(0, 1) = -0.2;
    az(0, 2) = 0.1;
    az(0, 3) = -0.4;
    az(0, 4) = 0.0;
    arma::mat el(1, 5);
    el(0, 0) = 0.1;
    el(0, 1) = -0.1;
    el(0, 2) = 0.05;
    el(0, 3) = -0.05;
    el(0, 4) = 0.0;
    arma::mat pw(1, 5);
    pw.ones();

    arma::vec as, es;
    arma::mat orient, phi, theta;

    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient, &phi, &theta);

    // Shape checks
    REQUIRE(phi.n_rows == 1);
    REQUIRE(phi.n_cols == 5);
    REQUIRE(theta.n_rows == 1);
    REQUIRE(theta.n_cols == 5);
    REQUIRE(orient.n_rows == 3);
    REQUIRE(orient.n_cols == 1);

    // All rotated angles should be in valid ranges
    for (arma::uword i = 0; i < 5; i++)
    {
        CHECK(phi(0, i) >= -3.15);
        CHECK(phi(0, i) <= 3.15);
        CHECK(theta(0, i) >= -1.58);
        CHECK(theta(0, i) <= 1.58);
    }
}

TEST_CASE("calc_angular_spreads_sphere - Error on empty input")
{
    arma::mat az, el, pw;
    arma::vec as;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as),
                    std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - Error on mismatched columns")
{
    arma::mat az(1, 3);
    az.zeros();
    arma::mat el(1, 2);
    el.zeros();
    arma::mat pw(1, 3);
    pw.ones();
    arma::vec as;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as),
                    std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - Selective outputs (nullptr)")
{
    arma::mat az(1, 3);
    az(0, 0) = 0.1;
    az(0, 1) = -0.1;
    az(0, 2) = 0.0;
    arma::mat el(1, 3);
    el.zeros();
    arma::mat pw(1, 3);
    pw.ones();

    // Request only azimuth spread
    arma::vec as;
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, pw, &as, (arma::vec *)nullptr,
                                                      (arma::mat *)nullptr, (arma::mat *)nullptr,
                                                      (arma::mat *)nullptr, true, 0.0);
    CHECK(as(0) > 0.0);
}

TEST_CASE("calc_angular_spreads_sphere - Example")
{
    arma::mat az = {0.1, 0.2, -0.1, 0.3}; // 1 angle set, 4 paths
    arma::mat el = {0.0, 0.05, -0.05, 0.02};
    arma::mat pw = {1.0, 2.0, 1.5, 0.5};

    arma::vec as, es;
    arma::mat orient;
    quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient);
}