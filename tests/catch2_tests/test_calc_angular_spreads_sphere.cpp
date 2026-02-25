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
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.5};
    el[0] = {0.3};
    powers[0] = {1.0};

    arma::vec as, es;
    arma::mat orient;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es, &orient);

    REQUIRE(as.n_elem == 1);
    REQUIRE(es.n_elem == 1);
    CHECK(as(0) < 1e-10);
    CHECK(es(0) < 1e-10);
    CHECK(std::abs(orient(2, 0) - 0.5) < 1e-10);  // heading ~ azimuth
    CHECK(std::abs(orient(1, 0) - 0.3) < 1e-10);  // tilt ~ elevation
}

TEST_CASE("calc_angular_spreads_sphere - Two symmetric azimuth paths")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.1, -0.1};
    el[0] = {0.0, 0.0};
    powers[0] = {1.0, 1.0};

    arma::vec as, es;
    arma::mat orient;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es, &orient);

    CHECK(std::abs(as(0) - 0.1) < 1e-6);
    CHECK(es(0) < 1e-6);
    CHECK(std::abs(orient(2, 0)) < 1e-6);
}

TEST_CASE("calc_angular_spreads_sphere - Two symmetric elevation paths")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.0, 0.0};
    el[0] = {0.2, -0.2};
    powers[0] = {1.0, 1.0};

    arma::vec as, es;
    arma::mat orient;
    std::vector<arma::vec> phi, theta;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es, &orient, &phi, &theta);

    REQUIRE(phi.size() == 1);
    REQUIRE(phi[0].n_elem == 2);

    // After bank rotation, AS >= ES
    CHECK(as(0) >= es(0) - 1e-6);
    // Total angular extent preserved
    double total = std::sqrt(as(0) * as(0) + es(0) * es(0));
    CHECK(std::abs(total - 0.2) < 1e-4);
}

TEST_CASE("calc_angular_spreads_sphere - Pole paths (near zenith)")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.0, 1.5707963, 3.1415927, -1.5707963};
    el[0] = {1.4, 1.4, 1.4, 1.4};
    powers[0] = {1.0, 1.0, 1.0, 1.0};

    arma::vec as, es;
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es);

    CHECK(as(0) < 0.5);
    CHECK(es(0) < 0.5);
}

TEST_CASE("calc_angular_spreads_sphere - Variable path counts per CIR")
{
    std::vector<arma::vec> az(3), el(3), powers(3);
    az[0] = {0.1, -0.1};                    // 2 paths
    el[0] = {0.0, 0.0};
    powers[0] = {1.0, 1.0};
    az[1] = {0.2, -0.2, 0.0};              // 3 paths
    el[1] = {0.0, 0.0, 0.0};
    powers[1] = {1.0, 1.0, 1.0};
    az[2] = {1.0, -1.0, 0.5, -0.5, 0.0};  // 5 paths
    el[2] = {0.0, 0.0, 0.0, 0.0, 0.0};
    powers[2] = {1.0, 1.0, 1.0, 1.0, 1.0};

    arma::vec as, es;
    std::vector<arma::vec> phi, theta;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es,
                                              (arma::mat *)nullptr, &phi, &theta);

    REQUIRE(as.n_elem == 3);
    REQUIRE(phi.size() == 3);
    CHECK(phi[0].n_elem == 2);
    CHECK(phi[1].n_elem == 3);
    CHECK(phi[2].n_elem == 5);
    CHECK(as(0) < as(1));
    CHECK(as(1) < as(2));
}

TEST_CASE("calc_angular_spreads_sphere - No bank angle calculation")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.0, 0.0};
    el[0] = {0.2, -0.2};
    powers[0] = {1.0, 1.0};

    arma::vec as_bank, es_bank, as_nobank, es_nobank;
    arma::mat orient_bank, orient_nobank;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_bank, &es_bank, &orient_bank,
                                              (std::vector<arma::vec> *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              false, true);
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_nobank, &es_nobank, &orient_nobank,
                                              (std::vector<arma::vec> *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              false, false);

    CHECK(std::abs(orient_nobank(0, 0)) < 1e-10); // bank = 0 when disabled
    CHECK(as_bank(0) >= as_nobank(0) - 1e-6);
}

TEST_CASE("calc_angular_spreads_sphere - Disable wrapping")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.1, -0.1, 0.05};
    el[0] = {0.2, -0.2, 0.0};
    powers[0] = {1.0, 1.0, 1.0};

    arma::vec as_wrap, es_wrap, as_raw, es_raw;
    arma::mat orient_raw;
    std::vector<arma::vec> phi_raw, theta_raw;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_wrap, &es_wrap);
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_raw, &es_raw, &orient_raw,
                                              &phi_raw, &theta_raw, true);

    // Orientation should be zero when wrapping is disabled
    CHECK(std::abs(orient_raw(0, 0)) < 1e-10);
    CHECK(std::abs(orient_raw(1, 0)) < 1e-10);
    CHECK(std::abs(orient_raw(2, 0)) < 1e-10);

    // phi/theta should equal the input az/el
    REQUIRE(phi_raw.size() == 1);
    CHECK(arma::approx_equal(phi_raw[0], az[0], "absdiff", 1e-14));
    CHECK(arma::approx_equal(theta_raw[0], el[0], "absdiff", 1e-14));
}

TEST_CASE("calc_angular_spreads_sphere - Quantization groups nearby paths")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.0, 0.01};
    el[0] = {0.0, 0.0};
    powers[0] = {1.0, 1.0};

    arma::vec as_raw, as_quant;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_raw,
                                              (arma::vec *)nullptr, (arma::mat *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              false, true, 0.0);
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_quant,
                                              (arma::vec *)nullptr, (arma::mat *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              false, true, 3.0);

    CHECK(as_quant(0) <= as_raw(0) + 1e-8);
}

TEST_CASE("calc_angular_spreads_sphere - Wrap-around at +/- pi")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {3.0, -3.0};
    el[0] = {0.0, 0.0};
    powers[0] = {1.0, 1.0};

    arma::vec as, es;
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es);

    double gap = 2.0 * (3.14159265358979 - 3.0);
    CHECK(as(0) < gap);
    CHECK(as(0) > 0.0);
}

TEST_CASE("calc_angular_spreads_sphere - Float precision")
{
    std::vector<arma::fvec> az(1), el(1), powers(1);
    az[0] = {0.1f, -0.1f, 0.0f};
    el[0] = {0.0f, 0.0f, 0.0f};
    powers[0] = {1.0f, 1.0f, 2.0f};

    arma::fvec as, es;
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es);

    REQUIRE(as.n_elem == 1);
    CHECK(as(0) > 0.0f);
    CHECK(es(0) < 0.01f);
}

TEST_CASE("calc_angular_spreads_sphere - Output shapes and valid ranges")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.3, -0.2, 0.1, -0.4, 0.0};
    el[0] = {0.1, -0.1, 0.05, -0.05, 0.0};
    powers[0] = {1.0, 1.0, 1.0, 1.0, 1.0};

    arma::vec as, es;
    arma::mat orient;
    std::vector<arma::vec> phi, theta;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es, &orient, &phi, &theta);

    REQUIRE(as.n_elem == 1);
    REQUIRE(es.n_elem == 1);
    REQUIRE(orient.n_rows == 3);
    REQUIRE(orient.n_cols == 1);
    REQUIRE(phi.size() == 1);
    REQUIRE(phi[0].n_elem == 5);
    REQUIRE(theta.size() == 1);
    REQUIRE(theta[0].n_elem == 5);

    for (arma::uword i = 0; i < 5; i++)
    {
        CHECK(phi[0](i) >= -3.15);
        CHECK(phi[0](i) <= 3.15);
        CHECK(theta[0](i) >= -1.58);
        CHECK(theta[0](i) <= 1.58);
    }
}

TEST_CASE("calc_angular_spreads_sphere - Error on empty input")
{
    std::vector<arma::vec> az, el, powers;
    arma::vec as;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as),
                    std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - Error on mismatched vector sizes")
{
    std::vector<arma::vec> az(2), el(1), powers(2);
    az[0] = {0.1};
    az[1] = {0.2};
    el[0] = {0.0};
    powers[0] = {1.0};
    powers[1] = {1.0};

    arma::vec as;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as),
                    std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - Error on mismatched path count within CIR")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.1, 0.2};
    el[0] = {0.0};         // Length mismatch
    powers[0] = {1.0, 1.0};

    arma::vec as;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as),
                    std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - Selective outputs (nullptr)")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.1, -0.1, 0.0};
    el[0] = {0.0, 0.0, 0.0};
    powers[0] = {1.0, 1.0, 1.0};

    arma::vec as;
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as,
                                                      (arma::vec *)nullptr,
                                                      (arma::mat *)nullptr,
                                                      (std::vector<arma::vec> *)nullptr,
                                                      (std::vector<arma::vec> *)nullptr,
                                                      false, true, 0.0);
    CHECK(as(0) > 0.0);
}
