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

TEST_CASE("calc_delay_spread - Basic single CIR")
{
    // Single CIR with 3 paths: delays = [0, 1e-6, 2e-6], powers = [1.0, 0.5, 0.25]
    arma::vec tau = {0.0, 1e-6, 2e-6};
    arma::vec pow = {1.0, 0.5, 0.25};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);

    REQUIRE(ds.n_elem == 1);
    REQUIRE(mean_delay.n_elem == 1);

    // Expected mean delay: (1.0*0 + 0.5*1e-6 + 0.25*2e-6) / (1.0+0.5+0.25) = 1e-6 / 1.75
    double expected_mean = 1.0e-6 / 1.75;
    CHECK(std::abs(mean_delay(0) - expected_mean) < 1e-14);

    // Verify delay spread is positive and reasonable
    CHECK(ds(0) > 0.0);
    CHECK(ds(0) < 2e-6); // Must be less than max delay spread
}

TEST_CASE("calc_delay_spread - Single path yields zero spread")
{
    arma::vec tau = {5e-6};
    arma::vec pow = {1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);

    REQUIRE(ds.n_elem == 1);
    CHECK(std::abs(ds(0)) < 1e-20);
    CHECK(std::abs(mean_delay(0) - 5e-6) < 1e-20);
}

TEST_CASE("calc_delay_spread - Equal power two paths")
{
    // Two equal-power paths: DS should be half the delay difference
    arma::vec tau = {0.0, 2e-6};
    arma::vec pow = {1.0, 1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);

    REQUIRE(ds.n_elem == 1);

    // Mean delay = 1e-6
    CHECK(std::abs(mean_delay(0) - 1e-6) < 1e-14);

    // DS = sqrt( 0.5*(0-1e-6)^2 + 0.5*(2e-6-1e-6)^2 ) = 1e-6
    CHECK(std::abs(ds(0) - 1e-6) < 1e-14);
}

TEST_CASE("calc_delay_spread - Multiple CIRs")
{
    arma::vec tau1 = {0.0, 1e-6};
    arma::vec pow1 = {1.0, 1.0};

    arma::vec tau2 = {0.0, 1e-6, 2e-6};
    arma::vec pow2 = {1.0, 1.0, 1.0};

    std::vector<arma::vec> delays = {tau1, tau2};
    std::vector<arma::vec> powers = {pow1, pow2};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);

    REQUIRE(ds.n_elem == 2);
    REQUIRE(mean_delay.n_elem == 2);

    // CIR 1: mean = 0.5e-6, DS = 0.5e-6
    CHECK(std::abs(mean_delay(0) - 0.5e-6) < 1e-14);
    CHECK(std::abs(ds(0) - 0.5e-6) < 1e-14);

    // CIR 2: mean = 1e-6
    CHECK(std::abs(mean_delay(1) - 1e-6) < 1e-14);
    CHECK(ds(1) > 0.0);
}

TEST_CASE("calc_delay_spread - Threshold filters weak paths")
{
    // Path 1: 1.0 W (0 dB), Path 2: 0.001 W (-30 dB), Path 3: 0.5 W (-3 dB)
    arma::vec tau = {0.0, 10e-6, 1e-6};
    arma::vec pow = {1.0, 0.001, 0.5};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    // With 20 dB threshold, path 2 (-30 dB) should be excluded
    arma::vec ds_20 = quadriga_lib::calc_delay_spread(delays, powers, 20.0);

    // With 100 dB threshold, all paths included
    arma::vec ds_all = quadriga_lib::calc_delay_spread(delays, powers, 100.0);

    // DS with all paths should be larger (path at 10e-6 increases spread)
    CHECK(ds_all(0) > ds_20(0));
}

TEST_CASE("calc_delay_spread - Granularity bins paths")
{
    // Two paths very close together at ~100 ns and ~110 ns, one path at 1000 ns
    arma::vec tau = {100e-9, 110e-9, 1000e-9};
    arma::vec pow = {1.0, 1.0, 1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    // Without granularity
    arma::vec ds_no_gran = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0);

    // With 50 ns granularity - first two paths should be binned together
    arma::vec ds_gran = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 50e-9);

    // Both should produce positive delay spreads
    CHECK(ds_no_gran(0) > 0.0);
    CHECK(ds_gran(0) > 0.0);
}

TEST_CASE("calc_delay_spread - Without mean_delay output")
{
    arma::vec tau = {0.0, 1e-6};
    arma::vec pow = {1.0, 1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    // Call without mean_delay pointer
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers);

    REQUIRE(ds.n_elem == 1);
    CHECK(std::abs(ds(0) - 0.5e-6) < 1e-14);
}

TEST_CASE("calc_delay_spread - Float precision")
{
    arma::fvec tau = {0.0f, 1e-6f, 2e-6f};
    arma::fvec pow = {1.0f, 1.0f, 1.0f};

    std::vector<arma::fvec> delays = {tau};
    std::vector<arma::fvec> powers = {pow};

    arma::fvec mean_delay;
    arma::fvec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0f, 0.0f, &mean_delay);

    REQUIRE(ds.n_elem == 1);
    CHECK(std::abs(mean_delay(0) - 1e-6f) < 1e-10f);
    CHECK(ds(0) > 0.0f);
}

TEST_CASE("calc_delay_spread - Input validation")
{
    std::vector<arma::vec> empty_delays;
    std::vector<arma::vec> powers = {{1.0}};

    // Empty delays
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(empty_delays, powers), std::invalid_argument);

    // Mismatched sizes
    std::vector<arma::vec> delays = {{0.0}, {1e-6}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(delays, powers), std::invalid_argument);

    // Mismatched path counts
    std::vector<arma::vec> delays2 = {{0.0, 1e-6}};
    std::vector<arma::vec> powers2 = {{1.0}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(delays2, powers2), std::invalid_argument);

    // Negative threshold
    std::vector<arma::vec> delays3 = {{0.0}};
    std::vector<arma::vec> powers3 = {{1.0}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(delays3, powers3, -10.0), std::invalid_argument);

    // Negative granularity
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(delays3, powers3, 100.0, -1.0), std::invalid_argument);
}

TEST_CASE("calc_delay_spread - Granularity with mean_delay output")
{
    arma::vec tau = {0.0, 50e-9, 500e-9, 550e-9};
    arma::vec pow = {1.0, 1.0, 1.0, 1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 100e-9, &mean_delay);

    REQUIRE(ds.n_elem == 1);
    REQUIRE(mean_delay.n_elem == 1);

    // With 100 ns granularity, paths at 0/50ns -> bin 0, paths at 500/550ns -> bin 500ns
    // Two bins with equal total power -> mean delay ≈ 250 ns
    CHECK(mean_delay(0) > 0.0);
    CHECK(ds(0) > 0.0);
}
