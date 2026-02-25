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

TEST_CASE("calc_rician_k_factor - Basic functionality")
{
    // Single snapshot: 3 paths, TX at origin, RX at (10, 0, 0) => dTR = 10.0
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0, 0.5, 0.25};
    path_length[0] = {10.0, 11.0, 12.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf, pg;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, &pg, 0.01);

    REQUIRE(kf.n_elem == 1);
    REQUIRE(pg.n_elem == 1);

    // LOS power = 1.0 (path at 10.0 <= 10.01), NLOS power = 0.5 + 0.25 = 0.75
    CHECK(std::abs(kf(0) - 1.0 / 0.75) < 1e-10);
    CHECK(std::abs(pg(0) - 1.75) < 1e-10);
}

TEST_CASE("calc_rician_k_factor - Multiple paths within LOS window")
{
    // Two paths within LOS window
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {2.0, 1.0, 0.5};
    path_length[0] = {10.0, 10.005, 15.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf, pg;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, &pg, 0.01);

    // LOS power = 2.0 + 1.0 = 3.0 (both paths <= 10.01), NLOS = 0.5
    CHECK(std::abs(kf(0) - 3.0 / 0.5) < 1e-10);
    CHECK(std::abs(pg(0) - 3.5) < 1e-10);
}

TEST_CASE("calc_rician_k_factor - No NLOS paths (infinite K-Factor)")
{
    // All paths within LOS window
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0, 0.5};
    path_length[0] = {10.0, 10.005};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 0.01);

    CHECK(std::isinf(kf(0)));
    CHECK(kf(0) > 0.0); // Positive infinity
}

TEST_CASE("calc_rician_k_factor - No LOS paths (zero K-Factor)")
{
    // All paths beyond LOS window
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0, 0.5};
    path_length[0] = {11.0, 12.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf, pg;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, &pg, 0.01);

    CHECK(kf(0) == 0.0);
    CHECK(std::abs(pg(0) - 1.5) < 1e-10);
}

TEST_CASE("calc_rician_k_factor - Multiple snapshots with mobile RX")
{
    // 2 snapshots, fixed TX, mobile RX
    std::vector<arma::vec> powers(2), path_length(2);
    powers[0] = {1.0, 0.5};
    path_length[0] = {10.0, 12.0};
    powers[1] = {2.0, 1.0};
    path_length[1] = {20.0, 25.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};

    arma::mat rx_pos(3, 2);
    rx_pos.col(0) = {10.0, 0.0, 0.0}; // dTR = 10.0
    rx_pos.col(1) = {20.0, 0.0, 0.0}; // dTR = 20.0

    arma::vec kf, pg;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, &pg, 0.01);

    REQUIRE(kf.n_elem == 2);
    REQUIRE(pg.n_elem == 2);

    CHECK(std::abs(kf(0) - 1.0 / 0.5) < 1e-10);
    CHECK(std::abs(kf(1) - 2.0 / 1.0) < 1e-10);
    CHECK(std::abs(pg(0) - 1.5) < 1e-10);
    CHECK(std::abs(pg(1) - 3.0) < 1e-10);
}

TEST_CASE("calc_rician_k_factor - 3D positions")
{
    // TX and RX in 3D space
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0, 0.5};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {3.0, 4.0, 0.0}; // dTR = 5.0

    path_length[0] = {5.0, 8.0};

    arma::vec kf;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 0.01);

    CHECK(std::abs(kf(0) - 1.0 / 0.5) < 1e-10);
}

TEST_CASE("calc_rician_k_factor - Float precision")
{
    std::vector<arma::fvec> powers(1), path_length(1);
    powers[0] = {1.0f, 0.5f, 0.25f};
    path_length[0] = {10.0f, 11.0f, 12.0f};

    arma::fmat tx_pos(3, 1);
    tx_pos.col(0) = {0.0f, 0.0f, 0.0f};
    arma::fmat rx_pos(3, 1);
    rx_pos.col(0) = {10.0f, 0.0f, 0.0f};

    arma::fvec kf, pg;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, &pg, 0.01f);

    CHECK(std::abs(kf(0) - 1.0f / 0.75f) < 1e-5f);
    CHECK(std::abs(pg(0) - 1.75f) < 1e-5f);
}

TEST_CASE("calc_rician_k_factor - Only KF requested")
{
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0, 0.5};
    path_length[0] = {10.0, 12.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 0.01);

    REQUIRE(kf.n_elem == 1);
    CHECK(std::abs(kf(0) - 1.0 / 0.5) < 1e-10);
}

TEST_CASE("calc_rician_k_factor - Only PG requested")
{
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0, 0.5};
    path_length[0] = {10.0, 12.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec pg;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, (arma::Col<double> *)nullptr, &pg, 0.01);

    REQUIRE(pg.n_elem == 1);
    CHECK(std::abs(pg(0) - 1.5) < 1e-10);
}

TEST_CASE("calc_rician_k_factor - Custom window size")
{
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0, 0.5, 0.25};
    path_length[0] = {10.0, 10.5, 12.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf;

    // With window_size = 0.01: only first path is LOS
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 0.01);
    CHECK(std::abs(kf(0) - 1.0 / 0.75) < 1e-10);

    // With window_size = 1.0: first two paths are LOS
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 1.0);
    CHECK(std::abs(kf(0) - 1.5 / 0.25) < 1e-10);
}

TEST_CASE("calc_rician_k_factor - Empty and zero-power snapshot")
{
    // Snapshot with zero-length paths
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0].set_size(0);
    path_length[0].set_size(0);

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf, pg;
    quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, &pg, 0.01);

    // No paths: both LOS and NLOS power = 0 => KF = 0, PG = 0
    CHECK(kf(0) == 0.0);
    CHECK(pg(0) == 0.0);
}

// --- Error handling tests ---

TEST_CASE("calc_rician_k_factor - Error: empty powers vector")
{
    std::vector<arma::vec> powers, path_length;
    arma::mat tx_pos(3, 1), rx_pos(3, 1);

    CHECK_THROWS_AS(
        quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos,
                                           (arma::Col<double> *)nullptr, (arma::Col<double> *)nullptr, 0.01),
        std::invalid_argument);
}

TEST_CASE("calc_rician_k_factor - Error: mismatched powers and path_length sizes")
{
    std::vector<arma::vec> powers(2), path_length(1);
    powers[0] = {1.0};
    powers[1] = {2.0};
    path_length[0] = {10.0};

    arma::mat tx_pos(3, 1), rx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf;
    CHECK_THROWS_AS(
        quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 0.01),
        std::invalid_argument);
}

TEST_CASE("calc_rician_k_factor - Error: wrong tx_pos shape")
{
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0};
    path_length[0] = {10.0};

    arma::mat tx_pos(2, 1); // Wrong: should be 3 rows
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf;
    CHECK_THROWS_AS(
        quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 0.01),
        std::invalid_argument);
}

TEST_CASE("calc_rician_k_factor - Error: wrong rx_pos columns")
{
    std::vector<arma::vec> powers(2), path_length(2);
    powers[0] = {1.0};
    powers[1] = {2.0};
    path_length[0] = {10.0};
    path_length[1] = {20.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 3); // Wrong: should be 1 or 2 columns

    arma::vec kf;
    CHECK_THROWS_AS(
        quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 0.01),
        std::invalid_argument);
}

TEST_CASE("calc_rician_k_factor - Error: mismatched element lengths")
{
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0, 0.5};
    path_length[0] = {10.0}; // Mismatch: 2 vs 1

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf;
    CHECK_THROWS_AS(
        quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, 0.01),
        std::invalid_argument);
}

TEST_CASE("calc_rician_k_factor - Error: negative window_size")
{
    std::vector<arma::vec> powers(1), path_length(1);
    powers[0] = {1.0};
    path_length[0] = {10.0};

    arma::mat tx_pos(3, 1);
    tx_pos.col(0) = {0.0, 0.0, 0.0};
    arma::mat rx_pos(3, 1);
    rx_pos.col(0) = {10.0, 0.0, 0.0};

    arma::vec kf;
    CHECK_THROWS_AS(
        quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, (arma::Col<double> *)nullptr, -0.01),
        std::invalid_argument);
}
