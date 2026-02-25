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
#include <vector>

TEST_CASE("calc_cross_polarization_ratio - Basic NLOS XPR")
{
    // Single CIR with 3 paths: 1 LOS + 2 NLOS
    // TX at origin, RX at (10, 0, 0) => dTR = 10.0
    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({10.0, 0.0, 0.0});

    // Path powers
    arma::vec pw(3);
    pw(0) = 1.0;  // LOS
    pw(1) = 0.5;  // NLOS
    pw(2) = 0.5;  // NLOS

    // Polarization matrices [8, n_path]
    // LOS: M = {{1,0},{0,-1}} => purely diagonal
    arma::mat M(8, 3);
    M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}); // LOS

    // NLOS path 1: some cross-pol leakage
    // M_vv=0.9, M_hv=0.1, M_vh=0.1, M_hh=0.8 (all real)
    M.col(1) = arma::vec({0.9, 0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0});

    // NLOS path 2: more cross-pol
    // M_vv=0.7, M_hv=0.3, M_vh=0.2, M_hh=0.6 (all real)
    M.col(2) = arma::vec({0.7, 0.0, 0.3, 0.0, 0.2, 0.0, 0.6, 0.0});

    // Path lengths: LOS = 10.0, NLOS = 12.0, 15.0
    arma::vec pl(3);
    pl(0) = 10.0;
    pl(1) = 12.0;
    pl(2) = 15.0;

    std::vector<arma::vec> powers_vec = {pw};
    std::vector<arma::mat> M_vec = {M};
    std::vector<arma::vec> pl_vec = {pl};

    arma::mat xpr;
    arma::vec pg;

    quadriga_lib::calc_cross_polarization_ratio(powers_vec, M_vec, pl_vec, tx, rx, &xpr, &pg);

    REQUIRE(xpr.n_rows == 1);
    REQUIRE(xpr.n_cols == 6);
    REQUIRE(pg.n_elem == 1);

    // pg includes ALL paths (including LOS)
    // LOS: 1.0 * (1 + 0 + 0 + 1) = 2.0
    // NLOS1: 0.5 * (0.81 + 0.01 + 0.01 + 0.64) = 0.5 * 1.47 = 0.735
    // NLOS2: 0.5 * (0.49 + 0.09 + 0.04 + 0.36) = 0.5 * 0.98 = 0.49
    CHECK(std::abs(pg(0) - 3.225) < 1e-10);

    // XPR only from NLOS paths (LOS excluded by default)
    // P_vv = 0.5*0.81 + 0.5*0.49 = 0.405 + 0.245 = 0.65
    // P_hv = 0.5*0.01 + 0.5*0.09 = 0.005 + 0.045 = 0.05
    // P_vh = 0.5*0.01 + 0.5*0.04 = 0.005 + 0.02  = 0.025
    // P_hh = 0.5*0.64 + 0.5*0.36 = 0.32  + 0.18  = 0.50

    // V-XPR = P_vv / P_hv = 0.65 / 0.05 = 13.0
    CHECK(std::abs(xpr(0, 1) - 13.0) < 1e-10);

    // H-XPR = P_hh / P_vh = 0.50 / 0.025 = 20.0
    CHECK(std::abs(xpr(0, 2) - 20.0) < 1e-10);

    // Aggregate linear = (P_vv + P_hh) / (P_hv + P_vh) = 1.15 / 0.075
    CHECK(std::abs(xpr(0, 0) - 1.15 / 0.075) < 1e-10);
}

TEST_CASE("calc_cross_polarization_ratio - Include LOS")
{
    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({10.0, 0.0, 0.0});

    arma::vec pw(2);
    pw(0) = 1.0;
    pw(1) = 0.5;

    arma::mat M(8, 2);
    // LOS: diagonal, purely co-pol
    M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0});
    // NLOS: some cross-pol
    M.col(1) = arma::vec({0.8, 0.0, 0.2, 0.0, 0.2, 0.0, 0.7, 0.0});

    arma::vec pl(2);
    pl(0) = 10.0;
    pl(1) = 15.0;

    std::vector<arma::vec> pv = {pw};
    std::vector<arma::mat> mv = {M};
    std::vector<arma::vec> plv = {pl};

    arma::mat xpr_no_los, xpr_with_los;

    // Without LOS (default)
    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr_no_los,
                                                (arma::Col<double> *)nullptr, false);

    // With LOS
    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr_with_los,
                                                (arma::Col<double> *)nullptr, true);

    // V-XPR without LOS: only NLOS path
    // P_vv = 0.5*0.64, P_hv = 0.5*0.04 => XPR_V = 0.64/0.04 = 16
    CHECK(std::abs(xpr_no_los(0, 1) - 16.0) < 1e-10);

    // V-XPR with LOS: includes LOS path which has zero cross-pol
    // P_vv = 1.0*1.0 + 0.5*0.64 = 1.32, P_hv = 0.5*0.04 = 0.02
    // XPR_V = 1.32 / 0.02 = 66.0
    CHECK(std::abs(xpr_with_los(0, 1) - 66.0) < 1e-10);
}

TEST_CASE("calc_cross_polarization_ratio - Circular polarization identity check")
{
    // For a purely diagonal M with M_vv = 1, M_hh = -1 (LOS-like),
    // the circular basis should show:
    // M_LL = (1 + (-1)) / 2 = 0 (all real)
    // M_RR = (1 + (-1)) / 2 = 0
    // M_LR = (1 - (-1)) / 2 = 1
    // M_RL = (1 - (-1)) / 2 = 1
    // => circular XPR = 0 (co-pol is zero, cross-pol is nonzero)
    // This reflects the physical fact that a single reflection reverses handedness.

    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({5.0, 0.0, 0.0});

    arma::vec pw(1);
    pw(0) = 1.0;

    arma::mat M(8, 1);
    M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0});

    arma::vec pl(1);
    pl(0) = 20.0; // NLOS (far from dTR=5)

    std::vector<arma::vec> pv = {pw};
    std::vector<arma::mat> mv = {M};
    std::vector<arma::vec> plv = {pl};

    arma::mat xpr;
    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr);

    // Linear XPR should be very high (zero cross-pol => returns 0 for undefined)
    // M_vv=1, M_hv=0, M_vh=0, M_hh=-1
    // P_vv = 1, P_hv = 0, P_vh = 0, P_hh = 1
    // Cross-pol = 0 => XPR = 0 (undefined)
    CHECK(xpr(0, 0) == 0.0);
    CHECK(xpr(0, 1) == 0.0);
    CHECK(xpr(0, 2) == 0.0);

    // Circular: M_LL = (1+(-1))/2 = 0, M_RR = (1+(-1))/2 = 0
    //           M_LR = (1-(-1))/2 = 1, M_RL = (1-(-1))/2 = 1
    // Co-pol_circ = 0, Cross-pol_circ = 2 => circ XPR = 0
    CHECK(xpr(0, 3) == 0.0);
    CHECK(xpr(0, 4) == 0.0);
    CHECK(xpr(0, 5) == 0.0);
}

TEST_CASE("calc_cross_polarization_ratio - Circular XPR for identity matrix")
{
    // M = identity: M_vv=1, M_hv=0, M_vh=0, M_hh=1
    // This is a double-bounce / polarization-preserving channel
    // Circular: M_LL = (1+1)/2 = 1, M_RR = (1+1)/2 = 1
    //           M_LR = (1-1)/2 = 0, M_RL = (1-1)/2 = 0
    // => Circular XPR = infinite (co-pol = 2, cross-pol = 0) => returns 0

    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({5.0, 0.0, 0.0});

    arma::vec pw(1);
    pw(0) = 1.0;

    arma::mat M(8, 1);
    M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}); // Identity

    arma::vec pl(1);
    pl(0) = 20.0;

    std::vector<arma::vec> pv = {pw};
    std::vector<arma::mat> mv = {M};
    std::vector<arma::vec> plv = {pl};

    arma::mat xpr;
    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr);

    // Linear XPR: P_vv=1, P_hh=1, P_hv=0, P_vh=0 => undefined (0)
    CHECK(xpr(0, 0) == 0.0);

    // Circular XPR: co=2, cross=0 => undefined (0)
    CHECK(xpr(0, 3) == 0.0);
}

TEST_CASE("calc_cross_polarization_ratio - Complex M elements")
{
    // Test with complex-valued M entries
    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({10.0, 0.0, 0.0});

    arma::vec pw(1);
    pw(0) = 1.0;

    // M_vv = 0.8+0.2j, M_hv = 0.1-0.1j, M_vh = 0.05+0.05j, M_hh = 0.7-0.3j
    arma::mat M(8, 1);
    M.col(0) = arma::vec({0.8, 0.2, 0.1, -0.1, 0.05, 0.05, 0.7, -0.3});

    arma::vec pl(1);
    pl(0) = 20.0; // NLOS

    std::vector<arma::vec> pv = {pw};
    std::vector<arma::mat> mv = {M};
    std::vector<arma::vec> plv = {pl};

    arma::mat xpr;
    arma::vec pg;
    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr, &pg);

    // Manual calculation
    double abs2_vv = 0.8 * 0.8 + 0.2 * 0.2;   // 0.68
    double abs2_hv = 0.1 * 0.1 + 0.1 * 0.1;   // 0.02
    double abs2_vh = 0.05 * 0.05 + 0.05 * 0.05; // 0.005
    double abs2_hh = 0.7 * 0.7 + 0.3 * 0.3;   // 0.58

    CHECK(std::abs(pg(0) - (abs2_vv + abs2_hv + abs2_vh + abs2_hh)) < 1e-14);
    CHECK(std::abs(xpr(0, 1) - abs2_vv / abs2_hv) < 1e-10);
    CHECK(std::abs(xpr(0, 2) - abs2_hh / abs2_vh) < 1e-10);
    CHECK(std::abs(xpr(0, 0) - (abs2_vv + abs2_hh) / (abs2_hv + abs2_vh)) < 1e-10);
}

TEST_CASE("calc_cross_polarization_ratio - Multiple CIRs with mobile TX/RX")
{
    // 2 CIRs with different TX/RX positions
    arma::mat tx(3, 2);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    tx.col(1) = arma::vec({1.0, 0.0, 0.0});

    arma::mat rx(3, 2);
    rx.col(0) = arma::vec({10.0, 0.0, 0.0});
    rx.col(1) = arma::vec({11.0, 0.0, 0.0});

    // CIR 0: 2 paths
    arma::vec pw0(2);
    pw0(0) = 1.0;
    pw0(1) = 0.5;
    arma::mat M0(8, 2);
    M0.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}); // LOS
    M0.col(1) = arma::vec({0.8, 0.0, 0.2, 0.0, 0.15, 0.0, 0.7, 0.0}); // NLOS
    arma::vec pl0(2);
    pl0(0) = 10.0;
    pl0(1) = 14.0;

    // CIR 1: 2 paths
    arma::vec pw1(2);
    pw1(0) = 0.8;
    pw1(1) = 0.4;
    arma::mat M1(8, 2);
    M1.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}); // LOS
    M1.col(1) = arma::vec({0.6, 0.0, 0.3, 0.0, 0.25, 0.0, 0.5, 0.0}); // NLOS
    arma::vec pl1(2);
    pl1(0) = 10.0;
    pl1(1) = 13.0;

    std::vector<arma::vec> pv = {pw0, pw1};
    std::vector<arma::mat> mv = {M0, M1};
    std::vector<arma::vec> plv = {pl0, pl1};

    arma::mat xpr;
    arma::vec pg;

    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr, &pg);

    REQUIRE(xpr.n_rows == 2);
    REQUIRE(xpr.n_cols == 6);
    REQUIRE(pg.n_elem == 2);

    // CIR 0: V-XPR from NLOS only (path 1)
    // P_vv = 0.5 * 0.64 = 0.32, P_hv = 0.5 * 0.04 = 0.02
    CHECK(std::abs(xpr(0, 1) - 0.32 / 0.02) < 1e-10);

    // CIR 1: V-XPR from NLOS only (path 1)
    // P_vv = 0.4 * 0.36 = 0.144, P_hv = 0.4 * 0.09 = 0.036
    CHECK(std::abs(xpr(1, 1) - 0.144 / 0.036) < 1e-10);
}

TEST_CASE("calc_cross_polarization_ratio - Window size effect")
{
    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({10.0, 0.0, 0.0});

    // 3 paths with path lengths 10.0, 10.005, 12.0
    arma::vec pw(3);
    pw(0) = 1.0;
    pw(1) = 0.8;
    pw(2) = 0.5;

    arma::mat M(8, 3);
    M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0});
    M.col(1) = arma::vec({0.9, 0.0, 0.15, 0.0, 0.1, 0.0, 0.85, 0.0}); // Near-LOS
    M.col(2) = arma::vec({0.7, 0.0, 0.3, 0.0, 0.2, 0.0, 0.6, 0.0});   // NLOS

    arma::vec pl(3);
    pl(0) = 10.0;
    pl(1) = 10.005;
    pl(2) = 12.0;

    std::vector<arma::vec> pv = {pw};
    std::vector<arma::mat> mv = {M};
    std::vector<arma::vec> plv = {pl};

    arma::mat xpr_small, xpr_large;

    // Small window (0.001): excludes only path 0
    quadriga_lib::calc_cross_polarization_ratio<double>(pv, mv, plv, tx, rx, &xpr_small,
                                                        (arma::Col<double> *)nullptr, false, 0.001);

    // Large window (0.01): excludes paths 0 and 1
    quadriga_lib::calc_cross_polarization_ratio<double>(pv, mv, plv, tx, rx, &xpr_large,
                                                        (arma::Col<double> *)nullptr, false, 0.01);

    // Small window includes path 1, large window excludes it
    // They should give different results
    CHECK(xpr_small(0, 1) != xpr_large(0, 1));

    // Large window: only path 2
    // V-XPR = 0.5*0.49 / (0.5*0.09) = 0.49/0.09
    CHECK(std::abs(xpr_large(0, 1) - 0.49 / 0.09) < 1e-10);
}

TEST_CASE("calc_cross_polarization_ratio - Only pg output requested")
{
    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({10.0, 0.0, 0.0});

    arma::vec pw(1);
    pw(0) = 2.0;
    arma::mat M(8, 1);
    M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
    arma::vec pl(1);
    pl(0) = 10.0;

    std::vector<arma::vec> pv = {pw};
    std::vector<arma::mat> mv = {M};
    std::vector<arma::vec> plv = {pl};

    arma::vec pg;
    quadriga_lib::calc_cross_polarization_ratio<double>(pv, mv, plv, tx, rx,
                                                        (arma::Mat<double> *)nullptr, &pg);

    REQUIRE(pg.n_elem == 1);
    CHECK(std::abs(pg(0) - 4.0) < 1e-14); // 2.0 * (1+0+0+1) = 4.0
}

TEST_CASE("calc_cross_polarization_ratio - Input validation errors")
{
    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({10.0, 0.0, 0.0});

    // Empty powers
    {
        std::vector<arma::vec> pv;
        std::vector<arma::mat> mv;
        std::vector<arma::vec> plv;
        arma::mat xpr;
        CHECK_THROWS_AS(
            quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr),
            std::invalid_argument);
    }

    // Mismatched M size
    {
        arma::vec pw(2);
        pw(0) = 1.0;
        pw(1) = 0.5;
        arma::mat M(8, 1); // Wrong: should be 2 columns
        M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
        arma::vec pl(2);
        pl(0) = 10.0;
        pl(1) = 12.0;

        std::vector<arma::vec> pv = {pw};
        std::vector<arma::mat> mv = {M};
        std::vector<arma::vec> plv = {pl};
        arma::mat xpr;
        CHECK_THROWS_AS(
            quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr),
            std::invalid_argument);
    }

    // Wrong tx_pos rows
    {
        arma::mat bad_tx(2, 1);
        bad_tx.col(0) = arma::vec({0.0, 0.0});
        arma::vec pw(1);
        pw(0) = 1.0;
        arma::mat M(8, 1);
        M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
        arma::vec pl(1);
        pl(0) = 10.0;
        std::vector<arma::vec> pv = {pw};
        std::vector<arma::mat> mv = {M};
        std::vector<arma::vec> plv = {pl};
        arma::mat xpr;
        CHECK_THROWS_AS(
            quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, bad_tx, rx, &xpr),
            std::invalid_argument);
    }

    // Negative window size
    {
        arma::vec pw(1);
        pw(0) = 1.0;
        arma::mat M(8, 1);
        M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
        arma::vec pl(1);
        pl(0) = 10.0;
        std::vector<arma::vec> pv = {pw};
        std::vector<arma::mat> mv = {M};
        std::vector<arma::vec> plv = {pl};
        arma::mat xpr;
        CHECK_THROWS_AS(
            (quadriga_lib::calc_cross_polarization_ratio<double>(pv, mv, plv, tx, rx, &xpr,
                                                                 (arma::Col<double> *)nullptr, false, -0.01)),
            std::invalid_argument);
    }
}

TEST_CASE("calc_cross_polarization_ratio - Float precision")
{
    arma::fmat tx(3, 1);
    tx.col(0) = arma::fvec({0.0f, 0.0f, 0.0f});
    arma::fmat rx(3, 1);
    rx.col(0) = arma::fvec({10.0f, 0.0f, 0.0f});

    arma::fvec pw(2);
    pw(0) = 1.0f;
    pw(1) = 0.5f;
    arma::fmat M(8, 2);
    M.col(0) = arma::fvec({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f});
    M.col(1) = arma::fvec({0.8f, 0.0f, 0.2f, 0.0f, 0.15f, 0.0f, 0.7f, 0.0f});
    arma::fvec pl(2);
    pl(0) = 10.0f;
    pl(1) = 15.0f;

    std::vector<arma::fvec> pv = {pw};
    std::vector<arma::fmat> mv = {M};
    std::vector<arma::fvec> plv = {pl};

    arma::fmat xpr;
    arma::fvec pg;
    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr, &pg);

    REQUIRE(xpr.n_rows == 1);
    REQUIRE(xpr.n_cols == 6);

    // V-XPR = 0.5*0.64 / (0.5*0.04) = 16.0
    CHECK(std::abs(xpr(0, 1) - 16.0f) < 1e-5f);
}

TEST_CASE("calc_cross_polarization_ratio - Circular XPR with known cross-pol")
{
    // Create a path with known circular polarization behavior
    // M = {{0.5, 0.5}, {0.5, 0.5}} (all real)
    // This distributes power equally in all polarization components
    arma::mat tx(3, 1);
    tx.col(0) = arma::vec({0.0, 0.0, 0.0});
    arma::mat rx(3, 1);
    rx.col(0) = arma::vec({10.0, 0.0, 0.0});

    arma::vec pw(1);
    pw(0) = 1.0;
    arma::mat M(8, 1);
    // M_vv=0.5, M_hv=0.5, M_vh=0.5, M_hh=0.5 (all real)
    M.col(0) = arma::vec({0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0});
    arma::vec pl(1);
    pl(0) = 20.0;

    std::vector<arma::vec> pv = {pw};
    std::vector<arma::mat> mv = {M};
    std::vector<arma::vec> plv = {pl};

    arma::mat xpr;
    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr);

    // Linear: all |M|²=0.25, so XPR = 1.0 everywhere (0 dB)
    CHECK(std::abs(xpr(0, 0) - 1.0) < 1e-10);
    CHECK(std::abs(xpr(0, 1) - 1.0) < 1e-10);
    CHECK(std::abs(xpr(0, 2) - 1.0) < 1e-10);

    // Circular: with all M elements equal and real:
    // a=g=0.5, c=e=0.5, b=d=f=h=0
    // M_LL: re=(0.5+0.5-0+0.5)=1.5, im=(0+0+0.5-0.5)=0 => |M_LL|²=1.5²/4=0.5625
    // M_RL: re=(0.5-0.5+0+0.5)=0.5, im=(0-0-0.5-0.5)=-1 => |M_RL|²=(0.25+1)/4=0.3125
    // M_LR: re=(0.5-0.5-0-0.5)=-0.5, im=(0-0+0.5+0.5)=1 => |M_LR|²=(0.25+1)/4=0.3125
    // M_RR: re=(0.5+0.5+0-0.5)=0.5, im=(0+0-0.5+0.5)=0 => |M_RR|²=0.25/4=0.0625
    double co_circ = 0.5625 + 0.0625;   // 0.625
    double cross_circ = 0.3125 + 0.3125; // 0.625
    CHECK(std::abs(xpr(0, 3) - co_circ / cross_circ) < 1e-10);
}
