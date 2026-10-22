// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"
#include <cmath>
#include <complex>
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
    pw(0) = 1.0; // LOS
    pw(1) = 0.5; // NLOS
    pw(2) = 0.5; // NLOS

    // Polarization matrices [8, n_path]
    // LOS: M = {{1,0},{0,-1}} => purely diagonal
    arma::mat M(8, 3);
    M.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}); // LOS

    // NLOS path 1: some cross-pol leakage
    // M_vv=0.9, M_hv=0.1, M_vh=0.1, M_hh=0.8 (all real)
    M.col(1) = arma::vec({0.9, 0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0});

    // NLOS path 2: more cross-pol
    // NLOS path 2: M_vv=0.7, M_vh=0.2, M_hv=0.3, M_hh=0.6 (all real)
    M.col(2) = arma::vec({0.7, 0.0, 0.2, 0.0, 0.3, 0.0, 0.6, 0.0});

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
    CHECK(std::abs(pg(0) - 0.5 * 3.225) < 1e-10);

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
    // M = diag(1, -1) is the LOS matrix of get_channels_* and the ray tracer (the receive frame's e_phi points the
    // other way). The circular entries couple the same circular vector at both ends, c(r, t) = r^T M t, with
    // L = (1, +j)/sqrt(2) and R = (1, -j)/sqrt(2) in (V, H):
    // M_LL = (M_vv - M_hh + j(M_vh + M_hv)) / 2 = (1 - (-1)) / 2 = 1, M_RR = 1
    // M_LR = (M_vv + M_hh + j(M_hv - M_vh)) / 2 = 0, M_RL = 0
    // => circular XPR = Inf (co-pol only), as get_channels_spherical gives |c| = 1 for the same circular
    //    pattern at both ends and 0 for opposite patterns: the LOS keeps the handedness

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

    // Linear XPR should be infinite (zero cross-pol, nonzero co-pol => perfect isolation)
    // M_vv=1, M_hv=0, M_vh=0, M_hh=-1
    // P_vv = 1, P_hv = 0, P_vh = 0, P_hh = 1
    // Cross-pol = 0, Co-pol = 2 => XPR = Inf
    CHECK(std::isinf(xpr(0, 0)));
    CHECK(std::isinf(xpr(0, 1)));
    CHECK(std::isinf(xpr(0, 2)));

    // Circular: co-pol_circ = 2, cross-pol_circ = 0 => Inf in all three circular columns
    CHECK(std::isinf(xpr(0, 3)));
    CHECK(std::isinf(xpr(0, 4)));
    CHECK(std::isinf(xpr(0, 5)));
}

TEST_CASE("calc_cross_polarization_ratio - Circular XPR for identity matrix")
{
    // M = identity: M_vv=1, M_hv=0, M_vh=0, M_hh=1
    // Relative to the LOS matrix diag(1, -1) the H component is mirrored, as by a single reflection: the
    // handedness is reversed
    // Circular: M_LL = (1-1)/2 = 0, M_RR = (1-1)/2 = 0
    //           M_LR = (1+1)/2 = 1, M_RL = (1+1)/2 = 1
    // => Circular XPR = 0 (co-pol = 0, cross-pol = 2)

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

    // Linear XPR: P_vv=1, P_hh=1, P_hv=0, P_vh=0 => perfect isolation (Inf)
    CHECK(std::isinf(xpr(0, 0)));

    // Circular XPR: co=0, cross=2 => 0
    CHECK(xpr(0, 3) == 0.0);
    CHECK(xpr(0, 4) == 0.0);
    CHECK(xpr(0, 5) == 0.0);
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

    // M_vv = 0.8+0.2j, M_vh = 0.05+0.05j, M_hv = 0.1-0.1j, M_hh = 0.7-0.3j
    arma::mat M(8, 1);
    M.col(0) = arma::vec({0.8, 0.2, 0.05, 0.05, 0.1, -0.1, 0.7, -0.3});

    arma::vec pl(1);
    pl(0) = 20.0; // NLOS

    std::vector<arma::vec> pv = {pw};
    std::vector<arma::mat> mv = {M};
    std::vector<arma::vec> plv = {pl};

    arma::mat xpr;
    arma::vec pg;
    quadriga_lib::calc_cross_polarization_ratio(pv, mv, plv, tx, rx, &xpr, &pg);

    // Manual calculation
    double abs2_vv = 0.8 * 0.8 + 0.2 * 0.2;     // 0.68
    double abs2_hv = 0.1 * 0.1 + 0.1 * 0.1;     // 0.02
    double abs2_vh = 0.05 * 0.05 + 0.05 * 0.05; // 0.005
    double abs2_hh = 0.7 * 0.7 + 0.3 * 0.3;     // 0.58

    CHECK(std::abs(pg(0) - 0.5 * (abs2_vv + abs2_hv + abs2_vh + abs2_hh)) < 1e-14);
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
    M0.col(1) = arma::vec({0.8, 0.0, 0.15, 0.0, 0.2, 0.0, 0.7, 0.0}); // NLOS
    arma::vec pl0(2);
    pl0(0) = 10.0;
    pl0(1) = 14.0;

    // CIR 1: 2 paths
    arma::vec pw1(2);
    pw1(0) = 0.8;
    pw1(1) = 0.4;
    arma::mat M1(8, 2);
    M1.col(0) = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}); // LOS
    M1.col(1) = arma::vec({0.6, 0.0, 0.25, 0.0, 0.3, 0.0, 0.5, 0.0}); // NLOS
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
    M.col(1) = arma::vec({0.9, 0.0, 0.1, 0.0, 0.15, 0.0, 0.85, 0.0}); // Near-LOS
    M.col(2) = arma::vec({0.7, 0.0, 0.2, 0.0, 0.3, 0.0, 0.6, 0.0});   // NLOS

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
    CHECK(std::abs(pg(0) - 2.0) < 1e-14); // (1+0+0+1) = 2.0
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
    M.col(1) = arma::fvec({0.8f, 0.0f, 0.15f, 0.0f, 0.2f, 0.0f, 0.7f, 0.0f});
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

    // Circular: with all M elements equal and real (M_vv = M_hv = M_vh = M_hh = 0.5):
    // M_LL = (M_vv - M_hh + j(M_vh + M_hv)) / 2 = 0.5j  => |M_LL|^2 = 0.25
    // M_RR = (M_vv - M_hh - j(M_vh + M_hv)) / 2 = -0.5j => |M_RR|^2 = 0.25
    // M_LR = (M_vv + M_hh + j(M_hv - M_vh)) / 2 = 0.5   => |M_LR|^2 = 0.25
    // M_RL = (M_vv + M_hh + j(M_vh - M_hv)) / 2 = 0.5   => |M_RL|^2 = 0.25
    double co_circ = 0.25 + 0.25;    // 0.5
    double cross_circ = 0.25 + 0.25; // 0.5
    CHECK(std::abs(xpr(0, 3) - co_circ / cross_circ) < 1e-10);
    CHECK(std::abs(xpr(0, 4) - 1.0) < 1e-10);
    CHECK(std::abs(xpr(0, 5) - 1.0) < 1e-10);
}

TEST_CASE("calc_cross_polarization_ratio - V-XPR and H-XPR are per receive polarisation")
{
    // M rows: 0-1 = (V rx, V tx), 2-3 = (H rx, V tx), 4-5 = (V rx, H tx), 6-7 = (H rx, H tx).
    // V-XPR = sum |M(V rx, V tx)|^2 / sum |M(V rx, H tx)|^2, H-XPR = sum |M(H rx, H tx)|^2 / sum |M(H rx, V tx)|^2
    arma::mat tx = arma::vec({0.0, 0.0, 0.0}), rx = arma::vec({100.0, 0.0, 0.0}), xpr;
    std::vector<arma::vec> pw = {{1.0}}, pl = {{200.0}};

    // Leakage from V transmit into H receive only (rows 2-3): V receive sees no cross-pol
    std::vector<arma::mat> Mv = {arma::mat(arma::vec({1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0}))};
    quadriga_lib::calc_cross_polarization_ratio<double>(pw, Mv, pl, tx, rx, &xpr);
    CHECK(std::isinf(xpr(0, 1)));                 // V-XPR
    CHECK(std::abs(xpr(0, 2) - 4.0) < 1e-12);     // H-XPR = 1 / 0.25
    CHECK(std::abs(xpr(0, 0) - 8.0) < 1e-12);     // aggregate (1 + 1) / 0.25

    // Leakage from H transmit into V receive only (rows 4-5): H receive sees no cross-pol
    Mv = {arma::mat(arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0}))};
    quadriga_lib::calc_cross_polarization_ratio<double>(pw, Mv, pl, tx, rx, &xpr);
    CHECK(std::abs(xpr(0, 1) - 4.0) < 1e-12);
    CHECK(std::isinf(xpr(0, 2)));
}

TEST_CASE("calc_cross_polarization_ratio - Circular columns are per receive polarisation and match get_channels_spherical")
{
    // Circular coupling matrix C (rows = RX L/R, columns = TX L/R) -> linear M = P^-T C P^-1 with P = [L R],
    // L = (1, +j)/sqrt(2), R = (1, -j)/sqrt(2) in (V, H), P^-1 = [1 -j; 1 j] / sqrt(2)
    typedef std::complex<double> cx;
    const cx j(0.0, 1.0);
    const cx C[2][2] = {{1.0, 0.5 * j}, {0.25, 2.0}}; // C_LL, C_LR (rx L, tx R); C_RL (rx R, tx L), C_RR
    const cx Pinv[2][2] = {{1.0 / std::sqrt(2.0), -j / std::sqrt(2.0)}, {1.0 / std::sqrt(2.0), j / std::sqrt(2.0)}};
    cx Ml[2][2];
    for (int r = 0; r < 2; ++r)
        for (int t = 0; t < 2; ++t)
        {
            Ml[r][t] = 0.0;
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    Ml[r][t] += Pinv[a][r] * C[a][b] * Pinv[b][t]; // (P^-T)(r, a) = P^-1(a, r)
        }
    arma::mat M = arma::vec({Ml[0][0].real(), Ml[0][0].imag(), Ml[1][0].real(), Ml[1][0].imag(),
                             Ml[0][1].real(), Ml[0][1].imag(), Ml[1][1].real(), Ml[1][1].imag()});

    arma::mat tx = arma::vec({0.0, 0.0, 0.0}), rx = arma::vec({100.0, 0.0, 0.0}), xpr;
    std::vector<arma::vec> pw = {{1.0}}, pl = {{100.0}};
    std::vector<arma::mat> Mv = {M};
    quadriga_lib::calc_cross_polarization_ratio<double>(pw, Mv, pl, tx, rx, &xpr, nullptr, true);
    CHECK(std::abs(xpr(0, 3) - (1.0 + 4.0) / (0.25 + 0.0625)) < 1e-12); // (LL + RR) / (LR + RL)
    CHECK(std::abs(xpr(0, 4) - 1.0 / 0.25) < 1e-12);                    // LHCP: |C_LL|^2 / |C_LR|^2
    CHECK(std::abs(xpr(0, 5) - 4.0 / 0.0625) < 1e-12);                  // RHCP: |C_RR|^2 / |C_RL|^2

    // The same powers from get_channels_spherical with circular patterns, (V, H) = (1, +j)/sqrt(2) (LHCP) and
    // (1, -j)/sqrt(2) (RHCP), at the receive and transmit ends
    auto L = quadriga_lib::generate_arrayant_omni<double>(10.0);
    L.e_theta_re *= std::sqrt(0.5);
    L.e_phi_im = L.e_theta_re;
    L.e_phi_re.zeros();
    auto R = L;
    R.e_phi_im *= -1.0;
    arma::mat F = arma::vec({50.0, 0.0, 0.0});
    arma::vec g = {1.0}, len = {100.0};
    auto p2 = [&](const quadriga_lib::arrayant<double> &tx_ant, const quadriga_lib::arrayant<double> &rx_ant)
    {
        arma::cube cr, ci, dl;
        quadriga_lib::get_channels_spherical<double>(&tx_ant, &rx_ant, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, &F, &F, &g,
                                                     &len, &M, &cr, &ci, &dl, 0.0);
        return cr(0) * cr(0) + ci(0) * ci(0);
    };
    double LL = p2(L, L), LR = p2(R, L), RL = p2(L, R), RR = p2(R, R); // <rx><tx>
    CHECK(std::abs(LL - 1.0) < 1e-5);
    CHECK(std::abs(LR - 0.25) < 1e-5);
    CHECK(std::abs(RL - 0.0625) < 1e-5);
    CHECK(std::abs(RR - 4.0) < 1e-5);
    CHECK(std::abs(xpr(0, 4) - LL / LR) < 1e-3);
    CHECK(std::abs(xpr(0, 5) - RR / RL) < 1e-2);

    // LOS matrix of get_channels_* (co-polar) and a mirrored H component as by one reflection (cross-polar)
    Mv = {arma::mat(arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0})), arma::mat(arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}))};
    std::vector<arma::vec> pw2 = {{1.0}, {1.0}}, pl2 = {{100.0}, {100.0}};
    quadriga_lib::calc_cross_polarization_ratio<double>(pw2, Mv, pl2, tx, rx, &xpr, nullptr, true);
    CHECK(std::isinf(xpr(0, 3)));
    CHECK(std::isinf(xpr(0, 4)));
    CHECK(std::isinf(xpr(0, 5)));
    CHECK(xpr(1, 3) == 0.0);
    CHECK(xpr(1, 4) == 0.0);
    CHECK(xpr(1, 5) == 0.0);
}
