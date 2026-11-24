// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// set_orientation: the orientation of tracks from their steps (bank 0, tilt and heading of the forward differences, the
// last step repeated or the first row for a closed track), the offset, force, nodes with relative = 1, point drops.
//
// Oracles: closed forms of the step directions of small tracks typed by hand (positive tilt points down, so a rising
// step has a negative tilt; a step without horizontal part keeps the previous heading). No random input.

#include <catch2/catch_test_macros.hpp>

#include "qd_test_layout.hpp"
#include "quadriga_layout.hpp"

#include <cmath>
#include <string>

namespace
{
const double D = arma::datum::pi / 180.0;
const double NaN = arma::datum::nan;
const double PI = arma::datum::pi;

bool has(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }
} // namespace

TEST_CASE("set_orientation")
{
    // Rows: A 0-2 straight along x; B 3-5 rising by 45 deg; C 6-9 closed triangle; D 10 point drop;
    //       E 11-14 with a step of zero length; F 15-17 starting with a step of zero length; G 18-19 descending
    auto make = []
    {
        auto N = qd_layout_nodes_from({arma::mat{{0.0, 10.0, 20.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
                                       arma::mat{{0.0, 10.0, 20.0}, {0.0, 0.0, 0.0}, {0.0, 10.0, 20.0}},
                                       arma::mat{{0.0, 10.0, 10.0, 0.0}, {0.0, 0.0, 10.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
                                       qd_layout_pt(5.0, 5.0, 5.0),
                                       arma::mat{{0.0, 10.0, 10.0, 10.0}, {0.0, 0.0, 0.0, 10.0}, {0.0, 0.0, 0.0, 0.0}},
                                       arma::mat{{0.0, 0.0, 0.0}, {0.0, 0.0, 10.0}, {0.0, 0.0, 0.0}},
                                       arma::mat{{0.0, 10.0}, {0.0, 0.0}, {10.0, 0.0}}});
        N.orientation.col(10) = arma::vec{5.0 * D, 6.0 * D, 7.0 * D};
        N.orientation.col(0) = arma::vec{0.3, 0.3, 0.3}; // Tracks lose their old orientation
        return N;
    };
    // Expected (bank, tilt, heading) per row without offset
    const arma::mat e0 = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5 * D, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          {0, 0, 0, -45 * D, -45 * D, -45 * D, 0, 0, 0, 0, 6 * D, 0, 0, 0, 0, 0, 0, 0, 45 * D, 45 * D},
                          {0, 0, 0, 0, 0, 0, 0, 90 * D, -135 * D, 0, 7 * D, 0, 0, 90 * D, 90 * D, 90 * D, 90 * D, 90 * D, 0, 0}};
    auto N = make();
    const auto N0 = N;

    SECTION("from the motion: bank 0, tilt and heading of the steps, the last step repeated or the first of a closed track")
    {
        quadriga_lib::set_orientation(N);
        CHECK(arma::abs(N.orientation - e0).max() < 1e-13);
        CHECK(arma::all(N.pos.col(0) == N0.pos.col(0)));
        CHECK(N.orientation(0, 9) == N.orientation(0, 6)); // Closed: exact copy of the first row
        CHECK(N.orientation(1, 9) == N.orientation(1, 6));
        CHECK(N.orientation(2, 9) == N.orientation(2, 6));
        CHECK(N.orientation(2, 10) == 7.0 * D); // A point drop keeps its orientation
        // Empty orientation vector, all nodes, same as zeros
        auto M = make();
        quadriga_lib::set_orientation(M, arma::vec{0.0, 0.0, 0.0}, {}, false);
        CHECK(arma::abs(M.orientation - N.orientation).max() == 0.0);
    }
    SECTION("the orientation is an offset")
    {
        quadriga_lib::set_orientation(N, arma::vec{10.0 * D, 20.0 * D, 30.0 * D});
        arma::mat e = e0;
        e.each_col() += arma::vec{10.0 * D, 20.0 * D, 30.0 * D};
        CHECK(arma::abs(N.orientation - e).max() < 1e-13);
    }
    SECTION("the result is wrapped to [-180, 180] deg")
    {
        quadriga_lib::set_orientation(N, arma::vec{0.0, 0.0, 200.0 * D});
        // Straight track along x: 0 + 200 = -160 deg; closed triangle: -135 + 200 = 65, 90 + 200 = -70 deg
        CHECK(std::abs(N.orientation(2, 0) + 160.0 * D) < 1e-13);
        CHECK(std::abs(N.orientation(2, 7) + 70.0 * D) < 1e-13);
        CHECK(std::abs(N.orientation(2, 8) - 65.0 * D) < 1e-13);
        CHECK(std::abs(N.orientation(2, 10) - qd_layout_wrap(7.0 * D + 200.0 * D)) < 1e-13);
        CHECK(arma::abs(N.orientation).max() <= PI + 1e-13);
    }
    SECTION("nodes of the list only")
    {
        quadriga_lib::set_orientation(N, arma::vec{}, {1, 3});
        CHECK(arma::abs(N.orientation.cols(3, 5) - e0.cols(3, 5)).max() < 1e-13);
        CHECK(arma::abs(N.orientation.col(0) - N0.orientation.col(0)).max() == 0.0); // Node 0 is not in the list
        CHECK(arma::abs(N.orientation.cols(6, 19) - N0.orientation.cols(6, 19)).max() == 0.0);
        quadriga_lib::set_orientation(N, arma::vec{0.1, 0.0, 0.0}, {3});
        CHECK(std::abs(N.orientation(0, 10) - (5.0 * D + 0.1)) < 1e-14);
    }
    SECTION("force: exactly the given orientation on all rows")
    {
        quadriga_lib::set_orientation(N, arma::vec{1.0 * D, 2.0 * D, 3.0 * D}, {}, true);
        for (arma::uword r = 0; r < N.n_row(); ++r)
        {
            CHECK(N.orientation(0, r) == 1.0 * D);
            CHECK(N.orientation(1, r) == 2.0 * D);
            CHECK(N.orientation(2, r) == 3.0 * D);
        }
        auto M = make();
        quadriga_lib::set_orientation(M, arma::vec{}, {2}, true); // Empty = 0
        CHECK(arma::abs(M.orientation.cols(6, 9)).max() == 0.0);
        CHECK(arma::abs(M.orientation.col(10) - N0.orientation.col(10)).max() == 0.0);
        // No wrapping
        quadriga_lib::set_orientation(M, arma::vec{0.0, 0.0, 7.0}, {2}, true);
        CHECK(M.orientation(2, 7) == 7.0);
    }
    SECTION("mounted antennas keep their orientation unless forced")
    {
        N.relative = {1, 0, 0, 1, 0, 0, 0};
        quadriga_lib::set_orientation(N);
        CHECK(arma::abs(N.orientation.cols(0, 2) - N0.orientation.cols(0, 2)).max() == 0.0);
        CHECK(arma::abs(N.orientation.col(10) - N0.orientation.col(10)).max() == 0.0);
        CHECK(arma::abs(N.orientation.cols(3, 5) - e0.cols(3, 5)).max() < 1e-13);
        quadriga_lib::set_orientation(N, arma::vec{1.0, 1.0, 1.0}, {}, false);
        CHECK(arma::abs(N.orientation.cols(0, 2) - N0.orientation.cols(0, 2)).max() == 0.0); // Not even the offset
        CHECK(arma::abs(N.orientation.col(10) - N0.orientation.col(10)).max() == 0.0);
        quadriga_lib::set_orientation(N, arma::vec{0.1, 0.2, 0.3}, {0, 3}, true);
        CHECK(N.orientation(1, 1) == 0.2);
        CHECK(N.orientation(2, 10) == 0.3);
    }
    SECTION("a point drop without offset keeps its angles bit for bit")
    {
        N.orientation.col(10) = arma::vec{0.1, 0.2, 6.1}; // Beyond pi: not wrapped by an empty offset
        quadriga_lib::set_orientation(N);
        CHECK(N.orientation(2, 10) == 6.1);
    }
}

TEST_CASE("set_orientation - input validation")
{
    auto N = qd_layout_nodes_from({arma::mat{{0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}}, qd_layout_pt(0.0, 0.0, 0.0)});
    const auto N0 = N;
    using quadriga_lib::set_orientation;
    auto fails = [&](const std::string &part, auto call)
    {
        const std::string msg = qd_layout_throws(call);
        INFO("message: " << msg);
        CHECK(has(msg, part));
        CHECK(qd_layout_diff_nodes(N, N0).empty());
    };
    fails("orientation", [&] { set_orientation(N, arma::vec{0.0, 0.0}); });
    fails("orientation", [&] { set_orientation(N, arma::vec{0.0, 0.0, NaN}); });
    fails("orientation", [&] { set_orientation(N, arma::vec{0.0, arma::datum::inf, 0.0}, {}, true); });
    fails("i_node", [&] { set_orientation(N, arma::vec{}, {2}); });
    fails("i_node", [&] { set_orientation(N, arma::vec{}, {0, 5}, true); });
    auto M = N;
    M.orientation.zeros(3, 2);
    CHECK(has(qd_layout_throws([&] { set_orientation(M, arma::vec{}, {}, true); }), "nodes"));
    M = N;
    M.offset = arma::uvec{0};
    CHECK(has(qd_layout_throws([&] { set_orientation(M); }), "nodes"));
}


TEST_CASE("set_orientation - heading and tilt change together")
{
    // Steps (10, 0, 0), (0, 10, 10), (-10, 0, -10): heading 0, 90, 180 deg; tilt 0, -45 (rising), +45 (descending) deg
    auto N = qd_layout_nodes_from({arma::mat{{0.0, 10.0, 10.0, 0.0}, {0.0, 0.0, 10.0, 10.0}, {0.0, 0.0, 10.0, 0.0}}});
    quadriga_lib::set_orientation(N);
    const arma::mat e = {{0.0, 0.0, 0.0, 0.0}, {0.0, -45.0 * D, 45.0 * D, 45.0 * D}, {0.0, 90.0 * D, 180.0 * D, 180.0 * D}};
    CHECK(arma::abs(N.orientation - e).max() < 1e-13);
}

TEST_CASE("set_orientation - vertical steps keep the heading")
{
    SECTION("east, then up, then up")
    {
        auto N = qd_layout_nodes_from({arma::mat{{0.0, 10.0, 10.0, 10.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 10.0, 20.0}}});
        quadriga_lib::set_orientation(N);
        CHECK(arma::abs(N.orientation.row(1) - arma::rowvec{0.0, -90.0 * D, -90.0 * D, -90.0 * D}).max() < 1e-13);
        CHECK(arma::abs(N.orientation.row(2)).max() < 1e-13);
        CHECK(arma::abs(N.orientation.row(0)).max() == 0.0);
    }
    SECTION("north, then down")
    {
        auto N = qd_layout_nodes_from({arma::mat{{0.0, 0.0, 0.0}, {0.0, 10.0, 10.0}, {0.0, 0.0, -10.0}}});
        quadriga_lib::set_orientation(N);
        CHECK(arma::abs(N.orientation.row(1) - arma::rowvec{0.0, 90.0 * D, 90.0 * D}).max() < 1e-13);
        CHECK(arma::abs(N.orientation.row(2) - arma::rowvec{90.0 * D, 90.0 * D, 90.0 * D}).max() < 1e-13);
    }
    SECTION("a vertical step at the start takes the heading of the first step that has a horizontal part")
    {
        auto N = qd_layout_nodes_from({arma::mat{{0.0, 0.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 10.0, 10.0}}});
        quadriga_lib::set_orientation(N);
        CHECK(arma::abs(N.orientation.row(1) - arma::rowvec{-90.0 * D, 0.0, 0.0}).max() < 1e-13);
        CHECK(arma::abs(N.orientation.row(2)).max() < 1e-13);
    }
    SECTION("a track that only climbs has heading 0")
    {
        auto N = qd_layout_nodes_from({arma::mat{{1.0, 1.0}, {2.0, 2.0}, {0.0, 5.0}}});
        quadriga_lib::set_orientation(N);
        CHECK(arma::abs(N.orientation.row(1) - arma::rowvec{-90.0 * D, -90.0 * D}).max() < 1e-13);
        CHECK(arma::abs(N.orientation.row(2)).max() == 0.0);
    }
}

TEST_CASE("set_orientation - a stationary track keeps its orientation")
{
    auto N = qd_layout_nodes_from({arma::mat{{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}}});
    N.orientation = arma::mat{{0.1, 0.4, 0.7}, {0.2, 0.5, 0.8}, {0.3, 0.6, 0.9}};
    const arma::mat O0 = N.orientation;
    quadriga_lib::set_orientation(N);
    CHECK(arma::abs(N.orientation - O0).max() == 0.0);
    quadriga_lib::set_orientation(N, arma::vec{0.01, 0.02, 0.03});
    arma::mat e = O0;
    e.each_col() += arma::vec{0.01, 0.02, 0.03};
    CHECK(arma::abs(N.orientation - e).max() < 1e-15);
}
