// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// sample_track: the snapshots of a node in time and space.
//
// Oracles: closed forms for each row of the sampling table (point drop with and without speed, with direction and with the
// direction taken from the orientation; track with times, track with speeds, ends held), the shortest-way interpolation
// of angles (90 deg turn, wrap at 180 deg), and, for mounted antennas, the product of the rotation matrix of the motion and
// of the stored orientation, built with the library function calc_rotation_matrix. The input values are small and typed
// by hand; no case depends on a random realisation.

#include <catch2/catch_test_macros.hpp>

#include "qd_test_layout.hpp"
#include "quadriga_layout.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace
{
const double D = arma::datum::pi / 180.0;
const double NaN = arma::datum::nan;

bool has(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }

struct snap
{
    arma::mat pos, ori;
    arma::uvec row;
};

snap sample(const quadriga_lib::layout_nodes &nodes, arma::uword i_node, const arma::vec &times = {})
{
    snap s;
    quadriga_lib::sample_track(nodes, i_node, times, s.pos, s.ori, &s.row);
    return s;
}

// Orientation of the snapshots as a rotation matrix against the expected one
double rot_err(const arma::mat &ori, arma::uword k, const arma::mat &R)
{
    return arma::abs(qd_layout_rot(ori(0, k), ori(1, k), ori(2, k)) - R).max();
}

// Track with three rows, times 1, 3, 7 s
quadriga_lib::layout_nodes timed_track()
{
    auto N = qd_layout_nodes_from({arma::mat{{0.0, 10.0, 10.0}, {0.0, 0.0, 20.0}, {0.0, 0.0, 0.0}}});
    N.orientation = arma::mat{{0.0, 0.0, 20.0 * D}, {0.0, 0.0, 10.0 * D}, {0.0, 90.0 * D, 180.0 * D}};
    N.time = {1.0, 3.0, 7.0};
    return N;
}
} // namespace

TEST_CASE("sample_track - input validation")
{
    auto N = timed_track();
    arma::mat p, o;
    arma::vec t = {0.0, 1.0};

    SECTION("node index and times")
    {
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(N, 1, t, p, o); }), "i_node"));
        t[1] = NaN;
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(N, 0, t, p, o); }), "times"));
        t[1] = arma::datum::inf;
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(N, 0, t, p, o); }), "times"));
    }
    SECTION("members that do not fit")
    {
        auto M = N;
        M.orientation.zeros(3, 2);
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }), "nodes"));
        M = N;
        M.time = {1.0, 3.0};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }), "nodes"));
        M = N;
        M.pos.set_size(2, 3);
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }), "nodes"));
    }
    SECTION("times of a track")
    {
        auto M = N;
        M.time = {1.0, 3.0, NaN};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, t, p, o); }), "time"));
        M.time = {1.0, 3.0, 3.0};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, t, p, o); }), "time"));
    }
    SECTION("track without time and speed")
    {
        auto M = N;
        M.time.reset();
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, t, p, o); }), "time"));
        CHECK(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }).empty()); // One snapshot per row needs none
        M.speed = {0.0, 0.0, 0.0};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, t, p, o); }), "speed"));
    }
    SECTION("moving step with speed 0")
    {
        auto M = N;
        M.time.reset();
        M.speed = {1.0, 0.0, 0.0};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, t, p, o); }), "speed"));
        CHECK(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }).empty());
    }
    SECTION("direction on a track")
    {
        auto M = N;
        M.direction = arma::mat{{NaN, 0.5, NaN}, {0.0, 0.0, 0.0}};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, t, p, o); }), "direction"));
    }
    SECTION("mounted antenna of a point drop needs its direction")
    {
        auto M = qd_layout_nodes_from({qd_layout_pt(1.0, 2.0, 3.0)});
        M.relative = {1};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }), "'N0'")); // The message names the node
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }), "direction"));
        M.direction = arma::vec{NaN, 0.0};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }), "direction"));
        M.direction = arma::vec{0.5, 0.0};
        CHECK(qd_layout_throws([&] { quadriga_lib::sample_track(M, 0, {}, p, o); }).empty());
    }
}

TEST_CASE("sample_track - point drop")
{
    // UT2 of the fixture: position (-50, 70, 1.5), heading 45 deg, speed 1.5 m/s, direction 120 deg azimuth, row 3
    const quadriga_lib::layout L = qd_layout_build();
    auto N = L.rx;

    SECTION("without times: one snapshot")
    {
        const snap s = sample(N, 1);
        REQUIRE(s.pos.n_cols == 1);
        CHECK(arma::abs(s.pos.col(0) - arma::vec{-50.0, 70.0, 1.5}).max() == 0.0);
        CHECK(arma::abs(s.ori.col(0) - arma::vec{0.0, 0.0, 45.0 * D}).max() == 0.0);
        CHECK(s.row.n_elem == 1);
        CHECK(s.row[0] == 3);
    }
    SECTION("with speed: p0 + v d t, constant orientation")
    {
        const arma::vec t = {0.0, 2.0, 4.0, -1.0};
        const snap s = sample(N, 1, t);
        REQUIRE(s.pos.n_cols == 4);
        for (arma::uword k = 0; k < 4; ++k)
        {
            const arma::vec e = {-50.0 + 1.5 * t[k] * std::cos(120.0 * D), 70.0 + 1.5 * t[k] * std::sin(120.0 * D), 1.5};
            CHECK(arma::abs(s.pos.col(k) - e).max() < 1e-13);
            CHECK(arma::abs(s.ori.col(k) - arma::vec{0.0, 0.0, 45.0 * D}).max() == 0.0);
            CHECK(s.row[k] == 3);
        }
    }
    SECTION("direction with elevation")
    {
        N.direction(1, 3) = 30.0 * D;
        const snap s = sample(N, 1, arma::vec{2.0});
        const arma::vec d = {std::cos(30.0 * D) * std::cos(120.0 * D), std::cos(30.0 * D) * std::sin(120.0 * D), std::sin(30.0 * D)};
        const arma::vec e = arma::vec{-50.0, 70.0, 1.5} + 3.0 * d;
        CHECK(arma::abs(s.pos.col(0) - e).max() < 1e-13);
    }
    SECTION("azimuth without elevation is horizontal")
    {
        N.direction(1, 3) = NaN;
        const snap s = sample(N, 1, arma::vec{2.0});
        CHECK(std::abs(s.pos(2, 0) - 1.5) == 0.0);
        CHECK(std::abs(s.pos(0, 0) - (-50.0 + 3.0 * std::cos(120.0 * D))) < 1e-13);
    }
    SECTION("direction from the orientation, with tilt")
    {
        N.direction(0, 3) = NaN;
        N.orientation.col(3) = arma::vec{20.0 * D, 30.0 * D, 40.0 * D}; // Bank has no effect, positive tilt points down
        const snap s = sample(N, 1, arma::vec{2.0});
        const arma::vec d = {std::cos(30.0 * D) * std::cos(40.0 * D), std::cos(30.0 * D) * std::sin(40.0 * D), -std::sin(30.0 * D)};
        const arma::vec e = arma::vec{-50.0, 70.0, 1.5} + 3.0 * d;
        CHECK(arma::abs(s.pos.col(0) - e).max() < 1e-13);
        CHECK(s.pos(2, 0) < 1.5);
        CHECK(arma::abs(s.ori.col(0) - arma::vec{20.0 * D, 30.0 * D, 40.0 * D}).max() == 0.0);
    }
    SECTION("without speed: the position at all times")
    {
        N.speed.zeros();
        const snap s = sample(N, 1, arma::vec{0.0, 5.0, 100.0});
        REQUIRE(s.pos.n_cols == 3);
        for (arma::uword k = 0; k < 3; ++k)
            CHECK(arma::abs(s.pos.col(k) - arma::vec{-50.0, 70.0, 1.5}).max() == 0.0);
        N.speed.reset();
        CHECK(sample(N, 1, arma::vec{3.0}).pos(0, 0) == -50.0);
    }
    SECTION("output sizes follow the call, not the old content")
    {
        snap s;
        s.pos.zeros(5, 7);
        s.ori.zeros(1, 1);
        s.row.zeros(9);
        quadriga_lib::sample_track(N, 1, arma::vec{1.0, 2.0}, s.pos, s.ori, &s.row);
        CHECK(s.pos.n_rows == 3);
        CHECK(s.pos.n_cols == 2);
        CHECK(s.ori.n_rows == 3);
        CHECK(s.ori.n_cols == 2);
        CHECK(s.row.n_elem == 2);
        arma::mat p, o;
        quadriga_lib::sample_track(N, 1, arma::vec{1.0, 2.0}, p, o); // Without i_row
        CHECK(arma::abs(p - s.pos).max() == 0.0);
    }
}

TEST_CASE("sample_track - rows without times")
{
    const quadriga_lib::layout L = qd_layout_build();
    // UT1: three rows without time and speed (one snapshot per row); BS2: one row at (-300, 50, 30), heading 270 deg
    const snap s = sample(L.rx, 0);
    REQUIRE(s.pos.n_cols == 3);
    CHECK(arma::abs(s.pos - L.rx.pos.cols(0, 2)).max() == 0.0);
    CHECK(arma::abs(s.ori - L.rx.orientation.cols(0, 2)).max() == 0.0);
    CHECK(arma::all(s.row == arma::uvec{0, 1, 2}));

    const snap b = sample(L.tx, 2);
    REQUIRE(b.pos.n_cols == 1);
    CHECK(arma::abs(b.pos.col(0) - arma::vec{-300.0, 50.0, 30.0}).max() == 0.0);
    CHECK(b.row[0] == 2);
    CHECK(std::abs(b.ori(2, 0) - 270.0 * D) == 0.0);
}

TEST_CASE("sample_track - track with times")
{
    // Rows at 1, 3 and 7 s; positions (0,0,0), (10,0,0), (10,20,0); headings 0, 90, 180 deg; bank 0, 0, 20 deg; tilt 0, 0, 10 deg
    auto N = timed_track();

    SECTION("linear positions, held ends, row at or before")
    {
        const arma::vec t = {0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0};
        const snap s = sample(N, 0, t);
        const arma::mat e = {{0.0, 0.0, 5.0, 10.0, 10.0, 10.0, 10.0}, {0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 20.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        REQUIRE(s.pos.n_cols == 7);
        CHECK(arma::abs(s.pos - e).max() < 1e-13);
        CHECK(arma::all(s.row == arma::uvec{0, 0, 0, 1, 1, 2, 2}));
    }
    SECTION("orientations: shortest-way interpolation of each angle")
    {
        const snap s = sample(N, 0, arma::vec{0.0, 2.0, 3.0, 5.0, 9.0});
        const arma::mat e = {{0.0, 0.0, 0.0, 10.0 * D, 20.0 * D}, {0.0, 0.0, 0.0, 5.0 * D, 10.0 * D}, {0.0, 45.0 * D, 90.0 * D, 135.0 * D, 180.0 * D}};
        CHECK(arma::abs(s.ori - e).max() < 1e-13);
    }
    SECTION("times of the snapshots need not increase")
    {
        const snap s = sample(N, 0, arma::vec{5.0, 2.0});
        CHECK(std::abs(s.pos(1, 0) - 10.0) < 1e-13);
        CHECK(std::abs(s.pos(0, 1) - 5.0) < 1e-13);
    }
    SECTION("without times: the rows")
    {
        const snap s = sample(N, 0);
        CHECK(arma::abs(s.pos - N.pos).max() == 0.0);
        CHECK(arma::abs(s.ori - N.orientation).max() == 0.0);
        CHECK(arma::all(s.row == arma::uvec{0, 1, 2}));
    }
    SECTION("the node is at its offset in the arrays")
    {
        auto M = qd_layout_nodes_from({qd_layout_pt(100.0, 100.0, 0.0), N.pos});
        M.orientation.cols(1, 3) = N.orientation;
        M.time = {NaN, 1.0, 3.0, 7.0};
        const snap s = sample(M, 1, arma::vec{2.0, 9.0});
        CHECK(std::abs(s.pos(0, 0) - 5.0) < 1e-13);
        CHECK(arma::all(s.row == arma::uvec{1, 3}));
        CHECK(std::abs(s.ori(2, 0) - 45.0 * D) < 1e-13);
    }
}

TEST_CASE("sample_track - angles take the shorter way")
{
    auto N = qd_layout_nodes_from({arma::mat{{0.0, 10.0}, {0.0, 0.0}, {0.0, 0.0}}});
    N.time = {0.0, 4.0};

    SECTION("a turn by 90 deg")
    {
        N.orientation = arma::mat{{0.0, 0.0}, {0.0, 0.0}, {0.0, 90.0 * D}};
        const snap s = sample(N, 0, arma::vec{1.0, 2.0, 3.0});
        CHECK(arma::abs(s.ori.row(2) - arma::rowvec{22.5 * D, 45.0 * D, 67.5 * D}).max() < 1e-13);
    }
    SECTION("across 180 deg")
    {
        N.orientation = arma::mat{{0.0, 0.0}, {0.0, 0.0}, {150.0 * D, -150.0 * D}};
        const snap s = sample(N, 0, arma::vec{1.0, 2.0, 3.0});
        // 150 -> 180 -> 210 deg: 165, 180, -165 deg (180 is -180 or 180)
        CHECK(std::abs(qd_layout_wrap(s.ori(2, 0) - 165.0 * D)) < 1e-12);
        CHECK(std::abs(qd_layout_wrap(s.ori(2, 1) - 180.0 * D)) < 1e-12);
        CHECK(std::abs(qd_layout_wrap(s.ori(2, 2) + 165.0 * D)) < 1e-12);
        CHECK(arma::abs(s.ori.row(2)).max() <= arma::datum::pi + 1e-12);
    }
    SECTION("each angle on its own")
    {
        N.orientation = arma::mat{{-170.0 * D, 170.0 * D}, {-30.0 * D, 50.0 * D}, {10.0 * D, 20.0 * D}};
        const snap s = sample(N, 0, arma::vec{2.0});
        CHECK(std::abs(qd_layout_wrap(s.ori(0, 0) - arma::datum::pi)) < 1e-12); // -170 -> 190: the middle is 180 deg
        CHECK(std::abs(s.ori(1, 0) - 10.0 * D) < 1e-13);
        CHECK(std::abs(s.ori(2, 0) - 15.0 * D) < 1e-13);
    }
}

TEST_CASE("sample_track - several angles change together")
{
    // Tilt 0 -> 60 deg and heading 0 -> 150 deg in 4 s: each angle on its own, in the middle 30 and 75 deg
    auto N = qd_layout_nodes_from({arma::mat{{0.0, 10.0}, {0.0, 0.0}, {0.0, 0.0}}});
    N.time = {0.0, 4.0};
    N.orientation = arma::mat{{0.0, 0.0}, {0.0, 60.0 * D}, {0.0, 150.0 * D}};
    const snap s = sample(N, 0, arma::vec{2.0});
    CHECK(std::abs(s.ori(0, 0)) < 1e-13);
    CHECK(std::abs(s.ori(1, 0) - 30.0 * D) < 1e-13);
    CHECK(std::abs(s.ori(2, 0) - 75.0 * D) < 1e-13);

    // The tilt is not periodic: from -90 to +90 deg it passes 0, not 180 deg
    N.orientation = arma::mat{{0.0, 0.0}, {-90.0 * D, 90.0 * D}, {0.0, 0.0}};
    const snap t = sample(N, 0, arma::vec{1.0, 2.0, 3.0});
    CHECK(arma::abs(t.ori.row(1) - arma::rowvec{-45.0 * D, 0.0, 45.0 * D}).max() < 1e-13);
}

TEST_CASE("sample_track - track with speeds")
{
    // Steps of 5 m (speed 1 m/s) and 10 m (speed 2 m/s): the rows are at 0, 5 and 10 s
    auto N = qd_layout_nodes_from({arma::mat{{0.0, 3.0, 3.0}, {0.0, 4.0, 4.0}, {0.0, 0.0, 10.0}}});
    N.orientation = arma::mat{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 60.0 * D, 60.0 * D}};
    N.speed = {1.0, 2.0, 0.0};

    SECTION("time from the arc length")
    {
        const snap s = sample(N, 0, arma::vec{0.0, 2.5, 5.0, 7.5, 10.0, 12.0});
        const arma::mat e = {{0.0, 1.5, 3.0, 3.0, 3.0, 3.0}, {0.0, 2.0, 4.0, 4.0, 4.0, 4.0}, {0.0, 0.0, 0.0, 5.0, 10.0, 10.0}};
        CHECK(arma::abs(s.pos - e).max() < 1e-13);
        CHECK(arma::all(s.row == arma::uvec{0, 0, 1, 1, 2, 2}));
        CHECK(std::abs(s.ori(2, 1) - 30.0 * D) < 1e-13);
        CHECK(std::abs(s.ori(2, 3) - 60.0 * D) < 1e-13);
    }
    SECTION("a step of zero length with speed 0 takes no time; the later row holds at that time")
    {
        auto M = qd_layout_nodes_from({arma::mat{{0.0, 0.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}});
        M.orientation = arma::mat{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {10.0 * D, 20.0 * D, 40.0 * D}};
        M.speed = {0.0, 1.0, 1.0};
        const snap s = sample(M, 0, arma::vec{0.0, 2.5, 5.0});
        CHECK(arma::all(s.row == arma::uvec{1, 1, 2}));
        CHECK(std::abs(s.pos(0, 1) - 2.5) < 1e-13);
        CHECK(std::abs(s.pos(0, 2) - 5.0) < 1e-13);
        CHECK(std::abs(s.ori(2, 0) - 20.0 * D) < 1e-13);
        CHECK(std::abs(s.ori(2, 1) - 30.0 * D) < 1e-13);
    }
    SECTION("a node with times uses them, not the arc length")
    {
        N.speed.reset();
        N.time = {0.0, 1.0, 2.0};
        const snap s = sample(N, 0, arma::vec{0.5});
        CHECK(std::abs(s.pos(0, 0) - 1.5) < 1e-13);
    }
}

TEST_CASE("sample_track - mounted antennas on a track")
{
    // Right-angle turn: 10 m along x, then 10 m along y; the antenna is mounted with a heading of 30 deg
    auto N = qd_layout_nodes_from({arma::mat{{0.0, 10.0, 10.0}, {0.0, 0.0, 10.0}, {0.0, 0.0, 0.0}}});
    N.relative = {1};

    SECTION("heading: the motion of the segment that holds the row, the last segment repeated")
    {
        N.orientation.row(2).fill(30.0 * D);
        const snap s = sample(N, 0);
        CHECK(arma::abs(s.ori.row(2) - arma::rowvec{30.0 * D, 120.0 * D, 120.0 * D}).max() < 1e-13);
        CHECK(arma::abs(s.ori.row(0)).max() < 1e-13);
        CHECK(arma::abs(s.ori.row(1)).max() < 1e-13);
    }
    SECTION("bank is relative to the vehicle")
    {
        N.orientation.row(0).fill(20.0 * D);
        const snap s = sample(N, 0);
        CHECK(arma::abs(s.ori.row(0) - arma::rowvec{20.0 * D, 20.0 * D, 20.0 * D}).max() < 1e-13);
        CHECK(arma::abs(s.ori.row(2) - arma::rowvec{0.0, 90.0 * D, 90.0 * D}).max() < 1e-13);
    }
    SECTION("rising track without a mounting rotation: the boresight is along the motion")
    {
        auto M = qd_layout_nodes_from({arma::mat{{0.0, 10.0}, {0.0, 0.0}, {0.0, 10.0}}});
        M.relative = {1};
        const snap s = sample(M, 0);
        const arma::mat R = qd_layout_rot(s.ori(0, 1), s.ori(1, 1), s.ori(2, 1));
        CHECK(std::abs(R(0, 0) - std::sqrt(0.5)) < 1e-13); // First column of R = boresight in the GCS
        CHECK(std::abs(R(1, 0)) < 1e-13);
        CHECK(std::abs(R(2, 0) - std::sqrt(0.5)) < 1e-13);
    }
    SECTION("vertical step: the mounting follows the tilt of -90 deg, the heading of the step before")
    {
        auto M = qd_layout_nodes_from({arma::mat{{0.0, 10.0, 10.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 10.0}}});
        M.relative = {1};
        M.orientation = arma::mat{{90.0 * D, 90.0 * D, 90.0 * D}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}; // Bank 90 deg on the vehicle
        const snap s = sample(M, 0);
        const arma::mat R = qd_layout_rot(0.0, -90.0 * D, 0.0) * qd_layout_rot(90.0 * D, 0.0, 0.0);
        CHECK(rot_err(s.ori, 1, R) < 1e-12);
        CHECK(rot_err(s.ori, 2, R) < 1e-12);
        CHECK(rot_err(s.ori, 0, qd_layout_rot(90.0 * D, 0.0, 0.0)) < 1e-12);
    }
    SECTION("without a mounting rotation the antenna looks along the motion")
    {
        const snap s = sample(N, 0);
        CHECK(arma::abs(s.ori.row(2) - arma::rowvec{0.0, 90.0 * D, 90.0 * D}).max() < 1e-13);
    }
    SECTION("rising track: R(motion) R(orientation)")
    {
        auto M = qd_layout_nodes_from({arma::mat{{0.0, 10.0}, {0.0, 0.0}, {0.0, 10.0}}}); // Rises by 45 deg, tilt -45 deg
        M.relative = {1};
        M.orientation = arma::mat{{10.0 * D, 10.0 * D}, {5.0 * D, 5.0 * D}, {30.0 * D, 30.0 * D}};
        const snap s = sample(M, 0);
        const arma::mat R = qd_layout_rot(0.0, -45.0 * D, 0.0) * qd_layout_rot(10.0 * D, 5.0 * D, 30.0 * D);
        for (arma::uword k = 0; k < 2; ++k)
            CHECK(rot_err(s.ori, k, R) < 1e-13);
    }
    SECTION("heading 90 deg with 30 deg tilt")
    {
        auto M = qd_layout_nodes_from({arma::mat{{0.0, 0.0}, {0.0, 10.0}, {0.0, -10.0}}}); // Along y, descending: tilt +45 deg
        M.relative = {1};
        M.orientation = arma::mat{{0.0, 0.0}, {30.0 * D, 30.0 * D}, {0.0, 0.0}};
        const snap s = sample(M, 0);
        const arma::mat R = qd_layout_rot(0.0, 45.0 * D, 90.0 * D) * qd_layout_rot(0.0, 30.0 * D, 0.0);
        CHECK(rot_err(s.ori, 0, R) < 1e-13);
        CHECK(rot_err(s.ori, 1, R) < 1e-13);
        // The pure product R(motion) Ry(tilt) has bank 0 and tilt 75 deg
        CHECK(std::abs(s.ori(1, 0) - 75.0 * D) < 1e-12);
        CHECK(std::abs(s.ori(2, 0) - 90.0 * D) < 1e-12);
    }
    SECTION("steps of zero length: the previous motion, at the start the first motion")
    {
        auto M = qd_layout_nodes_from({arma::mat{{0.0, 10.0, 10.0, 10.0}, {0.0, 0.0, 0.0, 10.0}, {0.0, 0.0, 0.0, 0.0}}});
        M.relative = {1};
        CHECK(arma::abs(sample(M, 0).ori.row(2) - arma::rowvec{0.0, 0.0, 90.0 * D, 90.0 * D}).max() < 1e-13);
        auto Q = qd_layout_nodes_from({arma::mat{{0.0, 0.0, 0.0}, {0.0, 0.0, 10.0}, {0.0, 0.0, 0.0}}});
        Q.relative = {1};
        CHECK(arma::abs(sample(Q, 0).ori.row(2) - arma::rowvec{90.0 * D, 90.0 * D, 90.0 * D}).max() < 1e-13);
    }
    SECTION("with times: the stored orientation is interpolated, then rotated by the motion of the segment")
    {
        N.orientation = arma::mat{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 40.0 * D, 60.0 * D}};
        N.time = {0.0, 10.0, 20.0};
        const snap s = sample(N, 0, arma::vec{-5.0, 5.0, 10.0, 15.0, 30.0});
        // Segment 0 (heading 0) holds the times up to 10 s exclusive, segment 1 (heading 90 deg) the rest; the ends are held
        const arma::rowvec e = {0.0, 20.0 * D, 130.0 * D, 140.0 * D, 150.0 * D};
        CHECK(arma::abs(s.ori.row(2) - e).max() < 1e-13);
        CHECK(arma::abs(s.pos.row(0) - arma::rowvec{0.0, 5.0, 10.0, 10.0, 10.0}).max() < 1e-13);
        CHECK(arma::all(s.row == arma::uvec{0, 0, 1, 1, 2}));
    }
}

TEST_CASE("sample_track - mounted antennas on a point drop")
{
    auto N = qd_layout_nodes_from({qd_layout_pt(1.0, 2.0, 3.0)});
    N.relative = {1};
    N.orientation = qd_layout_pt(0.0, 0.0, 30.0 * D);
    N.direction = arma::vec{90.0 * D, 0.0};
    N.speed = {2.0};

    SECTION("heading of the direction plus the mounting heading, constant in time")
    {
        const snap s = sample(N, 0, arma::vec{0.0, 3.0});
        CHECK(arma::abs(s.pos.col(1) - arma::vec{1.0, 8.0, 3.0}).max() < 1e-13);
        CHECK(arma::abs(s.ori.row(2) - arma::rowvec{120.0 * D, 120.0 * D}).max() < 1e-13);
        CHECK(arma::abs(s.ori.row(1)).max() < 1e-13);
    }
    SECTION("elevation of the direction is a negative tilt")
    {
        N.direction = arma::vec{0.0, 30.0 * D};
        N.orientation.zeros();
        const snap s = sample(N, 0);
        CHECK(std::abs(s.ori(1, 0) + 30.0 * D) < 1e-13);
        CHECK(std::abs(s.ori(2, 0)) < 1e-13);
    }
    SECTION("vertical direction: no heading")
    {
        N.direction = arma::vec{0.7, 90.0 * D};
        N.orientation.zeros();
        const snap s = sample(N, 0);
        CHECK(std::abs(s.ori(1, 0) + 90.0 * D) < 1e-12);
        CHECK(std::abs(s.ori(0, 0)) < 1e-12);
    }
    SECTION("not mounted: the orientation as stored")
    {
        N.relative = {0};
        const snap s = sample(N, 0);
        CHECK(arma::abs(s.ori.col(0) - arma::vec{0.0, 0.0, 30.0 * D}).max() == 0.0);
    }
}
