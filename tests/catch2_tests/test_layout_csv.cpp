// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// layout_csv_read, layout_csv_write and the is_valid members of the layout classes: the column classes (identity,
// fixed, per row, held), extra sectors, units, the composition of the antenna columns and of the band rows, the
// columns that the layout holds (seeds, pairing, wrap, scenario parameters), unknown columns, the messages with file and
// line, round trips, the all-or-nothing replacement of files, and each rule of is_valid.
//
// Oracles: CSV texts and the values they must give, typed by hand from the column rules; angles are 'deg' times
// pi / 180; antenna lines and the file text of the writer are typed by hand. The round trips use the reader and the
// writer against each other (the supporting function is the other one, checked by the typed cases). The layouts are
// read without evaluating the antennas (test_set_antenna.cpp covers the evaluation), so no antenna array is built here.

#include <catch2/catch_test_macros.hpp>

#include "qd_test_layout.hpp"
#include "quadriga_layout.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace
{
const double D = arma::datum::pi / 180.0;
const double NaN = arma::datum::nan;

// Reads the two texts through files ("" = no file for the side)
quadriga_lib::layout read_text(const qd_layout_files &dir, const std::string &tx, const std::string &rx, const arma::vec &freq = {},
                               std::vector<std::string> *extra_tx = nullptr, std::vector<std::string> *extra_rx = nullptr)
{
    const std::string fn_tx = tx.empty() ? "" : dir.write("tx.csv", tx), fn_rx = rx.empty() ? "" : dir.write("rx.csv", rx);
    return quadriga_lib::layout_csv_read(fn_tx, fn_rx, freq, false, extra_tx, extra_rx);
}

bool has(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }

const char *basic_tx = "name,sector,x,y,z,heading,tilt\nBS1,0,0,0,25,30,10\nBS1,1,,,,150,\nBS1,2,,,,270,\nBS2,0,500,0,25,90,0\n";
const char *basic_rx = "name,x,y,z,heading,speed,direction_az,scenario\n"
                       "UT1,10,20,1.5,45,0.833,120,UMa\n"
                       "UT2,0,0,1.5,0,,,\n"
                       "UT2,1,0,1.5,,,,\n"
                       "UT2,2,0,1.5,90,,,\n"
                       "UT2,2,1,1.5,,,,UMa_NLOS\n";
} // namespace

TEST_CASE("layout_csv_read - rows, units, held values and extra sectors")
{
    qd_layout_files dir("read_basic");
    const quadriga_lib::layout L = read_text(dir, basic_tx, basic_rx);
    CHECK(L.is_valid() == "");

    // Tx: three sectors of a static site and one more site; the empty cells of extra sectors take the values of sector 0
    const quadriga_lib::layout_nodes &T = L.tx;
    REQUIRE(T.n_node() == 4);
    CHECK(T.name == std::vector<std::string>{"BS1", "BS1", "BS1", "BS2"});
    CHECK(arma::all(T.sector == arma::u32_vec{0, 1, 2, 0}));
    CHECK(arma::all(T.offset == arma::uvec{0, 1, 2, 3}));
    CHECK(T.n_row() == 4);
    CHECK(T.n_row(2) == 1);
    const arma::mat pos = {{0, 0, 0, 500}, {0, 0, 0, 0}, {25, 25, 25, 25}};
    CHECK(arma::approx_equal(T.pos, pos, "absdiff", 0.0));
    const arma::vec heading = arma::vec{30, 150, 270, 90} * D, tilt = arma::vec{10, 10, 10, 0} * D;
    CHECK(arma::approx_equal(T.orientation.row(2).t(), heading, "absdiff", 1e-14));
    CHECK(arma::approx_equal(T.orientation.row(1).t(), tilt, "absdiff", 1e-14));
    CHECK(arma::all(T.orientation.row(0) == 0.0));
    CHECK(T.speed.is_empty());
    CHECK(T.direction.is_empty());
    CHECK(T.scenario.empty());
    CHECK(T.clutter.is_empty());
    CHECK(T.seed.is_empty());
    CHECK(arma::all(T.i_antenna == 0u));

    // Rx: a point drop with speed and direction, and a track with held cells
    const quadriga_lib::layout_nodes &R = L.rx;
    REQUIRE(R.n_node() == 2);
    CHECK(R.name == std::vector<std::string>{"UT1", "UT2"});
    CHECK(arma::all(R.offset == arma::uvec{0, 1}));
    CHECK(R.n_row(1) == 4);
    CHECK(arma::approx_equal(R.pos.col(4), arma::vec{2, 1, 1.5}, "absdiff", 0.0));
    CHECK(arma::approx_equal(R.orientation.row(2).t(), arma::vec{45, 0, 0, 90, 90} * D, "absdiff", 1e-14));
    CHECK(arma::approx_equal(R.speed, arma::vec{0.833, 0, 0, 0, 0}, "absdiff", 0.0));
    REQUIRE(R.direction.n_cols == 5);
    CHECK(std::abs(R.direction(0, 0) - 120.0 * D) < 1e-14);
    CHECK(R.direction(1, 0) == 0.0);
    CHECK(std::isnan(R.direction(0, 1)));
    CHECK(R.scenario == std::vector<std::string>{"UMa", "", "", "", "UMa_NLOS"});
    CHECK(L.name.empty());
    CHECK(L.links.empty());
    CHECK(L.antenna.empty());
    CHECK(L.antenna_args.empty());
    CHECK(L.center_freq.is_empty());
    CHECK(L.seed == -1);
}

TEST_CASE("layout_csv_read - one side only, empty files and column order")
{
    qd_layout_files dir("read_order");
    const quadriga_lib::layout L = read_text(dir, basic_tx, basic_rx);

    // Only one file
    const quadriga_lib::layout Lt = read_text(dir, basic_tx, "");
    CHECK(Lt.tx.n_node() == 4);
    CHECK(Lt.rx.n_node() == 0);
    CHECK(Lt.is_valid() == "");
    const quadriga_lib::layout Lr = read_text(dir, "", basic_rx);
    CHECK(Lr.tx.n_node() == 0);
    CHECK(qd_layout_diff_nodes(Lr.rx, L.rx, 0.0) == "");
    CHECK(quadriga_lib::layout_csv_read("", "", {}, false).tx.n_node() == 0);

    // A file with a header only has no nodes
    const quadriga_lib::layout Le = read_text(dir, "name,x,y,z\n", "# a comment\nname,x,y,z\n\n");
    CHECK(Le.tx.n_node() == 0);
    CHECK(Le.rx.n_node() == 0);
    CHECK(Le.is_valid() == "");

    // The order of the columns does not matter
    const std::string tx = "tilt,z,heading,sector,y,x,name\n10,25,30,0,0,0,BS1\n,,150,1,,,BS1\n,,270,2,,,BS1\n0,25,90,0,0,500,BS2\n";
    const std::string rx = "scenario,direction_az,speed,heading,z,y,x,name\n"
                           "UMa,120,0.833,45,1.5,20,10,UT1\n"
                           ",,,0,1.5,0,0,UT2\n"
                           ",,,,1.5,0,1,UT2\n"
                           ",,,90,1.5,0,2,UT2\n"
                           "UMa_NLOS,,,,1.5,1,2,UT2\n";
    CHECK(qd_layout_diff(read_text(dir, tx, rx), L) == "");

    // Rows of one node need not be consecutive; the file order is the track order
    const quadriga_lib::layout Ls = read_text(dir, "", "name,x,y,z\nA,1,0,0\nB,5,5,5\nA,2,0,0\nB,6,5,5\nA,3,0,0\n");
    REQUIRE(Ls.rx.n_node() == 2);
    CHECK(Ls.rx.name == std::vector<std::string>{"A", "B"});
    CHECK(arma::all(Ls.rx.offset == arma::uvec{0, 3}));
    CHECK(arma::approx_equal(Ls.rx.pos.row(0).t(), arma::vec{1, 2, 3, 5, 6}, "absdiff", 0.0));
}

TEST_CASE("layout_csv_read - column classes")
{
    qd_layout_files dir("read_class");

    SECTION("Fixed: the first row, identical repeats and empty cells")
    {
        const quadriga_lib::layout L = read_text(dir, "", "name,x,y,z,relative\nA,0,0,0,1\nA,1,0,0,\nA,2,0,0,1\nB,0,0,0,\n");
        REQUIRE(L.rx.relative.n_elem == 2);
        CHECK(arma::all(L.rx.relative == arma::u32_vec{1, 0}));
        // A later value that differs from the first one
        const std::string msg = qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,relative\nA,0,0,0,1\nA,1,0,0,\nA,2,0,0,0\n"); });
        CHECK(has(msg, "rx.csv, line 4, column 'relative'"));
        CHECK(has(msg, "differs from the first row"));
        // An empty first cell is the default; a later value differs from it
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,relative\nA,0,0,0,\nA,1,0,0,1\n"); }), "line 3, column 'relative'"));
    }

    SECTION("Per row: positions need a value, time is given on all rows or on none")
    {
        std::string msg = qd_layout_throws([&] { read_text(dir, "", "name,x,y,z\nA,0,0,0\nA,1,,0\n"); });
        CHECK(has(msg, "rx.csv, line 3, column 'y' is empty"));
        msg = qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,time\nA,0,0,0,0\nA,1,0,0,\n"); });
        CHECK(has(msg, "line 3, column 'time'"));
        const quadriga_lib::layout L = read_text(dir, "", "name,x,y,z,time\nA,0,0,0,0\nA,1,0,0,0.5\nB,0,0,0,\nB,1,0,0,\nC,0,0,0,7\n");
        CHECK(qd_layout_same_mat(L.rx.time, arma::vec{0, 0.5, NaN, NaN, 7}, 0.0));
        // Non-finite positions
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z\nA,NaN,0,0\n"); }), "line 2, column 'x'"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z\nA,0,Inf,0\n"); }), "line 2, column 'y'"));
    }

    SECTION("Held: an empty cell repeats the previous row, NaN is a value")
    {
        const std::string rx = "name,x,y,z,indoor,d2d_in,floor,scenario,speed\n"
                               "A,0,0,0,1,20,3,UMa,2\n"
                               "A,1,0,0,0,,,,\n"
                               "B,0,0,0,1,20,3,UMa,2\n"
                               "B,1,0,0,0,NaN,,UMa_NLOS,0\n";
        const quadriga_lib::layout L = read_text(dir, "", rx);
        CHECK(arma::all(L.rx.indoor == arma::s32_vec{1, 0, 1, 0}));
        CHECK(qd_layout_same_mat(L.rx.d2d_in, arma::vec{20, 20, 20, NaN}, 0.0));
        CHECK(arma::all(L.rx.floor == arma::s32_vec{3, 3, 3, 3}));
        CHECK(L.rx.scenario == std::vector<std::string>{"UMa", "UMa", "UMa", "UMa_NLOS"});
        CHECK(arma::approx_equal(L.rx.speed, arma::vec{2, 2, 2, 0}, "absdiff", 0.0));
        // The first row of a node starts from the default, not from the previous node
        const quadriga_lib::layout L2 = read_text(dir, "", "name,x,y,z,heading\nA,0,0,0,90\nB,0,0,0,\n");
        CHECK(std::abs(L2.rx.orientation(2, 0) - 90.0 * D) < 1e-14);
        CHECK(L2.rx.orientation(2, 1) == 0.0);
    }

    SECTION("Identity: name is required, sector is empty or an integer from 0")
    {
        const quadriga_lib::layout L = read_text(dir, "", "name,sector,x,y,z\nA,,0,0,0\nA,2,0,0,0\n");
        CHECK(arma::all(L.rx.sector == arma::u32_vec{0, 2}));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z\n,0,0,0\n"); }), "line 2, column 'name'"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,sector,x,y,z\nA,-1,0,0,0\n"); }), "line 2, column 'sector'"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,sector,x,y,z\nA,1.5,0,0,0\n"); }), "must be an integer"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,y,z\nA,0,0\n"); }), "required column 'x' is missing"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "x,y,z\n0,0,0\n"); }), "required column 'name' is missing"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z\n" + std::string(256, 'n') + ",0,0,0\n"); }), "1 to 255 bytes"));
    }

    SECTION("Values outside their range are rejected with the line")
    {
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,speed\nA,0,0,0,-1\n"); }), "line 2, column 'speed'"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,indoor\nA,0,0,0,4\n"); }), "line 2, column 'indoor'"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,heading\nA,0,0,0,abc\n"); }), "line 2, column 'heading' is not numeric"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,heading\nA,0,0,0,Inf\n"); }), "line 2, column 'heading'"));
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,d2d_in\nA,0,0,0,-3\n"); }), "line 2, column 'd2d_in'"));
    }
}

TEST_CASE("layout_csv_read - extra sectors")
{
    qd_layout_files dir("read_sector");

    // A moving site: all rows of each sector have positions
    const std::string tx = "name,sector,x,y,z,heading\nBS,0,0,0,25,10\nBS,0,5,0,25,\nBS,1,0,0,25,20\nBS,1,5,0,25,\n";
    const quadriga_lib::layout L = read_text(dir, tx, "");
    CHECK(L.tx.n_node() == 2);
    CHECK(arma::approx_equal(L.tx.orientation.row(2).t(), arma::vec{10, 10, 20, 20} * D, "absdiff", 1e-14));

    // An empty position is allowed only for the single row of an extra sector of a static site
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,sector,x,y,z\nBS,0,0,0,25\nBS,0,5,0,25\nBS,1,,,\n", ""); }), "line 4, column 'x' is empty"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,sector,x,y,z\nBS,0,0,0,25\nBS,1,,,\nBS,1,,,\n", ""); }), "line 3, column 'x' is empty"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,sector,x,y,z\nBS,1,,,\n", ""); }), "line 2, column 'x' is empty"));
    // Sector 0 may come later in the file
    const quadriga_lib::layout L2 = read_text(dir, "name,sector,x,y,z,heading,relative,direction_az\nBS,1,,,,40,,10\nBS,0,7,8,9,20,1,10\n", "");
    CHECK(arma::approx_equal(L2.tx.pos.col(0), arma::vec{7, 8, 9}, "absdiff", 0.0));
    CHECK(arma::all(L2.tx.relative == arma::u32_vec{1, 1}));
    // The shared values of a site must agree
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,sector,x,y,z,d_clutter\nBS,0,0,0,25,5\nBS,1,,,,6\n", ""); }), "must have equal 'clutter'"));
    // An empty cell of an extra sector is the value of sector 0; a value on an extra sector only is a difference
    const quadriga_lib::layout L3 = read_text(dir, "name,sector,x,y,z,d_clutter,seed\nBS,0,0,0,25,6,11\nBS,1,,,,,\n", "");
    CHECK(arma::all(L3.tx.clutter.row(0) == 6.0));
    CHECK(arma::all(L3.tx.seed == arma::s64_vec{11, 11}));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,sector,x,y,z,d_clutter\nBS,0,0,0,25,\nBS,1,,,,6\n", ""); }), "must have equal 'clutter'"));
    // The same (name, sector) twice in one node is one node
    CHECK(read_text(dir, "name,sector,x,y,z\nBS,0,0,0,25\nBS,0,1,0,25\n", "").tx.n_node() == 1);
}

TEST_CASE("layout_csv_read - direction, indoor state and scenario")
{
    qd_layout_files dir("read_dir");

    // Empty and NaN in a first row both mean: follow the orientation (no direction is stored)
    CHECK(read_text(dir, "", "name,x,y,z,direction_az,direction_el\nA,0,0,0,,\n").rx.direction.is_empty());
    CHECK(read_text(dir, "", "name,x,y,z,direction_az\nA,0,0,0,NaN\n").rx.direction.is_empty());
    CHECK(read_text(dir, "", "name,x,y,z,heading\nA,0,0,0,10\n").rx.direction.is_empty());

    // Azimuth and elevation of a point drop in degrees; positive elevation is up
    const quadriga_lib::layout L = read_text(dir, "", "name,x,y,z,direction_az,direction_el,speed\nA,0,0,0,200,-15,3\nB,0,0,0,,,\n");
    REQUIRE(L.rx.direction.n_cols == 2);
    CHECK(std::abs(L.rx.direction(0, 0) - 200.0 * D) < 1e-14);
    CHECK(std::abs(L.rx.direction(1, 0) + 15.0 * D) < 1e-14);
    CHECK(std::isnan(L.rx.direction(0, 1)));

    // An empty cell holds the previous direction, so a track with a direction is rejected, with NaN as with a value
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,direction_az\nA,0,0,0,90\nA,1,0,0,\n"); }), "'direction'"));
    CHECK(read_text(dir, "", "name,x,y,z,direction_az\nA,0,0,0,NaN\nA,1,0,0,\n").rx.direction.is_empty());

    // A scenario is a family or a conf name, never a path
    CHECK(read_text(dir, "", "name,x,y,z,scenario\nA,0,0,0,UMa_NLOS\n").rx.scenario[0] == "UMa_NLOS");
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,scenario\nA,0,0,0,../UMa\n"); }), "path separator"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,scenario\nA,0,0,0,a\\b\n"); }), "path separator"));

    // Floor and indoor distance change only where the scenario or the indoor state changes
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,floor\nA,0,0,0,2\nA,1,0,0,3\n"); }), "'floor' and 'd2d_in'"));
    const quadriga_lib::layout L2 = read_text(dir, "", "name,x,y,z,floor,indoor\nA,0,0,0,2,1\nA,1,0,0,3,3\n");
    CHECK(arma::all(L2.rx.floor == arma::s32_vec{2, 3}));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,floor\nA,0,0,0,0\n"); }), "'floor'"));
}

TEST_CASE("layout_csv_read - time and speed")
{
    qd_layout_files dir("read_time");
    const quadriga_lib::layout L = read_text(dir, "", "name,x,y,z,time\nA,0,0,0,0\nA,1,0,0,0.5\nA,2,0,0,2\n");
    CHECK(arma::approx_equal(L.rx.time, arma::vec{0, 0.5, 2}, "absdiff", 0.0));
    CHECK(L.rx.speed.is_empty());
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,time\nA,0,0,0,1\nA,1,0,0,1\n"); }), "'time' must increase"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,time,speed\nA,0,0,0,0,1\nA,1,0,0,1,\n"); }), "either 'time' or 'speed'"));
    // Moving at speed 0 without time is allowed only if the node has no speed at all
    CHECK(read_text(dir, "", "name,x,y,z\nA,0,0,0\nA,1,0,0\n").rx.n_row() == 2);
    CHECK(read_text(dir, "", "name,x,y,z,speed\nA,0,0,0,0\nA,1,0,0,\nB,0,0,0,2\nB,0,0,0,\n").rx.n_row() == 4);
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,speed\nA,0,0,0,1\nA,1,0,0,0\nA,2,0,0,\n"); }), "'speed' is 0"));
}

TEST_CASE("layout_csv_read - antenna columns")
{
    qd_layout_files dir("read_ant");
    const std::string tx = "name,x,y,z,antenna,panel\nBS1,0,0,25,3gpp,\"4,4,1,2\"\nBS2,50,0,25,3gpp,\"4,4,1,2\"\nBS3,100,0,25,dipole,\n";
    const std::string rx = "name,x,y,z,antenna\nUT1,1,1,1.5,dipole\nUT2,2,2,2,\nUT3,3,3,1.5,omni\n";
    const quadriga_lib::layout L = read_text(dir, tx, rx);

    // Equal text is one antenna; Tx nodes first, then Rx nodes; no antenna cells = generic antenna
    CHECK(L.antenna_cols == std::vector<std::string>{"antenna", "panel"});
    const std::vector<std::vector<std::string>> expect = {{"3gpp,\"4,4,1,2\""}, {"dipole,"}, {"omni,"}};
    CHECK(L.antenna_args == expect);
    CHECK(arma::all(L.tx.i_antenna == arma::u32_vec{1, 1, 2}));
    CHECK(arma::all(L.rx.i_antenna == arma::u32_vec{2, 0, 3}));
    CHECK(L.n_antenna() == 3);
    CHECK(L.antenna.empty());
    CHECK(L.is_valid() == "");

    // Read again: the same lists
    const quadriga_lib::layout L2 = quadriga_lib::layout_csv_read(dir.path("tx.csv"), dir.path("rx.csv"), {}, false);
    CHECK(qd_layout_diff(L, L2) == "");

    // Antenna columns of a position row repeat the first row or are empty
    const quadriga_lib::layout L3 = read_text(dir, "", "name,x,y,z,antenna,panel\nA,0,0,0,3gpp,\"2,2\"\nA,1,0,0,,\"2,2\"\nA,2,0,0,3gpp,\n");
    CHECK(L3.antenna_args == std::vector<std::vector<std::string>>{{"3gpp,\"2,2\""}});
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,antenna,panel\nA,0,0,0,3gpp,\"2,2\"\nA,1,0,0,3gpp,\"2,3\"\n"); }), "line 3, column 'panel'"));

    // Columns that no antenna uses are not in antenna_cols
    CHECK(read_text(dir, "", "name,x,y,z,antenna,panel\nA,0,0,0,omni,\n").antenna_cols == std::vector<std::string>{"antenna"});
    CHECK(read_text(dir, "", "name,x,y,z,antenna,panel\nA,0,0,0,,\n").antenna_cols.empty());
    // An extra sector without antenna cells has the antenna of sector 0
    const quadriga_lib::layout L4 = read_text(dir, "name,sector,x,y,z,antenna\nA,0,0,0,0,dipole\nA,1,,,,\nB,0,5,5,5,\n", "");
    CHECK(arma::all(L4.tx.i_antenna == arma::u32_vec{1, 1, 0}));
    const quadriga_lib::layout L5 = read_text(dir, "name,sector,x,y,z,antenna\nA,0,0,0,0,dipole\nA,1,,,,omni\n", "");
    CHECK(arma::all(L5.tx.i_antenna == arma::u32_vec{1, 2}));
}

TEST_CASE("layout_csv_read - antenna columns are checked against the type")
{
    qd_layout_files dir("read_ant_check");
    auto rx = [&](const std::string &row) { return qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,antenna,element,panel,spacing,pol,ports,steer,beams,separate_beams,apply_weights,device_size,locations,res\nA,0,0,0," + row + "\n"); }); };
    CHECK(rx("3gpp,\"65,65,30,30,8\",\"4,4,1,2\",\"0.5,0.5\",\"3,2\",\"2,2\",\"6,0;10,0\",,,1,,,2") == "");
    CHECK(rx("3GPP-UT,\"125\",,,\"2\",,,,,,\"0.15,0.07,0\",\"1,7,3\",") == "");
    CHECK(rx("multibeam,,\"2,2\",,,,,\"10,0,1;-10,5,1\",1,0,,,") == "");
    CHECK(rx("custom,\"90,90,10\",,,,,,,,,,,") == "");
    CHECK(has(rx("foo,,,,,,,,,,,,"), "line 2, column 'antenna' has an unknown antenna type ('foo')"));
    CHECK(has(rx(",,\"4,4\",,,,,,,,,,"), "antenna type ('antenna') is empty"));
    CHECK(has(rx("dipole,,\"4,4\",,,,,,,,,,"), "line 2, column 'panel' is not used by the antenna type 'dipole'"));
    CHECK(has(rx("ula,\"65\",,,,,,,,,,,"), "column 'element' is not used by the antenna type 'ula'"));
    CHECK(has(rx("3gpp,,,,,,,\"1,0\",,,,,"), "column 'beams' is not used by the antenna type '3gpp'"));
    CHECK(rx("omni,,,,,,,,,,,,10") == "");
    CHECK(has(rx("3gpp,,\"4,x\",,,,,,,,,,"), "line 2, column 'panel'"));
    CHECK(has(rx("3gpp,,\"4,1.5\",,,,,,,,,,"), "column 'panel' must have non-negative integer entries"));
    CHECK(has(rx("3gpp,,,,,,\"1,2;3\",,,,,,"), "line 2, column 'steer' has rows of different length"));
    CHECK(has(rx("multibeam,,,,,,,,2,,,,"), "column 'separate_beams'"));
    CHECK(has(rx("3gpp,,,,,,,,,,,,abc"), "column 'res' is not numeric"));
    CHECK(has(rx("qdant,,,,,,,,,,,,"), "needs a 'qdant_file'"));
    CHECK(has(rx("3gpp,\"1,,2\",,,,,,,,,,,"), "column 'element' has an empty entry"));
}

TEST_CASE("layout_csv_read - QDANT files are named relative to the CSV file")
{
    qd_layout_files dir("read_qdant");
    std::filesystem::create_directories(dir.path("a"));
    std::filesystem::create_directories(dir.path("b"));
    const std::string tx = dir.write("a/tx.csv", "name,x,y,z,antenna,qdant_file,qdant_id\nBS,0,0,25,qdant,../ants.qdant,2\nBS2,1,0,25,qdant,ants.qdant,\n");
    const std::string rx = dir.write("b/rx.csv", "name,x,y,z,antenna,qdant_file\nUT,0,0,1.5,qdant,ants.qdant\nUT2,0,0,1.5,qdant,\"../a/../ants.qdant\"\n");
    const quadriga_lib::layout L = quadriga_lib::layout_csv_read(tx, rx, {}, false);
    const std::string cwd = std::filesystem::current_path().generic_string() + "/";
    const std::string root = cwd + dir.dir + "/";
    CHECK(L.antenna_cols == std::vector<std::string>{"antenna", "qdant_file", "qdant_id"});
    REQUIRE(L.antenna_args.size() == 4);
    CHECK(L.antenna_args[0] == std::vector<std::string>{"qdant," + root + "ants.qdant,2"});
    CHECK(L.antenna_args[1] == std::vector<std::string>{"qdant," + root + "a/ants.qdant,"});
    CHECK(L.antenna_args[2] == std::vector<std::string>{"qdant," + root + "b/ants.qdant,"});
    CHECK(L.antenna_args[3] == std::vector<std::string>{"qdant," + root + "ants.qdant,"});
    CHECK(arma::all(L.tx.i_antenna == arma::u32_vec{1, 2}));
    CHECK(arma::all(L.rx.i_antenna == arma::u32_vec{3, 4}));
}

TEST_CASE("layout_csv_read - band rows")
{
    qd_layout_files dir("read_band");
    const std::string tx = "name,x,y,z,center_freq,antenna,panel\n"
                           "BS1,0,0,25,3.5e9,3gpp,\"8,8,1,2\"\n"
                           "BS1,,,,28e9,3gpp,\"16,16,1,4\"\n"
                           "BS2,500,0,25,3.5e9,3gpp,\"8,8,1,2\"\n"
                           "BS2,,,,28e9,,\n";
    const std::string rx = "name,x,y,z,antenna\nUT1,10,20,1.5,dipole\n";
    const quadriga_lib::layout L = read_text(dir, tx, rx);
    CHECK(arma::approx_equal(L.center_freq, arma::vec{3.5e9, 28e9}, "absdiff", 0.0));
    CHECK(L.antenna_cols == std::vector<std::string>{"center_freq", "antenna", "panel"});
    // One array per band for BS1; BS2 repeats its first band, so it is stored once without a frequency; UT1 inherits
    const std::vector<std::vector<std::string>> expect = {{"3.5e9,3gpp,\"8,8,1,2\"", "28e9,3gpp,\"16,16,1,4\""}, {",3gpp,\"8,8,1,2\""}, {",dipole,"}};
    CHECK(L.antenna_args == expect);
    CHECK(arma::all(L.tx.i_antenna == arma::u32_vec{1, 2}));
    CHECK(arma::all(L.rx.i_antenna == arma::u32_vec{3}));
    CHECK(L.tx.n_row() == 2);
    CHECK(L.is_valid() == "");

    // Band rows may be anywhere in the file, and the lines are sorted by band
    const std::string tx2 = "name,x,y,z,center_freq,antenna,panel\nBS1,,,,28e9,3gpp,\"16,16,1,4\"\nBS1,0,0,25,3.5e9,3gpp,\"8,8,1,2\"\n";
    CHECK(has(qd_layout_throws([&] { read_text(dir, tx2, ""); }), "line 2, column 'x' is empty")); // The first row of a node is not a band row
    const std::string tx3 = "name,x,y,z,center_freq,antenna,panel\nBS1,0,0,25,28e9,3gpp,\"16,16,1,4\"\nBS2,5,0,25,28e9,,\nBS2,5,1,25,,,\nBS1,,,,3.5e9,3gpp,\"8,8,1,2\"\nBS2,,,,3.5e9,,\n";
    const quadriga_lib::layout L3 = read_text(dir, tx3, "");
    CHECK(arma::approx_equal(L3.center_freq, arma::vec{3.5e9, 28e9}, "absdiff", 0.0));
    CHECK(L3.antenna_args == std::vector<std::vector<std::string>>{{"3.5e9,3gpp,\"8,8,1,2\"", "28e9,3gpp,\"16,16,1,4\""}});
    CHECK(arma::all(L3.tx.i_antenna == arma::u32_vec{1, 0}));
    CHECK(L3.tx.n_row() == 3);

    // A position row repeats the center_freq of the node or leaves it empty
    CHECK(qd_layout_throws([&] { read_text(dir, "name,x,y,z,center_freq\nA,0,0,0,1e9\nA,1,0,0,1e9\nA,2,0,0,\n", ""); }) == "");
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z,center_freq\nA,0,0,0,1e9\nA,,,,\n", ""); }), "line 3, column 'x' is empty"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z,center_freq\nA,0,0,0,1e9\nA,1,0,0,2e9\n", ""); }), "line 3, column 'center_freq' differs from the first row"));
    // The Rx side defines the frequencies if the Tx side has none
    const quadriga_lib::layout L4 = read_text(dir, "name,x,y,z\nBS,0,0,0\n", "name,x,y,z,center_freq\nUT,0,0,0,2e9\nUT,,,,1e9\n");
    CHECK(arma::approx_equal(L4.center_freq, arma::vec{1e9, 2e9}, "absdiff", 0.0));
    // A node without bands has all bands
    const quadriga_lib::layout L5 = read_text(dir, "name,x,y,z,center_freq\nBS,0,0,0,1e9\nBS,,,,2e9\n", "name,x,y,z,antenna\nUT,0,0,0,omni\n");
    CHECK(L5.antenna_args == std::vector<std::vector<std::string>>{{"omni"}});
    CHECK(arma::all(L5.tx.i_antenna == arma::u32_vec{0}));
    CHECK(arma::all(L5.rx.i_antenna == arma::u32_vec{1}));
}

TEST_CASE("layout_csv_read - band rule violations")
{
    qd_layout_files dir("read_band_err");
    const std::string head = "name,x,y,z,center_freq,antenna,heading\n";
    auto tx = [&](const std::string &rows, const std::string &rx = "") { return qd_layout_throws([&] { read_text(dir, head + rows, rx); }); };
    CHECK(has(tx("A,0,0,0,1e9,omni,\nA,,,,2e9,omni,\nB,1,0,0,1e9,omni,\n"), "defines 1 of 2 bands"));
    CHECK(has(tx("A,0,0,0,1e9,omni,\nA,,,,1e9,omni,\n"), "defined twice"));
    CHECK(has(tx("A,0,0,0,1e9,omni,\nA,,,,2e9,omni,30\n"), "line 3, column 'heading' must be empty in a band row"));
    CHECK(has(tx("A,0,0,0,1e9,,\nA,,,,2e9,omni,\n"), "a band row has antenna cells, but the first band has none"));
    CHECK(has(tx("A,0,0,0,,omni,\nA,,,,2e9,omni,\n"), "needs a 'center_freq'"));
    CHECK(has(tx("A,0,0,0,1e9,omni,\nA,,,,2e9,omni,\n", "name,x,y,z,center_freq\nUT,0,0,0,1e9\n"), "defines 1 of 2 bands"));
    CHECK(has(tx("A,0,0,0,1e9,omni,\n", "name,x,y,z,center_freq\nUT,0,0,0,3e9\nUT,,,,1e9\n"), "is not in the frequency list"));
    CHECK(has(tx("A,0,0,0,-1e9,omni,\n"), "column 'center_freq'"));
    // Unknown columns are not allowed in a band row either
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z,center_freq,foo\nA,0,0,0,1e9,\nA,,,,2e9,bar\n", ""); }), "line 3, column 'foo' must be empty in a band row"));
}

TEST_CASE("layout_csv_read - given frequencies")
{
    qd_layout_files dir("read_freq");
    const std::string tx = "name,x,y,z,center_freq,antenna\nA,0,0,0,3.5e9,omni\nA,,,,28e9,omni\n";
    const quadriga_lib::layout L = read_text(dir, tx, "", arma::vec{3.5e9, 28e9});
    CHECK(arma::approx_equal(L.center_freq, arma::vec{3.5e9, 28e9}, "absdiff", 0.0));
    // The file values must match the list within 1e-6; the list wins
    const arma::vec near = {3.5e9 * (1.0 + 5e-7), 28e9};
    CHECK(arma::approx_equal(read_text(dir, tx, "", near).center_freq, near, "absdiff", 0.0));
    CHECK(has(qd_layout_throws([&] { read_text(dir, tx, "", arma::vec{3.5e9 * (1.0 + 5e-6), 28e9}); }), "not in the frequency list"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, tx, "", arma::vec{3.5e9, 30e9}); }), "not in the frequency list"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, tx, "", arma::vec{3.5e9, 28e9, 40e9}); }), "defines 2 of 3 bands"));
    // Nodes without bands have all given frequencies
    const quadriga_lib::layout L2 = read_text(dir, "name,x,y,z\nA,0,0,0\n", "name,x,y,z,antenna\nU,0,0,0,omni\n", arma::vec{1e9, 2e9, 3e9});
    CHECK(L2.center_freq.n_elem == 3);
    CHECK(L2.antenna_args == std::vector<std::vector<std::string>>{{"omni"}});
    // Without any frequency in the files or the argument the list is empty
    CHECK(read_text(dir, "name,x,y,z\nA,0,0,0\n", "").center_freq.is_empty());
}

TEST_CASE("layout_csv_read - unknown columns are kept and reported")
{
    qd_layout_files dir("read_extra");
    std::vector<std::string> ex_tx, ex_rx;
    const quadriga_lib::layout L = read_text(dir, "name,x,y,z,Heading,note\nA,0,0,0,5,\"a,b\"\nA,1,0,0,,x\n", "name,x,y,z,owner\nU,0,0,0,me\n", {}, &ex_tx, &ex_rx);
    CHECK(ex_tx == std::vector<std::string>{"Heading", "note"});
    CHECK(ex_rx == std::vector<std::string>{"owner"});
    CHECK(L.tx.extra_names == ex_tx);
    CHECK(L.tx.extra == std::vector<std::vector<std::string>>{{"5", ""}, {"a,b", "x"}});
    CHECK(L.rx.extra == std::vector<std::vector<std::string>>{{"me"}});
    CHECK(arma::all(L.tx.orientation.row(2) == 0.0)); // 'Heading' is not 'heading'
    std::vector<std::string> none;
    read_text(dir, "name,x,y,z\nA,0,0,0\n", "", {}, &none, nullptr);
    CHECK(none.empty());
}

TEST_CASE("layout_csv_read - seeds, pairing, wrap and scenario parameters")
{
    qd_layout_files dir("read_misc");
    const std::string tx = "name,x,y,z,seed,layout_seed,wrap,d_clutter,r_clutter,h_clutter,scenpar.DS_VS\n"
                           "A,0,0,10,7,42,\"100,0;0,200\",5,0.5,3,0.2\n"
                           "B,300,0,10,,,,,,,\n"
                           "A,1,0,10,,42,\"100,0;0,200\",,,,\n";
    const std::string rx = "name,x,y,z,pairing,relative,direction_az\nU1,1,0,1.5,A,1,30\nU2,2,0,1.5,\"A, B\",,\nU3,3,0,1.5,,,\n";
    const quadriga_lib::layout L = read_text(dir, tx, rx);
    CHECK(L.seed == 42);
    CHECK(arma::all(L.tx.seed == arma::s64_vec{7, -1}));
    CHECK(arma::approx_equal(L.wrap, arma::mat{{100, 0}, {0, 200}}, "absdiff", 0.0));
    CHECK(qd_layout_same_mat(L.tx.clutter, arma::mat{{5, NaN}, {0.5, NaN}, {3, NaN}}, 0.0));
    REQUIRE(L.tx.scenpar_override.count("DS_VS") == 1);
    CHECK(qd_layout_same_mat(L.tx.scenpar_override.at("DS_VS"), arma::vec{0.2, NaN}, 0.0));
    CHECK(arma::all(L.rx.relative == arma::u32_vec{1, 0, 0}));
    // Pairs: (site, terminal); a terminal without entry is linked to all sites
    const arma::umat pairs = {{0, 0, 1, 0, 1}, {0, 1, 1, 2, 2}};
    CHECK(arma::all(arma::vectorise(L.pairing == pairs)));
    CHECK(L.is_valid() == "");

    // One lattice vector
    CHECK(arma::approx_equal(read_text(dir, "name,x,y,z,wrap\nA,0,0,0,\"1000,-500\"\n", "").wrap, arma::mat(arma::vec{1000, -500}), "absdiff", 0.0));

    // Errors
    auto err = [&](const std::string &t, const std::string &r) { return qd_layout_throws([&] { read_text(dir, t, r); }); };
    CHECK(has(err("name,x,y,z,pairing\nA,0,0,0,\n", ""), "the column 'pairing' is only used in the Rx file"));
    CHECK(has(err("", "name,x,y,z,wrap\nA,0,0,0,\n"), "the column 'wrap' is only used in the Tx file"));
    CHECK(has(err("", "name,x,y,z,seed\nA,0,0,0,\n"), "'seed'"));
    CHECK(has(err("name,x,y,z\nA,0,0,0\n", "name,x,y,z,pairing\nU,0,0,0,Z\n"), "names the site 'Z'"));
    CHECK(has(err("name,x,y,z\nA,0,0,0\n", "name,x,y,z,pairing\nU,0,0,0,\"A,A\"\n"), "twice"));
    CHECK(has(err("", "name,x,y,z,pairing\nU,0,0,0,A\n"), "names the site 'A'"));
    CHECK(has(err("name,x,y,z,wrap\nA,0,0,0,\"1,2,3\"\n", ""), "line 2, column 'wrap'"));
    CHECK(has(err("name,x,y,z,wrap\nA,0,0,0,\"1,0;2,0\"\n", ""), "linearly independent"));
    CHECK(has(err("name,x,y,z,wrap\nA,0,0,0,\"1,0\"\nA,1,0,0,\"2,0\"\n", ""), "differs from the 'wrap' of an earlier row"));
    CHECK(has(err("name,x,y,z,layout_seed\nA,0,0,0,1\nA,1,0,0,2\n", ""), "differs from the 'layout_seed'"));
    CHECK(has(err("name,x,y,z,seed\nA,0,0,0,-2\n", ""), "line 2, column 'seed'"));
    std::vector<std::string> ex;
    read_text(dir, "name,x,y,z,scenpar.\nA,0,0,0,1\n", "", {}, &ex); // 'scenpar.' without a key is an unknown column
    CHECK(ex == std::vector<std::string>{"scenpar."});
    CHECK(has(err("name,x,y,z,scenpar.DS_VS\nA,0,0,0,x\n", ""), "line 2, column 'scenpar.DS_VS' is not numeric"));
}

TEST_CASE("layout_csv_read - the cell column")
{
    // A Fixed Tx column: empty = 1; equal for the sectors of a site; written only if a site has 0
    qd_layout_files dir("read_cell");
    const std::string tx = "name,sector,x,y,z,cell\nA,0,0,0,25,\nA,1,,,,\nB,0,100,0,10,0\nC,0,200,0,10,1\n";
    const quadriga_lib::layout L = read_text(dir, tx, "");
    CHECK(arma::all(L.tx.cell == arma::u32_vec{1, 1, 0, 1}));
    CHECK(L.is_valid() == "");
    CHECK(read_text(dir, "name,x,y,z,cell\nA,0,0,0,1\n", "").tx.cell.is_empty()); // All 1 = empty

    quadriga_lib::layout_csv_write(dir.path("w.csv"), "", L);
    const std::string w = dir.read("w.csv");
    CHECK(has(w, "name,sector,x,y,z,cell\n"));
    CHECK(has(w, "\nB,0,100,0,10,0\nC,0,200,0,10,\n"));
    CHECK(arma::all(quadriga_lib::layout_csv_read(dir.path("w.csv"), "", {}, false).tx.cell == L.tx.cell));
    quadriga_lib::layout_csv_write(dir.path("w.csv"), "", L, false); // Every cell written
    CHECK(has(dir.read("w.csv"), ",cell"));
    CHECK(arma::all(quadriga_lib::layout_csv_read(dir.path("w.csv"), "", {}, false).tx.cell == L.tx.cell));
    quadriga_lib::layout M = L;
    M.tx.cell.reset();
    quadriga_lib::layout_csv_write(dir.path("w.csv"), "", M);
    CHECK(!has(dir.read("w.csv"), "cell"));

    auto err = [&](const std::string &t, const std::string &r) { return qd_layout_throws([&] { read_text(dir, t, r); }); };
    CHECK(has(err("", "name,x,y,z,cell\nU,0,0,0,1\n"), "the column 'cell' is only used in the Tx file"));
    CHECK(has(err("name,x,y,z,cell\nA,0,0,0,2\n", ""), "line 2, column 'cell'"));
    CHECK(has(err("name,x,y,z,cell\nA,0,0,0,0\nA,1,0,0,1\n", ""), "differs from the first row of the node"));
    CHECK(has(err("name,sector,x,y,z,cell\nA,0,0,0,0,0\nA,1,0,0,0,1\n", ""), "must have equal 'cell'"));
}

// ---- Writer ----

TEST_CASE("layout_csv_write - the file text")
{
    qd_layout_files dir("write_text");
    const quadriga_lib::layout L = qd_layout_build();
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L);
    // The extra sector BS1/1 has all cells in its first row, because an empty cell would be the value of sector 0
    CHECK(dir.read("tx.csv") == "name,sector,x,y,z,tilt,heading\nBS1,0,10,10,25,,30\nBS1,1,10,10,25,10,150\nBS2,0,-300,50,30,,270\n");
    CHECK(dir.read("rx.csv") == "name,x,y,z,heading,direction_az,speed\nUT1,100,200,1.5,,,\nUT1,101,200,1.5,,,\nUT1,101,202,1.5,,,\nUT2,-50,70,1.5,45,120,1.5\n");

    // Every cell written
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L, false);
    CHECK(dir.read("tx.csv") == "name,sector,x,y,z,tilt,heading\nBS1,0,10,10,25,0,30\nBS1,1,10,10,25,10,150\nBS2,0,-300,50,30,0,270\n");
    CHECK(dir.read("rx.csv") == "name,x,y,z,heading,direction_az,speed\nUT1,100,200,1.5,0,NaN,0\nUT1,101,200,1.5,0,NaN,0\nUT1,101,202,1.5,0,NaN,0\nUT2,-50,70,1.5,45,120,1.5\n");

    // One file only; the other is not touched
    std::filesystem::remove(dir.path("rx.csv"));
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), "", L);
    CHECK(std::filesystem::exists(dir.path("tx.csv")));
    CHECK(!std::filesystem::exists(dir.path("rx.csv")));
    quadriga_lib::layout_csv_write("", "", L);

    // Columns without content are not written; a layout without nodes has only the required columns
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), quadriga_lib::layout());
    CHECK(dir.read("tx.csv") == "name,x,y,z\n");
    CHECK(dir.read("rx.csv") == "name,x,y,z\n");
}

TEST_CASE("layout_csv_write - numbers have at most 9 significant digits")
{
    // Values with digits beyond the 9th are written rounded to 9 digits in the shortest form; integers, seeds and ids
    // are exact. Read back, values agree within 9 significant digits (relative 1e-8), angles in degrees likewise;
    // positions keep at least 9 digits and 1 um.
    qd_layout_files dir("csv_write_digits");
    quadriga_lib::layout L = qd_layout_build();
    L.center_freq = {3.7500000001e9};
    L.tx.pos(0, 0) = L.tx.pos(0, 1) = 1.0 / 3.0; // Both sectors of BS1
    L.rx.pos(1, 0) = 12345.678901234;
    L.tx.orientation(2, 0) = 0.1234567890123; // 7.07355295625... degrees
    L.tx.seed = {123456789012345, 123456789012345, 7};
    L.tx.scenpar_override["DS_mu"] = {0.1 + 0.2, 0.1 + 0.2, -7.25};
    L.rx.floor = {2147483647, 2147483647, 2147483647, -1};
    L.rx.indoor = {1, 1, 1, 0};
    REQUIRE(L.is_valid() == "");
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L);
    const std::string tx = dir.read("tx.csv"), rx = dir.read("rx.csv");
    CHECK(has(tx, ",3.75e9\n")); // Engineering notation from 1e6 on
    CHECK(has(tx, "\nBS1,0,0.333333333,"));
    CHECK(has(tx, ",7.07355296,"));
    CHECK(has(tx, ",123456789012345"));
    CHECK(has(tx, ",0.3,123456789012345,"));
    CHECK(has(tx, ",-7.25,7,"));
    CHECK(has(rx, ",12345.678901,")); // A position: 1 um
    CHECK(has(rx, ",2147483647"));

    const quadriga_lib::layout R = quadriga_lib::layout_csv_read(dir.path("tx.csv"), dir.path("rx.csv"), {}, false);
    auto rel = [](double a, double b) { return std::abs(a - b) <= 1.0e-8 * std::max(std::abs(a), std::abs(b)); };
    CHECK(rel(R.center_freq[0], L.center_freq[0]));
    CHECK(rel(R.tx.pos(0, 0), L.tx.pos(0, 0)));
    CHECK(std::abs(R.rx.pos(1, 0) - L.rx.pos(1, 0)) <= 0.5e-6);
    CHECK(rel(R.tx.orientation(2, 0), L.tx.orientation(2, 0)));
    CHECK(rel(R.tx.scenpar_override.at("DS_mu")[0], 0.3));
    CHECK(R.tx.seed[0] == 123456789012345);
    CHECK(R.rx.floor[0] == 2147483647);
}

TEST_CASE("layout_csv_write - compact files of computed tracks are stable")
{
    // Computed headings differ beyond the written digits from row to row (9 significant digits; positions 1 um); the
    // compact form leaves a held cell empty when its text repeats, so that write, read and write give the same text
    qd_layout_files dir("csv_write_stable");
    auto L = quadriga_lib::generate_layout("38.901_UMa", 1, {3.5e9}, {}, {0.0}, {}, {}, {}, {}, {}, true, 11);
    REQUIRE(L.rx.n_node() == 3);
    quadriga_lib::generate_track(L.rx, {0, 1, 2}, "street", 60.0, 5.0, {}, 0.5, {}, 3);
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L);
    const std::string tx = dir.read("tx.csv"), rx = dir.read("rx.csv");
    const auto L2 = quadriga_lib::layout_csv_read(dir.path("tx.csv"), dir.path("rx.csv"), {}, false);
    CHECK(qd_layout_diff(L, L2, 1.0e-8, 0.5e-6, 0.5e-9) == "");
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L2);
    CHECK(dir.read("tx.csv") == tx);
    CHECK(dir.read("rx.csv") == rx);
    // Along a straight street the heading repeats: most rows leave the heading cell empty
    const auto R = quadriga_lib::layout_csv_read("", dir.path("rx.csv"), {}, false);
    size_t n_rows = 0, n_empty = 0;
    std::istringstream is(rx);
    std::string line, head;
    std::getline(is, head);
    size_t h = 0, k = 0;
    for (size_t p = 0; p < head.size(); ++p)
        if (head.compare(p, 7, "heading") == 0 && (p == 0 || head[p - 1] == ','))
            h = k;
        else if (head[p] == ',')
            ++k;
    while (std::getline(is, line))
    {
        size_t c = 0, p = 0;
        for (; c < h; ++c)
            p = line.find(',', p) + 1;
        ++n_rows;
        n_empty += line[p] == ',' ? 1 : 0;
    }
    CHECK(n_rows == R.rx.n_row());
    CHECK(n_empty > n_rows / 2);
}

TEST_CASE("layout_csv_write - number format and the resolution of positions and times")
{
    // Values: plain digits from 1e-3 to below 1e6, else engineering notation (exponent a multiple of 3), zero as 0.
    // Positions and wrap vectors in m: the decimals for 1 um and at least 9 significant digits, plain also for map
    // coordinates; times: 1 ns. Read back, positions agree within 0.5 um and times within 0.5 ns.
    qd_layout_files dir("csv_write_format");
    quadriga_lib::layout L = qd_layout_build();
    L.center_freq = {1.5e6, 28e9};
    L.rx.pos(1, 0) = 5.4e6 + 0.1234564, L.rx.pos(1, 1) = 1.0e7 + 0.9876543, L.rx.pos(1, 2) = 1.0e7 + 1.5;
    L.rx.pos(0, 3) = 1.23456789012e-4;
    L.rx.time = {0.0, 1.0000000012345, 2.25, NaN};
    L.rx.speed = {0.0, 0.0, 0.0, 2.5e-6};
    L.rx.d2d_in = {0.00025, 0.00025, 0.00025, 0.5};
    L.rx.indoor = {1, 1, 1, 1};
    L.wrap = arma::mat(arma::vec{1299.03810567666, 2250.0});
    REQUIRE(L.is_valid() == "");
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L);
    const std::string tx = dir.read("tx.csv"), rx = dir.read("rx.csv");
    INFO(tx << rx);
    CHECK(has(tx, ",1.5e6\n"));
    CHECK(has(tx, ",28e9\n"));
    CHECK(has(tx, ",\"1299.038106,2250\","));
    CHECK(has(rx, "\nUT1,100,5400000.123456,1.5,0,"));
    CHECK(has(rx, "\nUT1,101,10000000.987654,1.5,1.000000001,"));
    CHECK(has(rx, "\nUT1,101,10000001.5,1.5,2.25,"));
    CHECK(has(rx, "\nUT2,0.000123456789,70,1.5,NaN,"));
    CHECK(has(rx, ",2.5e-6,"));
    CHECK(has(rx, ",250e-6\n"));
    CHECK(has(rx, ",0.5\n"));

    const quadriga_lib::layout R = quadriga_lib::layout_csv_read(dir.path("tx.csv"), dir.path("rx.csv"), {}, false);
    CHECK(arma::abs(R.rx.pos - L.rx.pos).max() <= 0.5e-6);
    CHECK(arma::abs(R.wrap - L.wrap).max() <= 0.5e-6);
    CHECK(std::abs(R.rx.time[1] - L.rx.time[1]) <= 0.5e-9);
    CHECK(std::abs(R.rx.pos(0, 3) - L.rx.pos(0, 3)) <= 1.0e-8 * L.rx.pos(0, 3)); // 9 significant digits
    CHECK(R.rx.speed[3] == 2.5e-6);
    CHECK(R.center_freq[1] == 28e9);
}

TEST_CASE("layout_csv_write - antennas, bands and shared values")
{
    qd_layout_files dir("write_ant");
    quadriga_lib::layout L = qd_layout_build();
    L.antenna_cols = {"antenna", "panel"};
    L.antenna_args = {{"3gpp,\"4,4,1,2\""}, {"dipole,"}};
    L.tx.i_antenna = {1, 1, 0};
    L.rx.i_antenna = {2, 0};
    L.center_freq = {3.5e9, 28e9};
    L.seed = 5;
    L.wrap = arma::mat{{100.0, 0.0}, {0.0, 200.0}};
    L.tx.seed = {7, 7, -1};
    L.tx.clutter = arma::mat{{5, 5, NaN}, {NaN, NaN, NaN}, {3, 3, NaN}};
    L.tx.scenpar_override["DS_VS"] = arma::vec{0.2, 0.2, NaN};
    L.rx.relative = {0, 1};
    L.tx.extra_names = {"note"};
    L.tx.extra = {{"a,b", "", "x"}};
    L.rx.scenario = {"UMa", "UMa", "UMa_NLOS", "UMa"};
    REQUIRE(L.is_valid() == "");
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L);

    // The frequencies are declared on the first Tx node; the extra sector has all its cells (the antenna of sector 0 too)
    CHECK(dir.read("tx.csv") == "name,sector,x,y,z,tilt,heading,d_clutter,h_clutter,scenpar.DS_VS,wrap,seed,layout_seed,center_freq,antenna,panel,note\n"
                                "BS1,0,10,10,25,,30,5,3,0.2,\"100,0;0,200\",7,5,3.5e9,3gpp,\"4,4,1,2\",\"a,b\"\n"
                                "BS1,0,,,,,,,,,,,,28e9,,,\n"
                                "BS1,1,10,10,25,10,150,5,3,0.2,,7,,,3gpp,\"4,4,1,2\",\n"
                                "BS2,0,-300,50,30,,270,,,,,,,,,,x\n");
    CHECK(dir.read("rx.csv") == "name,x,y,z,heading,direction_az,speed,scenario,relative,antenna\n"
                                "UT1,100,200,1.5,,,,UMa,,dipole\n"
                                "UT1,101,200,1.5,,,,,,\n"
                                "UT1,101,202,1.5,,,,UMa_NLOS,,\n"
                                "UT2,-50,70,1.5,45,120,1.5,UMa,1,\n");

    // The text reads back to the same layout; the frequencies of the declaration are the list, and its antenna is one line
    const quadriga_lib::layout L2 = quadriga_lib::layout_csv_read(dir.path("tx.csv"), dir.path("rx.csv"), {}, false);
    CHECK(qd_layout_diff(L, L2) == "");
    CHECK(arma::approx_equal(L2.center_freq, arma::vec{3.5e9, 28e9}, "absdiff", 0.0));

    // With every cell written the result is the same
    quadriga_lib::layout_csv_write(dir.path("tx2.csv"), dir.path("rx2.csv"), L, false);
    CHECK(qd_layout_diff(L, quadriga_lib::layout_csv_read(dir.path("tx2.csv"), dir.path("rx2.csv"), {}, false)) == "");

    // A frequency that an antenna line carries is not declared again
    quadriga_lib::layout L3 = L;
    L3.antenna_cols = {"center_freq", "antenna", "panel"};
    L3.antenna_args = {{"3.5e9,3gpp,\"4,4,1,2\"", "28e9,3gpp,\"8,8,1,4\""}, {",dipole,"}};
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L3);
    CHECK(has(dir.read("tx.csv"), "\nBS1,0,10,10,25,,30,5,3,0.2,\"100,0;0,200\",7,5,3.5e9,3gpp,\"4,4,1,2\",\"a,b\"\nBS1,0,,,,,,,,,,,,28e9,3gpp,\"8,8,1,4\",\n"));
    CHECK(has(dir.read("rx.csv"), ",UMa,,,dipole\n") == false);
    CHECK(qd_layout_diff(L3, quadriga_lib::layout_csv_read(dir.path("tx.csv"), dir.path("rx.csv"), {}, false)) == "");

    // QDANT files are written relative to the folder of the CSV file
    quadriga_lib::layout L4 = qd_layout_build();
    const std::string abs_file = std::filesystem::absolute(dir.path("q/ants.qdant")).lexically_normal().generic_string();
    L4.antenna_cols = {"antenna", "qdant_file", "qdant_id"};
    L4.antenna_args = {{"qdant," + abs_file + ",2"}};
    L4.tx.i_antenna = {1, 1, 1};
    std::filesystem::create_directories(dir.path("t"));
    quadriga_lib::layout_csv_write(dir.path("t/tx.csv"), "", L4);
    CHECK(has(dir.read("t/tx.csv"), "qdant,../q/ants.qdant,2"));
    CHECK(qd_layout_diff_nodes(quadriga_lib::layout_csv_read(dir.path("t/tx.csv"), "", {}, false).tx, L4.tx, 0.0) == "");
    CHECK(quadriga_lib::layout_csv_read(dir.path("t/tx.csv"), "", {}, false).antenna_args == L4.antenna_args);
}

TEST_CASE("layout_csv_write - an extra sector with the default value")
{
    qd_layout_files dir("write_extra_sector");
    quadriga_lib::layout L = qd_layout_build();
    L.tx.orientation(1, 0) = 10.0 * D; // Sector 0 has a tilt, the extra sector the default
    L.tx.orientation(1, 1) = 0.0;
    L.tx.orientation(0, 1) = 0.0;
    L.tx.orientation(0, 0) = 5.0 * D;
    L.rx = quadriga_lib::layout_nodes();
    for (bool compact : {true, false})
    {
        quadriga_lib::layout_csv_write(dir.path("tx.csv"), "", L, compact);
        CHECK(qd_layout_diff(L, quadriga_lib::layout_csv_read(dir.path("tx.csv"), "", {}, false)) == "");
    }
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), "", L);
    CHECK(has(dir.read("tx.csv"), "\nBS1,0,10,10,25,5,10,30\nBS1,1,10,10,25,0,0,150\n"));
}

TEST_CASE("layout_csv_write - pairing")
{
    qd_layout_files dir("write_pair");
    quadriga_lib::layout L = qd_layout_build();
    L.pairing = arma::umat{{1, 0, 1}, {0, 1, 1}}; // (BS2, UT1), (BS1, UT2), (BS2, UT2)
    REQUIRE(L.is_valid() == "");
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L);
    CHECK(has(dir.read("rx.csv"), "name,x,y,z,heading,direction_az,speed,pairing\nUT1,100,200,1.5,,,,BS2\n"));
    CHECK(has(dir.read("rx.csv"), "UT2,-50,70,1.5,45,120,1.5,\"BS1,BS2\"\n"));
    CHECK(qd_layout_diff(L, quadriga_lib::layout_csv_read(dir.path("tx.csv"), dir.path("rx.csv"), {}, false)) == "");
    // A terminal without a pair cannot be written, because an empty cell means all sites
    L.pairing = arma::umat(arma::uvec{1, 0});
    CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L); }), "Terminal 'UT2' has no entry in 'pairing'"));
}

TEST_CASE("layout_csv_read - read, write and read again")
{
    qd_layout_files dir("roundtrip");
    const std::string tx = "name,sector,x,y,z,heading,tilt,center_freq,antenna,element,panel,spacing,pol,steer,seed,layout_seed,wrap,d_clutter,scenpar.DS_VS,scenario,note\n"
                           "BS1,0,0,0,25,33.3,10,3.5e9,3gpp,\"65,65,30,30,8\",\"4,4,1,2\",\"0.5,0.5,2,2.5\",\"3,2\",\"6,0\",11,9,\"100,0;0,200\",5,0.25,UMa,hello\n"
                           "BS1,0,,,,,,28e9,3gpp,,\"8,8,1,4\"\n"
                           "BS1,1,,,,153.3\n"
                           "BS2,0,500,0,25,359.123456789012345,0,3.5e9,dipole,,,,,,,,,,,UMi\n"
                           "BS2,,,,,,,28e9\n"
                           "BS3,0,1000,0,25,0.1234567890123,,,omni\n";
    const std::string rx = "name,x,y,z,heading,speed,direction_az,direction_el,time,indoor,floor,d2d_in,scenario,antenna,locations,pairing,relative\n"
                           "UT1,10,20,1.5,45.5,0.833,120,-5,,,,,UMa,3gpp-ut,\"1,7,3\",\"BS1,BS2\",\n"
                           "UT2,0,0,1.5,0,,,,0,1,2,12.5,UMa,,,BS3,1\n"
                           "UT2,5,0,1.5,,,,,1,0,1,NaN,UMa_NLOS\n"
                           "UT2,5,5,1.5,90,,,,2.5\n"
                           "UT3,7,7,0,,2,10,,,,,,,omni,,,\n";
    for (bool compact : {true, false})
    {
        const quadriga_lib::layout L = read_text(dir, tx, rx);
        REQUIRE(L.is_valid() == "");
        quadriga_lib::layout_csv_write(dir.path("tx_w.csv"), dir.path("rx_w.csv"), L, compact);
        const quadriga_lib::layout L2 = quadriga_lib::layout_csv_read(dir.path("tx_w.csv"), dir.path("rx_w.csv"), {}, false);
        CHECK(qd_layout_diff(L, L2, 1.0e-8, 0.5e-6, 0.5e-9) == ""); // 9 significant digits, 1 um, 1 ns
        // The second cycle gives the same text
        const std::string text_tx = dir.read("tx_w.csv"), text_rx = dir.read("rx_w.csv");
        quadriga_lib::layout_csv_write(dir.path("tx_w.csv"), dir.path("rx_w.csv"), L2, compact);
        CHECK(dir.read("tx_w.csv") == text_tx);
        CHECK(dir.read("rx_w.csv") == text_rx);
    }
    const quadriga_lib::layout L = read_text(dir, tx, rx);
    CHECK(L.seed == 9);
    CHECK(arma::approx_equal(L.center_freq, arma::vec{3.5e9, 28e9}, "absdiff", 0.0));
    CHECK(std::abs(L.tx.orientation(2, 0) - 33.3 * D) < 1e-14);
    CHECK(std::abs(L.tx.orientation(2, 3) - 0.1234567890123 * D) < 1e-14);
    CHECK(std::abs(L.tx.orientation(2, 2) - 359.123456789012345 * D) < 1e-14); // the largest angle: its round trip sets the tolerance of qd_layout_diff
    CHECK(L.antenna_cols == std::vector<std::string>{"center_freq", "antenna", "element", "panel", "spacing", "pol", "steer", "locations"}); // BS1 has its band rows
    CHECK(arma::all(L.tx.i_antenna == arma::u32_vec{1, 1, 2, 3}));
    CHECK(arma::all(L.rx.i_antenna == arma::u32_vec{4, 0, 3}));
    CHECK(L.tx.relative.is_empty());
    CHECK(qd_layout_same_mat(L.rx.time, arma::vec{NaN, 0, 1, 2.5, NaN}, 0.0));
}

TEST_CASE("layout_csv_write - files are replaced as a whole")
{
    qd_layout_files dir("write_atomic");
    const quadriga_lib::layout L = qd_layout_build();
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), L);
    const std::string tx = dir.read("tx.csv"), rx = dir.read("rx.csv");
    auto files = [&] { std::vector<std::string> f; for (const auto &e : std::filesystem::directory_iterator(dir.dir)) f.push_back(e.path().filename().string()); std::sort(f.begin(), f.end()); return f; };
    const std::vector<std::string> expect = {"rx.csv", "tx.csv"};
    REQUIRE(files() == expect);

    quadriga_lib::layout M = L;
    M.rx.scenario = {"UMa", "", "", "UMa"}; // UT1: the scenario returns to none, which the file cannot express
    M.rx.scenario[0] = "UMa";
    M.rx.scenario[1] = "";
    quadriga_lib::layout M2 = L;
    M2.tx.scenario = {"UMa", "UMa", "UMi"};
    M2.tx.i_antenna = {1, 1, 0}; // Antenna 1 has neither a CSV description nor an array
    M2.antenna.resize(1);
    M2.antenna[0].resize(1);
    M2.antenna_args.clear();

    // Failure while composing the Rx file: the Tx file is not replaced either
    CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), M); }), "the scenario cannot return to none"));
    CHECK(files() == expect);
    CHECK(dir.read("tx.csv") == tx);
    CHECK(dir.read("rx.csv") == rx);
    CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), M2); }), "neither an antenna line nor an evaluated array"));
    CHECK(dir.read("tx.csv") == tx);

    // Failure while writing the Rx file: the Tx file is not replaced and no temporary file stays
    const std::string rx_missing = dir.path("no_such_folder/rx.csv");
    quadriga_lib::layout N = L;
    N.tx.scenario = {"UMa", "UMa", "UMi"};
    CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("tx.csv"), rx_missing, N); }), "Cannot open file"));
    CHECK(files() == expect);
    CHECK(dir.read("tx.csv") == tx);
    CHECK(dir.read("rx.csv") == rx);

    // Both files are replaced when everything works
    quadriga_lib::layout_csv_write(dir.path("tx.csv"), dir.path("rx.csv"), N);
    CHECK(dir.read("tx.csv") != tx);
    CHECK(dir.read("rx.csv") == rx);
    CHECK(files() == expect);
}

TEST_CASE("layout_csv_read - file syntax of the CSV unit applies")
{
    qd_layout_files dir("read_syntax");
    // Byte order mark, CRLF, comments, a quoted name with a comma and a quote
    const quadriga_lib::layout L = read_text(dir, "\xEF\xBB\xBF# layout\r\nname,x,y,z,note\r\n\"BS1\",1,2,3,\"a;b, \"\"c\"\"\"\r\n\r\n  # c\r\nBS2,4,5,6,\r\n", "");
    CHECK(L.tx.name == std::vector<std::string>{"BS1", "BS2"});
    CHECK(L.tx.extra == std::vector<std::vector<std::string>>{{"a;b, \"c\"", ""}});
    // The writer quotes what needs it
    quadriga_lib::layout_csv_write(dir.path("w.csv"), "", L);
    CHECK(has(dir.read("w.csv"), "\nBS1,1,2,3,\"a;b, \"\"c\"\"\"\nBS2,4,5,6,\n"));
    CHECK(qd_layout_diff(L, quadriga_lib::layout_csv_read(dir.path("w.csv"), "", {}, false)) == "");
    // A name that starts with '#'
    quadriga_lib::layout M = qd_layout_build();
    M.rx.name = {"#UT1", "#UT2"};
    quadriga_lib::layout_csv_write("", dir.path("w.csv"), M);
    CHECK(qd_layout_diff_nodes(quadriga_lib::layout_csv_read("", dir.path("w.csv"), {}, false).rx, M.rx, 0.0) == "");
    // Unknown columns of the layout must not repeat a column of the format
    M.rx.extra_names = {"x"};
    M.rx.extra = {{"1", "2", "3", "4"}};
    CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write("", dir.path("w.csv"), M); }), "'extra_names' must not repeat"));
}

TEST_CASE("layout_csv_read - failure leaves the outputs alone")
{
    qd_layout_files dir("read_fail");
    std::vector<std::string> ex_tx = {"keep"}, ex_rx = {"keep"};
    CHECK(!qd_layout_throws([&] { read_text(dir, "name,x,y,z,a\nA,0,0,0,1\n", "name,x,y\nB,0,0\n", {}, &ex_tx, &ex_rx); }).empty());
    CHECK(ex_tx == std::vector<std::string>{"keep"});
    CHECK(ex_rx == std::vector<std::string>{"keep"});
}

TEST_CASE("layout_csv_read - a long file")
{
    qd_layout_files dir("read_long");
    std::string rx = "name,x,y,z,heading\n";
    const int n_ut = 400, n_row = 30;
    for (int u = 0; u < n_ut; ++u)
        for (int r = 0; r < n_row; ++r)
            rx += "UT" + std::to_string(u) + "," + std::to_string(r) + "," + std::to_string(u) + ",1.5," + (r == 0 ? std::to_string(u % 360) : "") + "\n";
    const quadriga_lib::layout L = read_text(dir, "", rx);
    REQUIRE(L.rx.n_node() == (arma::uword)n_ut);
    CHECK(L.rx.n_row() == (arma::uword)(n_ut * n_row));
    CHECK(L.rx.n_row(399) == (arma::uword)n_row);
    CHECK(L.rx.pos(0, 399 * 30 + 29) == 29.0);
    CHECK(L.rx.pos(1, 399 * 30 + 29) == 399.0);
    CHECK(std::abs(L.rx.orientation(2, 399 * 30 + 29) - 39.0 * D) < 1e-14);
    CHECK(L.is_valid() == "");
}

TEST_CASE("layout_csv_read - input validation")
{
    qd_layout_files dir("validation_read");
    const std::string tx = dir.write("tx.csv", basic_tx), rx = dir.write("rx.csv", basic_rx);
    REQUIRE(qd_layout_throws([&] { quadriga_lib::layout_csv_read(tx, rx, {}, false); }).empty());
    {
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_read(dir.path("missing.csv"), "", {}, false); }), "Cannot open file"));
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_read(dir.dir, "", {}, false); }), "it is a folder"));
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_read("", dir.write("empty.csv", ""), {}, false); }), "there is no header line"));
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_read("", dir.write("dup.csv", "name,x,y,z,x\nA,0,0,0,0\n"), {}, false); }), "used twice"));
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_read("", dir.write("long.csv", "name,x,y,z\nA,0,0,0,0\n"), {}, false); }), "the row has 5 cells"));
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_read(tx, rx, arma::vec{2e9, 1e9}); }), "Input 'center_freq' must be strictly ascending"));
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_read(tx, rx, arma::vec{-1.0}); }), "Input 'center_freq' must be finite and positive"));
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_read(tx, rx, arma::vec{NaN}); }), "Input 'center_freq'"));
        // The layout of the two files must be valid: here the terminals' sectors have unequal positions
        CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,sector,x,y,z\nA,0,0,0,0\nA,1,1,0,0\n"); }), "must have equal positions"));
    }

}

TEST_CASE("layout_csv_write - input validation")
{
    qd_layout_files dir("validation_write");
    const quadriga_lib::layout L = read_text(dir, basic_tx, basic_rx);
    REQUIRE(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), dir.path("b.csv"), L); }).empty());
    {
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), dir.path("a.csv"), L); }), "must be different files"));
        quadriga_lib::layout M = L;
        M.tx.name[0] = "";
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), dir.path("b.csv"), M); }), "Input 'layout' is not valid: tx: "));
        M = L;
        M.antenna_cols = {"foo"};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), dir.path("b.csv"), M); }), "antenna_cols"));
        M = L;
        M.tx.i_antenna = {1, 0, 0, 0};
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), dir.path("b.csv"), M); }), "i_antenna"));
        // An extra sector without antenna next to a sector 0 with one is written as 'none' (test_set_antenna.cpp)
        M = L;
        M.antenna_cols = {"antenna"};
        M.antenna_args = {{"omni"}};
        M.tx.i_antenna = {1, 0, 1, 0};
        quadriga_lib::layout_csv_write(dir.path("n_tx.csv"), dir.path("n_rx.csv"), M);
        CHECK(qd_layout_diff(M, quadriga_lib::layout_csv_read(dir.path("n_tx.csv"), dir.path("n_rx.csv"), {}, false)) == "");
        // A name with a line break is not valid
        M = L;
        M.rx.name[0] = "U\nT";
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), dir.path("b.csv"), M); }), "Node names"));
        // The same file under two spellings: nothing is replaced
        const std::string before = dir.read("a.csv");
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), dir.dir + "/./a.csv", L); }), "must be different files"));
        CHECK(dir.read("a.csv") == before);
        CHECK(!std::filesystem::exists(dir.path("a.csv.part")));
        // Values that only the Tx file can hold, when the layout has no Tx nodes
        M = qd_layout_build();
        M.tx = quadriga_lib::layout_nodes();
        M.seed = 7;
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), dir.path("b.csv"), M); }), "can only be written to the Tx file"));
        CHECK(qd_layout_throws([&] { quadriga_lib::layout_csv_write("", dir.path("b.csv"), M); }).empty());
        M.seed = -1;
        M.wrap = arma::mat(arma::vec{10.0, 0.0});
        CHECK(has(qd_layout_throws([&] { quadriga_lib::layout_csv_write(dir.path("a.csv"), "", M); }), "can only be written to the Tx file"));
    }
}

TEST_CASE("is_valid - input validation")
{
    const quadriga_lib::layout base = qd_layout_build();
    REQUIRE(base.is_valid() == "");
    CHECK(quadriga_lib::layout().is_valid() == "");
    CHECK(quadriga_lib::layout_nodes().is_valid() == "");
    CHECK(quadriga_lib::layout_nodes().n_node() == 0);
    CHECK(quadriga_lib::layout_nodes().n_row() == 0);
    CHECK(base.tx.n_node() == 3);
    CHECK(base.tx.n_row() == 3);
    CHECK(base.rx.n_row() == 4);
    CHECK(base.rx.n_row(0) == 3);
    CHECK(base.rx.n_row(1) == 1);
    CHECK(has(qd_layout_throws([&] { base.rx.n_row(2); }), "Input 'i_node'"));
    CHECK(base.n_antenna() == 0);

    auto msg = [&](auto change) { quadriga_lib::layout L = base; change(L); return L.is_valid(); };
    using LL = quadriga_lib::layout;

    // Sizes
    CHECK(has(msg([](LL &L) { L.tx.sector = {0, 1}; }), "one entry per node"));
    CHECK(has(msg([](LL &L) { L.tx.offset = {0, 1}; }), "one entry per node"));
    CHECK(has(msg([](LL &L) { L.rx.pos.set_size(2, 4); }), "'pos' must have 3 rows"));
    CHECK(has(msg([](LL &L) { L.rx.orientation.set_size(3, 3); }), "'orientation'"));
    CHECK(has(msg([](LL &L) { L.rx.direction.set_size(2, 3); }), "'direction'"));
    CHECK(has(msg([](LL &L) { L.rx.speed.set_size(3); }), "one entry per row"));
    CHECK(has(msg([](LL &L) { L.rx.scenario = {"a"}; }), "one entry per row"));
    CHECK(has(msg([](LL &L) { L.tx.clutter.set_size(3, 2); }), "'clutter'"));
    CHECK(has(msg([](LL &L) { L.tx.scenpar_override["DS_VS"] = arma::vec{1.0}; }), "'DS_VS'"));
    CHECK(has(msg([](LL &L) { L.tx.seed = {1, 2}; }), "'seed'"));
    CHECK(has(msg([](LL &L) { L.tx.cell = {1, 1}; }), "'cell'"));
    CHECK(has(msg([](LL &L) { L.tx.cell = {1, 1, 2}; }), "Member 'cell' must be 0 or 1."));
    CHECK(has(msg([](LL &L) { L.tx.cell = {1, 0, 1}; }), "tx: Nodes 'BS1' (sectors 0 and 1): must have equal 'cell'."));
    CHECK(has(msg([](LL &L) { L.rx.cell = {1, 1}; }), "rx: Member 'cell' is a member of the Tx side and must be empty."));
    CHECK(msg([](LL &L) { L.tx.cell = {0, 0, 1}; }) == "");
    CHECK(has(msg([](LL &L) { L.tx.extra_names = {"a"}; }), "'extra_names' and 'extra'"));
    CHECK(has(msg([](LL &L) { L.tx.extra_names = {"a"}; L.tx.extra = {{"1"}}; }), "one cell per row"));
    // Offsets and values
    CHECK(has(msg([](LL &L) { L.rx.offset = {1, 3}; }), "'offset' must start at the first row"));
    CHECK(has(msg([](LL &L) { L.rx.offset = {0, 0}; }), "'offset' must start at the first row"));
    CHECK(has(msg([](LL &L) { L.rx.offset = {0, 4}; }), "'offset' must start at the first row"));
    CHECK(has(msg([](LL &L) { L.rx.pos(1, 1) = NaN; }), "'pos' must be finite"));
    CHECK(has(msg([](LL &L) { L.rx.orientation(0, 2) = arma::datum::inf; }), "'orientation' must be finite"));
    CHECK(has(msg([](LL &L) { L.tx.name[1] = ""; }), "1 to 255 bytes"));
    CHECK(has(msg([](LL &L) { L.tx.name[1] = std::string(256, 'x'); }), "1 to 255 bytes"));
    CHECK(msg([](LL &L) { L.tx.name[0] = L.tx.name[1] = std::string(255, 'x'); L.tx.sector[1] = 1; L.tx.name[2] = "BS2"; }) == "");
    CHECK(has(msg([](LL &L) { L.tx.sector = {0, 0, 0}; }), "is used by two nodes"));
    CHECK(has(msg([](LL &L) { L.rx.speed = arma::vec{0, 0, 0, -1}; }), "'speed' must be finite and >= 0"));
    CHECK(has(msg([](LL &L) { L.rx.speed = arma::vec{0, 0, 0, NaN}; }), "'speed' must be finite and >= 0"));
    CHECK(has(msg([](LL &L) { L.rx.d2d_in = arma::vec{NaN, NaN, NaN, -1}; }), "'d2d_in'"));
    CHECK(has(msg([](LL &L) { L.rx.indoor = {0, 0, 0, 4}; }), "'indoor'"));
    CHECK(has(msg([](LL &L) { L.rx.floor = {-1, -1, -1, 0}; }), "'floor' must be -1 or >= 1"));
    CHECK(has(msg([](LL &L) { L.rx.scenario = {"", "", "a/b", ""}; }), "path separator"));
    CHECK(has(msg([](LL &L) { L.tx.seed = {-2, -1, -1}; }), "'seed' must be >= -1"));
    CHECK(has(msg([](LL &L) { L.tx.relative = {0, 2, 0}; }), "'relative'"));
    // Time and speed
    CHECK(has(msg([](LL &L) { L.rx.time = arma::vec{NaN, NaN, NaN, 5}; }), "either 'time' or 'speed'"));
    CHECK(has(msg([](LL &L) { L.rx.speed.reset(); L.rx.time = arma::vec{0, NaN, 2, NaN}; }), "'time' must be given on all rows or on none"));
    CHECK(has(msg([](LL &L) { L.rx.speed.reset(); L.rx.time = arma::vec{0, 2, 1, NaN}; }), "'time' must increase strictly"));
    CHECK(msg([](LL &L) { L.rx.speed.reset(); L.rx.time = arma::vec{0, 1, 2, 7}; }) == "");
    CHECK(has(msg([](LL &L) { L.rx.speed = arma::vec{1, 0, 0, 1}; }), "'speed' is 0"));
    CHECK(msg([](LL &L) { L.rx.speed = arma::vec{1, 1, 0, 1}; }) == ""); // The last row of a track has no segment
    // Direction of a track
    CHECK(has(msg([](LL &L) { L.rx.direction(0, 1) = 1.0; }), "a track must not have a 'direction'"));
    // Floor and indoor distance change only at the start of a segment
    CHECK(has(msg([](LL &L) { L.rx.floor = {1, 2, 2, -1}; }), "'floor' and 'd2d_in'"));
    CHECK(msg([](LL &L) { L.rx.floor = {1, 2, 2, -1}; L.rx.indoor = {1, 3, 3, 0}; }) == "");
    CHECK(msg([](LL &L) { L.rx.d2d_in = arma::vec{5, 6, 6, NaN}; L.rx.scenario = {"UMa", "UMi", "UMi", ""}; }) == "");

    // Sites and terminals
    CHECK(has(msg([](LL &L) { L.tx.pos(0, 1) += 1.0; }), "must have equal positions"));
    CHECK(has(msg([](LL &L) { L.tx.pos.insert_cols(0, arma::vec{0, 0, 25}); L.tx.orientation.insert_cols(0, arma::vec{0, 0, 0}); L.tx.offset = {0, 2, 3}; }), "must have the same number of rows"));
    CHECK(has(msg([](LL &L) { L.tx.scenario = {"UMa", "UMi", "UMa"}; }), "must have equal 'scenario'"));
    CHECK(has(msg([](LL &L) { L.tx.clutter = arma::mat{{5, 6, 5}, {1, 1, 1}, {2, 2, 2}}; }), "must have equal 'clutter'"));
    CHECK(has(msg([](LL &L) { L.tx.scenpar_override["DS_VS"] = arma::vec{1, 2, 3}; }), "equal 'scenpar_override' for the key 'DS_VS'"));
    CHECK(has(msg([](LL &L) { L.tx.seed = {5, 6, 7}; }), "must have equal 'seed'"));
    CHECK(has(msg([](LL &L) { L.tx.pos.insert_cols(3, arma::vec{0, 0, 30}); L.tx.orientation.insert_cols(3, arma::vec{0, 0, 0}); L.tx.scenario = {"", "", "UMa", "UMi"}; }), "constant along its rows"));
    CHECK(has(msg([](LL &L) { L.rx.name = {"UT1", "UT1"}; L.rx.sector = {0, 1}; L.rx.pos = arma::mat{{0, 0}, {0, 0}, {0, 0}}; L.rx.orientation.zeros(3, 2); L.rx.offset = {0, 1}; L.rx.speed.reset(); L.rx.direction.reset(); L.rx.scenario = {"UMa", "UMi"}; }), "rx: Nodes 'UT1' (sectors 0 and 1): must have equal 'scenario', 'indoor', 'floor' and 'd2d_in'"));
    CHECK(has(msg([](LL &L) { L.rx.name = {"UT1", "UT1"}; L.rx.sector = {0, 1}; L.rx.pos = arma::mat{{0, 0}, {0, 0}, {0, 0}}; L.rx.orientation.zeros(3, 2); L.rx.offset = {0, 1}; L.rx.speed.reset(); L.rx.direction.reset(); L.rx.indoor = {1, 0}; }), "'indoor'"));

    // Antennas and frequencies
    CHECK(has(msg([](LL &L) { L.tx.i_antenna = {1, 0, 0}; }), "'i_antenna' is larger than the number of antennas"));
    CHECK(has(msg([](LL &L) { L.antenna_args = {{"a"}}; L.antenna_cols = {"antenna"}; L.antenna.resize(2); }), "must have the same length"));
    CHECK(has(msg([](LL &L) { L.center_freq = arma::vec{2e9, 1e9}; }), "'center_freq' must be finite, positive and strictly ascending"));
    CHECK(has(msg([](LL &L) { L.center_freq = arma::vec{1e9, 1e9}; }), "'center_freq'"));
    CHECK(has(msg([](LL &L) { L.center_freq = arma::vec{-1e9}; }), "'center_freq'"));
    CHECK(has(msg([](LL &L) { L.center_freq = arma::vec{1e9, 2e9, 3e9}; L.antenna_cols = {"antenna"}; L.antenna_args = {{"omni", "omni"}}; }), "0, 1 or n_freq lines"));
    CHECK(msg([](LL &L) { L.center_freq = arma::vec{1e9, 2e9}; L.antenna_cols = {"center_freq", "antenna"}; L.antenna_args = {{"1e9,omni", "2e9,omni"}, {}, {",dipole"}}; L.tx.i_antenna = {1, 3, 0}; }) == "");
    CHECK(has(msg([](LL &L) { L.antenna.resize(1); L.antenna[0].resize(2); L.center_freq = arma::vec{1e9, 2e9, 3e9}; }), "1 or n_freq arrays"));
    CHECK(has(msg([](LL &L) { L.antenna_cols = {"foo"}; }), "antenna columns"));
    CHECK(has(msg([](LL &L) { L.antenna_cols = {"antenna", "antenna"}; }), "each once"));
    CHECK(has(msg([](LL &L) { L.antenna_cols = {"heading"}; }), "antenna columns"));
    CHECK(has(msg([](LL &L) { L.antenna_cols = {"antenna"}; L.antenna_args = {{"omni,omni"}}; }), "'antenna_args' has a line that is not valid"));
    // The lines are checked as the reader checks the antenna columns
    CHECK(has(msg([](LL &L) { L.antenna_cols = {"antenna"}; L.antenna_args = {{"bogus"}}; }), "unknown antenna type"));
    CHECK(has(msg([](LL &L) { L.antenna_cols = {"antenna", "panel"}; L.antenna_args = {{"dipole,\"4,4\""}}; }), "not used by the antenna type 'dipole'"));
    CHECK(has(msg([](LL &L) { L.antenna_cols = {"antenna"}; L.antenna_args = {{"#x"}}; }), "'antenna_args' has a line that is not valid"));
    CHECK(has(msg([](LL &L) { L.antenna_cols = {"antenna"}; L.antenna_args = {{"omni\nomni"}}; }), "'antenna_args' has a line that is not valid"));
    CHECK(msg([](LL &L) { L.antenna_cols = {"antenna", "panel"}; L.antenna_args = {{"3gpp,\"4,4,1,2\""}}; L.tx.i_antenna = {1, 1, 0}; }) == "");
    // Lines per band: one line has no frequency, n_freq lines carry the frequencies of center_freq
    const auto bands = [](LL &L) { L.center_freq = arma::vec{3.5e9, 28e9}; L.antenna_cols = {"center_freq", "antenna"}; };
    CHECK(msg([&](LL &L) { bands(L); L.antenna_args = {{"3.5e9,omni", "28e9,omni"}}; }) == "");
    CHECK(msg([&](LL &L) { bands(L); L.antenna_args = {{"3500000001,omni", "28000000001,omni"}}; }) == "");
    CHECK(has(msg([&](LL &L) { bands(L); L.antenna_args = {{"3.5e9,omni", "30e9,omni"}}; }), "'center_freq' of the antenna line"));
    CHECK(has(msg([&](LL &L) { bands(L); L.antenna_args = {{",omni", "28e9,omni"}}; }), "'center_freq' of the antenna line"));
    CHECK(has(msg([&](LL &L) { bands(L); L.antenna_args = {{"3.5e9,omni"}}; }), "'center_freq' of a single antenna line"));
    CHECK(msg([&](LL &L) { bands(L); L.antenna_args = {{",omni"}}; }) == "");
    CHECK(has(msg([](LL &L) { L.center_freq = arma::vec{3.5e9, 28e9}; L.antenna_cols = {"antenna"}; L.antenna_args = {{"omni", "omni"}}; }), "'center_freq' of the antenna line"));

    // Lattice vectors and pairs
    CHECK(has(msg([](LL &L) { L.wrap = arma::mat{{1, 2}, {2, 4}}; }), "'wrap' must have linearly independent columns"));
    CHECK(has(msg([](LL &L) { L.wrap = arma::mat(arma::vec{0, 0}); }), "'wrap' must not have a zero vector"));
    CHECK(has(msg([](LL &L) { L.wrap = arma::mat(3, 1, arma::fill::ones); }), "'wrap' must be finite with 2 rows"));
    CHECK(has(msg([](LL &L) { L.wrap = arma::mat(2, 3, arma::fill::eye); }), "'wrap' must be finite with 2 rows"));
    CHECK(has(msg([](LL &L) { L.wrap = arma::mat(arma::vec{100.0, NaN}); }), "'wrap'"));
    CHECK(msg([](LL &L) { L.wrap = arma::mat{{100, 0}, {0, 100}}; }) == "");
    CHECK(msg([](LL &L) { L.wrap = arma::mat(arma::vec{100, 0}); }) == "");
    CHECK(has(msg([](LL &L) { L.pairing = arma::umat(arma::uvec{2, 0}); }), "'pairing' has an index outside"));
    CHECK(has(msg([](LL &L) { L.pairing = arma::umat(arma::uvec{0, 2}); }), "'pairing' has an index outside"));
    CHECK(has(msg([](LL &L) { L.pairing = arma::umat{{0, 0}, {1, 1}}; }), "'pairing' must not list a pair twice"));
    CHECK(has(msg([](LL &L) { L.pairing = arma::umat(3, 1, arma::fill::zeros); }), "'pairing' must have 2 rows"));
    // Sites are numbered by name: the two sectors of BS1 are one site
    CHECK(msg([](LL &L) { L.pairing = arma::umat{{1, 0}, {1, 1}}; }) == "");
    CHECK(has(msg([](LL &L) { L.links.i_tx = {0}; }), "links: "));

    // Names must fit a pairing cell and a CSV file
    for (const std::string &bad : {"a,b", "a;b", "a\"b", "a\nb", "a\rb", " a", "a ", "\ta"})
    {
        CHECK(has(msg([&](LL &L) { L.tx.name[2] = bad; }), "tx: Node names must not contain"));
        CHECK(has(msg([&](LL &L) { L.rx.name[1] = bad; }), "rx: Node names must not contain"));
    }
    CHECK(msg([](LL &L) { L.tx.name[2] = "a b#c'd"; }) == "");

    // What the files cannot hold
    CHECK(has(msg([](LL &L) { L.seed = -5; }), "'seed' must be >= -1"));
    CHECK(has(msg([](LL &L) { L.links.seed = -2; }), "'seed' must be >= -1"));
    CHECK(msg([](LL &L) { L.seed = -1; L.tx.seed = {1LL << 62, 1LL << 62, -1}; }) == "");
    CHECK(has(msg([](LL &L) { L.rx.scenario = {"", "", "a\nb", ""}; }), "path separator or a line break"));
    CHECK(has(msg([](LL &L) { L.tx.scenpar_override[""] = arma::vec{1, 1, 1}; }), "'scenpar_override' must not have an empty key"));
    CHECK(has(msg([](LL &L) { L.tx.scenpar_override["a\nb"] = arma::vec{1, 1, 1}; }), "'scenpar_override'"));
    CHECK(has(msg([](LL &L) { L.tx.extra_names = {"x"}; L.tx.extra = {{"1", "2", "3"}}; }), "'extra_names'"));
    CHECK(has(msg([](LL &L) { L.tx.extra_names = {"a", "a"}; L.tx.extra = {{"1", "2", "3"}, {"1", "2", "3"}}; }), "'extra_names'"));
    CHECK(has(msg([](LL &L) { L.tx.extra_names = {"scenpar.DS_VS"}; L.tx.extra = {{"1", "2", "3"}}; }), "'extra_names'"));
    CHECK(has(msg([](LL &L) { L.tx.extra_names = {"a\nb"}; L.tx.extra = {{"1", "2", "3"}}; }), "'extra_names'"));
    CHECK(has(msg([](LL &L) { L.tx.extra_names = {"a"}; L.tx.extra = {{"1", "2\n", "3"}}; }), "line break"));
    CHECK(msg([](LL &L) { L.tx.extra_names = {"a"}; L.tx.extra = {{"1", "2,3", "\"q\""}}; }) == "");

    // A node with one row and relative = 1 needs a direction
    CHECK(has(msg([](LL &L) { L.rx.relative = {0, 1}; L.rx.direction(0, 3) = NaN; }), "'relative' = 1 needs a 'direction'"));
    CHECK(msg([](LL &L) { L.rx.relative = {0, 1}; }) == ""); // UT2 moves along 120 degrees
    CHECK(msg([](LL &L) { L.rx.relative = {1, 0}; }) == ""); // a track
    CHECK(has(msg([](LL &L) { L.rx.relative = {0, 1}; L.rx.direction.reset(); }), "'relative' = 1 needs a 'direction'"));
}

TEST_CASE("layout_links - members and is_valid")
{
    quadriga_lib::layout_links K;
    CHECK(K.empty());
    CHECK(K.n_link() == 0);
    CHECK(K.is_valid() == "");
    CHECK(K.seed == -1);
    CHECK(K.anchor_freq == 2.0e9);
    CHECK(!K.pol_variability);

    K.i_tx = {0, 1, 1};
    K.i_rx = {0, 0, 1};
    K.i_segment = {0, 0, 0};
    K.image = {0, 0, 2};
    K.i_conf = {0, 1, 1};
    K.conf_names = {"UMa_LOS", "UMa_NLOS"};
    K.conf_files = {"a/UMa_LOS.conf", "a/UMa_NLOS.conf"};
    K.los = {1, 0, 0};
    K.indoor = {0, 1, -1};
    K.floor_id = {0, 2, 0};
    K.d2d_in = arma::vec{0.0, 12.0, NaN};
    K.o2i_rnd = arma::mat(2, 4, arma::fill::value(0.5));
    K.o2i_model = {0, 1};
    K.gbsm_seed = {5, 6, 7};
    K.seed = 3;
    K.method = "38.901";
    K.layout_hash = 99;
    CHECK(!K.empty());
    CHECK(K.n_link() == 3);
    CHECK(K.is_valid() == "");

    auto bad = [&](auto change) { quadriga_lib::layout_links Q = K; change(Q); return Q.is_valid(); };
    using KK = quadriga_lib::layout_links;
    CHECK(has(bad([](KK &Q) { Q.los = {1, 0}; }), "one entry per link"));
    CHECK(has(bad([](KK &Q) { Q.gbsm_seed = {1, 2}; }), "one entry per link"));
    CHECK(has(bad([](KK &Q) { Q.i_conf = {0, 1, 2}; }), "'i_conf'"));
    CHECK(has(bad([](KK &Q) { Q.los = {1, 0, 2}; }), "'los' must be 0 or 1"));
    CHECK(has(bad([](KK &Q) { Q.indoor = {0, 1, 4}; }), "'indoor'"));
    CHECK(has(bad([](KK &Q) { Q.d2d_in = arma::vec{0.0, -1.0, NaN}; }), "'d2d_in'"));
    CHECK(has(bad([](KK &Q) { Q.conf_files = {"a"}; }), "'conf_files'"));
    CHECK(has(bad([](KK &Q) { Q.o2i_model = {0}; }), "'o2i_rnd'"));
    CHECK(has(bad([](KK &Q) { Q.o2i_rnd = arma::mat(2, 3); }), "'o2i_rnd'"));
    CHECK(has(bad([](KK &Q) { Q.anchor_freq = 0.0; }), "'anchor_freq'"));

    // reset empties all members and restores the defaults
    K.pol_variability = true;
    K.anchor_freq = 5e9;
    K.sos_preset = "x";
    K.reset();
    CHECK(K.empty());
    CHECK(K.i_rx.is_empty());
    CHECK(K.conf_names.empty());
    CHECK(K.conf_files.empty());
    CHECK(K.o2i_rnd.is_empty());
    CHECK(K.d2d_in.is_empty());
    CHECK(K.method.empty());
    CHECK(K.sos_preset.empty());
    CHECK(K.seed == -1);
    CHECK(K.anchor_freq == 2.0e9);
    CHECK(!K.pol_variability);
    CHECK(K.layout_hash == 0);
    CHECK(K.is_valid() == "");
}

TEST_CASE("layout_csv_read - seeds are 64-bit integers")
{
    qd_layout_files dir("seed64");
    const std::string tx = "name,x,y,z,seed,layout_seed\nA,0,0,0,9007199254740993,4611686018427387904\nB,1,0,0,4611686018427387905,\n";
    const quadriga_lib::layout L = read_text(dir, tx, "");
    CHECK(L.seed == (1LL << 62));
    CHECK(L.tx.seed[0] == (1LL << 53) + 1);
    CHECK(L.tx.seed[1] == (1LL << 62) + 1);
    quadriga_lib::layout_csv_write(dir.path("w.csv"), "", L);
    CHECK(has(dir.read("w.csv"), "\nA,0,0,0,9007199254740993,4611686018427387904\nB,1,0,0,4611686018427387905,\n"));
    const quadriga_lib::layout L2 = quadriga_lib::layout_csv_read(dir.path("w.csv"), "", {}, false);
    CHECK(L2.seed == (1LL << 62));
    CHECK(L2.tx.seed[0] == (1LL << 53) + 1);
    CHECK(L2.tx.seed[1] == (1LL << 62) + 1);

    // A layout object
    quadriga_lib::layout M = qd_layout_build();
    M.seed = (1LL << 62) + 3;
    M.tx.seed = {(1LL << 53) + 1, (1LL << 53) + 1, std::numeric_limits<long long>::max()};
    quadriga_lib::layout_csv_write(dir.path("w.csv"), "", M);
    CHECK(has(dir.read("w.csv"), "9007199254740993"));
    CHECK(has(dir.read("w.csv"), "9223372036854775807"));
    const quadriga_lib::layout M2 = quadriga_lib::layout_csv_read(dir.path("w.csv"), "", {}, false);
    CHECK(M2.seed == M.seed);
    CHECK(arma::all(M2.tx.seed == M.tx.seed));

    // Not an integer, below -1, above the range
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z,seed\nA,0,0,0,9223372036854775808\n", ""); }), "line 2, column 'seed'"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z,seed\nA,0,0,0,1.5\n", ""); }), "must be an integer"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z,seed\nA,0,0,0,1e3\n", ""); }), "must be an integer"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z,layout_seed\nA,0,0,0,-2\n", ""); }), "line 2, column 'layout_seed'"));
}

TEST_CASE("layout_csv_read - names, pairing and relative are checked after reading")
{
    qd_layout_files dir("read_checked");
    // A name that cannot be stored in a pairing cell
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z\n\"a,b\",0,0,0\n", ""); }), "Node names must not contain"));
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z\n\" a\",0,0,0\n", ""); }), "Node names must not contain"));

    // A terminal at the only site: its empty pairing cell means all sites, and there is none
    const std::string tx = "name,x,y,z\nA,0,0,10\n";
    const std::string msg = qd_layout_throws([&] { read_text(dir, tx, "name,x,y,z,pairing\nU1,5,0,1.5,A\nU2,0,0,10,\n"); });
    CHECK(has(msg, "rx.csv, line 3"));
    CHECK(has(msg, "terminal 'U2' has no pair"));
    CHECK(qd_layout_throws([&] { read_text(dir, tx, "name,x,y,z,pairing\nU1,5,0,1.5,A\nU2,1,0,10,\n"); }).empty());

    // The nodes of a terminal have one pairing
    CHECK(has(qd_layout_throws([&] { read_text(dir, "name,x,y,z\nA,0,0,10\nB,5,0,10\n", "name,sector,x,y,z,pairing\nU,0,5,0,1.5,A\nU,1,,,,B\n"); }), "'pairing'"));

    // relative = 1 on a node with one row needs a direction
    CHECK(has(qd_layout_throws([&] { read_text(dir, "", "name,x,y,z,relative,speed\nU,0,0,1.5,1,3\n"); }), "'relative' = 1 needs a 'direction'"));
    CHECK(read_text(dir, "", "name,x,y,z,relative,speed,direction_az\nU,0,0,1.5,1,3,45\n").rx.n_node() == 1);
}
