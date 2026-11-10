// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "quadriga_lib.hpp"

TEST_CASE("Tools - Path 2 Tube")
{
    arma::mat path = {{1.0, 10.0, 10.0, 12.0},
                      {0.0, 0.0, 10.0, 0.0},
                      {0.0, 0.0, 0.0, 0.0}};

    arma::mat vert;
    arma::umat faces;
    quadriga_lib::path_to_tube(&path, &vert, &faces, 0.1, 3);
}

TEST_CASE("Tools - Path 2 Tube Short segments")
{
    arma::fmat path = {{1.0, 1.0, 1.0, 10.0, 10.0, 12.0, 12.001},
                       {0.0, 0.0, 0.001, 5.0, 5.0, 0.0, 0.0},
                       {0.0, 0.0, 0.0, 1.0, 1.001, 0.0, 0.0}};

    arma::fmat vert;
    arma::umat faces;
    quadriga_lib::path_to_tube(&path, &vert, &faces, 0.1f, 4);

    CHECK(vert.n_cols == 12ULL);
}

TEST_CASE("Channel - OBJ Export")
{
    quadriga_lib::channel<double> c;

    REQUIRE_THROWS_AS(c.write_paths_to_obj_file("test_chan.obj"), std::invalid_argument);

    // Single transmitter
    c.tx_pos = {0.0, 0.0, 1.0};
    c.tx_pos = c.tx_pos.t();

    // 2 snapshots
    c.rx_pos = {{20.0, 3.2},
                {0.0, 1.0},
                {1.0, 1.0}};

    REQUIRE_THROWS_AS(c.write_paths_to_obj_file("test_chan.obj"), std::invalid_argument);

    // 2 Paths: LOS and 1 single-bounce
    arma::u32_vec no_interact = {0, 1};
    c.no_interact.push_back(no_interact);
    c.no_interact.push_back(no_interact);

    arma::mat interact_coord = {0.0, 10.0, 1.0};
    c.interact_coord.push_back(interact_coord.t());
    c.interact_coord.push_back(interact_coord.t());

    REQUIRE_THROWS_AS(c.write_paths_to_obj_file("test_chan.obj"), std::invalid_argument);

    arma::mat M(8, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 0.1;
    M(6, 0) = -1.0;
    M(6, 1) = -0.1;
    c.path_polarization.push_back(M);
    c.path_polarization.push_back(M);

    c.center_frequency = 1.0e9;

    // This is the minimum data required from the RT simulation:
    // - TX and RX positions, Polarization Matrix and Path interaction coordinates
    c.write_paths_to_obj_file("test_chan.obj");
    c.write_paths_to_obj_file("test_chan.obj", 1);
    c.write_paths_to_obj_file("test_chan.obj", 0, -50.0);
    c.write_paths_to_obj_file("test_chan.obj", 0, -50.0, -82.6);

    REQUIRE_THROWS_AS(c.write_paths_to_obj_file("test_chan.obj", 0, -50.0, -82.6, "bla"), std::invalid_argument);

    c.write_paths_to_obj_file("test_chan.obj", 0, -50.0, -82.6, "turbo");
    c.write_paths_to_obj_file("test_chan.obj", 0, -50.0, -82.6, "jet", {1, 0, 0, 1}, 0.5, 0.2, 12);

    std::remove("test_chan.obj");
    std::remove("test_chan.mtl");
}
TEST_CASE("Channel - OBJ Export skips paths with non-finite coordinates")
{
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat({0.0, 0.0, 1.0}).t();
    c.rx_pos = arma::mat({20.0, 0.0, 1.0}).t();
    c.center_frequency = 1.0e9;

    // 2 paths: LOS and a single-bounce path with a NaN interaction point
    c.no_interact.push_back(arma::u32_vec({0, 1}));
    c.interact_coord.push_back(arma::mat({arma::datum::nan, 10.0, 1.0}).t());
    c.coeff_re.push_back(arma::cube(1, 1, 2, arma::fill::value(1.0e-3)));
    c.coeff_im.push_back(arma::cube(1, 1, 2, arma::fill::zeros));
    c.delay.push_back(arma::cube(1, 1, 2, arma::fill::value(1.0e-7)));

    // The gain range must be positive
    REQUIRE_THROWS_AS(c.write_paths_to_obj_file("test_chan_nonfinite.obj", 0, -100.0, -100.0), std::invalid_argument);
    REQUIRE_THROWS_AS(c.write_paths_to_obj_file("test_chan_nonfinite.obj", 0, -100.0, -60.0), std::invalid_argument);

    c.write_paths_to_obj_file("test_chan_nonfinite.obj", 0, 0.0, -100.0);

    std::ifstream file("test_chan_nonfinite.obj");
    REQUIRE(file.is_open());
    std::string line;
    int n_obj = 0, n_nan = 0;
    while (std::getline(file, line))
    {
        if (line.rfind("o QuaDRiGa_path", 0) == 0)
            ++n_obj;
        if (line.rfind("#", 0) != 0 && line.find("nan") != std::string::npos) // the comment table also lists skipped paths
            ++n_nan;
    }
    file.close();
    std::remove("test_chan_nonfinite.obj");
    std::remove("test_chan_nonfinite.mtl");

    CHECK(n_obj == 1);
    CHECK(n_nan == 0);
}
