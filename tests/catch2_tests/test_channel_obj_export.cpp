// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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
#include <iostream>

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

    REQUIRE_THROWS_AS(c.export_obj_file("test_chan.obj"), std::invalid_argument);

    // Single transmitter
    c.tx_pos = {0.0, 0.0, 1.0};
    c.tx_pos = c.tx_pos.t();

    // 2 snapshots
    c.rx_pos = {{20.0, 3.2},
                {0.0, 1.0},
                {1.0, 1.0}};

    REQUIRE_THROWS_AS(c.export_obj_file("test_chan.obj"), std::invalid_argument);

    // 2 Paths: LOS and 1 single-bounce
    arma::u32_vec no_interact = {0, 1};
    c.no_interact.push_back(no_interact);
    c.no_interact.push_back(no_interact);

    arma::mat interact_coord = {0.0, 10.0, 1.0};
    c.interact_coord.push_back(interact_coord.t());
    c.interact_coord.push_back(interact_coord.t());

    REQUIRE_THROWS_AS(c.export_obj_file("test_chan.obj"), std::invalid_argument);

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
    c.export_obj_file("test_chan.obj");
    c.export_obj_file("test_chan.obj", 1);
    c.export_obj_file("test_chan.obj", 0, -50.0);
    c.export_obj_file("test_chan.obj", 0, -50.0, -82.6);

    REQUIRE_THROWS_AS(c.export_obj_file("test_chan.obj", 0, -50.0, -82.6, "bla"), std::invalid_argument);

    c.export_obj_file("test_chan.obj", 0, -50.0, -82.6, "turbo");
    c.export_obj_file("test_chan.obj", 0, -50.0, -82.6, "jet", {1,0,0,1}, 0.5, 0.2, 12);

    std::remove("test_chan.obj");
    std::remove("test_chan.mtl");
}