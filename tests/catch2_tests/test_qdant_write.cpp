// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"

#include <iostream>
#include <fstream>
#include <string>

TEST_CASE("Arrayant writing to QDANT file - Minimal test")
{
    std::remove("test.qdant");

    float pi = arma::datum::pi;

    quadriga_lib::arrayant<float> ant;
    ant.azimuth_grid = {-0.75f * pi, 0.0f, 0.75f * pi, pi};
    ant.elevation_grid = {-0.45f * pi, 0.0f, 0.45f * pi};

    arma::mat A = arma::linspace(1.0, 12.0, 12);
    A.reshape(3, 4);

    arma::fcube B;
    B.zeros(3, 4, 1);
    B.slice(0) = arma::conv_to<arma::fmat>::from(A);

    ant.e_theta_re = B * 0.5f;
    ant.e_theta_im = B * 0.002f;
    ant.e_phi_re = -B;
    ant.e_phi_im = -B * 0.001f;

    arma::fmat C = {1.0f, 2.0f, 4.0f};
    ant.element_pos = C.t();

    ant.coupling_re = {1.0f};
    ant.coupling_im = {0.1f};
    ant.center_frequency = 2.0e9f;
    ant.name = "name";

    unsigned id = ant.qdant_write("test.qdant");
    CHECK(id == 1);

    // Load file again and compare
    quadriga_lib::arrayant<float> antI("test.qdant");

    CHECK(arma::approx_equal(antI.azimuth_grid, ant.azimuth_grid, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.elevation_grid, ant.elevation_grid, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.e_theta_re, ant.e_theta_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_theta_im, ant.e_theta_im, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_phi_re, ant.e_phi_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.e_phi_im, ant.e_phi_im, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.element_pos, ant.element_pos, "absdiff", 1e-6));
    CHECK(arma::approx_equal(antI.coupling_re, ant.coupling_re, "absdiff", 1e-4));
    CHECK(arma::approx_equal(antI.coupling_im, ant.coupling_im, "absdiff", 1e-4));
    CHECK(std::abs(antI.center_frequency - ant.center_frequency) < 0.1f);
    CHECK(antI.name == ant.name);

    std::remove("test.qdant");
}
