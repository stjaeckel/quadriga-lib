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
    auto antI = quadriga_lib::qdant_read<float>("test.qdant");

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

TEST_CASE("Arrayant writing/reading from QDANT file - Multi-Element Array")
{
    double pi = arma::datum::pi;

    quadriga_lib::arrayant<double> ant;
    ant.name = "bla";
    ant.elevation_grid = arma::linspace<arma::vec>(-90.0, 90.0, 5) * pi / 180.0;
    ant.azimuth_grid = {-pi, 0.0f, 0.5f * pi};

    ant.e_theta_re = arma::cube(5, 3, 2, arma::fill::value(1.0));
    ant.e_theta_im = arma::cube(5, 3, 2, arma::fill::value(2.0));
    ant.e_phi_re = arma::cube(5, 3, 2, arma::fill::value(3.0));
    ant.e_phi_im = arma::cube(5, 3, 2, arma::fill::value(4.0));

    ant.e_theta_re.slice(1).fill(5.0);
    ant.e_theta_im.slice(1).fill(6.0);
    ant.e_phi_re.slice(1).fill(7.0);
    ant.e_phi_im.slice(1).fill(8.0);

    arma::mat A = arma::reshape(arma::linspace<arma::vec>(1.0, 6.0, 6), 2, 3);
    arma::mat B = arma::reshape(arma::linspace<arma::vec>(2.0, 7.0, 6), 2, 3);
    ant.coupling_re = A;
    ant.coupling_im = B;

    unsigned id = ant.qdant_write("test2.qdant");
    CHECK(id == 1);

    auto x = quadriga_lib::qdant_read<double>("test2.qdant");
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(1.0)), x.e_theta_re.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(2.0)), x.e_theta_im.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(3.0)), x.e_phi_re.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(4.0)), x.e_phi_im.slice(0), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(5.0)), x.e_theta_re.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(6.0)), x.e_theta_im.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(7.0)), x.e_phi_re.slice(1), "absdiff", 1e-5));
    CHECK(arma::approx_equal(arma::mat(5, 3, arma::fill::value(8.0)), x.e_phi_im.slice(1), "absdiff", 2e-5));
    CHECK(arma::approx_equal(A, x.coupling_re, "absdiff", 1e-5));
    CHECK(arma::approx_equal(B, x.coupling_im, "absdiff", 1e-5));

    std::remove("test2.qdant");
}

TEST_CASE("Arrayant writing to QDANT file - Complex test")
{
    // Create new file
    std::ofstream f;
    f.open("test.qdant");
    f << "<qdant><arrayant>" << std::endl;
    f << "<name>bla</name>" << std::endl;
    f << "<ElevationGrid>-90 -45 0 45 90</ElevationGrid>" << std::endl;
    f << "<AzimuthGrid>-180 0 90</AzimuthGrid>" << std::endl;
    f << "<EthetaMag>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15</EthetaMag>" << std::endl;
    f << "</arrayant></qdant>" << std::endl;
    f.close();

    double pi = arma::datum::pi;

    quadriga_lib::arrayant<double> ant;
    ant.azimuth_grid = {-0.75 * pi, 0.0, 0.75 * pi, pi};
    ant.elevation_grid = {-0.45 * pi, 0.0, 0.45 * pi};

    arma::mat A = arma::linspace(1.0, 12.0, 12);
    A.reshape(3, 4);

    arma::cube B;
    B.zeros(3, 4, 1);
    B.slice(0) = A;

    ant.e_theta_re = B * 0.5;
    ant.e_theta_im = B * 0.002;
    ant.e_phi_re = -B;
    ant.e_phi_im = -B * 0.001;

    arma::mat C = {1.0, 2.0, 4.0};
    ant.element_pos = C.t();

    ant.coupling_re = {1.0};
    ant.coupling_im = {0.1};
    ant.center_frequency = 2.0e9;
    ant.name = "name";

    // Append new antenna
    unsigned id = ant.qdant_write("test.qdant");

    CHECK(id == 2);

    arma::Mat<unsigned> layout, layout2;

    auto antI = quadriga_lib::qdant_read<float>("test.qdant", 1, &layout2);

    CHECK(antI.name == "bla");

    antI = quadriga_lib::qdant_read<float>("test.qdant", 2);
    CHECK(antI.name == "name");

    layout = {1, 2};
    CHECK(arma::all(arma::vectorise(layout2 - layout) == 0));

    id = ant.qdant_write("test.qdant", 112);
    CHECK(id == 112);

    // Layout can only contain valid entries (error expected)
    layout = {{1, 2, 112}, {112, 112, 6}};
    REQUIRE_THROWS_AS(ant.qdant_write("test.qdant", 5, layout), std::invalid_argument);

    id = ant.qdant_write("test.qdant", 6, layout);
    CHECK(id == 6);

    antI = quadriga_lib::qdant_read<float>("test.qdant", 6, &layout2);

    CHECK(arma::all(arma::vectorise(layout2 - layout) == 0));

    std::remove("test.qdant");
}
