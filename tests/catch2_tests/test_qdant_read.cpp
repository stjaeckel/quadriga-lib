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

TEST_CASE("Arrayant reading from QDANT file - Minimal test")
{
    std::ofstream f;
    f.open("test.qdant");
    f << "<qdant><arrayant>" << std::endl;
    f << "<name>bla</name>" << std::endl;
    f << "<ElevationGrid>-90 -45 0 45 90</ElevationGrid>" << std::endl;
    f << "<AzimuthGrid>-180 0 90</AzimuthGrid>" << std::endl;
    f << "<EthetaMag>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15</EthetaMag>" << std::endl;
    f << "</arrayant></qdant>" << std::endl;
    f.close();

    float pi = arma::datum::pi;
    quadriga_lib::arrayant<float> x("test.qdant");

    arma::fmat A = 20 * log10(x.e_theta_re.slice(0));
    arma::fmat B = arma::reshape(arma::linspace<arma::fvec>(1, 15, 15), 3, 5).t();
    CHECK(arma::approx_equal(A, B, "absdiff", 1e-6));

    B.zeros();
    CHECK(arma::approx_equal(x.e_theta_im.slice(0), B, "absdiff", 1e-6));
    CHECK(arma::approx_equal(x.e_phi_re.slice(0), B, "absdiff", 1e-6));
    CHECK(arma::approx_equal(x.e_phi_im.slice(0), B, "absdiff", 1e-6));

    B = {-pi, 0.0f, pi / 2.0f};
    CHECK(arma::approx_equal(x.azimuth_grid, B.t(), "absdiff", 1e-6));

    B = {-pi / 2.0f, -pi / 4.0f, 0.0f, pi / 4.0f, pi / 2.0f};
    CHECK(arma::approx_equal(x.elevation_grid, B.t(), "absdiff", 1e-6));

    B = {0.0f, 0.0f, 0.0f};
    CHECK(arma::approx_equal(x.element_pos, B.t(), "absdiff", 1e-6));

    B = {1.0f};
    CHECK(arma::approx_equal(x.coupling_re, B, "absdiff", 1e-6));

    B = {0.0f};
    CHECK(arma::approx_equal(x.coupling_im, B, "absdiff", 1e-6));
    CHECK(std::abs(x.center_frequency - 299792447.0f) < 0.1f);

    std::remove("test.qdant");
}

TEST_CASE("Arrayant reading from QDANT file - More complex test")
{
    std::ofstream f;
    f.open("test.qdant");
    f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?><qdant xmlns:xx=\"test\">" << std::endl;
    f << "<xx:layout>1,1 1,1 1,1</xx:layout>" << std::endl;
    f << "<xx:arrayant id=\"1\">" << std::endl;
    f << "<xx:AzimuthGrid>-90 -45 0 45 90</xx:AzimuthGrid>" << std::endl;
    f << "<xx:ElevationGrid>-90 0 90</xx:ElevationGrid>" << std::endl;
    f << "<xx:EphiMag>0  0  0  0  0  0  0  0  0  0  0  0  0  0  0</xx:EphiMag>" << std::endl;
    f << "<xx:EphiPhase>90  90  90  90  90  90  90  90  90  90  90  90  90  90  90</xx:EphiPhase>" << std::endl;
    f << "<xx:EthetaMag>3  3  3  3  3  3  3  3  3  3  3  3  3  3  3</xx:EthetaMag>" << std::endl;
    f << "<xx:EthetaPhase>-90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90 -90</xx:EthetaPhase>" << std::endl;
    f << "<xx:ElementPosition>1,2,3</xx:ElementPosition>" << std::endl;
    f << "<xx:CouplingAbs>1</xx:CouplingAbs>" << std::endl;
    f << "<xx:CouplingPhase>45</xx:CouplingPhase>" << std::endl;
    f << "<xx:CenterFrequency>3e9</xx:CenterFrequency>" << std::endl;
    f << "</xx:arrayant></qdant>" << std::endl;
    f.close();

    double pi = arma::datum::pi;

    arma::Mat<unsigned> layout;
    quadriga_lib::arrayant<double> x("test.qdant", 1, &layout);

    arma::mat B = {-pi / 2.0, -pi / 4.0, 0.0, pi / 4.0, pi / 2.0};
    CHECK(arma::approx_equal(x.azimuth_grid, B.t(), "absdiff", 1e-13));

    B = {-pi / 2.0, 0.0, pi / 2.0};
    CHECK(arma::approx_equal(x.elevation_grid, B.t(), "absdiff", 1e-13));

    B.zeros(3, 5);
    CHECK(arma::approx_equal(x.e_theta_re.slice(0), B, "absdiff", 1e-13));
    CHECK(arma::approx_equal(x.e_phi_re.slice(0), B, "absdiff", 1e-13));

    B.fill(-std::sqrt(std::pow(10, 0.3)));
    CHECK(arma::approx_equal(x.e_theta_im.slice(0), B, "absdiff", 1e-13));

    B.ones();
    CHECK(arma::approx_equal(x.e_phi_im.slice(0), B, "absdiff", 1e-13));

    B = {1.0, 2.0, 3.0};
    CHECK(arma::approx_equal(x.element_pos, B.t(), "absdiff", 1e-13));

    B = {1.0 / std::sqrt(2.0)};
    CHECK(arma::approx_equal(x.coupling_re, B, "absdiff", 1e-13));
    CHECK(arma::approx_equal(x.coupling_im, B, "absdiff", 1e-13));

    CHECK(std::abs(x.center_frequency - 3.0e9) < 0.1f);
    CHECK(x.name == "unknown");
    CHECK(arma::all(arma::vectorise(layout) == 1));

    std::remove("test.qdant");
}