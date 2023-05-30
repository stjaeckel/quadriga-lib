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

#include <iostream>
#include <string>

TEST_CASE("Arrayant interpolation - Minimal test")
{
    double pi = arma::datum::pi;
    quadriga_lib::arrayant<double> ant;
    ant.e_theta_re.zeros(1, 2, 1), ant.e_theta_re.slice(0) = {-2.0, 2.0};
    ant.e_theta_im.zeros(1, 2, 1), ant.e_theta_im.slice(0) = {-1.0, 1.0};
    ant.e_phi_re.zeros(1, 2, 1), ant.e_phi_re.slice(0) = {3.0, 1.0};
    ant.e_phi_im.zeros(1, 2, 1), ant.e_phi_im.slice(0) = {6.0, 2.0};
    ant.azimuth_grid = {0.0, pi};
    ant.elevation_grid = {0.0};

    arma::mat azimuth = {0.0, pi / 4.0, pi / 2.0, 3.0 * pi / 4.0};
    arma::mat elevation = {-0.5, 0.0, 0.0, 0.5};
    arma::Col<unsigned> i_element = {1};
    arma::Cube<double> orientation(3, 1, 1);
    arma::mat element_pos_i(3, 1);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);

    arma::mat T = {-2.0, -1.0, 0.0, 1.0};
    CHECK(arma::approx_equal(V_re, T, "absdiff", 1e-13));

    T = {-1.0, -0.5, 0.0, 0.5};
    CHECK(arma::approx_equal(V_im, T, "absdiff", 1e-13));

    T = {3.0, 2.5, 2.0, 1.5};
    CHECK(arma::approx_equal(H_re, T, "absdiff", 1e-13));

    T = {6.0, 5.0, 4.0, 3.0};
    CHECK(arma::approx_equal(H_im, T, "absdiff", 1e-13));

    T.zeros();
    CHECK(arma::approx_equal(dist, T, "absdiff", 1e-13));

    CHECK(arma::approx_equal(azimuth_loc, azimuth, "absdiff", 1e-13));
    CHECK(arma::approx_equal(elevation_loc, elevation, "absdiff", 1e-13));
}

TEST_CASE("Arrayant interpolation - Simple interpolation in el-direction")
{
    double pi = arma::datum::pi;
    quadriga_lib::arrayant<double> ant;

    arma::mat T;
    T = {-2.0, 2.0}, ant.e_theta_re.zeros(2, 1, 1), ant.e_theta_re.slice(0) = T.t();
    T = {-1.0, 1.0}, ant.e_theta_im.zeros(2, 1, 1), ant.e_theta_im.slice(0) = T.t();
    T = {3.0, 1.0}, ant.e_phi_re.zeros(2, 1, 1), ant.e_phi_re.slice(0) = T.t();
    T = {6.0, 2.0}, ant.e_phi_im.zeros(2, 1, 1), ant.e_phi_im.slice(0) = T.t();
    ant.azimuth_grid = {0.0};
    ant.elevation_grid = {0.0, pi / 2.0};

    arma::mat azimuth = {-0.5, 0.0, 0.0, 0.5};
    arma::mat elevation = {0.0, pi / 8.0, pi / 4.0, 3.0 * pi / 8.0};
    arma::Col<unsigned> i_element = {1};
    arma::Cube<double> orientation(3, 1, 1);
    arma::mat element_pos_i(3, 1);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);

    T = {-2.0, -1.0, 0.0, 1.0};
    CHECK(arma::approx_equal(V_re, T, "absdiff", 1e-14));

    T = {-1.0, -0.5, 0.0, 0.5};
    CHECK(arma::approx_equal(V_im, T, "absdiff", 1e-14));

    T = {3.0, 2.5, 2.0, 1.5};
    CHECK(arma::approx_equal(H_re, T, "absdiff", 1e-14));

    T = {6.0, 5.0, 4.0, 3.0};
    CHECK(arma::approx_equal(H_im, T, "absdiff", 1e-14));

    T.zeros();
    CHECK(arma::approx_equal(dist, T, "absdiff", 1e-14));

    CHECK(arma::approx_equal(azimuth_loc, azimuth, "absdiff", 1e-14));
    CHECK(arma::approx_equal(elevation_loc, elevation, "absdiff", 1e-14));
}

TEST_CASE("Arrayant interpolation - Spheric interpolation in az-direction")
{
    double pi = arma::datum::pi;
    quadriga_lib::arrayant<double> ant;

    arma::mat T;
    T = {1.0, 0.0}, ant.e_theta_re.zeros(1, 2, 1), ant.e_theta_re.slice(0) = T;
    T = {0.0, 1.0}, ant.e_theta_im.zeros(1, 2, 1), ant.e_theta_im.slice(0) = T;
    T = {-2.0, 0.0}, ant.e_phi_re.zeros(1, 2, 1), ant.e_phi_re.slice(0) = T;
    T = {0.0, -1.0}, ant.e_phi_im.zeros(1, 2, 1), ant.e_phi_im.slice(0) = T;
    ant.azimuth_grid = {0.0, pi};
    ant.elevation_grid = {0.0};

    arma::mat C = {0.0, 1.0, 2.0, 3.0};
    arma::mat azimuth = C * pi / 4.0;
    arma::mat elevation = {0.0, 0.0, 0.0, 0.0};
    arma::Cube<double> orientation(3, 1, 1);

    arma::mat V_re, V_im, H_re, H_im, dist;
    ant.interpolate(azimuth, elevation, orientation, &V_re, &V_im, &H_re, &H_im, &dist);

    T = arma::cos(C * pi / 8.0);
    CHECK(arma::approx_equal(V_re, T, "absdiff", 1e-14));

    T = arma::sin(C * pi / 8.0);
    CHECK(arma::approx_equal(V_im, T, "absdiff", 1e-14));

    T = {2, 1.75, 1.5, 1.25};
    T = -arma::cos(C * pi / 8.0) % T;
    CHECK(arma::approx_equal(H_re, T, "absdiff", 1e-14));

    T = {2, 1.75, 1.5, 1.25};
    T = -arma::sin(C * pi / 8.0) % T;
    CHECK(arma::approx_equal(H_im, T, "absdiff", 1e-14));
}

TEST_CASE("Arrayant interpolation - Spheric interpolation in el-direction")
{
    double pi = arma::datum::pi;
    quadriga_lib::arrayant<double> ant;

    arma::mat T;
    T = {1.0, 0.0}, ant.e_theta_re.zeros(2, 1, 1), ant.e_theta_re.slice(0) = T.t();
    T = {0.0, 1.0}, ant.e_theta_im.zeros(2, 1, 1), ant.e_theta_im.slice(0) = T.t();
    T = {-2.0, 0.0}, ant.e_phi_re.zeros(2, 1, 1), ant.e_phi_re.slice(0) = T.t();
    T = {0.0, -1.0}, ant.e_phi_im.zeros(2, 1, 1), ant.e_phi_im.slice(0) = T.t();
    ant.azimuth_grid = {0.0};
    ant.elevation_grid = {0.0, pi / 2.0};

    arma::mat C = {0.0, 1.0, 2.0, 3.0};
    arma::mat azimuth = {0.0, 0.0, 0.0, 0.0};
    arma::mat elevation = C * pi / 8.0;
    arma::Cube<double> orientation(3, 1, 1);

    arma::mat V_re, V_im, H_re, H_im, dist;
    ant.interpolate(azimuth, elevation, orientation, &V_re, &V_im, &H_re, &H_im, &dist);

    T = arma::cos(C * pi / 8.0);
    CHECK(arma::approx_equal(V_re, T, "absdiff", 1e-14));

    T = arma::sin(C * pi / 8.0);
    CHECK(arma::approx_equal(V_im, T, "absdiff", 1e-14));

    T = {2, 1.75, 1.5, 1.25};
    T = -arma::cos(C * pi / 8.0) % T;
    CHECK(arma::approx_equal(H_re, T, "absdiff", 1e-14));

    T = {2, 1.75, 1.5, 1.25};
    T = -arma::sin(C * pi / 8.0) % T;
    CHECK(arma::approx_equal(H_im, T, "absdiff", 1e-14));
}

TEST_CASE("Arrayant interpolation - Spheric interpolation in az-direction with z-rotation")
{
    double pi = arma::datum::pi;
    quadriga_lib::arrayant<double> ant;

    arma::mat T;
    T = {1.0, 0.0}, ant.e_theta_re.zeros(1, 2, 1), ant.e_theta_re.slice(0) = T;
    T = {0.0, 1.0}, ant.e_theta_im.zeros(1, 2, 1), ant.e_theta_im.slice(0) = T;
    T = {-2.0, 0.0}, ant.e_phi_re.zeros(1, 2, 1), ant.e_phi_re.slice(0) = T;
    T = {0.0, -1.0}, ant.e_phi_im.zeros(1, 2, 1), ant.e_phi_im.slice(0) = T;
    ant.azimuth_grid = {0.0, pi};
    ant.elevation_grid = {0.0};

    arma::mat C = {0.0, 1.0, 2.0, 3.0};
    arma::mat azimuth = C * pi / 4.0;
    arma::mat elevation = {0.0, 0.0, 0.0, 0.0};
    arma::Col<unsigned> i_element = {1};

    T = {0.0, 0.0, -pi / 8.0};
    arma::Cube<double> orientation(3, 1, 1);
    orientation.slice(0) = T.t();
    arma::mat element_pos_i(3, 1);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);

    CHECK(arma::approx_equal(azimuth_loc, azimuth + pi / 8.0, "absdiff", 1e-14));

    T = arma::cos(C * pi / 8.0 + pi / 16.0);
    CHECK(arma::approx_equal(V_re, T, "absdiff", 1e-14));

    T = arma::sin(C * pi / 8.0 + pi / 16.0);
    CHECK(arma::approx_equal(V_im, T, "absdiff", 1e-14));

    T = {1.875, 1.625, 1.375, 1.125};
    T = -arma::cos(C * pi / 8.0 + pi / 16.0) % T;
    CHECK(arma::approx_equal(H_re, T, "absdiff", 1e-14));

    T = {1.875, 1.625, 1.375, 1.125};
    T = -arma::sin(C * pi / 8.0 + pi / 16.0) % T;
    CHECK(arma::approx_equal(H_im, T, "absdiff", 1e-14));
}

TEST_CASE("Arrayant interpolation - Spheric interpolation in el-direction with y-rotation")
{
    double pi = arma::datum::pi;
    quadriga_lib::arrayant<double> ant;

    arma::mat T;
    T = {1.0, 0.0}, ant.e_theta_re.zeros(2, 1, 1), ant.e_theta_re.slice(0) = T.t();
    T = {0.0, 1.0}, ant.e_theta_im.zeros(2, 1, 1), ant.e_theta_im.slice(0) = T.t();
    T = {-2.0, 0.0}, ant.e_phi_re.zeros(2, 1, 1), ant.e_phi_re.slice(0) = T.t();
    T = {0.0, -1.0}, ant.e_phi_im.zeros(2, 1, 1), ant.e_phi_im.slice(0) = T.t();
    ant.azimuth_grid = {0.0};
    ant.elevation_grid = {0.0, pi / 2.0};

    arma::mat C = {0.0, 1.0, 2.0, 3.0};
    arma::mat azimuth = {0.0, 0.0, 0.0, 0.0};
    arma::mat elevation = C * pi / 8.0;
    arma::Col<unsigned> i_element = {1};
    arma::mat element_pos_i(3, 1);

    T = {0.0, -pi / 16.0, 0.0};
    arma::Cube<double> orientation(3, 1, 1);
    orientation.slice(0) = T.t();

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);

    CHECK(arma::approx_equal(azimuth_loc, azimuth, "absdiff", 1e-14));
    CHECK(arma::approx_equal(elevation_loc, elevation + pi / 16.0, "absdiff", 1e-14));

    T = arma::cos(C * pi / 8.0 + pi / 16.0);
    CHECK(arma::approx_equal(V_re, T, "absdiff", 1e-14));

    T = arma::sin(C * pi / 8.0 + pi / 16.0);
    CHECK(arma::approx_equal(V_im, T, "absdiff", 1e-14));

    T = {1.875, 1.625, 1.375, 1.125};
    T = -arma::cos(C * pi / 8.0 + pi / 16.0) % T;
    CHECK(arma::approx_equal(H_re, T, "absdiff", 1e-14));

    T = {1.875, 1.625, 1.375, 1.125};
    T = -arma::sin(C * pi / 8.0 + pi / 16.0) % T;
    CHECK(arma::approx_equal(H_im, T, "absdiff", 1e-14));
}

TEST_CASE("Arrayant interpolation - Polarization rotation using x-rotation")
{
    double pi = arma::datum::pi;
    quadriga_lib::arrayant<double> ant;

    arma::mat T;
    T = {1.0, 0.0}, ant.e_theta_re.zeros(1, 2, 1), ant.e_theta_re.slice(0) = T;
    T = {1.0, 0.0}, ant.e_theta_im.zeros(1, 2, 1), ant.e_theta_im.slice(0) = T;
    ant.e_phi_re.zeros(1, 2, 1), ant.e_phi_im.zeros(1, 2, 1);
    ant.azimuth_grid = {0.0, pi};
    ant.elevation_grid = {0.0};

    arma::mat azimuth(1, 1);
    arma::mat elevation(1, 1);
    arma::Col<unsigned> i_element = {1, 1};

    T = {{pi / 4.0, -pi / 4.0}, {0.0, 0.0}, {0.0, 0.0}};
    arma::Cube<double> orientation(3, 2, 1);
    orientation.slice(0) = T;
    arma::mat element_pos_i(3, 2);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);

    double rs2 = 1.0 / std::sqrt(2.0);
    T = {rs2, rs2};
    CHECK(arma::approx_equal(V_re, T.t(), "absdiff", 1e-14));
    CHECK(arma::approx_equal(V_im, T.t(), "absdiff", 1e-14));

    T = {rs2, -rs2};
    CHECK(arma::approx_equal(H_re, T.t(), "absdiff", 1e-14));
    CHECK(arma::approx_equal(H_im, T.t(), "absdiff", 1e-14));
}

TEST_CASE("Arrayant interpolation - Test projected distance")
{
    double pi = arma::datum::pi;
    double rs2 = 1.0 / std::sqrt(2.0);
    quadriga_lib::arrayant<double> ant;

    ant.e_theta_re.ones(1, 1, 1), ant.e_theta_im.zeros(1, 1, 1);
    ant.e_phi_re.zeros(1, 1, 1), ant.e_phi_im.zeros(1, 1, 1);
    ant.azimuth_grid = {0.0}, ant.elevation_grid = {0.0};

    arma::mat azimuth(1, 1);
    arma::mat elevation(1, 1);
    arma::Col<unsigned> i_element = {1, 1, 1};
    arma::Cube<double> orientation(3, 1, 1);
    arma::mat element_pos_i(3, 3, arma::fill::eye);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);

    arma::mat T(3, 1, arma::fill::ones);
    CHECK(arma::approx_equal(V_re, T, "absdiff", 1e-14));

    T.zeros();
    CHECK(arma::approx_equal(V_im, T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(H_re, T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(H_im, T, "absdiff", 1e-14));

    T = {-1.0, 0.0, 0.0};
    CHECK(arma::approx_equal(dist, T.t(), "absdiff", 1e-14));

    azimuth.at(0) = 3.0 * pi / 4.0;
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);
    T = {rs2, -rs2, 0.0};
    CHECK(arma::approx_equal(dist, T.t(), "absdiff", 1e-14));

    azimuth.at(0) = 0.0;
    elevation.at(0) = -pi / 4.0;
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);
    T = {-rs2, 0.0, rs2};
    CHECK(arma::approx_equal(dist, T.t(), "absdiff", 1e-14));

    azimuth = {-pi, -pi / 2.0, 0.0};
    elevation = {0.0, 0.0, -pi / 2.0};
    element_pos_i = -element_pos_i; // -eye(3)
    ant.interpolate(azimuth, elevation, i_element, orientation, element_pos_i,
                    &V_re, &V_im, &H_re, &H_im, &dist, &azimuth_loc, &elevation_loc, &gamma);
    CHECK(arma::approx_equal(dist, element_pos_i, "absdiff", 1e-14));
}