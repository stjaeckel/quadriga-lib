// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <iostream>
#include <string>
#include <cmath>
#include <limits>

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
    arma::uvec i_element = {0};
    arma::Cube<double> orientation(3, 1, 1);
    arma::mat element_pos_i(3, 1);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im,
                    i_element, &orientation, &element_pos_i,
                    &dist, &azimuth_loc, &elevation_loc, &gamma);

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
    arma::uvec i_element = {0};
    arma::Cube<double> orientation(3, 1, 1);
    arma::mat element_pos_i(3, 1);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im,
                    i_element, &orientation, &element_pos_i,
                    &dist, &azimuth_loc, &elevation_loc, &gamma);

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

    arma::mat V_re, V_im, H_re, H_im;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, {}, &orientation);

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

    arma::mat V_re, V_im, H_re, H_im;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im);

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

    T = {0.0, 0.0, -pi / 8.0};
    arma::Cube<double> orientation(3, 1, 1);
    orientation.slice(0) = T.t();
    arma::mat element_pos_i(3, 1);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, {},
                    &orientation, &element_pos_i, &dist, &azimuth_loc);

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

    arma::mat element_pos_i(3, 1);

    T = {0.0, -pi / 16.0, 0.0};
    arma::Cube<double> orientation(3, 1, 1);
    orientation.slice(0) = T.t();

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, {},
                    &orientation, &element_pos_i, &dist, &azimuth_loc, &elevation_loc, &gamma);

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
    arma::uvec i_element = {0, 0};

    T = {{pi / 4.0, -pi / 4.0}, {0.0, 0.0}, {0.0, 0.0}};
    arma::Cube<double> orientation(3, 2, 1);
    orientation.slice(0) = T;
    arma::mat element_pos_i(3, 2);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, i_element,
                    &orientation, &element_pos_i);

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
    arma::uvec i_element = {0, 0, 0};
    arma::Cube<double> orientation(3, 1, 1);
    arma::mat element_pos_i(3, 3, arma::fill::eye);

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, i_element,
                    &orientation, &element_pos_i, &dist, &azimuth_loc, &elevation_loc, &gamma);

    arma::mat T(3, 1, arma::fill::ones);
    CHECK(arma::approx_equal(V_re, T, "absdiff", 1e-14));

    T.zeros();
    CHECK(arma::approx_equal(V_im, T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(H_re, T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(H_im, T, "absdiff", 1e-14));

    T = {-1.0, 0.0, 0.0};
    CHECK(arma::approx_equal(dist, T.t(), "absdiff", 1e-14));

    azimuth.at(0) = 3.0 * pi / 4.0;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, i_element, &orientation, &element_pos_i,
                    &dist, &azimuth_loc, &elevation_loc, &gamma);
    T = {rs2, -rs2, 0.0};
    CHECK(arma::approx_equal(dist, T.t(), "absdiff", 1e-14));

    azimuth.at(0) = 0.0;
    elevation.at(0) = -pi / 4.0;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, i_element, &orientation, &element_pos_i,
                    &dist, &azimuth_loc, &elevation_loc, &gamma);
    T = {-rs2, 0.0, rs2};
    CHECK(arma::approx_equal(dist, T.t(), "absdiff", 1e-14));

    azimuth = {-pi, -pi / 2.0, 0.0};
    elevation = {0.0, 0.0, -pi / 2.0};
    element_pos_i = -element_pos_i; // -eye(3)
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, i_element, &orientation, &element_pos_i,
                    &dist, &azimuth_loc, &elevation_loc, &gamma);
    CHECK(arma::approx_equal(dist, element_pos_i, "absdiff", 1e-14));
}

TEST_CASE("Arrayant interpolation - Example")
{
    double pi = arma::datum::pi;

    auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
    arma::mat azimuth = {0.0, 0.5 * pi, -0.5 * pi, pi}; // Azimuth angles: East, North, South, West
    arma::mat elevation(1, azimuth.n_elem);             // Initialize to 0

    arma::mat V_re, V_im, H_re, H_im;
    ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im);
}
// B-007: NaN or infinite angles must give NaN output without reading outside the pattern (the bracket search used
// to wrap an index to SIZE_MAX; the out-of-bounds read is visible only in the ASan suite). Finite angles in the same
// call are not affected.
template <typename dtype>
static void arrayant_interpolate_nonfinite_check()
{
    const dtype nan = std::numeric_limits<dtype>::quiet_NaN(), inf = std::numeric_limits<dtype>::infinity();
    auto ant = quadriga_lib::generate_arrayant_xpol<dtype>(5.0); // 37 x 73 grid, 2 elements
    ant.element_pos = {{0.1, -0.1}, {0.2, 0.0}, {0.0, 0.3}};

    // Shared angles [1, n_ang]: columns 1..6 are non-finite, columns 0 and 7 are finite
    arma::Mat<dtype> az = {{0.3, nan, inf, -inf, 0.2, 0.2, 0.2, -2.0}};
    arma::Mat<dtype> el = {{0.1, 0.0, 0.0, 0.0, nan, inf, -inf, 0.5}};
    arma::Mat<dtype> Vr, Vi, Hr, Hi, dist, az_loc, el_loc, gam;
    ant.interpolate(&az, &el, &Vr, &Vi, &Hr, &Hi, {}, nullptr, nullptr, &dist, &az_loc, &el_loc, &gam);

    arma::Mat<dtype> az_f = {{0.3, -2.0}}, el_f = {{0.1, 0.5}};
    arma::Mat<dtype> Vr_f, Vi_f, Hr_f, Hi_f;
    ant.interpolate(&az_f, &el_f, &Vr_f, &Vi_f, &Hr_f, &Hi_f);

    for (arma::uword o = 0; o < 2; ++o)
    {
        for (arma::uword a = 1; a < 7; ++a)
        {
            CHECK(std::isnan(Vr(o, a)));
            CHECK(std::isnan(Vi(o, a)));
            CHECK(std::isnan(Hr(o, a)));
            CHECK(std::isnan(Hi(o, a)));
            CHECK(std::isnan(dist(o, a)));
        }
        CHECK(Vr(o, 0) == Vr_f(o, 0));
        CHECK(Hr(o, 0) == Hr_f(o, 0));
        CHECK(Vr(o, 7) == Vr_f(o, 1));
        CHECK(Hi(o, 7) == Hi_f(o, 1));
    }

    // Per-element angles [n_out, n_ang]: non-finite angles for some elements only
    arma::Mat<dtype> az2 = {{nan, 0.3, 0.1, inf}, {0.3, -inf, 0.2, 0.0}};
    arma::Mat<dtype> el2 = {{0.1, 0.1, nan, 0.0}, {0.1, 0.1, 0.0, -inf}};
    const int bad[2][4] = {{1, 0, 1, 1}, {0, 1, 0, 1}};
    ant.interpolate(&az2, &el2, &Vr, &Vi, &Hr, &Hi);
    for (arma::uword o = 0; o < 2; ++o)
        for (arma::uword a = 0; a < 4; ++a)
        {
            CHECK(std::isnan(Vr(o, a)) == (bad[o][a] == 1));
            CHECK(std::isnan(Hi(o, a)) == (bad[o][a] == 1));
        }
}

TEST_CASE("Arrayant interpolation - NaN and infinite angles give NaN (B-007)")
{
    arrayant_interpolate_nonfinite_check<float>();
    arrayant_interpolate_nonfinite_check<double>();
}
