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
#include "quadriga_tools.hpp"
#include <iostream>

// Function to calculate the gain
#ifndef calc_transition_gain_HELPER
#define calc_transition_gain_HELPER
static double calc_transition_gain(int interaction_type,       // (0) Reflection, (1) Transmission, (2) Refraction
                                   double incidence_angle_deg, // Angle between face normal and ray (as in ITU P.2040-1) (degree)
                                   double dist1,               // Medium 1 travel distance (meters)
                                   double dist2,               // Medium 2 travel distance (meters) OR distance after reflection
                                   std::complex<double> eta1,  // relative permittivity of medium 1
                                   std::complex<double> eta2)  // relative permittivity of medium 2
{
    double deg2rad = arma::datum::pi / 180.0;

    // Calculate gain from ITU-R P.2040:
    double cos_th = std::cos(incidence_angle_deg * deg2rad); // Incidence on boundary
    double sin_th = std::sqrt(1.0 - cos_th * cos_th);        // Trigonometric identity
    std::complex<double> cos_th2 = std::sqrt(1.0 - eta1 / eta2 * sin_th * sin_th);

    // Medium 1 loss
    double tan_delta = std::imag(eta1) / std::real(eta1); // Loss tangent, Rec. ITU-R P.2040-1, eq. (13)
    double cos_delta = std::cos(std::atan(tan_delta));
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta1)));
    double A = 8.686 * dist1 / Delta;                // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double medium_1_gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    // Medium 2 loss
    if (interaction_type != 0) // Use eta1 for reflection
    {
        tan_delta = std::imag(eta2) / std::real(eta2); // Loss tangent, Rec. ITU-R P.2040-1, eq. (13)
        cos_delta = std::cos(std::atan(tan_delta));
        Delta = 2.0 * cos_delta / (1.0 - cos_delta);
        Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta2)));
    }
    A = 8.686 * dist2 / Delta;                       // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double medium_2_gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    eta1 = std::sqrt(eta1);
    eta2 = std::sqrt(eta2);

    // Reflection coefficients
    std::complex<double> R_te = (eta1 * cos_th - eta2 * cos_th2) / (eta1 * cos_th + eta2 * cos_th2);
    std::complex<double> R_tm = (eta2 * cos_th - eta1 * cos_th2) / (eta2 * cos_th + eta1 * cos_th2);

    // Transmission coefficients
    std::complex<double> T_te = (2.0 * eta1 * cos_th) / (eta1 * cos_th + eta2 * cos_th2);
    std::complex<double> T_tm = (2.0 * eta1 * cos_th) / (eta2 * cos_th + eta1 * cos_th2);

    double reflection_gain = 0.5 * (std::norm(R_te) + std::norm(R_tm));
    double refraction_gain = 0.5 * (std::norm(T_te) + std::norm(T_tm));

    double total_gain = 0.0;
    if (interaction_type == 0) // Refection
        total_gain = medium_1_gain * reflection_gain * medium_2_gain;
    else if (interaction_type == 1) // Transmission
        total_gain = medium_1_gain * (1.0 - reflection_gain) * medium_2_gain;
    else if (interaction_type == 2) // Refraction
        total_gain = medium_1_gain * refraction_gain * medium_2_gain;

    return total_gain;
}
#endif

TEST_CASE("Calc Diffraction Gain")
{
    double deg2rad = arma::datum::pi / 180.0;

    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
                      {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0},  //  2 South Lower
                      {-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0}, //  3 West Lower
                      {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},  //  4 Bottom NorthWest
                      {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0},     //  5 East Lower
                      {-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0},    //  6 North Lower
                      {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0},    //  7 Top SouthWest
                      {1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0},  //  8 South Upper
                      {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0},   //  9 West Upper
                      {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0},  // 10 Bottom SouthEast
                      {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0},     // 11 East Upper
                      {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}};     // 12 North Upper

    arma::mat mtl_prop, orig, dest, tm;
    arma::vec gain, tv;
    arma::cube coord, tc;

    mtl_prop = {{1.5, 0.0, 0.001, 0.0, 0.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    // Entire path outside
    orig = {{-10.0, 0.0, 0.5}};
    dest = {{-5.0, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 1.0e9, 1, &gain, &coord, 0);

    tv = {1.0};
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));

    tc.set_size(3, 2, 1);
    tc.slice(0) = {{-8.75, -6.25}, {0.0, 0.0}, {0.5, 0.5}};
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 1.0e9, 2, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    tc.set_size(3, 3, 1);
    tc.slice(0) = {{-10.0 + 5.0 * 0.1464, -7.5, -10.0 + 5.0 * 0.8536}, {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 1.0e9, 3, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    tc.set_size(3, 4, 1);
    tc.slice(0) = {{-10.0 + 5.0 * 0.0955, -10.0 + 5.0 * 0.3455, -10.0 + 5.0 * 0.6545, -10.0 + 5.0 * 0.9045}, {0.0, 0.0, 0.0, 0.0}, {0.5, 0.5, 0.5, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 1.0e9, 4, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    // Single path outside to inside
    std::complex<double> eta1(1.0, 0.0);                                // Air
    std::complex<double> eta2(mtl_prop(0, 0), -1.798 * mtl_prop(0, 2)); // @ 10 GHz

    double total_gain = calc_transition_gain(1, 0.0, 1.0, 1.5, eta1, eta2);

    tv = {total_gain};
    orig = {{-10.0, 0.0, 0.5}};
    dest = {{0.5, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 10.0e9, 0, &gain, &coord);

    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-10));

    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 10.0e9, 5, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-10));

    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 10.0e9, 6, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-10));

    // 2 segments, (1) outside to inside, (2) inside
    orig = {{-1.5, 0.0, 0.5}};
    dest = {{0.5, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 10.0e9, 5, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-10));
}