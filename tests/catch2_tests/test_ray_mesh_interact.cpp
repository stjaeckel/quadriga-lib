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
#include <complex>

TEST_CASE("Ray-Mesh Interact - Air to Air (x-z plane)")
{
    // Default cube
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

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

    // Ray start 10 m west of cube center, pointing east
    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}};
    arma::vec orig_length = {2.7}; // Assuming 2.7 m  previous length

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir(1, 6);
    tridir.at(4) = 1.0 * deg2rad;
    tridir.at(1) = 1.0 * deg2rad;

    // Calculate interaction location
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-14));

    T = {{1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-14));

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;
    double a;

    // Test case 1 : Cube of air
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 0.0}}; // Air
    mtl_prop = repmat(mtl_prop, 12, 1);

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, &orig_length,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN,
                                    &orig_lengthN, &fbs_angleN, &thicknessN, &edge_lengthN, &normal_vecN);

    T = {{-1.001, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-14));

    T = {{-12.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-14));

    U = {0.0}; // Air does not reflect anything
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-14));

    T = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-14));

    a = std::tan(1.0 * deg2rad) * 9.0 + 0.2;
    T = {{0.001, -0.1, a, 0.001, -0.1, -0.2, 0.001, a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-14));

    T = {{180.0, 1.0, 180.0, 0.0, 179.0, 0.0}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-14));

    U = {2.7 + 9.0 + 0.001};
    CHECK(arma::approx_equal(orig_lengthN, U, "absdiff", 1e-14));

    U = {90.0 * deg2rad};
    CHECK(arma::approx_equal(fbs_angleN, U, "absdiff", 1e-14));

    U = {2.0};
    CHECK(arma::approx_equal(thicknessN, U, "absdiff", 1e-14));

    U = {std::sqrt(a * a + (a + 0.1) * (a + 0.1))};
    CHECK(arma::approx_equal(edge_lengthN, U, "absdiff", 1e-14));

    T = {{-1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(normal_vecN, T, "absdiff", 1e-14));

    // Test transmission on air
    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, &orig_length,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    T = {{-0.999, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-14));

    T = dest;
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-14));

    U = {1.0}; // Air does transmit everything
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-14));

    T = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-14));

    a = std::tan(1.0 * deg2rad) * 9.0 + 0.2;
    T = {{-0.001, -0.1, a, -0.001, -0.1, -0.2, -0.001, a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-14));

    T = {{0.0, 1.0, 0.0, 0.0, 1.0, 0.0}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-14));
}

TEST_CASE("Ray-Mesh Interact - Air to Dielectric Medium (x-z plane)")
{
    // Default cube
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

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

    // 45° incidence on dielectric medium
    arma::mat orig = {{-1.5, 0.0, 0.0}};
    arma::mat dest = {{0.0, 0.0, 1.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-14));

    T = {{-0.5, 0.0, 1.0}}; // Ceiling of cube
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-14));

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{0.0, 45.0, 0.0, 45.0, 0.0, 45.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;
    double a;

    arma::mat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    a = 0.001 * std::cos(45 * deg2rad);
    T = {{-1.0 - a, 0.0, 0.5 + a}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-14));

    T = {{-2.0, 0.0, 1.5}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-14));

    // See Jaeckel, S.; Raschkowski, L.; Wu, S.; Thiele, L. & Keusgen, W.
    // An Explicit Ground Reflection Model for mm-Wave Channels Proc. IEEE WCNC Workshops '17, 2017
    double eps = mtl_prop.at(0, 0);
    double cos_th = std::cos(45.0 * deg2rad), sin_th = std::sin(45.0 * deg2rad);
    double Z = std::sqrt(eps - cos_th * cos_th);
    double R_par = (eps * sin_th - Z) / (eps * sin_th + Z); // = R_tm
    double R_per = (sin_th - Z) / (sin_th + Z);             // = R_te

    U = {0.5 * R_par * R_par + 0.5 * R_per * R_per}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-14));

    T = {{-R_par, 0.0, 0.0, 0.0, 0.0, 0.0, -R_per, 0.0}}; // 180° phase shift for reflection
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-14));

    T = {{a, -0.1, 0.2 - a, a, -0.1, -0.2 - a, a, 0.2, -a}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-14));

    T = {{180.0, 45.0, 180.0, 45.0, 180.0, 45.0}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-14));

    // Test refraction into medium
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    // Refraction angle calculated from Snell's law
    double th2 = std::asin(std::sin(45.0 * deg2rad) / std::sqrt(eps));

    double x = std::cos(th2) * 0.001 - 1;
    double z = std::sin(th2) * 0.001 + 0.5;
    T = {{x, 0.0, z}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-14));

    x = std::cos(th2) * std::sqrt(2) - 1;
    z = std::sin(th2) * std::sqrt(2) + 0.5;
    T = {{x, 0.0, z}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-14));

    // Transmission coefficients, ITU-R P.2040-1, eq. (32a) and (32b)
    double T_te = 2.0 * cos_th / (cos_th + std::sqrt(eps) * std::cos(th2));
    double T_tm = 2.0 * cos_th / (std::sqrt(eps) * cos_th + std::cos(th2));

    U = {0.5 * T_tm * T_tm + 0.5 * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-14));

    T = {{T_tm, 0.0, 0.0, 0.0, 0.0, 0.0, T_te, 0.0}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-14));

    a = std::cos(th2) * 0.001;
    double b = std::sin(th2) * 0.001;
    T = {{-a, -0.1, 0.2 - b, -a, -0.1, -0.2 - b, -a, 0.2, -b}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-14));

    T = {{0.0, th2, 0.0, th2, 0.0, th2}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-14));

    // Test transmission into medium
    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    a = 0.001 * std::cos(45 * deg2rad); // Same as reflection
    T = {{-1.0 + a, 0.0, 0.5 + a}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-14));

    T = dest;
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-14));

    // Gain
    std::complex<double> eta1(1.0, 0.0);
    std::complex<double> eta2(1.5, 0.0);
    double G = calc_transition_gain(0, 45.0, 0.0, 0.0, eta1, eta2); // Reflection
    double H = calc_transition_gain(2, 45.0, 0.0, 0.0, eta1, eta2); // Refraction

    U = {1.0 - G};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-14));

    T = {{T_tm, 0.0, 0.0, 0.0, 0.0, 0.0, T_te, 0.0}}; // Same as refraction
    T = T * std::sqrt((1 - G) / H);
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-14));

    T = {{-a, -0.1, 0.2 - a, -a, -0.1, -0.2 - a, -a, 0.2, -a}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-14));

    a = 45.0 * deg2rad;
    T = {{0.0, a, 0.0, a, 0.0, a}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-14));
}

TEST_CASE("Ray-Mesh Interact - Dielectric Medium to Air (x-y plane, float)")
{
    // Default cube
    arma::fmat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
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

    // Conversions
    float deg2rad = arma::datum::pi / 180.0;
    float rad2deg = 180.0 / arma::datum::pi;

    // 45° incidence, x-y plane
    arma::fmat orig = {{0.5, 0.1, 0.0}}; // start inside the cube
    arma::fmat dest = {{2.0, 1.6, 0.0}};

    arma::fmat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::fmat T = {{1.0, 0.6, 0.0}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-7));

    T = {{2.0, 1.6, 0.0}}; // End of path
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-7));

    arma::fmat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::fmat tridir = {{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::fmat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::fvec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;

    arma::fmat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    float cos_th = std::cos(45.0 * deg2rad);
    float a = 0.001 * cos_th;
    T = {{1.0f - a, 0.6f + a, 0.0f}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-7));

    T = {{0.0f, 1.6f, 0.0f}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-7));

    // Gain
    std::complex<double> eta1(1.5, 0.0);
    std::complex<double> eta2(1.0, 0.0);
    double G = calc_transition_gain(0, 45.0, 0.0, 0.0, eta1, eta2);
    U = {(float)G}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    // Reflection coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    float cos_th2 = std::sqrt(1.0 - 1.5 / 1.0 * cos_th * cos_th);
    float sqrt_eps = std::sqrt(mtl_prop.at(0, 0));
    float R_te = (sqrt_eps * cos_th - cos_th2) / (sqrt_eps * cos_th + cos_th2);
    float R_tm = (cos_th - sqrt_eps * cos_th2) / (cos_th + sqrt_eps * cos_th2);
    T = {{R_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, R_tm, 0.0f}}; // x-y plane swaps base vectors
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-7));

    T = {{a, -0.1f - a, 0.2, a, -0.1f - a, -0.2f, a, 0.2f - a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-7));

    T = {{135.0f, 0.0f, 135.0f, 0.0f, 135.0f, 0.0f}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-07));

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    float th2 = std::acos(cos_th2);
    float x = std::cos(th2) * 0.001 + 1.0;
    float y = std::sin(th2) * 0.001 + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-7));

    x = std::cos(th2) * std::sqrt(2) + 1.0;
    y = std::sin(th2) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-7));

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    float T_te = (2.0 * sqrt_eps * cos_th) / (sqrt_eps * cos_th + cos_th2);
    float T_tm = (2.0 * sqrt_eps * cos_th) / (cos_th + sqrt_eps * cos_th2);

    U = {0.5f * T_tm * T_tm + 0.5f * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    G = calc_transition_gain(2, 45.0, 0.0, 0.0, eta1, eta2);
    U = {(float)G}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    // x-y plane swaps base vectors
    T = {{T_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, T_tm, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-7));

    a = std::cos(th2) * 0.001;
    float b = std::sin(th2) * 0.001;
    T = {{-a, -0.1f - b, 0.2f, -a, -0.1f - b, -0.2f, -a, 0.2f - b, 0.0f}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-7));

    T = {{th2, 0.0f, th2, 0.0f, th2, 0.0f}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-7));

    // Transmission from inside to outside without refraction
    quadriga_lib::ray_mesh_interact(1, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    U = {1.0};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    U = {0.5f * xprmatN(6) * xprmatN(6) + 0.5f * xprmatN(0) * xprmatN(0)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    // Total reflection occurs at eta = 2, theta = 45°
    mtl_prop = {{2.0, 0.0, 0.0, 0.0, 0.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    quadriga_lib::ray_mesh_interact(0, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);
    U = {1.0f};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    T = {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-7));

    quadriga_lib::ray_mesh_interact(2, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);
    U = {0.0f};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    T = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-7));
}

TEST_CASE("Ray-Mesh Interact - Medium to Medium (x-y plane, double)")
{

    // Default cube
    arma::mat msh = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
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

    auto tmp = msh; // Second cube
    tmp.col(0) = tmp.col(0) + 2.0;
    tmp.col(3) = tmp.col(3) + 2.0;
    tmp.col(6) = tmp.col(6) + 2.0;
    msh = arma::join_cols(msh, tmp);

    arma::mat mtl_prop = {{1.2, std::log10(1.5 / 1.2), 0.0, 0.0, 0.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    tmp = {{1.33, 0.0, 0.0, 0.0, 3.0}};
    tmp = repmat(tmp, 12, 1);
    mtl_prop = arma::join_cols(mtl_prop, tmp);

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

    // 45° incidence, x-y plane
    arma::mat orig = {{0.5, 0.1, 0.0}}; // start inside the cube
    arma::mat dest = {{2.0, 1.6, 0.0}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &msh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{1.0, 0.6, 0.0}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-14));

    T = {{1.0, 0.6, 0.0}}; // End of path
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-14));

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &msh, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    double cos_th = std::cos(45.0 * deg2rad);
    double a = 0.001 * cos_th;
    T = {{1.0 - a, 0.6 + a, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-7));

    T = {{0.0, 1.6, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-7));

    // Cosine of refraction angle,  ITU-R P.2040-1, eq. (33) with sin(theta) = cos(theta) @ 45°
    double cos_th2 = std::sqrt(1.0 - 1.5 / 1.33 * cos_th * cos_th);

    // Reflection coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    double sqrt_eps1 = std::sqrt(1.5);
    double sqrt_eps2 = std::sqrt(1.33);
    double R_te = (sqrt_eps1 * cos_th - sqrt_eps2 * cos_th2) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    double R_tm = (sqrt_eps2 * cos_th - sqrt_eps1 * cos_th2) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    U = {0.5f * R_tm * R_tm + 0.5f * R_te * R_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    // x-y plane swaps base vectors
    T = {{R_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, R_tm, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-7));

    T = {{a, -0.1 - a, 0.2, a, -0.1 - a, -0.2, a, 0.2 - a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-7));

    T = {{135.0, 0.0, 135.0, 0.0, 135.0, 0.0}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-07));

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &msh, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    double transition_gain = std::pow(10.0, -0.1 * 3.0);

    double th2 = std::acos(cos_th2);
    double x = std::cos(th2) * 0.001 + 1.0;
    double y = std::sin(th2) * 0.001 + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-7));

    x = std::cos(th2) * std::sqrt(2) + 1.0;
    y = std::sin(th2) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-7));

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    double T_te = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    double T_tm = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    T_te *= std::sqrt(transition_gain);
    T_tm *= std::sqrt(transition_gain);

    U = {0.5f * T_tm * T_tm + 0.5f * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-7));

    // x-y plane swaps base vectors
    T = {{T_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, T_tm, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-7));

    a = std::cos(th2) * 0.001;
    double b = std::sin(th2) * 0.001;
    T = {{-a, -0.1f - b, 0.2f, -a, -0.1f - b, -0.2f, -a, 0.2f - b, 0.0f}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-7));

    T = {{th2, 0.0f, th2, 0.0f, th2, 0.0f}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-7));
}

TEST_CASE("Ray-Mesh Interact - Conductive to Dielectric (x-y plane, double)")
{

    // Default cube
    arma::mat msh = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
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

    auto tmp = msh; // Second cube
    tmp.col(0) = tmp.col(0) + 2.0;
    tmp.col(3) = tmp.col(3) + 2.0;
    tmp.col(6) = tmp.col(6) + 2.0;
    msh = arma::join_cols(msh, tmp);

    arma::mat mtl_prop = {{1.2, std::log10(1.5 / 1.2), 0.01, std::log10(0.02 / 0.01), 0.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    tmp = {{1.33, 0.0, 0.0, 0.0, 0.0}};
    tmp = repmat(tmp, 12, 1);
    mtl_prop = arma::join_cols(mtl_prop, tmp);

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

    // 45° incidence, x-y plane
    arma::mat orig = {{0.5, 0.1, 0.0}}; // start inside the cube
    arma::mat dest = {{2.0, 1.6, 0.0}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &msh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{1.0, 0.6, 0.0}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-14));

    T = {{1.0, 0.6, 0.0}}; // End of path
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-14));

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &msh, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    double cos_th = std::cos(45.0 * deg2rad);

    std::complex<double> eta1(1.5, -1.798 * 0.02); // @ 10 GHz
    std::complex<double> eta2(1.33, 0.0);

    // Cosine of refraction angle,  ITU-R P.2040-1, eq. (33) with sin(theta) = cos(theta) @ 45°
    std::complex<double> cos_th2 = std::sqrt(1.0 - eta1 / eta2 * cos_th * cos_th);

    // Reflection coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    std::complex<double> sqrt_eps1 = std::sqrt(eta1);
    std::complex<double> sqrt_eps2 = std::sqrt(eta2);
    std::complex<double> R_te = (sqrt_eps1 * cos_th - sqrt_eps2 * cos_th2) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    std::complex<double> R_tm = (sqrt_eps2 * cos_th - sqrt_eps1 * cos_th2) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    // In-medium attenuation
    double thickness = std::sqrt(2.0 * 0.5 * 0.5) + 0.001;
    double tan_delta = std::imag(eta1) / std::real(eta1); // Loss tangent, Rec. ITU-R P.2040-1, eq. (13)
    double cos_delta = std::cos(std::atan(tan_delta));
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta1)));
    double A = 8.686 * thickness / Delta;   // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    R_te *= std::sqrt(gain);
    R_tm *= std::sqrt(gain);
    U = {0.5 * std::abs(R_tm) * std::abs(R_tm) + 0.5 * std::abs(R_te) * std::abs(R_te)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-14));

    // x-y plane swaps base vectors
    T = {{std::real(R_te), std::imag(R_te), 0.0, 0.0, 0.0, 0.0, std::real(R_tm), std::imag(R_tm)}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-14));

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &msh, &mtl_prop, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    // Wave direction is not defined precisely due to the complex angles
    std::complex<double> th2 = std::acos(cos_th2);
    double x = std::real(cos_th2) * 0.001 + 1.0;
    double y = std::real(std::sin(th2)) * 0.001 + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-7));

    x = std::real(cos_th2) * std::sqrt(2.0) + 1.0;
    y = std::real(std::sin(th2)) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-3));

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    std::complex<double> T_te = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    std::complex<double> T_tm = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    T_te *= std::sqrt(gain);
    T_tm *= std::sqrt(gain);
    U = {0.5 * std::abs(T_tm) * std::abs(T_tm) + 0.5 * std::abs(T_te) * std::abs(T_te)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-3));

   // x-y plane swaps base vectors
    T = {{std::real(T_te), std::imag(T_te), 0.0, 0.0, 0.0, 0.0, std::real(T_tm), std::imag(T_tm)}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-3));
}