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
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_prop, 1.0e9, 1, &gain, &coord);

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