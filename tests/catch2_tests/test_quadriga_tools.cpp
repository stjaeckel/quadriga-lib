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
#include <string>

TEST_CASE("Quadriga tools - 2D Interpolation")
{
    arma::cube I(2, 3, 2);
    I.slice(0) = {{1.0, 0.0, -2.0}, {1.0, 0.0, -2.0}};
    I.slice(1) = {{2.0, 2.0, 2.0}, {4.0, 4.0, 4.0}};

    arma::vec x = {0.0, 1.5, 2.0}, xo;
    arma::vec y = {10.0, 20.0}, yo;
    arma::cube O, T;

    // Interpolation at the same sample points should create identical output
    quadriga_tools::interp(&I, &x, &y, &x, &y, &O);
    CHECK(arma::approx_equal(I, O, "absdiff", 1e-13));

    // Regular interpolation, y-values out of bound
    xo = {0.75, 1.875};
    yo = {0.0, 100.0};
    quadriga_tools::interp(&I, &x, &y, &xo, &yo, &O);
    T.set_size(2, 2, 2);
    T.slice(0) = {{0.5, -1.5}, {0.5, -1.5}};
    T.slice(1) = {{2.0, 2.0}, {4.0, 4.0}};
    CHECK(arma::approx_equal(T, O, "absdiff", 1e-13));

    // Edge-Case : co-located x-variables
    I.set_size(1, 4, 1);
    I.slice(0) = {{0.0, 1.0, 2.0, 3.0}};
    x = {0.0, 1.0, 1.0, 2.0}, y = {0.0}, xo = {0.9, 1.0, 1.1}, yo = {-1.0, 1.0};
    quadriga_tools::interp(&I, &x, &y, &xo, &yo, &O);
    T.set_size(2, 3, 1);
    T.slice(0) = {{0.9, 2.0, 2.1}, {0.9, 2.0, 2.1}};
    CHECK(arma::approx_equal(T, O, "absdiff", 1e-13));

    // Reverse order
    x = {3.0, 2.0, 1.0, 0.0}, y = {0.0}, xo = {2.5, 2.1, 2.0, 1.9}, yo = {0.0};
    quadriga_tools::interp(&I, &x, &y, &xo, &yo, &O);
    T.set_size(1, 4, 1);
    T.slice(0) = {{0.5, 0.9, 1.0, 1.1}};
    CHECK(arma::approx_equal(T, O, "absdiff", 1e-13));
}

TEST_CASE("Quadriga tools - 1D Interpolation")
{
    arma::fmat I = {{1.0f, 0.0f, -2.0f}, {2.0f, 3.0f, 4.0f}};
    arma::fvec x = {0.0f, 1.5f, 2.0f}, xo;
    arma::fmat O, T;
    I = I.t();

    // Interpolation at the same sample points should create identical output
    quadriga_tools::interp(&I, &x, &x, &O);
    CHECK(arma::approx_equal(I, O, "absdiff", 1e-6));

    // Regular interpolation, y-values out of bound
    xo = {0.75f, 1.875f};
    quadriga_tools::interp(&I, &x, &xo, &O);
    T = {{0.5f, 2.5f}, {-1.5f, 3.75f}};
    CHECK(arma::approx_equal(T, O, "absdiff", 1e-6));
}