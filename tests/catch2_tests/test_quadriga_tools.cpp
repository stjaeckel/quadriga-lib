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
    quadriga_lib::interp(&I, &x, &y, &x, &y, &O);
    CHECK(arma::approx_equal(I, O, "absdiff", 1e-13));

    // Regular interpolation, y-values out of bound
    xo = {0.75, 1.875};
    yo = {0.0, 100.0};
    quadriga_lib::interp(&I, &x, &y, &xo, &yo, &O);
    T.set_size(2, 2, 2);
    T.slice(0) = {{0.5, -1.5}, {0.5, -1.5}};
    T.slice(1) = {{2.0, 2.0}, {4.0, 4.0}};
    CHECK(arma::approx_equal(T, O, "absdiff", 1e-13));

    // Edge-Case : co-located x-variables
    I.set_size(1, 4, 1);
    I.slice(0) = {{0.0, 1.0, 2.0, 3.0}};
    x = {0.0, 1.0, 1.0, 2.0}, y = {0.0}, xo = {0.9, 1.0, 1.1}, yo = {-1.0, 1.0};
    quadriga_lib::interp(&I, &x, &y, &xo, &yo, &O);
    T.set_size(2, 3, 1);
    T.slice(0) = {{0.9, 2.0, 2.1}, {0.9, 2.0, 2.1}};
    CHECK(arma::approx_equal(T, O, "absdiff", 1e-13));

    // Reverse order
    x = {3.0, 2.0, 1.0, 0.0}, y = {0.0}, xo = {2.5, 2.1, 2.0, 1.9}, yo = {0.0};
    quadriga_lib::interp(&I, &x, &y, &xo, &yo, &O);
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
    quadriga_lib::interp(&I, &x, &x, &O);
    CHECK(arma::approx_equal(I, O, "absdiff", 1e-6));

    // Regular interpolation, y-values out of bound
    xo = {0.75f, 1.875f};
    quadriga_lib::interp(&I, &x, &xo, &O);
    T = {{0.5f, 2.5f}, {-1.5f, 3.75f}};
    CHECK(arma::approx_equal(T, O, "absdiff", 1e-6));
}

TEST_CASE("Quadriga tools - Icosphere")
{
    arma::Mat<double> center;
    arma::Col<double> length;
    arma::Mat<double> vert;
    arma::Mat<double> direction;

    // Number of subdivisions cannot be zero
    REQUIRE_THROWS_AS(quadriga_lib::icosphere<double>(0, 1.0, &center), std::invalid_argument);

    auto n = quadriga_lib::icosphere<double>(1, 1.0, &center);
    CHECK(n == 20ULL);

    n = quadriga_lib::icosphere<double>(1, 1.0, &center, &length);
    arma::vec sumRow = sqrt(sum(arma::abs(pow(center, 2)), 1));
    CHECK(arma::approx_equal(length, sumRow, "absdiff", 1e-14));

    n = quadriga_lib::icosphere<double>(2, 1.0, &center, &length);
    CHECK(n == 80ULL);
    sumRow = sqrt(sum(arma::abs(pow(center, 2)), 1));
    CHECK(arma::approx_equal(length, sumRow, "absdiff", 1e-14));

    n = quadriga_lib::icosphere<double>(2, 1.0, &center, &length, &vert);
    arma::vec squaredSum = sum(arma::square(arma::abs(center + vert.cols(0, 2))), 1);
    auto test = arma::vec(80, arma::fill::ones);
    CHECK(arma::approx_equal(squaredSum, test, "absdiff", 1e-14));

    squaredSum = sum(arma::square(arma::abs(center + vert.cols(3, 5))), 1);
    CHECK(arma::approx_equal(squaredSum, test, "absdiff", 1e-14));

    squaredSum = sum(arma::square(arma::abs(center + vert.cols(6, 8))), 1);
    CHECK(arma::approx_equal(squaredSum, test, "absdiff", 1e-14));

    n = quadriga_lib::icosphere<double>(2, 2.0, &center, &length, &vert, &direction);
    CHECK(direction.n_rows == 80ULL);
    CHECK(direction.n_cols == 6ULL);

    test = 4.0 * test;
    squaredSum = sum(arma::square(arma::abs(center + vert.cols(6, 8))), 1);
    CHECK(arma::approx_equal(squaredSum, test, "absdiff", 1e-14));

    bool allElementsWithinRange = arma::all(arma::vectorise(arma::abs(direction)) <= arma::datum::pi);
    CHECK(allElementsWithinRange);
}

TEST_CASE("Quadriga tools - Mesh reorganization")
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

    arma::fmat mtl_prop = {{1.5, 0.1, 0.2, 0.3, 0.4}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    mtl_prop.col(0) = arma::regspace<arma::fvec>(1.1f, 0.1f, 2.2f);

    // Subdivide the mesh into smaller chunks
    arma::fmat cube_sub = cube, mtl_prop_sub = mtl_prop;
    quadriga_lib::subdivide_triangles(3, &cube, &cube_sub, &mtl_prop, &mtl_prop_sub);

    // Case 1 - Mesh size is already below threshold, test padding
    arma::fmat cube_re, mtl_prop_re;
    arma::u32_vec cube_index, mesh_index;
    auto n_sub = quadriga_lib::triangle_mesh_segmentation(&cube_sub, &cube_re, &cube_index, 1024, 8,
                                                &mtl_prop_sub, &mtl_prop_re, &mesh_index);

    CHECK(n_sub == 1);
    CHECK(cube_re.n_rows == 112);     // Multiple of 8
    CHECK(mtl_prop_re.n_rows == 112); // Multiple of 8
    CHECK(cube_index.n_elem == 1);
    CHECK(cube_index.at(0) == 0);

    arma::fmat T(4, 9);
    CHECK(arma::approx_equal(cube_re.submat(0, 0, 107, 8), cube_sub, "absdiff", 1e-14));
    CHECK(arma::approx_equal(cube_re.submat(108, 0, 111, 8), T, "absdiff", 1e-14));

    T.zeros(4, 5);
    T.col(0).ones();
    CHECK(arma::approx_equal(mtl_prop_re.submat(0, 0, 107, 4), mtl_prop_sub, "absdiff", 1e-14));
    CHECK(arma::approx_equal(mtl_prop_re.submat(108, 0, 111, 4), T, "absdiff", 1e-14));

    auto U = arma::regspace<arma::u32_vec>(1, 108);
    CHECK(arma::all(mesh_index.subvec(0, 107) == U));

    U.zeros(4);
    CHECK(arma::all(mesh_index.subvec(108, 111) == U));

    // Case 2 - Subdivide, no padding
    n_sub = quadriga_lib::triangle_mesh_segmentation(&cube_sub, &cube_re, &cube_index, 64, 1,
                                           &mtl_prop_sub, &mtl_prop_re, &mesh_index);

    CHECK(n_sub == 3);
    CHECK(cube_re.n_rows == 108);
    CHECK(mtl_prop_re.n_rows == 108);
    CHECK(cube_index.n_elem == n_sub);
    CHECK(cube_index.at(0) == 0);

    CHECK(!mesh_index.is_sorted());
    U = arma::regspace<arma::u32_vec>(1, 108);
    CHECK(arma::all(arma::sort(mesh_index) == U));

    auto I = arma::conv_to<arma::uvec>::from(mesh_index - 1);
    CHECK(arma::approx_equal(cube_re, cube_sub.rows(I), "absdiff", 1e-14));
    CHECK(arma::approx_equal(mtl_prop_re, mtl_prop_sub.rows(I), "absdiff", 1e-14));
}