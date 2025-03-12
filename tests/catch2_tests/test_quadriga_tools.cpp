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
    arma::Mat<float> center;
    arma::Col<float> length;
    arma::Mat<float> vert;
    arma::Mat<float> direction;

    // Number of subdivisions cannot be zero
    REQUIRE_THROWS_AS(quadriga_lib::icosphere<float>(0, 1.0, &center), std::invalid_argument);

    size_t n = quadriga_lib::icosphere<float>(1, 1.0, &center);
    CHECK(n == 20);

    n = quadriga_lib::icosphere<float>(1, 1.0, &center, &length);
    arma::fvec sumRow = arma::sqrt(sum(arma::abs(pow(center, 2)), 1));
    CHECK(arma::approx_equal(length, sumRow, "absdiff", 2e-7));

    n = quadriga_lib::icosphere<float>(2, 1.0, &center, &length);
    CHECK(n == 80);
    sumRow = arma::sqrt(sum(arma::abs(pow(center, 2)), 1));
    CHECK(arma::approx_equal(length, sumRow, "absdiff", 2e-7));

    n = quadriga_lib::icosphere<float>(2, 1.0, &center, &length, &vert);
    arma::fvec squaredSum = arma::sum(arma::square(arma::abs(center + vert.cols(0, 2))), 1);
    auto test = arma::fvec(80, arma::fill::ones);
    CHECK(arma::approx_equal(squaredSum, test, "absdiff", 2e-7));

    squaredSum = sum(arma::square(arma::abs(center + vert.cols(3, 5))), 1);
    CHECK(arma::approx_equal(squaredSum, test, "absdiff", 2e-7));

    squaredSum = sum(arma::square(arma::abs(center + vert.cols(6, 8))), 1);
    CHECK(arma::approx_equal(squaredSum, test, "absdiff", 2e-7));

    n = quadriga_lib::icosphere<float>(2, 2.0, &center, &length, &vert, &direction);
    CHECK(direction.n_rows == 80);
    CHECK(direction.n_cols == 6);

    test = 4.0 * test;
    squaredSum = sum(arma::square(arma::abs(center + vert.cols(6, 8))), 1);
    CHECK(arma::approx_equal(squaredSum, test, "absdiff", 5e-7));

    bool allElementsWithinRange = arma::all(arma::vectorise(arma::abs(direction)) <= arma::datum::pi);
    CHECK(allElementsWithinRange);

    quadriga_lib::icosphere<float>(2, 2.0, &center, &length, &vert, &direction, 1);
    test.ones(80);

    arma::fvec x = direction.col(0) % direction.col(0) +
                   direction.col(1) % direction.col(1) +
                   direction.col(2) % direction.col(2);

    CHECK(arma::approx_equal(x, test, "absdiff", 2e-7));

    x = direction.col(3) % direction.col(3) +
        direction.col(4) % direction.col(4) +
        direction.col(5) % direction.col(5);

    CHECK(arma::approx_equal(x, test, "absdiff", 2e-7));

    x = direction.col(6) % direction.col(6) +
        direction.col(7) % direction.col(7) +
        direction.col(8) % direction.col(8);

    CHECK(arma::approx_equal(x, test, "absdiff", 2e-7));
}

TEST_CASE("Quadriga tools - Mesh Segmentation")
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

TEST_CASE("Quadriga tools - Point Cloud Segmentation")
{
    // Generate set of points
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(0.0f, 0.1f, 0.3f);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0f;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0f;
    points.submat(8, 1, 15, 1) += 50.0f;
    points.col(2) += 1.0f;

    // Calculate bounding box
    auto aabb = quadriga_lib::point_cloud_aabb(&points);

    arma::fmat T;
    T = {{0.0f, 40.3f, -50.0f, 50.0f, 1.0f, 1.0f}};
    CHECK(arma::approx_equal(aabb, T, "absdiff", 1e-14));

    arma::fmat pointsA, pointsB;
    arma::s32_vec split_ind;

    // Split along longest axis
    int res = quadriga_lib::point_cloud_split(&points, &pointsA, &pointsB, 0, &split_ind);

    CHECK(res == 2);
    CHECK(arma::approx_equal(pointsA, points.submat(0, 0, 7, 2), "absdiff", 1e-14));
    CHECK(arma::approx_equal(pointsB, points.submat(8, 0, 15, 2), "absdiff", 1e-14));
    CHECK(split_ind.n_elem == 16);
    CHECK(split_ind.at(0) == 1);
    CHECK(split_ind.at(8) == 2);

    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index, forward_index, reverse_index;

    points.col(1) *= 0.1f;

    size_t n_sub = quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 5,
                                                          &forward_index, &reverse_index);

    arma::u32_vec I;
    I = {0, 5, 10, 15};
    CHECK(arma::all(sub_cloud_index == I));

    CHECK(arma::approx_equal(pointsR.submat(0, 0, 3, 2), points.submat(0, 0, 3, 2), "absdiff", 1e-14));
    T = {{0.15f, -5.0f, 1.0f}};
    CHECK(arma::approx_equal(pointsR.submat(4, 0, 4, 2), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(pointsR.submat(5, 0, 8, 2), points.submat(8, 0, 11, 2), "absdiff", 1e-14));
    T = {{0.15f, 5.0f, 1.0f}};
    CHECK(arma::approx_equal(pointsR.submat(9, 0, 9, 2), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(pointsR.submat(10, 0, 13, 2), points.submat(4, 0, 7, 2), "absdiff", 1e-14));
    T = {{40.15f, -5.0f, 1.0f}};
    CHECK(arma::approx_equal(pointsR.submat(14, 0, 14, 2), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(pointsR.submat(15, 0, 18, 2), points.submat(12, 0, 15, 2), "absdiff", 1e-14));
    T = {{40.15f, 5.0f, 1.0f}};
    CHECK(arma::approx_equal(pointsR.submat(19, 0, 19, 2), T, "absdiff", 1e-14));

    I = {1, 2, 3, 4, 0, 9, 10, 11, 12, 0, 5, 6, 7, 8, 0, 13, 14, 15, 16, 0};
    CHECK(arma::all(forward_index == I));

    I = {0, 1, 2, 3, 10, 11, 12, 13, 5, 6, 7, 8, 15, 16, 17, 18};
    CHECK(arma::all(reverse_index == I));

    // Test AABB of sub-clouds
    aabb = quadriga_lib::point_cloud_aabb(&pointsR, &sub_cloud_index);

    T = {{0.0f, 0.3f, -5.0f, -5.0f, 1.0f, 1.0f},
         {0.0f, 0.3f, 5.0f, 5.0f, 1.0f, 1.0f},
         {40.0f, 40.3f, -5.0f, -5.0f, 1.0f, 1.0f},
         {40.0f, 40.3f, 5.0f, 5.0f, 1.0f, 1.0f}};

    CHECK(arma::approx_equal(aabb, T, "absdiff", 1e-14));
}

TEST_CASE("Quadriga tools - Subdivide Rays")
{
    arma::Mat<float> orig;
    arma::Mat<float> trivec;
    arma::Mat<float> tridir;

    quadriga_lib::icosphere<float>(2, 2.0, &orig, nullptr, &trivec, &tridir);

    arma::Mat<float> origN;
    arma::Mat<float> trivecN;
    arma::Mat<float> tridirN;

    // No index given
    size_t n_rayN = quadriga_lib::subdivide_rays<float>(&orig, &trivec, &tridir, nullptr, &origN, &trivecN, &tridirN);
    CHECK(n_rayN == 4 * orig.n_rows);

    // Empty index
    arma::u32_vec index;
    n_rayN = quadriga_lib::subdivide_rays<float>(&orig, &trivec, &tridir, nullptr, &origN, &trivecN, &tridirN, nullptr, &index);
    CHECK(n_rayN == 4 * orig.n_rows);

    // Select two beams
    index = {2, 1};
    n_rayN = quadriga_lib::subdivide_rays<float>(&orig, &trivec, &tridir, nullptr, &origN, &trivecN, &tridirN, nullptr, &index);
    CHECK(n_rayN == 8);

    // Check for directions that are equal (beam 1)
    CHECK(tridir(2, 0) == tridirN(0, 0));
    CHECK(tridir(2, 1) == tridirN(0, 1));
    CHECK(tridir(2, 2) == tridirN(3, 2));
    CHECK(tridir(2, 3) == tridirN(3, 3));
    CHECK(tridir(2, 4) == tridirN(2, 4));
    CHECK(tridir(2, 5) == tridirN(2, 5));

    // Check for directions that are equal (beam 2)
    CHECK(tridir(1, 0) == tridirN(4, 0));
    CHECK(tridir(1, 1) == tridirN(4, 1));
    CHECK(tridir(1, 2) == tridirN(7, 2));
    CHECK(tridir(1, 3) == tridirN(7, 3));
    CHECK(tridir(1, 4) == tridirN(6, 4));
    CHECK(tridir(1, 5) == tridirN(6, 5));

    // Check corner points
    CHECK(arma::approx_equal(orig.row(2) + trivec.submat(2, 0, 2, 2), origN.row(0) + trivecN.submat(0, 0, 0, 2), "absdiff", 1e-6));
    CHECK(arma::approx_equal(orig.row(2) + trivec.submat(2, 3, 2, 5), origN.row(3) + trivecN.submat(3, 3, 3, 5), "absdiff", 1e-6));
    CHECK(arma::approx_equal(orig.row(2) + trivec.submat(2, 6, 2, 8), origN.row(2) + trivecN.submat(2, 6, 2, 8), "absdiff", 1e-6));

    CHECK(arma::approx_equal(orig.row(1) + trivec.submat(1, 0, 1, 2), origN.row(4) + trivecN.submat(4, 0, 4, 2), "absdiff", 1e-6));
    CHECK(arma::approx_equal(orig.row(1) + trivec.submat(1, 3, 1, 5), origN.row(7) + trivecN.submat(7, 3, 7, 5), "absdiff", 1e-6));
    CHECK(arma::approx_equal(orig.row(1) + trivec.submat(1, 6, 1, 8), origN.row(6) + trivecN.submat(6, 6, 6, 8), "absdiff", 1e-6));

    arma::fmat dest = 2.0f * orig;
    arma::fmat destN;

    index = {0};
    n_rayN = quadriga_lib::subdivide_rays<float>(&orig, &trivec, &tridir, &dest, &origN, &trivecN, &tridirN, &destN, &index, 0.1);

    CHECK(n_rayN == 4);
}

TEST_CASE("Quadriga tools - Coord 2 Path")
{
    double Tx = 0.0, Ty = 0.0, Tz = 0.0;
    double Rx = 10.0, Ry = 10.0, Rz = 10.0;

    arma::u32_vec no_interact = {0};
    arma::mat interact_coord(3, 0);

    arma::vec path_length;
    arma::mat fbs_pos, lbs_pos, path_angles;
    std::vector<arma::mat> path_coord;

    // Forward LOS path
    quadriga_lib::coord2path(Tx, Ty, Tz, Rx, Ry, Rz, &no_interact, &interact_coord,
                             &path_length, &fbs_pos, &lbs_pos, &path_angles, &path_coord);

    arma::mat M = {5.0, 5.0, 5.0};
    CHECK(arma::approx_equal(fbs_pos, M.t(), "absdiff", 1e-14));
    CHECK(arma::approx_equal(lbs_pos, M.t(), "absdiff", 1e-14));

    M = {10.0 * std::sqrt(3.0)};
    CHECK(arma::approx_equal(path_length, M.col(0), "absdiff", 1e-14));

    double pi = 3.141592653589793;
    double el = std::asin(1.0 / std::sqrt(3.0));
    M = {0.25 * pi, el, -0.75 * pi, -el};
    CHECK(arma::approx_equal(path_angles, M, "absdiff", 1e-14));

    M = {{0.0, 10.0}, {0.0, 10.0}, {0.0, 10.0}};
    CHECK(arma::approx_equal(path_coord[0], M, "absdiff", 1e-14));

    // Reverse LOS path
    quadriga_lib::coord2path(Tx, Ty, Tz, Rx, Ry, Rz, &no_interact, &interact_coord,
                             &path_length, &fbs_pos, &lbs_pos, &path_angles, &path_coord, true);

    M = {5.0, 5.0, 5.0};
    CHECK(arma::approx_equal(fbs_pos, M.t(), "absdiff", 1e-14));
    CHECK(arma::approx_equal(lbs_pos, M.t(), "absdiff", 1e-14));

    M = {10.0 * std::sqrt(3.0)};
    CHECK(arma::approx_equal(path_length, M.col(0), "absdiff", 1e-14));

    M = {-0.75 * pi, -el, 0.25 * pi, el};
    CHECK(arma::approx_equal(path_angles, M, "absdiff", 1e-14));

    // Forward path
    Rx = 10.0, Ry = 0.0, Rz = 0.0;
    no_interact = {2};
    interact_coord = {{2.0, 8.0}, {2.0, 2.0}, {2.0, 2.0}};

    M = {{10.0, 0.0}, {10.0, 0.0}, {10.0, 0.0}};
    CHECK(arma::approx_equal(path_coord[0], M, "absdiff", 1e-14));

    quadriga_lib::coord2path(Tx, Ty, Tz, Rx, Ry, Rz, &no_interact, &interact_coord,
                             &path_length, &fbs_pos, &lbs_pos, &path_angles, &path_coord);

    M = {2.0, 2.0, 2.0};
    CHECK(arma::approx_equal(fbs_pos, M.t(), "absdiff", 1e-14));

    M = {8.0, 2.0, 2.0};
    CHECK(arma::approx_equal(lbs_pos, M.t(), "absdiff", 1e-14));

    M = {6.0 + 4.0 * std::sqrt(3.0)};
    CHECK(arma::approx_equal(path_length, M.col(0), "absdiff", 1e-14));

    M = {0.25 * pi, el, 0.75 * pi, el};
    CHECK(arma::approx_equal(path_angles, M, "absdiff", 1e-14));

    M = {{0.0, 2.0, 8.0, 10.0}, {0.0, 2.0, 2.0, 0.0}, {0.0, 2.0, 2.0, 0.0}};
    CHECK(arma::approx_equal(path_coord[0], M, "absdiff", 1e-14));

    // Reverse path

    quadriga_lib::coord2path(Tx, Ty, Tz, Rx, Ry, Rz, &no_interact, &interact_coord,
                             &path_length, &fbs_pos, &lbs_pos, &path_angles, &path_coord, true);

    M = {8.0, 2.0, 2.0};
    CHECK(arma::approx_equal(fbs_pos, M.t(), "absdiff", 1e-14));

    M = {2.0, 2.0, 2.0};
    CHECK(arma::approx_equal(lbs_pos, M.t(), "absdiff", 1e-14));

    M = {6.0 + 4.0 * std::sqrt(3.0)};
    CHECK(arma::approx_equal(path_length, M.col(0), "absdiff", 1e-14));

    M = {0.75 * pi, el, 0.25 * pi, el};
    CHECK(arma::approx_equal(path_angles, M, "absdiff", 1e-14));

    M = {{10.0, 8.0, 2.0, 0.0}, {0.0, 2.0, 2.0, 0.0}, {0.0, 2.0, 2.0, 0.0}};
    CHECK(arma::approx_equal(path_coord[0], M, "absdiff", 1e-14));
}