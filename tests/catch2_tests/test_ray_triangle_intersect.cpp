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

TEST_CASE("Ray-Triangle Intersect - Simple Mode")
{
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

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}};

    arma::mat fbs, sbs;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;

    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind);

    arma::mat T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));

    T = {{1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    CHECK(no_interact.at(0) == 2);
    CHECK(fbs_ind.at(0) == 9);
    CHECK(sbs_ind.at(0) == 11);
}

TEST_CASE("Ray-Triangle Intersect - Sub-mesh Mode")
{

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

    // Subdivide the mesh into smaller chunks
    arma::mat cube_sub = cube;
    quadriga_lib::subdivide_triangles(3, &cube, &cube_sub);

    // Build sub-meshes
    arma::mat cube_re;
    arma::u32_vec cube_index;

    auto n_sub = quadriga_lib::triangle_mesh_segmentation(&cube_sub, &cube_re, &cube_index, 64, 8);
    CHECK(n_sub == 3);

    // Single ray
    arma::mat orig = {{1.0, -10.0, 0.5}};
    arma::mat dest = {{1.0, 10.0, 0.5}};

    arma::mat fbs, sbs;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;

    // Run intersect algorithm without sub-mesh
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube_re, &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind);

    // Check results
    arma::mat T = {{1.0, -1.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));
    T = {{1.0, 1.0, 0.5}};
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    // Store FBS and SBS indices
    unsigned fbs_0 = fbs_ind.at(0);
    unsigned sbs_0 = sbs_ind.at(0);
    
    // Run intersect algorithm with sub-mesh
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube_re, &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind, &cube_index);

    // Check results
    T = {{1.0, -1.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));
    T = {{1.0, 1.0, 0.5}};
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    // Store FBS and SBS indices
    unsigned fbs_1 = fbs_ind.at(0);
    unsigned sbs_1 = sbs_ind.at(0);

    CHECK(fbs_1 == fbs_0);
    CHECK(sbs_1 == sbs_0);
}
