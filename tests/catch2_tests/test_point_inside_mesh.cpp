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

TEST_CASE("Test Point inside Mesh")
{
    arma::mat mesh = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  0 Top NorthEast
                      {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0},  //  1 South Lower
                      {-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0}, //  2 West Lower
                      {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},  //  3 Bottom NorthWest
                      {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0},     //  4 East Lower
                      {-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0},    //  5 North Lower
                      {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0},    //  6 Top SouthWest
                      {1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0},  //  7 South Upper
                      {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0},   //  8 West Upper
                      {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0},  //  9 Bottom SouthEast
                      {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0},     // 10 East Upper
                      {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}};     // 11 North Upper

    arma::mat points = {{0.0, 0.0, 0.5}, {-1.1, 0.0, 0.0}};
    arma::u32_vec obj_ind(12, arma::fill::value(2)), res;

    res = quadriga_lib::point_inside_mesh(&points, &mesh, &obj_ind);
    CHECK(res.n_elem == 2);
    CHECK(res.at(0) == 2);
    CHECK(res.at(1) == 0);

    res = quadriga_lib::point_inside_mesh(&points, &mesh);
    CHECK(res.n_elem == 2);
    CHECK(res.at(0) == 1);
    CHECK(res.at(1) == 0);

    res = quadriga_lib::point_inside_mesh<double>(&points, &mesh, nullptr, 0.12);
    CHECK(res.n_elem == 2);
    CHECK(res.at(0) == 1);
    CHECK(res.at(1) == 1);

    res = quadriga_lib::point_inside_mesh<double>(&points, &mesh, &obj_ind, 0.09);
    CHECK(res.n_elem == 2);
    CHECK(res.at(0) == 2);
    CHECK(res.at(1) == 0);

    res = quadriga_lib::point_inside_mesh<double>(&points, &mesh, &obj_ind, 2.0);
    CHECK(res.n_elem == 2);
    CHECK(res.at(0) == 2);
    CHECK(res.at(1) == 2);
}
