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
#include "quadriga_tools.hpp"

#include <iostream>
#include <fstream>
#include <string>

TEST_CASE("Test OBJ Overlap - Identical objects")
{
    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  0 Top NorthEast
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

    arma::mat tmp = cube; // Second cube
    arma::mat mesh = arma::join_cols(cube, tmp);

    // Object indices (1,4)
    arma::uvec cube_ind(12);
    cube_ind.ones();
    arma::uvec tmp_ind = cube_ind + 3;
    arma::uvec obj_ind = arma::join_cols(cube_ind, tmp_ind);

    std::vector<std::string> reason;
    auto overlap = quadriga_lib::obj_overlap_test(&mesh, &obj_ind, &reason);

    arma::uvec test = {1, 4};
    CHECK(arma::all(overlap == test));

    CHECK(reason[0] == "Identical with OBJ-ID 4");
    CHECK(reason[1] == "Identical with OBJ-ID 1");
}

TEST_CASE("Test OBJ Overlap - Touching cubes")
{
    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  0 Top NorthEast
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

    arma::mat tmp = cube; // Second cube
    tmp.col(0) = tmp.col(0) + 2.0;
    tmp.col(3) = tmp.col(3) + 2.0;
    tmp.col(6) = tmp.col(6) + 2.0;
    arma::mat mesh = arma::join_cols(cube, tmp);

    // Object indices (1,4)
    arma::uvec cube_ind(12);
    cube_ind.ones();
    arma::uvec tmp_ind = cube_ind + 3;
    arma::uvec obj_ind = arma::join_cols(cube_ind, tmp_ind);

    auto overlap = quadriga_lib::obj_overlap_test(&mesh, &obj_ind);
    CHECK(overlap.n_elem == 0ULL);
}

TEST_CASE("Test OBJ Overlap - 3D overlap")
{
    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  0 Top NorthEast
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

    arma::mat tmp = cube; // Second cube
    tmp.col(0) = tmp.col(0) + 1.0;
    tmp.col(3) = tmp.col(3) + 1.0;
    tmp.col(6) = tmp.col(6) + 1.0;
    tmp.col(1) = tmp.col(1) + 1.0;
    tmp.col(4) = tmp.col(4) + 1.0;
    tmp.col(7) = tmp.col(7) + 1.0;
    tmp.col(2) = tmp.col(2) + 1.0;
    tmp.col(5) = tmp.col(5) + 1.0;
    tmp.col(8) = tmp.col(8) + 1.0;
    arma::mat mesh = arma::join_cols(cube, tmp);

    // Object indices (1,4)
    arma::uvec cube_ind(12);
    cube_ind.ones();
    arma::uvec tmp_ind = cube_ind + 3;
    arma::uvec obj_ind = arma::join_cols(cube_ind, tmp_ind);

    std::vector<std::string> reason;
    auto overlap = quadriga_lib::obj_overlap_test(&mesh, &obj_ind, &reason);

    arma::uvec test = {1, 4};
    CHECK(arma::all(overlap == test));

    CHECK(reason[0].substr(0, 21) == "3D Intersect: OBJ-IDs");
    CHECK(reason[1].substr(0, 21) == "3D Intersect: OBJ-IDs");
}

TEST_CASE("Test OBJ Overlap - Overlapping Edges")
{
    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  0 Top NorthEast
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

    arma::mat tmp = cube * 1.0; // Second cube
    tmp.col(0) = tmp.col(0) + 1.0;
    tmp.col(3) = tmp.col(3) + 1.0;
    tmp.col(6) = tmp.col(6) + 1.0;
    arma::mat mesh = arma::join_cols(cube, tmp);

    // Object indices (1,4)
    arma::uvec cube_ind(12);
    cube_ind.ones();
    arma::uvec tmp_ind = cube_ind + 3;
    arma::uvec obj_ind = arma::join_cols(cube_ind, tmp_ind);

    std::vector<std::string> reason;
    auto overlap = quadriga_lib::obj_overlap_test(&mesh, &obj_ind, &reason);

    arma::uvec test = {1, 4};
    CHECK(arma::all(overlap == test));

    CHECK(reason[0].substr(0, 21) == "2D Intersect: OBJ-IDs");
    CHECK(reason[1].substr(0, 21) == "2D Intersect: OBJ-IDs");

    CHECK(reason[0].find("co-linear edges (7)") < reason[0].size());
}

TEST_CASE("Test OBJ Overlap - Overlapping Faces")
{
    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  0 Top NorthEast
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

    arma::mat tmp = cube * 1.0; // Second cube
    tmp.col(0) = tmp.col(0) + 1.0;
    tmp.col(3) = tmp.col(3) + 1.0;
    tmp.col(6) = tmp.col(6) + 1.0;
    tmp.col(1) = tmp.col(1) + 0.5;
    tmp.col(4) = tmp.col(4) + 0.5;
    tmp.col(7) = tmp.col(7) + 0.5;
    arma::mat mesh = arma::join_cols(cube, tmp);

    // Object indices (1,4)
    arma::uvec cube_ind(12);
    cube_ind.ones();
    arma::uvec tmp_ind = cube_ind + 3;
    arma::uvec obj_ind = arma::join_cols(cube_ind, tmp_ind);

    std::vector<std::string> reason;
    auto overlap = quadriga_lib::obj_overlap_test(&mesh, &obj_ind, &reason);

    arma::uvec test = {1, 4};
    CHECK(arma::all(overlap == test));

    CHECK(reason[0].substr(0, 21) == "2D Intersect: OBJ-IDs");
}

// TEST_CASE("Test OBJ Overlap - File")
// {

//     std::string fn_obj = "/tmp/qrt_overlap_test.obj";
//     arma::fmat mesh;
//     arma::uvec obj_ind;
//     std::vector<std::string> obj_names;
//     quadriga_lib::obj_file_read<float>(fn_obj, &mesh, nullptr, nullptr, nullptr, &obj_ind, nullptr, &obj_names);

//     std::vector<std::string> reason;
//     arma::uvec intersecting = quadriga_lib::obj_overlap_test(&mesh, &obj_ind, &reason);

//     intersecting.print();

// }