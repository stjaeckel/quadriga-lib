// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"

#include <iostream>
#include <fstream>
#include <string>

TEST_CASE("Test OBJ Overlap - Identical objects")
{
    arma::mat cube = quadriga_lib::cube<double>();

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
    arma::mat cube = quadriga_lib::cube<double>();

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
    arma::mat cube = quadriga_lib::cube<double>();

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
    arma::mat cube = quadriga_lib::cube<double>();

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
    arma::mat cube = quadriga_lib::cube<double>();

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
