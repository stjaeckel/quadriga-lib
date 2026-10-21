// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

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

TEST_CASE("Test OBJ Overlap - Vertices on the planes of the other object's faces")
{
    // Object 1: cube [-1,1]^3. Object 2: tetrahedron with its apex in the cube centre and its base vertices at x = 3,
    // each on the extension of a cube face plane (y = 1, y = -1, z = 1). The apex edges leave the cube through the
    // face x = 1; their end points lie on the planes of other faces. Control: base vertices 0.05 off the planes.
    for (double off : {0.0, 0.05})
    {
        INFO("offset of the base vertices: " << off);
        arma::mat A = quadriga_lib::cube<double>();
        arma::rowvec P0 = {0, 0, 0}, P1 = {3, 1 + off, 0.2}, P2 = {3, -1 - off, -1.5}, P3 = {3, 2.5, 1 + off};
        arma::mat B(4, 9);
        B.row(0) = arma::join_rows(P0, P1, P2);
        B.row(1) = arma::join_rows(P0, P2, P3);
        B.row(2) = arma::join_rows(P0, P3, P1);
        B.row(3) = arma::join_rows(P1, P3, P2);
        arma::mat mesh = arma::join_cols(A, B);
        arma::uvec obj = arma::join_cols(arma::uvec(12, arma::fill::value(1)), arma::uvec(4, arma::fill::value(2)));
        std::vector<std::string> reason;
        arma::uvec res = quadriga_lib::obj_overlap_test(&mesh, &obj, &reason);
        REQUIRE(res.n_elem == 2);
        CHECK(res(0) == 1);
        CHECK(res(1) == 2);
        CHECK(reason[0].substr(0, 21) == "3D Intersect: OBJ-IDs");
    }
}

TEST_CASE("Test OBJ Overlap - Touching boxes with offset and larger faces")
{
    // Boxes that touch face to face: the second box is shifted by half its width along the shared face, or its
    // touching face is larger than the first box's face. The diagonals of the touching faces cross the edges of the
    // first box, which the ray-triangle test may report as hits on its side faces. Nothing overlaps.
    arma::mat A = quadriga_lib::cube<double>();
    arma::uvec obj = arma::join_cols(arma::uvec(12, arma::fill::value(1)), arma::uvec(12, arma::fill::value(2)));
    for (int c = 0; c < 4; ++c)
    {
        INFO("case " << c);
        arma::mat B;
        if (c == 0)
            B = quadriga_lib::cube<double>({1.0}, {}, {2.0, 1.0, 0.0}); // shifted by half in y
        else if (c == 1)
            B = quadriga_lib::cube<double>({1.0}, {}, {2.0, 1.0, 1.0}); // shifted by half in y and z
        else if (c == 2)
            B = quadriga_lib::cube<double>({1.0, 2.0, 2.0}, {}, {2.0, 0.0, 0.0}); // larger touching face
        else
            B = quadriga_lib::cube<double>({1.0, 3.0, 1.5}, {}, {2.0, 0.5, -0.25}); // larger, off-centre
        arma::mat mesh = arma::join_cols(A, B);
        std::vector<std::string> reason;
        arma::uvec res = quadriga_lib::obj_overlap_test(&mesh, &obj, &reason);
        CHECK(res.n_elem == 0);
        if (res.n_elem != 0)
            UNSCOPED_INFO(reason[0]);
    }
}

TEST_CASE("Test OBJ Overlap - Empty mesh")
{
    arma::mat mesh(0, 9);
    arma::uvec obj;
    std::vector<std::string> reason = {"old"};
    arma::uvec res;
    REQUIRE_NOTHROW(res = quadriga_lib::obj_overlap_test(&mesh, &obj, &reason));
    CHECK(res.n_elem == 0);
    CHECK(reason.empty());
    REQUIRE_NOTHROW(res = quadriga_lib::obj_overlap_test(&mesh, &obj));
    CHECK(res.n_elem == 0);
}

TEST_CASE("Test OBJ Overlap - Object IDs are returned as given")
{
    // 0-based IDs as obj_file_read returns them
    arma::mat A = quadriga_lib::cube<double>(), B = quadriga_lib::cube<double>({1, 0.5, 0.5}, {}, {1, 0, 0});
    arma::mat mesh = arma::join_cols(A, B);
    for (arma::uword id0 : {0, 7})
    {
        arma::uvec obj = arma::join_cols(arma::uvec(12, arma::fill::value(id0)), arma::uvec(12, arma::fill::value(id0 + 1)));
        arma::uvec res = quadriga_lib::obj_overlap_test(&mesh, &obj);
        REQUIRE(res.n_elem == 2);
        CHECK(res(0) == id0);
        CHECK(res(1) == id0 + 1);
    }
}

TEST_CASE("Test OBJ Overlap - Reasons do not depend on the call or the thread count")
{
    // Six mutually overlapping cubes; the reasons must be identical over repeated calls with 1 and 4 threads
    arma::mat mesh;
    arma::uvec obj;
    for (int i = 0; i < 6; ++i)
    {
        arma::mat c = quadriga_lib::cube<double>({1.0}, {}, {0.5 * i, 0.3 * i, 0.1 * i});
        mesh = arma::join_cols(mesh, c);
        obj = arma::join_cols(obj, arma::uvec(12, arma::fill::value(i + 1)));
    }
    const int n_threads_before = omp_get_max_threads();
    std::vector<std::string> ref;
    arma::uvec ref_ids;
    for (int nt : {1, 1, 4, 4, 1, 4})
    {
        omp_set_num_threads(nt);
        std::vector<std::string> why;
        arma::uvec res = quadriga_lib::obj_overlap_test(&mesh, &obj, &why);
        if (ref.empty())
        {
            ref = why, ref_ids = res;
            REQUIRE(res.n_elem == 6);
            continue;
        }
        INFO("threads " << nt);
        CHECK(arma::all(res == ref_ids));
        CHECK(why == ref);
    }
    omp_set_num_threads(n_threads_before);

    // The reason of object 1 is the finding of its first pair (1,2)
    CHECK(ref[0].find("OBJ-IDs (1,2)") != std::string::npos);
}
