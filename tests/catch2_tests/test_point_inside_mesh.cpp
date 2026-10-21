// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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
    arma::uvec obj_ind(12, arma::fill::value(1)), res; // 0-based object 1 -> returns 2

    res = quadriga_lib::point_inside_mesh<double>(&points, &mesh, &obj_ind);
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

TEST_CASE("Test Point inside Mesh - Cube far from the origin")
{
    // A 10 m cube at UTM-like coordinates and 12 points 5 cm inside (1) or outside (0) each face; the default kernel
    // (AVX2 in AVX2 builds) must classify them as at the origin
    for (double off : {0.0, 5e6, 1e7})
    {
        INFO("offset (" << off << ", " << off << ")");
        arma::mat mesh = quadriga_lib::cube<double>({5.0}, {}, {off, off, 5.0});
        arma::mat pts(12, 3);
        arma::uvec expect(12);
        arma::uword n = 0;
        for (int ax = 0; ax < 3; ++ax)
            for (int s = -1; s <= 1; s += 2)
                for (double dd : {-0.05, 0.05})
                {
                    pts(n, 0) = off, pts(n, 1) = off, pts(n, 2) = 5.0;
                    pts(n, ax) += s * (5.0 + dd);
                    expect(n++) = dd < 0.0 ? 1 : 0;
                }
        arma::uvec res = quadriga_lib::point_inside_mesh<double>(&pts, &mesh);
        CHECK(arma::all(res == expect));
    }
}
