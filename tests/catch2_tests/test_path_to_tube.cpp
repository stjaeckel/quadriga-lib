// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <cmath>

// Distance of each vertex of ring r (n_edges vertices) from point c, and its projection onto the direction d
static void check_ring(const arma::mat &vert, arma::uword r, arma::uword n_edges, const arma::vec &c, const arma::vec &d,
                       double radius, double tol = 1e-9)
{
    arma::vec dn = d / std::sqrt(d(0) * d(0) + d(1) * d(1) + d(2) * d(2));
    for (arma::uword e = 0; e < n_edges; ++e)
    {
        arma::vec v = vert.col(r * n_edges + e) - c;
        double len = std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2));
        double proj = v(0) * dn(0) + v(1) * dn(1) + v(2) * dn(2);
        INFO("ring " << r << ", edge " << e);
        CHECK(std::abs(len - radius) < tol);
        CHECK(std::abs(proj) < tol);
    }
}

TEST_CASE("path_to_tube - straight, nearly straight and reversed paths give finite meshes")
{
    const arma::uword n_edges = 4;
    const double radius = 0.5;
    arma::mat vert, vA, vB;
    arma::umat faces, fA, fB;

    // 3 collinear points along x: same vertices as the two 2-point segments
    arma::mat pc = {{0.0, 10.0, 20.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    quadriga_lib::path_to_tube<double>(&pc, &vert, &faces, radius, n_edges);
    REQUIRE(vert.n_cols == 3 * n_edges);
    REQUIRE(faces.n_cols == 2 * n_edges);
    CHECK(vert.is_finite());
    arma::mat pA = pc.cols(0, 1), pB = pc.cols(1, 2);
    quadriga_lib::path_to_tube<double>(&pA, &vA, &fA, radius, n_edges);
    quadriga_lib::path_to_tube<double>(&pB, &vB, &fB, radius, n_edges);
    CHECK(arma::approx_equal(vert.cols(0, 2 * n_edges - 1), vA, "absdiff", 1e-12));
    CHECK(arma::approx_equal(vert.cols(2 * n_edges, 3 * n_edges - 1), vB.cols(n_edges, 2 * n_edges - 1), "absdiff", 1e-12));

    // 5 collinear points in an oblique direction, and float precision
    arma::vec d = {1.0, 2.0, 3.0};
    arma::mat po(3, 5);
    for (arma::uword i = 0; i < 5; ++i)
        po.col(i) = arma::vec({1.0, -2.0, 0.5}) + (double)i * 3.0 * d;
    quadriga_lib::path_to_tube<double>(&po, &vert, &faces, radius, n_edges);
    REQUIRE(vert.n_cols == 5 * n_edges);
    CHECK(vert.is_finite());
    for (arma::uword r = 0; r < 5; ++r)
        check_ring(vert, r, n_edges, po.col(r), d, radius);
    arma::fmat pof = arma::conv_to<arma::fmat>::from(po), vf;
    quadriga_lib::path_to_tube<float>(&pof, &vf, &faces, (float)radius, n_edges);
    CHECK(vf.is_finite());

    // A bend of 1e-9 rad at the middle point (the ring there is projected onto the bisecting plane by
    // ray_triangle_intersect, which computes in single precision)
    arma::mat pn = {{0.0, 10.0, 20.0}, {0.0, 0.0, 10.0 * std::sin(1e-9)}, {0.0, 0.0, 0.0}};
    quadriga_lib::path_to_tube<double>(&pn, &vert, &faces, radius, n_edges);
    CHECK(vert.is_finite());
    for (arma::uword r = 0; r < 3; ++r)
        check_ring(vert, r, n_edges, pn.col(r), arma::vec({1.0, 0.0, 0.0}), radius, 1e-6);

    // Reversed path (180 deg turn): the split inserts an extra ring at the turning point
    arma::mat pr = {{0.0, 10.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};
    quadriga_lib::path_to_tube<double>(&pr, &vert, &faces, radius, n_edges);
    REQUIRE(vert.n_cols == 4 * n_edges);
    CHECK(vert.is_finite());
    check_ring(vert, 0, n_edges, pr.col(0), arma::vec({1.0, 0.0, 0.0}), radius);
    check_ring(vert, 1, n_edges, pr.col(1), arma::vec({1.0, 0.0, 0.0}), radius);
    check_ring(vert, 2, n_edges, pr.col(1), arma::vec({1.0, 0.0, 0.0}), radius);
    check_ring(vert, 3, n_edges, pr.col(2), arma::vec({1.0, 0.0, 0.0}), radius);

    // Reversal followed by a straight continuation and a normal bend
    arma::mat pm = {{0.0, 10.0, 5.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 8.0}, {0.0, 0.0, 0.0, 0.0, 0.0}};
    quadriga_lib::path_to_tube<double>(&pm, &vert, &faces, radius, n_edges);
    CHECK(vert.is_finite());
}
