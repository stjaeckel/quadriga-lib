// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_tools.hpp"
#include <iostream>

TEST_CASE("Ray-Point Intersect - Simple Mode")
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
    points.col(1) *= 0.1f;

    // Create a sub-cloud index
    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index, reverse_index;
    quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8, nullptr, &reverse_index);

    // Generate a set of ray beams
    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);

    // Change the location
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;

    // Call intersect
    arma::u32_vec hit_count;
    auto ind = quadriga_lib::ray_point_intersect(&pointsR, &orig, &trivec, &tridir, &sub_cloud_index, &hit_count);
    CHECK(arma::all(hit_count == 1));
}

TEST_CASE("Ray-Point Intersect - Ray Subdivision")
{
    double res = 0.1;
    arma::vec rx_pos = {-10, -10, 0.1}; // Lower left point
    arma::vec rx_xy = {20, 20};         // x-y scale

    // Generate x and y vectors
    arma::vec x = arma::regspace(rx_pos(0), res, rx_pos(0) + rx_xy(0));
    arma::vec y = arma::regspace(rx_pos(1), res, rx_pos(1) + rx_xy(1));

    // Create meshgrid
    arma::mat X(y.n_elem, x.n_elem);
    arma::mat Y(y.n_elem, x.n_elem);

    for (arma::uword i = 0; i < y.n_elem; ++i)
        X.row(i) = x.t();

    for (arma::uword j = 0; j < x.n_elem; ++j)
        Y.col(j) = y;

    // Flatten the meshgrid and create the points matrix
    arma::mat points(X.n_elem, 3);
    points.col(0) = arma::vectorise(X);
    points.col(1) = arma::vectorise(Y);
    points.col(2).fill(rx_pos(2));

    // Generate a set of ray beams
    arma::mat orig, trivec, tridir;
    quadriga_lib::icosphere<double>(12, 1.0, &orig, nullptr, &trivec, &tridir, true);

    // Change the location
    orig.col(0) -= 10.0;
    orig.col(1) -= 20.0;
    orig.col(2) -= 30.0;

    // Call intersect
    arma::u32_vec hit_count;
    auto ind = quadriga_lib::ray_point_intersect<double>(&points, &orig, &trivec, &tridir, nullptr, &hit_count);
    CHECK(hit_count.n_elem == points.n_rows);
    CHECK(arma::all(hit_count == 1));

    // Subdivide all rays
    arma::mat origN, trivecN, tridirN;
    quadriga_lib::subdivide_rays<double>(&orig, &trivec, &tridir, nullptr, &origN, &trivecN, &tridirN);
    CHECK(origN.n_rows == 4 * orig.n_rows);

    // Call intersect
    hit_count.reset();
    quadriga_lib::ray_point_intersect<double>(&points, &origN, &trivecN, &tridirN, nullptr, &hit_count);
    CHECK(arma::all(hit_count == 1));

    // Subdivide selected rays
    arma::u32_vec index(points.n_rows);
    for (arma::uword i = 0; i < points.n_rows; ++i)
        index.at(i) = ind[i].at(0);

    index = arma::unique(index);
    CHECK(index.n_elem < 4 * orig.n_rows);

    quadriga_lib::subdivide_rays<double>(&orig, &trivec, &tridir, nullptr, &origN, &trivecN, &tridirN, nullptr, &index);
    CHECK(origN.n_rows == 4 * index.n_elem);

    // Call intersect
    hit_count.reset();
    quadriga_lib::ray_point_intersect<double>(&points, &origN, &trivecN, &tridirN, nullptr, &hit_count);
    CHECK(arma::all(hit_count == 1));
}