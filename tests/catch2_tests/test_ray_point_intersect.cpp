// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

    // Generate a set of ray tubes
    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);

    // Change the location
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;

    // Call intersect
    arma::u32_vec hit_count;
    auto ind = quadriga_lib::ray_point_intersect(&pointsR, &orig, &trivec, &tridir, &sub_cloud_index, &hit_count);
}