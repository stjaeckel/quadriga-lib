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

#include <iostream>
#include <string>

TEST_CASE("Arrayant combine pattern - Minimal test")
{
    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    ant.copy_element(0, 2);
    ant.coupling_re.ones(3, 1);
    ant.coupling_im.reset();

    ant = ant.combine_pattern();

    arma::fcube T(181, 361, 1, arma::fill::value(2.0f));
    CHECK(arma::approx_equal(ant.e_theta_re, T, "absdiff", 1e-6));

    T.zeros();
    CHECK(arma::approx_equal(ant.e_theta_im, T, "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.e_phi_re, T, "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.e_phi_im, T, "absdiff", 1e-6));

    arma::fmat Q(3, 1);
    CHECK(arma::approx_equal(ant.element_pos, Q, "absdiff", 1e-6));

    Q.ones(1, 1);
    CHECK(arma::approx_equal(ant.coupling_re, Q, "absdiff", 1e-6));

    Q.zeros(1, 1);
    CHECK(arma::approx_equal(ant.coupling_im, Q, "absdiff", 1e-6));

    // Multiple copies
    ant.copy_element(0, {1,2});
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), ant.e_theta_re.slice(1), "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), ant.e_theta_re.slice(2), "absdiff", 1e-6));
}

TEST_CASE("Arrayant rotation - Minimal test")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(5.0, 5.0);

    arma::uword i = ant.e_theta_re.index_max();
    arma::uvec s = arma::ind2sub(arma::size(ant.e_theta_re), i);
    CHECK(s(0) == 90);
    CHECK(s(1) == 180);

    ant.rotate_pattern(0.0, 0.0, 90.0);

    i = ant.e_theta_re.index_max();
    s = arma::ind2sub(arma::size(ant.e_theta_re), i);
    CHECK(s(0) == 90);
    CHECK(s(1) == 270);

    ant = quadriga_lib::generate_arrayant_custom<float>(5.0, 5.0);
    ant.rotate_pattern(0.0, -45.0);

    i = ant.e_theta_re.index_max();
    s = arma::ind2sub(arma::size(ant.e_theta_re), i);
    CHECK(s(0) == 135);
    CHECK(s(1) == 180);
}

TEST_CASE("Arrayant rotation - Full sphere, separate output")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(5.0, 5.0);
    arma::fmat T = {1.0, 2.0, 3.0};
    ant.element_pos = T.t();
    ant.copy_element(0, 1);
    ant.e_theta_re.slice(1) = ant.e_theta_re.slice(0) * 2.0f;
    ant.element_pos.col(1) = 10.0 * T.t();
    ant.read_only = true;

    arma::uword n_elem = ant.e_theta_re.n_elem;
    float *a = ant.e_theta_re.memptr(), *b = ant.e_theta_im.memptr(),
          *c = ant.e_phi_re.memptr(), *d = ant.e_phi_im.memptr();

    // Check that the two elements point into the same direction
    float val = ant.e_theta_re.at(90, 180, 0);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 1) - 2.0 * val) < 0.001);

    // Rotate first element around z-axis
    quadriga_lib::arrayant<float> out;
    ant.rotate_pattern(0.0f, 0.0f, 90.0f, 0, 0, &out);

    // Check if there is no change in the input
    CHECK(a == ant.e_theta_re.memptr());
    CHECK(b == ant.e_theta_im.memptr());
    CHECK(c == ant.e_phi_re.memptr());
    CHECK(d == ant.e_phi_im.memptr());
    CHECK(ant.e_theta_re.n_elem == n_elem);
    CHECK(ant.e_theta_im.n_elem == n_elem);
    CHECK(ant.e_phi_re.n_elem == n_elem);
    CHECK(ant.e_phi_im.n_elem == n_elem);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 0) - val) < 0.001);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 1) - 2.0 * val) < 0.001);

    // Test output
    T = {-2.0, 1.0, 3.0};
    CHECK(out.e_theta_re.n_elem == n_elem / 2);
    CHECK(out.e_theta_im.n_elem == n_elem / 2);
    CHECK(out.e_phi_re.n_elem == n_elem / 2);
    CHECK(out.e_phi_im.n_elem == n_elem / 2);
    CHECK(std::abs(out.e_theta_re.at(90, 270, 0) - val) < 0.001);
    CHECK(arma::approx_equal(out.element_pos, T.t(), "absdiff", 1e-5));

    // Second element
    T = {20.0, -10.0, 30.0};
    ant.rotate_pattern(0.0f, 0.0f, -90.0f, 0, 1, &out);
    CHECK(std::abs(out.e_theta_re.at(90, 90, 0) - 2.0 * val) < 0.001);
    CHECK(arma::approx_equal(out.element_pos, T.t(), "absdiff", 1e-5));

    // Both elements
    T = {{-1.0, -10.0}, {-2.0, -20.0}, {3.0, 30.0}};
    ant.rotate_pattern(0.0f, 0.0f, 180.0f, 0, -1, &out);
    CHECK(out.e_theta_re.n_elem == n_elem);
    CHECK(out.e_theta_im.n_elem == n_elem);
    CHECK(out.e_phi_re.n_elem == n_elem);
    CHECK(out.e_phi_im.n_elem == n_elem);
    CHECK(std::abs(out.e_theta_re.at(90, 0, 0) - val) < 0.001);
    CHECK(std::abs(out.e_theta_re.at(90, 360, 1) - 2.0 * val) < 0.001);
    CHECK(arma::approx_equal(out.element_pos, T, "absdiff", 1e-5));

    // Only polarization
    T = {1.0, 2.0, 3.0};
    ant.rotate_pattern(90.0f, 0.0f, 0.0f, 2, 0, &out);
    CHECK(out.e_theta_re.n_elem == n_elem / 2);
    CHECK(std::abs(out.e_phi_re.at(90, 180, 0) - val) < 0.001);
    CHECK(arma::approx_equal(out.element_pos, T.t(), "absdiff", 1e-5));
}

TEST_CASE("Arrayant rotation - Partial sphere, separate output")
{
    arma::fmat T;

    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    ant.e_theta_re.zeros();
    ant.e_theta_re.at(90, 180, 0) = 1.0f;
    ant.remove_zeros();

    T = {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), T, "absdiff", 1e-5));

    T = {1.0, 2.0, 3.0};
    ant.element_pos = T.t();
    ant.copy_element(0, 1);
    ant.e_theta_re.slice(1) = ant.e_theta_re.slice(0) * 2.0f;
    ant.element_pos.col(1) = 10.0 * T.t();
    ant.read_only = true;

    arma::uword n_elem = ant.e_theta_re.n_elem;
    float *a = ant.e_theta_re.memptr(), *b = ant.e_theta_im.memptr(),
          *c = ant.e_phi_re.memptr(), *d = ant.e_phi_im.memptr();

    // Rotate first element
    quadriga_lib::arrayant<float> out;
    ant.rotate_pattern(0.0f, 0.0f, 90.0f, 0, 0, &out);

    T = {-2.0, 1.0, 3.0};
    CHECK(a == ant.e_theta_re.memptr());
    CHECK(b == ant.e_theta_im.memptr());
    CHECK(c == ant.e_phi_re.memptr());
    CHECK(d == ant.e_phi_im.memptr());
    CHECK(ant.e_theta_re.n_elem == n_elem);
    CHECK(ant.e_theta_im.n_elem == n_elem);
    CHECK(ant.e_phi_re.n_elem == n_elem);
    CHECK(ant.e_phi_im.n_elem == n_elem);
    CHECK(out.n_elevation() == 3);
    CHECK(out.n_elements() == 1);
    CHECK(arma::approx_equal(out.elevation_grid, ant.elevation_grid, "absdiff", 1e-5f));
    CHECK(std::abs(arma::max(arma::vectorise(out.e_theta_re)) - 1.0f) < 1e-5f);
    CHECK(arma::approx_equal(out.element_pos, T.t(), "absdiff", 1e-5f));

    T = out.e_theta_re.slice(0).t();
    arma::fvec U = out.azimuth_grid % T.col(1);
    CHECK(std::abs(arma::accu(U) - arma::fdatum::pi / 2.0f) < 2e-5f);

    // Second element
    ant.rotate_pattern(0.0f, 0.0f, -90.0f, 0, 1, &out);

    T = {20.0, -10.0, 30.0};
    CHECK(out.n_elevation() == 3);
    CHECK(out.n_elements() == 1);
    CHECK(arma::approx_equal(out.elevation_grid, ant.elevation_grid, "absdiff", 1e-5f));
    CHECK(std::abs(arma::max(arma::vectorise(out.e_theta_re)) - 2.0f) < 1e-5f);
    CHECK(arma::approx_equal(out.element_pos, T.t(), "absdiff", 1e-5));

    T = out.e_theta_re.slice(0).t() / 2.0f;
    U = out.azimuth_grid % T.col(1);
    CHECK(std::abs(arma::accu(U) + arma::fdatum::pi / 2.0f) < 2e-5f);

    // Both elements
    ant.rotate_pattern(0.0f, 0.0f, 180.0f, 0, -1, &out);

    T = {{-1.0, -10.0}, {-2.0, -20.0}, {3.0, 30.0}};
    arma::uword end = out.n_azimuth() - 1;
    CHECK(out.n_elevation() == 3);
    CHECK(out.n_elements() == 2);
    CHECK(arma::approx_equal(out.elevation_grid, ant.elevation_grid, "absdiff", 1e-5f));
    CHECK(std::abs(out.azimuth_grid.at(0) + arma::fdatum::pi) < 1e-5f);
    CHECK(std::abs(out.azimuth_grid.at(end) - arma::fdatum::pi) < 1e-5f);
    CHECK(std::abs(out.e_theta_re.at(1, 0, 0) - 1.0f) < 1e-4f);
    CHECK(std::abs(out.e_theta_re.at(1, end, 0) - 1.0f) < 1e-4f);
    CHECK(std::abs(out.e_theta_re.at(1, 0, 1) - 2.0f) < 1e-4f);
    CHECK(std::abs(out.e_theta_re.at(1, end, 1) - 2.0f) < 1e-4f);
    CHECK(arma::approx_equal(out.element_pos, T, "absdiff", 1e-5));
}

TEST_CASE("Arrayant rotation - Full sphere, inplace")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(5.0, 5.0);
    arma::fmat T = {1.0, 2.0, 3.0};
    ant.element_pos = T.t();
    ant.copy_element(0, 1);
    ant.e_theta_re.slice(1) = ant.e_theta_re.slice(0) * 2.0f;
    ant.element_pos.col(1) = 10.0 * T.t();
    ant.read_only = true;

    arma::uword n_elem = ant.e_theta_re.n_elem;
    float val = ant.e_theta_re.at(90, 180, 0);

    // Check error message
    REQUIRE_THROWS_AS(ant.rotate_pattern(0.0f, 0.0f, 90.0f), std::invalid_argument);
    ant.read_only = false;

    // First element
    T = {{-2.0, 10.0}, {1, 20.0}, {3.0, 30.0}};
    ant.rotate_pattern(0.0f, 0.0f, 90.0f, 0, 0);
    CHECK(ant.e_theta_re.n_elem == n_elem);
    CHECK(ant.e_theta_im.n_elem == n_elem);
    CHECK(ant.e_phi_re.n_elem == n_elem);
    CHECK(ant.e_phi_im.n_elem == n_elem);
    CHECK(std::abs(ant.e_theta_re.at(90, 270, 0) - val) < 0.001);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 1) - 2.0 * val) < 0.001);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));

    // Second element
    T = {{-2.0, 20.0}, {1.0, -10.0}, {3.0, 30.0}};
    ant.rotate_pattern(0.0f, 0.0f, -90.0f, 0, 1);
    CHECK(std::abs(ant.e_theta_re.at(90, 270, 0) - val) < 0.001);
    CHECK(std::abs(ant.e_theta_re.at(90, 90, 1) - 2.0 * val) < 0.001);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));

    // Both elements
    T = {{2.0, -20.0}, {-1.0, 10.0}, {3.0, 30.0}};
    ant.rotate_pattern(0.0f, 0.0f, 180.0f, 0);
    CHECK(std::abs(ant.e_theta_re.at(90, 90, 0) - val) < 0.001);
    CHECK(std::abs(ant.e_theta_re.at(90, 270, 1) - 2.0 * val) < 0.001);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));

    // Polarization
    T = {{1.0, -20.0}, {2.0, 10.0}, {3.0, 30.0}};
    ant.rotate_pattern(0.0f, 0.0f, 90.0f, 0, 0);
    ant.rotate_pattern(90.0f, 0.0f, 0.0f, 2, 0);
    CHECK(std::abs(ant.e_phi_re.at(90, 180, 0) - val) < 0.001);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));
}

TEST_CASE("Arrayant rotation - Partial sphere, inplace")
{
    arma::fmat T = {1.0, 2.0, 3.0};

    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    ant.e_theta_re.zeros();
    ant.e_theta_re.at(90, 180, 0) = 1.0f;
    ant.remove_zeros();
    ant.element_pos = T.t();
    ant.copy_element(0, 1);
    ant.e_theta_re.slice(1) = ant.e_theta_re.slice(0) * 2.0f;
    ant.element_pos.col(1) = 10.0 * T.t();
    // Check error message if trying to rotate single element
    REQUIRE_THROWS_AS(ant.rotate_pattern(0.0f, 0.0f, 90.0f, 0, 0), std::invalid_argument);

    // Both elements
    ant.rotate_pattern(0.0f, 0.0f, 180.0f);

    T = {{-1.0, -10.0}, {-2.0, -20.0}, {3.0, 30.0}};
    arma::uword end = ant.n_azimuth() - 1;
    CHECK(ant.n_elevation() == 3);
    CHECK(ant.n_elements() == 2);
    CHECK(arma::approx_equal(ant.elevation_grid, ant.elevation_grid, "absdiff", 1e-5f));
    CHECK(std::abs(ant.azimuth_grid.at(0) + arma::fdatum::pi) < 1e-5f);
    CHECK(std::abs(ant.azimuth_grid.at(end) - arma::fdatum::pi) < 1e-5f);
    CHECK(std::abs(ant.e_theta_re.at(1, 0, 0) - 1.0f) < 1e-4f);
    CHECK(std::abs(ant.e_theta_re.at(1, end, 0) - 1.0f) < 1e-4f);
    CHECK(std::abs(ant.e_theta_re.at(1, 0, 1) - 2.0f) < 1e-4f);
    CHECK(std::abs(ant.e_theta_re.at(1, end, 1) - 2.0f) < 1e-4f);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));
}
