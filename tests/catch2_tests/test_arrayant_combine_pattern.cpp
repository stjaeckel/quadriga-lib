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

    auto out = ant.combine_pattern();

    arma::fcube T(181, 361, 1, arma::fill::value(2.0f));
    CHECK(arma::approx_equal(out.e_theta_re, T, "absdiff", 1e-6));

    T.zeros();
    CHECK(arma::approx_equal(out.e_theta_im, T, "absdiff", 1e-6));
    CHECK(arma::approx_equal(out.e_phi_re, T, "absdiff", 1e-6));
    CHECK(arma::approx_equal(out.e_phi_im, T, "absdiff", 1e-6));

    arma::fmat Q(3, 1);
    CHECK(arma::approx_equal(out.element_pos, Q, "absdiff", 1e-6));

    Q.ones(1, 1);
    CHECK(arma::approx_equal(out.coupling_re, Q, "absdiff", 1e-6));

    Q.zeros(1, 1);
    CHECK(arma::approx_equal(out.coupling_im, Q, "absdiff", 1e-6));
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
