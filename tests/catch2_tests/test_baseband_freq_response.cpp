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
#include "quadriga_channel.hpp"
#include <iostream>

TEST_CASE("Baseband Frequency Response - Single")
{
    arma::cube coeff_re(4, 3, 2, arma::fill::zeros), coeff_im(4, 3, 2, arma::fill::zeros);
    coeff_re(arma::span::all, arma::span(0), arma::span(0)) = arma::linspace<arma::vec>(0.25, 1.0, 4);
    coeff_re(arma::span::all, arma::span(1), arma::span(0)) = arma::linspace<arma::vec>(1.0, 4.0, 4);
    coeff_im(arma::span::all, arma::span(2), arma::span(0)) = arma::linspace<arma::vec>(1.0, 4.0, 4);
    coeff_re.slice(1) = -coeff_re.slice(0);
    coeff_im.slice(1) = -coeff_im.slice(0);

    double fc = 299792458.0; // Wavelength = 1 m
    arma::cube delay(4, 3, 2, arma::fill::zeros);
    delay.slice(0).fill(1.0 / fc);
    delay.slice(1).fill(1.5 / fc);

    auto pilots = arma::linspace<arma::vec>(0.0, 2.0, 21);

    arma::cube hmat_re, hmat_im;
    quadriga_lib::baseband_freq_response(&coeff_re, &coeff_im, &delay, &pilots, fc, &hmat_re, &hmat_im);

    auto T = arma::mat(4, 3);

    CHECK(arma::approx_equal(hmat_im.slice(0), T, "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_im.slice(20), T, "absdiff", 3e-6));

    CHECK(arma::approx_equal(hmat_re.slice(0), T, "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_re.slice(20), T, "absdiff", 3e-6));

    for (arma::uword i = 0; i < 21; ++i)
    {
        arma::cube A = hmat_re.subcube(0, 0, i, 3, 0, i) * 4.0;
        arma::cube B = hmat_re.subcube(0, 1, i, 3, 1, i);
        CHECK(arma::approx_equal(A, B, "absdiff", 1e-8));

        A = hmat_im.subcube(0, 0, i, 3, 0, i) * 4.0;
        B = hmat_im.subcube(0, 1, i, 3, 1, i);
        CHECK(arma::approx_equal(A, B, "absdiff", 1e-8));
    }

    T = {{-0.25, -1, 1}, {-0.5, -2, 2}, {-0.75, -3, 3}, {-1, -4, 4}};
    CHECK(arma::approx_equal(hmat_re.slice(5), T, "absdiff", 1.5e-6));

    T = {{0.5, 2, 0}, {1, 4, 0}, {1.5, 6, 0}, {2, 8, 0}};
    CHECK(arma::approx_equal(hmat_re.slice(10), T, "absdiff", 1.5e-6));

    T = {{-0.25, -1, -1}, {-0.5, -2, -2}, {-0.75, -3, -3}, {-1, -4, -4}};
    CHECK(arma::approx_equal(hmat_re.slice(15), T, "absdiff", 5e-6));
    CHECK(arma::approx_equal(hmat_im.slice(5), T, "absdiff", 1.5e-6));

    T = {{0.25, 1, -1}, {0.5, 2, -2}, {0.75, 3, -3}, {1, 4, -4}};
    CHECK(arma::approx_equal(hmat_im.slice(15), T, "absdiff", 1e-5));

    T = {{0, 0, 2}, {0, 0, 4}, {0, 0, 6}, {0, 0, 8}};
    CHECK(arma::approx_equal(hmat_im.slice(10), T, "absdiff", 1.5e-6));
}

TEST_CASE("Baseband Frequency Response - Multi")
{

    arma::fcube coeff_re(4, 3, 2, arma::fill::zeros), coeff_im(4, 3, 2, arma::fill::zeros);
    coeff_re(arma::span::all, arma::span(0), arma::span(0)) = arma::linspace<arma::fvec>(0.25, 1.0, 4);
    coeff_re(arma::span::all, arma::span(1), arma::span(0)) = arma::linspace<arma::fvec>(1.0, 4.0, 4);
    coeff_im(arma::span::all, arma::span(2), arma::span(0)) = arma::linspace<arma::fvec>(1.0, 4.0, 4);
    coeff_re.slice(1) = -coeff_re.slice(0);
    coeff_im.slice(1) = -coeff_im.slice(0);

    float fc = 299792458.0; // Wavelength = 1 m
    arma::fcube delay(1, 1, 2, arma::fill::zeros);
    delay.slice(0).fill(1.0 / fc);
    delay.slice(1).fill(1.5 / fc);

    std::vector<arma::fcube> coeff_re_vec;
    coeff_re_vec.push_back(coeff_re);
    coeff_re_vec.push_back(coeff_re * 2.0f);
    coeff_re_vec.push_back(coeff_re * 3.0f);

    std::vector<arma::fcube> coeff_im_vec;
    coeff_im_vec.push_back(coeff_im);
    coeff_im_vec.push_back(coeff_im * 2.0f);
    coeff_im_vec.push_back(coeff_im * 3.0f);

    std::vector<arma::fcube> delay_vec;
    delay_vec.push_back(delay);
    delay_vec.push_back(delay);
    delay_vec.push_back(delay);

    auto pilots = arma::linspace<arma::fvec>(0.0, 2.0, 21);

    std::vector<arma::fcube> hmat_re, hmat_im;
    quadriga_lib::baseband_freq_response_vec(&coeff_re_vec, &coeff_im_vec, &delay_vec, &pilots, fc, &hmat_re, &hmat_im);

    CHECK(hmat_re.size() == 3);
    CHECK(hmat_im.size() == 3);

    auto T = arma::fmat(4, 3);

    CHECK(arma::approx_equal(hmat_im[0].slice(0), T, "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_im[0].slice(20), T, "absdiff", 3e-6));

    CHECK(arma::approx_equal(hmat_re[0].slice(0), T, "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_re[0].slice(20), T, "absdiff", 3e-6));

    for (arma::uword i = 0; i < 21; ++i)
    {
        arma::fcube A = hmat_re[0].subcube(0, 0, i, 3, 0, i) * 4.0;
        arma::fcube B = hmat_re[0].subcube(0, 1, i, 3, 1, i);
        CHECK(arma::approx_equal(A, B, "absdiff", 1e-8));

        A = hmat_im[0].subcube(0, 0, i, 3, 0, i) * 4.0;
        B = hmat_im[0].subcube(0, 1, i, 3, 1, i);
        CHECK(arma::approx_equal(A, B, "absdiff", 1e-8));
    }

    T = {{-0.25, -1, 1}, {-0.5, -2, 2}, {-0.75, -3, 3}, {-1, -4, 4}};
    CHECK(arma::approx_equal(hmat_re[0].slice(5), T, "absdiff", 1.5e-6));

    T = {{0.5, 2, 0}, {1, 4, 0}, {1.5, 6, 0}, {2, 8, 0}};
    CHECK(arma::approx_equal(hmat_re[0].slice(10), T, "absdiff", 1.5e-6));

    T = {{-0.25, -1, -1}, {-0.5, -2, -2}, {-0.75, -3, -3}, {-1, -4, -4}};
    CHECK(arma::approx_equal(hmat_re[0].slice(15), T, "absdiff", 5e-6));
    CHECK(arma::approx_equal(hmat_im[0].slice(5), T, "absdiff", 1.5e-6));

    T = {{0.25, 1, -1}, {0.5, 2, -2}, {0.75, 3, -3}, {1, 4, -4}};
    CHECK(arma::approx_equal(hmat_im[0].slice(15), T, "absdiff", 1e-5));

    T = {{0, 0, 2}, {0, 0, 4}, {0, 0, 6}, {0, 0, 8}};
    CHECK(arma::approx_equal(hmat_im[0].slice(10), T, "absdiff", 1.5e-6));

    CHECK(arma::approx_equal(hmat_re[0] * 2.0, hmat_re[1], "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_im[0] * 2.0, hmat_im[1], "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_re[0] * 3.0, hmat_re[2], "absdiff", 2e-6));
    CHECK(arma::approx_equal(hmat_im[0] * 3.0, hmat_im[2], "absdiff", 2e-6));

    // Change the snapshot order
    arma::u32_vec i_snap = {1, 2, 1, 0};

    quadriga_lib::baseband_freq_response_vec(&coeff_re_vec, &coeff_im_vec, &delay_vec, &pilots, fc, &hmat_re, &hmat_im, &i_snap);

    CHECK(hmat_re.size() == 4);
    CHECK(hmat_im.size() == 4);

    T = {{-0.25, -1, 1}, {-0.5, -2, 2}, {-0.75, -3, 3}, {-1, -4, 4}};
    CHECK(arma::approx_equal(hmat_re[3].slice(5), T, "absdiff", 1.5e-6));

    T = {{0.5, 2, 0}, {1, 4, 0}, {1.5, 6, 0}, {2, 8, 0}};
    CHECK(arma::approx_equal(hmat_re[3].slice(10), T, "absdiff", 1.5e-6));

    T = {{-0.25, -1, -1}, {-0.5, -2, -2}, {-0.75, -3, -3}, {-1, -4, -4}};
    CHECK(arma::approx_equal(hmat_re[3].slice(15), T, "absdiff", 5e-6));
    CHECK(arma::approx_equal(hmat_im[3].slice(5), T, "absdiff", 1.5e-6));

    T = {{0.25, 1, -1}, {0.5, 2, -2}, {0.75, 3, -3}, {1, 4, -4}};
    CHECK(arma::approx_equal(hmat_im[3].slice(15), T, "absdiff", 1e-5));

    T = {{0, 0, 2}, {0, 0, 4}, {0, 0, 6}, {0, 0, 8}};
    CHECK(arma::approx_equal(hmat_im[3].slice(10), T, "absdiff", 1.5e-6));

    CHECK(arma::approx_equal(hmat_re[3] * 2.0, hmat_re[0], "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_im[3] * 2.0, hmat_im[0], "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_re[3] * 3.0, hmat_re[1], "absdiff", 2e-6));
    CHECK(arma::approx_equal(hmat_im[3] * 3.0, hmat_im[1], "absdiff", 2e-6));
    CHECK(arma::approx_equal(hmat_re[3] * 2.0, hmat_re[2], "absdiff", 1.5e-6));
    CHECK(arma::approx_equal(hmat_im[3] * 2.0, hmat_im[2], "absdiff", 1.5e-6));
}