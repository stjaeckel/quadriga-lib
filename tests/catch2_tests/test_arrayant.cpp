// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

TEST_CASE("Arrayant append")
{
    auto ant1 = quadriga_lib::generate_arrayant_dipole<float>();
    auto ant2 = quadriga_lib::generate_arrayant_custom<float>(10.0, 10.0);

    ant1.coupling_re.at(0) = 2.0;
    ant2.coupling_re.at(0) = 3.0;

    auto ant = ant1.append(&ant2);

    arma::fmat T;
    T = {{2.0, 0.0}, {0.0, 3.0}};
    CHECK(arma::approx_equal(ant.coupling_re, T, "absdiff", 1.0e-6f));

    float g0 = ant.calc_directivity_dBi(0);
    float g1 = ant.calc_directivity_dBi(1);

    CHECK(std::abs(g0 - 1.760964f) < 0.0001f);
    CHECK(std::abs(g1 - 25.627f) < 0.001f);
}

// B-043: the default constructor must initialise check_ptr. The arrayant is created twice in the same stack slot;
// small matrices live inside the Armadillo objects, so stale check_ptr values of the first object would match the
// data pointers of the second one and let the quick is_valid() pass an invalid, never validated object.
static __attribute__((noinline)) std::string arrayant_check_ptr_helper(bool first)
{
    quadriga_lib::arrayant<double> a;
    a.e_theta_re.ones(2, 2, 1), a.e_theta_im.zeros(2, 2, 1), a.e_phi_re.zeros(2, 2, 1), a.e_phi_im.zeros(2, 2, 1);
    a.elevation_grid = {-0.5, 0.5};
    a.element_pos.zeros(3, 1), a.coupling_re.ones(1, 1), a.coupling_im.zeros(1, 1);
    if (first)
    {
        a.azimuth_grid = {-1.0, 1.0};
        return a.validate();
    }
    a.azimuth_grid = {1.0, -1.0}; // invalid: unsorted; never validated
    return a.is_valid();
}

TEST_CASE("Arrayant - Default constructor initialises check_ptr")
{
    quadriga_lib::arrayant<double> a;
    for (auto p : a.check_ptr)
        CHECK(p == nullptr);

    CHECK(arrayant_check_ptr_helper(true).empty());
    CHECK(!arrayant_check_ptr_helper(false).empty());
}

// B-008: append must build the block-diagonal coupling [C1 0; 0 C2] (as arrayant_concat_multi), also when the number
// of ports differs from the number of elements, and treat an empty coupling_im as zeros (it used to crash).
TEST_CASE("Arrayant append - Block-diagonal coupling with fewer ports than elements (B-008)")
{
    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.copy_element(0, 1);
    a.coupling_re = arma::mat(2, 1, arma::fill::ones); // 2 elements -> 1 port
    a.coupling_im = arma::mat(arma::colvec{0.5, -0.5});

    auto b = quadriga_lib::generate_arrayant_xpol<double>(10.0); // 2 elements, 2 ports (identity)
    b.coupling_re = {{2.0, 0.0}, {0.0, 3.0}};
    b.coupling_im = {{0.0, 0.25}, {0.0, 0.0}};

    auto c = a.append(&a);
    arma::mat T = {{1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}};
    CHECK(arma::approx_equal(c.coupling_re, T, "absdiff", 1e-14));
    T = {{0.5, 0.0}, {-0.5, 0.0}, {0.0, 0.5}, {0.0, -0.5}};
    CHECK(arma::approx_equal(c.coupling_im, T, "absdiff", 1e-14));
    CHECK(c.n_ports() == 2);

    auto cm = quadriga_lib::arrayant_concat_multi(std::vector<quadriga_lib::arrayant<double>>{a},
                                                  std::vector<quadriga_lib::arrayant<double>>{a});
    CHECK(arma::approx_equal(c.coupling_re, cm[0].coupling_re, "absdiff", 1e-14));
    CHECK(arma::approx_equal(c.coupling_im, cm[0].coupling_im, "absdiff", 1e-14));

    // Different port counts: 2x1 block followed by a 2x2 block
    auto d = a.append(&b);
    T = {{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
    CHECK(arma::approx_equal(d.coupling_re, T, "absdiff", 1e-14));
    T = {{0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {0.0, 0.0, 0.25}, {0.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(d.coupling_im, T, "absdiff", 1e-14));
    auto dm = quadriga_lib::arrayant_concat_multi(std::vector<quadriga_lib::arrayant<double>>{a},
                                                  std::vector<quadriga_lib::arrayant<double>>{b});
    CHECK(arma::approx_equal(d.coupling_re, dm[0].coupling_re, "absdiff", 1e-14));
    CHECK(arma::approx_equal(d.coupling_im, dm[0].coupling_im, "absdiff", 1e-14));
    CHECK(d.is_valid(false).empty());
}

TEST_CASE("Arrayant append - Empty coupling_im is treated as zeros (B-008)")
{
    auto b = quadriga_lib::generate_arrayant_omni<double>(10.0);
    b.coupling_re = arma::mat(1, 1, arma::fill::value(2.0));
    b.coupling_im.reset();
    REQUIRE(b.is_valid(false).empty());

    quadriga_lib::arrayant<double> d;
    REQUIRE_NOTHROW(d = b.append(&b));
    arma::mat T = {{2.0, 0.0}, {0.0, 2.0}};
    CHECK(arma::approx_equal(d.coupling_re, T, "absdiff", 1e-14));
    CHECK(d.coupling_im.n_rows == 2);
    CHECK(d.coupling_im.n_cols == 2);
    CHECK(arma::all(arma::vectorise(d.coupling_im) == 0.0));

    // Both couplings empty (allowed by is_valid): identity blocks
    auto e = quadriga_lib::generate_arrayant_xpol<double>(10.0);
    e.coupling_re.reset(), e.coupling_im.reset();
    REQUIRE(e.is_valid(false).empty());
    auto f = e.append(&b);
    T = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}};
    CHECK(arma::approx_equal(f.coupling_re, T, "absdiff", 1e-14));
    CHECK(arma::all(arma::vectorise(f.coupling_im) == 0.0));

    // arrayant_concat_multi shares the coupling assembly (empty couplings used to hit an Armadillo bounds error)
    std::vector<quadriga_lib::arrayant<double>> fm;
    REQUIRE_NOTHROW(fm = quadriga_lib::arrayant_concat_multi(std::vector<quadriga_lib::arrayant<double>>{e},
                                                             std::vector<quadriga_lib::arrayant<double>>{b}));
    CHECK(arma::approx_equal(fm[0].coupling_re, T, "absdiff", 1e-14));
    CHECK(arma::all(arma::vectorise(fm[0].coupling_im) == 0.0));
}

// B-046: remove_zeros and calc_directivity_dBi must not depend on the absolute scale of the pattern; an all-zero
// pattern is left unchanged.
TEST_CASE("Arrayant remove_zeros and calc_directivity_dBi - Low-level patterns (B-046)")
{
    auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, 5.0); // 37 x 73
    const double d_ref = a.calc_directivity_dBi(0);
    quadriga_lib::arrayant<double> o_ref;
    a.remove_zeros(&o_ref);

    for (double s : {1e-6, 1e-8, 1e-12, 1e-30})
    {
        auto b = a.copy();
        b.e_theta_re *= s;
        CHECK(std::abs(b.calc_directivity_dBi(0) - d_ref) < 1e-9);
        quadriga_lib::arrayant<double> o;
        b.remove_zeros(&o);
        CHECK(o.n_elevation() == o_ref.n_elevation());
        CHECK(o.n_azimuth() == o_ref.n_azimuth());
        CHECK(arma::approx_equal(o.e_theta_re, o_ref.e_theta_re * s, "reldiff", 1e-12));
    }

    // Float pattern scaled by 1e-8 (power 1e-16, below the old absolute limits)
    auto af = quadriga_lib::generate_arrayant_custom<float>(90.0f, 90.0f, 0.0f, 5.0f);
    const float df_ref = af.calc_directivity_dBi(0);
    quadriga_lib::arrayant<float> of_ref;
    af.remove_zeros(&of_ref);
    af.e_theta_re *= 1e-8f;
    CHECK(std::abs(af.calc_directivity_dBi(0) - df_ref) < 1e-4f);
    quadriga_lib::arrayant<float> of;
    af.remove_zeros(&of);
    CHECK(of.n_elevation() == of_ref.n_elevation());
    CHECK(of.n_azimuth() == of_ref.n_azimuth());

    // All-zero pattern: unchanged (in place and into an output), directivity 0
    auto z = quadriga_lib::generate_arrayant_omni<double>(10.0);
    z.e_theta_re.zeros();
    CHECK(z.calc_directivity_dBi(0) == 0.0);
    quadriga_lib::arrayant<double> oz;
    z.remove_zeros(&oz);
    CHECK(oz.n_elevation() == z.n_elevation());
    CHECK(oz.n_azimuth() == z.n_azimuth());
    CHECK(arma::approx_equal(oz.azimuth_grid, z.azimuth_grid, "absdiff", 0.0));
    CHECK(oz.is_valid(false).empty());
    z.remove_zeros();
    CHECK(z.n_elevation() == 19);
    CHECK(z.n_azimuth() == 37);
    CHECK(z.is_valid(false).empty());
}

// B-101: copy_element must respect read_only and reject an empty destination with a clear message.
TEST_CASE("Arrayant copy_element - Read-only object and empty destination (B-101)")
{
    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.read_only = true;
    CHECK_THROWS_AS(a.copy_element(0, 3), std::invalid_argument);
    CHECK(a.n_elements() == 1);
    a.read_only = false;

    try
    {
        a.copy_element(0, arma::uvec());
        FAIL("no exception for an empty destination");
    }
    catch (const std::invalid_argument &e)
    {
        CHECK(std::string(e.what()).find("destination") != std::string::npos);
    }
    CHECK(a.n_elements() == 1);

    REQUIRE_NOTHROW(a.copy_element(0, 2));
    CHECK(a.n_elements() == 3);
}
