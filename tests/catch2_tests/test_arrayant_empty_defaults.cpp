// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_test_arrayant.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// An empty element_pos stands for zeros, an empty coupling_re for identity and an empty coupling_im for zeros. Every
// function that takes an arrayant must give the same result for the empty members as for the explicit values.

// Three-element arrayant on a 30 degree grid with distinct, deterministic patterns; element_pos, coupling_re and
// coupling_im filled with zeros, identity and zeros
static quadriga_lib::arrayant<double> filled_arrayant(double center_frequency = 3.0e9)
{
    const double d2r = arma::datum::pi / 180.0;
    quadriga_lib::arrayant<double> a;
    a.name = "test";
    a.azimuth_grid = arma::regspace<arma::vec>(-180.0, 30.0, 180.0) * d2r;
    a.elevation_grid = arma::regspace<arma::vec>(-90.0, 30.0, 90.0) * d2r;
    const arma::uword n_el = a.elevation_grid.n_elem, n_az = a.azimuth_grid.n_elem, n = 3;
    a.e_theta_re.set_size(n_el, n_az, n), a.e_theta_im.set_size(n_el, n_az, n);
    a.e_phi_re.set_size(n_el, n_az, n), a.e_phi_im.set_size(n_el, n_az, n);
    for (arma::uword s = 0; s < n; ++s)
        for (arma::uword j = 0; j < n_az; ++j)
            for (arma::uword i = 0; i < n_el; ++i)
            {
                const double az = a.azimuth_grid(j), el = a.elevation_grid(i), k = double(s + 1);
                a.e_theta_re(i, j, s) = std::cos(el) * (1.0 + 0.5 * std::cos(az - k));
                a.e_theta_im(i, j, s) = 0.3 * std::sin(k * az) * std::cos(el);
                a.e_phi_re(i, j, s) = 0.2 * k * std::sin(el + az);
                a.e_phi_im(i, j, s) = 0.1 * std::cos(k * el);
            }
    a.element_pos.zeros(3, n);
    a.coupling_re.eye(n, n);
    a.coupling_im.zeros(n, n);
    a.center_frequency = center_frequency;
    return a;
}

static quadriga_lib::arrayant<double> emptied(quadriga_lib::arrayant<double> a)
{
    a.element_pos.reset(), a.coupling_re.reset(), a.coupling_im.reset();
    return a;
}

// Element positions and coupling with the defaults written out
static arma::mat pos_of(const quadriga_lib::arrayant<double> &a)
{
    return a.element_pos.empty() ? arma::mat(3, a.n_elements(), arma::fill::zeros) : a.element_pos;
}
static arma::mat cpl_re_of(const quadriga_lib::arrayant<double> &a)
{
    return a.coupling_re.empty() ? arma::mat(a.n_elements(), a.n_elements(), arma::fill::eye) : a.coupling_re;
}
static arma::mat cpl_im_of(const quadriga_lib::arrayant<double> &a)
{
    return a.coupling_im.empty() ? arma::mat(a.n_elements(), a.n_ports(), arma::fill::zeros) : a.coupling_im;
}

static void check_same(const quadriga_lib::arrayant<double> &a, const quadriga_lib::arrayant<double> &b)
{
    CHECK(a.is_valid().empty());
    CHECK(b.is_valid().empty());
    CHECK(arma::approx_equal(a.azimuth_grid, b.azimuth_grid, "absdiff", 1e-14));
    CHECK(arma::approx_equal(a.elevation_grid, b.elevation_grid, "absdiff", 1e-14));
    CHECK(arma::approx_equal(a.e_theta_re, b.e_theta_re, "absdiff", 1e-12));
    CHECK(arma::approx_equal(a.e_theta_im, b.e_theta_im, "absdiff", 1e-12));
    CHECK(arma::approx_equal(a.e_phi_re, b.e_phi_re, "absdiff", 1e-12));
    CHECK(arma::approx_equal(a.e_phi_im, b.e_phi_im, "absdiff", 1e-12));
    CHECK(arma::approx_equal(pos_of(a), pos_of(b), "absdiff", 1e-12));
    CHECK(arma::approx_equal(cpl_re_of(a), cpl_re_of(b), "absdiff", 1e-14));
    CHECK(arma::approx_equal(cpl_im_of(a), cpl_im_of(b), "absdiff", 1e-14));
    CHECK(a.n_ports() == b.n_ports());
}

static std::string file_text(const std::string &fn)
{
    std::ifstream f(fn);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

TEST_CASE("Arrayant functions - Empty element_pos and coupling equal zeros and identity")
{
    const auto F = filled_arrayant();
    const auto E = emptied(F);
    REQUIRE(E.is_valid().empty());

    SECTION("arrayant_interpolate")
    {
        const arma::mat az = {{0.3, -1.2, 2.5}}, el = {{0.1, -0.4, 0.7}};
        const arma::cube ori(3, 1, 1, arma::fill::value(0.2));
        arma::mat vr[2], vi[2], hr[2], hi[2], d[2];
        quadriga_lib::arrayant_interpolate<double>(F, az, el, vr[0], vi[0], hr[0], hi[0], nullptr, &ori, nullptr, &d[0]);
        quadriga_lib::arrayant_interpolate<double>(E, az, el, vr[1], vi[1], hr[1], hi[1], nullptr, &ori, nullptr, &d[1]);
        CHECK(arma::approx_equal(vr[0], vr[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(vi[0], vi[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(hr[0], hr[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(hi[0], hi[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(d[0], d[1], "absdiff", 0.0));
    }

    SECTION("arrayant_combine_pattern")
    {
        check_same(quadriga_lib::arrayant_combine_pattern(F), quadriga_lib::arrayant_combine_pattern(E));
    }

    SECTION("arrayant_concat")
    {
        check_same(quadriga_lib::arrayant_concat(F, F), quadriga_lib::arrayant_concat(E, E));
        check_same(quadriga_lib::arrayant_concat(F, F), quadriga_lib::arrayant_concat(F, E));
        check_same(quadriga_lib::arrayant_concat(F, F), quadriga_lib::arrayant_concat(E, F));
    }

    SECTION("arrayant_copy_element")
    {
        // Growing: empty members are written out, so that the new elements can be edited
        auto f = F, e = E;
        quadriga_lib::arrayant_copy_element(f, 1, {0, 4});
        quadriga_lib::arrayant_copy_element(e, 1, {0, 4});
        check_same(f, e);
        CHECK(arma::size(e.element_pos) == arma::size(3, 5));
        CHECK(arma::size(e.coupling_re) == arma::size(5, 5));
        CHECK(arma::size(e.coupling_im) == arma::size(5, 5));
        CHECK(e.n_ports() == 5);
        e.element_pos(1, 4) = 0.5, e.coupling_re(4, 4) = 0.7; // Editing a new element works

        // Coupling given, coupling_im empty: the new elements get their own port
        f = F, e = F, e.coupling_im.reset();
        quadriga_lib::arrayant_copy_element(f, 2, {3});
        quadriga_lib::arrayant_copy_element(e, 2, {3});
        check_same(f, e);

        // Not growing: empty members stay empty
        f = F, e = E;
        quadriga_lib::arrayant_copy_element(f, 0, {1, 2});
        quadriga_lib::arrayant_copy_element(e, 0, {1, 2});
        check_same(f, e);
        CHECK(e.element_pos.empty());
        CHECK(e.coupling_re.empty());
    }

    SECTION("arrayant_remove_zeros")
    {
        auto f = F, e = E;
        f.e_theta_re.cols(0, 3).zeros(), f.e_theta_im.cols(0, 3).zeros(), f.e_phi_re.cols(0, 3).zeros(), f.e_phi_im.cols(0, 3).zeros();
        e.e_theta_re = f.e_theta_re, e.e_theta_im = f.e_theta_im, e.e_phi_re = f.e_phi_re, e.e_phi_im = f.e_phi_im;
        quadriga_lib::arrayant_remove_zeros(f);
        quadriga_lib::arrayant_remove_zeros(e);
        CHECK(f.n_azimuth() < F.n_azimuth());
        check_same(f, e);
        CHECK(e.element_pos.empty());
    }

    SECTION("qdant_write and qdant_read")
    {
        const std::string fn_f = "test_empty_defaults_f.qdant", fn_e = "test_empty_defaults_e.qdant";
        std::remove(fn_f.c_str()), std::remove(fn_e.c_str());
        quadriga_lib::qdant_write(fn_f, F);
        quadriga_lib::qdant_write(fn_e, E);
        check_same(quadriga_lib::qdant_read<double>(fn_f), quadriga_lib::qdant_read<double>(fn_e));
        std::remove(fn_f.c_str()), std::remove(fn_e.c_str());
    }

    SECTION("arrayant_export_obj_file")
    {
        const std::string fn = "test_empty_defaults.obj";
        quadriga_lib::arrayant_export_obj_file(fn, F, 30.0, "jet", 1.0, 1);
        const std::string text_f = file_text(fn);
        quadriga_lib::arrayant_export_obj_file(fn, E, 30.0, "jet", 1.0, 1);
        CHECK(text_f == file_text(fn));
        std::remove(fn.c_str()), std::remove("test_empty_defaults.mtl");
    }

    SECTION("arrayant_rotate_pattern")
    {
        for (unsigned usage : {0U, 1U, 2U, 3U})
        {
            auto f = F, e = E;
            qd_rotate(f, arma::datum::pi / 18.0, arma::datum::pi / 9.0, arma::datum::pi / 6.0, usage);
            qd_rotate(e, arma::datum::pi / 18.0, arma::datum::pi / 9.0, arma::datum::pi / 6.0, usage);
            check_same(f, e);

            f = F, e = E; // Single element
            qd_rotate(f, arma::datum::pi / 18.0, arma::datum::pi / 9.0, arma::datum::pi / 6.0, usage, 1);
            qd_rotate(e, arma::datum::pi / 18.0, arma::datum::pi / 9.0, arma::datum::pi / 6.0, usage, 1);
            check_same(f, e);
        }

        // A non-uniform azimuth grid makes usage 0 resample the pattern onto a new grid
        auto fn = F;
        const arma::vec t = arma::linspace<arma::vec>(-1.0, 1.0, F.n_azimuth());
        fn.azimuth_grid = arma::sign(t) % arma::pow(arma::abs(t), 1.5) * (arma::datum::pi - 0.01);
        auto en = emptied(fn), f = fn, e = en;
        qd_rotate(f, arma::datum::pi / 18.0, arma::datum::pi / 9.0, arma::datum::pi / 6.0, 0);
        qd_rotate(e, arma::datum::pi / 18.0, arma::datum::pi / 9.0, arma::datum::pi / 6.0, 0);
        REQUIRE(f.n_azimuth() != fn.n_azimuth());
        check_same(f, e);
    }

    SECTION("arrayant_calc_beamwidth and arrayant_calc_directivity")
    {
        double bf = 0.0, be = 0.0;
        qd_beamwidth(F, 1, 3.0, &bf);
        qd_beamwidth(E, 1, 3.0, &be);
        CHECK(bf == be);
        CHECK(qd_directivity(F, 2) == qd_directivity(E, 2));
    }
}

TEST_CASE("Arrayant multi-frequency functions - Empty element_pos and coupling equal zeros and identity")
{
    const std::vector<quadriga_lib::arrayant<double>> F = {filled_arrayant(1.0e9), filled_arrayant(2.0e9)};
    const std::vector<quadriga_lib::arrayant<double>> E = {emptied(F[0]), emptied(F[1])};
    REQUIRE(quadriga_lib::arrayant_is_valid_multi(E).empty());

    auto check_same_multi = [](const std::vector<quadriga_lib::arrayant<double>> &a, const std::vector<quadriga_lib::arrayant<double>> &b)
    {
        REQUIRE(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i)
            check_same(a[i], b[i]);
    };

    SECTION("arrayant_combine_pattern_multi")
    {
        check_same_multi(quadriga_lib::arrayant_combine_pattern_multi(F), quadriga_lib::arrayant_combine_pattern_multi(E));
    }

    SECTION("arrayant_concat_multi")
    {
        check_same_multi(quadriga_lib::arrayant_concat_multi(F, F), quadriga_lib::arrayant_concat_multi(E, F));
    }

    SECTION("arrayant_copy_element_multi")
    {
        auto f = F, e = E;
        quadriga_lib::arrayant_copy_element_multi(f, 0, {3});
        quadriga_lib::arrayant_copy_element_multi(e, 0, {3});
        check_same_multi(f, e);
    }

    SECTION("arrayant_rotate_pattern_multi")
    {
        auto f = F, e = E;
        qd_rotate_multi(f, arma::datum::pi / 18.0, arma::datum::pi / 9.0, arma::datum::pi / 6.0);
        qd_rotate_multi(e, arma::datum::pi / 18.0, arma::datum::pi / 9.0, arma::datum::pi / 6.0);
        check_same_multi(f, e);
    }

    SECTION("arrayant_set_element_pos_multi")
    {
        auto f = F, e = E;
        const arma::mat p = arma::vec{0.1, 0.2, 0.3};
        quadriga_lib::arrayant_set_element_pos_multi(f, p, arma::uvec{1});
        quadriga_lib::arrayant_set_element_pos_multi(e, p, arma::uvec{1});
        check_same_multi(f, e);
    }

    SECTION("arrayant_interpolate_multi")
    {
        const arma::mat az = {{0.3, -1.2}}, el = {{0.1, -0.4}};
        const arma::vec freq = {1.5e9};
        arma::cube vr[2], vi[2], hr[2], hi[2];
        quadriga_lib::arrayant_interpolate_multi(F, az, el, vr[0], vi[0], hr[0], hi[0], &freq);
        quadriga_lib::arrayant_interpolate_multi(E, az, el, vr[1], vi[1], hr[1], hi[1], &freq);
        CHECK(arma::approx_equal(vr[0], vr[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(vi[0], vi[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(hr[0], hr[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(hi[0], hi[1], "absdiff", 0.0));
    }

    SECTION("qdant_write_multi and qdant_read_multi")
    {
        const std::string fn_f = "test_empty_defaults_mf.qdant", fn_e = "test_empty_defaults_me.qdant";
        quadriga_lib::qdant_write_multi(fn_f, F);
        quadriga_lib::qdant_write_multi(fn_e, E);
        check_same_multi(quadriga_lib::qdant_read_multi<double>(fn_f), quadriga_lib::qdant_read_multi<double>(fn_e));
        std::remove(fn_f.c_str()), std::remove(fn_e.c_str());
    }
}

// The channel functions with empty members on one or both sides, also with a coupling matrix on the other side
TEST_CASE("Channel functions - Empty element_pos and coupling equal zeros and identity")
{
    auto F = filled_arrayant();
    F.element_pos = {{0.0, 0.05, -0.05}, {0.1, -0.1, 0.0}, {0.0, 0.02, 0.0}}; // Positions stay given
    auto E = F;
    E.coupling_re.reset(), E.coupling_im.reset();

    // Array with two ports: element 0 and the phased sum of elements 1 and 2
    auto C = F;
    C.coupling_re = {{1.0, 0.0}, {0.0, 0.6}, {0.0, 0.5}};
    C.coupling_im = {{0.0, 0.0}, {0.0, 0.3}, {0.0, -0.2}};

    // Positions only: empty vs zeros
    auto P0 = filled_arrayant(), PE = emptied(filled_arrayant());
    PE.coupling_re = P0.coupling_re, PE.coupling_im = P0.coupling_im;

    const arma::mat fbs = {{10.0, 5.0}, {0.0, 8.0}, {1.0, 2.0}};
    const arma::vec gain = {1.0, 0.25}, len = {20.0, 22.0};
    arma::mat M(8, 2, arma::fill::zeros);
    M.row(0).fill(1.0), M.row(6).fill(-1.0), M(2, 1) = 0.3;

    // (TX, RX) pairs that must give the same channel
    struct Case
    {
        const quadriga_lib::arrayant<double> *tx_a, *rx_a, *tx_b, *rx_b;
    };
    const std::vector<Case> cases = {{&F, &F, &E, &E}, {&C, &F, &C, &E}, {&F, &C, &E, &C}, {&P0, &P0, &PE, &PE}, {&C, &P0, &C, &PE}};

    for (const auto &c : cases)
    {
        // Spherical waves
        arma::cube re[2], im[2], dl[2];
        for (int k = 0; k < 2; ++k)
            quadriga_lib::get_channels_spherical(k == 0 ? c.tx_a : c.tx_b, k == 0 ? c.rx_a : c.rx_b,
                                                 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0, 0.0, 3.0,
                                                 &fbs, &fbs, &gain, &len, &M, &re[k], &im[k], &dl[k], 3.0e9);
        REQUIRE(arma::abs(re[0]).max() > 0.01);
        CHECK(arma::approx_equal(re[0], re[1], "absdiff", 1e-12));
        CHECK(arma::approx_equal(im[0], im[1], "absdiff", 1e-12));
        CHECK(arma::approx_equal(dl[0], dl[1], "absdiff", 1e-15));

        // Planar waves
        const arma::vec aod = {0.0, 0.6}, eod = {0.0, 0.1}, aoa = {3.0, -2.5}, eoa = {0.0, -0.1};
        for (int k = 0; k < 2; ++k)
            quadriga_lib::get_channels_planar(k == 0 ? c.tx_a : c.tx_b, k == 0 ? c.rx_a : c.rx_b,
                                              0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0, 0.0, 3.0,
                                              &aod, &eod, &aoa, &eoa, &gain, &len, &M, &re[k], &im[k], &dl[k], 3.0e9);
        REQUIRE(arma::abs(re[0]).max() > 0.01);
        CHECK(arma::approx_equal(re[0], re[1], "absdiff", 1e-12));
        CHECK(arma::approx_equal(im[0], im[1], "absdiff", 1e-12));
        CHECK(arma::approx_equal(dl[0], dl[1], "absdiff", 1e-15));

        // Multi-frequency (two entries with the same pattern at 2 and 4 GHz)
        std::vector<arma::cube> mre[2], mim[2], mdl[2];
        const arma::vec freq_in = {2.0e9, 4.0e9}, freq_out = {3.0e9};
        const arma::mat gain_f = arma::join_horiz(gain, gain);
        const arma::cube M_f = arma::join_slices(arma::cube(M.memptr(), 8, 2, 1), arma::cube(M.memptr(), 8, 2, 1));
        for (int k = 0; k < 2; ++k)
        {
            auto tx = *(k == 0 ? c.tx_a : c.tx_b), rx = *(k == 0 ? c.rx_a : c.rx_b);
            auto tx2 = tx, rx2 = rx;
            tx.center_frequency = rx.center_frequency = 2.0e9, tx2.center_frequency = rx2.center_frequency = 4.0e9;
            quadriga_lib::get_channels_multifreq<double>({tx, tx2}, {rx, rx2},
                                                         0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0, 0.0, 3.0,
                                                         fbs, fbs, gain_f, len, M_f, freq_in, freq_out, mre[k], mim[k], mdl[k]);
        }
        REQUIRE(arma::abs(mre[0][0]).max() > 0.01);
        CHECK(arma::approx_equal(mre[0][0], mre[1][0], "absdiff", 1e-12));
        CHECK(arma::approx_equal(mim[0][0], mim[1][0], "absdiff", 1e-12));
        CHECK(arma::approx_equal(mdl[0][0], mdl[1][0], "absdiff", 1e-15));
    }

    // IRS: TX and RX with empty members, the IRS with empty element_pos (its coupling holds the IRS weights)
    {
        auto irs_a = C, irs_b = C;
        irs_a.element_pos.zeros(3, 3), irs_b.element_pos.reset();
        const arma::mat fbs_1 = arma::vec{5.0, 5.0, 1.0}, fbs_2 = arma::vec{15.0, 5.0, 1.0};
        const double l = std::sqrt(50.0) + 5.0; // TX -> scatterer -> IRS and IRS -> scatterer -> RX
        const arma::vec g1 = {1.0}, l1 = {l}, g2 = {1.0}, l2 = {l};
        const arma::mat M1 = M.col(0), M2 = M.col(0);
        arma::cube re[2], im[2], dl[2];
        for (int k = 0; k < 2; ++k)
            quadriga_lib::get_channels_irs<double>(k == 0 ? &P0 : &PE, k == 0 ? &F : &E, k == 0 ? &irs_a : &irs_b,
                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                   10.0, 5.0, 1.0, 0.0, 0.0, -1.57,
                                                   &fbs_1, &fbs_1, &g1, &l1, &M1, &fbs_2, &fbs_2, &g2, &l2, &M2,
                                                   &re[k], &im[k], &dl[k], 1, -200.0, 3.0e9);
        REQUIRE(arma::abs(re[0]).max() > 1e-6);
        CHECK(arma::approx_equal(re[0], re[1], "absdiff", 1e-12));
        CHECK(arma::approx_equal(im[0], im[1], "absdiff", 1e-12));
        CHECK(arma::approx_equal(dl[0], dl[1], "absdiff", 1e-15));
    }
}
