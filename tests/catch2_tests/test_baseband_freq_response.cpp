// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_channel.hpp"
#include <iostream>
#include <cmath>
#include <complex>

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
    arma::uvec i_snap = {1, 2, 1, 0};

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

TEST_CASE("Baseband Frequency Response - Long delays with the phase in double precision (B-020)")
{
    // One path, coefficient 1, bandwidth 100 MHz, tau = 0.12 s: H = exp(-j * 2 * pi * B * p * tau)
    const double B = 1.0e8, tau = 0.12;
    arma::cube cr(1, 1, 1, arma::fill::ones), ci(1, 1, 1, arma::fill::zeros), dl(1, 1, 1);
    dl(0) = tau;
    arma::vec pg = {0.5, 1.0};
    arma::cx_cube h;
    quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl, &pg, B, nullptr, nullptr, &h);
    for (arma::uword k = 0; k < 2; ++k)
    {
        std::complex<double> ref = std::polar(1.0, -2.0 * arma::datum::pi * B * pg(k) * tau);
        CHECK(std::abs(h(k) - ref) < 1e-4);
    }

    // Float version: delays and coefficients in single precision
    arma::fcube crf(1, 1, 1, arma::fill::ones), cif(1, 1, 1, arma::fill::zeros), dlf(1, 1, 1);
    dlf(0) = 0.125f; // exact in float
    arma::fvec pgf = {0.5f, 1.0f};
    arma::cx_fcube hf;
    quadriga_lib::baseband_freq_response<float>(&crf, &cif, &dlf, &pgf, B, nullptr, nullptr, &hf);
    for (arma::uword k = 0; k < 2; ++k)
    {
        std::complex<double> ref = std::polar(1.0, -2.0 * arma::datum::pi * B * (double)pgf(k) * 0.125);
        CHECK(std::abs(std::complex<double>(hf(k)) - ref) < 1e-4);
    }
}

TEST_CASE("Baseband Frequency Response - Delays up to 10 s against a double-precision reference (B-020)")
{
    // 2x2 MIMO with per-antenna delays from 1 us to 10 s, 64 carriers over 100 MHz. The reference evaluates
    // exp(-j * 2 * pi * x) with x = (-B * p) * tau in double, reduced to x - round(x), as the library does;
    // the library then evaluates sin / cos in single precision. Each build (AVX2 and generic) stays within
    // 5e-7 of the reference, so both agree to 1e-6.
    const double B = 1.0e8;
    arma::uword n_rx = 2, n_tx = 2, n_path = 4, n_ant = n_rx * n_tx;
    arma::cube cr(n_rx, n_tx, n_path), ci(n_rx, n_tx, n_path), dl(n_rx, n_tx, n_path);
    arma::vec amp = {0.4, 0.3, 0.2, 0.1}, tau = {1.0e-6, 1.0e-3, 0.5, 10.0};
    for (arma::uword p = 0; p < n_path; ++p)
        for (arma::uword a = 0; a < n_ant; ++a)
        {
            double ph = 0.7 * (double)(p + 1) + 0.3 * (double)a;
            cr(a + p * n_ant) = amp(p) * std::cos(ph);
            ci(a + p * n_ant) = amp(p) * std::sin(ph);
            dl(a + p * n_ant) = tau(p) * (1.0 + 1.0e-7 * (double)a) + 3.3e-9 * (double)a;
        }
    arma::vec pg = arma::linspace<arma::vec>(0.0, 1.0, 64);
    pg(17) = 0.123456789, pg(40) = 0.987654321;

    arma::cx_cube h;
    quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl, &pg, B, nullptr, nullptr, &h);
    REQUIRE(h.n_slices == pg.n_elem);

    double err = 0.0;
    for (arma::uword k = 0; k < pg.n_elem; ++k)
        for (arma::uword a = 0; a < n_ant; ++a)
        {
            std::complex<double> ref(0.0, 0.0);
            for (arma::uword p = 0; p < n_path; ++p)
            {
                double x = (-B * pg(k)) * dl(a + p * n_ant);
                x -= std::round(x);
                ref += std::complex<double>(cr(a + p * n_ant), ci(a + p * n_ant)) * std::polar(1.0, 2.0 * arma::datum::pi * x);
            }
            err = std::max(err, std::abs(h(a + k * n_ant) - ref));
        }
    CHECK(err < 5.0e-7);

    // Planar-wave delays [1, 1, n_path] give the same result as identical per-antenna delays
    arma::cube dl1(1, 1, n_path), dlx(n_rx, n_tx, n_path);
    for (arma::uword p = 0; p < n_path; ++p)
        dl1(p) = tau(p), dlx.slice(p).fill(tau(p));
    arma::cx_cube h1, hx;
    quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl1, &pg, B, nullptr, nullptr, &h1);
    quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dlx, &pg, B, nullptr, nullptr, &hx);
    CHECK(arma::approx_equal(h1, hx, "absdiff", 1e-12));

    // Non-finite delays give NaN (as before); a phase beyond 2^50 cycles (1e8 s at 100 MHz) stays finite
    arma::cube c1(1, 1, 1, arma::fill::ones), c0(1, 1, 1, arma::fill::zeros), d1(1, 1, 1);
    arma::vec pg2 = {0.25, 0.5, 1.0};
    for (double d : {arma::datum::nan, arma::datum::inf, 1.0e8})
    {
        d1(0) = d;
        arma::cx_cube hn;
        quadriga_lib::baseband_freq_response<double>(&c1, &c0, &d1, &pg2, B, nullptr, nullptr, &hn);
        for (arma::uword k = 0; k < 3; ++k)
            if (std::isfinite(d))
                CHECK(std::abs(std::abs(hn(k)) - 1.0) < 1e-6);
            else
                CHECK(std::isnan(hn(k).real()));
    }
}

TEST_CASE("Baseband Frequency Response - Carrier counts that are not a multiple of 8, unaligned buffers")
{
    // 7, 13 and 1021 carriers (padding path of the AVX2 kernel; the internal buffers are ordinary, unaligned
    // allocations), float and double, the three output forms, and a call without outputs
    const double B = 2.0e7;
    arma::uword n_rx = 3, n_tx = 2, n_path = 5, n_ant = n_rx * n_tx;
    arma::cube cr(n_rx, n_tx, n_path), ci(n_rx, n_tx, n_path), dl(n_rx, n_tx, n_path);
    for (arma::uword i = 0; i < cr.n_elem; ++i)
        cr(i) = std::cos(0.37 * (double)i), ci(i) = std::sin(0.91 * (double)i), dl(i) = 1.0e-7 * (double)(i % 17) + 3.0e-9;

    for (arma::uword n_carrier : {7ULL, 13ULL, 1021ULL})
    {
        arma::vec pg = arma::linspace<arma::vec>(0.0, 1.0, n_carrier);
        arma::cube hr, hi;
        arma::cx_cube h;
        quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl, &pg, B, &hr, &hi, &h);
        REQUIRE(h.n_slices == n_carrier);
        REQUIRE(hr.n_slices == n_carrier);
        CHECK(arma::approx_equal(hr, arma::cube(arma::real(h)), "absdiff", 0.0));
        CHECK(arma::approx_equal(hi, arma::cube(arma::imag(h)), "absdiff", 0.0));

        double err = 0.0;
        for (arma::uword k = 0; k < n_carrier; ++k)
            for (arma::uword a = 0; a < n_ant; ++a)
            {
                std::complex<double> ref(0.0, 0.0);
                for (arma::uword p = 0; p < n_path; ++p)
                {
                    double x = (-B * pg(k)) * dl(a + p * n_ant);
                    x -= std::round(x);
                    ref += std::complex<double>(cr(a + p * n_ant), ci(a + p * n_ant)) * std::polar(1.0, 2.0 * arma::datum::pi * x);
                }
                err = std::max(err, std::abs(h(a + k * n_ant) - ref));
            }
        INFO("n_carrier = " << n_carrier);
        CHECK(err < 2.0e-6);

        // Float inputs with the same carrier count
        arma::fcube crf = arma::conv_to<arma::fcube>::from(cr), cif = arma::conv_to<arma::fcube>::from(ci);
        arma::fcube dlf = arma::conv_to<arma::fcube>::from(dl), hrf;
        arma::fvec pgf = arma::conv_to<arma::fvec>::from(pg);
        quadriga_lib::baseband_freq_response<float>(&crf, &cif, &dlf, &pgf, B, &hrf, nullptr);
        REQUIRE(hrf.n_slices == n_carrier);
        CHECK(hrf.is_finite());

        // No output requested: returns without computing
        quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl, &pg, B, nullptr, nullptr, nullptr);
    }
}
