// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for arrayant_interpolate_multi.
//
// The suite pins the frequency dimension: which entries bracket a target frequency, the blending between them,
// clamping outside the entries, the default frequencies, and that the order of the query frequencies (which drives
// the reuse of spatial results) and of the entries does not change the result. The spatial interpolation is pinned
// by test_arrayant_interpolate.cpp; here it serves as the oracle at the center frequencies of the entries, where the
// result must be identical to arrayant_interpolate of that entry.
//
// Oracles:
//  - Between two entries the field is blended spherically: amplitude and phase linear in the frequency between the
//    two center frequencies, linear interpolation for opposite phases or a zero amplitude. Single-sample entries give
//    hand values; entries that are positive multiples of one pattern blend linearly in amplitude.
//  - Loudspeaker models from generate_speaker: a piston becomes more directional with frequency; a woofer dominates
//    a tweeter at low frequencies and the other way round.
//
// Conventions: frequencies in Hz; entries are built with their center frequency and need not be sorted.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_test_diag.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

QD_TEST_DIAG_SWITCH("QD_DIAG_ANT")

namespace
{
    constexpr double pi = 3.14159265358979323846;
    using cxd = std::complex<double>;
    using ant_vec = std::vector<quadriga_lib::arrayant<double>>;

    std::vector<int> kernels()
    {
        std::vector<int> k = {1};
        if (quadriga_lib::quadriga_lib_has_AVX2())
            k.push_back(2);
        return k;
    }

    template <typename T>
    bool identical(const T &a, const T &b)
    {
        return arma::size(a) == arma::size(b) && std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(*a.memptr())) == 0;
    }

    struct qd_multi_result
    {
        arma::cube V_re, V_im, H_re, H_im;
    };

    qd_multi_result run(const ant_vec &vec, const arma::mat &az, const arma::mat &el, const arma::vec *freq,
                        const arma::uvec *i_element = nullptr, const arma::cube *ori = nullptr, int kernel = 1)
    {
        qd_multi_result r;
        quadriga_lib::arrayant_interpolate_multi<double>(vec, az, el, r.V_re, r.V_im, r.H_re, r.H_im, freq, i_element, ori, nullptr, kernel);
        return r;
    }

    // True if slice s of r equals the single-frequency interpolation of ant
    bool slice_equals(const qd_multi_result &r, arma::uword s, const quadriga_lib::arrayant<double> &ant, const arma::mat &az,
                      const arma::mat &el, const arma::uvec *i_element = nullptr, const arma::cube *ori = nullptr, int kernel = 1)
    {
        arma::mat V_re, V_im, H_re, H_im;
        quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, i_element, ori, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, kernel);
        return arma::any(arma::vectorise(V_re) != 0.0) && // Not a trivial (zero) result
               identical(arma::mat(r.V_re.slice(s)), V_re) && identical(arma::mat(r.V_im.slice(s)), V_im) &&
               identical(arma::mat(r.H_re.slice(s)), H_re) && identical(arma::mat(r.H_im.slice(s)), H_im);
    }

    bool identical(const qd_multi_result &a, const qd_multi_result &b)
    {
        return identical(a.V_re, b.V_re) && identical(a.V_im, b.V_im) && identical(a.H_re, b.H_re) && identical(a.H_im, b.H_im);
    }

    // Entry with one element and one sample (constant pattern) of e-theta V and e-phi H
    quadriga_lib::arrayant<double> const_entry(cxd V, cxd H, double freq)
    {
        quadriga_lib::arrayant<double> ant;
        ant.e_theta_re.set_size(1, 1, 1), ant.e_theta_im.set_size(1, 1, 1);
        ant.e_phi_re.set_size(1, 1, 1), ant.e_phi_im.set_size(1, 1, 1);
        ant.e_theta_re(0) = V.real(), ant.e_theta_im(0) = V.imag(), ant.e_phi_re(0) = H.real(), ant.e_phi_im(0) = H.imag();
        ant.azimuth_grid = {0.0}, ant.elevation_grid = {0.0};
        ant.center_frequency = freq;
        return ant;
    }

    // Entries that are positive multiples scale[i] of one dual-polarized 4-element panel (10 degree grid), at the
    // center frequencies cf[i]; the blend of two of them is linear in the amplitude
    ant_vec scaled_entries(const arma::vec &cf, const arma::vec &scale)
    {
        const auto base = quadriga_lib::generate_arrayant_3GPP<double>(1, 2, 3.0e9, 3, 0.0, 0.5, 1, 1, 0.5, 0.5, nullptr, 10.0);
        ant_vec vec;
        for (arma::uword i = 0; i < cf.n_elem; ++i)
        {
            auto ant = base;
            ant.e_theta_re *= scale(i), ant.e_theta_im *= scale(i), ant.e_phi_re *= scale(i), ant.e_phi_im *= scale(i);
            ant.center_frequency = cf(i);
            vec.push_back(ant);
        }
        return vec;
    }

    // Blending oracle between two complex values at weight w in [0, 1]: linear for opposite phases or a zero
    // amplitude, otherwise amplitude and phase linear (shorter arc)
    cxd blend_oracle(cxd A, cxd B, double w)
    {
        const double dphi = std::remainder(std::arg(B) - std::arg(A), 2.0 * pi);
        if (std::abs(A) == 0.0 || std::abs(B) == 0.0 || std::abs(std::abs(dphi) - pi) < 1e-12)
            return (1.0 - w) * A + w * B;
        return std::polar((1.0 - w) * std::abs(A) + w * std::abs(B), std::arg(A) + w * dphi);
    }
} // namespace

TEST_CASE("arrayant_interpolate_multi - the center frequencies give the single-frequency result")
{
    const ant_vec vec = scaled_entries({2.0e9, 1.0e9, 3.0e9}, {1.0, 0.5, 2.0}); // Not sorted by frequency
    const arma::mat az = {{0.1, -0.7, 2.0, 3.0}}, el = {{0.0, 0.3, -0.4, 1.2}};
    const arma::vec freq = {1.0e9, 3.0e9, 2.0e9};
    for (int kernel : kernels())
    {
        INFO("kernel " << kernel);
        const auto r = run(vec, az, el, &freq, nullptr, nullptr, kernel);
        REQUIRE(r.V_re.n_slices == 3);
        CHECK(slice_equals(r, 0, vec[1], az, el, nullptr, nullptr, kernel));
        CHECK(slice_equals(r, 1, vec[2], az, el, nullptr, nullptr, kernel));
        CHECK(slice_equals(r, 2, vec[0], az, el, nullptr, nullptr, kernel));
    }
}

TEST_CASE("arrayant_interpolate_multi - frequencies outside the entries are clamped")
{
    const ant_vec vec = scaled_entries({1.0e9, 2.0e9, 3.0e9}, {1.0, 2.0, 3.0});
    const arma::mat az = {{0.2, 1.0}}, el = {{0.1, -0.2}};
    const arma::vec freq = {1.0e3, 0.5e9, 3.5e9, 1.0e12, -1.0e9};
    const auto r = run(vec, az, el, &freq);
    CHECK(slice_equals(r, 0, vec[0], az, el));
    CHECK(slice_equals(r, 1, vec[0], az, el));
    CHECK(slice_equals(r, 2, vec[2], az, el));
    CHECK(slice_equals(r, 3, vec[2], az, el));
    CHECK(slice_equals(r, 4, vec[0], az, el));

    // A single entry gives its pattern at every frequency
    const ant_vec one = {vec[1]};
    const arma::vec freq1 = {1.0, 2.0e9, 1.0e15};
    const auto r1 = run(one, az, el, &freq1);
    for (arma::uword s = 0; s < 3; ++s)
        CHECK(slice_equals(r1, s, vec[1], az, el));
}

TEST_CASE("arrayant_interpolate_multi - blending between two entries is spherical in amplitude and phase")
{
    // Pairs of single-sample entries (V, H) at 1 and 2 kHz; target frequencies inside the pair
    struct qd_pair
    {
        cxd VA, HA, VB, HB;
        const char *what;
    };
    const double p3 = pi / 3.0;
    const qd_pair pairs[] = {{{1.0, 0.0}, {4.0, 0.0}, {3.0, 0.0}, {2.0, 0.0}, "in phase"},
                             {{1.0, 0.0}, {2.0, 0.0}, {0.0, 1.0}, {0.0, 4.0}, "90 degree phase step"},
                             {{2.0, 0.0}, {-2.0, 0.0}, {2.0 * std::cos(p3), 2.0 * std::sin(p3)}, {0.0, -2.0}, "60 and 90 degree steps"},
                             {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, "zero amplitude"},
                             {{1.0, 0.0}, {0.5, 0.5}, {-1.0, 0.0}, {-0.5, -0.5}, "opposite phases"}};
    const arma::vec freq = arma::linspace(1000.0, 2000.0, 11);
    const arma::mat az(1, 1, arma::fill::zeros), el(1, 1, arma::fill::zeros);

    QD_DIAG("\n# diag: blending of two single-sample entries, largest deviation from the oracle\n");
    QD_DIAG("pair,d_V,d_H\n");
    for (const auto &p : pairs)
    {
        INFO(p.what);
        const ant_vec vec = {const_entry(p.VA, p.HA, 1000.0), const_entry(p.VB, p.HB, 2000.0)};
        const auto r = run(vec, az, el, &freq);
        double dV = 0.0, dH = 0.0;
        for (arma::uword f = 0; f < freq.n_elem; ++f)
        {
            const double w = (freq(f) - 1000.0) / 1000.0;
            dV = std::max(dV, std::abs(cxd(r.V_re(0, 0, f), r.V_im(0, 0, f)) - blend_oracle(p.VA, p.VB, w)));
            dH = std::max(dH, std::abs(cxd(r.H_re(0, 0, f), r.H_im(0, 0, f)) - blend_oracle(p.HA, p.HB, w)));
        }
        QD_DIAG("%s,%.3e,%.3e\n", p.what, dV, dH);
        CHECK(dV < 1e-12);
        CHECK(dH < 1e-12);
    }

    // Both entries zero: zero
    const ant_vec zero = {const_entry(0.0, 0.0, 1000.0), const_entry(0.0, 0.0, 2000.0)};
    const auto r = run(zero, az, el, &freq);
    CHECK(arma::all(arma::vectorise(arma::abs(r.V_re) + arma::abs(r.V_im) + arma::abs(r.H_re) + arma::abs(r.H_im)) == 0.0));
}

TEST_CASE("arrayant_interpolate_multi - the neighbouring entries are found for every frequency")
{
    // 20 entries with scales 1 ... 20 at frequencies spread from 100 Hz to 10 kHz, stored in reversed order; the
    // result at any frequency is the base pattern times the linearly interpolated scale, clamped at both ends
    const arma::vec cf = arma::reverse(arma::linspace(100.0, 10000.0, 20));
    const arma::vec scale = arma::reverse(arma::linspace(1.0, 20.0, 20));
    const ant_vec vec = scaled_entries(cf, scale);
    const arma::mat az = {{0.3, -1.1, 2.5}}, el = {{0.2, -0.5, 0.0}};
    const arma::vec freq = arma::linspace(50.0, 15000.0, 100);

    arma::mat V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate<double>(vec.back(), az, el, V_re, V_im, H_re, H_im); // Scale 1
    CHECK(arma::abs(V_re).max() > 0.1);                                                      // Not a trivial (zero) result
    const auto r = run(vec, az, el, &freq);
    REQUIRE(r.V_re.n_slices == 100);
    double d = 0.0;
    for (arma::uword f = 0; f < freq.n_elem; ++f)
    {
        const double s = 1.0 + 19.0 * std::max(0.0, std::min(1.0, (freq(f) - 100.0) / 9900.0));
        d = std::max(d, arma::abs(arma::mat(r.V_re.slice(f)) - s * V_re).max());
        d = std::max(d, arma::abs(arma::mat(r.H_im.slice(f)) - s * H_im).max());
    }
    QD_DIAG("\n# diag: bracketing over 20 entries, largest deviation\nd,%.3e\n", d);
    CHECK(d < 1e-12);
}

TEST_CASE("arrayant_interpolate_multi - the order of frequencies and entries does not change the result")
{
    // Query frequencies in an order that makes the reuse of spatial results go up, down and repeat; every slice
    // equals the call with that frequency alone. Shuffled entries give the same result.
    const ant_vec vec = scaled_entries({1.0e9, 2.0e9, 3.0e9, 4.0e9}, {1.0, 3.0, 2.0, 0.5});
    const arma::mat az = {{0.3, -1.1, 2.5, 0.0}}, el = {{0.2, -0.5, 0.0, 1.0}};
    const arma::vec freq = {1.2e9, 1.8e9, 2.5e9, 3.5e9, 3.9e9, 3.1e9, 2.9e9, 1.5e9, 1.5e9, 2.0e9, 4.0e9, 1.1e9, 3.0e9};
    const auto r = run(vec, az, el, &freq);
    CHECK(arma::abs(r.V_re).max() > 0.1); // Not a trivial (zero) result
    bool all_equal = true;
    for (arma::uword f = 0; f < freq.n_elem; ++f)
    {
        const arma::vec f1 = {freq(f)};
        const auto s = run(vec, az, el, &f1);
        all_equal = all_equal && identical(arma::mat(r.V_re.slice(f)), arma::mat(s.V_re.slice(0))) &&
                    identical(arma::mat(r.H_im.slice(f)), arma::mat(s.H_im.slice(0)));
    }
    CHECK(all_equal);

    const ant_vec shuffled = {vec[2], vec[0], vec[3], vec[1]};
    CHECK(identical(run(shuffled, az, el, &freq), r));
}

TEST_CASE("arrayant_interpolate_multi - element indices, orientation and angles reach the spatial interpolation")
{
    const ant_vec vec = scaled_entries({1.0e9, 2.0e9}, {1.0, 2.0});
    const arma::uvec i_element = {3, 0, 3}; // Permuted, with a duplicate
    const arma::mat az = {{0.3, -1.1}, {2.5, 0.0}, {-0.4, 1.4}}, el = {{0.2, -0.5}, {0.0, 1.0}, {-0.3, 0.6}}; // Per element
    const arma::mat az1 = az.row(0), el1 = el.row(0);                                                       // Planar
    arma::cube ori_e(3, 3, 1), ori_a(3, 1, 2);
    ori_e.slice(0) = {{0.1, -0.2, 0.3}, {0.2, 0.0, -0.1}, {-0.4, 0.5, 0.25}};
    ori_a.slice(0) = arma::mat({0.3, -0.2, 0.1}).t(), ori_a.slice(1) = arma::mat({-0.1, 0.4, -0.6}).t();
    const arma::vec freq = {1.0e9, 1.25e9, 2.0e9};

    for (int kernel : kernels())
        for (int variant = 0; variant < 3; ++variant)
        {
            INFO("kernel " << kernel << ", variant " << variant);
            const arma::mat &a = variant == 1 ? az1 : az, &e = variant == 1 ? el1 : el;
            const arma::cube *o = variant == 0 ? &ori_e : (variant == 1 ? &ori_a : nullptr);
            const auto r = run(vec, a, e, &freq, &i_element, o, kernel);
            REQUIRE(r.V_re.n_rows == 3);
            REQUIRE(r.V_re.n_cols == 2);
            CHECK(slice_equals(r, 0, vec[0], a, e, &i_element, o, kernel));
            CHECK(slice_equals(r, 2, vec[1], a, e, &i_element, o, kernel));
            // At a quarter of the way the entries (scales 1 and 2, in phase) blend to 1.25 times entry 0
            CHECK(arma::approx_equal(r.V_re.slice(1), 1.25 * r.V_re.slice(0), "absdiff", 1e-12));
            CHECK(arma::approx_equal(r.H_im.slice(1), 1.25 * r.H_im.slice(0), "absdiff", 1e-12));
        }
}

TEST_CASE("arrayant_interpolate_multi - omitted frequencies are the center frequencies in vector order")
{
    const ant_vec vec = scaled_entries({2.0e9, 1.0e9, 3.0e9}, {1.0, 2.0, 3.0});
    const arma::mat az = {{0.3, -1.1}}, el = {{0.2, -0.5}};
    const arma::vec cf = {2.0e9, 1.0e9, 3.0e9}, empty;
    const auto r = run(vec, az, el, &cf);
    CHECK(arma::abs(r.V_re).max() > 0.1); // Not a trivial (zero) result
    CHECK(identical(run(vec, az, el, nullptr), r));
    CHECK(identical(run(vec, az, el, &empty), r));
}

TEST_CASE("arrayant_interpolate_multi - empty angle input gives empty outputs")
{
    const ant_vec vec = scaled_entries({1.0e9, 2.0e9}, {1.0, 2.0});
    const arma::vec freq = {1.0e9, 1.5e9, 2.0e9};
    const arma::uvec i_element = {1, 2};
    for (arma::uword n_rows : {1, 2})
    {
        qd_multi_result r;
        r.V_re.ones(2, 2, 2), r.H_im.ones(1, 1, 1); // Stale content of another size
        quadriga_lib::arrayant_interpolate_multi<double>(vec, arma::mat(n_rows, 0), arma::mat(n_rows, 0), r.V_re, r.V_im, r.H_re, r.H_im,
                                                 &freq, &i_element);
        for (const arma::cube *c : {&r.V_re, &r.V_im, &r.H_re, &r.H_im})
        {
            CHECK(c->n_rows == 2);
            CHECK(c->n_cols == 0);
            CHECK(c->n_slices == 3);
        }
    }
}

TEST_CASE("arrayant_interpolate_multi - pre-sized outputs are written in place, others are resized and overwritten")
{
    const ant_vec vec = scaled_entries({1.0e9, 2.0e9}, {1.0, 2.0});
    const arma::mat az = {{0.3, -1.1, 0.7}}, el = {{0.2, -0.5, 0.1}};
    const arma::vec freq = {1.0e9, 1.7e9};
    const auto ref = run(vec, az, el, &freq);
    CHECK(arma::abs(ref.V_re).max() > 0.1); // Not a trivial (zero) result

    qd_multi_result r;
    std::vector<arma::cube *> outs = {&r.V_re, &r.V_im, &r.H_re, &r.H_im};
    std::vector<const double *> ptr;
    for (arma::cube *c : outs)
        c->set_size(4, 3, 2), c->fill(std::numeric_limits<double>::quiet_NaN()), ptr.push_back(c->memptr());
    quadriga_lib::arrayant_interpolate_multi<double>(vec, az, el, r.V_re, r.V_im, r.H_re, r.H_im, &freq);
    for (size_t i = 0; i < outs.size(); ++i)
        CHECK(outs[i]->memptr() == ptr[i]);
    CHECK(identical(r, ref));

    for (arma::cube *c : outs)
        c->set_size(4, 2, 3), c->fill(std::numeric_limits<double>::quiet_NaN());
    quadriga_lib::arrayant_interpolate_multi<double>(vec, az, el, r.V_re, r.V_im, r.H_re, r.H_im, &freq);
    CHECK(identical(r, ref));
}

TEST_CASE("arrayant_interpolate_multi - use_kernel selects the kernel")
{
    const ant_vec vec = scaled_entries({1.0e9, 2.0e9}, {1.0, 2.0});
    const arma::mat az = {{0.3, -1.1, 0.7}}, el = {{0.2, -0.5, 0.1}};
    const arma::vec freq = {1.0e9, 1.6e9};
    CHECK(identical(run(vec, az, el, &freq, nullptr, nullptr, 0), run(vec, az, el, &freq, nullptr, nullptr, 1)));
    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        const auto r2 = run(vec, az, el, &freq, nullptr, nullptr, 2);
        CHECK(slice_equals(r2, 0, vec[0], az, el, nullptr, nullptr, 2));
        CHECK(!slice_equals(r2, 0, vec[0], az, el, nullptr, nullptr, 1));
    }
    else
    {
        CHECK_THROWS_AS(run(vec, az, el, &freq, nullptr, nullptr, 2), std::invalid_argument);
    }
}

TEST_CASE("arrayant_interpolate_multi - float and double agree")
{
    const ant_vec vec = scaled_entries({1.0e9, 2.0e9}, {1.0, 2.0});
    std::vector<quadriga_lib::arrayant<float>> vec_f;
    for (const auto &a : vec)
    {
        quadriga_lib::arrayant<float> b;
        b.e_theta_re = arma::conv_to<arma::fcube>::from(a.e_theta_re), b.e_theta_im = arma::conv_to<arma::fcube>::from(a.e_theta_im);
        b.e_phi_re = arma::conv_to<arma::fcube>::from(a.e_phi_re), b.e_phi_im = arma::conv_to<arma::fcube>::from(a.e_phi_im);
        b.azimuth_grid = arma::conv_to<arma::fvec>::from(a.azimuth_grid), b.elevation_grid = arma::conv_to<arma::fvec>::from(a.elevation_grid);
        b.element_pos = arma::conv_to<arma::fmat>::from(a.element_pos);
        b.center_frequency = (float)a.center_frequency;
        vec_f.push_back(b);
    }
    const arma::fmat az = {{0.3f, -1.1f, 0.7f}}, el = {{0.2f, -0.5f, 0.1f}};
    const arma::fvec freq = {1.0e9f, 1.5e9f, 2.0e9f};
    arma::fcube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi<float>(vec_f, az, el, V_re, V_im, H_re, H_im, &freq);
    const arma::vec freq_d = arma::conv_to<arma::vec>::from(freq);
    const auto r = run(vec, arma::conv_to<arma::mat>::from(az), arma::conv_to<arma::mat>::from(el), &freq_d);
    CHECK(arma::abs(r.V_re).max() > 0.1); // Not a trivial (zero) result
    // Field magnitudes up to about 3 (scale 2 of a panel with peak 1.5); float rounding of the angles (about 3e-7
    // rad) times the slope of the 10 degree pattern (up to about 10 per rad), plus a few float epsilons of the value
    const double d = std::max(arma::abs(arma::conv_to<arma::cube>::from(V_re) - r.V_re).max(),
                              arma::abs(arma::conv_to<arma::cube>::from(H_im) - r.H_im).max());
    QD_DIAG("\n# diag: float against double, multi-frequency\nd,%.3e\n", d);
    CHECK(d < 2e-5);
}

TEST_CASE("arrayant_interpolate_multi - loudspeaker entries: center frequencies and directivity")
{
    // A piston at its entry frequencies equals the single-frequency interpolation; it becomes more directional with
    // frequency: the ratio of the on-axis to the 90 degree off-axis field grows
    const arma::vec entry_freqs = {200.0, 1000.0, 5000.0, 15000.0};
    const auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0, "hemisphere",
                                                            0.0, 0.0, 0.0, 0.15, 0.25, entry_freqs, 10.0);
    const arma::mat az = {{0.0, 0.5 * pi}}, el(1, 2, arma::fill::zeros);
    const auto r = run(spk, az, el, &entry_freqs);
    for (arma::uword f = 0; f < 4; ++f)
        CHECK(slice_equals(r, f, spk[f], az, el));

    const arma::vec freq = {200.0, 5000.0, 15000.0};
    const auto d = run(spk, az, el, &freq);
    auto ratio = [&d](arma::uword f)
    { return std::abs(cxd(d.V_re(0, 0, f), d.V_im(0, 0, f))) / (std::abs(cxd(d.V_re(0, 1, f), d.V_im(0, 1, f))) + 1e-20); };
    CHECK(ratio(1) > ratio(0));
    CHECK(ratio(2) > ratio(1));
}

TEST_CASE("arrayant_interpolate_multi - two-way loudspeaker example")
{
    // The example of the documentation: woofer and tweeter in one model; the woofer dominates at 150 Hz, the
    // tweeter at 12 kHz (above the last entry of 16 kHz the model is clamped, below it blended)
    const arma::vec entry_freqs = {100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 16000.0};
    const auto woofer = quadriga_lib::generate_speaker<double>("piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
                                                               0.0, 0.0, 0.0, 0.20, 0.30, entry_freqs, 10.0);
    const auto tweeter = quadriga_lib::generate_speaker<double>("piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
                                                                0.0, 0.0, 0.0, 0.20, 0.30, entry_freqs, 10.0);
    const auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);

    const arma::mat az = {0.0, 1.5708, -1.5708, 3.14159};
    const arma::mat el(1, 4, arma::fill::zeros);
    const arma::vec freq = {150.0, 750.0, 3000.0, 12000.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi<double>(speaker, az, el, V_re, V_im, H_re, H_im, &freq);

    REQUIRE(V_re.n_rows == 2);
    REQUIRE(V_re.n_cols == 4);
    REQUIRE(V_re.n_slices == 4);
    CHECK(V_re.is_finite());
    auto mag = [&](arma::uword e, arma::uword f)
    { return std::abs(cxd(V_re(e, 0, f), V_im(e, 0, f))); };
    CHECK(mag(0, 0) > 5.0 * mag(1, 0));
    CHECK(mag(1, 3) > 2.0 * mag(0, 3));
}

TEST_CASE("arrayant_interpolate_multi - input validation")
{
    ant_vec vec = scaled_entries({1.0e9, 2.0e9}, {1.0, 2.0}); // 4 elements
    const arma::mat az = {{0.1, 0.2, 0.3}}, el = {{0.0, 0.1, -0.1}};
    const arma::vec freq = {1.0e9, 1.5e9};
    const arma::uvec i_element = {3, 0};
    const arma::cube ori(3, 2, 3, arma::fill::zeros);
    const arma::mat pos(3, 2, arma::fill::zeros);
    arma::cube V_re, V_im, H_re, H_im;

    CHECK_NOTHROW(quadriga_lib::arrayant_interpolate_multi<double>(vec, az, el, V_re, V_im, H_re, H_im, &freq, &i_element, &ori, &pos, 1));

    // An invalid call leaves the outputs untouched
    V_re.fill(7.0);
    const double *V_re_ptr = V_re.memptr();
    auto call = [&](const ant_vec &v, const arma::mat &a, const arma::mat &e, const arma::vec *f, const arma::uvec *ie,
                    const arma::cube *o, const arma::mat *p, int k)
    {
        quadriga_lib::arrayant_interpolate_multi<double>(v, a, e, V_re, V_im, H_re, H_im, f, ie, o, p, k);
    };

    SECTION("arrayant_vec")
    {
        CHECK_THROWS_AS(call(ant_vec(), az, el, &freq, &i_element, &ori, &pos, 1), std::invalid_argument);
        ant_vec bad = vec;
        bad[1].e_theta_im.set_size(1, 1, 1); // Invalid entry
        CHECK_THROWS_AS(call(bad, az, el, &freq, &i_element, &ori, &pos, 1), std::invalid_argument);
        bad = vec;
        bad[1].azimuth_grid(3) += 0.01; // Entries do not match
        CHECK_THROWS_AS(call(bad, az, el, &freq, &i_element, &ori, &pos, 1), std::invalid_argument);
    }
    SECTION("azimuth")
    {
        CHECK_THROWS_AS(call(vec, arma::mat(0, 3), arma::mat(0, 3), &freq, &i_element, &ori, &pos, 1), std::invalid_argument);
        CHECK_THROWS_AS(call(vec, arma::mat(3, 3), arma::mat(3, 3), &freq, &i_element, &ori, &pos, 1), std::invalid_argument);
    }
    SECTION("elevation")
    {
        CHECK_THROWS_AS(call(vec, az, arma::mat(1, 2), &freq, &i_element, &ori, &pos, 1), std::invalid_argument);
    }
    SECTION("frequency")
    {
        for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
        {
            const arma::vec f = {1.0e9, bad};
            CHECK_THROWS_AS(call(vec, az, el, &f, &i_element, &ori, &pos, 1), std::invalid_argument);
        }
    }
    SECTION("i_element")
    {
        const arma::uvec bad = {4, 0};
        CHECK_THROWS_AS(call(vec, az, el, &freq, &bad, &ori, &pos, 1), std::invalid_argument);
    }
    SECTION("orientation")
    {
        for (const arma::cube &bad : {arma::cube(2, 2, 3), arma::cube(3, 3, 3), arma::cube(3, 2, 2)})
            CHECK_THROWS_AS(call(vec, az, el, &freq, &i_element, &bad, &pos, 1), std::invalid_argument);
    }
    SECTION("element_pos")
    {
        for (const arma::mat &bad : {arma::mat(2, 2), arma::mat(3, 4)})
            CHECK_THROWS_AS(call(vec, az, el, &freq, &i_element, &ori, &bad, 1), std::invalid_argument);
    }
    SECTION("use_kernel")
    {
        CHECK_THROWS_AS(call(vec, az, el, &freq, &i_element, &ori, &pos, -1), std::invalid_argument);
        CHECK_THROWS_AS(call(vec, az, el, &freq, &i_element, &ori, &pos, 3), std::invalid_argument);
    }
    SECTION("an output shares memory with an input")
    {
        arma::cube on_az(const_cast<double *>(az.memptr()), 1, 3, 1, false, true);
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate_multi<double>(vec, az, el, V_re, on_az, H_re, H_im), std::invalid_argument);
        arma::cube on_freq(const_cast<double *>(freq.memptr()), 1, 1, 2, false, true);
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate_multi<double>(vec, az, el, V_re, V_im, on_freq, H_im, &freq), std::invalid_argument);
    }
    SECTION("an output shares memory with a data member of an entry")
    {
        arma::cube on_pattern(vec[1].e_phi_re.memptr(), 4, 3, 2, false, true);
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate_multi<double>(vec, az, el, V_re, V_im, H_re, on_pattern), std::invalid_argument);
    }
    SECTION("two outputs are the same object")
    {
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate_multi<double>(vec, az, el, V_re, V_im, V_re, H_im), std::invalid_argument);
    }

    CHECK(V_re.memptr() == V_re_ptr);
    CHECK(arma::all(arma::vectorise(V_re) == 7.0));
}
