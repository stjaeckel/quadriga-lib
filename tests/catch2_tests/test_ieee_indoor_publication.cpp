// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Anchors the results of the published report
//   S. Jaeckel, "A modern C++ framework for the IEEE indoor channel models", IEEE 802.11-25/2318r0, 2025
//   (html_docu/publications/11-25-2318-modern-cpp-framework.md)
// as fast regression tests. Each case cites the section of the report.
//
// - Deterministic quantities (tables, seeds, closed-form values) are checked exactly.
// - Statistical results use fixed seeds and fewer realisations than the report. The tolerances are three standard
//   deviations of the estimate for the sample size used here; the standard deviations were measured over 1000
//   realisations (2000 for the capacities) and are given next to each check.
// - Where the code deliberately differs from the report, the test pins the code and the comment says how and why:
//   - Section 3.9: the uplink is the plain transpose of the downlink (the report states the conjugate transpose).
//   - Section 3.8: a sub-path is rotated by exp(+j 2 pi f t); the report states exp(-j 2 pi f t). Approaching
//     scatterers give positive Doppler shifts in the convention of get_channels_spherical (coefficients proportional
//     to exp(-j k d)). The report's Doppler figures were computed with an IFFT over time, which mirrors the Doppler
//     axis, so the published spectra (positive Jakes peak and vehicle spike) are reproduced here with an FFT over
//     time; the report's script now shows them at negative frequencies.
//   - Section 3.4, Table 3-3: the code assigns the four LCG seeds to AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; the
//     table lists them for AoD LOS, AoA LOS, AoD NLOS, AoA NLOS.
//   - Section 3.6: the Jones matrix has no cross-circular coupling (the report claims XPR_circ = XPR).
//   - Floors add attenuation but no delay (not stated in the report); paths are pruned on the normalised PDP.

#include <catch2/catch_test_macros.hpp>
#include "ieee_channel_model_functions.hpp"
#include "quadriga_lib.hpp"

#include <complex>
#include <vector>

namespace
{
    using cx = std::complex<double>;
    const double c0 = 299792458.0;

    struct ieee_param
    {
        arma::mat rx_pos, rx_orientation;
        std::vector<arma::mat> aod, aoa, pow;
        std::vector<arma::vec> delay;
        std::vector<arma::cube> M;
    };

    double sum_prod(const arma::vec &a, const arma::vec &b) // no BLAS
    {
        double s = 0.0;
        for (arma::uword i = 0; i < a.n_elem; ++i)
            s += a[i] * b[i];
        return s;
    }

    double rms_delay_spread(const arma::vec &d, const arma::vec &p)
    {
        double P = arma::accu(p), m1 = sum_prod(p, d) / P, m2 = sum_prod(p, d % d) / P;
        return std::sqrt(std::max(0.0, m2 - m1 * m1));
    }

    // Doppler power spectrum of the given taps of a SISO channel: FFT over time (bin k = +k / (N dt) for an
    // exp(+j 2 pi f t) coefficient), optional Hann window, summed over taps
    arma::vec doppler_spectrum(const quadriga_lib::channel<double> &c, const arma::uvec &taps, bool hann)
    {
        arma::uword N = c.n_snap();
        arma::vec S(N, arma::fill::zeros);
        for (auto p : taps)
        {
            arma::cx_vec h(N);
            for (arma::uword n = 0; n < N; ++n)
            {
                double w = hann ? 0.5 - 0.5 * std::cos(2.0 * arma::datum::pi * double(n) / double(N - 1)) : 1.0;
                h(n) = cx(c.coeff_re[n](0, 0, p), c.coeff_im[n](0, 0, p)) * w;
            }
            S += arma::square(arma::abs(arma::fft(h)));
        }
        return S;
    }

    double bin_freq(arma::uword k, arma::uword N, double dt)
    {
        return (k < (N + 1) / 2 ? double(k) : double(k) - double(N)) / (double(N) * dt);
    }

    arma::uvec all_taps(const quadriga_lib::channel<double> &c)
    {
        return arma::regspace<arma::uvec>(0, c.n_path()[0] - 1);
    }

    // log2 det(I + a H H^H) by Gaussian elimination (no LAPACK)
    double capacity(const arma::cx_mat &H, double a)
    {
        arma::uword n = H.n_rows;
        arma::cx_mat A(n, n);
        for (arma::uword i = 0; i < n; ++i)
            for (arma::uword j = 0; j < n; ++j)
            {
                cx s = 0.0;
                for (arma::uword k = 0; k < H.n_cols; ++k)
                    s += H(i, k) * std::conj(H(j, k));
                A(i, j) = a * s + (i == j ? 1.0 : 0.0);
            }
        double ld = 0.0;
        for (arma::uword c = 0; c < n; ++c)
        {
            arma::uword piv = c;
            for (arma::uword r = c + 1; r < n; ++r)
                if (std::abs(A(r, c)) > std::abs(A(piv, c)))
                    piv = r;
            A.swap_rows(piv, c);
            ld += std::log2(std::abs(A(c, c)));
            for (arma::uword r = c + 1; r < n; ++r)
            {
                cx f = A(r, c) / A(c, c);
                for (arma::uword k = c; k < n; ++k)
                    A(r, k) -= f * A(c, k);
            }
        }
        return ld;
    }
}

TEST_CASE("IEEE indoor publication - Sec. 2.2.3 and 3.8: API, tap spacing and snapshots")
{
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);

    // N_snap = 1 + floor(T_obs / dt)
    CHECK(quadriga_lib::get_channels_ieee_indoor(omni, omni, "A", 2.4e9, 10e-9, 1, 0.0, 1e-3)[0].n_snap() == 1);
    CHECK(quadriga_lib::get_channels_ieee_indoor(omni, omni, "A", 2.4e9, 10e-9, 1, 0.0105, 1e-3)[0].n_snap() == 11);

    // Tap spacing 10 ns / 2^k
    CHECK_NOTHROW(quadriga_lib::get_channels_ieee_indoor(omni, omni, "A", 2.4e9, 2.5e-9));
    CHECK_THROWS_AS(quadriga_lib::get_channels_ieee_indoor(omni, omni, "A", 2.4e9, 3e-9), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::get_channels_ieee_indoor(omni, omni, "A", 2.4e9, 20e-9), std::invalid_argument);

    // Output length = number of users; floors limited to 4; channel types A-F
    CHECK(quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", 2.4e9, 10e-9, 3).size() == 3);
    CHECK_THROWS_AS(quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", 2.4e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {5.0}, {5}),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::get_channels_ieee_indoor(omni, omni, "G"), std::invalid_argument);

    // Fixed seed: deterministic
    auto a = quadriga_lib::get_channels_ieee_indoor(omni, omni, "D", 2.4e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {3.0}, {0}, false, {}, 20, 50.0, 17);
    auto b = quadriga_lib::get_channels_ieee_indoor(omni, omni, "D", 2.4e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {3.0}, {0}, false, {}, 20, 50.0, 17);
    CHECK(arma::approx_equal(a[0].coeff_re[0], b[0].coeff_re[0], "absdiff", 0.0));
}

TEST_CASE("IEEE indoor publication - Sec. 3.1, Table 3-1: TGn profiles")
{
    // N_path (LOS / NLOS), max. delay, breakpoint distance, K-factor, SF std (LOS / NLOS); exact.
    // The report lists 6 clusters for model F; the code stores clusters 4 and 6 (no common tap) in one row, which
    // gives the same 41 / 42 paths.
    struct row
    {
        const char *m;
        arma::uword n_los, n_nlos;
        double max_delay_ns, dBP, K, sf_los, sf_nlos;
    };
    std::vector<row> T = {{"A", 2, 1, 0.0, 5.0, 1.0, 3.0, 4.0},
                          {"B", 13, 12, 80.0, 5.0, 1.0, 3.0, 4.0},
                          {"C", 19, 18, 200.0, 5.0, 1.0, 3.0, 5.0},
                          {"D", 28, 27, 390.0, 10.0, 2.0, 3.0, 5.0},
                          {"E", 39, 38, 730.0, 20.0, 4.0, 3.0, 6.0},
                          {"F", 42, 41, 1050.0, 30.0, 4.0, 3.0, 6.0}};

    for (auto &t : T)
    {
        ieee_param p;
        auto run = [&](double d, double fc = 2.4e9, double sf_los = NAN, double sf_nlos = NAN)
        {
            qd_ieee_indoor_param(p.rx_pos, p.rx_orientation, p.aod, p.aoa, p.pow, p.delay, p.M, t.m, fc, 10e-9, 1, {d}, {0}, {},
                                 20, 5, NAN, NAN, sf_los, sf_nlos);
        };
        auto total_dB = [&]()
        { return 10.0 * std::log10(arma::accu(p.pow[0])); };

        run(t.dBP - 0.01);
        CHECK(p.pow[0].n_cols == t.n_los);
        CHECK(std::abs(p.delay[0].max() * 1e9 - t.max_delay_ns) < 1e-9);
        CHECK(std::abs(arma::accu(p.pow[0].col(0)) / arma::accu(p.pow[0].col(1)) - t.K) < 1e-12); // LOS : NLOS of tap 1
        run(t.dBP);
        CHECK(p.pow[0].n_cols == t.n_nlos);

        // SF std: the same seed gives the same standard normal draw Z; an SF override of 1 dB measures Z
        for (double fc : {2.4e9, 0.9e9}) // TGah: 1 dB less below 1 GHz
        {
            double dl = fc < 1e9 ? 1.0 : 0.0;
            run(1.0, fc, 0.0);
            double ref = total_dB();
            run(1.0, fc, 1.0);
            double z = total_dB() - ref;
            run(1.0, fc);
            CHECK(std::abs((total_dB() - ref) / z - (t.sf_los - dl)) < 1e-9);

            run(100.0, fc, NAN, 0.0);
            ref = total_dB();
            run(100.0, fc, NAN, 1.0);
            z = total_dB() - ref;
            run(100.0, fc);
            CHECK(std::abs((total_dB() - ref) / z - (t.sf_nlos - dl)) < 1e-9);
        }
    }
}

TEST_CASE("IEEE indoor publication - Sec. 3.2: TGac tap interpolation")
{
    // Model B at 5 ns: tau_{p,i} = tau_p + i T0 / I, powers interpolated linearly in dB within each cluster
    ieee_param p;
    qd_ieee_indoor_param(p.rx_pos, p.rx_orientation, p.aod, p.aoa, p.pow, p.delay, p.M, "B", 2.4e9, 5e-9, 1, {20.0}, {0},
                         {}, 20, 5, NAN, NAN, 0.0, 0.0);
    arma::vec d = p.delay[0] * 1e9;
    CHECK(arma::approx_equal(arma::unique(d), arma::regspace<arma::vec>(0.0, 5.0, 80.0), "absdiff", 1e-9));
    CHECK(d.n_elem == 22); // cluster 1: 0..40 ns (9 taps), cluster 2: 20..80 ns (13 taps), NLOS

    // Cluster 1 at 5 ns (between 0 dB and -5.4 dB) and cluster 2 at 25 ns (between -3.2 and -6.3 dB)
    arma::vec P = arma::sum(p.pow[0], 0).t();
    P /= P(0);
    CHECK(std::abs(10.0 * std::log10(P(1)) + 2.7) < 1e-9);
    arma::uvec i25 = arma::find(arma::abs(d - 25.0) < 1e-6);
    REQUIRE(i25.n_elem == 2);
    CHECK(std::abs(10.0 * std::log10(P(i25(1))) + 4.75) < 1e-9);
}

TEST_CASE("IEEE indoor publication - Sec. 3.3, Table 3-2: path loss and floors")
{
    // PL(d) = 20 log10(4 pi fc / c) + 20 log10(d) below dBP, + 20 log10(dBP) + 35 log10(d / dBP) above (c = 3e8);
    // TGah floor loss L_F and SF std sigma_F below 1 GHz (Table 3-2); above 1 GHz the TGax floor loss
    // 18.3 n^((n + 2) / (n + 1) - 0.46) applies and the SF std is not changed (not in the report).
    ieee_param p;
    auto total_dB = [&](const char *m, double fc, double d, arma::uword nf, arma::sword seed, double sf)
    {
        qd_ieee_indoor_param(p.rx_pos, p.rx_orientation, p.aod, p.aoa, p.pow, p.delay, p.M, m, fc, 10e-9, 1, {d}, {nf}, {},
                             20, seed, NAN, NAN, sf, sf);
        return 10.0 * std::log10(arma::accu(p.pow[0]));
    };

    for (double d : {3.0, 9.99, 10.0, 25.0, 50.0})
    {
        double pl0 = 20.0 * std::log10(4.0 * arma::datum::pi * 2.4e9 / 3.0e8);
        double pl = d < 10.0 ? pl0 + 20.0 * std::log10(d) : pl0 + 20.0 + 35.0 * std::log10(d / 10.0);
        CHECK(std::abs(total_dB("D", 2.4e9, d, 0, 5, 0.0) + pl) < 1e-9);
    }

    // L_F and sigma_F from two seeds (two different standard normal draws)
    const double L_F[4] = {12.9, 18.7, 24.4, 27.7}, S_F[4] = {7.0, 2.8, 1.7, 1.5};
    for (double fc : {0.9e9, 5.25e9})
        for (arma::uword nf = 1; nf <= 4; ++nf)
        {
            double z1 = total_dB("B", fc, 3.0, 0, 1, 1.0) - total_dB("B", fc, 3.0, 0, 1, 0.0);
            double z2 = total_dB("B", fc, 3.0, 0, 2, 1.0) - total_dB("B", fc, 3.0, 0, 2, 0.0);
            double a1 = total_dB("B", fc, 3.0, nf, 1, NAN) - total_dB("B", fc, 3.0, 0, 1, 0.0);
            double a2 = total_dB("B", fc, 3.0, nf, 2, NAN) - total_dB("B", fc, 3.0, 0, 2, 0.0);
            double sigma = (a1 - a2) / (z1 - z2), L = z1 * sigma - a1;
            double n = double(nf);
            CHECK(std::abs(L - (fc < 1e9 ? L_F[nf - 1] : 18.3 * std::pow(n, (n + 2.0) / (n + 1.0) - 0.46))) < 1e-9);
            CHECK(std::abs(sigma - (fc < 1e9 ? S_F[nf - 1] : 3.0)) < 1e-9);
        }
}

TEST_CASE("IEEE indoor publication - Sec. 3.4, Table 3-3: MU-MIMO offset angles")
{
    // LCG with a = 16807, m = 2^31 - 1; offset (deg) = (s_k / m - 0.5) * 360, starting with s_0 = seed
    auto lcg = [](arma::uword s, arma::uword n)
    {
        arma::vec o(n);
        for (arma::uword i = 0; i < n; ++i)
        {
            o(i) = ((double)s / 2147483647.0 - 0.5) * 360.0;
            s = (16807ULL * s) % 2147483647ULL;
        }
        return o;
    };
    const arma::uword s1 = 608341199, s2 = 1468335517, s3 = 266639588, s4 = 115415752;

    // Code: rows of offset_angles (AoD LOS, AoD NLOS, AoA LOS, AoA NLOS) use s1, s2, s3, s4.
    // The report's table assigns s2 to AoA LOS and s3 to AoD NLOS.
    arma::mat code_map(4, 3);
    code_map.row(0) = lcg(s1, 3).t();
    code_map.row(1) = lcg(s2, 3).t();
    code_map.row(2) = lcg(s3, 3).t();
    code_map.row(3) = lcg(s4, 3).t();

    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto run = [&](const arma::mat &off)
    {
        return quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", 2.4e9, 10e-9, 3, 0.0, 1e-3, 0.0, 1.2, {3.0}, {0}, false,
                                                      off, 20, 50.0, 3);
    };
    auto c_auto = run(arma::mat()), c_code = run(code_map);
    for (int u = 0; u < 3; ++u)
    {
        CHECK(arma::approx_equal(c_auto[u].coeff_re[0], c_code[u].coeff_re[0], "absdiff", 0.0));
        CHECK(arma::approx_equal(c_auto[u].coeff_im[0], c_code[u].coeff_im[0], "absdiff", 0.0));

        // Station at the AoD LOS offset; heading AoD_LOS + 180 - AoA_LOS (station sees the AP at AoA LOS)
        double ang = std::atan2(c_auto[u].rx_pos(1, 0), c_auto[u].rx_pos(0, 0)) * 180.0 / arma::datum::pi;
        CHECK(std::abs(std::remainder(ang - code_map(0, u), 360.0)) < 1e-9);
        double yaw = c_auto[u].rx_orientation(2, 0) * 180.0 / arma::datum::pi;
        CHECK(std::abs(std::remainder(yaw - (code_map(0, u) + 180.0 - code_map(2, u)), 360.0)) < 1e-9);
    }
    CHECK(std::abs(code_map(0, 0) + 78.0189) < 1e-4); // first AoD LOS offset
}

TEST_CASE("IEEE indoor publication - Sec. 3.5: sub-path mapping")
{
    // Weighted RMS of the sub-path deviations equals the cluster angular spread (both ends), weights sum to 1;
    // LOS: one sub-path with all power; K splits the first tap (Sec. 3.5, LOS and NLOS taps).
    ieee_param p;
    qd_ieee_indoor_param(p.rx_pos, p.rx_orientation, p.aod, p.aoa, p.pow, p.delay, p.M, "B", 2.4e9, 10e-9, 1, {3.0}, {0}, {},
                         20, 9, NAN, NAN, 0.0, 0.0);

    // LOS path
    CHECK(p.pow[0](0, 0) > 0.0);
    CHECK(arma::accu(p.pow[0].col(0).tail(19)) == 0.0);

    // Path 1 = cluster 1, tap 1: AoD 225.1 deg, AoA 4.3 deg (+ 180 deg station yaw), ASD = ASA = 14.4 deg
    arma::vec w = p.pow[0].col(1) / arma::accu(p.pow[0].col(1));
    arma::vec dd = p.aod[0].col(1) * 180.0 / arma::datum::pi - 225.1;
    arma::vec da = p.aoa[0].col(1) * 180.0 / arma::datum::pi - (4.3 + 180.0);
    dd.transform([](double x) { return std::remainder(x, 360.0); });
    da.transform([](double x) { return std::remainder(x, 360.0); });
    CHECK(std::abs(std::sqrt(sum_prod(w, dd % dd)) - 14.4) < 1e-9);
    CHECK(std::abs(std::sqrt(sum_prod(w, da % da)) - 14.4) < 1e-9);
    CHECK(arma::all(w > 0.0));
}

TEST_CASE("IEEE indoor publication - Sec. 3.6: polarisation model")
{
    // M = g [alpha, j beta; -j beta, alpha], alpha^2 = X / (1 + X), beta^2 = 1 / (1 + X): linear XPR = X exactly,
    // quadrature between co- and cross-polar terms, axial ratio sqrt(X).
    // The report also claims XPR_circ = X. In the code U M U^H is diagonal (no cross-circular coupling); the two
    // circular polarisations get |alpha + beta|^2 and |alpha - beta|^2 (33.97 = 15.3 dB apart for X = 2).
    for (double X : {2.0, 10.0})
    {
        ieee_param p;
        qd_ieee_indoor_param(p.rx_pos, p.rx_orientation, p.aod, p.aoa, p.pow, p.delay, p.M, "B", 2.4e9, 10e-9, 1, {20.0}, {0},
                             {}, 20, 4, NAN, X);
        double a = std::sqrt(X / (1.0 + X)), b = std::sqrt(1.0 / (1.0 + X));
        double r = 1.0 / std::sqrt(2.0);
        double e_lin = 0.0, e_quad = 0.0, e_off = 0.0, e_diag = 0.0;
        for (arma::uword i = 0; i < p.M[0].n_slices; ++i)
            for (arma::uword s = 0; s < p.M[0].n_cols; ++s)
            {
                const double *m = p.M[0].slice_colptr(i, s);
                cx J11(m[0], m[1]), J21(m[2], m[3]), J12(m[4], m[5]), J22(m[6], m[7]);
                e_lin = std::max(e_lin, std::abs(std::norm(J11) / std::norm(J21) - X));
                e_quad = std::max(e_quad, std::abs(J21 / J11 - cx(0.0, -b / a)) + std::abs(J12 / J11 - cx(0.0, b / a)) +
                                              std::abs(J22 / J11 - 1.0));
                // U = [1, j; 1, -j] / sqrt(2): C = U M U^H
                cx U[2][2] = {{r, cx(0, r)}, {r, cx(0, -r)}}, Mm[2][2] = {{J11, J12}, {J21, J22}}, C[2][2];
                for (int x = 0; x < 2; ++x)
                    for (int y = 0; y < 2; ++y)
                    {
                        C[x][y] = 0.0;
                        for (int k = 0; k < 2; ++k)
                            for (int l = 0; l < 2; ++l)
                                C[x][y] += U[x][k] * Mm[k][l] * std::conj(U[y][l]);
                    }
                e_off = std::max(e_off, std::abs(C[0][1]) + std::abs(C[1][0]));
                e_diag = std::max(e_diag, std::abs(std::abs(C[0][0]) - (a + b)) + std::abs(std::abs(C[1][1]) - (a - b)));
            }
        CHECK(e_lin < 1e-9);
        CHECK(e_quad < 1e-12);
        CHECK(e_off < 1e-12);
        CHECK(e_diag < 1e-12);
    }
}

TEST_CASE("IEEE indoor publication - Sec. 3.7 and 4.1, Table 4-1: normalisation and RMS delay spread")
{
    // Sec. 3.7: the tap power is normalised with a dual-polarised probe (power = 1/2 sum over the 2 x 2 probe links;
    // the report writes 1 / (2 N_rx N_tx)). With the co-polar omni antennas of Table 4-1, an NLOS tap then keeps
    // alpha^2 = X / (1 + X) = 2/3 of its path gain and the LOS tap all of it. This (and not the interpolation) is
    // why the "TGn" and "TGac" rows of Table 4-1 differ from the reference rows. Models B, C, F are deterministic;
    // D and E vary with the fluorescent modulation, which is also applied to a static channel.
    auto omni = quadriga_lib::generate_arrayant_omni<double>(30.0);
    auto run = [&](const char *m, double ts, arma::sword seed)
    {
        return quadriga_lib::get_channels_ieee_indoor(omni, omni, m, 2.4e9, ts, 1, 0.0, 1e-3, 0.0, 1.2, {4.99}, {0}, false, {}, 20,
                                                      50.0, seed)[0];
    };
    auto rds = [](const quadriga_lib::channel<double> &c)
    {
        arma::vec p = arma::vectorise(arma::square(c.coeff_re[0]) + arma::square(c.coeff_im[0]));
        return rms_delay_spread(arma::vectorise(c.delay[0]) * 1e9, p);
    };

    {
        auto c = run("B", 10e-9, 1);
        arma::vec p = arma::vectorise(arma::square(c.coeff_re[0]) + arma::square(c.coeff_im[0]));
        CHECK(std::abs(p(0) / c.path_gain[0](0) - 1.0) < 1e-9);
        for (arma::uword i = 1; i < p.n_elem; ++i)
            CHECK(std::abs(p(i) / c.path_gain[0](i) - 2.0 / 3.0) < 1e-9);
    }

    // Published "TGn" / "TGac" rows (rounded to the printed digits), deterministic models
    struct pub
    {
        const char *m;
        double ts, value, rounding;
    };
    for (auto &t : std::vector<pub>{{"B", 10e-9, 15.436, 5e-4}, {"C", 10e-9, 33.122, 5e-4}, {"F", 10e-9, 148.69, 5e-3},
                                    {"B", 5e-9, 15.924, 5e-4}, {"C", 5e-9, 33.198, 5e-4}, {"F", 5e-9, 142.06, 5e-3}})
    {
        CHECK(std::abs(rds(run(t.m, t.ts, 1)) - t.value) <= t.rounding);
        CHECK(std::abs(rds(run(t.m, t.ts, 2)) - t.value) <= t.rounding);
    }

    // Models D and E: the published value is one realisation. Mean of 40 realisations here; per-realisation std
    // measured over 1000: D 0.92 / 0.93 ns, E 0.64 / 0.62 ns (10 / 5 ns). Check |published - mean| below 3 std of
    // the difference (std * sqrt(1 + 1/40)).
    for (auto &t : std::vector<pub>{{"D", 10e-9, 49.538, 0.92}, {"D", 5e-9, 48.502, 0.93}, {"E", 10e-9, 98.334, 0.64}, {"E", 5e-9, 96.832, 0.62}})
    {
        double m = 0.0;
        for (int s = 1; s <= 40; ++s)
            m += rds(run(t.m, t.ts, s)) / 40.0;
        CHECK(std::abs(t.value - m) < 3.0 * t.rounding * std::sqrt(1.0 + 1.0 / 40.0));
    }

    // Model level (sub-path powers, no antennas): exact values of the code, and the reference rows "11n [4]" and
    // "11ac [4]" of the table (deviation at most 0.21 ns, model D)
    struct model_level
    {
        const char *m;
        double ts, code, reference;
    };
    for (auto &t : std::vector<model_level>{{"B", 10e-9, 15.6466, 15.648}, {"C", 10e-9, 33.4393, 33.433}, {"D", 10e-9, 50.1626, 49.953},
                                            {"E", 10e-9, 98.9842, 98.990}, {"F", 10e-9, 148.8037, 148.92}, {"B", 5e-9, 15.9281, 15.933},
                                            {"C", 5e-9, 33.2835, 33.278}, {"D", 5e-9, 49.6120, 49.402}, {"E", 5e-9, 97.2362, 97.249},
                                            {"F", 5e-9, 142.0418, 142.14}})
    {
        ieee_param p;
        qd_ieee_indoor_param(p.rx_pos, p.rx_orientation, p.aod, p.aoa, p.pow, p.delay, p.M, t.m, 2.4e9, t.ts);
        double r = rms_delay_spread(p.delay[0] * 1e9, arma::sum(p.pow[0], 0).t());
        CHECK(std::abs(r - t.code) < 5e-4);
        CHECK(std::abs(r - t.reference) < 0.25);
    }
}

TEST_CASE("IEEE indoor publication - Sec. 4.2, Table 4-2: 4x4 MIMO NLOS capacity")
{
    // 4 x 4 ULA (lambda / 2, isotropic), 50 m (NLOS for all models), SNR 10 dB, H normalised to unit mean power.
    // 100 realisations per model (report: 2000). Per-realisation std measured over 2000 (A..F, IID):
    // 1.03, 0.81, 0.87, 0.83, 0.90, 0.86, 0.70 b/s/Hz. Tolerance: printed rounding (0.05) + 3 std of the difference
    // of the two means, std * sqrt(1/100 + 1/2000).
    auto ula = quadriga_lib::generate_arrayant_ula<double>(4, 2.4e9, 0.5, nullptr, 30.0);
    arma::vec pilot = {0.0};
    const char *models[6] = {"A", "B", "C", "D", "E", "F"};
    const double published[7] = {9.0, 9.4, 9.1, 10.2, 9.6, 10.4, 11.0};
    const double sd[7] = {1.03, 0.81, 0.87, 0.83, 0.90, 0.86, 0.70};
    const int N = 100;

    double mean[7];
    for (int m = 0; m < 7; ++m)
    {
        double s = 0.0;
        for (int n = 1; n <= N; ++n)
        {
            arma::cx_mat H(4, 4);
            if (m < 6)
            {
                auto c = quadriga_lib::get_channels_ieee_indoor(ula, ula, models[m], 2.4e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {50.0}, {0},
                                                                false, {}, 20, 50.0, n);
                arma::cx_cube hm;
                quadriga_lib::baseband_freq_response<double>(&c[0].coeff_re[0], &c[0].coeff_im[0], &c[0].delay[0], &pilot,
                                                             100e6, nullptr, nullptr, &hm);
                H = hm.slice(0);
            }
            else
            {
                std::mt19937_64 g(n);
                std::normal_distribution<double> N01;
                for (auto &h : H)
                    h = cx(N01(g), N01(g)) / std::sqrt(2.0);
            }
            double P = 0.0;
            for (auto &h : H)
                P += std::norm(h);
            H /= std::sqrt(P / 16.0);
            s += capacity(H, 10.0 / 4.0);
        }
        mean[m] = s / N;
        CHECK(std::abs(mean[m] - published[m]) < 0.05 + 3.0 * sd[m] * std::sqrt(1.0 / N + 1.0 / 2000.0));
    }

    // Ordering of Figure 4.1: IID right-most; D and F right of A, B, C, E (margins > 4 std of the differences)
    CHECK(mean[6] > std::max(mean[3], mean[5]));
    CHECK(std::min(mean[3], mean[5]) > std::max(std::max(mean[0], mean[1]), std::max(mean[2], mean[4])));
}

TEST_CASE("IEEE indoor publication - Sec. 4.3, Figure 4.2: bell-shaped environment Doppler")
{
    // Model C, v_env = 1.2 km/h, 2.4 GHz, T_obs = 2 s, dt = 0.02 s, 100 sub-paths, seeds 1234 + n (as the report).
    // Bell spectrum S(f) = 1 / (1 + 9 (f / f_d)^2): fraction of power within |f| <= f_d = (2 / pi) atan(3) = 0.795.
    // 16 realisations; measured over 1000: ensemble fraction 0.806 (rectangular window), std over batches of 16:
    // 0.022; asymmetry std 0.066. The draws are truncated at 5 f_d (13.3 Hz).
    auto omni = quadriga_lib::generate_arrayant_omni<double>(30.0);
    const double fd = 1.2 / 3.6 / (c0 / 2.4e9), dt = 0.02;
    double in = 0.0, tot = 0.0, pos = 0.0, neg = 0.0, beyond = 0.0, tot_h = 0.0;
    for (int n = 1; n <= 16; ++n)
    {
        auto c = quadriga_lib::get_channels_ieee_indoor(omni, omni, "C", 2.4e9, 10e-9, 1, 2.0, dt, 0.0, 1.2, {}, {}, false, {}, 100,
                                                        50.0, 1234 + n)[0];
        arma::vec S = doppler_spectrum(c, all_taps(c), false), Sh = doppler_spectrum(c, all_taps(c), true);
        arma::uword N = S.n_elem;
        REQUIRE(N == 101);
        for (arma::uword k = 0; k < N; ++k)
        {
            double f = bin_freq(k, N, dt);
            tot += S(k), tot_h += Sh(k);
            in += std::abs(f) <= fd ? S(k) : 0.0;
            pos += f > 0.0 ? S(k) : 0.0;
            neg += f < 0.0 ? S(k) : 0.0;
            beyond += std::abs(f) > 5.0 * fd + 2.0 / (N * dt) ? Sh(k) : 0.0;
        }
    }
    CHECK(std::abs(in / tot - 0.795) < 3.0 * 0.022 + 0.011); // + bias of the finite window (0.806 - 0.795)
    CHECK(std::abs((pos - neg) / (pos + neg)) < 3.0 * 0.066);
    CHECK(beyond / tot_h < 1e-3);
}

TEST_CASE("IEEE indoor publication - Sec. 4.3, Figure 4.3: Jakes Doppler of a moving station")
{
    // Model C, v_sta = 5 km/h towards the AP, no environment motion: |f| <= f_max = 11.12 Hz, peak near +f_max and
    // most power at positive Doppler (report: peak at +10.6 Hz on its axis, i.e. the bin at +10.89 Hz).
    // 8 realisations; measured over 1000: fraction at f > 0 = 0.897, std over batches of 8: 0.0095.
    // The report's script (IFFT over time) now shows this spectrum mirrored, see the header.
    auto omni = quadriga_lib::generate_arrayant_omni<double>(30.0);
    const double fmax = 5.0 / 3.6 / (c0 / 2.4e9), dt = 0.02;
    arma::vec S_sum(101, arma::fill::zeros), Sh_sum(101, arma::fill::zeros);
    for (int n = 1; n <= 8; ++n)
    {
        auto c = quadriga_lib::get_channels_ieee_indoor(omni, omni, "C", 2.4e9, 10e-9, 1, 2.0, dt, 5.0, 0.0, {}, {}, false, {}, 100,
                                                        50.0, 1234 + n)[0];
        S_sum += doppler_spectrum(c, all_taps(c), false);
        Sh_sum += doppler_spectrum(c, all_taps(c), true);
    }
    double tot = arma::accu(Sh_sum), pos = 0.0, out = 0.0;
    for (arma::uword k = 0; k < 101; ++k)
    {
        double f = bin_freq(k, 101, dt);
        pos += f > 0.0 ? Sh_sum(k) : 0.0;
        out += std::abs(f) > fmax + 2.0 / (101 * dt) ? Sh_sum(k) : 0.0;
    }
    CHECK(std::abs(pos / tot - 0.897) < 3.0 * 0.0095);
    CHECK(out / tot < 1e-3);
    CHECK(std::abs(bin_freq(S_sum.index_max(), 101, dt) - 10.891) < 0.5); // within one bin (0.495 Hz)
}

TEST_CASE("IEEE indoor publication - Sec. 4.3, Figure 4.4: Doppler spike of a moving vehicle")
{
    // Model F, v_veh = 40 km/h, v_env = 0.089 km/h (as the figure; the listing says 1.2), T_obs = 1 s, dt = 4 ms.
    // Spike on the 20 ns tap (tap 3) at f_spike = v / lambda = 88.95 Hz, drawn within +-0.1 f_spike.
    // 12 realisations; measured over 1000: energy centroid of the spike 88.96 Hz, std over batches of 12: 0.24 Hz.
    // The report states that the spike is 3.3 dB below the same tap at 0 Hz (its figure shows 4.3 dB between the
    // peak bins); the energy ratio is not a stable statistic (ensemble 1.7 dB, batch std 1.8 dB), not checked.
    auto omni = quadriga_lib::generate_arrayant_omni<double>(30.0);
    const double fs = 40.0 / 3.6 / (c0 / 2.4e9), dt = 0.004;
    double hi = 0.0, fw = 0.0, mirror = 0.0, edge = 0.0;
    for (int n = 1; n <= 12; ++n)
    {
        auto c = quadriga_lib::get_channels_ieee_indoor(omni, omni, "F", 2.4e9, 10e-9, 1, 1.0, dt, 0.0, 0.089, {}, {}, false, {}, 100,
                                                        40.0, 1234 + n)[0];
        REQUIRE(std::abs(c.delay[0](0, 0, 3) * 1e9 - 20.0) < 1e-6);
        arma::vec S = doppler_spectrum(c, {3}, true);
        arma::uword N = S.n_elem;
        for (arma::uword k = 0; k < N; ++k)
        {
            double f = bin_freq(k, N, dt);
            if (f > 0.85 * fs && f < 1.15 * fs)
                hi += S(k), fw += S(k) * f;
            if (f < -0.85 * fs && f > -1.15 * fs)
                mirror += S(k);
            if ((f > 0.85 * fs && f < 0.88 * fs) || (f > 1.12 * fs && f < 1.15 * fs))
                edge += S(k);
        }
    }
    CHECK(std::abs(fw / hi - fs) < 3.0 * 0.24);
    CHECK(mirror < 1e-3 * hi); // positive Doppler (approaching vehicle)
    CHECK(edge < 1e-2 * hi);   // truncated at +-0.1 f_spike
}

TEST_CASE("IEEE indoor publication - Sec. 4.3, Figure 4.5: fluorescent lights")
{
    // Model E, f_m = 50 Hz, v_env = 1.2 km/h, T_obs = 0.4 s, dt = 0.8 ms: real modulation at 100, 300, 500 Hz
    // (symmetric lines), 100 Hz strongest; I/C = X^2, X ~ N(0.0203, 0.0107^2), E[I/C] = 5.27e-4.
    // 12 realisations, Hann window, +-20 Hz around each line; measured over 1000: line energy / total = 4.85e-4,
    // std over batches of 12: 1.58e-4. Line power ratios are not checked: the per-realisation normalisation of I/C
    // makes them biased and heavy-tailed (ensemble -11.6 / -15.9 dB instead of -15 / -20 dB).
    auto omni = quadriga_lib::generate_arrayant_omni<double>(30.0);
    const double dt = 0.0008;
    double L[3] = {0.0, 0.0, 0.0}, Lp = 0.0, Ln = 0.0, tot = 0.0;
    for (int n = 1; n <= 12; ++n)
    {
        auto c = quadriga_lib::get_channels_ieee_indoor(omni, omni, "E", 2.4e9, 10e-9, 1, 0.4, dt, 0.0, 1.2, {}, {}, false, {}, 100,
                                                        50.0, 1234 + n)[0];
        arma::vec S = doppler_spectrum(c, all_taps(c), true);
        arma::uword N = S.n_elem;
        REQUIRE(N == 501);
        tot += arma::accu(S);
        for (arma::uword k = 0; k < N; ++k)
        {
            double f = bin_freq(k, N, dt);
            for (int l = 0; l < 3; ++l)
                if (std::abs(std::abs(f) - 100.0 * (2 * l + 1)) < 20.0)
                    L[l] += S(k), (f > 0.0 ? Lp : Ln) += S(k);
        }
    }
    CHECK(std::abs((L[0] + L[1] + L[2]) / tot - 4.85e-4) < 3.0 * 1.58e-4);
    CHECK(L[0] > L[1]);
    CHECK(L[0] > L[2]);
    CHECK(std::abs(10.0 * std::log10(Lp / Ln)) < 0.05);
}

TEST_CASE("IEEE indoor publication - Sec. 3.9: uplink")
{
    // The report states H_UL = H_DL^H. The code returns H_UL = H_DL^T (reciprocal channel in the baseband
    // convention of the library; the conjugate would reverse the Doppler shift of a moving station). Delays are
    // transposed, positions and orientations swapped, path gains equal.
    auto ap = quadriga_lib::generate_arrayant_ula<double>(3, 2.4e9, 0.5);
    auto sta = quadriga_lib::generate_arrayant_ula<double>(2, 2.4e9, 0.5);
    auto dl = quadriga_lib::get_channels_ieee_indoor(ap, sta, "D", 2.4e9, 10e-9, 1, 0.003, 1e-3, 3.0, 1.2, {6.0}, {0}, false, {}, 20, 50.0, 8)[0];
    auto ul = quadriga_lib::get_channels_ieee_indoor(ap, sta, "D", 2.4e9, 10e-9, 1, 0.003, 1e-3, 3.0, 1.2, {6.0}, {0}, true, {}, 20, 50.0, 8)[0];
    REQUIRE(dl.n_snap() == 4);
    double e = 0.0;
    for (arma::uword s = 0; s < 4; ++s)
        for (arma::uword p = 0; p < dl.coeff_re[s].n_slices; ++p)
        {
            e = std::max(e, arma::abs(ul.coeff_re[s].slice(p).t() - dl.coeff_re[s].slice(p)).max());
            e = std::max(e, arma::abs(ul.coeff_im[s].slice(p).t() - dl.coeff_im[s].slice(p)).max());
            e = std::max(e, arma::abs(ul.delay[s].slice(p).t() - dl.delay[s].slice(p)).max());
        }
    CHECK(e == 0.0);
    CHECK(arma::approx_equal(ul.tx_pos, dl.rx_pos, "absdiff", 0.0));
    CHECK(arma::approx_equal(ul.path_gain[3], dl.path_gain[3], "absdiff", 0.0));
}
