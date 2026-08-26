// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Cross-validation of the 'apply_fspl' and 'apply_phase' options of calc_diffraction_gain
// against the two places in the library that own those models:
//
//   - Free-space / spreading loss: qrt_file_read, which folds the path loss into 'path_gain'
//     when reading a QRT file. A non-normalized file (normalize_M = 0) carries a unit
//     polarization matrix, so the returned path_gain IS the path loss and nothing else.
//   - Propagation phase: get_channels_spherical (EM) and get_channels_multifreq (acoustic),
//     which rotate the channel coefficient by exp(-j*2*pi*d/lambda) derived from 'path_length'.
//
// Both models are exercised in EM and scalar mode. No reference values are hardcoded: each
// test compares two independent code paths against each other.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_lib.hpp"

#include <unordered_map>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <cstdio>

// Speed of sound used for the acoustic tests. This must match the value the library uses to
// form the acoustic wavelength in calc_diffraction_gain, otherwise the phase tests below fail.
static constexpr double SPEED_OF_SOUND = 343.2;

// A lossless material assigned to every face of the reference cube. The cross-validation runs
// on clear paths, so the material is never evaluated; it only has to be well-formed.
static inline void gains_test_scene(arma::mat &cube, arma::uvec &mtl_ind,
                                    std::unordered_map<std::string, std::vector<double>> &mtl_map)
{
    cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});

    static const char *names[9] = {"a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"};
    static const double vals[9] = {1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

    mtl_ind = arma::ones<arma::uvec>(cube.n_rows);
    mtl_map.clear();
    for (int c = 0; c < 9; ++c)
        mtl_map[names[c]] = std::vector<double>{vals[c]};
}

// Write a QRT file holding one TX and 'n_cir' receivers, each reached by a single
// zero-interaction (LOS) path with a unit polarization matrix. Reading it back with
// normalize_M = 0 therefore returns path_gain == path loss.
//
// TX and RX coordinates are chosen to be exactly representable in float, since QRT stores
// positions as float and the comparison distance must survive the round trip unchanged.
static inline void write_los_qrt(const std::string &fn, const arma::vec &freq,
                                 const arma::vec &tx_pos, const arma::mat &rx_pos, bool scalar_mode)
{
    const unsigned n_cir = (unsigned)rx_pos.n_rows;
    const size_t n_freq = freq.n_elem;

    arma::mat cir_orientation; // empty: no orientation
    arma::u32_vec cir_offset(1, arma::fill::zeros);
    quadriga_lib::qrt_file_init<double>(fn, freq, rx_pos, cir_orientation,
                                        {"RX"}, cir_offset, 1, scalar_mode);

    // One path per CIR, nSEG = 0. init() leaves the transfer coefficients at unity for every
    // frequency, which is exactly the "non-normalized, no material loss" case we want.
    std::vector<quadriga_lib::path> paths;
    paths.reserve(n_cir);
    for (unsigned c = 0; c < n_cir; ++c)
    {
        quadriga_lib::path p(0, n_freq, scalar_mode);
        p.iC = c;
        paths.push_back(std::move(p));
    }

    arma::vec tx_orientation(3, arma::fill::zeros);
    size_t written = quadriga_lib::qrt_file_append<double>(fn, paths, tx_pos, tx_orientation, "TX");
    REQUIRE(written == (size_t)n_cir);
}

// The spreading-loss model must be bit-for-bit the same whether it arrives through
// qrt_file_read or through calc_diffraction_gain(apply_fspl = true). EM mode: the loss is
// 32.45 + 20*log10(f_GHz) + 20*log10(d), so both the distance and the frequency dependence
// are exercised by sweeping three distances against three widely separated bands.
TEST_CASE("Gains - FSPL matches qrt_file_read, EM mode")
{
    const std::string fn = "test_gains_em.qrt";

    arma::vec freq_GHz = {0.9, 3.5, 28.0};
    arma::vec tx_pos = {-200.0, 0.0, 2.0};
    arma::mat rx_pos = {{-100.0, 0.0, 2.0},    // 100 m
                        {-160.0, 0.0, 2.0},    // 40 m
                        {-187.5, 0.0, 2.0}};   // 12.5 m
    arma::vec dist = {100.0, 40.0, 12.5};

    write_los_qrt(fn, freq_GHz, tx_pos, rx_pos, false);

    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    gains_test_scene(cube, mtl_ind, mtl_map);

    for (arma::uword ic = 0; ic < rx_pos.n_rows; ++ic)
    {
        arma::vec cf, txp, rxp, pl;
        arma::mat pg;
        quadriga_lib::qrt_file_read<double>(fn, ic, 0, true, &cf, &txp, nullptr, &rxp, nullptr,
                                            nullptr, nullptr, &pg, &pl, nullptr,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, 0);

        REQUIRE(pg.n_rows == 1);
        REQUIRE(pg.n_cols == freq_GHz.n_elem);
        REQUIRE(cf.n_elem == freq_GHz.n_elem);

        // Precondition: the QRT path length is the direct TX-RX distance, which is what
        // calc_diffraction_gain uses. If this drifts, the comparison below is meaningless.
        CHECK(std::abs(pl(0) - dist(ic)) < 1e-9);

        // Feed calc_diffraction_gain the positions as they came back from the file, so both
        // sides see identical (float-rounded) coordinates.
        arma::mat orig(1, 3), dest(1, 3);
        for (arma::uword k = 0; k < 3; ++k)
            orig(0, k) = txp(k), dest(0, k) = rxp(k);

        for (arma::uword f = 0; f < freq_GHz.n_elem; ++f)
        {
            arma::vec g_rel, g_abs;
            quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, cf(f), 0,
                                                        &g_rel, nullptr, nullptr, 0, nullptr, 0, 0,
                                                        false, 0.0, false, false);
            quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, cf(f), 0,
                                                        &g_abs, nullptr, nullptr, 0, nullptr, 0, 0,
                                                        false, 0.0, true, false);

            // The path is clear, so the diffraction gain is 1 and g_abs is pure path loss
            CHECK(std::abs(g_rel(0) - 1.0) < 1e-13);
            CHECK(std::abs(g_abs(0) / pg(0, f) - 1.0) < 1e-9);
        }

        // Frequency dependence is real: 0.9 GHz vs 28 GHz differ by 20*log10(28/0.9)
        double ratio_dB = 10.0 * std::log10(pg(0, 0) / pg(0, 2));
        CHECK(std::abs(ratio_dB - 20.0 * std::log10(28.0 / 0.9)) < 1e-6);
    }

    std::remove(fn.c_str());
}

// Same cross-check in scalar (acoustic) mode. Here the model is 20*log10(d) + alpha(f)*d with
// alpha from ISO 9613-1, so this is the test that pins the shared air-absorption code: the QRT
// reader and calc_diffraction_gain must evaluate alpha identically. The bands are chosen to
// span the range where alpha grows by three orders of magnitude.
TEST_CASE("Gains - FSPL matches qrt_file_read, scalar mode")
{
    const std::string fn = "test_gains_scalar.qrt";

    arma::vec freq_Hz = {125.0, 1000.0, 8000.0};
    arma::vec tx_pos = {-200.0, 0.0, 2.0};
    arma::mat rx_pos = {{-100.0, 0.0, 2.0},    // 100 m
                        {-160.0, 0.0, 2.0},    // 40 m
                        {-187.5, 0.0, 2.0}};   // 12.5 m
    arma::vec dist = {100.0, 40.0, 12.5};

    write_los_qrt(fn, freq_Hz, tx_pos, rx_pos, true);

    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    gains_test_scene(cube, mtl_ind, mtl_map);

    for (arma::uword ic = 0; ic < rx_pos.n_rows; ++ic)
    {
        arma::vec cf, txp, rxp, pl;
        arma::mat pg;
        quadriga_lib::qrt_file_read<double>(fn, ic, 0, true, &cf, &txp, nullptr, &rxp, nullptr,
                                            nullptr, nullptr, &pg, &pl, nullptr,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, 0);

        REQUIRE(pg.n_rows == 1);
        REQUIRE(pg.n_cols == freq_Hz.n_elem);

        CHECK(std::abs(pl(0) - dist(ic)) < 1e-9);

        arma::mat orig(1, 3), dest(1, 3);
        for (arma::uword k = 0; k < 3; ++k)
            orig(0, k) = txp(k), dest(0, k) = rxp(k);

        for (arma::uword f = 0; f < freq_Hz.n_elem; ++f)
        {
            arma::vec g_abs;
            quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, cf(f), 0,
                                                        &g_abs, nullptr, nullptr, 0, nullptr, 0, 0,
                                                        true, 0.0, true, false);
            CHECK(std::abs(g_abs(0) / pg(0, f) - 1.0) < 1e-9);
        }

        // Spreading alone is frequency-independent in scalar mode, so any band-to-band
        // difference is air absorption. Solve for the absorption difference and check it is
        // positive and grows with distance, i.e. it really is a per-meter term.
        double extra_dB = -10.0 * std::log10(pg(0, 2) / pg(0, 0));
        CHECK(extra_dB > 0.0);

        // At 125 Hz the loss is dominated by spreading: within 0.1 dB of 20*log10(d)
        CHECK(std::abs(-10.0 * std::log10(pg(0, 0)) - 20.0 * std::log10(dist(ic))) < 0.1);
    }

    // Absorption must scale with distance: the 8 kHz-vs-125 Hz excess at 100 m is 8x the
    // excess at 12.5 m. This is read entirely from the QRT side, then confirmed above to
    // match calc_diffraction_gain.
    auto excess_dB = [&](arma::uword ic) -> double
    {
        arma::mat pg;
        quadriga_lib::qrt_file_read<double>(fn, ic, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, &pg, nullptr, nullptr,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, 0);
        return -10.0 * std::log10(pg(0, 2) / pg(0, 0));
    };
    CHECK(std::abs(excess_dB(0) / excess_dB(2) - 8.0) < 1e-6);

    std::remove(fn.c_str());
}

// The propagation phase applied by calc_diffraction_gain must be the same rotation that
// get_channels_spherical derives from 'path_length'. The check is an equivalence between two
// generator calls that differ only in where the phase comes from:
//
//   A: M without phase, generator phase ON  ->  coeff = M * exp(-j*k*d) * antenna
//   B: M with phase,    generator phase OFF ->  coeff = (M * exp(-j*k*d)) * antenna
//
// If the two agree, calc_diffraction_gain reproduces the generator's phase exactly, including
// its sign. Run on a clear path (M = identity) and on a wall crossing (M non-trivial), so the
// test covers both the bare phasor and its composition with a material Jones matrix.
TEST_CASE("Gains - Phase matches get_channels_spherical, EM mode")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    gains_test_scene(cube, mtl_ind, mtl_map);

    // A lossy wall for the obstructed case
    arma::uvec mtl_ind_w = arma::ones<arma::uvec>(cube.n_rows);
    std::unordered_map<std::string, std::vector<double>> mtl_map_w;
    {
        static const char *names[9] = {"a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"};
        static const double vals[9] = {4.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        for (int c = 0; c < 9; ++c)
            mtl_map_w[names[c]] = std::vector<double>{vals[c]};
    }

    auto tx_ant = quadriga_lib::generate_arrayant_omni<double>();
    auto rx_ant = quadriga_lib::generate_arrayant_omni<double>();

    struct Case
    {
        arma::vec T, R;
        bool obstructed;
    };
    std::vector<Case> cases = {
        {{-40.0, 0.0, 2.0}, {-10.0, 0.0, 2.0}, false}, // 30 m clear
        {{-10.0, 0.0, 0.5}, {10.0, 0.0, 0.5}, true},   // 20 m through the cube
    };

    for (const auto &cs : cases)
    {
        const auto &ind = cs.obstructed ? mtl_ind_w : mtl_ind;
        const auto &map = cs.obstructed ? mtl_map_w : mtl_map;

        for (double freq : {0.9e9, 28.0e9})
        {
            arma::mat orig(1, 3), dest(1, 3);
            for (arma::uword k = 0; k < 3; ++k)
                orig(0, k) = cs.T(k), dest(0, k) = cs.R(k);

            arma::vec gain;
            arma::mat M_plain, M_phase;
            quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, ind, map, freq, 0,
                                                        &gain, &M_plain, nullptr, 0, nullptr, 0, 0,
                                                        false, 0.0, false, false);
            quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, ind, map, freq, 0,
                                                        nullptr, &M_phase, nullptr, 0, nullptr, 0, 0,
                                                        false, 0.0, false, true);

            REQUIRE(M_plain.n_rows == 8);

            // Straight transmission path: present it to the generator as LOS so the generator
            // uses the same TX-RX distance that calc_diffraction_gain used.
            double dx = cs.R(0) - cs.T(0), dy = cs.R(1) - cs.T(1), dz = cs.R(2) - cs.T(2);
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            arma::mat fbs(3, 1), lbs(3, 1);
            for (arma::uword k = 0; k < 3; ++k)
                fbs(k, 0) = 0.5 * (cs.T(k) + cs.R(k)), lbs(k, 0) = fbs(k, 0);

            arma::vec pg = {gain(0)}, plen = {dist};

            arma::cube crA, ciA, dlA, crB, ciB, dlB;

            // A: phase supplied by the generator
            quadriga_lib::get_channels_spherical<double>(&tx_ant, &rx_ant,
                                                         cs.T(0), cs.T(1), cs.T(2), 0.0, 0.0, 0.0,
                                                         cs.R(0), cs.R(1), cs.R(2), 0.0, 0.0, 0.0,
                                                         &fbs, &lbs, &pg, &plen, &M_plain,
                                                         &crA, &ciA, &dlA, freq, false, false);

            // B: phase already in M, generator phase disabled
            quadriga_lib::get_channels_spherical<double>(&tx_ant, &rx_ant,
                                                         cs.T(0), cs.T(1), cs.T(2), 0.0, 0.0, 0.0,
                                                         cs.R(0), cs.R(1), cs.R(2), 0.0, 0.0, 0.0,
                                                         &fbs, &lbs, &pg, &plen, &M_phase,
                                                         &crB, &ciB, &dlB, 0.0, false, false);

            REQUIRE(crA.n_elem == crB.n_elem);
            CHECK(arma::abs(crA - crB).max() < 1e-10);
            CHECK(arma::abs(ciA - ciB).max() < 1e-10);

            // The phase is not trivially zero: with phase disabled and no phase in M the
            // coefficient would be real, so the imaginary part proves rotation happened.
            arma::cube crC, ciC, dlC;
            quadriga_lib::get_channels_spherical<double>(&tx_ant, &rx_ant,
                                                         cs.T(0), cs.T(1), cs.T(2), 0.0, 0.0, 0.0,
                                                         cs.R(0), cs.R(1), cs.R(2), 0.0, 0.0, 0.0,
                                                         &fbs, &lbs, &pg, &plen, &M_plain,
                                                         &crC, &ciC, &dlC, 0.0, false, false);
            CHECK(arma::abs(ciA - ciC).max() > 1e-6);

            // Rotation preserves magnitude
            arma::cube magA = arma::sqrt(crA % crA + ciA % ciA);
            arma::cube magC = arma::sqrt(crC % crC + ciC % ciC);
            CHECK(arma::abs(magA - magC).max() < 1e-10);
        }
    }
}

// Acoustic counterpart of the previous test. get_channels_multifreq is the generator that
// carries a configurable propagation speed and a 2-row scalar M, so it owns the acoustic phase
// model. Same A/B equivalence; freq_out = 0 disables the generator's own rotation.
//
// This test also pins the speed of sound: if calc_diffraction_gain formed the acoustic
// wavelength from anything other than SPEED_OF_SOUND, case A and case B would disagree.
TEST_CASE("Gains - Phase matches get_channels_multifreq, scalar mode")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    gains_test_scene(cube, mtl_ind, mtl_map);

    // Acoustic wall: mass-law style material for the obstructed case
    arma::uvec mtl_ind_w = arma::ones<arma::uvec>(cube.n_rows);
    std::unordered_map<std::string, std::vector<double>> mtl_map_w;
    {
        static const char *names[9] = {"a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"};
        static const double vals[9] = {2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0};
        for (int c = 0; c < 9; ++c)
            mtl_map_w[names[c]] = std::vector<double>{vals[c]};
    }

    struct Case
    {
        arma::vec T, R;
        bool obstructed;
    };
    std::vector<Case> cases = {
        {{-40.0, 0.0, 2.0}, {-10.0, 0.0, 2.0}, false}, // 30 m clear
        {{-10.0, 0.0, 0.5}, {10.0, 0.0, 0.5}, true},   // 20 m through the cube
    };

    for (const auto &cs : cases)
    {
        const auto &ind = cs.obstructed ? mtl_ind_w : mtl_ind;
        const auto &map = cs.obstructed ? mtl_map_w : mtl_map;

        for (double freq : {125.0, 4000.0})
        {
            std::vector<quadriga_lib::arrayant<double>> tx_v{quadriga_lib::generate_arrayant_omni<double>()};
            std::vector<quadriga_lib::arrayant<double>> rx_v{quadriga_lib::generate_arrayant_omni<double>()};
            tx_v[0].center_frequency = freq;
            rx_v[0].center_frequency = freq;

            arma::mat orig(1, 3), dest(1, 3);
            for (arma::uword k = 0; k < 3; ++k)
                orig(0, k) = cs.T(k), dest(0, k) = cs.R(k);

            arma::vec gain;
            arma::mat X_plain, X_phase;
            quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, ind, map, freq, 0,
                                                        &gain, &X_plain, nullptr, 0, nullptr, 0, 0,
                                                        true, 0.0, false, false);
            quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, ind, map, freq, 0,
                                                        nullptr, &X_phase, nullptr, 0, nullptr, 0, 0,
                                                        true, 0.0, false, true);

            REQUIRE(X_plain.n_rows == 2);

            // Scalar M is [2, n_path, n_freq_in]
            arma::cube M_plain(2, 1, 1), M_phase(2, 1, 1);
            for (arma::uword k = 0; k < 2; ++k)
                M_plain(k, 0, 0) = X_plain(k, 0), M_phase(k, 0, 0) = X_phase(k, 0);

            double dx = cs.R(0) - cs.T(0), dy = cs.R(1) - cs.T(1), dz = cs.R(2) - cs.T(2);
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            arma::mat fbs(3, 1), lbs(3, 1);
            for (arma::uword k = 0; k < 3; ++k)
                fbs(k, 0) = 0.5 * (cs.T(k) + cs.R(k)), lbs(k, 0) = fbs(k, 0);

            arma::mat pg(1, 1);
            pg(0, 0) = gain(0);
            arma::vec plen = {dist};
            arma::vec freq_in = {freq};

            std::vector<arma::cube> crA, ciA, dlA, crB, ciB, dlB;

            // A: phase supplied by the generator, at the acoustic propagation speed
            arma::vec freq_out_on = {freq};
            quadriga_lib::get_channels_multifreq<double>(tx_v, rx_v,
                                                         cs.T(0), cs.T(1), cs.T(2), 0.0, 0.0, 0.0,
                                                         cs.R(0), cs.R(1), cs.R(2), 0.0, 0.0, 0.0,
                                                         fbs, lbs, pg, plen, M_plain,
                                                         freq_in, freq_out_on, crA, ciA, dlA,
                                                         false, false, SPEED_OF_SOUND);

            // B: phase already in M; freq_out = 0 disables the generator's rotation
            arma::vec freq_out_off = {0.0};
            quadriga_lib::get_channels_multifreq<double>(tx_v, rx_v,
                                                         cs.T(0), cs.T(1), cs.T(2), 0.0, 0.0, 0.0,
                                                         cs.R(0), cs.R(1), cs.R(2), 0.0, 0.0, 0.0,
                                                         fbs, lbs, pg, plen, M_phase,
                                                         freq_in, freq_out_off, crB, ciB, dlB,
                                                         false, false, SPEED_OF_SOUND);

            REQUIRE(crA.size() == 1);
            REQUIRE(crB.size() == 1);
            REQUIRE(crA[0].n_elem == crB[0].n_elem);

            CHECK(arma::abs(crA[0] - crB[0]).max() < 1e-10);
            CHECK(arma::abs(ciA[0] - ciB[0]).max() < 1e-10);

            // Reference with neither phase source: proves the rotation is non-trivial and
            // that it preserves magnitude.
            std::vector<arma::cube> crC, ciC, dlC;
            quadriga_lib::get_channels_multifreq<double>(tx_v, rx_v,
                                                         cs.T(0), cs.T(1), cs.T(2), 0.0, 0.0, 0.0,
                                                         cs.R(0), cs.R(1), cs.R(2), 0.0, 0.0, 0.0,
                                                         fbs, lbs, pg, plen, M_plain,
                                                         freq_in, freq_out_off, crC, ciC, dlC,
                                                         false, false, SPEED_OF_SOUND);

            CHECK(arma::abs(ciA[0] - ciC[0]).max() > 1e-6);

            arma::cube magA = arma::sqrt(crA[0] % crA[0] + ciA[0] % ciA[0]);
            arma::cube magC = arma::sqrt(crC[0] % crC[0] + ciC[0] % ciC[0]);
            CHECK(arma::abs(magA - magC).max() < 1e-10);
        }
    }
}

// The acoustic and EM phase must genuinely differ at the same frequency and distance: the
// wavelength comes from the propagation speed, and the two speeds differ by six orders of
// magnitude. Guards against a scalar_mode branch that silently falls back to the speed of light.
TEST_CASE("Gains - Acoustic and EM phase use different wave speeds")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    gains_test_scene(cube, mtl_ind, mtl_map);

    arma::mat orig = {{-40.0, 0.0, 2.0}};
    arma::mat dest = {{-10.0, 0.0, 2.0}}; // 30 m
    const double freq = 4000.0;

    arma::mat X_em, X_ac;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                nullptr, &X_em, nullptr, 0, nullptr, 0, 0,
                                                false, 0.0, false, true);
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                nullptr, &X_ac, nullptr, 0, nullptr, 0, 0,
                                                true, 0.0, false, true);

    std::complex<double> em(X_em(0, 0), X_em(1, 0));
    std::complex<double> ac(X_ac(0, 0), X_ac(1, 0));

    // At 4 kHz over 30 m the EM phase is microscopic; the acoustic phase wraps many times
    CHECK(std::abs(em - std::complex<double>(1.0, 0.0)) < 1e-2);
    CHECK(std::abs(ac - em) > 1e-2);

    // Both remain unit-magnitude, so xprmat stays normalized either way
    CHECK(std::abs(std::abs(em) - 1.0) < 1e-12);
    CHECK(std::abs(std::abs(ac) - 1.0) < 1e-12);

    // The acoustic phasor advances by exactly one turn per wavelength of extra distance
    arma::mat dest2 = {{-10.0 + SPEED_OF_SOUND / freq, 0.0, 2.0}};
    arma::mat X_ac2;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest2, cube, mtl_ind, mtl_map, freq, 0,
                                                nullptr, &X_ac2, nullptr, 0, nullptr, 0, 0,
                                                true, 0.0, false, true);
    std::complex<double> ac2(X_ac2(0, 0), X_ac2(1, 0));
    CHECK(std::abs(ac2 - ac) < 1e-9);
}
