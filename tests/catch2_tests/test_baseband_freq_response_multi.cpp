// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

#include <cmath>
#include <complex>
#include <vector>

// Helper: build constant-envelope test data (no baked-in delay phase)
// coeff = amplitude * exp(j * env_phase) at all input frequencies
static void build_flat_envelope(arma::uword n_rx, arma::uword n_tx, arma::uword n_path, arma::uword n_freq_in,
                                double amplitude, double env_phase,
                                std::vector<arma::Cube<double>> &cre, std::vector<arma::Cube<double>> &cim)
{
    cre.resize(n_freq_in);
    cim.resize(n_freq_in);
    double re = amplitude * std::cos(env_phase);
    double im = amplitude * std::sin(env_phase);
    for (arma::uword f = 0; f < n_freq_in; ++f)
    {
        cre[f].set_size(n_rx, n_tx, n_path);
        cim[f].set_size(n_rx, n_tx, n_path);
        cre[f].fill(re);
        cim[f].fill(im);
    }
}

// Helper: build delay vector (planar wave)
static std::vector<arma::Cube<double>> build_planar_delays(arma::uword n_path, arma::uword n_freq_in,
                                                           const arma::Col<double> &tau)
{
    std::vector<arma::Cube<double>> dl(n_freq_in);
    for (arma::uword f = 0; f < n_freq_in; ++f)
    {
        dl[f].set_size(1, 1, n_path);
        for (arma::uword p = 0; p < n_path; ++p)
            dl[f](0, 0, p) = tau(p);
    }
    return dl;
}

// Helper: build delay vector (spherical wave)
static std::vector<arma::Cube<double>> build_spherical_delays(arma::uword n_rx, arma::uword n_tx, arma::uword n_path,
                                                              arma::uword n_freq_in, const arma::Cube<double> &tau)
{
    std::vector<arma::Cube<double>> dl(n_freq_in);
    for (arma::uword f = 0; f < n_freq_in; ++f)
        dl[f] = tau;
    return dl;
}

// Helper: bake delay phase into coefficients (simulates get_channels_multifreq output)
// coeff_baked[f] = coeff_env[f] * exp(-j * 2 * pi * freq_in[f] * delay)
static void bake_delay_phase(std::vector<arma::Cube<double>> &cre, std::vector<arma::Cube<double>> &cim,
                             const std::vector<arma::Cube<double>> &delay, const arma::Col<double> &freq_in)
{
    arma::uword n_freq_in = freq_in.n_elem;
    bool planar = (delay[0].n_rows == 1 && delay[0].n_cols == 1);
    arma::uword n_rx = cre[0].n_rows, n_tx = cre[0].n_cols, n_path = cre[0].n_slices;
    arma::uword n_ant = n_rx * n_tx;

    for (arma::uword f = 0; f < n_freq_in; ++f)
    {
        double *pr = cre[f].memptr();
        double *pi = cim[f].memptr();
        const double *pd = delay[0].memptr();

        for (arma::uword p = 0; p < n_path; ++p)
            for (arma::uword a = 0; a < n_ant; ++a)
            {
                arma::uword idx = p * n_ant + a;
                double dl = planar ? pd[p] : pd[idx];
                double phase = -6.283185307179586 * (double)freq_in(f) * dl;
                double c = std::cos(phase), s = std::sin(phase);
                double re = pr[idx], im = pi[idx];
                pr[idx] = re * c - im * s;
                pi[idx] = re * s + im * c;
            }
    }
}

// ================================================================================================
// Error handling tests
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Error handling")
{
    // Minimal valid data: 1 RX, 1 TX, 1 path, 1 freq_in
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {1.0e9};
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 1); cre[0].fill(1.0);
    cim[0].set_size(1, 1, 1); cim[0].fill(0.0);
    dl[0].set_size(1, 1, 1);  dl[0].fill(0.0);

    arma::Cube<double> Hr, Hi;

    // --- Empty freq_in ---
    {
        arma::Col<double> fin_empty;
        std::vector<arma::Cube<double>> cre_e, cim_e, dl_e;
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre_e, cim_e, dl_e, fin_empty, freq_out, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }

    // --- Empty freq_out ---
    {
        arma::Col<double> fout_empty;
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, fout_empty, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }

    // --- coeff_re size mismatch ---
    {
        std::vector<arma::Cube<double>> cre_bad(2);
        cre_bad[0].set_size(1, 1, 1); cre_bad[0].fill(1.0);
        cre_bad[1].set_size(1, 1, 1); cre_bad[1].fill(1.0);
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre_bad, cim, dl, freq_in, freq_out, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }

    // --- coeff_im size mismatch ---
    {
        std::vector<arma::Cube<double>> cim_bad(2);
        cim_bad[0].set_size(1, 1, 1); cim_bad[0].fill(0.0);
        cim_bad[1].set_size(1, 1, 1); cim_bad[1].fill(0.0);
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre, cim_bad, dl, freq_in, freq_out, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }

    // --- delay size mismatch ---
    {
        std::vector<arma::Cube<double>> dl_bad(2);
        dl_bad[0].set_size(1, 1, 1); dl_bad[0].fill(0.0);
        dl_bad[1].set_size(1, 1, 1); dl_bad[1].fill(0.0);
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre, cim, dl_bad, freq_in, freq_out, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }

    // --- Empty coefficient cubes ---
    {
        std::vector<arma::Cube<double>> cre_e(1), cim_e(1);
        cre_e[0].set_size(0, 1, 1);
        cim_e[0].set_size(0, 1, 1);
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre_e, cim_e, dl, freq_in, freq_out, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }

    // --- Inconsistent coeff_re dimensions across frequencies ---
    {
        arma::Col<double> fin2 = {1.0e9, 2.0e9};
        std::vector<arma::Cube<double>> cre2(2), cim2(2), dl2(2);
        cre2[0].set_size(1, 1, 1); cre2[0].fill(1.0);
        cre2[1].set_size(2, 1, 1); cre2[1].fill(1.0); // Different n_rx
        cim2[0].set_size(1, 1, 1); cim2[0].fill(0.0);
        cim2[1].set_size(2, 1, 1); cim2[1].fill(0.0);
        dl2[0].set_size(1, 1, 1); dl2[0].fill(0.0);
        dl2[1].set_size(1, 1, 1); dl2[1].fill(0.0);
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre2, cim2, dl2, fin2, freq_out, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }

    // --- Inconsistent coeff_im dimensions ---
    {
        arma::Col<double> fin2 = {1.0e9, 2.0e9};
        std::vector<arma::Cube<double>> cre2(2), cim2(2), dl2(2);
        cre2[0].set_size(1, 1, 1); cre2[0].fill(1.0);
        cre2[1].set_size(1, 1, 1); cre2[1].fill(1.0);
        cim2[0].set_size(1, 1, 1); cim2[0].fill(0.0);
        cim2[1].set_size(1, 1, 2); cim2[1].fill(0.0); // Different n_path
        dl2[0].set_size(1, 1, 1); dl2[0].fill(0.0);
        dl2[1].set_size(1, 1, 1); dl2[1].fill(0.0);
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre2, cim2, dl2, fin2, freq_out, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }

    // --- Invalid delay[0] shape ---
    {
        std::vector<arma::Cube<double>> dl_bad(1);
        dl_bad[0].set_size(2, 2, 1); dl_bad[0].fill(0.0); // n_rx=1,n_tx=1 but delay is 2x2
        CHECK_THROWS_AS(quadriga_lib::baseband_freq_response_multi(cre, cim, dl_bad, freq_in, freq_out, &Hr, &Hi,
                         (arma::Cube<std::complex<double>> *)nullptr, true), std::invalid_argument);
    }
}

// ================================================================================================
// Zero paths - output should be zeros
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Zero paths")
{
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {1.0e9, 2.0e9, 3.0e9};
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(2, 2, 0);
    cim[0].set_size(2, 2, 0);
    dl[0].set_size(1, 1, 0);

    arma::Cube<double> Hr, Hi;
    arma::Cube<std::complex<double>> Hc;

    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi, &Hc, false);

    REQUIRE(Hr.n_rows == 2);
    REQUIRE(Hr.n_cols == 2);
    REQUIRE(Hr.n_slices == 3);
    CHECK(arma::accu(arma::abs(Hr)) == 0.0);
    CHECK(arma::accu(arma::abs(Hi)) == 0.0);

    REQUIRE(Hc.n_rows == 2);
    REQUIRE(Hc.n_cols == 2);
    REQUIRE(Hc.n_slices == 3);
}

// ================================================================================================
// Single input frequency, single path, zero delay - simplest case
// H(f_out) = coeff * exp(-j*2*pi*f_out*0) = coeff at all output carriers
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Single freq, single path, zero delay, no phase removal")
{
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {0.5e9, 1.0e9, 2.0e9};

    double amp = 0.7, phi = 0.3;
    std::vector<arma::Cube<double>> cre, cim;
    build_flat_envelope(1, 1, 1, 1, amp, phi, cre, cim);

    arma::Col<double> tau = {0.0};
    auto dl = build_planar_delays(1, 1, tau);

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    REQUIRE(Hr.n_slices == 3);
    double expected_re = amp * std::cos(phi);
    double expected_im = amp * std::sin(phi);

    for (arma::uword k = 0; k < 3; ++k)
    {
        CHECK(std::abs(Hr(0, 0, k) - expected_re) < 1e-7);
        CHECK(std::abs(Hi(0, 0, k) - expected_im) < 1e-7);
    }
}

// ================================================================================================
// Single input frequency, single path, nonzero delay - verify delay phase rotation
// H(f_out) = amp * exp(j*env_phase) * exp(-j*2*pi*f_out*tau)
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Single freq, nonzero delay, no phase removal")
{
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {1.0e9, 1.5e9, 2.0e9};

    double amp = 1.0, phi = 0.0;
    std::vector<arma::Cube<double>> cre, cim;
    build_flat_envelope(1, 1, 1, 1, amp, phi, cre, cim);

    double tau_val = 10.0e-9; // 10 ns
    arma::Col<double> tau = {tau_val};
    auto dl = build_planar_delays(1, 1, tau);

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    for (arma::uword k = 0; k < 3; ++k)
    {
        double f = freq_out(k);
        double expected_phase = -6.283185307179586 * f * tau_val;
        double expected_re = amp * std::cos(expected_phase);
        double expected_im = amp * std::sin(expected_phase);
        CHECK(std::abs(Hr(0, 0, k) - expected_re) < 1e-7);
        CHECK(std::abs(Hi(0, 0, k) - expected_im) < 1e-7);
    }
}

// ================================================================================================
// Two-path superposition - verify accumulation
// H(f) = A1*exp(-j*2*pi*f*tau1) + A2*exp(-j*2*pi*f*tau2)
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Two-path superposition, no phase removal")
{
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {1.0e9, 1.25e9, 1.5e9, 2.0e9};

    double A1 = 1.0, A2 = 0.5;
    double tau1 = 5.0e-9, tau2 = 20.0e-9;

    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 2);
    cim[0].set_size(1, 1, 2);
    cre[0](0, 0, 0) = A1;  cim[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = A2;  cim[0](0, 0, 1) = 0.0;
    dl[0].set_size(1, 1, 2);
    dl[0](0, 0, 0) = tau1;
    dl[0](0, 0, 1) = tau2;

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    for (arma::uword k = 0; k < freq_out.n_elem; ++k)
    {
        double f = freq_out(k);
        double phase1 = -6.283185307179586 * f * tau1;
        double phase2 = -6.283185307179586 * f * tau2;
        double expected_re = A1 * std::cos(phase1) + A2 * std::cos(phase2);
        double expected_im = A1 * std::sin(phase1) + A2 * std::sin(phase2);
        CHECK(std::abs(Hr(0, 0, k) - expected_re) < 1e-7);
        CHECK(std::abs(Hi(0, 0, k) - expected_im) < 1e-7);
    }
}

// ================================================================================================
// SLERP interpolation - magnitude varies linearly across 2 input frequencies
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - SLERP magnitude interpolation, no phase removal")
{
    arma::Col<double> freq_in = {1.0e9, 2.0e9};
    arma::Col<double> freq_out = {1.0e9, 1.25e9, 1.5e9, 1.75e9, 2.0e9};

    // Path with zero delay: magnitude 1.0 at 1 GHz, 3.0 at 2 GHz, phase zero
    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
    for (int f = 0; f < 2; ++f)
    {
        cre[f].set_size(1, 1, 1);
        cim[f].set_size(1, 1, 1);
        cim[f].fill(0.0);
        dl[f].set_size(1, 1, 1);
        dl[f].fill(0.0);
    }
    cre[0](0, 0, 0) = 1.0; // mag=1.0 at 1 GHz
    cre[1](0, 0, 0) = 3.0; // mag=3.0 at 2 GHz

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // With zero delay and zero phase, H(f) = interpolated_magnitude
    // Linear interpolation: mag(f) = 1.0 + (f - 1e9) / 1e9 * 2.0
    for (arma::uword k = 0; k < freq_out.n_elem; ++k)
    {
        double t = (freq_out(k) - 1.0e9) / 1.0e9;
        double expected_mag = 1.0 + t * 2.0;
        CHECK(std::abs(Hr(0, 0, k) - expected_mag) < 5e-7);
        CHECK(std::abs(Hi(0, 0, k)) < 1e-7);
    }
}

// ================================================================================================
// Constant extrapolation - output freq outside input range
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Constant extrapolation outside input range")
{
    arma::Col<double> freq_in = {1.0e9, 2.0e9};
    // Output extends below and above the input range
    arma::Col<double> freq_out = {0.5e9, 1.0e9, 1.5e9, 2.0e9, 3.0e9};

    // Amplitudes: 1.0 at 1 GHz, 2.0 at 2 GHz, zero delay and phase
    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
    for (int f = 0; f < 2; ++f)
    {
        cre[f].set_size(1, 1, 1);
        cim[f].set_size(1, 1, 1);
        cim[f].fill(0.0);
        dl[f].set_size(1, 1, 1);
        dl[f].fill(0.0);
    }
    cre[0](0, 0, 0) = 1.0;
    cre[1](0, 0, 0) = 2.0;

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // Below range: clamped to 1.0
    CHECK(std::abs(Hr(0, 0, 0) - 1.0) < 5e-7);
    // At 1 GHz: exactly 1.0
    CHECK(std::abs(Hr(0, 0, 1) - 1.0) < 5e-7);
    // At 1.5 GHz: interpolated to 1.5
    CHECK(std::abs(Hr(0, 0, 2) - 1.5) < 5e-7);
    // At 2 GHz: exactly 2.0
    CHECK(std::abs(Hr(0, 0, 3) - 2.0) < 5e-7);
    // Above range: clamped to 2.0
    CHECK(std::abs(Hr(0, 0, 4) - 2.0) < 5e-7);
}

// ================================================================================================
// remove_delay_phase - baked-in phase undo/re-apply
// Construct envelopes, bake in delay phase, then verify that remove_delay_phase=true
// produces the same result as the pure-envelope case with remove_delay_phase=false
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - remove_delay_phase correctness")
{
    arma::Col<double> freq_in = {1.0e9, 2.0e9, 3.0e9};
    arma::Col<double> freq_out = arma::linspace<arma::Col<double>>(1.0e9, 3.0e9, 32);

    double amp = 0.8, env_phi = 0.5;
    double tau_val = 50.0e-9; // 50 ns - significant delay

    // --- Build pure envelope (no baked-in delay phase) ---
    std::vector<arma::Cube<double>> cre_env, cim_env;
    build_flat_envelope(1, 1, 1, 3, amp, env_phi, cre_env, cim_env);
    arma::Col<double> tau = {tau_val};
    auto dl = build_planar_delays(1, 3, tau);

    arma::Cube<double> Hr_ref, Hi_ref;
    quadriga_lib::baseband_freq_response_multi(cre_env, cim_env, dl, freq_in, freq_out, &Hr_ref, &Hi_ref,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // --- Build baked version (simulate get_channels_multifreq output) ---
    std::vector<arma::Cube<double>> cre_baked, cim_baked;
    build_flat_envelope(1, 1, 1, 3, amp, env_phi, cre_baked, cim_baked);
    bake_delay_phase(cre_baked, cim_baked, dl, freq_in);

    arma::Cube<double> Hr_test, Hi_test;
    quadriga_lib::baseband_freq_response_multi(cre_baked, cim_baked, dl, freq_in, freq_out, &Hr_test, &Hi_test,
                                               (arma::Cube<std::complex<double>> *)nullptr, true);

    // Both should produce identical results
    REQUIRE(Hr_ref.n_slices == 32);
    REQUIRE(Hr_test.n_slices == 32);
    for (arma::uword k = 0; k < 32; ++k)
    {
        CHECK(std::abs(Hr_test(0, 0, k) - Hr_ref(0, 0, k)) < 1e-7);
        CHECK(std::abs(Hi_test(0, 0, k) - Hi_ref(0, 0, k)) < 1e-7);
    }
}

// ================================================================================================
// remove_delay_phase with multiple paths and varying amplitudes
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - remove_delay_phase multi-path")
{
    arma::Col<double> freq_in = {1.0e9, 1.5e9, 2.0e9};
    arma::Col<double> freq_out = arma::linspace<arma::Col<double>>(1.0e9, 2.0e9, 16);

    arma::uword n_path = 3;
    arma::uword n_freq_in = 3;

    // Different amplitudes per path, frequency-flat envelopes
    double amps[3] = {1.0, 0.5, 0.3};
    double phis[3] = {0.0, 0.7, -0.4};
    double taus[3] = {10.0e-9, 30.0e-9, 80.0e-9};

    // Build pure envelope
    std::vector<arma::Cube<double>> cre_env(n_freq_in), cim_env(n_freq_in);
    for (arma::uword f = 0; f < n_freq_in; ++f)
    {
        cre_env[f].set_size(1, 1, n_path);
        cim_env[f].set_size(1, 1, n_path);
        for (arma::uword p = 0; p < n_path; ++p)
        {
            cre_env[f](0, 0, p) = amps[p] * std::cos(phis[p]);
            cim_env[f](0, 0, p) = amps[p] * std::sin(phis[p]);
        }
    }

    arma::Col<double> tau(n_path);
    tau(0) = taus[0]; tau(1) = taus[1]; tau(2) = taus[2];
    auto dl = build_planar_delays(n_path, n_freq_in, tau);

    // Reference: pure envelope, no phase removal
    arma::Cube<double> Hr_ref, Hi_ref;
    quadriga_lib::baseband_freq_response_multi(cre_env, cim_env, dl, freq_in, freq_out, &Hr_ref, &Hi_ref,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // Test: baked + phase removal
    std::vector<arma::Cube<double>> cre_baked = cre_env;
    std::vector<arma::Cube<double>> cim_baked = cim_env;
    // Deep copy needed since bake_delay_phase modifies in-place
    for (arma::uword f = 0; f < n_freq_in; ++f)
    {
        cre_baked[f] = arma::Cube<double>(cre_env[f]);
        cim_baked[f] = arma::Cube<double>(cim_env[f]);
    }
    bake_delay_phase(cre_baked, cim_baked, dl, freq_in);

    arma::Cube<double> Hr_test, Hi_test;
    quadriga_lib::baseband_freq_response_multi(cre_baked, cim_baked, dl, freq_in, freq_out, &Hr_test, &Hi_test,
                                               (arma::Cube<std::complex<double>> *)nullptr, true);

    for (arma::uword k = 0; k < freq_out.n_elem; ++k)
    {
        CHECK(std::abs(Hr_test(0, 0, k) - Hr_ref(0, 0, k)) < 1e-7);
        CHECK(std::abs(Hi_test(0, 0, k) - Hi_ref(0, 0, k)) < 1e-7);
    }
}

// ================================================================================================
// SLERP phase interpolation - verify phase unwrapping across pi boundary
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Phase unwrapping across pi boundary")
{
    arma::Col<double> freq_in = {1.0e9, 2.0e9};
    arma::Col<double> freq_out = {1.5e9};

    // Coefficient at 1 GHz: phase = +2.8 rad (near +pi)
    // Coefficient at 2 GHz: phase = -2.8 rad (near -pi)
    // The phase wraps across the pi boundary; correct midpoint is pi (or -pi), not 0
    double amp = 1.0;
    double phi_lo = 2.8, phi_hi = -2.8;

    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
    for (int f = 0; f < 2; ++f)
    {
        cre[f].set_size(1, 1, 1);
        cim[f].set_size(1, 1, 1);
        dl[f].set_size(1, 1, 1);
        dl[f].fill(0.0);
    }
    cre[0](0, 0, 0) = amp * std::cos(phi_lo);
    cim[0](0, 0, 0) = amp * std::sin(phi_lo);
    cre[1](0, 0, 0) = amp * std::cos(phi_hi);
    cim[1](0, 0, 0) = amp * std::sin(phi_hi);

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // Unwrapped phase at 2 GHz: 2.8 + wrap(-2.8 - 2.8) = 2.8 + (-5.6 + 2*pi) ≈ 2.8 + 0.683 = 3.483
    // Midpoint phase: (2.8 + 3.483) / 2 ≈ 3.1416 ≈ pi
    double unwrapped_hi = phi_lo + (phi_hi - phi_lo + 6.283185307179586); // Shortest arc through +pi
    // Actually let's compute it properly
    double d = phi_hi - phi_lo; // -5.6
    while (d > 3.141592653589793) d -= 6.283185307179586;
    while (d < -3.141592653589793) d += 6.283185307179586;
    double unwrapped_phi_hi = phi_lo + d; // 2.8 + 0.6832 = 3.4832
    double mid_phase = 0.5 * (phi_lo + unwrapped_phi_hi);

    double expected_re = amp * std::cos(mid_phase);
    double expected_im = amp * std::sin(mid_phase);

    CHECK(std::abs(Hr(0, 0, 0) - expected_re) < 1e-7);
    CHECK(std::abs(Hi(0, 0, 0) - expected_im) < 1e-7);

    // The result should be near phase = pi, i.e. Re ≈ -1.0
    CHECK(Hr(0, 0, 0) < -0.99);
}

// ================================================================================================
// Planar wave vs spherical wave delays
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Spherical wave delays")
{
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {1.0e9, 2.0e9};

    // 2 RX, 1 TX, 1 path - different delay per antenna
    arma::uword n_rx = 2, n_tx = 1, n_path = 1;
    std::vector<arma::Cube<double>> cre(1), cim(1);
    cre[0].set_size(n_rx, n_tx, n_path); cre[0].fill(1.0);
    cim[0].set_size(n_rx, n_tx, n_path); cim[0].fill(0.0);

    double tau_rx0 = 5.0e-9, tau_rx1 = 15.0e-9;
    arma::Cube<double> tau_cube(n_rx, n_tx, n_path);
    tau_cube(0, 0, 0) = tau_rx0;
    tau_cube(1, 0, 0) = tau_rx1;
    auto dl = build_spherical_delays(n_rx, n_tx, n_path, 1, tau_cube);

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    REQUIRE(Hr.n_rows == 2);
    REQUIRE(Hr.n_cols == 1);
    REQUIRE(Hr.n_slices == 2);

    // Verify different phase rotations for each RX antenna at each output freq
    for (arma::uword k = 0; k < 2; ++k)
    {
        double f = freq_out(k);

        double phase0 = -6.283185307179586 * f * tau_rx0;
        CHECK(std::abs(Hr(0, 0, k) - std::cos(phase0)) < 1e-7);
        CHECK(std::abs(Hi(0, 0, k) - std::sin(phase0)) < 1e-7);

        double phase1 = -6.283185307179586 * f * tau_rx1;
        CHECK(std::abs(Hr(1, 0, k) - std::cos(phase1)) < 1e-7);
        CHECK(std::abs(Hi(1, 0, k) - std::sin(phase1)) < 1e-7);
    }
}

// ================================================================================================
// MIMO (2x2) - verify output dimensions and per-link processing
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - 2x2 MIMO")
{
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {1.0e9};

    arma::uword n_rx = 2, n_tx = 2, n_path = 1;

    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(n_rx, n_tx, n_path);
    cim[0].set_size(n_rx, n_tx, n_path);
    dl[0].set_size(1, 1, n_path);
    dl[0](0, 0, 0) = 0.0;

    // Set different amplitudes for each TX-RX pair
    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.0;  // RX0-TX0
    cre[0](1, 0, 0) = 2.0; cim[0](1, 0, 0) = 0.0;  // RX1-TX0
    cre[0](0, 1, 0) = 3.0; cim[0](0, 1, 0) = 0.0;  // RX0-TX1
    cre[0](1, 1, 0) = 4.0; cim[0](1, 1, 0) = 0.0;  // RX1-TX1

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    REQUIRE(Hr.n_rows == 2);
    REQUIRE(Hr.n_cols == 2);
    REQUIRE(Hr.n_slices == 1);

    CHECK(std::abs(Hr(0, 0, 0) - 1.0) < 1e-7);
    CHECK(std::abs(Hr(1, 0, 0) - 2.0) < 1e-7);
    CHECK(std::abs(Hr(0, 1, 0) - 3.0) < 1e-7);
    CHECK(std::abs(Hr(1, 1, 0) - 4.0) < 1e-7);
}

// ================================================================================================
// Complex output matches real + j*imag
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Complex output consistency")
{
    arma::Col<double> freq_in = {1.0e9, 2.0e9};
    arma::Col<double> freq_out = arma::linspace<arma::Col<double>>(1.0e9, 2.0e9, 8);

    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
    for (int f = 0; f < 2; ++f)
    {
        cre[f].set_size(1, 1, 2);
        cim[f].set_size(1, 1, 2);
        dl[f].set_size(1, 1, 2);
        dl[f](0, 0, 0) = 10.0e-9;
        dl[f](0, 0, 1) = 30.0e-9;
    }
    cre[0](0, 0, 0) = 1.0;  cim[0](0, 0, 0) = 0.2;
    cre[0](0, 0, 1) = 0.5;  cim[0](0, 0, 1) = -0.3;
    cre[1](0, 0, 0) = 0.8;  cim[1](0, 0, 0) = 0.4;
    cre[1](0, 0, 1) = 0.6;  cim[1](0, 0, 1) = -0.1;

    arma::Cube<double> Hr, Hi;
    arma::Cube<std::complex<double>> Hc;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi, &Hc, false);

    REQUIRE(Hc.n_slices == 8);
    for (arma::uword k = 0; k < 8; ++k)
    {
        CHECK(std::abs(Hc(0, 0, k).real() - Hr(0, 0, k)) < 1e-7);
        CHECK(std::abs(Hc(0, 0, k).imag() - Hi(0, 0, k)) < 1e-7);
    }
}

// ================================================================================================
// Only request subset of outputs (hmat_re only, hmat_im only, hmat only)
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Selective output pointers")
{
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {1.0e9};
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 1); cre[0].fill(1.0);
    cim[0].set_size(1, 1, 1); cim[0].fill(0.0);
    dl[0].set_size(1, 1, 1);  dl[0].fill(0.0);

    // Only hmat_re
    {
        arma::Cube<double> Hr;
        quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr,
                                                   (arma::Cube<double> *)nullptr,
                                                   (arma::Cube<std::complex<double>> *)nullptr, false);
        REQUIRE(Hr.n_slices == 1);
        CHECK(std::abs(Hr(0, 0, 0) - 1.0) < 1e-7);
    }

    // Only hmat_im
    {
        arma::Cube<double> Hi;
        quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out,
                                                   (arma::Cube<double> *)nullptr, &Hi,
                                                   (arma::Cube<std::complex<double>> *)nullptr, false);
        REQUIRE(Hi.n_slices == 1);
        CHECK(std::abs(Hi(0, 0, 0)) < 1e-7);
    }

    // Only hmat (complex)
    {
        arma::Cube<std::complex<double>> Hc;
        quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out,
                                                   (arma::Cube<double> *)nullptr,
                                                   (arma::Cube<double> *)nullptr, &Hc, false);
        REQUIRE(Hc.n_slices == 1);
        CHECK(std::abs(Hc(0, 0, 0).real() - 1.0) < 1e-7);
        CHECK(std::abs(Hc(0, 0, 0).imag()) < 1e-7);
    }
}

// ================================================================================================
// Float precision template instantiation
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Float precision")
{
    arma::Col<float> freq_in = {1.0e9f, 2.0e9f};
    arma::Col<float> freq_out = {1.0e9f, 1.5e9f, 2.0e9f};

    std::vector<arma::Cube<float>> cre(2), cim(2), dl(2);
    for (int f = 0; f < 2; ++f)
    {
        cre[f].set_size(1, 1, 1);
        cim[f].set_size(1, 1, 1);
        cim[f].fill(0.0f);
        dl[f].set_size(1, 1, 1);
        dl[f].fill(0.0f);
    }
    cre[0](0, 0, 0) = 1.0f;
    cre[1](0, 0, 0) = 2.0f;

    arma::Cube<float> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<float>> *)nullptr, false);

    REQUIRE(Hr.n_slices == 3);
    CHECK(std::abs(Hr(0, 0, 0) - 1.0f) < 1e-5f);
    CHECK(std::abs(Hr(0, 0, 1) - 1.5f) < 1e-5f);
    CHECK(std::abs(Hr(0, 0, 2) - 2.0f) < 1e-5f);
}

// ================================================================================================
// Multiple input frequencies (3 segments) - verify correct segment selection
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Three input frequencies, segment selection")
{
    arma::Col<double> freq_in = {1.0e9, 2.0e9, 4.0e9};
    // Test points: in first segment, at boundary, in second segment
    arma::Col<double> freq_out = {1.5e9, 2.0e9, 3.0e9};

    // Amplitudes: 1.0, 3.0, 7.0 at each input freq (zero delay, zero phase)
    std::vector<arma::Cube<double>> cre(3), cim(3), dl(3);
    for (int f = 0; f < 3; ++f)
    {
        cre[f].set_size(1, 1, 1);
        cim[f].set_size(1, 1, 1);
        cim[f].fill(0.0);
        dl[f].set_size(1, 1, 1);
        dl[f].fill(0.0);
    }
    cre[0](0, 0, 0) = 1.0;
    cre[1](0, 0, 0) = 3.0;
    cre[2](0, 0, 0) = 7.0;

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // At 1.5 GHz: first segment, t=0.5, mag = 1.0 + 0.5*2.0 = 2.0
    CHECK(std::abs(Hr(0, 0, 0) - 2.0) < 5e-7);

    // At 2.0 GHz: boundary, t=1.0 in first segment, mag = 3.0
    CHECK(std::abs(Hr(0, 0, 1) - 3.0) < 5e-7);

    // At 3.0 GHz: second segment [2.0, 4.0], t=0.5, mag = 3.0 + 0.5*4.0 = 5.0
    CHECK(std::abs(Hr(0, 0, 2) - 5.0) < 5e-7);
}

// ================================================================================================
// Physical consistency: verify against direct DFT for narrowband case
// When input coefficients are identical at all input frequencies and remove_delay_phase=false,
// the result should match the classic DFT baseband_freq_response (up to phase convention)
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Consistency with baseband_freq_response")
{
    double center_freq = 2.0e9;
    double bandwidth = 100.0e6; // 100 MHz
    arma::uword n_carrier = 16;

    // Build pilot grid for narrowband function: carrier offsets relative to bandwidth
    arma::Col<double> pilot_grid(n_carrier);
    for (arma::uword k = 0; k < n_carrier; ++k)
        pilot_grid(k) = (double)k / (double)n_carrier;

    // Build freq_out for multi function: absolute frequencies
    arma::Col<double> freq_out(n_carrier);
    for (arma::uword k = 0; k < n_carrier; ++k)
        freq_out(k) = center_freq + bandwidth * pilot_grid(k);

    // Single input frequency at center_freq (no interpolation needed)
    arma::Col<double> freq_in = {center_freq};

    // Channel: 2 RX, 1 TX, 2 paths
    arma::uword n_rx = 2, n_tx = 1, n_path = 2;
    double tau1 = 0.0e-9, tau2 = 50.0e-9; // relative delays

    // Narrowband coefficients with delay phase baked in at center_freq
    arma::Cube<double> coeff_re_nb(n_rx, n_tx, n_path);
    arma::Cube<double> coeff_im_nb(n_rx, n_tx, n_path);
    arma::Cube<double> delay_nb(1, 1, n_path);
    delay_nb(0, 0, 0) = tau1;
    delay_nb(0, 0, 1) = tau2;

    // Envelope: path 1 amp=1.0, path 2 amp=0.5
    double A1 = 1.0, A2 = 0.5;

    // Bake delay phase at center_freq into narrowband coefficients
    double phase1 = -6.283185307179586 * center_freq * tau1;
    double phase2 = -6.283185307179586 * center_freq * tau2;
    for (arma::uword r = 0; r < n_rx; ++r)
    {
        coeff_re_nb(r, 0, 0) = A1 * std::cos(phase1);
        coeff_im_nb(r, 0, 0) = A1 * std::sin(phase1);
        coeff_re_nb(r, 0, 1) = A2 * std::cos(phase2);
        coeff_im_nb(r, 0, 1) = A2 * std::sin(phase2);
    }

    // --- Run narrowband DFT ---
    arma::Cube<double> Hr_nb, Hi_nb;
    quadriga_lib::baseband_freq_response(&coeff_re_nb, &coeff_im_nb, &delay_nb, &pilot_grid,
                                         bandwidth, &Hr_nb, &Hi_nb);

    // --- Run multi with same baked coefficients, remove_delay_phase=true ---
    std::vector<arma::Cube<double>> cre_multi(1), cim_multi(1), dl_multi(1);
    cre_multi[0] = coeff_re_nb;
    cim_multi[0] = coeff_im_nb;
    dl_multi[0] = delay_nb;

    arma::Cube<double> Hr_multi, Hi_multi;
    quadriga_lib::baseband_freq_response_multi(cre_multi, cim_multi, dl_multi, freq_in, freq_out,
                                               &Hr_multi, &Hi_multi,
                                               (arma::Cube<std::complex<double>> *)nullptr, true);

    REQUIRE(Hr_nb.n_slices == n_carrier);
    REQUIRE(Hr_multi.n_slices == n_carrier);

    // The power |H|^2 should match between both methods
    for (arma::uword k = 0; k < n_carrier; ++k)
    {
        for (arma::uword r = 0; r < n_rx; ++r)
        {
            double power_nb = Hr_nb(r, 0, k) * Hr_nb(r, 0, k) + Hi_nb(r, 0, k) * Hi_nb(r, 0, k);
            double power_multi = Hr_multi(r, 0, k) * Hr_multi(r, 0, k) + Hi_multi(r, 0, k) * Hi_multi(r, 0, k);
            CHECK(std::abs(power_nb - power_multi) / (power_nb + 1e-30) < 1e-5);
        }
    }
}

// ================================================================================================
// remove_delay_phase with spherical wave delays (per-antenna delays)
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - remove_delay_phase with spherical delays")
{
    arma::Col<double> freq_in = {1.0e9, 2.0e9};
    arma::Col<double> freq_out = arma::linspace<arma::Col<double>>(1.0e9, 2.0e9, 8);

    arma::uword n_rx = 2, n_tx = 1, n_path = 1;

    // Different delay per RX antenna
    arma::Cube<double> tau_cube(n_rx, n_tx, n_path);
    tau_cube(0, 0, 0) = 10.0e-9;
    tau_cube(1, 0, 0) = 40.0e-9;

    // Build pure envelopes
    std::vector<arma::Cube<double>> cre_env(2), cim_env(2);
    for (int f = 0; f < 2; ++f)
    {
        cre_env[f].set_size(n_rx, n_tx, n_path); cre_env[f].fill(1.0);
        cim_env[f].set_size(n_rx, n_tx, n_path); cim_env[f].fill(0.0);
    }
    auto dl = build_spherical_delays(n_rx, n_tx, n_path, 2, tau_cube);

    // Reference: envelope, no removal
    arma::Cube<double> Hr_ref, Hi_ref;
    quadriga_lib::baseband_freq_response_multi(cre_env, cim_env, dl, freq_in, freq_out, &Hr_ref, &Hi_ref,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // Bake and test
    std::vector<arma::Cube<double>> cre_baked(2), cim_baked(2);
    for (int f = 0; f < 2; ++f)
    {
        cre_baked[f] = arma::Cube<double>(cre_env[f]);
        cim_baked[f] = arma::Cube<double>(cim_env[f]);
    }
    bake_delay_phase(cre_baked, cim_baked, dl, freq_in);

    arma::Cube<double> Hr_test, Hi_test;
    quadriga_lib::baseband_freq_response_multi(cre_baked, cim_baked, dl, freq_in, freq_out, &Hr_test, &Hi_test,
                                               (arma::Cube<std::complex<double>> *)nullptr, true);

    for (arma::uword k = 0; k < freq_out.n_elem; ++k)
        for (arma::uword r = 0; r < n_rx; ++r)
        {
            CHECK(std::abs(Hr_test(r, 0, k) - Hr_ref(r, 0, k)) < 1e-7);
            CHECK(std::abs(Hi_test(r, 0, k) - Hi_ref(r, 0, k)) < 1e-7);
        }
}

// ================================================================================================
// Frequency-dependent envelope (amplitude ramp) with baked delay phase
// Verifies full pipeline: undo → SLERP → re-apply
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Frequency-dependent envelope with baked phase")
{
    arma::Col<double> freq_in = {1.0e9, 2.0e9, 3.0e9};
    arma::Col<double> freq_out = arma::linspace<arma::Col<double>>(1.0e9, 3.0e9, 64);

    double tau_val = 100.0e-9; // Large delay to stress-test phase undo

    // Envelope amplitude ramps from 1.0 to 3.0 across frequency, phase = 0
    std::vector<arma::Cube<double>> cre_env(3), cim_env(3);
    double env_amps[3] = {1.0, 2.0, 3.0};
    for (int f = 0; f < 3; ++f)
    {
        cre_env[f].set_size(1, 1, 1);
        cim_env[f].set_size(1, 1, 1);
        cre_env[f](0, 0, 0) = env_amps[f];
        cim_env[f](0, 0, 0) = 0.0;
    }

    arma::Col<double> tau = {tau_val};
    auto dl = build_planar_delays(1, 3, tau);

    // Reference: envelope only
    arma::Cube<double> Hr_ref, Hi_ref;
    quadriga_lib::baseband_freq_response_multi(cre_env, cim_env, dl, freq_in, freq_out, &Hr_ref, &Hi_ref,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // Bake delay phase (100 ns * 1 GHz = 100 full rotations per input step — very fast)
    std::vector<arma::Cube<double>> cre_baked(3), cim_baked(3);
    for (int f = 0; f < 3; ++f)
    {
        cre_baked[f] = arma::Cube<double>(cre_env[f]);
        cim_baked[f] = arma::Cube<double>(cim_env[f]);
    }
    bake_delay_phase(cre_baked, cim_baked, dl, freq_in);

    arma::Cube<double> Hr_test, Hi_test;
    quadriga_lib::baseband_freq_response_multi(cre_baked, cim_baked, dl, freq_in, freq_out, &Hr_test, &Hi_test,
                                               (arma::Cube<std::complex<double>> *)nullptr, true);

    // Verify magnitude and phase match
    for (arma::uword k = 0; k < freq_out.n_elem; ++k)
    {
        double power_ref = Hr_ref(0, 0, k) * Hr_ref(0, 0, k) + Hi_ref(0, 0, k) * Hi_ref(0, 0, k);
        double power_test = Hr_test(0, 0, k) * Hr_test(0, 0, k) + Hi_test(0, 0, k) * Hi_test(0, 0, k);
        CHECK(std::abs(power_ref - power_test) / (power_ref + 1e-30) < 1e-7);
        CHECK(std::abs(Hr_test(0, 0, k) - Hr_ref(0, 0, k)) < 1e-7);
        CHECK(std::abs(Hi_test(0, 0, k) - Hi_ref(0, 0, k)) < 1e-7);
    }
}

// ================================================================================================
// Output size auto-allocation
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Output auto-resize")
{
    arma::Col<double> freq_in = {1.0e9};
    arma::Col<double> freq_out = {1.0e9, 2.0e9, 3.0e9, 4.0e9};

    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(2, 3, 1); cre[0].fill(1.0);
    cim[0].set_size(2, 3, 1); cim[0].fill(0.0);
    dl[0].set_size(1, 1, 1);  dl[0].fill(0.0);

    // Pass uninitialized output cubes (default constructed, size 0x0x0)
    arma::Cube<double> Hr, Hi;
    arma::Cube<std::complex<double>> Hc;

    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi, &Hc, false);

    REQUIRE(Hr.n_rows == 2);
    REQUIRE(Hr.n_cols == 3);
    REQUIRE(Hr.n_slices == 4);
    REQUIRE(Hi.n_rows == 2);
    REQUIRE(Hi.n_cols == 3);
    REQUIRE(Hi.n_slices == 4);
    REQUIRE(Hc.n_rows == 2);
    REQUIRE(Hc.n_cols == 3);
    REQUIRE(Hc.n_slices == 4);
}

// ================================================================================================
// Large-delay acoustic scenario (speed of sound ≈ 343 m/s → large delays in seconds)
// Verify that double-precision phase computation handles this correctly
// ================================================================================================

TEST_CASE("baseband_freq_response_multi - Acoustic scenario large delay")
{
    // Acoustic: speed of sound 343 m/s, 10 m distance → ~29.15 ms delay
    // At 1000 Hz carrier, phase = 2*pi*1000*0.02915 ≈ 183 rad
    double tau_val = 10.0 / 343.0; // ~29.15 ms

    arma::Col<double> freq_in = {500.0, 1000.0, 2000.0};
    arma::Col<double> freq_out = {500.0, 750.0, 1000.0, 1500.0, 2000.0};

    std::vector<arma::Cube<double>> cre(3), cim(3);
    for (int f = 0; f < 3; ++f)
    {
        cre[f].set_size(1, 1, 1); cre[f].fill(1.0);
        cim[f].set_size(1, 1, 1); cim[f].fill(0.0);
    }

    arma::Col<double> tau = {tau_val};
    auto dl = build_planar_delays(1, 3, tau);

    arma::Cube<double> Hr, Hi;
    quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, &Hr, &Hi,
                                               (arma::Cube<std::complex<double>> *)nullptr, false);

    // Verify power is unity at all carriers (|H(f)|^2 = 1 for unit envelope)
    for (arma::uword k = 0; k < freq_out.n_elem; ++k)
    {
        double power = Hr(0, 0, k) * Hr(0, 0, k) + Hi(0, 0, k) * Hi(0, 0, k);
        CHECK(std::abs(power - 1.0) < 5e-7);
    }

    // Verify phase at exact input frequency matches expected value
    // At f=1000 Hz (index 2): phase = -2*pi*1000*tau
    double expected_phase = -6.283185307179586 * 1000.0 * tau_val;
    double expected_re = std::cos(expected_phase);
    double expected_im = std::sin(expected_phase);
    CHECK(std::abs(Hr(0, 0, 2) - expected_re) < 1e-7);
    CHECK(std::abs(Hi(0, 0, 2) - expected_im) < 1e-7);
}