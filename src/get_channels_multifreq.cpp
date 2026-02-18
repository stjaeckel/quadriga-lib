// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <limits>

#include "quadriga_arrayant.hpp"
#include "qd_arrayant_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_channels_multifreq
Calculate channel coefficients for spherical waves across multiple frequencies

## Description:
- Extends `get_channels_spherical` to support frequency-dependent antenna patterns, path gains,
  and polarization transfer (Jones) matrices across multiple output frequencies.
- **Geometry is computed once**: departure angles, arrival angles, element-resolved path delays, and LOS
  path detection are frequency-independent and reused for all output frequencies. This avoids redundant
  trigonometry and distance calculations.
- **Four frequency grids** are aligned by interpolation:
  1. | TX array frequencies (defined by `tx_array[i].center_frequency`)
  2. | RX array frequencies (defined by `rx_array[i].center_frequency`)
  3. | Input sample frequencies (`freq_in`) at which `path_gain` and `M` are provided
  4. | Target output frequencies (`freq_out`) at which coefficients and delays are returned
- For each output frequency, TX and RX antenna patterns are interpolated from their respective
  multi-frequency vectors using spherical interpolation (SLERP) with linear fallback, the same
  algorithm used in `arrayant_interpolate_multi`. The private `qd_arrayant_interpolate` function is
  called directly for maximum performance.
- Path gain is interpolated linearly across frequency. The Jones matrix `M` is interpolated using
  SLERP for each complex entry pair to preserve phase coherence.
- **Extrapolation** is handled by clamping to the nearest available frequency entry in all four grids.
- **Propagation speed** can be set to support both radio (speed of light, default) and acoustic
  (speed of sound, ~343 m/s) simulations. This affects wavelength, wave number, and delay calculations.
- The Jones matrix `M` supports two formats: 8 rows for full polarimetric
  (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH), or 2 rows for scalar pressure waves
  (ReVV, ImVV only), where VH, HV, and HH entries are implicitly zero.
- Antenna element coupling is applied using the coupling matrices from the first entry of each
  multi-frequency vector (consistent across all entries by `arrayant_is_valid_multi` constraints).
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::get_channels_multifreq(
        const std::vector<arrayant<dtype>> &tx_array,
        const std::vector<arrayant<dtype>> &rx_array,
        dtype Tx, dtype Ty, dtype Tz,
        dtype Tb, dtype Tt, dtype Th,
        dtype Rx, dtype Ry, dtype Rz,
        dtype Rb, dtype Rt, dtype Rh,
        const arma::Mat<dtype> &fbs_pos,
        const arma::Mat<dtype> &lbs_pos,
        const arma::Mat<dtype> &path_gain,
        const arma::Col<dtype> &path_length,
        const arma::Cube<dtype> &M,
        const arma::Col<dtype> &freq_in,
        const arma::Col<dtype> &freq_out,
        std::vector<arma::Cube<dtype>> &coeff_re,
        std::vector<arma::Cube<dtype>> &coeff_im,
        std::vector<arma::Cube<dtype>> &delay,
        bool use_absolute_delays = false,
        bool add_fake_los_path = false,
        dtype propagation_speed = dtype(299792458.0))
```

## Arguments:
- `const std::vector<arrayant<dtype>> &**tx_array**` (input)<br>
  Multi-frequency transmit array antenna vector. All entries must pass `arrayant_is_valid_multi`.

- `const std::vector<arrayant<dtype>> &**rx_array**` (input)<br>
  Multi-frequency receive array antenna vector. All entries must pass `arrayant_is_valid_multi`.

- `dtype **Tx**, **Ty**, **Tz**` (input)<br>
  Transmitter position in Cartesian coordinates [m].

- `dtype **Tb**, **Tt**, **Th**` (input)<br>
  Transmitter orientation (bank, tilt, heading) in [rad].

- `dtype **Rx**, **Ry**, **Rz**` (input)<br>
  Receiver position in Cartesian coordinates [m].

- `dtype **Rb**, **Rt**, **Rh**` (input)<br>
  Receiver orientation (bank, tilt, heading) in [rad].

- `const arma::Mat<dtype> &**fbs_pos**` (input)<br>
  First-bounce scatterer positions, Size: `[3, n_path]`.

- `const arma::Mat<dtype> &**lbs_pos**` (input)<br>
  Last-bounce scatterer positions, Size: `[3, n_path]`.

- `const arma::Mat<dtype> &**path_gain**` (input)<br>
  Path gain in linear scale, Size: `[n_path, n_freq_in]`. Each column corresponds to one input frequency.

- `const arma::Col<dtype> &**path_length**` (input)<br>
  Absolute path lengths from TX to RX phase center, Length: `n_path`.

- `const arma::Cube<dtype> &**M**` (input)<br>
  Polarization transfer matrix, Size: `[8, n_path, n_freq_in]` for full polarimetric or
  `[2, n_path, n_freq_in]` for scalar pressure. Each slice corresponds to one input frequency.
  Interleaved complex format: (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH) for 8 rows,
  or (ReVV, ImVV) for 2 rows.

- `const arma::Col<dtype> &**freq_in**` (input)<br>
  Input sample frequencies in [Hz] at which `path_gain` and `M` are defined, Length: `n_freq_in`.

- `const arma::Col<dtype> &**freq_out**` (input)<br>
  Target frequencies in [Hz] at which to compute output coefficients and delays, Length: `n_freq_out`.

- `std::vector<arma::Cube<dtype>> &**coeff_re**` (output)<br>
  Real part of channel coefficients. Vector of length `n_freq_out`, each cube of size `[n_rx, n_tx, n_path]`.

- `std::vector<arma::Cube<dtype>> &**coeff_im**` (output)<br>
  Imaginary part of channel coefficients. Same structure as `coeff_re`.

- `std::vector<arma::Cube<dtype>> &**delay**` (output)<br>
  Propagation delays in seconds. Same structure as `coeff_re`.

- `bool **use_absolute_delays** = false` (optional input)<br>
  If true, LOS delay is included in all paths. Default: `false`.

- `bool **add_fake_los_path** = false` (optional input)<br>
  Adds a zero-power LOS path if no LOS path was detected. Default: `false`.

- `dtype **propagation_speed** = 299792458.0` (optional input)<br>
  Wave propagation speed in [m/s]. Default is the speed of light for radio simulations.
  Set to ~343.0 for acoustic simulations in air.

## Example:
```
// Build a 2-way speaker as TX (source)
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto tx_woofer = quadriga_lib::generate_speaker<double>(
    "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
    0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
auto tx_tweeter = quadriga_lib::generate_speaker<double>(
    "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
    0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
auto tx = quadriga_lib::arrayant_concat_multi(tx_woofer, tx_tweeter);

// Omnidirectional microphone as RX (single entry, clamped for all frequencies)
std::vector<quadriga_lib::arrayant<double>> rx = { quadriga_lib::generate_arrayant_omni<double>() };

// Simple LOS path setup
arma::mat fbs = arma::mat({0.5, 0.0, 0.0}).t();
arma::mat lbs = arma::mat({0.5, 0.0, 0.0}).t();
arma::vec path_length = {1.0};               // 1 meter distance

// Frequency-flat path gain and scalar Jones matrix
arma::vec freq_in = {100.0, 10000.0};
arma::mat path_gain_mat(1, 2, arma::fill::ones);
arma::cube M_cube(2, 1, 2, arma::fill::zeros);
M_cube(0, 0, 0) = 1.0; M_cube(0, 0, 1) = 1.0;  // ReVV = 1 at both freqs

arma::vec freq_out = {200.0, 1000.0, 5000.0};
std::vector<arma::cube> coeff_re, coeff_im, delays;

quadriga_lib::get_channels_multifreq(tx, rx,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   // TX at origin
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,   // RX at (1,0,0)
    fbs_pos, lbs_pos, path_gain_mat, path_length, M_cube,
    freq_in, freq_out, coeff_re, coeff_im, delays,
    false, false, 343.0);             // Speed of sound for acoustics
```

## See also:
- <a href="#get_channels_spherical">get_channels_spherical</a>
- <a href="#arrayant_interpolate_multi">arrayant_interpolate_multi</a>
- <a href="#arrayant_concat_multi">arrayant_concat_multi</a>
- <a href="#generate_speaker">generate_speaker</a>
MD!*/

// ------------------------------------------------------------------------------------------------
// Spherical interpolation with linear fallback for a single complex value pair
// Same algorithm as used in arrayant_interpolate_multi and qd_arrayant_interpolate
// ------------------------------------------------------------------------------------------------
template <typename dtype>
static inline void slerp_complex_mf(dtype Ar, dtype Ai, dtype Br, dtype Bi, dtype w,
                                    dtype &Xr, dtype &Xi)
{
    constexpr dtype one = dtype(1.0), zero = dtype(0.0), neg_one = dtype(-1.0);
    const dtype R0 = std::numeric_limits<dtype>::epsilon() * std::numeric_limits<dtype>::epsilon() * std::numeric_limits<dtype>::epsilon();
    const dtype R1 = std::numeric_limits<dtype>::epsilon();
    constexpr dtype tL = dtype(-0.999), tS = dtype(-0.99), dT = one / (tS - tL);

    dtype wB = w, wA = one - w;
    dtype ampA = std::sqrt(Ar * Ar + Ai * Ai);
    dtype ampB = std::sqrt(Br * Br + Bi * Bi);

    if (ampA < R1 && ampB < R1)
    {
        Xr = zero;
        Xi = zero;
        return;
    }

    dtype gAr = (ampA < R1) ? zero : Ar / ampA;
    dtype gAi = (ampA < R1) ? zero : Ai / ampA;
    dtype gBr = (ampB < R1) ? zero : Br / ampB;
    dtype gBi = (ampB < R1) ? zero : Bi / ampB;
    dtype cPhase = (ampA < R1 || ampB < R1) ? neg_one : gAr * gBr + gAi * gBi;
    bool linear_int = cPhase < tS;

    dtype fXr = zero, fXi = zero;
    if (linear_int)
        fXr = wA * Ar + wB * Br, fXi = wA * Ai + wB * Bi;

    if (cPhase > tL)
    {
        dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0;
        dtype sPhase = one / std::sin(Phase);
        dtype wp = std::sin(wB * Phase) * sPhase;
        dtype wn = std::sin(wA * Phase) * sPhase;
        dtype gXr = wn * gAr + wp * gBr;
        dtype gXi = wn * gAi + wp * gBi;
        dtype ampX = wA * ampA + wB * ampB;

        if (linear_int) // Transition zone: blend spherical and linear
        {
            dtype m = (tS - cPhase) * dT, n = one - m;
            fXr = n * gXr * ampX + m * fXr;
            fXi = n * gXi * ampX + m * fXi;
        }
        else
            fXr = gXr * ampX, fXi = gXi * ampX;
    }
    Xr = fXr;
    Xi = fXi;
}

// ------------------------------------------------------------------------------------------------
// Find bracketing indices and weight for frequency interpolation
// Returns: i_lo, i_hi, weight (0=lo, 1=hi), exact_match flag
// ------------------------------------------------------------------------------------------------
template <typename dtype>
static inline void find_freq_bracket(const arma::Col<dtype> &center_freqs, dtype freq,
                                     arma::uword &i_lo, arma::uword &i_hi, dtype &w, bool &exact)
{
    arma::uword n = center_freqs.n_elem;
    exact = false;
    i_lo = 0;
    i_hi = 0;
    w = dtype(0.0);

    if (n == 1)
    {
        exact = true;
        return;
    }

    if (freq <= center_freqs[0])
    {
        exact = true;
        return;
    }

    if (freq >= center_freqs[n - 1])
    {
        exact = true;
        i_lo = n - 1;
        return;
    }

    // Find bracket: i_lo is the last entry <= freq
    for (arma::uword i = 1; i < n; ++i)
    {
        if (center_freqs[i] <= freq)
            i_lo = i;
        else
            break;
    }
    i_hi = i_lo + 1;

    // Check for exact match (relative tolerance 1e-6)
    if (std::abs((double)center_freqs[i_lo] - (double)freq) < 1.0e-6 * std::abs((double)center_freqs[i_lo]))
    {
        exact = true;
        return;
    }

    w = (freq - center_freqs[i_lo]) / (center_freqs[i_hi] - center_freqs[i_lo]);
}

// ------------------------------------------------------------------------------------------------
// Interpolate a complex-valued matrix pair using SLERP per element
// Handles empty imaginary matrices (treated as zeros)
// ------------------------------------------------------------------------------------------------
template <typename dtype>
static inline void interpolate_complex_matrix(const arma::Mat<dtype> &re_lo, const arma::Mat<dtype> &im_lo,
                                              const arma::Mat<dtype> &re_hi, const arma::Mat<dtype> &im_hi,
                                              dtype w, arma::Mat<dtype> &re_out, arma::Mat<dtype> &im_out)
{
    arma::uword nR = re_lo.n_rows, nC = re_lo.n_cols, nE = nR * nC;
    re_out.set_size(nR, nC);
    im_out.set_size(nR, nC);

    const dtype *rlo = re_lo.memptr();
    const dtype *rhi = re_hi.memptr();
    bool has_im_lo = (im_lo.n_elem == nE);
    bool has_im_hi = (im_hi.n_elem == nE);
    const dtype *ilo = has_im_lo ? im_lo.memptr() : nullptr;
    const dtype *ihi = has_im_hi ? im_hi.memptr() : nullptr;
    dtype *ro = re_out.memptr();
    dtype *io = im_out.memptr();
    constexpr dtype zero = dtype(0.0);

    for (arma::uword k = 0; k < nE; ++k)
    {
        dtype ai = ilo ? ilo[k] : zero;
        dtype bi = ihi ? ihi[k] : zero;
        slerp_complex_mf(rlo[k], ai, rhi[k], bi, w, ro[k], io[k]);
    }
}

// ================================================================================================
// get_channels_multifreq implementation
// ================================================================================================

template <typename dtype>
void quadriga_lib::get_channels_multifreq(const std::vector<quadriga_lib::arrayant<dtype>> &tx_array,
                                          const std::vector<quadriga_lib::arrayant<dtype>> &rx_array,
                                          dtype Tx, dtype Ty, dtype Tz, dtype Tb, dtype Tt, dtype Th,
                                          dtype Rx, dtype Ry, dtype Rz, dtype Rb, dtype Rt, dtype Rh,
                                          const arma::Mat<dtype> &fbs_pos, const arma::Mat<dtype> &lbs_pos,
                                          const arma::Mat<dtype> &path_gain, const arma::Col<dtype> &path_length,
                                          const arma::Cube<dtype> &M,
                                          const arma::Col<dtype> &freq_in, const arma::Col<dtype> &freq_out,
                                          std::vector<arma::Cube<dtype>> &coeff_re, std::vector<arma::Cube<dtype>> &coeff_im,
                                          std::vector<arma::Cube<dtype>> &delay,
                                          bool use_absolute_delays, bool add_fake_los_path, dtype propagation_speed)
{
    constexpr dtype los_limit = dtype(1.0e-4);
    constexpr dtype zero = dtype(0.0);
    constexpr dtype one = dtype(1.0);
    const dtype rC = one / propagation_speed;
    const dtype two_pi = dtype(2.0) * dtype(arma::datum::pi);

    // ============================================================================================
    // Input validation
    // ============================================================================================

    if (tx_array.empty())
        throw std::invalid_argument("get_channels_multifreq: TX array vector is empty.");
    if (rx_array.empty())
        throw std::invalid_argument("get_channels_multifreq: RX array vector is empty.");

    {
        std::string err = quadriga_lib::arrayant_is_valid_multi(tx_array, false);
        if (!err.empty())
            throw std::invalid_argument("get_channels_multifreq: TX array validation failed: " + err);
        err = quadriga_lib::arrayant_is_valid_multi(rx_array, false);
        if (!err.empty())
            throw std::invalid_argument("get_channels_multifreq: RX array validation failed: " + err);
    }

    if (fbs_pos.n_elem == 0 || lbs_pos.n_elem == 0 || path_gain.n_elem == 0 ||
        path_length.n_elem == 0 || M.n_elem == 0)
        throw std::invalid_argument("get_channels_multifreq: Missing path data.");

    if (fbs_pos.n_rows != 3 || lbs_pos.n_rows != 3)
        throw std::invalid_argument("get_channels_multifreq: 'fbs_pos' and 'lbs_pos' must have 3 rows.");

    if (M.n_rows != 8 && M.n_rows != 2)
        throw std::invalid_argument("get_channels_multifreq: 'M' must have 8 rows (full polarimetric) or 2 rows (scalar pressure).");

    const bool scalar_M = (M.n_rows == 2);

    const arma::uword n_path = fbs_pos.n_cols;
    const arma::uword n_freq_in = freq_in.n_elem;
    const arma::uword n_freq_out = freq_out.n_elem;
    const arma::uword n_out = add_fake_los_path ? n_path + 1 : n_path;

    if (n_freq_in == 0)
        throw std::invalid_argument("get_channels_multifreq: 'freq_in' is empty.");
    if (n_freq_out == 0)
        throw std::invalid_argument("get_channels_multifreq: 'freq_out' is empty.");

    if (lbs_pos.n_cols != n_path)
        throw std::invalid_argument("get_channels_multifreq: 'lbs_pos' must have n_path columns.");
    if (path_gain.n_rows != n_path || path_gain.n_cols != n_freq_in)
        throw std::invalid_argument("get_channels_multifreq: 'path_gain' must have size [n_path, n_freq_in].");
    if (path_length.n_elem != n_path)
        throw std::invalid_argument("get_channels_multifreq: 'path_length' must have n_path elements.");
    if (M.n_cols != n_path || M.n_slices != n_freq_in)
        throw std::invalid_argument("get_channels_multifreq: 'M' must have size [8|2, n_path, n_freq_in].");
    if (propagation_speed <= zero)
        throw std::invalid_argument("get_channels_multifreq: 'propagation_speed' must be positive.");

    // Antenna dimensions (from first entry, consistent across all entries by validation)
    const arma::uword n_tx = tx_array[0].n_elements();
    const arma::uword n_rx = rx_array[0].n_elements();
    const arma::uword n_links = n_rx * n_tx;
    const arma::uword n_tx_ports = tx_array[0].n_ports();
    const arma::uword n_rx_ports = rx_array[0].n_ports();
    const arma::uword n_ports = n_tx_ports * n_rx_ports;

    // ============================================================================================
    // Resize outputs (only if needed, preserves externally borrowed memory)
    // ============================================================================================

    if (coeff_re.size() != n_freq_out)
        coeff_re.resize(n_freq_out);
    if (coeff_im.size() != n_freq_out)
        coeff_im.resize(n_freq_out);
    if (delay.size() != n_freq_out)
        delay.resize(n_freq_out);
    for (arma::uword f = 0; f < n_freq_out; ++f)
    {
        if (coeff_re[f].n_rows != n_rx_ports || coeff_re[f].n_cols != n_tx_ports || coeff_re[f].n_slices != n_out)
            coeff_re[f].set_size(n_rx_ports, n_tx_ports, n_out);
        if (coeff_im[f].n_rows != n_rx_ports || coeff_im[f].n_cols != n_tx_ports || coeff_im[f].n_slices != n_out)
            coeff_im[f].set_size(n_rx_ports, n_tx_ports, n_out);
        if (delay[f].n_rows != n_rx_ports || delay[f].n_cols != n_tx_ports || delay[f].n_slices != n_out)
            delay[f].set_size(n_rx_ports, n_tx_ports, n_out);
    }

    // ============================================================================================
    // Build center-frequency vectors for TX, RX, and freq_in grids
    // ============================================================================================

    arma::Col<dtype> tx_center_freqs((arma::uword)tx_array.size());
    arma::Col<dtype> rx_center_freqs((arma::uword)rx_array.size());
    for (arma::uword i = 0; i < tx_center_freqs.n_elem; ++i)
        tx_center_freqs[i] = (dtype)tx_array[i].center_frequency;
    for (arma::uword i = 0; i < rx_center_freqs.n_elem; ++i)
        rx_center_freqs[i] = (dtype)rx_array[i].center_frequency;

    // ============================================================================================
    // Phase 1: Geometry computation (frequency-independent, sequential)
    // ============================================================================================

    // Transmitter and receiver orientation
    arma::Cube<dtype> tx_orientation(3, 1, 1, arma::fill::zeros);
    arma::Cube<dtype> rx_orientation(3, 1, 1, arma::fill::zeros);
    bool tx_orientation_not_zero = (Tb != zero || Tt != zero || Th != zero);
    bool rx_orientation_not_zero = (Rb != zero || Rt != zero || Rh != zero);

    if (tx_orientation_not_zero)
    {
        dtype *ptr = tx_orientation.memptr();
        ptr[0] = Tb;
        ptr[1] = Tt;
        ptr[2] = Th;
    }
    if (rx_orientation_not_zero)
    {
        dtype *ptr = rx_orientation.memptr();
        ptr[0] = Rb;
        ptr[1] = Rt;
        ptr[2] = Rh;
    }

    // Calculate antenna element positions in GCS
    arma::Mat<dtype> tx_element_pos(3, n_tx, arma::fill::zeros);
    arma::Mat<dtype> rx_element_pos(3, n_rx, arma::fill::zeros);
    dtype *p_tx = tx_element_pos.memptr();
    dtype *p_rx = rx_element_pos.memptr();

    if (!tx_array[0].element_pos.is_empty())
        std::memcpy(p_tx, tx_array[0].element_pos.memptr(), 3 * n_tx * sizeof(dtype));
    if (tx_orientation_not_zero)
        qd_rotate_inplace(Tb, -Tt, Th, p_tx, n_tx);
    for (arma::uword t = 0; t < 3 * n_tx; t += 3)
        p_tx[t] += Tx, p_tx[t + 1] += Ty, p_tx[t + 2] += Tz;

    if (!rx_array[0].element_pos.is_empty())
        std::memcpy(p_rx, rx_array[0].element_pos.memptr(), 3 * n_rx * sizeof(dtype));
    if (rx_orientation_not_zero)
        qd_rotate_inplace(Rb, -Rt, Rh, p_rx, n_rx);
    for (arma::uword r = 0; r < 3 * n_rx; r += 3)
        p_rx[r] += Rx, p_rx[r + 1] += Ry, p_rx[r + 2] += Rz;

    // Free-space distance
    dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
    dtype dist_rx_tx = std::sqrt(x * x + y * y + z * z);

    // Allocate geometry arrays (reused across all frequencies)
    arma::Mat<dtype> AOD(n_links, n_out), EOD(n_links, n_out);
    arma::Mat<dtype> AOA(n_links, n_out), EOA(n_links, n_out);
    arma::Mat<dtype> path_delays_geom(n_links, n_out); // Element-resolved path lengths [m]

    dtype *p_aod = AOD.memptr(), *p_eod = EOD.memptr();
    dtype *p_aoa = AOA.memptr(), *p_eoa = EOA.memptr();
    dtype *p_path_dl = path_delays_geom.memptr();

    const dtype *p_fbs = fbs_pos.memptr();
    const dtype *p_lbs = lbs_pos.memptr();
    const dtype *p_length = path_length.memptr();

    // LOS path detection
    arma::uword true_los_path = 0;
    dtype shortest_path = los_limit;

    for (arma::uword i_out = 0; i_out < n_out; ++i_out)
    {
        const arma::uword i = add_fake_los_path ? i_out - 1 : i_out;
        const arma::uword ix = 3 * i, iy = ix + 1, iz = ix + 2;
        const arma::uword o_slice = i_out * n_links;

        dtype d_shortest = dist_rx_tx, d_length = dist_rx_tx, d_fbs_lbs = zero;
        if (!add_fake_los_path || i_out != 0)
        {
            x = p_fbs[ix] - Tx;
            y = p_fbs[iy] - Ty;
            z = p_fbs[iz] - Tz;
            d_shortest = std::sqrt(x * x + y * y + z * z);
            x = p_lbs[ix] - p_fbs[ix];
            y = p_lbs[iy] - p_fbs[iy];
            z = p_lbs[iz] - p_fbs[iz];
            d_fbs_lbs = std::sqrt(x * x + y * y + z * z);
            d_shortest += d_fbs_lbs;
            x = Rx - p_lbs[ix];
            y = Ry - p_lbs[iy];
            z = Rz - p_lbs[iz];
            d_shortest += std::sqrt(x * x + y * y + z * z);
            d_length = (p_length[i] < d_shortest) ? d_shortest : p_length[i];
        }

        if (std::abs(d_length - dist_rx_tx) < shortest_path) // LOS path
        {
            if (!add_fake_los_path || i_out != 0)
                true_los_path = i_out, shortest_path = std::abs(d_length - dist_rx_tx);

            for (arma::uword t = 0; t < n_tx; ++t)
            {
                dtype *pt = &p_tx[3 * t];
                for (arma::uword r = 0; r < n_rx; ++r)
                {
                    dtype *pr = &p_rx[3 * r];
                    x = pr[0] - pt[0];
                    y = pr[1] - pt[1];
                    z = pr[2] - pt[2];
                    dtype d = std::sqrt(x * x + y * y + z * z);
                    arma::uword io = o_slice + t * n_rx + r;
                    p_aod[io] = std::atan2(y, x);
                    p_eod[io] = (d < los_limit) ? zero : std::asin(z / d);
                    p_aoa[io] = std::atan2(-y, -x);
                    p_eoa[io] = -p_eod[io];
                    p_path_dl[io] = d;
                }
            }
        }
        else // NLOS path
        {
            dtype *dr = new dtype[n_rx];
            for (arma::uword r = 0; r < n_rx; ++r)
            {
                dtype *pr = &p_rx[3 * r];
                x = p_lbs[ix] - pr[0];
                y = p_lbs[iy] - pr[1];
                z = p_lbs[iz] - pr[2];
                dr[r] = std::sqrt(x * x + y * y + z * z);
                p_aoa[o_slice + r] = std::atan2(y, x);
                p_eoa[o_slice + r] = (dr[r] < los_limit) ? zero : std::asin(z / dr[r]);
            }
            for (arma::uword t = 0; t < n_tx; ++t)
            {
                dtype *pt = &p_tx[3 * t];
                x = p_fbs[ix] - pt[0];
                y = p_fbs[iy] - pt[1];
                z = p_fbs[iz] - pt[2];
                dtype dt = std::sqrt(x * x + y * y + z * z);
                dtype at = std::atan2(y, x);
                dtype et = (dt < los_limit) ? zero : std::asin(z / dt);
                for (arma::uword r = 0; r < n_rx; ++r)
                {
                    arma::uword io = o_slice + t * n_rx + r;
                    p_aod[io] = at;
                    p_eod[io] = et;
                    p_aoa[io] = p_aoa[o_slice + r];
                    p_eoa[io] = p_eoa[o_slice + r];
                    p_path_dl[io] = dt + d_fbs_lbs + dr[r];
                }
            }
            delete[] dr;
        }
    }

    // ============================================================================================
    // Prepare TX/RX element indices and interpolation positions (frequency-independent)
    // ============================================================================================

    arma::Col<unsigned> i_tx_element(n_links);
    arma::Mat<dtype> tx_element_pos_interp(3, n_links);
    {
        unsigned *pe = i_tx_element.memptr();
        dtype *pp = tx_element_pos_interp.memptr();
        arma::Mat<dtype> tx_local(3, n_tx, arma::fill::zeros);
        if (!tx_array[0].element_pos.is_empty())
            std::memcpy(tx_local.memptr(), tx_array[0].element_pos.memptr(), 3 * n_tx * sizeof(dtype));
        dtype *ptl = tx_local.memptr();
        for (unsigned t = 0; t < (unsigned)n_tx; ++t)
            for (unsigned r = 0; r < (unsigned)n_rx; ++r)
                *pe++ = t + 1, *pp++ = ptl[3 * t], *pp++ = ptl[3 * t + 1], *pp++ = ptl[3 * t + 2];
    }

    arma::Col<unsigned> i_rx_element(n_links);
    arma::Mat<dtype> rx_element_pos_interp(3, n_links);
    {
        unsigned *pe = i_rx_element.memptr();
        dtype *pp = rx_element_pos_interp.memptr();
        arma::Mat<dtype> rx_local(3, n_rx, arma::fill::zeros);
        if (!rx_array[0].element_pos.is_empty())
            std::memcpy(rx_local.memptr(), rx_array[0].element_pos.memptr(), 3 * n_rx * sizeof(dtype));
        dtype *prl = rx_local.memptr();
        for (unsigned t = 0; t < (unsigned)n_tx; ++t)
            for (unsigned r = 0; r < (unsigned)n_rx; ++r)
                *pe++ = r + 1, *pp++ = prl[3 * r], *pp++ = prl[3 * r + 1], *pp++ = prl[3 * r + 2];
    }

    // ============================================================================================
    // Determine if element coupling is needed (check ALL entries across frequencies)
    // ============================================================================================

    const bool different_output_size = (n_tx != n_tx_ports || n_rx != n_rx_ports);
    bool apply_element_coupling = different_output_size;

    if (!apply_element_coupling)
    {
        // Check all TX entries for non-identity coupling
        for (arma::uword e = 0; e < tx_array.size() && !apply_element_coupling; ++e)
        {
            const auto &ta = tx_array[e];
            if (ta.coupling_re.n_elem != 0 && ta.coupling_re.n_rows == n_tx && ta.coupling_re.n_cols == n_tx)
            {
                const dtype *p = ta.coupling_re.memptr();
                for (arma::uword r = 0; r < n_tx && !apply_element_coupling; ++r)
                    for (arma::uword c = 0; c < n_tx && !apply_element_coupling; ++c)
                    {
                        if (r == c && std::abs(p[c * n_tx + r] - one) > dtype(1e-6))
                            apply_element_coupling = true;
                        else if (r != c && std::abs(p[c * n_tx + r]) > dtype(1e-6))
                            apply_element_coupling = true;
                    }
            }
            if (!apply_element_coupling && ta.coupling_im.n_elem != 0)
                for (const dtype *p = ta.coupling_im.begin(); p < ta.coupling_im.end(); ++p)
                    if (std::abs(*p) > dtype(1e-6))
                    {
                        apply_element_coupling = true;
                        break;
                    }
        }

        // Check all RX entries for non-identity coupling
        for (arma::uword e = 0; e < rx_array.size() && !apply_element_coupling; ++e)
        {
            const auto &ra = rx_array[e];
            if (ra.coupling_re.n_elem != 0 && ra.coupling_re.n_rows == n_rx && ra.coupling_re.n_cols == n_rx)
            {
                const dtype *p = ra.coupling_re.memptr();
                for (arma::uword r = 0; r < n_rx && !apply_element_coupling; ++r)
                    for (arma::uword c = 0; c < n_rx && !apply_element_coupling; ++c)
                    {
                        if (r == c && std::abs(p[c * n_rx + r] - one) > dtype(1e-6))
                            apply_element_coupling = true;
                        else if (r != c && std::abs(p[c * n_rx + r]) > dtype(1e-6))
                            apply_element_coupling = true;
                    }
            }
            if (!apply_element_coupling && ra.coupling_im.n_elem != 0)
                for (const dtype *p = ra.coupling_im.begin(); p < ra.coupling_im.end(); ++p)
                    if (std::abs(*p) > dtype(1e-6))
                    {
                        apply_element_coupling = true;
                        break;
                    }
        }
    }

    // ============================================================================================
    // Phase 2: Per-frequency processing (parallelized over output frequencies)
    // ============================================================================================

    const int n_freq_out_i = (int)n_freq_out;
#pragma omp parallel for schedule(static)
    for (int f_i = 0; f_i < n_freq_out_i; ++f_i)
    {
        const arma::uword f = (arma::uword)f_i;
        const dtype freq = freq_out[f];
        const dtype wavelength = (freq > zero) ? propagation_speed / freq : one;
        const dtype wave_number = (freq > zero) ? two_pi * freq / propagation_speed : zero;

        // ------------------------------------------------------------------------------------
        // Find frequency brackets for TX, RX, and freq_in grids
        // ------------------------------------------------------------------------------------

        arma::uword tx_lo, tx_hi, rx_lo, rx_hi, fin_lo, fin_hi;
        dtype tx_w, rx_w, fin_w;
        bool tx_exact, rx_exact, fin_exact;

        find_freq_bracket(tx_center_freqs, freq, tx_lo, tx_hi, tx_w, tx_exact);
        find_freq_bracket(rx_center_freqs, freq, rx_lo, rx_hi, rx_w, rx_exact);
        find_freq_bracket(freq_in, freq, fin_lo, fin_hi, fin_w, fin_exact);

        // ------------------------------------------------------------------------------------
        // Interpolate path gain at this frequency (linear interpolation)
        // ------------------------------------------------------------------------------------

        const dtype *p_gain_in = path_gain.memptr(); // Column-major [n_path, n_freq_in]
        arma::Col<dtype> gain_f(n_path);
        if (fin_exact)
        {
            std::memcpy(gain_f.memptr(), &p_gain_in[fin_lo * n_path], n_path * sizeof(dtype));
        }
        else
        {
            const dtype *col_lo = &p_gain_in[fin_lo * n_path];
            const dtype *col_hi = &p_gain_in[fin_hi * n_path];
            dtype wA = one - fin_w, wB = fin_w;
            dtype *pg = gain_f.memptr();
            for (arma::uword p = 0; p < n_path; ++p)
                pg[p] = wA * col_lo[p] + wB * col_hi[p];
        }

        // ------------------------------------------------------------------------------------
        // Interpolate M (Jones matrix) at this frequency (SLERP per complex entry)
        // ------------------------------------------------------------------------------------

        const arma::uword M_rows = M.n_rows; // 8 or 2
        arma::Mat<dtype> M_f(M_rows, n_path);
        if (fin_exact)
        {
            std::memcpy(M_f.memptr(), M.slice_memptr(fin_lo), M_rows * n_path * sizeof(dtype));
        }
        else
        {
            const dtype *M_lo = M.slice_memptr(fin_lo);
            const dtype *M_hi = M.slice_memptr(fin_hi);
            dtype *Mf = M_f.memptr();
            arma::uword n_pairs = M_rows / 2; // 4 complex pairs for full, 1 for scalar

            for (arma::uword p = 0; p < n_path; ++p)
            {
                arma::uword base = p * M_rows;
                for (arma::uword k = 0; k < n_pairs; ++k)
                {
                    arma::uword idx = base + 2 * k;
                    slerp_complex_mf(M_lo[idx], M_lo[idx + 1],
                                     M_hi[idx], M_hi[idx + 1],
                                     fin_w, Mf[idx], Mf[idx + 1]);
                }
            }
        }

        // ------------------------------------------------------------------------------------
        // Interpolate coupling matrices at this frequency (SLERP per complex entry)
        // ------------------------------------------------------------------------------------

        arma::Mat<dtype> tx_cpl_re_f, tx_cpl_im_f;
        arma::Mat<dtype> rx_cpl_re_f, rx_cpl_im_f;
        arma::Mat<dtype> tx_coupling_pwr_f, rx_coupling_pwr_f;

        if (apply_element_coupling)
        {
            // --- TX coupling interpolation ---
            const arma::Mat<dtype> &tx_cpl_re_lo = tx_array[tx_lo].coupling_re;
            const arma::Mat<dtype> &tx_cpl_im_lo = tx_array[tx_lo].coupling_im;

            if (tx_exact)
            {
                // Use the bracket entry directly
                if (tx_cpl_re_lo.n_elem != 0)
                    tx_cpl_re_f = tx_cpl_re_lo;
                else
                {
                    tx_cpl_re_f.eye(n_tx, n_tx_ports);
                }
                if (tx_cpl_im_lo.n_elem != 0)
                    tx_cpl_im_f = tx_cpl_im_lo;
                else
                    tx_cpl_im_f.zeros(n_tx, n_tx_ports);
            }
            else
            {
                // Build effective matrices for lo and hi (handle empty → identity/zeros)
                arma::Mat<dtype> re_lo_eff, im_lo_eff, re_hi_eff, im_hi_eff;
                const arma::Mat<dtype> &tx_cpl_re_hi = tx_array[tx_hi].coupling_re;
                const arma::Mat<dtype> &tx_cpl_im_hi = tx_array[tx_hi].coupling_im;

                if (tx_cpl_re_lo.n_elem != 0)
                    re_lo_eff = tx_cpl_re_lo;
                else
                    re_lo_eff.eye(n_tx, n_tx_ports);

                if (tx_cpl_re_hi.n_elem != 0)
                    re_hi_eff = tx_cpl_re_hi;
                else
                    re_hi_eff.eye(n_tx, n_tx_ports);

                // im can be empty → treated as zeros inside interpolate_complex_matrix
                interpolate_complex_matrix(re_lo_eff, tx_cpl_im_lo,
                                           re_hi_eff, tx_cpl_im_hi,
                                           tx_w, tx_cpl_re_f, tx_cpl_im_f);
            }

            // --- RX coupling interpolation ---
            const arma::Mat<dtype> &rx_cpl_re_lo = rx_array[rx_lo].coupling_re;
            const arma::Mat<dtype> &rx_cpl_im_lo = rx_array[rx_lo].coupling_im;

            if (rx_exact)
            {
                if (rx_cpl_re_lo.n_elem != 0)
                    rx_cpl_re_f = rx_cpl_re_lo;
                else
                    rx_cpl_re_f.eye(n_rx, n_rx_ports);

                if (rx_cpl_im_lo.n_elem != 0)
                    rx_cpl_im_f = rx_cpl_im_lo;
                else
                    rx_cpl_im_f.zeros(n_rx, n_rx_ports);
            }
            else
            {
                arma::Mat<dtype> re_lo_eff, im_lo_eff, re_hi_eff, im_hi_eff;
                const arma::Mat<dtype> &rx_cpl_re_hi = rx_array[rx_hi].coupling_re;
                const arma::Mat<dtype> &rx_cpl_im_hi = rx_array[rx_hi].coupling_im;

                if (rx_cpl_re_lo.n_elem != 0)
                    re_lo_eff = rx_cpl_re_lo;
                else
                    re_lo_eff.eye(n_rx, n_rx_ports);

                if (rx_cpl_re_hi.n_elem != 0)
                    re_hi_eff = rx_cpl_re_hi;
                else
                    re_hi_eff.eye(n_rx, n_rx_ports);

                interpolate_complex_matrix(re_lo_eff, rx_cpl_im_lo,
                                           re_hi_eff, rx_cpl_im_hi,
                                           rx_w, rx_cpl_re_f, rx_cpl_im_f);
            }

            // Compute power matrices for delay coupling
            tx_coupling_pwr_f.set_size(n_tx, n_tx_ports);
            rx_coupling_pwr_f.set_size(n_rx, n_rx_ports);
            qd_power_mat(n_tx, n_tx_ports, tx_coupling_pwr_f.memptr(), true,
                         tx_cpl_re_f.memptr(), tx_cpl_im_f.memptr());
            qd_power_mat(n_rx, n_rx_ports, rx_coupling_pwr_f.memptr(), true,
                         rx_cpl_re_f.memptr(), rx_cpl_im_f.memptr());
        }

        // ------------------------------------------------------------------------------------
        // Internal storage for this frequency (if output size differs from element size)
        // ------------------------------------------------------------------------------------

        arma::Cube<dtype> CR, CI, DL;
        if (different_output_size)
        {
            CR.set_size(n_rx, n_tx, n_out);
            CI.set_size(n_rx, n_tx, n_out);
            DL.set_size(n_rx, n_tx, n_out);
        }

        dtype *p_coeff_re = different_output_size ? CR.memptr() : coeff_re[f].memptr();
        dtype *p_coeff_im = different_output_size ? CI.memptr() : coeff_im[f].memptr();
        dtype *p_delays_f = different_output_size ? DL.memptr() : delay[f].memptr();

        // Copy geometric path delays (same for all frequencies, in meters)
        std::memcpy(p_delays_f, p_path_dl, n_links * n_out * sizeof(dtype));

        // ------------------------------------------------------------------------------------
        // MIMO coefficient calculation — sequential loop over paths
        // ------------------------------------------------------------------------------------

        for (arma::uword j = 0; j < n_out; ++j)
        {
            const arma::uword i = add_fake_los_path ? (j == 0 ? 0 : j - 1) : j;
            const arma::uword o_slice = j * n_links;

            // --- Interpolate TX antenna at departure angles ---
            const arma::Mat<dtype> AOD_j(&p_aod[o_slice], n_links, 1, false, true);
            const arma::Mat<dtype> EOD_j(&p_eod[o_slice], n_links, 1, false, true);
            arma::Mat<dtype> EMPTY;

            arma::Mat<dtype> Vt_re(n_links, 1), Vt_im(n_links, 1);
            arma::Mat<dtype> Ht_re(n_links, 1), Ht_im(n_links, 1);

            if (tx_exact)
            {
                const auto &ta = tx_array[tx_lo];
                qd_arrayant_interpolate(&ta.e_theta_re, &ta.e_theta_im, &ta.e_phi_re, &ta.e_phi_im,
                                        &ta.azimuth_grid, &ta.elevation_grid, &AOD_j, &EOD_j,
                                        &i_tx_element, &tx_orientation, &tx_element_pos_interp,
                                        &Vt_re, &Vt_im, &Ht_re, &Ht_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);
            }
            else
            {
                arma::Mat<dtype> Vt_re_lo(n_links, 1), Vt_im_lo(n_links, 1);
                arma::Mat<dtype> Ht_re_lo(n_links, 1), Ht_im_lo(n_links, 1);
                arma::Mat<dtype> Vt_re_hi(n_links, 1), Vt_im_hi(n_links, 1);
                arma::Mat<dtype> Ht_re_hi(n_links, 1), Ht_im_hi(n_links, 1);

                const auto &ta_lo = tx_array[tx_lo];
                qd_arrayant_interpolate(&ta_lo.e_theta_re, &ta_lo.e_theta_im, &ta_lo.e_phi_re, &ta_lo.e_phi_im,
                                        &ta_lo.azimuth_grid, &ta_lo.elevation_grid, &AOD_j, &EOD_j,
                                        &i_tx_element, &tx_orientation, &tx_element_pos_interp,
                                        &Vt_re_lo, &Vt_im_lo, &Ht_re_lo, &Ht_im_lo, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

                const auto &ta_hi = tx_array[tx_hi];
                qd_arrayant_interpolate(&ta_hi.e_theta_re, &ta_hi.e_theta_im, &ta_hi.e_phi_re, &ta_hi.e_phi_im,
                                        &ta_hi.azimuth_grid, &ta_hi.elevation_grid, &AOD_j, &EOD_j,
                                        &i_tx_element, &tx_orientation, &tx_element_pos_interp,
                                        &Vt_re_hi, &Vt_im_hi, &Ht_re_hi, &Ht_im_hi, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

                // SLERP blend between bracket entries
                dtype *vtr = Vt_re.memptr(), *vti = Vt_im.memptr();
                dtype *htr = Ht_re.memptr(), *hti = Ht_im.memptr();
                const dtype *vlr = Vt_re_lo.memptr(), *vli = Vt_im_lo.memptr();
                const dtype *vhr = Vt_re_hi.memptr(), *vhi = Vt_im_hi.memptr();
                const dtype *hlr = Ht_re_lo.memptr(), *hli = Ht_im_lo.memptr();
                const dtype *hhr = Ht_re_hi.memptr(), *hhi = Ht_im_hi.memptr();
                for (arma::uword k = 0; k < n_links; ++k)
                {
                    slerp_complex_mf(vlr[k], vli[k], vhr[k], vhi[k], tx_w, vtr[k], vti[k]);
                    slerp_complex_mf(hlr[k], hli[k], hhr[k], hhi[k], tx_w, htr[k], hti[k]);
                }
            }

            // --- Interpolate RX antenna at arrival angles ---
            const arma::Mat<dtype> AOA_j(&p_aoa[o_slice], n_links, 1, false, true);
            const arma::Mat<dtype> EOA_j(&p_eoa[o_slice], n_links, 1, false, true);

            arma::Mat<dtype> Vr_re(n_links, 1), Vr_im(n_links, 1);
            arma::Mat<dtype> Hr_re(n_links, 1), Hr_im(n_links, 1);

            if (rx_exact)
            {
                const auto &ra = rx_array[rx_lo];
                qd_arrayant_interpolate(&ra.e_theta_re, &ra.e_theta_im, &ra.e_phi_re, &ra.e_phi_im,
                                        &ra.azimuth_grid, &ra.elevation_grid, &AOA_j, &EOA_j,
                                        &i_rx_element, &rx_orientation, &rx_element_pos_interp,
                                        &Vr_re, &Vr_im, &Hr_re, &Hr_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);
            }
            else
            {
                arma::Mat<dtype> Vr_re_lo(n_links, 1), Vr_im_lo(n_links, 1);
                arma::Mat<dtype> Hr_re_lo(n_links, 1), Hr_im_lo(n_links, 1);
                arma::Mat<dtype> Vr_re_hi(n_links, 1), Vr_im_hi(n_links, 1);
                arma::Mat<dtype> Hr_re_hi(n_links, 1), Hr_im_hi(n_links, 1);

                const auto &ra_lo = rx_array[rx_lo];
                qd_arrayant_interpolate(&ra_lo.e_theta_re, &ra_lo.e_theta_im, &ra_lo.e_phi_re, &ra_lo.e_phi_im,
                                        &ra_lo.azimuth_grid, &ra_lo.elevation_grid, &AOA_j, &EOA_j,
                                        &i_rx_element, &rx_orientation, &rx_element_pos_interp,
                                        &Vr_re_lo, &Vr_im_lo, &Hr_re_lo, &Hr_im_lo, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

                const auto &ra_hi = rx_array[rx_hi];
                qd_arrayant_interpolate(&ra_hi.e_theta_re, &ra_hi.e_theta_im, &ra_hi.e_phi_re, &ra_hi.e_phi_im,
                                        &ra_hi.azimuth_grid, &ra_hi.elevation_grid, &AOA_j, &EOA_j,
                                        &i_rx_element, &rx_orientation, &rx_element_pos_interp,
                                        &Vr_re_hi, &Vr_im_hi, &Hr_re_hi, &Hr_im_hi, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

                dtype *vrr = Vr_re.memptr(), *vri = Vr_im.memptr();
                dtype *hrr = Hr_re.memptr(), *hri = Hr_im.memptr();
                const dtype *vlr = Vr_re_lo.memptr(), *vli = Vr_im_lo.memptr();
                const dtype *vhr = Vr_re_hi.memptr(), *vhi = Vr_im_hi.memptr();
                const dtype *hlr = Hr_re_lo.memptr(), *hli = Hr_im_lo.memptr();
                const dtype *hhr = Hr_re_hi.memptr(), *hhi = Hr_im_hi.memptr();
                for (arma::uword k = 0; k < n_links; ++k)
                {
                    slerp_complex_mf(vlr[k], vli[k], vhr[k], vhi[k], rx_w, vrr[k], vri[k]);
                    slerp_complex_mf(hlr[k], hli[k], hhr[k], hhi[k], rx_w, hrr[k], hri[k]);
                }
            }

            // --- Compute MIMO coefficients for this path ---
            const dtype *pM = M_f.colptr(i);
            const dtype *pVrr = Vr_re.memptr(), *pVri = Vr_im.memptr();
            const dtype *pHrr = Hr_re.memptr(), *pHri = Hr_im.memptr();
            const dtype *pVtr = Vt_re.memptr(), *pVti = Vt_im.memptr();
            const dtype *pHtr = Ht_re.memptr(), *pHti = Ht_im.memptr();
            const dtype path_amplitude = (add_fake_los_path && j == 0) ? zero : std::sqrt(gain_f[i]);

            for (arma::uword t = 0; t < n_tx; ++t)
                for (arma::uword r = 0; r < n_rx; ++r)
                {
                    arma::uword R = t * n_rx + r;
                    dtype re = zero, im = zero;

                    if (scalar_M)
                    {
                        // Scalar pressure: only VV component, M = [ReVV, ImVV]
                        re += pVrr[R] * pM[0] * pVtr[R] - pVri[R] * pM[1] * pVtr[R] - pVrr[R] * pM[1] * pVti[R] - pVri[R] * pM[0] * pVti[R];
                        im += pVrr[R] * pM[1] * pVtr[R] + pVri[R] * pM[0] * pVtr[R] + pVrr[R] * pM[0] * pVti[R] - pVri[R] * pM[1] * pVti[R];
                    }
                    else
                    {
                        // Full 2x2 Jones: [ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH]
                        re += pVrr[R] * pM[0] * pVtr[R] - pVri[R] * pM[1] * pVtr[R] - pVrr[R] * pM[1] * pVti[R] - pVri[R] * pM[0] * pVti[R];
                        re += pHrr[R] * pM[2] * pVtr[R] - pHri[R] * pM[3] * pVtr[R] - pHrr[R] * pM[3] * pVti[R] - pHri[R] * pM[2] * pVti[R];
                        re += pVrr[R] * pM[4] * pHtr[R] - pVri[R] * pM[5] * pHtr[R] - pVrr[R] * pM[5] * pHti[R] - pVri[R] * pM[4] * pHti[R];
                        re += pHrr[R] * pM[6] * pHtr[R] - pHri[R] * pM[7] * pHtr[R] - pHrr[R] * pM[7] * pHti[R] - pHri[R] * pM[6] * pHti[R];

                        im += pVrr[R] * pM[1] * pVtr[R] + pVri[R] * pM[0] * pVtr[R] + pVrr[R] * pM[0] * pVti[R] - pVri[R] * pM[1] * pVti[R];
                        im += pHrr[R] * pM[3] * pVtr[R] + pHri[R] * pM[2] * pVtr[R] + pHrr[R] * pM[2] * pVti[R] - pHri[R] * pM[3] * pVti[R];
                        im += pVrr[R] * pM[5] * pHtr[R] + pVri[R] * pM[4] * pHtr[R] + pVrr[R] * pM[4] * pHti[R] - pVri[R] * pM[5] * pHti[R];
                        im += pHrr[R] * pM[7] * pHtr[R] + pHri[R] * pM[6] * pHtr[R] + pHrr[R] * pM[6] * pHti[R] - pHri[R] * pM[7] * pHti[R];
                    }

                    // Apply phase rotation from path delay
                    dtype dl = p_delays_f[o_slice + R];
                    dtype phase = wave_number * std::fmod(dl, wavelength);
                    dtype cp = std::cos(phase), sp = std::sin(phase);

                    p_coeff_re[o_slice + R] = (re * cp + im * sp) * path_amplitude;
                    p_coeff_im[o_slice + R] = (-re * sp + im * cp) * path_amplitude;

                    // Convert path length [m] to propagation delay [s]
                    dl = use_absolute_delays ? dl : dl - dist_rx_tx;
                    p_delays_f[o_slice + R] = dl * rC;
                }

        } // End path loop

        // ------------------------------------------------------------------------------------
        // Apply antenna element coupling (using interpolated coupling matrices)
        // ------------------------------------------------------------------------------------

        if (apply_element_coupling)
        {
            arma::uword N = (n_ports > n_links) ? n_ports : n_links;
            dtype *tempX = new dtype[N];
            dtype *tempY = new dtype[N];
            dtype *tempT = new dtype[N];

            const dtype *tx_cpl_re_ptr = tx_cpl_re_f.memptr();
            const dtype *tx_cpl_im_ptr = tx_cpl_im_f.memptr();
            const dtype *rx_cpl_re_ptr = rx_cpl_re_f.memptr();
            const dtype *rx_cpl_im_ptr = rx_cpl_im_f.memptr();

            for (arma::uword j = 0; j < n_out; ++j)
            {
                arma::uword o_slice = j * n_links;

                if (different_output_size)
                {
                    // Coupling writes to output cubes; internal storage holds element-space data
                    qd_multiply_3_complex_mat(rx_cpl_re_ptr, rx_cpl_im_ptr,
                                              &p_coeff_re[o_slice], &p_coeff_im[o_slice],
                                              tx_cpl_re_ptr, tx_cpl_im_ptr,
                                              coeff_re[f].slice_memptr(j), coeff_im[f].slice_memptr(j),
                                              n_rx, n_rx_ports, n_tx, n_tx_ports);

                    qd_multiply_3_mat(rx_coupling_pwr_f.memptr(), &p_delays_f[o_slice],
                                      tx_coupling_pwr_f.memptr(), delay[f].slice_memptr(j),
                                      n_rx, n_rx_ports, n_tx, n_tx_ports);
                }
                else
                {
                    // Coupling applied in-place via temporaries
                    qd_multiply_3_complex_mat(rx_cpl_re_ptr, rx_cpl_im_ptr,
                                              &p_coeff_re[o_slice], &p_coeff_im[o_slice],
                                              tx_cpl_re_ptr, tx_cpl_im_ptr,
                                              tempX, tempY,
                                              n_rx, n_rx_ports, n_tx, n_tx_ports);
                    std::memcpy(&p_coeff_re[o_slice], tempX, n_ports * sizeof(dtype));
                    std::memcpy(&p_coeff_im[o_slice], tempY, n_ports * sizeof(dtype));

                    qd_multiply_3_mat(rx_coupling_pwr_f.memptr(), &p_delays_f[o_slice],
                                      tx_coupling_pwr_f.memptr(), tempT,
                                      n_rx, n_rx_ports, n_tx, n_tx_ports);
                    std::memcpy(&p_delays_f[o_slice], tempT, n_ports * sizeof(dtype));
                }
            }

            delete[] tempX;
            delete[] tempY;
            delete[] tempT;
        }

        // ------------------------------------------------------------------------------------
        // Move true LOS path to slot 0 if fake LOS was added
        // ------------------------------------------------------------------------------------

        if (add_fake_los_path && true_los_path != 0)
        {
            dtype *ptrR = different_output_size ? CR.slice_memptr(true_los_path) : coeff_re[f].slice_memptr(true_los_path);
            dtype *ptrI = different_output_size ? CI.slice_memptr(true_los_path) : coeff_im[f].slice_memptr(true_los_path);

            std::memcpy(p_coeff_re, ptrR, n_links * sizeof(dtype));
            std::memcpy(p_coeff_im, ptrI, n_links * sizeof(dtype));

            for (arma::uword k = 0; k < n_links; ++k)
                ptrR[k] = zero, ptrI[k] = zero;
        }

    } // End frequency loop (OMP parallel)
}

// Explicit template instantiations
template void quadriga_lib::get_channels_multifreq(
    const std::vector<quadriga_lib::arrayant<float>> &, const std::vector<quadriga_lib::arrayant<float>> &,
    float, float, float, float, float, float, float, float, float, float, float, float,
    const arma::Mat<float> &, const arma::Mat<float> &, const arma::Mat<float> &,
    const arma::Col<float> &, const arma::Cube<float> &, const arma::Col<float> &, const arma::Col<float> &,
    std::vector<arma::Cube<float>> &, std::vector<arma::Cube<float>> &, std::vector<arma::Cube<float>> &,
    bool, bool, float);

template void quadriga_lib::get_channels_multifreq(
    const std::vector<quadriga_lib::arrayant<double>> &, const std::vector<quadriga_lib::arrayant<double>> &,
    double, double, double, double, double, double, double, double, double, double, double, double,
    const arma::Mat<double> &, const arma::Mat<double> &, const arma::Mat<double> &,
    const arma::Col<double> &, const arma::Cube<double> &, const arma::Col<double> &, const arma::Col<double> &,
    std::vector<arma::Cube<double>> &, std::vector<arma::Cube<double>> &, std::vector<arma::Cube<double>> &,
    bool, bool, double);