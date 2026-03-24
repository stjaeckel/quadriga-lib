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

#include "quadriga_math.hpp"
#include "quadriga_channel.hpp"
#include "quadriga_lib_generic_functions.hpp"

#if defined(_MSC_VER) // Windows
#include <malloc.h>   // Include for _aligned_malloc and _aligned_free
#endif

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#define VEC_SIZE 8ULL
#else // AVX2 disabled
#define VEC_SIZE 1ULL
#endif

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# baseband_freq_response
Compute the baseband frequency response of a MIMO channel

## Description:
- Computes the frequency-domain response of a time-domain MIMO channel using a discrete Fourier transform (DFT).
- Input consists of real and imaginary channel coefficients and corresponding delays for each MIMO sub-link.
- Outputs the complex channel response matrix `H` at given sub-carrier frequency positions.
- Internally uses AVX2 instructions for fast parallel computation of 8 carriers at once.
- Can be efficiently called in a loop (e.g., over snapshots) and parallelized with OpenMP.
- Internal arithmetic is performed in single precision for speed.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::baseband_freq_response(
                const arma::Cube<dtype> *coeff_re,
                const arma::Cube<dtype> *coeff_im,
                const arma::Cube<dtype> *delay,
                const arma::Col<dtype> *pilot_grid,
                const double bandwidth,
                arma::Cube<dtype> *hmat_re,
                arma::Cube<dtype> *hmat_im);
```

## Arguments:
- `const arma::Cube<dtype> ***coeff_re**` (input)<br>
  Real part of channel coefficients in time domain, Size `[n_rx, n_tx, n_path]`.

- `const arma::Cube<dtype> ***coeff_im**` (input)<br>
  Imaginary part of channel coefficients in time domain, Size `[n_rx, n_tx, n_path]`.

- `const arma::Cube<dtype> ***delay**` (input)<br>
  Path delays in seconds. Size `[n_rx, n_tx, n_path]` or broadcastable shape `[1, 1, n_path]`.

- `const arma::Col<dtype> ***pilot_grid**` (input)<br>
  Normalized sub-carrier positions relative to bandwidth. Range: `0.0` (center freq) to `1.0` (center + bandwidth). Length: `n_carriers`.

- `const double **bandwidth**` (input)<br>
  Total baseband bandwidth in Hz (defines absolute frequency spacing of the pilot grid).

- `arma::Cube<dtype> ***hmat_re**` (output)<br>
  Output: Real part of the frequency-domain channel matrix, Size `[n_rx, n_tx, n_carriers]`.

- `arma::Cube<dtype> ***hmat_im**` (output)<br>
  Output: Imaginary part of the frequency-domain channel matrix, Size `[n_rx, n_tx, n_carriers]`.
MD!*/

template <typename dtype>
void quadriga_lib::baseband_freq_response(const arma::Cube<dtype> *coeff_re,     // Channel coefficients, real part, cube of size [n_rx, n_tx, n_path]
                                          const arma::Cube<dtype> *coeff_im,     // Channel coefficients, imaginary part, cube of size [n_rx, n_tx, n_path]
                                          const arma::Cube<dtype> *delay,        // Path delays in seconds, cube of size [n_rx, n_tx, n_path] or [1, 1, n_path]
                                          const arma::Col<dtype> *pilot_grid,    // Sub-carrier positions, relative to the bandwidth, 0.0 = fc, 1.0 = fc+bandwidth, Size: [ n_carriers ]
                                          const double bandwidth,                // The baseband bandwidth in [Hz]
                                          arma::Cube<dtype> *hmat_re,            // Output: Channel matrix (H), real part, Size [n_rx, n_tx, n_carriers]
                                          arma::Cube<dtype> *hmat_im,            // Output: Channel matrix (H), imaginary part, Size [n_rx, n_tx, n_carriers]
                                          arma::Cube<std::complex<dtype>> *hmat) // Output: Channel matrix (H), Size [n_rx, n_tx, n_carriers]
{

    // Input validation
    if (coeff_re == nullptr || coeff_im == nullptr || delay == nullptr || pilot_grid == nullptr)
        throw std::invalid_argument("Arguments cannot be NULL.");

    if (pilot_grid->n_elem == 0)
        throw std::invalid_argument("Pilot grid must have at least one element.");

    arma::uword n_rx = coeff_re->n_rows;
    arma::uword n_tx = coeff_re->n_cols;
    arma::uword n_ant = n_rx * n_tx;
    arma::uword n_path = coeff_re->n_slices;
    arma::uword n_carrier = pilot_grid->n_elem;
    arma::uword n_carrier_s = (n_carrier % VEC_SIZE == 0) ? n_carrier : VEC_SIZE * (n_carrier / VEC_SIZE + 1);

    if (coeff_im->n_rows != n_rx || coeff_im->n_cols != n_tx || coeff_im->n_slices != n_path)
        throw std::invalid_argument("Input 'coeff_im' must have same size as 'coeff_re'");

    if (!(delay->n_rows == n_rx && delay->n_cols == n_tx && delay->n_slices == n_path) &&
        !(delay->n_rows == 1 && delay->n_cols == 1 && delay->n_slices == n_path))
        throw std::invalid_argument("Input 'delay' must match the size as 'coeff_re'");

    if (bandwidth < 0.0)
        throw std::invalid_argument("Bandwidth cannot be negative");

    // Set output size
    if (hmat_re != nullptr)
    {
        if (hmat_re->n_rows != n_rx || hmat_re->n_cols != n_tx || hmat_re->n_slices != n_carrier)
            hmat_re->set_size(n_rx, n_tx, n_carrier);
        if (n_path == 0)
            hmat_re->zeros();
    }
    if (hmat_im != nullptr)
    {
        if (hmat_im->n_rows != n_rx || hmat_im->n_cols != n_tx || hmat_im->n_slices != n_carrier)
            hmat_im->set_size(n_rx, n_tx, n_carrier);
        if (n_path == 0)
            hmat_im->zeros();
    }
    if (hmat != nullptr)
    {
        if (hmat->n_rows != n_rx || hmat->n_cols != n_tx || hmat->n_slices != n_carrier)
            hmat->set_size(n_rx, n_tx, n_carrier);
        if (n_path == 0)
            hmat->zeros();
    }
    if (n_path == 0)
        return;

    // Declare constants
    const double scale = -6.283185307179586 * bandwidth;
    const bool planar_wave = delay->n_rows == 1 && delay->n_cols == 1;

// Allocate aligned memory
#if defined(_MSC_VER) // Windows
    float *phasor = (float *)_aligned_malloc(n_carrier_s * sizeof(float), 32);
    float *Hr = (float *)_aligned_malloc(n_carrier_s * n_ant * sizeof(float), 32);
    float *Hi = (float *)_aligned_malloc(n_carrier_s * n_ant * sizeof(float), 32);
#else // Linux
    float *phasor = (float *)aligned_alloc(32, n_carrier_s * sizeof(float));
    float *Hr = (float *)aligned_alloc(32, n_carrier_s * n_ant * sizeof(float));
    float *Hi = (float *)aligned_alloc(32, n_carrier_s * n_ant * sizeof(float));
#endif

    // Convert pilot grid to phasor
    const dtype *p_carrier = pilot_grid->memptr();
    for (arma::uword i_carrier = 0; i_carrier < n_carrier; ++i_carrier)
    {
        double tmp = scale * (double)p_carrier[i_carrier];
        phasor[i_carrier] = (float)tmp;
    }
    for (arma::uword i_carrier = n_carrier; i_carrier < n_carrier_s; ++i_carrier)
        phasor[i_carrier] = 0.0f;

    // Call DFT function
#if BUILD_WITH_AVX2
    if (runtime_AVX2_Check()) // CPU support for AVX2
    {
        qd_DFT_AVX2(coeff_re->memptr(), coeff_im->memptr(), delay->memptr(),
                    n_ant, n_path, planar_wave, phasor, n_carrier_s, Hr, Hi);
    }
    else
    {
        qd_DFT_GENERIC(coeff_re->memptr(), coeff_im->memptr(), delay->memptr(),
                       n_ant, n_path, planar_wave, phasor, n_carrier_s, Hr, Hi);
    }
#else // AVX2 disabled
    qd_DFT_GENERIC(coeff_re->memptr(), coeff_im->memptr(), delay->memptr(),
                   n_ant, n_path, planar_wave, phasor, n_carrier_s, Hr, Hi);
#endif

#if defined(_MSC_VER) // Windows
    _aligned_free(phasor);
#else // Linux
    free(phasor);
#endif

    // Copy data to output
    bool use_re = hmat_re != nullptr, use_im = hmat_im != nullptr, use_cplx = hmat != nullptr;
    dtype *p_hmat_re = use_re ? hmat_re->memptr() : nullptr;
    dtype *p_hmat_im = use_im ? hmat_im->memptr() : nullptr;
    std::complex<dtype> *p_hmat = use_cplx ? hmat->memptr() : nullptr;

    for (arma::uword i_carrier = 0; i_carrier < n_carrier; ++i_carrier)
        for (arma::uword i_ant = 0; i_ant < n_ant; ++i_ant)
        {
            arma::uword i = i_ant * n_carrier_s + i_carrier;
            arma::uword o = i_carrier * n_ant + i_ant;

            if (use_re)
                p_hmat_re[o] = (dtype)Hr[i];
            if (use_im)
                p_hmat_im[o] = (dtype)Hi[i];
            if (use_cplx)
                p_hmat[o] = {(dtype)Hr[i], (dtype)Hi[i]};
        }

// Free aligned memory
#if defined(_MSC_VER) // Windows
    _aligned_free(Hr);
    _aligned_free(Hi);
#else // Linux
    free(Hr);
    free(Hi);
#endif
}

template void quadriga_lib::baseband_freq_response(const arma::Cube<float> *coeff_re, const arma::Cube<float> *coeff_im, const arma::Cube<float> *delay,
                                                   const arma::Col<float> *pilot_grid, const double bandwidth,
                                                   arma::Cube<float> *hmat_re, arma::Cube<float> *hmat_im, arma::Cube<std::complex<float>> *hmat);

template void quadriga_lib::baseband_freq_response(const arma::Cube<double> *coeff_re, const arma::Cube<double> *coeff_im, const arma::Cube<double> *delay,
                                                   const arma::Col<double> *pilot_grid, const double bandwidth,
                                                   arma::Cube<double> *hmat_re, arma::Cube<double> *hmat_im, arma::Cube<std::complex<double>> *hmat);

/*!MD
# baseband_freq_response_vec
Compute the baseband frequency response of multiple MIMO channels

## Description:
- Computes the frequency-domain response of a batch of time-domain MIMO channels using a discrete Fourier transform (DFT).
- This function wraps `quadriga_lib::baseband_freq_response` and applies it across multiple snapshots in parallel using OpenMP.
- Input consists of vectors of real/imaginary coefficients and delay Cubes for each snapshot.
- Output is a vector of frequency-domain channel matrices `H` (one per snapshot).
- Can optionally compute a selected subset of snapshots using `i_snap`.
- Internal arithmetic is performed in single precision for performance.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::baseband_freq_response_vec(
                const std::vector<arma::Cube<dtype>> *coeff_re,
                const std::vector<arma::Cube<dtype>> *coeff_im,
                const std::vector<arma::Cube<dtype>> *delay,
                const arma::Col<dtype> *pilot_grid,
                const double bandwidth,
                std::vector<arma::Cube<dtype>> *hmat_re,
                std::vector<arma::Cube<dtype>> *hmat_im,
                const arma::u32_vec *i_snap = nullptr);
```

## Arguments:
- `const std::vector<arma::Cube<dtype>> ***coeff_re**` (input)<br>
  Real part of channel coefficients, vector of length `n_snap`. Each cube has shape `[n_rx, n_tx, n_path]`.

- `const std::vector<arma::Cube<dtype>> ***coeff_im**` (input)<br>
  Imaginary part of channel coefficients, same structure as `coeff_re`.

- `const std::vector<arma::Cube<dtype>> ***delay**` (input)<br>
  Path delays in seconds, same structure as `coeff_re`, shape can be broadcasted `[1, 1, n_path]`.

- `const arma::Col<dtype> ***pilot_grid**` (input)<br>
  Normalized sub-carrier positions relative to bandwidth. Range: `0.0` (center freq) to `1.0` (center + bandwidth). Length: `n_carriers`.

- `const double **bandwidth**` (input)<br>
  Total baseband bandwidth in Hz, used to compute sub-carrier frequencies.

- `std::vector<arma::Cube<dtype>> ***hmat_re**` (output)<br>
  Output: Real part of the frequency-domain channel matrices. Vector of length `n_out`. Each cube is `[n_rx, n_tx, n_carriers]`.

- `std::vector<arma::Cube<dtype>> ***hmat_im**` (output)<br>
  Output: Imaginary part of the frequency-domain channel matrices. Same structure as `hmat_re`.

- `const arma::u32_vec ***i_snap** = nullptr` (optional input)<br>
  Optional subset of snapshot indices to process. If omitted, all `n_snap` snapshots are processed. Length: `n_out`.

## See also:
- [[baseband_freq_response]] (for processing a single snapshot)
MD!*/

// Compute the baseband frequency response of multiple MIMO channels
template <typename dtype>
void quadriga_lib::baseband_freq_response_vec(const std::vector<arma::Cube<dtype>> *coeff_re, const std::vector<arma::Cube<dtype>> *coeff_im,
                                              const std::vector<arma::Cube<dtype>> *delay, const arma::Col<dtype> *pilot_grid, const double bandwidth,
                                              std::vector<arma::Cube<dtype>> *hmat_re, std::vector<arma::Cube<dtype>> *hmat_im, const arma::Col<unsigned> *i_snap)
{
    if (coeff_re == nullptr || coeff_im == nullptr || delay == nullptr || pilot_grid == nullptr || hmat_re == nullptr || hmat_im == nullptr)
        throw std::invalid_argument("Arguments cannot be NULL.");

    if (coeff_re->size() == 0 || pilot_grid->n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty");

    size_t n_snap_t = coeff_re->size();
    unsigned n_snap_u = (unsigned)n_snap_t;

    if (n_snap_t >= INT32_MAX)
        throw std::invalid_argument("Number of snapshots exceeds maximum supported number.");

    if (coeff_im->size() != n_snap_t || delay->size() != n_snap_t)
        throw std::invalid_argument("Coefficients and delays must have the same number of snapshots");

    if (bandwidth < 0.0)
        throw std::invalid_argument("Bandwidth cannot be negative");

    // Process the snapshot indices
    size_t n_snap_o = n_snap_t;
    arma::Col<unsigned> snap;
    if (i_snap == nullptr || i_snap->n_elem == 0)
    {
        snap.set_size(n_snap_o);
        unsigned *p = snap.memptr();
        for (unsigned i = 0; i < n_snap_u; ++i)
            p[i] = i;
    }
    else
    {
        n_snap_o = (size_t)i_snap->n_elem;
        snap.set_size(n_snap_o);
        unsigned *p = snap.memptr();
        const unsigned *q = i_snap->memptr();

        for (size_t i = 0; i < n_snap_o; ++i)
        {
            if (q[i] >= n_snap_u)
                throw std::invalid_argument("Snapshot indices cannot exceed number of mesh elements.");
            p[i] = q[i];
        }
    }

    // Reset output
    hmat_re->clear();
    hmat_im->clear();

    hmat_re->reserve(n_snap_o);
    hmat_im->reserve(n_snap_o);

    for (size_t i = 0; i < n_snap_o; ++i)
    {
        hmat_re->emplace_back();
        hmat_im->emplace_back();
    }

    unsigned *p_snap = snap.memptr();
    int n_snap_i = (int)n_snap_o;

#pragma omp parallel for
    for (int i = 0; i < n_snap_i; ++i)
    {
        unsigned i_snap = p_snap[i];
        quadriga_lib::baseband_freq_response<dtype>(&coeff_re->at(i_snap), &coeff_im->at(i_snap), &delay->at(i_snap),
                                                    pilot_grid, bandwidth, &hmat_re->at(i), &hmat_im->at(i));
    }
}

template void quadriga_lib::baseband_freq_response_vec(const std::vector<arma::Cube<float>> *coeff_re, const std::vector<arma::Cube<float>> *coeff_im,
                                                       const std::vector<arma::Cube<float>> *delay, const arma::Col<float> *pilot_grid, const double bandwidth,
                                                       std::vector<arma::Cube<float>> *hmat_re, std::vector<arma::Cube<float>> *hmat_im, const arma::Col<unsigned> *i_snap);

template void quadriga_lib::baseband_freq_response_vec(const std::vector<arma::Cube<double>> *coeff_re, const std::vector<arma::Cube<double>> *coeff_im,
                                                       const std::vector<arma::Cube<double>> *delay, const arma::Col<double> *pilot_grid, const double bandwidth,
                                                       std::vector<arma::Cube<double>> *hmat_re, std::vector<arma::Cube<double>> *hmat_im, const arma::Col<unsigned> *i_snap);

/*!MD
# baseband_freq_response_multi
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

## Description:
- Computes the frequency-domain channel transfer function H(f) from multi-frequency channel coefficients
  obtained by sampling the channel at a coarse set of carrier frequencies (e.g., every 500 MHz).
- Designed for ultra-wideband (UWB) channels where antenna patterns, path gains, and polarization transfer
  matrices vary across a large bandwidth (e.g., 500 MHz to 20 GHz).
- For each multipath component (MPC), the complex coefficient envelope is interpolated from the coarse
  input frequency grid `freq_in` to the dense output frequency grid `freq_out` using SLERP (Spherical
  Linear Interpolation): magnitude is interpolated linearly, phase is unwrapped and interpolated linearly
  along the shortest arc.
- The delay-induced phase rotation `exp(-j * 2 * pi * freq_out * delay)` is applied per output carrier.
  Since the output frequencies are absolute (not baseband offsets), the phase computation uses double
  precision internally to avoid loss of accuracy at high carrier frequencies.
- **Delay phase removal** (`remove_delay_phase = true`, default): Channel generation functions such as
  `get_channels_multifreq` bake the delay-induced phase `exp(-j * 2 * pi * freq_in[f] * delay)` into
  the output coefficients at each input frequency. This means the coefficient phase rotates rapidly across
  the input frequency grid (e.g., 50 full cycles per 500 MHz step for a 100 ns excess delay), which
  prevents meaningful SLERP interpolation. When `remove_delay_phase` is enabled, this function undoes the
  baked-in delay phase by multiplying each coefficient by `exp(+j * 2 * pi * freq_in[f] * delay)` before
  extracting the slowly-varying envelope for interpolation. The full delay phase at the output frequency
  is then re-applied analytically. This must be enabled when consuming output from `get_channels_multifreq`
  or `get_channels_spherical`. Set to `false` only if the input coefficients are already pure envelopes
  without baked-in delay phase (e.g., from manual construction or external sources).
- Delays are taken from `delay[0]` only, since path geometry (and thus propagation delay) is
  frequency-independent. All entries in the `delay` vector should contain identical values.
- The delay cube supports broadcastable shape `[1, 1, n_path]` (planar wave, same delay for all antenna
  pairs) or full shape `[n_rx, n_tx, n_path]` (spherical wave, per-antenna-pair delays).
- For output frequencies outside the range of `freq_in`, the nearest endpoint coefficient is used
  (constant extrapolation in magnitude/phase domain).
- The Nyquist condition for the coarse sampling grid applies to the coefficient envelope only (magnitude
  and phase of antenna patterns, path gains, and Jones matrix elements). The fast-rotating delay phase is
  handled analytically and does not require dense sampling. Adjacent envelope phase differences should stay
  well below pi to avoid interpolation artifacts.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::baseband_freq_response_multi(
                const std::vector<arma::Cube<dtype>> &coeff_re,
                const std::vector<arma::Cube<dtype>> &coeff_im,
                const std::vector<arma::Cube<dtype>> &delay,
                const arma::Col<dtype> &freq_in,
                const arma::Col<dtype> &freq_out,
                arma::Cube<dtype> *hmat_re = nullptr,
                arma::Cube<dtype> *hmat_im = nullptr,
                arma::Cube<std::complex<dtype>> *hmat = nullptr,
                bool remove_delay_phase = true);
```

## Arguments:
- `const std::vector<arma::Cube<dtype>> &**coeff_re**` (input)<br>
  Real part of channel coefficients at each input frequency. Vector of length `n_freq_in`, each cube
  has shape `[n_rx, n_tx, n_path]`.

- `const std::vector<arma::Cube<dtype>> &**coeff_im**` (input)<br>
  Imaginary part of channel coefficients at each input frequency. Same structure as `coeff_re`.

- `const std::vector<arma::Cube<dtype>> &**delay**` (input)<br>
  Path delays in seconds. Vector of length `n_freq_in`. Only `delay[0]` is used (delays are
  frequency-independent). Shape `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`.

- `const arma::Col<dtype> &**freq_in**` (input)<br>
  Input sample frequencies in Hz, sorted in ascending order. Length `[n_freq_in]`.

- `const arma::Col<dtype> &**freq_out**` (input)<br>
  Output carrier frequencies in Hz (absolute, not baseband offsets). Length `[n_carrier]`.

- `arma::Cube<dtype> ***hmat_re** = nullptr` (optional output)<br>
  Real part of the frequency-domain channel matrix. Size `[n_rx, n_tx, n_carrier]`.

- `arma::Cube<dtype> ***hmat_im** = nullptr` (optional output)<br>
  Imaginary part of the frequency-domain channel matrix. Size `[n_rx, n_tx, n_carrier]`.

- `arma::Cube<std::complex<dtype>> ***hmat** = nullptr` (optional output)<br>
  Complex-valued frequency-domain channel matrix. Size `[n_rx, n_tx, n_carrier]`.

- `bool **remove_delay_phase** = true` (optional input)<br>
  If `true` (default), the delay-induced phase `exp(-j * 2 * pi * freq_in[f] * delay)` that is baked
  into the input coefficients by channel generation functions (e.g., `get_channels_multifreq`) is removed
  before SLERP interpolation and re-applied at each output frequency. Must be enabled when the input
  comes from `get_channels_multifreq` or `get_channels_spherical`. Set to `false` only when the input
  coefficients contain pure slowly-varying envelopes without delay phase.

## Example:
```
// Using output from get_channels_multifreq (delay phase is baked in)
std::vector<arma::Cube<double>> coeff_re, coeff_im, delays;
arma::Col<double> freq_in = {0.5e9, 1.0e9, 1.5e9, 2.0e9};
arma::Col<double> freq_out = arma::linspace<arma::Col<double>>(0.5e9, 2.0e9, 2048);

// ... call get_channels_multifreq to populate coeff_re, coeff_im, delays ...

arma::Cube<double> Hr, Hi;
quadriga_lib::baseband_freq_response_multi(coeff_re, coeff_im, delays,
    freq_in, freq_out, &Hr, &Hi, nullptr, true);  // remove_delay_phase = true (default)
```

## See also:
- [[baseband_freq_response]] (for single-frequency narrowband channels)
- [[baseband_freq_response_vec]] (for batched narrowband channels)
- [[get_channels_multifreq]] (produces the multi-frequency coefficients consumed by this function)
MD!*/

template <typename dtype>
void quadriga_lib::baseband_freq_response_multi(const std::vector<arma::Cube<dtype>> &coeff_re, // Real part of coefficients, length [n_freq_in], each [n_rx, n_tx, n_path]
                                                const std::vector<arma::Cube<dtype>> &coeff_im, // Imag part of coefficients, length [n_freq_in], each [n_rx, n_tx, n_path]
                                                const std::vector<arma::Cube<dtype>> &delay,    // Delays [s], length [n_freq_in], each [n_rx, n_tx, n_path] or [1, 1, n_path]
                                                const arma::Col<dtype> &freq_in,                // Input sample frequencies [Hz], length [n_freq_in]
                                                const arma::Col<dtype> &freq_out,               // Output carrier frequencies [Hz], length [n_carrier]
                                                arma::Cube<dtype> *hmat_re,                     // Output: Channel matrix (H), real part, Size [n_rx, n_tx, n_carrier]
                                                arma::Cube<dtype> *hmat_im,                     // Output: Channel matrix (H), imaginary part, Size [n_rx, n_tx, n_carrier]
                                                arma::Cube<std::complex<dtype>> *hmat,          // Output: Channel matrix (H), complex-valued, Size [n_rx, n_tx, n_carrier]
                                                bool remove_delay_phase)                        // Remove baked-in delay phase before SLERP interpolation
{
    // --- Input validation ---

    size_t n_freq_in = (size_t)freq_in.n_elem;
    size_t n_carrier = (size_t)freq_out.n_elem;

    if (n_freq_in == 0)
        throw std::invalid_argument("Input 'freq_in' must have at least one element.");

    if (n_carrier == 0)
        throw std::invalid_argument("Input 'freq_out' must have at least one element.");

    if (coeff_re.size() != n_freq_in)
        throw std::invalid_argument("Length of 'coeff_re' must match length of 'freq_in'.");

    if (coeff_im.size() != n_freq_in)
        throw std::invalid_argument("Length of 'coeff_im' must match length of 'freq_in'.");

    if (delay.size() != n_freq_in)
        throw std::invalid_argument("Length of 'delay' must match length of 'freq_in'.");

    // Verify that freq_in is sorted in strictly ascending order
    for (size_t f = 1; f < n_freq_in; ++f)
        if (freq_in[f] <= freq_in[f - 1])
            throw std::invalid_argument("Input 'freq_in' must be sorted in strictly ascending order.");

    // Get dimensions from first element
    arma::uword n_rx = coeff_re[0].n_rows;
    arma::uword n_tx = coeff_re[0].n_cols;
    arma::uword n_ant = n_rx * n_tx;
    arma::uword n_path = coeff_re[0].n_slices;

    if (n_rx == 0 || n_tx == 0)
        throw std::invalid_argument("Coefficient cubes must not be empty.");

    // Validate all coefficient cubes have consistent dimensions
    for (size_t f = 0; f < n_freq_in; ++f)
    {
        if (coeff_re[f].n_rows != n_rx || coeff_re[f].n_cols != n_tx || coeff_re[f].n_slices != n_path)
            throw std::invalid_argument("All cubes in 'coeff_re' must have the same dimensions.");
        if (coeff_im[f].n_rows != n_rx || coeff_im[f].n_cols != n_tx || coeff_im[f].n_slices != n_path)
            throw std::invalid_argument("All cubes in 'coeff_im' must have the same dimensions.");
    }

    // Validate delay[0] shape (only delay[0] is used)
    const arma::Cube<dtype> &delay0 = delay[0];
    bool planar_wave = (delay0.n_rows == 1 && delay0.n_cols == 1 && delay0.n_slices == n_path);

    if (!planar_wave && !(delay0.n_rows == n_rx && delay0.n_cols == n_tx && delay0.n_slices == n_path))
        throw std::invalid_argument("Input 'delay[0]' must have shape [n_rx, n_tx, n_path] or [1, 1, n_path].");

    // --- Set output size ---

    bool use_re = (hmat_re != nullptr);
    bool use_im = (hmat_im != nullptr);
    bool use_cplx = (hmat != nullptr);

    if (use_re)
    {
        if (hmat_re->n_rows != n_rx || hmat_re->n_cols != n_tx || hmat_re->n_slices != (arma::uword)n_carrier)
            hmat_re->set_size(n_rx, n_tx, (arma::uword)n_carrier);
        if (n_path == 0)
            hmat_re->zeros();
    }
    if (use_im)
    {
        if (hmat_im->n_rows != n_rx || hmat_im->n_cols != n_tx || hmat_im->n_slices != (arma::uword)n_carrier)
            hmat_im->set_size(n_rx, n_tx, (arma::uword)n_carrier);
        if (n_path == 0)
            hmat_im->zeros();
    }
    if (use_cplx)
    {
        if (hmat->n_rows != n_rx || hmat->n_cols != n_tx || hmat->n_slices != (arma::uword)n_carrier)
            hmat->set_size(n_rx, n_tx, (arma::uword)n_carrier);
        if (n_path == 0)
            hmat->zeros();
    }
    if (n_path == 0)
        return;

    // --- Precompute interpolation table ---
    // For each output carrier k, find the left segment index and fractional weight in freq_in

    const dtype *p_fin = freq_in.memptr();
    const dtype *p_fout = freq_out.memptr();

    std::vector<size_t> seg(n_carrier);    // Left segment index into freq_in
    std::vector<double> frac(n_carrier);   // Fractional position within segment [0, 1]
    std::vector<double> phasor(n_carrier); // -2 * pi * freq_out[k]

    for (size_t k = 0; k < n_carrier; ++k)
    {
        phasor[k] = -6.283185307179586 * (double)p_fout[k];

        if (n_freq_in == 1)
        {
            seg[k] = 0;
            frac[k] = 0.0;
        }
        else
        {
            // Clamp to input frequency range (constant extrapolation)
            if (p_fout[k] <= p_fin[0])
            {
                seg[k] = 0;
                frac[k] = 0.0;
            }
            else if (p_fout[k] >= p_fin[n_freq_in - 1])
            {
                seg[k] = n_freq_in - 2;
                frac[k] = 1.0;
            }
            else
            {
                // Linear search (n_freq_in is small, typically ~40)
                size_t s = 0;
                for (size_t i = 1; i < n_freq_in - 1; ++i)
                    if (p_fout[k] >= p_fin[i])
                        s = i;
                seg[k] = s;
                double denom = (double)p_fin[s + 1] - (double)p_fin[s];
                frac[k] = (denom > 0.0) ? ((double)p_fout[k] - (double)p_fin[s]) / denom : 0.0;
            }
        }
    }

    // --- Get raw pointers to coefficient data for each input frequency ---

    std::vector<const dtype *> p_cre(n_freq_in), p_cim(n_freq_in);
    for (size_t f = 0; f < n_freq_in; ++f)
    {
        p_cre[f] = coeff_re[f].memptr();
        p_cim[f] = coeff_im[f].memptr();
    }
    const dtype *p_delay = delay0.memptr();

    // --- Precompute delay-phase undo phasors for each input frequency ---
    // When remove_delay_phase is true, we need +2*pi*freq_in[f] to undo the baked-in exp(-j*2*pi*f*tau)
    std::vector<double> undo_phasor(n_freq_in, 0.0);
    if (remove_delay_phase)
    {
        for (size_t f = 0; f < n_freq_in; ++f)
            undo_phasor[f] = 6.283185307179586 * (double)p_fin[f];
    }

    // --- Get output pointers ---

    dtype *p_hmat_re = use_re ? hmat_re->memptr() : nullptr;
    dtype *p_hmat_im = use_im ? hmat_im->memptr() : nullptr;
    std::complex<dtype> *p_hmat = use_cplx ? hmat->memptr() : nullptr;

    // --- SLERP + accumulate core (vectorized) ---
    // Loop order: path-outer, antenna-inner.
    // For planar waves, the delay-phase sincos depends only on the path (not the antenna),
    // so fast_sincos is called once per path and reused across all n_ant antenna pairs.
    // For spherical waves, fast_sincos is called per (path, antenna) pair.
    // SLERP interpolation is vectorized across all n_carrier output frequencies per call.
    // Accumulators are kept in double precision to preserve summation accuracy across paths.

    const arma::uword n_carrier_u = (arma::uword)n_carrier;

    // Pre-build SLERP weight vector from fractional interpolation table (path/antenna-independent)
    arma::Col<dtype> w_vec;
    if (n_freq_in > 1)
    {
        w_vec.set_size(n_carrier_u);
        dtype *pw = w_vec.memptr();
        for (size_t k = 0; k < n_carrier; ++k)
            pw[k] = (dtype)frac[k];
    }

    // Working buffers (allocated once, reused across iterations)
    arma::dvec theta(n_carrier_u);       // delay-phase angles, double for range reduction
    arma::fvec s_dl, c_dl;               // sincos output, single precision
    arma::Col<dtype> slerp_Ar, slerp_Ai; // SLERP gather: left bracket
    arma::Col<dtype> slerp_Br, slerp_Bi; // SLERP gather: right bracket
    arma::fvec Xr, Xi;                   // SLERP output, single precision
    if (n_freq_in > 1)
    {
        slerp_Ar.set_size(n_carrier_u);
        slerp_Ai.set_size(n_carrier_u);
        slerp_Br.set_size(n_carrier_u);
        slerp_Bi.set_size(n_carrier_u);
    }

    // Envelope samples for one (antenna, path) pair across input frequencies
    std::vector<double> env_re(n_freq_in), env_im(n_freq_in);

    // Accumulators for all antenna pairs, flat layout [i_ant * n_carrier + k], double precision
    std::vector<double> Hr(n_ant * n_carrier, 0.0), Hi(n_ant * n_carrier, 0.0);

    for (size_t i_path = 0; i_path < n_path; ++i_path)
    {
        // --- Planar wave: compute delay-phase sincos once per path ---
        double dl_planar = 0.0;
        if (planar_wave)
        {
            dl_planar = (double)p_delay[i_path];
            double *p_theta = theta.memptr();
            for (size_t k = 0; k < n_carrier; ++k)
                p_theta[k] = phasor[k] * dl_planar;
            quadriga_lib::fast_sincos(theta, &s_dl, &c_dl); // double input: range-reduced internally
        }

        for (size_t i_ant = 0; i_ant < n_ant; ++i_ant)
        {
            const size_t idx = i_path * n_ant + i_ant;

            // --- Get delay for this path/antenna pair ---
            double dl;
            if (planar_wave)
            {
                dl = dl_planar;
            }
            else
            {
                dl = (double)p_delay[idx];
                // Spherical wave: compute delay-phase sincos per (path, antenna)
                double *p_theta = theta.memptr();
                for (size_t k = 0; k < n_carrier; ++k)
                    p_theta[k] = phasor[k] * dl;
                quadriga_lib::fast_sincos(theta, &s_dl, &c_dl);
            }

            // --- Extract coefficient envelope samples ---
            // If remove_delay_phase is enabled, undo the baked-in exp(-j*2*pi*freq_in[f]*delay)
            // by multiplying with exp(+j*2*pi*freq_in[f]*delay) to recover the slowly-varying envelope
            for (size_t f = 0; f < n_freq_in; ++f)
            {
                double re = (double)p_cre[f][idx];
                double im = (double)p_cim[f][idx];

                if (remove_delay_phase)
                {
                    double undo_phase = undo_phasor[f] * dl; // +2*pi*freq_in[f]*delay
                    double c_undo = std::cos(undo_phase);
                    double s_undo = std::sin(undo_phase);
                    double re_env = re * c_undo - im * s_undo;
                    double im_env = re * s_undo + im * c_undo;
                    re = re_env;
                    im = im_env;
                }

                env_re[f] = re;
                env_im[f] = im;
            }

            // --- SLERP interpolation across output carriers ---
            const float *pXr, *pXi;

            if (n_freq_in == 1)
            {
                // No interpolation: broadcast the single envelope sample into Xr, Xi
                Xr.set_size(n_carrier_u);
                Xi.set_size(n_carrier_u);
                Xr.fill((float)env_re[0]);
                Xi.fill((float)env_im[0]);
                pXr = Xr.memptr();
                pXi = Xi.memptr();
            }
            else
            {
                // Gather SLERP bracket inputs from envelope via the segment table
                dtype *pAr = slerp_Ar.memptr(), *pAi = slerp_Ai.memptr();
                dtype *pBr = slerp_Br.memptr(), *pBi = slerp_Bi.memptr();
                for (size_t k = 0; k < n_carrier; ++k)
                {
                    const size_t s = seg[k];
                    pAr[k] = (dtype)env_re[s];
                    pAi[k] = (dtype)env_im[s];
                    pBr[k] = (dtype)env_re[s + 1];
                    pBi[k] = (dtype)env_im[s + 1];
                }

                quadriga_lib::fast_slerp(slerp_Ar, slerp_Ai, slerp_Br, slerp_Bi, w_vec, Xr, Xi);
                pXr = Xr.memptr();
                pXi = Xi.memptr();
            }

            // --- Accumulate: interpolated envelope * exp(-j*2*pi*freq_out*delay) ---
            // Cast float products to double before accumulating to preserve summation precision
            const float *pC = c_dl.memptr(), *pS = s_dl.memptr();
            double *pHr = &Hr[i_ant * n_carrier];
            double *pHi = &Hi[i_ant * n_carrier];

            for (size_t k = 0; k < n_carrier; ++k)
            {
                const double xr = (double)pXr[k], xi = (double)pXi[k];
                const double cd = (double)pC[k], sd = (double)pS[k];
                pHr[k] += xr * cd - xi * sd;
                pHi[k] += xr * sd + xi * cd;
            }
        }
    }

    // --- Write output for all antenna pairs ---
    for (size_t i_ant = 0; i_ant < n_ant; ++i_ant)
    {
        const double *pHr = &Hr[i_ant * n_carrier];
        const double *pHi = &Hi[i_ant * n_carrier];
        for (size_t k = 0; k < n_carrier; ++k)
        {
            size_t o = k * n_ant + i_ant; // Column-major index [n_rx, n_tx, n_carrier]
            if (use_re)
                p_hmat_re[o] = (dtype)pHr[k];
            if (use_im)
                p_hmat_im[o] = (dtype)pHi[k];
            if (use_cplx)
                p_hmat[o] = {(dtype)pHr[k], (dtype)pHi[k]};
        }
    }
}

template void quadriga_lib::baseband_freq_response_multi(const std::vector<arma::Cube<float>> &coeff_re, const std::vector<arma::Cube<float>> &coeff_im,
                                                         const std::vector<arma::Cube<float>> &delay, const arma::Col<float> &freq_in,
                                                         const arma::Col<float> &freq_out, arma::Cube<float> *hmat_re, arma::Cube<float> *hmat_im,
                                                         arma::Cube<std::complex<float>> *hmat, bool remove_delay_phase);

template void quadriga_lib::baseband_freq_response_multi(const std::vector<arma::Cube<double>> &coeff_re, const std::vector<arma::Cube<double>> &coeff_im,
                                                         const std::vector<arma::Cube<double>> &delay, const arma::Col<double> &freq_in,
                                                         const arma::Col<double> &freq_out, arma::Cube<double> *hmat_re, arma::Cube<double> *hmat_im,
                                                         arma::Cube<std::complex<double>> *hmat, bool remove_delay_phase);