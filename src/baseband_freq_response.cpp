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

#include "quadriga_channel.hpp"
#include "quadriga_lib_avx2_functions.hpp"

// Vector size for AVX2
#define VEC_SIZE 8

// Testing for AVX2 support at runtime
#if defined(_MSC_VER) // Windows
#include <intrin.h>
#include <malloc.h> // Include for _aligned_malloc and _aligned_free
#else               // Linux
#include <cpuid.h>
#endif

static bool isAVX2Supported()
{
    std::vector<int> cpuidInfo(4);

#if defined(_MSC_VER) // Windows
    return false;
#else // Linux
    __cpuid_count(7, 0, cpuidInfo[0], cpuidInfo[1], cpuidInfo[2], cpuidInfo[3]);
#endif

    return (cpuidInfo[1] & (1 << 5)) != 0; // Check the AVX2 bit in EBX
}

// Generic C++ implementation of DFT
template <typename dtype>
static inline void qd_DFT_GENERIC(const dtype *CFr,       // Channel coefficients, real part, Size [n_ant, n_path]
                                  const dtype *CFi,       // Channel coefficients, imaginary part, Size [n_ant, n_path]
                                  const dtype *DL,        // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                                  const size_t n_ant,     // Number of MIMO sub-links
                                  const size_t n_path,    // Number multipath components
                                  const bool planar_wave, // Indicator that same delays are used for all antennas
                                  const float *phasor,    // Phasor, -pi/2 to pi/2, aligned to 32 byte, Size [ n_carrier ]
                                  const size_t n_carrier, // Number of carriers, mutiple of 8
                                  float *Hr,              // Channel matrix, real part, Size [ n_carrier, n_ant ]
                                  float *Hi)              // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]
{

    for (size_t i_path = 0; i_path < n_path; ++i_path) // Path loop
        for (size_t i_ant = 0; i_ant < n_ant; ++i_ant) // Antenna loop
        {
            size_t i = i_path * n_ant + i_ant;
            float cr = (float)CFr[i];
            float ci = (float)CFi[i];
            float dl = planar_wave ? (float)DL[i_path] : (float)DL[i];

            for (size_t i_carrier = 0; i_carrier < n_carrier; ++i_carrier)
            {
                float phase = phasor[i_carrier] * dl;
                float si = std::sin(phase), co = std::cos(phase);

                size_t o = i_ant * n_carrier + i_carrier;
                Hr[o] += co * cr - si * ci;
                Hi[o] += co * ci + si * cr;
            }
        }
}

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

    // Initialize output memory
    for (arma::uword i = 0; i < n_carrier_s * n_ant; ++i)
        Hr[i] = 0.0f, Hi[i] = 0.0f;

    // Call DFT function
#if defined(_MSC_VER) // Windows
    qd_DFT_GENERIC(coeff_re->memptr(), coeff_im->memptr(), delay->memptr(),
                   n_ant, n_path, planar_wave, phasor, n_carrier_s, Hr, Hi);
#else // Linux
    if (isAVX2Supported()) // CPU support for AVX2
    {
        qd_DFT_AVX2(coeff_re->memptr(), coeff_im->memptr(), delay->memptr(),
                    n_ant, n_path, planar_wave, phasor, n_carrier_s, Hr, Hi);
    }
    else
    {
        qd_DFT_GENERIC(coeff_re->memptr(), coeff_im->memptr(), delay->memptr(),
                       n_ant, n_path, planar_wave, phasor, n_carrier_s, Hr, Hi);
    }
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
