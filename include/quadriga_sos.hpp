// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_sos_H
#define quadriga_sos_H

#include <armadillo>
#include <string>
#include <vector>

// If arma::uword and size_t are not the same width (e.g. 64 bit), the compiler will throw an error here
// This allows the use of "uword", "size_t" and "unsigned long long" interchangeably
// This requires a 64 bit platform, but will compile on Linux, Windows and macOS
static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");

namespace quadriga_lib
{

    // ---- Sum-of-sinusoids random number generator ----

    // A generator with N sinusoids produces at position x the value s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n )
    // with the frequencies f_n (rows of 'freq') and the phases phi_n (one column of 'phase' per generator)
    // The functions are documented in src/sos_*.cpp

    // Frequencies and ACF of a predefined sum-of-sinusoids generator
    // - Presets: Exp, Gauss or Comb followed by the number of sinusoids 128, 256, 512 or 1024, e.g. "Comb256"
    template <typename dtype>
    void sos_preset(const std::string &name,          // Name of the preset, e.g. "Comb256"
                    arma::Mat<dtype> &freq,           // Output: Frequencies in [1/m], [n_coeff, 3]
                    arma::Col<dtype> *acf = nullptr,  // Optional output: Target ACF, [200]
                    arma::Col<dtype> *dist = nullptr, // Optional output: Sample points of the ACF in [m], [200]
                    dtype dist_decorr = 10.0);        // Decorrelation distance in [m], must be > 0

    // Draw random phases for a set of sum-of-sinusoids generators
    // - Uniform in [-pi, pi); for the same seed, column g does not depend on n_gen
    template <typename dtype>
    void sos_init_phases(arma::uword n_coeff,     // Number of sinusoids, must be > 0
                         arma::uword n_gen,       // Number of generators, must be > 0
                         arma::Mat<dtype> &phase, // Output: Phases in [rad], [n_coeff, n_gen]
                         arma::sword seed = -1);  // Seed, -1 = system random device

    // Target ACF at given distances
    // - Linear interpolation of the ACF table; distances beyond the last sample point return the last value
    template <typename dtype>
    void sos_acfi(const arma::Col<dtype> &acf,  // Target ACF, [n_acf]
                  const arma::Col<dtype> &dist, // Sample points of the ACF in [m], [n_acf]
                  const arma::Col<dtype> &d,    // Distances in [m], must be >= 0, [n_d]
                  arma::Col<dtype> &val);       // Output: ACF at d, [n_d]

    // Spatially correlated random values at given positions
    // - One device (pos_b omitted), two devices on the same map (pos_b given, phase_b omitted; needs acf and dist)
    //   or two devices on different maps (pos_b and phase_b given)
    template <typename dtype>
    void sos_val(const arma::Mat<dtype> &freq,              // Frequencies in [1/m], [n_coeff, n_dim]
                 const arma::Mat<dtype> &phase,             // Phases in [rad], [n_coeff, n_gen]
                 const arma::Mat<dtype> &pos,               // Positions of the first device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                 arma::Mat<dtype> &val,                     // Output: Values, [n_pos, n_gen]
                 const arma::Mat<dtype> *pos_b = nullptr,   // Optional: Positions of the second device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                 const arma::Mat<dtype> *phase_b = nullptr, // Optional: Phases of the second device in [rad], [n_coeff, n_gen]
                 const arma::Col<dtype> *acf = nullptr,     // Optional: Target ACF, [n_acf]
                 const arma::Col<dtype> *dist = nullptr,    // Optional: Sample points of the ACF in [m], [n_acf]
                 dtype freq_scale_b = 1.0,                  // Frequency scale of the second device, d_first / d_second, must be > 0
                 bool uniform = false,                      // Output distribution: false = Normal, true = Uniform in [0, 1]
                 int use_kernel = 0);                       // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Spatially correlated random values on a regular grid
    // - One device; output in meshgrid order: map[g](iy, ix, iz) belongs to position {x(ix), y(iy), z(iz)}
    template <typename dtype>
    void sos_map(const arma::Mat<dtype> &freq,          // Frequencies in [1/m], [n_coeff, n_dim]
                 const arma::Mat<dtype> &phase,         // Phases in [rad], [n_coeff, n_gen]
                 const arma::Col<dtype> &x,             // Grid points on the x-axis in [m], [nx]
                 std::vector<arma::Cube<dtype>> &map,   // Output: Values, vector (n_gen) of cubes [ny, nx, nz]
                 const arma::Col<dtype> *y = nullptr,   // Optional: Grid points on the y-axis in [m], [ny], default: ny = 1
                 const arma::Col<dtype> *z = nullptr,   // Optional: Grid points on the z-axis in [m], [nz], default: nz = 1
                 bool uniform = false,                  // Output distribution: false = Normal, true = Uniform in [0, 1]
                 int use_kernel = 0);                   // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Spatially correlated random numbers: Normal (sos_randn), Uniform (sos_rand), integer (sos_randi)
    // - One independent generator per entry of dist_decorr, built from the preset with phases drawn from the seed
    // - With pos_b both devices share one map, unless dist_decorr_b differs for that generator (two maps)
    template <typename dtype>
    void sos_randn(const arma::Col<dtype> &dist_decorr,     // Decorrelation distances in [m], must be >= 0, [n_gen]
                   const arma::Mat<dtype> &pos,             // Positions of the first device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                   arma::Mat<dtype> &val,                   // Output: Values, Normal(0,1), [n_pos, n_gen]
                   const arma::Mat<dtype> *pos_b = nullptr, // Optional: Positions of the second device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                   const arma::Col<dtype> *dist_decorr_b = nullptr, // Optional: Decorrelation distances of the second device in [m], [n_gen]
                   const std::string &preset = "Comb256",   // Name of the preset, see sos_preset
                   arma::sword seed = -1);                  // Seed, -1 = system random device

    template <typename dtype>
    void sos_rand(const arma::Col<dtype> &dist_decorr,     // Decorrelation distances in [m], must be >= 0, [n_gen]
                  const arma::Mat<dtype> &pos,             // Positions of the first device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                  arma::Mat<dtype> &val,                   // Output: Values, Uniform in [0, 1], [n_pos, n_gen]
                  const arma::Mat<dtype> *pos_b = nullptr, // Optional: Positions of the second device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                  const arma::Col<dtype> *dist_decorr_b = nullptr, // Optional: Decorrelation distances of the second device in [m], [n_gen]
                  const std::string &preset = "Comb256",   // Name of the preset, see sos_preset
                  arma::sword seed = -1);                  // Seed, -1 = system random device

    template <typename dtype>
    void sos_randi(const arma::Col<dtype> &dist_decorr,     // Decorrelation distances in [m], must be >= 0, [n_gen]
                   const arma::Mat<dtype> &pos,             // Positions of the first device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                   arma::uword imax,                        // Largest value, must be > 0
                   arma::Mat<dtype> &val,                   // Output: Integer values 1 ... imax, [n_pos, n_gen]
                   const arma::Mat<dtype> *pos_b = nullptr, // Optional: Positions of the second device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                   const arma::Col<dtype> *dist_decorr_b = nullptr, // Optional: Decorrelation distances of the second device in [m], [n_gen]
                   const std::string &preset = "Comb256",   // Name of the preset, see sos_preset
                   arma::sword seed = -1);                  // Seed, -1 = system random device

    // Remove the sample mean of the generators over a set of positions by a phase offset
    // - Closed form, one offset per generator (two with phase_b); same device modes as sos_val
    template <typename dtype>
    void sos_remove_bias(const arma::Mat<dtype> &freq,            // Frequencies in [1/m], [n_coeff, n_dim]
                         arma::Mat<dtype> &phase,                 // Phases in [rad], updated in-place, [n_coeff, n_gen]
                         const arma::Mat<dtype> &pos,             // Positions of the first device in [m], [n_pos, 1], [n_pos, 2] or [n_pos, 3]
                         const arma::Mat<dtype> *pos_b = nullptr, // Optional: Positions of the second device in [m], [n_pos_b, 1], [n_pos_b, 2] or [n_pos_b, 3]
                         arma::Mat<dtype> *phase_b = nullptr,     // Optional: Phases of the second device in [rad], updated in-place, [n_coeff, n_gen]
                         const arma::Col<dtype> *acf = nullptr,   // Optional: Target ACF, [n_acf]
                         const arma::Col<dtype> *dist = nullptr,  // Optional: Sample points of the ACF in [m], [n_acf]
                         arma::Mat<dtype> *offset = nullptr,      // Optional output: Applied offsets in [rad], [n_gen, 1], or [n_gen, 2] with phase_b
                         arma::Mat<dtype> *mean_before = nullptr, // Optional output: Sample mean before the correction, [n_gen, 1]; with phase_b the mean of each device's own field, [n_gen, 2]
                         dtype freq_scale_b = 1.0);               // Frequency scale of the second device, d_first / d_second, must be > 0

    // Target ACF on a 2-D grid
    // - Grid axis from -2·Dmax to 2·Dmax with the spacing of 'dist', Dmax = last sample point, n_grid = 4·n_acf - 3
    template <typename dtype>
    void sos_acf_2d(const arma::Col<dtype> &acf,          // Target ACF, [n_acf]
                    const arma::Col<dtype> &dist,         // Sample points of the ACF in [m], [n_acf]
                    arma::Mat<dtype> &acf_2d,             // Output: 2-D ACF, [n_grid, n_grid]
                    arma::Col<dtype> *dist_2d = nullptr); // Optional output: Grid axis in [m], [n_grid]

    // ACF produced by a set of sinusoids, on the grid axis of sos_acf_2d
    // - n_dim = 1: along the axis; n_dim = 2: x-y plane; n_dim = 3: x-y, x-z and y-z plane (slices 0, 1, 2)
    template <typename dtype>
    void sos_acf_approx(const arma::Mat<dtype> &freq,         // Frequencies in [1/m], [n_coeff, n_dim]
                        const arma::Col<dtype> &dist,         // Sample points of the ACF in [m], [n_acf]
                        arma::Cube<dtype> &acf_approx,        // Output: Approximated ACF, see above
                        arma::Col<dtype> *dist_2d = nullptr); // Optional output: Grid axis in [m], [n_grid]

    // Approximation error of the ACF of a set of sinusoids
    // - All errors are returned as -10·log10( mean squared error ), larger is better
    // - Return value: error along the test directions on the sample points 'dist'
    template <typename dtype>
    dtype sos_calc_mse(const arma::Mat<dtype> &freq,         // Frequencies in [1/m], [n_coeff, n_dim]
                       const arma::Col<dtype> &acf,          // Target ACF, [n_acf]
                       const arma::Col<dtype> &dist,         // Sample points of the ACF in [m], equally spaced, [n_acf]
                       dtype *mse_core = nullptr,            // Optional output: Error within the core radius in [dB]
                       dtype *mse_all = nullptr,             // Optional output: Error on the complete grid in [dB]
                       arma::Col<dtype> *acf_tail = nullptr, // Optional output: Approximated ACF beyond the fitted range, [4]
                       arma::Mat<dtype> *test_dir = nullptr, // Optional output: Test directions in [rad], [n_dir, 2] = {azimuth, elevation}; [n_dir, 1] for n_dim = 2; empty for n_dim = 1
                       arma::uword n_dir = 10);              // Number of test directions, must be > 0

    // Monte-Carlo estimate of the ACF of one generator
    // - Correlates the values at 2·n_pos random positions for all position pairs whose distance falls into a bin
    template <typename dtype>
    void sos_acf_estimate(const arma::Mat<dtype> &freq,             // Frequencies in [1/m], [n_coeff, 3]
                          const arma::Col<dtype> &phase,            // Phases of the generator in [rad], [n_coeff]
                          const arma::Col<dtype> &bins,             // Bin centers in [m], >= 0, increasing, finite, [n_bin]
                          arma::uword n_pos,                        // Number of random positions per device, must be > 1
                          arma::Col<dtype> &acf_est,                // Output: Estimated ACF, [n_bin]
                          arma::Col<dtype> *dist_est = nullptr,     // Optional output: Mean distance of the pairs in each bin in [m], [n_bin]
                          arma::Col<dtype> *acf_est_dual = nullptr, // Optional output: Estimated ACF for two devices, [n_bin]
                          const arma::Col<dtype> *acf = nullptr,    // Optional: Target ACF, required for acf_est_dual, [n_acf]
                          const arma::Col<dtype> *dist = nullptr,   // Optional: Sample points of the ACF in [m], [n_acf]
                          arma::sword seed = -1);                   // Seed for the positions, -1 = system random device

    // Find the frequencies of a sum-of-sinusoids generator for a given ACF
    // - Iterative fit, one sinusoid at a time; computation in double precision for both instantiations
    template <typename dtype>
    void sos_generate(const arma::Col<dtype> &acf,          // Target ACF, [n_acf]
                      const arma::Col<dtype> &dist,         // Sample points of the ACF in [m], [n_acf]
                      arma::uword n_coeff,                  // Number of sinusoids, must be > 0
                      arma::Mat<dtype> &freq,               // Output: Frequencies in [1/m], [n_coeff, n_dim]
                      arma::Col<dtype> *ase = nullptr,      // Optional output: Mean squared error after each pass, linear, [n_pass]
                      arma::Mat<dtype> *test_dir = nullptr, // Optional output: Test directions in [rad], as in sos_calc_mse
                      arma::uword n_dim = 3,                // Number of dimensions, 1, 2 or 3
                      bool uniform_smp = true,              // Regular (true) or random (false) directions of the sinusoids
                      arma::uword n_dir = 10,               // Number of test directions, must be > 0
                      bool random_init = true,              // Start from random frequencies (true) or build up sinusoid by sinusoid (false)
                      dtype freq_min = 0.0,                 // Lower limit of abs(f) in [1/m], 0 = none
                      dtype freq_max = 0.0,                 // Upper limit of abs(f) in [1/m], 0 = 15 / max(dist)
                      arma::sword seed = -1);               // Seed, -1 = system random device
}

#endif
