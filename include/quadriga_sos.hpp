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

    // Conventions shared by all functions of this group:
    // - A generator with N sinusoids in n_dim dimensions (1, 2 or 3) produces at position x the value
    //   s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n ); the amplitude follows from N and is not an argument
    // - 'freq' holds the spatial frequencies f_n in [1/m], [n_coeff, n_dim]; all generators of one call share it
    // - 'phase' holds the phases phi_n in [rad], one column per generator, [n_coeff, n_gen]
    // - Positions in [m] have one row per position; the number of columns sets the number of dimensions:
    //   [n_pos, 1] = {x}, [n_pos, 2] = {x, y} or [n_pos, 3] = {x, y, z}; coordinates that are not given are 0;
    //   not more columns than 'freq'
    // - Results have one row per position and one column per generator, [n_pos, n_gen]
    // - The target ACF is a table: 'acf' sampled at the distances 'dist' in [m], both [n_acf]; 'dist' starts at 0
    //   and increases strictly; beyond the last sample point the last value is used
    // - Seeds: a seed >= 0 gives the same random numbers on all platforms (std::mt19937_64, upper 53 bits,
    //   no distribution classes of the standard library); -1 seeds from the system random device
    // - Directions on the sphere are a spiral (Fibonacci) lattice on the upper hemisphere, directions on the
    //   circle are equally spaced on the half circle; a direction and its opposite are equivalent
    // - The "MATLAB:" line above each declaration is the usage line of its MEX wrapper

    // Frequencies and ACF of a predefined sum-of-sinusoids generator
    // - Presets: Exp128, Exp256, Exp512, Exp1024, Gauss128, Gauss256, Gauss512, Gauss1024, Comb128, Comb256,
    //   Comb512, Comb1024; the number is the number of sinusoids; the name is matched exactly
    // - Exp: R = exp(-D/d); Gauss: R = exp(-(D/d)^2); Comb: Gauss up to D = d, Exp beyond; d = decorrelation distance
    // - The tables are stored for d = 10 m (200 ACF samples, 0 to 49.75 m) and 3 dimensions; for another
    //   decorrelation distance: dist = dist_10 · d/10, freq = freq_10 · 10/d
    // - No sinusoid has a wavelength above 20 decorrelation distances, so the ACF stays near 0 beyond the table
    // MATLAB: [ freq, acf, dist ] = quadriga_lib.sos_preset( name, dist_decorr );
    template <typename dtype>
    void sos_preset(const std::string &name,          // Name of the preset, e.g. "Comb256"
                    arma::Mat<dtype> &freq,           // Output: Frequencies in [1/m], [n_coeff, 3]
                    arma::Col<dtype> *acf = nullptr,  // Optional output: Target ACF, [200]
                    arma::Col<dtype> *dist = nullptr, // Optional output: Sample points of the ACF in [m], [200]
                    dtype dist_decorr = 10.0);        // Decorrelation distance in [m], must be > 0

    // Draw random phases for a set of sum-of-sinusoids generators
    // - Values are uniform in [-pi, pi)
    // - Filled column by column from one random stream: for the same seed, column g does not depend on n_gen
    // MATLAB: phase = quadriga_lib.sos_init_phases( n_coeff, n_gen, seed );
    template <typename dtype>
    void sos_init_phases(arma::uword n_coeff,     // Number of sinusoids, must be > 0
                         arma::uword n_gen,       // Number of generators, must be > 0
                         arma::Mat<dtype> &phase, // Output: Phases in [rad], [n_coeff, n_gen]
                         arma::sword seed = -1);  // Seed, -1 = system random device

    // Target ACF at given distances
    // - Linear interpolation of the ACF table; distances beyond the last sample point return the last value
    // - The same rule is used inside sos_val and sos_remove_bias for two devices on the same map
    // MATLAB: val = quadriga_lib.sos_acfi( acf, dist, d );
    template <typename dtype>
    void sos_acfi(const arma::Col<dtype> &acf,  // Target ACF, [n_acf]
                  const arma::Col<dtype> &dist, // Sample points of the ACF in [m], [n_acf]
                  const arma::Col<dtype> &d,    // Distances in [m], must be >= 0, [n_d]
                  arma::Col<dtype> &val);       // Output: ACF at d, [n_d]

    // Spatially correlated random values at given positions
    // - One device (pos_b omitted): val = s(pos) for each phase column
    // - Two devices on the same map (pos_b given, phase_b omitted): both devices use 'phase';
    //   val = ( s(pos) + s(pos_b) ) / sqrt( 2 + 2·A(D) ), D = distance between pos and pos_b, A = target ACF;
    //   'acf' and 'dist' are required
    // - Two devices on different maps (pos_b and phase_b given): val = ( s_a(pos) + s_b(pos_b) ) / sqrt(2),
    //   s_b uses 'phase_b' and the frequencies freq · freq_scale_b; 'acf' and 'dist' are not used
    // - The mode follows from the arguments only; phase values are never compared
    // - freq_scale_b != 1 requires phase_b (two decorrelation distances on one map have no consistent ACF)
    // - uniform = false: values are approximately Normal(0,1); uniform = true: 0.5 · erfc( -val / sqrt(2) )
    // - AVX2 kernel: sin and cos in single precision after range reduction of the argument in double,
    //   absolute error of val below 1e-5 independent of the distance from the origin; GENERIC: double throughout
    // - The value at a position does not depend on the other positions or generators of the call,
    //   nor on the number of threads
    // MATLAB: val = quadriga_lib.sos_val( freq, phase, pos, pos_b, phase_b, acf, dist, freq_scale_b, uniform, use_kernel );
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
                 bool uniform = false,                      // Output distribution: false = Normal, true = Uniform in (0, 1)
                 int use_kernel = 0);                       // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Spatially correlated random values on a regular grid
    // - One device only; evaluates sos_val at all combinations of the grid vectors
    // - Output in meshgrid order: map[g](iy, ix, iz) belongs to position {x(ix), y(iy), z(iz)}
    // - The number of grid vectors given must not exceed n_dim; omitted vectors mean a single sample at 0
    // MATLAB: map = quadriga_lib.sos_map( freq, phase, x, y, z, uniform, use_kernel );   % [ny, nx, nz, n_gen]
    template <typename dtype>
    void sos_map(const arma::Mat<dtype> &freq,          // Frequencies in [1/m], [n_coeff, n_dim]
                 const arma::Mat<dtype> &phase,         // Phases in [rad], [n_coeff, n_gen]
                 const arma::Col<dtype> &x,             // Grid points on the x-axis in [m], [nx]
                 std::vector<arma::Cube<dtype>> &map,   // Output: Values, vector (n_gen) of cubes [ny, nx, nz]
                 const arma::Col<dtype> *y = nullptr,   // Optional: Grid points on the y-axis in [m], [ny], default: ny = 1
                 const arma::Col<dtype> *z = nullptr,   // Optional: Grid points on the z-axis in [m], [nz], default: nz = 1
                 bool uniform = false,                  // Output distribution: false = Normal, true = Uniform in (0, 1)
                 int use_kernel = 0);                   // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

    // Spatially correlated random numbers: Normal (sos_randn), Uniform (sos_rand), integer (sos_randi)
    // - One independent generator per entry of dist_decorr, built from the preset with new phases
    //   (sos_init_phases with the given seed); a decorrelation distance of 0 is replaced by 0.1 m
    // - With pos_b both devices are on the same map (see sos_val), unless dist_decorr_b is given and differs
    //   from dist_decorr for that generator: then the second device gets its own phases and its own
    //   decorrelation distance (two maps)
    // - The phases of the first device do not depend on whether a second device is given
    // - sos_randi returns ceil( u · imax ) with u from sos_rand, limited to 1 ... imax
    // MATLAB: val = quadriga_lib.sos_randn( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed );
    // MATLAB: val = quadriga_lib.sos_rand( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed );
    // MATLAB: val = quadriga_lib.sos_randi( dist_decorr, pos, imax, pos_b, dist_decorr_b, preset, seed );
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
                  arma::Mat<dtype> &val,                   // Output: Values, Uniform in (0, 1), [n_pos, n_gen]
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
    // - Adds one offset to all phases of a generator so that the mean of the Normal-distributed values of
    //   sos_val over the given positions is zero; closed form, no iteration:
    //   offset = pi/2 - arg(Z) or that value minus pi, whichever is smaller in magnitude,
    //   Z = sum_n z_n · exp( j·phi_n ), z_n = mean over the positions of exp( j·2·pi · f_n·x )
    // - One device: one offset per generator
    // - Two devices on the same map (pos_b given, phase_b omitted): one offset per generator that zeroes the
    //   mean of the combined value; 'acf' and 'dist' are required
    // - Two devices on different maps (pos_b and phase_b given): each device is zeroed on its own positions,
    //   two offsets per generator; freq_scale_b as in sos_val
    // - This conditions the field on a zero sample mean: the variance at the positions drops by about the
    //   variance of the sample mean, which is small only if the positions spread over many decorrelation distances
    // - The zero mean holds for exactly these positions, frequencies and phases
    // - Phases are returned in [-pi, pi); a generator with abs(Z) below 1e-14 · n_coeff is left unchanged
    // MATLAB: [ phase_out, phase_b_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( freq, phase_in, pos,
    //             pos_b, phase_b_in, acf, dist, freq_scale_b );
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
    // - Rotationally symmetric; 0 at radial distances above Dmax
    // - 'dist' must be equally spaced
    // MATLAB: [ acf_2d, dist_2d ] = quadriga_lib.sos_acf_2d( acf, dist );
    template <typename dtype>
    void sos_acf_2d(const arma::Col<dtype> &acf,          // Target ACF, [n_acf]
                    const arma::Col<dtype> &dist,         // Sample points of the ACF in [m], [n_acf]
                    arma::Mat<dtype> &acf_2d,             // Output: 2-D ACF, [n_grid, n_grid]
                    arma::Col<dtype> *dist_2d = nullptr); // Optional output: Grid axis in [m], [n_grid]

    // ACF produced by a set of sinusoids
    // - R(dx) = (1/N) · sum_n cos( 2·pi · f_n·dx ), on the same grid axis as sos_acf_2d
    // - n_dim = 1: along the axis, [n_grid, 1, 1]; n_dim = 2: x-y plane, [n_grid, n_grid, 1], element (iy, ix);
    //   n_dim = 3: x-y, x-z and y-z plane, [n_grid, n_grid, 3], row index = second axis of the plane
    // - 'dist' must be equally spaced
    // MATLAB: [ acf_approx, dist_2d ] = quadriga_lib.sos_acf_approx( freq, dist );
    template <typename dtype>
    void sos_acf_approx(const arma::Mat<dtype> &freq,         // Frequencies in [1/m], [n_coeff, n_dim]
                        const arma::Col<dtype> &dist,         // Sample points of the ACF in [m], [n_acf]
                        arma::Cube<dtype> &acf_approx,        // Output: Approximated ACF, see above
                        arma::Col<dtype> *dist_2d = nullptr); // Optional output: Grid axis in [m], [n_grid]

    // Approximation error of the ACF of a set of sinusoids
    // - All errors are returned as -10·log10( mean squared error ), larger is better
    // - Return value: error along the test directions on the sample points 'dist'
    //   (n_dim = 1: one direction; n_dim = 2: n_dir directions on the half circle; n_dim = 3: n_dir directions
    //   on the hemisphere)
    // - mse_core: error on the grid of sos_acf_approx against sos_acf_2d within a radius of n_acf grid steps, averaged over the planes
    // - mse_all: same on the complete grid
    // - acf_tail: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times Dmax;
    //   the target is 0 there; a table with sinusoids near zero frequency shows a plateau
    // - For n_dim = 1, mse_core and mse_all are evaluated on the 1-D axis and the return value equals mse_core
    // MATLAB: [ mse, mse_core, mse_all, acf_tail, test_dir ] = quadriga_lib.sos_calc_mse( freq, acf, dist, n_dir );
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
    // - Draws 2·n_pos random positions in a cube with half edge 0.4·max(bins), evaluates the generator and
    //   correlates the values of all position pairs whose distance falls into a bin
    // - Bin n collects distances between the midpoints to the neighbouring bin centers; NaN for bins with
    //   fewer than 6 pairs
    // - acf_est_dual: same for two devices on the same map (first n_pos positions = first device, second
    //   n_pos = second device), pairs selected by the distance of the first devices while the second devices
    //   are no further apart than the bin limit; requires 'acf' and 'dist'
    // MATLAB: [ acf_est, dist_est, acf_est_dual ] = quadriga_lib.sos_acf_estimate( freq, phase, bins, n_pos, acf, dist, seed );
    template <typename dtype>
    void sos_acf_estimate(const arma::Mat<dtype> &freq,             // Frequencies in [1/m], [n_coeff, 3]
                          const arma::Col<dtype> &phase,            // Phases of the generator in [rad], [n_coeff]
                          const arma::Col<dtype> &bins,             // Bin centers in [m], increasing, [n_bin]
                          arma::uword n_pos,                        // Number of random positions per device, must be > 1
                          arma::Col<dtype> &acf_est,                // Output: Estimated ACF, [n_bin]
                          arma::Col<dtype> *dist_est = nullptr,     // Optional output: Mean distance of the pairs in each bin in [m], [n_bin]
                          arma::Col<dtype> *acf_est_dual = nullptr, // Optional output: Estimated ACF for two devices, [n_bin]
                          const arma::Col<dtype> *acf = nullptr,    // Optional: Target ACF, required for acf_est_dual, [n_acf]
                          const arma::Col<dtype> *dist = nullptr,   // Optional: Sample points of the ACF in [m], [n_acf]
                          arma::sword seed = -1);                   // Seed for the positions, -1 = system random device

    // Find the frequencies of a sum-of-sinusoids generator for a given ACF
    // - Iterative fit: one sinusoid at a time is re-fitted against the residual of all others along its own
    //   direction (coarse search over the frequency grid, then step refinement); an update is kept only if
    //   the mean squared error over the test directions decreases; stops when a pass changes nothing or
    //   gains less than a factor of 1e-6, at the latest after 1000 passes
    // - Directions of the sinusoids: n_dim = 1: none; n_dim = 2: on the half circle; n_dim = 3: on the hemisphere;
    //   uniform_smp = true places them regularly, false draws them at random
    // - The fit covers the range of 'dist' only; sinusoids near zero frequency keep the ACF above 0 at all
    //   larger distances, set freq_min to prevent this
    // - With few test directions the result fits those directions only; the presets were made with 300
    // - A seed makes the result repeatable on one platform, not bit-identical across math libraries
    // - freq_min, freq_max limit abs(f) of the search; 0 selects the default: no lower limit, 15 / max(dist)
    // - Computation in double precision for both instantiations
    // MATLAB: [ freq, ase, test_dir ] = quadriga_lib.sos_generate( acf, dist, n_coeff, n_dim, uniform_smp, n_dir,
    //             random_init, freq_min, freq_max, seed );
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
