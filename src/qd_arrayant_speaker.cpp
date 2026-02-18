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

#include "quadriga_arrayant.hpp"
#include "quadriga_tools.hpp"
#include "qd_arrayant_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <stdexcept>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# generate_speaker
Generate a parametric loudspeaker directivity model

## Description:
- Generates frequency-dependent loudspeaker radiation patterns by combining a driver directivity model 
  with an enclosure radiation modifier and a Butterworth-style bandpass frequency response.
- Returns a vector of `arrayant` objects, one per frequency sample, where each contains the complex-valued 
  directivity balloon at that frequency stored in `e_theta_re`. The `center_frequency` field of each 
  arrayant is set to the corresponding frequency in Hz.
- Multi-driver speakers (e.g. two-way systems) are modelled by generating each driver separately and 
  combining them via `append` and `element_pos`. Crossover behavior emerges naturally from overlapping 
  bandpass responses.
- The frequency response follows a Butterworth-style bandpass filter: 
  H(f) = 1/sqrt(1 + (f_low/f)^(2n_low)) * 1/sqrt(1 + (f/f_high)^(2n_high)), where the filter order 
  n = slope_dB_per_octave / 6. The response is -3 dB at the specified cutoff frequencies.
- Sensitivity is applied as a linear amplitude scaling relative to 85 dB SPL: 
  sens_lin = 10^((sensitivity - 85) / 20).
- If no frequency vector is provided, third-octave band center frequencies are auto-generated 
  covering the range from one band below the lower cutoff to one band above the upper cutoff, 
  clipped to 20-20000 Hz.
- The speed of sound is assumed to be 344 m/s.
- Allowed datatypes (`dtype`): `float` or `double`

Three driver directivity models are supported:
- **Piston**: Circular piston in a baffle, using the classical formula 
  D(theta) = 2*J1(ka*sin(theta)) / (ka*sin(theta)), where k = 2*pi*f/c is the wavenumber and a is 
  the driver radius. Rotationally symmetric around the forward axis (+x). Transitions from omnidirectional 
  at low ka to progressively narrower beaming at high ka. Uses C++17 `std::cyl_bessel_j` for the Bessel f
  unction J1.
- **Horn**: Separable cosine-power directivity with frequency-dependent pattern control. Horizontal 
  and vertical coverage are modelled independently as cos^n(angle), where the exponent n is derived 
  from the coverage angle: n = log(0.5) / log(cos(theta_3dB/2)). Below the horn control frequency, 
  the pattern blends toward omnidirectional using a second-order transition: 
  blend = (f/f_ctrl)^2 / (1 + (f/f_ctrl)^2). Horn parameters can be auto-derived from the mouth radius.
- **Omni**: Frequency-independent omnidirectional pattern (D = 1 at all angles). Suitable for 
  subwoofers at frequencies where the driver is small relative to the wavelength.

Four enclosure radiation types modify the base driver pattern:
- **Monopole**: No modification (4*pi radiation, R = 1 everywhere). Appropriate for subwoofers in free space.
- **Hemisphere**: Sealed box on a finite baffle with frequency-dependent transition from omnidirectional 
  to forward-weighted radiation. Uses a baffle step model: alpha = (f/f_baffle)^2 / (1 + (f/f_baffle)^2), 
  where f_baffle = c / (pi * sqrt(W*H)) is derived from the baffle dimensions. The radiation modifier is 
  R = (1-alpha) + alpha * 0.5 * (1 + cos(theta_off)). At low frequencies the pattern is nearly 
  omnidirectional; at high frequencies it approaches a cardioid-like front weighting with approximately
  6 dB front-to-back ratio.
- **Dipole**: Open baffle / planar speaker with figure-8 pattern. R = abs(cos(theta_off)) with a 180-degree 
  phase inversion (negative sign in `e_theta_re`) in the rear hemisphere. Nulls occur at 90 degrees off-axis.
- **Cardioid**: Monopole + dipole combination, R = 0.5 * (1 + cos(theta_off)). Null at the rear, 
  half-amplitude at the sides. No phase inversion.

## Declaration:
```
std::vector<arrayant<dtype>> quadriga_lib::generate_speaker(
        std::string driver_type = "piston",
        dtype radius = 0.05,
        dtype lower_cutoff = 80.0,
        dtype upper_cutoff = 12000.0,
        dtype lower_rolloff_slope = 12.0,
        dtype upper_rolloff_slope = 12.0,
        dtype sensitivity = 85.0,
        std::string radiation_type = "hemisphere",
        dtype hor_coverage = 0.0,
        dtype ver_coverage = 0.0,
        dtype horn_control_freq = 0.0,
        dtype baffle_width = 0.15,
        dtype baffle_height = 0.25,
        arma::Col<dtype> frequencies = arma::Col<dtype>(),
        dtype angular_resolution = 5.0)
```

## Arguments:
- `std::string **driver_type** = "piston"` (optional input)<br>
  Driver directivity model. Supported values: `"piston"` (cone/dome via Bessel function), `"horn"` 
  (cosine-power with frequency-dependent pattern control), `"omni"` (omnidirectional subwoofer).

- `dtype **radius** = 0.05` (optional input)<br>
  Effective radiating radius in meters. For `"piston"`: the cone or dome radius. For `"horn"`: the horn 
  mouth radius, from which the pattern control frequency is derived if not specified. Default: 0.05 m 
  (approximately a 4-inch driver).

- `dtype **lower_cutoff** = 80.0` (optional input)<br>
  Lower -3 dB frequency of the bandpass response in Hz. Default: 80 Hz.

- `dtype **upper_cutoff** = 12000.0` (optional input)<br>
  Upper -3 dB frequency of the bandpass response in Hz. Default: 12000 Hz.

- `dtype **lower_rolloff_slope** = 12.0` (optional input)<br>
  Low-frequency rolloff slope in dB per octave. Corresponds to a Butterworth filter of order 
  n = slope/6 (e.g. 12 dB/oct = 2nd order). Default: 12.0.

- `dtype **upper_rolloff_slope** = 12.0` (optional input)<br>
  High-frequency rolloff slope in dB per octave. Default: 12.0.

- `dtype **sensitivity** = 85.0` (optional input)<br>
  On-axis sensitivity in dB SPL at 1W/1m. Scales the pattern amplitude linearly relative to 85 dB as 
  reference (i.e. 85 dB gives unity amplitude). Default: 85.0.

- `std::string **radiation_type** = "hemisphere"` (optional input)<br>
  Enclosure radiation model. Supported values: `"monopole"` (4*pi, no modification), `"hemisphere"` 
  (sealed box with frequency-dependent baffle step), `"dipole"` (open baffle, figure-8), `"cardioid"` 
  (monopole + dipole combination).

- `dtype **hor_coverage** = 0.0` (optional input)<br>
  Horizontal coverage angle in degrees. Horn driver only. A value of 0 auto-defaults to 90 degrees. Default: 0.0.

- `dtype **ver_coverage** = 0.0` (optional input)<br>
  Vertical coverage angle in degrees. Horn driver only. A value of 0 auto-defaults to 60 degrees. Default: 0.0.

- `dtype **horn_control_freq** = 0.0` (optional input)<br>
  Frequency in Hz above which the horn maintains pattern control. A value of 0 auto-derives from the 
  mouth radius as f_ctrl = c / (2*pi*radius). Default: 0.0.

- `dtype **baffle_width** = 0.15` (optional input)<br>
  Enclosure baffle width in meters. Used by the `"hemisphere"` radiation model to compute the baffle 
  step frequency. Default: 0.15 m.

- `dtype **baffle_height** = 0.25` (optional input)<br>
  Enclosure baffle height in meters. Used by the `"hemisphere"` radiation model to compute the baffle 
  step frequency. Default: 0.25 m.

- `arma::Col<dtype> **frequencies** = arma::Col<dtype>()` (optional input)<br>
  Frequency sample points in Hz. If empty, third-octave band center frequencies are auto-generated 
  overing one band beyond each cutoff frequency, clipped to the 20-20000 Hz audible range.

- `dtype **angular_resolution** = 5.0` (optional input)<br>
  Resolution of the azimuth and elevation sampling grids in degrees. Default: 5.0.

## Returns:
- `std::vector<arrayant<dtype>>`<br>
  Vector of arrayant objects, one per frequency sample. Each arrayant has one element with the directivity 
  pattern stored in `e_theta_re` (real-valued except for dipole rear hemisphere where negative values encode 
  180-degree phase inversion). The `center_frequency` field is set to the corresponding frequency in Hz.

## Example:
```
// Generate a default cheap piston speaker at specific frequencies
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
                12.0, 12.0, 85.0, "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);

// Generate a horn driver with auto-derived parameters at third-octave bands
auto horn = quadriga_lib::generate_speaker<double>("horn");

// Generate a subwoofer
arma::vec sub_freqs = {30.0, 50.0, 80.0, 120.0, 200.0};
auto sub = quadriga_lib::generate_speaker<double>("omni", 0.13, 30.0, 200.0,
                12.0, 24.0, 92.0, "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, sub_freqs, 10.0);
```
MD!*/


// Piston directivity function: 2*J1(x)/x, returns 1.0 at x=0
// Uses C++17 std::cyl_bessel_j for J1
static double qd_piston_directivity(double x)
{
    double ax = std::abs(x);
    if (ax < 1.0e-10)
        return 1.0;
    return 2.0 * std::cyl_bessel_j(1.0, ax) / ax;
}

template <typename dtype>
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::generate_speaker(std::string driver_type, dtype radius,
                                                                          dtype lower_cutoff, dtype upper_cutoff,
                                                                          dtype lower_rolloff_slope, dtype upper_rolloff_slope,
                                                                          dtype sensitivity, std::string radiation_type,
                                                                          dtype hor_coverage, dtype ver_coverage, dtype horn_control_freq,
                                                                          dtype baffle_width, dtype baffle_height,
                                                                          arma::Col<dtype> frequencies, dtype angular_resolution)
{
    // Constants
    const double c_sound = 344.0;        // Speed of sound in [m/s]
    const double pi = arma::datum::pi;   // 3.14159...
    const double two_pi = 2.0 * pi;     // 6.28318...

    // --- Input validation ---
    if (driver_type != "piston" && driver_type != "horn" && driver_type != "omni")
        throw std::invalid_argument("generate_speaker: driver_type must be 'piston', 'horn', or 'omni'.");

    if (radiation_type != "monopole" && radiation_type != "hemisphere" && radiation_type != "dipole" && radiation_type != "cardioid")
        throw std::invalid_argument("generate_speaker: radiation_type must be 'monopole', 'hemisphere', 'dipole', or 'cardioid'.");

    // --- Fix input ranges ---
    radius = (radius <= dtype(0.0)) ? dtype(0.05) : radius;
    lower_cutoff = (lower_cutoff <= dtype(0.0)) ? dtype(80.0) : lower_cutoff;
    upper_cutoff = (upper_cutoff <= lower_cutoff) ? dtype(12000.0) : upper_cutoff;
    lower_rolloff_slope = (lower_rolloff_slope <= dtype(0.0)) ? dtype(12.0) : lower_rolloff_slope;
    upper_rolloff_slope = (upper_rolloff_slope <= dtype(0.0)) ? dtype(12.0) : upper_rolloff_slope;
    sensitivity = (sensitivity <= dtype(0.0)) ? dtype(85.0) : sensitivity;
    angular_resolution = (angular_resolution <= dtype(0.0)) ? dtype(5.0) : (angular_resolution > dtype(90.0) ? dtype(90.0) : angular_resolution);
    baffle_width = (baffle_width <= dtype(0.0)) ? dtype(0.15) : baffle_width;
    baffle_height = (baffle_height <= dtype(0.0)) ? dtype(0.25) : baffle_height;

    // --- Auto-derive horn parameters from radius when set to 0 ---
    if (driver_type == "horn")
    {
        if (horn_control_freq <= dtype(0.0))
            horn_control_freq = dtype(c_sound / (two_pi * (double)radius));
        if (hor_coverage <= dtype(0.0))
            hor_coverage = dtype(90.0);
        if (ver_coverage <= dtype(0.0))
            ver_coverage = dtype(60.0);
    }

    // --- Generate third-octave frequency samples if not provided ---
    if (frequencies.is_empty())
    {
        // Third-octave bands: f_n = 1000 * 2^(n/3)
        double f_ref = 1000.0;
        int n_start = (int)std::floor(3.0 * std::log2((double)lower_cutoff / f_ref)) - 1;
        int n_end = (int)std::ceil(3.0 * std::log2((double)upper_cutoff / f_ref)) + 1;

        std::vector<dtype> freq_vec;
        for (int n = n_start; n <= n_end; ++n)
        {
            double f = f_ref * std::pow(2.0, (double)n / 3.0);
            if (f >= 20.0 && f <= 20000.0)
                freq_vec.push_back(dtype(f));
        }

        if (freq_vec.empty())
            throw std::invalid_argument("generate_speaker: Could not generate frequency samples in the audible range.");

        frequencies.set_size((arma::uword)freq_vec.size());
        for (arma::uword i = 0; i < (arma::uword)freq_vec.size(); ++i)
            frequencies[i] = freq_vec[i];
    }

    arma::uword n_freq = frequencies.n_elem;

    // --- Build angular grids ---
    arma::uword no_az = arma::uword(360.0 / (double)angular_resolution) + 1ULL;
    arma::uword no_el = arma::uword(180.0 / (double)angular_resolution) + 1ULL;

    dtype pi_f = dtype(pi);
    dtype pih_f = dtype(pi * 0.5);
    arma::Col<dtype> az_grid = arma::linspace<arma::Col<dtype>>(-pi_f, pi_f, no_az);
    arma::Col<dtype> el_grid = arma::linspace<arma::Col<dtype>>(-pih_f, pih_f, no_el);

    // --- Precompute trigonometric values for all grid points ---
    // Forward direction is +x axis: azimuth=0, elevation=0
    // Off-axis angle theta_off: cos(theta_off) = cos(el) * cos(az)
    arma::Mat<double> cos_az_d(1, no_az);
    arma::Mat<double> cos_el_d(no_el, 1);

    for (arma::uword ia = 0; ia < no_az; ++ia)
    {
        cos_az_d(0, ia) = std::cos((double)az_grid[ia]);
    }
    for (arma::uword ie = 0; ie < no_el; ++ie)
    {
        cos_el_d(ie, 0) = std::cos((double)el_grid[ie]);
    }

    // Precompute cos(theta_off) and sin(theta_off) for the full grid
    arma::Mat<double> cos_off(no_el, no_az), sin_off(no_el, no_az);
    for (arma::uword ia = 0; ia < no_az; ++ia)
        for (arma::uword ie = 0; ie < no_el; ++ie)
        {
            double ct = cos_el_d(ie, 0) * cos_az_d(0, ia);
            double st_sq = 1.0 - ct * ct;
            cos_off(ie, ia) = ct;
            sin_off(ie, ia) = std::sqrt(st_sq > 0.0 ? st_sq : 0.0);
        }

    // --- Baffle step frequency for hemisphere radiation type ---
    // Frequency where the transition from omnidirectional to hemispherical occurs
    // Based on geometric mean of baffle dimensions
    double baffle_mean = std::sqrt((double)baffle_width * (double)baffle_height);
    double f_baffle = c_sound / (pi * baffle_mean);

    // --- Sensitivity: linear amplitude relative to 85 dB SPL reference ---
    double sens_lin = std::pow(10.0, ((double)sensitivity - 85.0) / 20.0);

    // --- Horn: compute cosine-power exponents from coverage angles ---
    // For cos^n(theta) pattern: -3dB at theta_3dB means cos^n(theta_3dB) = 0.5
    // Solving: n = log(0.5) / log(cos(theta_3dB/2))
    // (theta_3dB/2 because the coverage angle is the full -3dB width)
    double horn_exp_h = 1.0, horn_exp_v = 1.0;
    if (driver_type == "horn")
    {
        double theta_h = (double)hor_coverage * 0.5 * pi / 180.0; // Half-angle in radians
        double theta_v = (double)ver_coverage * 0.5 * pi / 180.0;
        double cos_h = std::cos(theta_h);
        double cos_v = std::cos(theta_v);
        if (cos_h > 1.0e-10 && cos_h < 1.0)
            horn_exp_h = std::log(0.5) / std::log(cos_h);
        if (cos_v > 1.0e-10 && cos_v < 1.0)
            horn_exp_v = std::log(0.5) / std::log(cos_v);
    }

    // --- Frequency response filter order ---
    // Butterworth: n = slope_dB_per_octave / 6
    double n_low = (double)lower_rolloff_slope / 6.0;
    double n_high = (double)upper_rolloff_slope / 6.0;

    // --- Generate output: one arrayant per frequency sample ---
    std::vector<quadriga_lib::arrayant<dtype>> output(n_freq);

    for (arma::uword i_freq = 0; i_freq < n_freq; ++i_freq)
    {
        double freq = (double)frequencies[i_freq];
        double wavelength = c_sound / freq;
        double k = two_pi / wavelength;       // Wavenumber [rad/m]
        double ka = k * (double)radius;        // ka product (dimensionless)

        // --- Frequency response: Butterworth-style bandpass ---
        // H(f) = 1/sqrt(1 + (f_low/f)^(2n)) * 1/sqrt(1 + (f/f_high)^(2n))
        double ratio_low = (double)lower_cutoff / freq;
        double ratio_high = freq / (double)upper_cutoff;
        double H_low = 1.0 / std::sqrt(1.0 + std::pow(ratio_low, 2.0 * n_low));
        double H_high = 1.0 / std::sqrt(1.0 + std::pow(ratio_high, 2.0 * n_high));
        double freq_response = H_low * H_high * sens_lin;

        // --- Initialize the arrayant for this frequency ---
        quadriga_lib::arrayant<dtype> &ant = output[i_freq];
        ant.name = "speaker_" + driver_type;
        ant.azimuth_grid = az_grid;
        ant.elevation_grid = el_grid;
        ant.e_theta_re.zeros(no_el, no_az, 1);
        ant.e_theta_im.zeros(no_el, no_az, 1);
        ant.e_phi_re.zeros(no_el, no_az, 1);
        ant.e_phi_im.zeros(no_el, no_az, 1);
        ant.element_pos.zeros(3, 1);
        ant.coupling_re.ones(1, 1);
        ant.coupling_im.zeros(1, 1);
        ant.center_frequency = dtype(freq);

        dtype *pat_re = ant.e_theta_re.memptr();

        // --- Compute the directivity balloon for each grid point ---
        for (arma::uword ia = 0; ia < no_az; ++ia)
        {
            double c_az = cos_az_d(0, ia);  // cos(azimuth)

            for (arma::uword ie = 0; ie < no_el; ++ie)
            {
                arma::uword idx = ia * no_el + ie; // Column-major index
                double c_el = cos_el_d(ie, 0);
                double ct = cos_off(ie, ia);        // cos(off-axis angle from +x)
                double st = sin_off(ie, ia);        // sin(off-axis angle from +x)

                // ===== Step 1: Driver directivity =====
                double D = 1.0;

                if (driver_type == "piston")
                {
                    // Circular piston in an infinite baffle
                    // D(theta) = 2*J1(ka*sin(theta)) / (ka*sin(theta))
                    // Rotationally symmetric around the forward axis
                    D = qd_piston_directivity(ka * st);
                }
                else if (driver_type == "horn")
                {
                    // Horn with separable horizontal/vertical coverage
                    // Pattern control degrades below horn_control_freq

                    // Frequency-dependent pattern control blend factor
                    // blend = 0 at f << f_ctrl (omni), blend = 1 at f >> f_ctrl (full control)
                    double ctrl_ratio = freq / (double)horn_control_freq;
                    double blend = ctrl_ratio * ctrl_ratio / (1.0 + ctrl_ratio * ctrl_ratio);

                    // Horizontal directivity: cos^n_h(az) in the forward hemisphere
                    double D_h = 1.0;
                    if (c_az > 0.0)
                        D_h = std::pow(c_az, horn_exp_h);
                    else
                        D_h = 0.0;

                    // Vertical directivity: cos^n_v(el), always non-negative since |el| <= pi/2
                    // Clamp to non-negative to avoid NaN from pow(tiny_negative, non_integer)
                    double c_el_pos = (c_el > 0.0) ? c_el : 0.0;
                    double D_v = std::pow(c_el_pos, horn_exp_v);

                    double D_horn = D_h * D_v;

                    // Blend: at low freq, horn acts increasingly omnidirectional
                    D = (1.0 - blend) * 1.0 + blend * D_horn;
                }
                // else "omni": D stays 1.0

                // ===== Step 2: Radiation type modifier =====
                // Modifies the pattern based on enclosure type
                double R = 1.0;
                bool rear_phase_flip = false;

                if (radiation_type == "hemisphere")
                {
                    // Sealed box on a finite baffle: frequency-dependent transition from
                    // monopole (4pi) to hemispherical (2pi) radiation
                    // alpha(f) = (f/f_baffle)^2 / (1 + (f/f_baffle)^2)
                    //   alpha ~ 0 at low freq -> omnidirectional (R = 1 everywhere)
                    //   alpha ~ 1 at high freq -> cardioid-like front weighting
                    // On-axis (ct=1): R = (1-a) + a*1 = 1 (always unity on-axis)
                    double f_ratio = freq / f_baffle;
                    double alpha = f_ratio * f_ratio / (1.0 + f_ratio * f_ratio);
                    R = (1.0 - alpha) + alpha * 0.5 * (1.0 + ct);
                }
                else if (radiation_type == "dipole")
                {
                    // Open baffle / planar speaker: figure-8 pattern
                    // Rear hemisphere is 180 degrees out of phase
                    R = std::abs(ct);
                    if (ct < 0.0)
                        rear_phase_flip = true;
                }
                else if (radiation_type == "cardioid")
                {
                    // Cardioid subwoofer design: monopole + dipole combination
                    // R = 0.5 * (1 + cos(theta_off))
                    // Null at rear (theta_off = 180 deg)
                    R = 0.5 * (1.0 + ct);
                }
                // else "monopole": R stays 1.0

                // ===== Step 3: Combine amplitude =====
                double amplitude = std::abs(D * R * freq_response);

                if (rear_phase_flip)
                {
                    // Dipole rear: 180-degree phase inversion
                    pat_re[idx] = dtype(-amplitude);
                }
                else
                {
                    pat_re[idx] = dtype(amplitude);
                }
            }
        }

        // Set the data pointers for quick validation
        ant.check_ptr[0] = ant.e_theta_re.memptr();
        ant.check_ptr[1] = ant.e_theta_im.memptr();
        ant.check_ptr[2] = ant.e_phi_re.memptr();
        ant.check_ptr[3] = ant.e_phi_im.memptr();
        ant.check_ptr[4] = ant.azimuth_grid.memptr();
        ant.check_ptr[5] = ant.elevation_grid.memptr();
        ant.check_ptr[6] = ant.element_pos.memptr();
        ant.check_ptr[7] = ant.coupling_re.memptr();
        ant.check_ptr[8] = ant.coupling_im.memptr();
    }

    return output;
}

template std::vector<quadriga_lib::arrayant<float>> quadriga_lib::generate_speaker(
    std::string, float, float, float, float, float, float, std::string, float, float, float, float, float, arma::Col<float>, float);
    
template std::vector<quadriga_lib::arrayant<double>> quadriga_lib::generate_speaker(
    std::string, double, double, double, double, double, double, std::string, double, double, double, double, double, arma::Col<double>, double);