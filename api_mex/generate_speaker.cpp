// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# GENERATE_SPEAKER
Generate a parametric frequency-dependent loudspeaker directivity model

- Returns one arrayant per frequency sample; each has a single element with the real-valued
  directivity pattern in `e_theta_re` and `center_frequency` set to the corresponding frequency.
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining
  results via `append` and `element_pos`; crossover behavior emerges from overlapping bandpass
  responses.
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`,
  where `n = slope_dB_per_octave / 6`; -3 dB at the cutoff frequencies.
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity - 85) / 20)`.
- If `frequencies` is empty, third-octave band center frequencies are auto-generated from one band
  below `lower_cutoff` to one band above `upper_cutoff`, clipped to 20-20000 Hz.
- Speed of sound assumed to be 343 m/s.
- Driver models (`driver_type`):
  - `piston` - circular piston in baffle, `D(theta) = 2·J1(ka·sin theta)/(ka·sin theta)`,
    rotationally symmetric, narrows with increasing `ka`
  - `horn` - separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni
    below `horn_control_freq`
  - `omni` - frequency-independent omnidirectional pattern.
- Enclosure models (`radiation_type`):
  - `monopole` - no modification
  - `hemisphere` - sealed box with baffle-step transition, `f_baffle = c/(pi·sqrt(W·H))`
  - `dipole` - figure-8, `R = abs(cos(theta_off))` with sign inversion in rear hemisphere
  - `cardioid` - `R = 0.5·(1+cos(theta_off))`
- For `horn`, if `horn_control_freq = 0`, it is auto-derived as `f_ctrl = c/(2pi·radius)`.

## Usage:
```
arrayant = quadriga_lib.generate_speaker( driver_type, radius, lower_cutoff, upper_cutoff, ...
    lower_rolloff_slope, upper_rolloff_slope, sensitivity, radiation_type, hor_coverage, ...
    ver_coverage, horn_control_freq, baffle_width, baffle_height, frequencies, ...
    angular_resolution );
```

## Inputs:
- **`driver_type`** - Driver directivity model: `piston`, `horn`, or `omni`; default: `piston`
- **`radius`** - Effective radiating radius in m; cone/dome radius for piston, mouth radius for horn; default: 0.05
- **`lower_cutoff`** - Lower -3 dB bandpass frequency in Hz; default: 80
- **`upper_cutoff`** - Upper -3 dB bandpass frequency in Hz; default: 12000
- **`lower_rolloff_slope`** - Low-frequency rolloff in dB/octave (12 dB/oct = 2nd-order Butterworth); default: 12
- **`upper_rolloff_slope`** - High-frequency rolloff in dB/octave; default: 12
- **`sensitivity`** - On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** - Enclosure radiation model: `monopole`, `hemisphere`, `dipole`, or `cardioid`; default: `hemisphere`
- **`hor_coverage`** - Horn horizontal coverage angle in degrees; `0` defaults to 90;  default: 90
- **`ver_coverage`** - Horn vertical coverage angle in degrees; `0` defaults to 60;  default: 60
- **`horn_control_freq`** - Horn pattern control frequency; `0` auto-derives from `radius`; default: 0
- **`baffle_width`** - Baffle width; used by `hemisphere` model; default: 0.15
- **`baffle_height`** - Baffle height; used by `hemisphere` model; default: 0.25
- **`frequencies`** - Frequency sample points; auto-generated third-octave bands if empty; `[n_freq]`
- **`angular_resolution`** - Azimuth and elevation sampling grid resolution in degrees; default: 5

## Outputs:
- **`arrayant`** - Struct array with one arrayant per frequency sample; directivity is stored in
  `e_theta_re`; dipole rear hemisphere encoded with negative sign for 180 degree phase inversion; 
  see [[arrayant_generate]] for struct fields; `[n_freq]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const std::string driver_type = (nrhs < 1) ? "piston" : qd_mex_get_string(prhs[0], "piston");
    const double radius = (nrhs < 2) ? 0.05 : qd_mex_get_scalar<double>(prhs[1], "radius", 0.05);
    const double lower_cutoff = (nrhs < 3) ? 80.0 : qd_mex_get_scalar<double>(prhs[2], "lower_cutoff", 80.0);
    const double upper_cutoff = (nrhs < 4) ? 12000.0 : qd_mex_get_scalar<double>(prhs[3], "upper_cutoff", 12000.0);
    const double lower_rolloff_slope = (nrhs < 5) ? 12.0 : qd_mex_get_scalar<double>(prhs[4], "lower_rolloff_slope", 12.0);
    const double upper_rolloff_slope = (nrhs < 6) ? 12.0 : qd_mex_get_scalar<double>(prhs[5], "upper_rolloff_slope", 12.0);
    const double sensitivity = (nrhs < 7) ? 85.0 : qd_mex_get_scalar<double>(prhs[6], "sensitivity", 85.0);
    const std::string radiation_type = (nrhs < 8) ? "hemisphere" : qd_mex_get_string(prhs[7], "hemisphere");
    const double hor_coverage = (nrhs < 9) ? 0.0 : qd_mex_get_scalar<double>(prhs[8], "hor_coverage", 0.0);
    const double ver_coverage = (nrhs < 10) ? 0.0 : qd_mex_get_scalar<double>(prhs[9], "ver_coverage", 0.0);
    const double horn_control_freq = (nrhs < 11) ? 0.0 : qd_mex_get_scalar<double>(prhs[10], "horn_control_freq", 0.0);
    const double baffle_width = (nrhs < 12) ? 0.15 : qd_mex_get_scalar<double>(prhs[11], "baffle_width", 0.15);
    const double baffle_height = (nrhs < 13) ? 0.25 : qd_mex_get_scalar<double>(prhs[12], "baffle_height", 0.25);
    const arma::vec frequencies = (nrhs < 14) ? arma::vec() : qd_mex_get_Col<double>(prhs[13]);
    const double angular_resolution = (nrhs < 15) ? 5.0 : qd_mex_get_scalar<double>(prhs[14], "angular_resolution", 5.0);

    // Call library function
    std::vector<quadriga_lib::arrayant<double>> ant;
    CALL_QD(ant = quadriga_lib::generate_speaker<double>(driver_type, radius, lower_cutoff,
                                                         upper_cutoff, lower_rolloff_slope,
                                                         upper_rolloff_slope, sensitivity,
                                                         radiation_type, hor_coverage, ver_coverage,
                                                         horn_control_freq, baffle_width,
                                                         baffle_height, frequencies,
                                                         angular_resolution));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_arrayant2struct_multi(ant);
}