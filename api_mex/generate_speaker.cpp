// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# GENERATE_SPEAKER
Generate a parametric frequency-dependent loudspeaker directivity model

- Creates one arrayant per frequency with a single element at the origin. Its real-valued pattern in `e_theta_re` is the
  product of the driver directivity, the enclosure radiation and the bandpass frequency response; a negative value is a
  180° phase inversion (piston side lobes, dipole rear)
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining the results with
  [[join_arrays]] and the field `element_pos`; the crossover follows from the overlapping bandpass responses

## Usage:
```
arrayant = quadriga_lib.generate_speaker( driver_type, radius, lower_cutoff, upper_cutoff, ...
    lower_rolloff_slope_dB, upper_rolloff_slope_dB, sensitivity_dB, radiation_type, hor_coverage_deg, ...
    ver_coverage_deg, horn_control_freq, baffle_width, baffle_height, center_freq, ...
    angular_resolution_deg );
```

## Inputs:
- **`driver_type`** — Driver directivity model: `piston` (circular piston in a baffle, narrows with increasing
  frequency), `horn` (set coverage, toward omni at low frequencies) or `omni` (independent of frequency); default:
  `piston`
- **`radius`** — Effective radiating radius in m, > 0; cone or dome radius for `piston`, mouth radius for `horn`;
  default: 0.05
- **`lower_cutoff`** — Lower -3 dB frequency of the bandpass in Hz, > 0; default: 80
- **`upper_cutoff`** — Upper -3 dB frequency of the bandpass in Hz, larger than `lower_cutoff`; default: 12000
- **`lower_rolloff_slope_dB`** — Low-frequency rolloff in dB per octave, > 0; 12 is a second-order Butterworth filter;
  default: 12
- **`upper_rolloff_slope_dB`** — High-frequency rolloff in dB per octave, > 0; default: 12
- **`sensitivity_dB`** — On-axis sensitivity in dB SPL at 1 W and 1 m, finite; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** — Enclosure radiation model: `monopole` (no change), `hemisphere` (sealed box with baffle-step
  transition), `dipole` (figure-8, rear hemisphere inverted) or `cardioid`; default: `hemisphere`
- **`hor_coverage_deg`** — `horn`: horizontal coverage in degrees, below 180, between the -6 dB points at full pattern
  control; NaN or <= 0: 90°; default: NaN
- **`ver_coverage_deg`** — `horn`: vertical coverage in degrees, below 180, as above; NaN or <= 0: 60°; default: NaN
- **`horn_control_freq`** — `horn`: frequency in Hz above which the horn controls its pattern; NaN or <= 0: `c / (2π a)`
  with the speed of sound c and a = `radius`; default: NaN
- **`baffle_width`** — Baffle width in m, > 0; used by `hemisphere`; default: 0.15
- **`baffle_height`** — Baffle height in m, > 0; used by `hemisphere`; default: 0.25
- **`center_freq`** — Frequency samples in Hz, positive and finite; `[n_freq]`; default: third-octave bands
  `1000 · 2^(n/3)` Hz, those enclosing the cutoffs plus one more on each side, limited to 20 to 20000 Hz
- **`angular_resolution_deg`** — Resolution of the azimuth and elevation grid in degrees; 0.1 to 90; a value r gives
  `floor(360 / r)` azimuth and `floor(180 / r)` elevation steps; default: 5

## Output:
- **`arrayant`** — Struct array with one arrayant per frequency, with `center_freq` set to that frequency; fields as in
  [[generate_antenna]]; `[1, n_freq]`

## Notes:
- **Frequency response**: a Butterworth-style bandpass, -3 dB at the cutoffs, scaled by the sensitivity S =
  `sensitivity_dB` relative to 85 dB SPL:

  `H(f) = 10^((S - 85) / 20) / ( √(1 + (f_l / f)^(2 n_l)) · √(1 + (f / f_u)^(2 n_u)) )`

  with f_l = `lower_cutoff`, f_u = `upper_cutoff` and n_l, n_u the rolloff slopes in dB per octave divided by 6
- **Driver directivity** D, with the angle α from the forward axis +x (cos α = cos θ cos φ), the wavenumber k = 2π f /
  c, the speed of sound c = 343 m/s and a = `radius`:
  - `piston`: `D = 2 J1(x) / x` with `x = k a sin α`, rotationally symmetric about +x
  - `horn`: `D = 1 - b + b cos^m_h(φ) cos^m_v(θ)` in the front half (φ between -90° and 90°) and `D = 1 - b` behind it,
    with `b = f² / (f² + f_c²)` and f_c = `horn_control_freq`. The exponents follow from the coverage angle Θ by
    `cos^m(Θ / 2) = 1/2`
  - `omni`: D = 1
- **Enclosure radiation** R:
  - `monopole`: R = 1
  - `hemisphere`: `R = 1 - b + b (1 + cos α) / 2` with `b = f² / (f² + f_b²)` and the baffle-step frequency
    `f_b = c / (π √(W H))`, W = `baffle_width`, H = `baffle_height`; omnidirectional at low and cardioid-like at high
    frequencies
  - `dipole`: `R = cos α`, negative in the rear hemisphere
  - `cardioid`: `R = (1 + cos α) / 2`

## See also:
- [[join_arrays]] (builds multi-driver systems)
- [[interpolate_pattern]] (evaluates the model at any direction and frequency)
- [[calc_beamwidth]] (measures the coverage angle, with `threshold_dB = 6`)
- [[qdant_write]] (stores the model in a file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string driver_type = (nrhs < 1) ? "piston" : qd_mex_get_string(prhs[0], "piston", "driver_type");
    const double radius = (nrhs < 2) ? 0.05 : qd_mex_get_scalar<double>(prhs[1], "radius", 0.05);
    const double lower_cutoff = (nrhs < 3) ? 80.0 : qd_mex_get_scalar<double>(prhs[2], "lower_cutoff", 80.0);
    const double upper_cutoff = (nrhs < 4) ? 12000.0 : qd_mex_get_scalar<double>(prhs[3], "upper_cutoff", 12000.0);
    const double lower_rolloff_slope_dB = (nrhs < 5) ? 12.0 : qd_mex_get_scalar<double>(prhs[4], "lower_rolloff_slope_dB", 12.0);
    const double upper_rolloff_slope_dB = (nrhs < 6) ? 12.0 : qd_mex_get_scalar<double>(prhs[5], "upper_rolloff_slope_dB", 12.0);
    const double sensitivity_dB = (nrhs < 7) ? 85.0 : qd_mex_get_scalar<double>(prhs[6], "sensitivity_dB", 85.0);
    const std::string radiation_type = (nrhs < 8) ? "hemisphere" : qd_mex_get_string(prhs[7], "hemisphere", "radiation_type");
    const double hor_coverage_deg = (nrhs < 9) ? NAN : qd_mex_get_scalar<double>(prhs[8], "hor_coverage_deg", NAN);
    const double ver_coverage_deg = (nrhs < 10) ? NAN : qd_mex_get_scalar<double>(prhs[9], "ver_coverage_deg", NAN);
    const double horn_control_freq = (nrhs < 11) ? NAN : qd_mex_get_scalar<double>(prhs[10], "horn_control_freq", NAN);
    const double baffle_width = (nrhs < 12) ? 0.15 : qd_mex_get_scalar<double>(prhs[11], "baffle_width", 0.15);
    const double baffle_height = (nrhs < 13) ? 0.25 : qd_mex_get_scalar<double>(prhs[12], "baffle_height", 0.25);
    const arma::vec center_freq = (nrhs < 14) ? arma::vec() : qd_mex_get_Col<double>(prhs[13], false, "center_freq");
    const double angular_resolution_deg = (nrhs < 15) ? 5.0 : qd_mex_get_scalar<double>(prhs[14], "angular_resolution_deg", 5.0);

    std::vector<quadriga_lib::arrayant<double>> ant;
    CALL_QD(ant = quadriga_lib::generate_speaker<double>(driver_type, radius, lower_cutoff,
                                                         upper_cutoff, lower_rolloff_slope_dB,
                                                         upper_rolloff_slope_dB, sensitivity_dB,
                                                         radiation_type, hor_coverage_deg, ver_coverage_deg,
                                                         horn_control_freq, baffle_width,
                                                         baffle_height, center_freq,
                                                         angular_resolution_deg));

    plhs[0] = qd_mex_arrayant2struct_multi(ant);
}
