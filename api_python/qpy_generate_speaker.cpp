// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

- Returns a multi-frequency arrayant dict; pattern fields are 4D arrays where the 4th dim is frequency; for a
  single frequency the dict is a single-frequency arrayant dict (3D pattern fields, scalar `center_freq`)
- Directivity is stored as real-valued data in `e_theta_re`; dipole rear hemisphere is encoded with negative sign for 180° phase inversion
- Multi-driver systems (e.g. two-way) are built by calling this per driver and combining via [[concat]];
  crossover behavior emerges from overlapping bandpass responses
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`,
  where `n = slope_dB_per_octave / 6`; -3 dB at the cutoff frequencies
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity_dB - 85) / 20)`
- If `center_freq` is None or empty, third-octave band centers are auto-generated from one band below
  `lower_cutoff` to one band above `upper_cutoff`, clipped to 20-20000 Hz
- Speed of sound assumed to be 343 m/s
- Driver models (`driver_type`):
  - `piston` - circular piston in baffle, `D(theta) = 2·J1(ka·sin theta)/(ka·sin theta)`, rotationally symmetric, narrows with increasing `ka`
  - `horn` - separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni below `horn_control_freq`
  - `omni` - frequency-independent omnidirectional pattern
- Enclosure models (`radiation_type`):
  - `monopole` - no modification
  - `hemisphere` - sealed box with baffle-step transition, `f_baffle = c/(pi·sqrt(W·H))`
  - `dipole` - figure-8, `R = abs(cos(theta_off))` with sign inversion in rear hemisphere
  - `cardioid` - `R = 0.5·(1+cos(theta_off))`
- For `horn`, `hor_coverage_deg`, `ver_coverage_deg` and `horn_control_freq` set to NaN (the default) or <= 0 select 90°, 60°
  and `f_ctrl = c/(2pi·radius)`

## Usage:
```
speaker = quadriga_lib.arrayant.generate_speaker( driver_type, *, radius, lower_cutoff, upper_cutoff, \
    lower_rolloff_slope_dB, upper_rolloff_slope_dB, sensitivity_dB, radiation_type, hor_coverage_deg, \
    ver_coverage_deg, horn_control_freq, baffle_width, baffle_height, center_freq, angular_resolution_deg )
```

## Inputs:
- **`driver_type`** — Driver directivity model: `piston`, `horn`, or `omni`; default: `piston`
- **`radius`** — Effective radiating radius in m, > 0; cone/dome radius for piston, mouth radius for horn;
  default: 0.05
- **`lower_cutoff`** — Lower -3 dB bandpass frequency in Hz, > 0; default: 80
- **`upper_cutoff`** — Upper -3 dB bandpass frequency in Hz, larger than `lower_cutoff`;
  default: 12000
- **`lower_rolloff_slope_dB`** — Low-frequency rolloff in dB/octave, > 0 (12 dB/oct = 2nd-order Butterworth);
  default: 12
- **`upper_rolloff_slope_dB`** — High-frequency rolloff in dB/octave, > 0; default: 12
- **`sensitivity_dB`** — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** — Enclosure radiation model: `monopole`, `hemisphere`, `dipole`, or `cardioid`;
  default: `hemisphere`
- **`hor_coverage_deg`** — Horn horizontal coverage angle in degrees; NaN or <= 0: default 90; horn only;
  default: NaN
- **`ver_coverage_deg`** — Horn vertical coverage angle in degrees; NaN or <= 0: default 60; horn only;
  default: NaN
- **`horn_control_freq`** — Horn pattern control frequency in Hz; NaN or <= 0: default `c/(2pi·radius)`;
  horn only; default: NaN
- **`baffle_width`** — Baffle width in m, > 0; used by `hemisphere` model; default: 0.15
- **`baffle_height`** — Baffle height in m, > 0; used by `hemisphere` model; default: 0.25
- **`center_freq`** — Frequency sample points in Hz; auto-generated third-octave bands if None;
  `(n_freq,)` or None; default: None
- **`angular_resolution_deg`** — Azimuth and elevation grid resolution in degrees, 0.1 to 90; default: 5

## Outputs:
- **`speaker`** — Multi-frequency arrayant dict with the fields below; for `n_freq = 1` the pattern fields
  are 3D `(n_elevation, n_azimuth, n_elements)` and `center_freq` is a float:
  - `e_theta_re` — e-theta field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_theta_im` — e-theta field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_re` — e-phi field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_im` — e-phi field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `azimuth_grid` — Azimuth angles in rad, -π to π, strictly increasing; `(n_azimuth,)`
  - `elevation_grid` — Elevation angles in rad, -π/2 to π/2, strictly increasing; `(n_elevation,)`
  - `element_pos` — Element (x,y,z) positions in meters; `(3, n_elements)`
  - `coupling_re` — Coupling matrix, real part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `coupling_im` — Coupling matrix, imaginary part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `center_freq` — Frequency samples in Hz; `(n_freq,)`
  - `name` — Name of the array antenna object; str
MD!*/

py::dict generate_speaker(const std::string &driver_type,
                          double radius,
                          double lower_cutoff,
                          double upper_cutoff,
                          double lower_rolloff_slope_dB,
                          double upper_rolloff_slope_dB,
                          double sensitivity_dB,
                          const std::string &radiation_type,
                          double hor_coverage_deg,
                          double ver_coverage_deg,
                          double horn_control_freq,
                          double baffle_width,
                          double baffle_height,
                          py::handle center_freq,
                          double angular_resolution_deg)
{
    // Empty arma::vec when center_freq is None or empty
    const auto freq_vec = qd_python_numpy2arma_Col<double>(center_freq, true);

    auto speaker = quadriga_lib::generate_speaker<double>(
        driver_type, radius, lower_cutoff, upper_cutoff,
        lower_rolloff_slope_dB, upper_rolloff_slope_dB, sensitivity_dB,
        radiation_type, hor_coverage_deg, ver_coverage_deg, horn_control_freq,
        baffle_width, baffle_height, freq_vec, angular_resolution_deg);

    return qd_python_arrayant2dict_multi(speaker);
}
