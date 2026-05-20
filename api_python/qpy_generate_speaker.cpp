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

- Returns a multi-frequency arrayant dict; pattern fields are 4D arrays where the 4th dim is frequency
- Directivity is stored as real-valued data in `e_theta_re`; dipole rear hemisphere is encoded with negative sign for 180° phase inversion
- Multi-driver systems (e.g. two-way) are built by calling this per driver and combining via arrayant.[[concat]];
  crossover behavior emerges from overlapping bandpass responses
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`,
  where `n = slope_dB_per_octave / 6`; -3 dB at the cutoff frequencies
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity - 85) / 20)`
- If `frequencies` is None or empty, third-octave band centers are auto-generated from one band below
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
- For `horn`, if `horn_control_freq = 0`, it is auto-derived as `f_ctrl = c/(2pi·radius)`

## Usage:
```
# Default piston driver (4-inch, 80 Hz – 12 kHz)
speaker = quadriga_lib.arrayant.generate_speaker()

# Horn tweeter with custom coverage
speaker = quadriga_lib.arrayant.generate_speaker(driver_type='horn', radius=0.025,
    lower_cutoff=1500.0, upper_cutoff=20000.0, hor_coverage=90.0, ver_coverage=60.0)

# Omnidirectional subwoofer with steep rolloff
speaker = quadriga_lib.arrayant.generate_speaker(driver_type='omni', radius=0.165,
    lower_cutoff=30.0, upper_cutoff=300.0, lower_rolloff_slope=24.0, upper_rolloff_slope=24.0,
    sensitivity=90.0, radiation_type='monopole')

# Piston driver at specific frequencies
speaker = quadriga_lib.arrayant.generate_speaker(
    frequencies=np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0]), angular_resolution=5.0)
```

## Inputs:
- **`driver_type`** — Driver directivity model: `piston`, `horn`, or `omni`; default: `piston`
- **`radius`** — Effective radiating radius in m; cone/dome radius for piston, mouth radius for horn;
  default: 0.05
- **`lower_cutoff`** — Lower -3 dB bandpass frequency in Hz; default: 80
- **`upper_cutoff`** — Upper -3 dB bandpass frequency in Hz; default: 12000
- **`lower_rolloff_slope`** — Low-frequency rolloff in dB/octave (12 dB/oct = 2nd-order Butterworth);
  default: 12
- **`upper_rolloff_slope`** — High-frequency rolloff in dB/octave; default: 12
- **`sensitivity`** — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** — Enclosure radiation model: `monopole`, `hemisphere`, `dipole`, or `cardioid`;
  default: `hemisphere`
- **`hor_coverage`** — Horn horizontal coverage angle in degrees; 0 auto-defaults to 90; horn only;
  default: 0
- **`ver_coverage`** — Horn vertical coverage angle in degrees; 0 auto-defaults to 60; horn only;
  default: 0
- **`horn_control_freq`** — Horn pattern control frequency in Hz; 0 auto-derives from `radius`;
  default: 0
- **`baffle_width`** — Baffle width in m; used by `hemisphere` model; default: 0.15
- **`baffle_height`** — Baffle height in m; used by `hemisphere` model; default: 0.25
- **`frequencies`** — Frequency sample points in Hz; auto-generated third-octave bands if None;
  `(n_freq,)` or None; default: None
- **`angular_resolution`** — Azimuth and elevation grid resolution in degrees; default: 5

## Outputs:
- **`speaker`** — Multi-frequency arrayant dict with fields:
  - `e_theta_re` — e-theta field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_theta_im` — e-theta field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_re` — e-phi field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_im` — e-phi field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `azimuth_grid` — Azimuth angles in rad, -π to π, sorted; `(n_azimuth,)`
  - `elevation_grid` — Elevation angles in rad, -π/2 to π/2, sorted; `(n_elevation,)`
  - `element_pos` — Element (x,y,z) positions in meters; `(3, n_elements)`
  - `coupling_re` — Coupling matrix, real part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `coupling_im` — Coupling matrix, imaginary part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `center_freq` — Frequency samples in Hz; `(n_freq,)`
  - `name` — Name of the array antenna object; str
MD!*/

py::dict arrayant_generate_speaker(const std::string &driver_type,
                                   double radius,
                                   double lower_cutoff,
                                   double upper_cutoff,
                                   double lower_rolloff_slope,
                                   double upper_rolloff_slope,
                                   double sensitivity,
                                   const std::string &radiation_type,
                                   double hor_coverage,
                                   double ver_coverage,
                                   double horn_control_freq,
                                   double baffle_width,
                                   double baffle_height,
                                   py::handle frequencies,
                                   double angular_resolution)
{
    // Empty arma::vec when frequencies is None or empty
    const auto freq_vec = qd_python_numpy2arma_Col<double>(frequencies, true);

    auto speaker = quadriga_lib::generate_speaker<double>(
        driver_type, radius, lower_cutoff, upper_cutoff,
        lower_rolloff_slope, upper_rolloff_slope, sensitivity,
        radiation_type, hor_coverage, ver_coverage, horn_control_freq,
        baffle_width, baffle_height, freq_vec, angular_resolution);

    return qd_python_arrayant2dict_multi(speaker);
}

// pybind11 declaration:
// m.def("generate_speaker", &arrayant_generate_speaker,
//       py::arg("driver_type") = "piston",
//       py::arg("radius") = 0.05,
//       py::arg("lower_cutoff") = 80.0,
//       py::arg("upper_cutoff") = 12000.0,
//       py::arg("lower_rolloff_slope") = 12.0,
//       py::arg("upper_rolloff_slope") = 12.0,
//       py::arg("sensitivity") = 85.0,
//       py::arg("radiation_type") = "hemisphere",
//       py::arg("hor_coverage") = 0.0,
//       py::arg("ver_coverage") = 0.0,
//       py::arg("horn_control_freq") = 0.0,
//       py::arg("baffle_width") = 0.15,
//       py::arg("baffle_height") = 0.25,
//       py::arg("frequencies") = py::none(),
//       py::arg("angular_resolution") = 5.0);