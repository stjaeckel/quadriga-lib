// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# generate_speaker
Generates a parametric loudspeaker directivity model

## Description:
This function generates frequency-dependent loudspeaker radiation patterns by combining a driver
directivity model with an enclosure radiation modifier and a Butterworth-style bandpass frequency
response. Returns a multi-frequency arrayant dictionary where the pattern fields are 4D arrays
(the 4th dimension is frequency) and `center_freq` is a 1D array of frequency samples in Hz.

Multi-driver speakers (e.g. two-way systems) are modelled by generating each driver separately and
combining them via `arrayant_concat_multi`. Crossover behavior emerges naturally from overlapping
bandpass responses.<br><br>

Three driver types are supported:
- **`piston`**: Circular piston in a baffle using the classical Bessel J1 formula. Transitions from
  omnidirectional at low ka to progressively narrower beaming at high ka.
- **`horn`**: Separable cosine-power directivity with frequency-dependent pattern control. Below the
  horn control frequency, the pattern blends toward omnidirectional.
- **`omni`**: Frequency-independent omnidirectional pattern (suitable for subwoofers).<br><br>

Four enclosure radiation types modify the base driver pattern:
- **`monopole`**: No modification (4π radiation). Appropriate for subwoofers in free space.
- **`hemisphere`**: Sealed box on a finite baffle with frequency-dependent baffle step transition.
- **`dipole`**: Open baffle / planar speaker with figure-8 pattern.
- **`cardioid`**: Monopole + dipole combination with null at the rear.<br><br>

The frequency response follows a Butterworth-style bandpass filter. If no frequency vector is
provided, third-octave band center frequencies are auto-generated covering the range from one band
below the lower cutoff to one band above the upper cutoff, clipped to 20–20000 Hz.

## Usage:
```
from quadriga_lib import arrayant

# Default piston driver (4-inch, 80 Hz – 12 kHz)
speaker = arrayant.generate_speaker()

# Horn tweeter with custom coverage
speaker = arrayant.generate_speaker(
    driver_type='horn',
    radius=0.025,
    lower_cutoff=1500.0,
    upper_cutoff=20000.0,
    radiation_type='hemisphere',
    hor_coverage=90.0,
    ver_coverage=60.0
)

# Omnidirectional subwoofer with steep rolloff
speaker = arrayant.generate_speaker(
    driver_type='omni',
    radius=0.165,
    lower_cutoff=30.0,
    upper_cutoff=300.0,
    lower_rolloff_slope=24.0,
    upper_rolloff_slope=24.0,
    sensitivity=90.0,
    radiation_type='monopole'
)

# Piston driver at specific frequencies
import numpy as np
speaker = arrayant.generate_speaker(
    frequencies=np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0]),
    angular_resolution=5.0
)
```

## Input Arguments:
- **`driver_type`**<br>
  Driver directivity model, string. Supported values: `"piston"` (cone/dome via Bessel function),
  `"horn"` (cosine-power with frequency-dependent pattern control), `"omni"` (omnidirectional
  subwoofer). Default: `"piston"`

- **`radius`**<br>
  Effective radiating radius in meters, scalar. For `"piston"`: cone or dome radius. For `"horn"`:
  mouth radius (pattern control frequency auto-derived if not specified). Default: `0.05` (~4-inch driver)

- **`lower_cutoff`**<br>
  Lower −3 dB frequency of the bandpass response in Hz, scalar. Default: `80.0`

- **`upper_cutoff`**<br>
  Upper −3 dB frequency of the bandpass response in Hz, scalar. Default: `12000.0`

- **`lower_rolloff_slope`**<br>
  Low-frequency rolloff slope in dB per octave, scalar. Butterworth order = slope / 6
  (e.g. 12 dB/oct = 2nd order). Default: `12.0`

- **`upper_rolloff_slope`**<br>
  High-frequency rolloff slope in dB per octave, scalar. Default: `12.0`

- **`sensitivity`**<br>
  On-axis sensitivity in dB SPL at 1W/1m, scalar. Scales amplitude linearly relative to 85 dB
  reference. Default: `85.0`

- **`radiation_type`**<br>
  Enclosure radiation modifier, string. Supported values: `"monopole"`, `"hemisphere"`,
  `"dipole"`, `"cardioid"`. Default: `"hemisphere"`

- **`hor_coverage`**<br>
  Horizontal coverage angle in degrees, scalar. Horn driver only. 0 = auto (90°). Default: `0.0`

- **`ver_coverage`**<br>
  Vertical coverage angle in degrees, scalar. Horn driver only. 0 = auto (60°). Default: `0.0`

- **`horn_control_freq`**<br>
  Horn pattern control frequency in Hz, scalar. 0 = auto-derived from mouth radius. Default: `0.0`

- **`baffle_width`**<br>
  Enclosure baffle width in meters, scalar. Piston driver only (used for baffle step model).
  Default: `0.15`

- **`baffle_height`**<br>
  Enclosure baffle height in meters, scalar. Piston driver only. Default: `0.25`

- **`frequencies`**<br>
  Frequency sample points in Hz, 1D numpy array. If empty, third-octave bands are auto-generated.
  Default: empty (auto)

- **`angular_resolution`**<br>
  Angular grid resolution in degrees, scalar. Used to generate azimuth and elevation grids.
  Default: `5.0`

## Output Argument:
- **`speaker`**<br>
  Dictionary containing the multi-frequency arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part           | Shape: `(n_elevation, n_azimuth, n_elements, n_freq)`
  `e_theta_im`     | e-theta field component, imaginary part      | Shape: `(n_elevation, n_azimuth, n_elements, n_freq)`
  `e_phi_re`       | e-phi field component, real part             | Shape: `(n_elevation, n_azimuth, n_elements, n_freq)`
  `e_phi_im`       | e-phi field component, imaginary part        | Shape: `(n_elevation, n_azimuth, n_elements, n_freq)`
  `azimuth_grid`   | Azimuth angles in [rad], −π to π, sorted     | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], −π/2 to π/2       | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions            | Shape: `(3, n_elements)`
  `coupling_re`    | Coupling matrix, real part                   | Shape: `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)`
  `coupling_im`    | Coupling matrix, imaginary part              | Shape: `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)`
  `center_freq`    | Frequency samples in Hz                      | Shape: `(n_freq)` - 1D array
  `name`           | Name of the array antenna object             | String
MD!*/

py::dict arrayant_generate_speaker(const std::string driver_type,          // Driver type: "piston", "horn", "omni"
                                   double radius,                          // Effective radiating radius in [m]
                                   double lower_cutoff,                    // Lower -3 dB frequency in [Hz]
                                   double upper_cutoff,                    // Upper -3 dB frequency in [Hz]
                                   double lower_rolloff_slope,             // Low-frequency rolloff in [dB/octave]
                                   double upper_rolloff_slope,             // High-frequency rolloff in [dB/octave]
                                   double sensitivity,                     // On-axis sensitivity in [dB SPL] at 1W/1m
                                   const std::string radiation_type,       // Radiation type: "monopole", "hemisphere", "dipole", "cardioid"
                                   double hor_coverage,                    // Horizontal coverage angle in [deg], horn only
                                   double ver_coverage,                    // Vertical coverage angle in [deg], horn only
                                   double horn_control_freq,               // Horn pattern control frequency in [Hz]
                                   double baffle_width,                    // Enclosure baffle width in [m], piston only
                                   double baffle_height,                   // Enclosure baffle height in [m], piston only
                                   const py::array_t<double> &frequencies, // Frequency sample points in [Hz]
                                   double angular_resolution)              // Angular grid resolution in [deg]
{
    // Convert frequencies: empty numpy array → empty arma::vec, otherwise copy
    arma::vec freq_vec;
    if (frequencies.size() > 0)
        freq_vec = qd_python_numpy2arma_Col<double>(frequencies, true);

    auto speaker = quadriga_lib::generate_speaker<double>(
        driver_type, radius, lower_cutoff, upper_cutoff,
        lower_rolloff_slope, upper_rolloff_slope, sensitivity,
        radiation_type, hor_coverage, ver_coverage, horn_control_freq,
        baffle_width, baffle_height, freq_vec, angular_resolution);

    return qd_python_arrayant2dict_multi(speaker);
}
