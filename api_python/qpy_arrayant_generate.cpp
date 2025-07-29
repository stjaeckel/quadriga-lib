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
# GENERATE
Generates predefined array antenna models

## Description:
This functions can be used to generate a variety of pre-defined array antenna models, including 3GPP
array antennas used for 5G-NR simulations. The first argument is the array type. The following input
arguments are then specific to this type.

## Usage:

```
from quadriga_lib import arrayant

# Isotropic radiator, vertical polarization
arrayant = arrayant.generate('omni', res)

# Short dipole radiating with vertical polarization
arrayant = arrayant.generate('dipole', res)

# Half-wave dipole radiating with vertical polarization
arrayant = arrayant.generate('half-wave-dipole', res)

# Cross-polarized isotropic radiator
arrayant = arrayant.generate('xpol', res)

# An antenna with a custom 3dB beam with (in degree)
arrayant = arrayant.generate('custom', res, az_3dB, el_3db, rear_gain_lin)

# 3GPP-NR antenna model (example for 2x2, V-polarized, 0.7λ spacing)
arrayant = arrayant.generate('3gpp', M=2, N=2, freq=3.7e9, pol=1, spacing=0.7)

# Planar multi-element antenna with support for multiple beam directions
arrayant = arrayant.generate('multibeam', M=6, N=6, freq=3.7e9, pol=1, spacing=0.7, az=[-30.0, 30.0], el=[0.0, 0.0])
```

## Input Arguments:
- **`type`**<br>
  Antenna model type, string

- **`res`**<br>
  Pattern resolution in [deg], scalar, default = 1 deg

- **`freq`**<br>
  The center frequency in [Hz], scalar, default = 299792458 Hz

## Input arguments for type `custom`, `3gpp` and `multibeam`:
- **`az_3dB`**<br>
  3dB beam width in azimuth direction in [deg], scalar,
  default for `custom` = 90 deg, default for `3gpp` = 67 deg, `multibeam` = 120 deg

- **`el_3db`**<br>
  3dB beam width in elevation direction in [deg], scalar,
  default for `custom` = 90 deg, default for `3gpp` = 67 deg, `multibeam` = 120 deg

- **`rear_gain_lin`**<br>
  Isotropic gain (linear scale) at the back of the antenna, scalar, default = 0.0

## Input arguments for type `3gpp` and `multibeam`:
- **`M`**<br>
  Number of vertically stacked elements for `3gpp` and `multibeam`, scalar, default = 1

- **`N`**<br>
  Number of horizontally stacked elements for `3gpp` and `multibeam`, scalar, default = 1

- **`pol`**<br>
  Polarization indicator to be applied for each of the M elements:<br>
  `pol = 1` | vertical polarization (default value), `3gpp` and `multibeam`
  `pol = 2` | H/V polarized elements, results in 2NM elements, `3gpp` and `multibeam`
  `pol = 3` | +/-45° polarized elements, results in 2NM elements, `3gpp` and `multibeam`
  `pol = 4` | vertical polarization, combines elements in vertical direction, results in N elements, `3gpp` only
  `pol = 5` | H/V polarization, combines elements in vertical direction, results in 2N elements, `3gpp` only
  `pol = 6` | +/-45° polarization, combines elements in vertical direction, results in 2N elements, `3gpp` only
  Polarization indicator is ignored when a custom pattern is provided.

- **`tilt`**<br>
  The electric downtilt angle in [deg], Only relevant for `pol = 4/5/6`, `3gpp` only, scalar, default = 0

- **`spacing`**<br>
  Element spacing in [λ] for `3gpp` and `multibeam`, scalar, default = 0.5

- **`Mg`**<br>
  Number of nested panels in a column, `3gpp` only, scalar, default = 1

- **`Ng`**<br>
  Number of nested panels in a row, `3gpp` only, scalar, default = 1

- **`dgv`**<br>
  Panel spacing in vertical direction in [λ], `3gpp` only, scalar, default = 0.5

- **`dgh`**<br>
  Panel spacing in horizontal direction in [λ], `3gpp` only, scalar, default = 0.5

- **`beam_az`**<br>
  Azimuth beam angles (degrees), `multibeam` only, Vector of length `n_beams`. Default: `[0.0]`

- **`beam_el`**<br>
  Elevation beam angles (degrees), `multibeam` only, Vector of length `n_beams`. Default: `[0.0]`

- **`beam_weight`**<br>
  Scaling factors for each beam, `multibeam` only, The vector must have the same length as `beam_az` and `beam_el`.
  Values are normalized so that their sum equals 1. Can be used to prioritize beams.
  Default: `{1.0}`

- **`separate_beams`**<br>
  If set to true, create a separate beam for each angle pair (ignores weights), `multibeam` only

- **`apply_weights`**<br>
  Switch to apply the beam-forming weights

- **`pattern`** (optional)<br>
  Dictionary containing a custom pattern (default = empty) with at least the following keys:
  `e_theta_re`     | Real part of e-theta field component             | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_theta_im`     | Imaginary part of e-theta field component        | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_phi_re`       | Real part of e-phi field component               | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_phi_im`       | Imaginary part of e-phi field component          | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation]`

## Output Arguments:
- **`arrayant`**<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted             | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions                     | Size: `[3, n_elements]`
  `coupling_re`    | Coupling matrix, real part                            | Size: `[n_elements, n_ports]`
  `coupling_im`    | Coupling matrix, imaginary part                       | Size: `[n_elements, n_ports]`
  `center_freq`    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
  `name`           | Name of the array antenna object                      | String
MD!*/

py::dict arrayant_generate(const std::string type,                 // Array type
                           double res,                             // Pattern resolution in [deg]
                           double freq,                            // The center frequency in [Hz]
                           double az_3dB,                          // 3dB beam width in azimuth direction in [deg]
                           double el_3dB,                          // 3dB beam width in elevation direction in [deg]
                           double rear_gain_lin,                   // Isotropic gain (linear scale) at the back of the antenna
                           arma::uword M,                          // Number of vertically stacked elements
                           arma::uword N,                          // Number of horizontally stacked elements
                           unsigned pol,                           // Polarization indicator to be applied for each of the M elements
                           double tilt,                            // The electric downtilt angle in [deg]
                           double spacing,                         // Element spacing in [λ]
                           arma::uword Mg,                         // Number of nested panels in a column
                           arma::uword Ng,                         // Number of nested panels in a row
                           double dgv,                             // Panel spacing in vertical horizontal in [λ]
                           double dgh,                             // Panel spacing in horizontal direction in [λ]
                           const py::array_t<double> &beam_az,     // Azimuth beam angles in degree
                           const py::array_t<double> &beam_el,     // Elevation beam angles in degree
                           const py::array_t<double> &beam_weight, // Scaling factor for the beams
                           bool separate_beams,                    // If true, create a separate beam for each angle pair (ignores weights)
                           bool apply_weights,                     // Switch to apply the beamforming weights
                           const py::dict &pattern)                // 3GPP custom pattern
{
    quadriga_lib::arrayant<double> arrayant;

    if (type == "omni")
        arrayant = quadriga_lib::generate_arrayant_omni<double>(res);
    else if (type == "dipole" || type == "short-dipole")
        arrayant = quadriga_lib::generate_arrayant_dipole<double>(res);
    else if (type == "half-wave-dipole")
        arrayant = quadriga_lib::generate_arrayant_half_wave_dipole<double>(res);
    else if (type == "xpol")
        arrayant = quadriga_lib::generate_arrayant_xpol<double>(res);
    else if (type == "custom")
        arrayant = quadriga_lib::generate_arrayant_custom<double>(az_3dB, el_3dB, rear_gain_lin, res);
    else if (type == "3GPP" || type == "3gpp")
    {
        if (pattern.size() != 0) // Use custom pattern
        {
            quadriga_lib::arrayant<double> custom_array;
            custom_array.e_theta_re = qd_python_numpy2arma_Cube<double>(pattern["e_theta_re"], true);
            custom_array.e_theta_im = qd_python_numpy2arma_Cube<double>(pattern["e_theta_im"], true);
            custom_array.e_phi_re = qd_python_numpy2arma_Cube<double>(pattern["e_phi_re"], true);
            custom_array.e_phi_im = qd_python_numpy2arma_Cube<double>(pattern["e_phi_im"], true);
            custom_array.azimuth_grid = qd_python_numpy2arma_Col<double>(pattern["azimuth_grid"], true);
            custom_array.elevation_grid = qd_python_numpy2arma_Col<double>(pattern["elevation_grid"], true);
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, &custom_array);
        }
        else if (az_3dB > 0.0 && el_3dB > 0.0) // Use custom beam width
        {
            auto custom_array = quadriga_lib::generate_arrayant_custom<double>(az_3dB, el_3dB, rear_gain_lin, res);
            if (pol == 2 || pol == 5)
            {
                custom_array.copy_element(0, 1);
                custom_array.rotate_pattern(90.0, 0.0, 0.0, 2, 1);
            }
            else if (pol == 3 || pol == 6)
            {
                custom_array.copy_element(0, 1);
                custom_array.rotate_pattern(45.0, 0.0, 0.0, 2, 0);
                custom_array.rotate_pattern(-45.0, 0.0, 0.0, 2, 1);
            }
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, &custom_array);
        }
        else // Use 3GPP default pattern
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, nullptr, res);
    }
    else if (type == "multibeam")
    {
        const auto az = qd_python_numpy2arma_Col(beam_az, true);
        const auto el = qd_python_numpy2arma_Col(beam_el, true);
        const auto weight = qd_python_numpy2arma_Col(beam_weight, true);

        arrayant = quadriga_lib::generate_arrayant_multibeam(M, N, az, el, weight, freq, pol, spacing,
                                                             az_3dB, el_3dB, rear_gain_lin, res, 
                                                             separate_beams, apply_weights);
    }
    else
        throw std::invalid_argument("Array type not supported!");

    arrayant.center_frequency = freq;
    return qd_python_arrayant2dict(arrayant);
}
