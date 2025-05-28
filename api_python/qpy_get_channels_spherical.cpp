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
# GET_CHANNELS_SPHERICAL
Calculate channel coefficients from path data and antenna patterns

## Description:
In this function, the wireless propagation channel between a transmitter and a receiver is calculated,
based on a single transmit and receive position. Additionally, interaction points with the environment,
which are derived from either Ray Tracing or Geometric Stochastic Models such as QuaDRiGa, are
considered. The calculation is performed under the assumption of spherical wave propagation. For accurate
execution of this process, several pieces of input data are required:<br><br>

- The 3D Cartesian (local) coordinates of both the transmitter and the receiver.
- The specific interaction positions of the propagation paths within the environment.
- The polarization transfer matrix for each propagation path.
- Antenna models for both the transmitter and the receiver.
- The orientations of the antennas.

## Usage:

```
coeff_re, coeff_im, delays = quadriga_lib.get_channels_spherical( ant_tx, ant_rx, 
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, 
    center_freq, use_absolute_delays, add_fake_los_path );

coeff_re, coeff_im, delays, aod, eod, aoa, eoa = quadriga_lib.get_channels_spherical( ant_tx, ant_rx, 
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, 
    center_freq, use_absolute_delays, add_fake_los_path, angles=1 );
```

## Input Arguments:
- **`ant_tx`** (required)<br>
  Dictionary containing the transmit (TX) arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation_tx, n_azimuth_tx, n_elements_tx]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation_tx, n_azimuth_tx, n_elements_tx]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation_tx, n_azimuth_tx, n_elements_tx]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation_tx, n_azimuth_tx, n_elements_tx]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth_tx]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation_tx]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements_tx]`
  `coupling_re`    | Coupling matrix, real part, optional                  | Size: `[n_elements_tx, n_ports_tx]`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Size: `[n_elements_tx, n_ports_tx]`

- **`ant_rx`** (required)<br>
  Dictionary containing the receive (RX) arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation_rx, n_azimuth_rx, n_elements_rx]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation_rx, n_azimuth_rx, n_elements_rx]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation_rx, n_azimuth_rx, n_elements_rx]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation_rx, n_azimuth_rx, n_elements_rx]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth_rx]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation_rx]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements_rx]`
  `coupling_re`    | Coupling matrix, real part, optional                  | Size: `[n_elements_rx, n_ports_rx]`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Size: `[n_elements_rx, n_ports_rx]`

- **`fbs_pos`** (required)<br>
  First interaction point of the rays and the environment, Size: `[ 3, n_path ]`

- **`lbs_pos`** (required)<br>
  Last interaction point of the rays and the environment; For single-bounce models, this must be
  identical to `fbs_pos`, Size: `[ 3, n_path ]`

- **`path_gain`** (required)<br>
  Path gain (linear scale), Size: `[ n_path ]`

- **`path_length`** (required)<br>
  Total path length in meters, Size: `[ n_path ]`

- **`M`** (required)<br>
  Polarization transfer matrix, interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH),
  Size: `[ 8, n_path ]`

- **`tx_pos`** (required)<br>
  Transmitter position in 3D Cartesian coordinates; Size: `[3]`

- **`tx_orientation`** (required)<br>
  3-element vector describing the orientation of the transmit antenna in Euler angles (bank, tilt, heading),
  Size: `[3,1]` or `[1,3]`

- **`rx_pos`** (required)<br>
  Receiver position in 3D Cartesian coordinates, Size: `[3]`

- **`rx_orientation`** (required)]<br>
  3-element vector describing the orientation of the receive antenna in Euler angles,
  Size: `[3]`

- **`center_freq`** (optional)<br>
  Center frequency in [Hz]; optional; If the value is not provided or set to 0, phase calculation
  in coefficients is disabled, i.e. that path length has not influence on the results. This can be
  used to calculate the antenna response for a specific angle and polarization. Scalar value

- **`use_absolute_delays`** (optional)<br>
  If true, the LOS delay is included for all paths; Default is `false`, i.e. delays are normalized
  to the LOS delay.

- **`add_fake_los_path`** (optional)<br>
  If true, adds a zero-power LOS path as the first path in case where no LOS path was present.
  Default: `false`

- **`angles`** (optional flag)<br>
  Switch to return the angles in antenna-local coordinates. Default: 0, false

## Derived inputs:
  `n_azimuth_tx`   | Number of azimuth angles in the TX antenna pattern
  `n_elevation_tx` | Number of elevation angles in the TX antenna pattern
  `n_elements_tx`  | Number of physical antenna elements in the TX array antenna
  `n_ports_tx`     | Number of ports (after coupling) in the TX array antenna
  `n_azimuth_rx`   | Number of azimuth angles in the RX antenna pattern
  `n_elevation_rx` | Number of elevation angles in the RX antenna pattern
  `n_elements_rx`  | Number of physical antenna elements in the RX array antenna
  `n_ports_rx`     | Number of ports (after coupling) in the RX array antenna
  `n_path`         | Number of propagation paths

## Output Arguments:
- **`coeff_re`**<br>
  Channel coefficients, real part, Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`coeff_im`**<br>
  Channel coefficients, imaginary part, Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`delays`**<br>
  Propagation delay in seconds, Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`aod`** (optional)<br>
  Azimuth of Departure angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`,
  Only returned when `angles` flag is set to 1.

- **`eod`** (optional)<br>
  Elevation of Departure angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`,
  Only returned when `angles` flag is set to 1.

- **`aoa`** (optional)<br>
  Azimuth of Arrival angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`,
  Only returned when `angles` flag is set to 1.

- **`eoa`** (optional)<br>
  Elevation of Arrival angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`,
  Only returned when `angles` flag is set to 1.

## Caveat:
- Input data is directly accessed from Python memory, without copying if it is provided in
  **double** precision and is in F-contiguous (column-major) order
- Other formats (e.g. single precision inputs or C-contiguous (row-major) order) will be converted
  to double automatically, causing additional computation steps.
- To improve performance of repeated computations (e.g. in loops), consider preparing the data
  accordingly to avoid unecessary computations.
MD!*/

py::tuple get_channels_spherical(const py::dict &ant_tx,
                                 const py::dict &ant_rx,
                                 const py::array_t<double> &fbs_pos,
                                 const py::array_t<double> &lbs_pos,
                                 const py::array_t<double> &path_gain,
                                 const py::array_t<double> &path_length,
                                 const py::array_t<double> &M,
                                 const py::array_t<double> &tx_pos,
                                 const py::array_t<double> &tx_orientation,
                                 const py::array_t<double> &rx_pos,
                                 const py::array_t<double> &rx_orientation,
                                 const double center_freq,
                                 const bool use_absolute_delays,
                                 const bool add_fake_los_path,
                                 const bool angles)
{
    // Parse input arguments
    const auto ant_tx_a = qd_python_dict2arrayant(ant_tx, true);
    const auto ant_rx_a = qd_python_dict2arrayant(ant_rx, true);
    const auto fbs_pos_a = qd_python_numpy2arma_Mat(fbs_pos, true);
    const auto lbs_pos_a = qd_python_numpy2arma_Mat(lbs_pos, true);
    const auto path_gain_a = qd_python_numpy2arma_Col(path_gain, true);
    const auto path_length_a = qd_python_numpy2arma_Col(path_length, true);
    const auto M_a = qd_python_numpy2arma_Mat(M, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation, true, false, "rx_orientation", 3);

    // Extract scalar values
    double Tx = tx_pos_a.at(0), Ty = tx_pos_a.at(1), Tz = tx_pos_a.at(2);
    double Tb = tx_orientation_a.at(0), Tt = tx_orientation_a.at(1), Th = tx_orientation_a.at(2);
    double Rx = rx_pos_a.at(0), Ry = rx_pos_a.at(1), Rz = rx_pos_a.at(2);
    double Rb = rx_orientation_a.at(0), Rt = rx_orientation_a.at(1), Rh = rx_orientation_a.at(2);

    // Derived inputs
    arma::uword n_ports_tx = ant_tx_a.n_ports();
    arma::uword n_ports_rx = ant_rx_a.n_ports();
    arma::uword n_path = fbs_pos_a.n_cols;

    // Initialize output memory
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    arma::cube *p_aod = nullptr, *p_eod = nullptr, *p_aoa = nullptr, *p_eoa = nullptr;

    auto coeff_re_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_re);
    auto coeff_im_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_im);
    auto delay_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &delay);

    pybind11::array_t<double> aod_p, eod_p, aoa_p, eoa_p;
    if (angles)
    {
        aod_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aod);
        eod_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eod);
        aoa_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aoa);
        eoa_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eoa);
        p_aod = &aod;
        p_eod = &eod;
        p_aoa = &aoa;
        p_eoa = &eoa;
    }

    // Call member function
    quadriga_lib::get_channels_spherical<double>(&ant_tx_a, &ant_rx_a,
                                                 Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, Rz, Rb, Rt, Rh,
                                                 &fbs_pos_a, &lbs_pos_a, &path_gain_a, &path_length_a, &M_a,
                                                 &coeff_re, &coeff_im, &delay,
                                                 center_freq, use_absolute_delays, add_fake_los_path,
                                                 p_aod, p_eod, p_aoa, p_eoa);

    if (angles)
        return py::make_tuple(coeff_re_p, coeff_im_p, delay_p, aod_p, eod_p, aoa_p, eoa_p);
    else
        return py::make_tuple(coeff_re_p, coeff_im_p, delay_p);
}
