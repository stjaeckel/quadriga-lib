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

#include "mex.h"
#include "quadriga_arrayant.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel generation functions
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
[ coeff_re, coeff_im, delays, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( ant_tx, ant_rx, ...
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, ...
    center_freq, use_absolute_delays, add_fake_los_path );
```

## Input Arguments:
- **`ant_tx`** [1] (required)<br>
  Struct containing the transmit (TX) arrayant data with the following fields:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation_tx, n_azimuth_tx, n_elements_tx]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation_tx, n_azimuth_tx, n_elements_tx]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation_tx, n_azimuth_tx, n_elements_tx]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation_tx, n_azimuth_tx, n_elements_tx]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth_tx]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation_tx]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements_tx]`
  `coupling_re`    | Coupling matrix, real part, optional                  | Size: `[n_elements_tx, n_ports_tx]`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Size: `[n_elements_tx, n_ports_tx]`

- **`ant_rx`** [2] (required)<br>
  Struct containing the receive (RX) arrayant data with the following fields:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation_rx, n_azimuth_rx, n_elements_rx]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation_rx, n_azimuth_rx, n_elements_rx]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation_rx, n_azimuth_rx, n_elements_rx]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation_rx, n_azimuth_rx, n_elements_rx]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth_rx]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation_rx]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements_rx]`
  `coupling_re`    | Coupling matrix, real part, optional                  | Size: `[n_elements_rx, n_ports_rx]`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Size: `[n_elements_rx, n_ports_rx]`

- **`fbs_pos`** [3] (required)<br>
  First interaction point of the rays and the environment, Size: `[ 3, n_path ]`

- **`lbs_pos`** [4] (required)<br>
  Last interaction point of the rays and the environment; For single-bounce models, this must be
  identical to `fbs_pos`, Size: `[ 3, n_path ]`

- **`path_gain`** [5] (required)<br>
  Path gain (linear scale), Size: `[ 1, n_path ]` or `[ n_path, 1 ]`

- **`path_length`** [6] (required)<br>
  Total path length in meters; If `path_length` is shorter than the shortest possible path from TX to
  FBS to LBS to RX, it is replaced by the shortest path length, Size: `[ 1, n_path ]` or `[ n_path, 1 ]`

- **`M`** [7] (required)<br>
  Polarization transfer matrix; interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH),
  Size: `[ 8, n_path ]`

- **`tx_pos`** [8] (required)<br>
  Transmitter position in 3D Cartesian coordinates, Size: `[3,1]` or `[1,3]`

- **`tx_orientation`** [9] (required)<br>
  3-element vector describing the orientation of the transmit antenna in Euler angles (bank, tilt, heading),
  Size: `[3,1]` or `[1,3]`

- **`rx_pos`** [10] (required)<br>
  Receiver position in 3D Cartesian coordinates, Size: `[3,1]` or `[1,3]`

- **`rx_orientation`** [11] (required)<br>
  3-element vector describing the orientation of the receive antenna, Size: `[3,1]` or `[1,3]`

- **`center_freq`** [12] (optional)<br>
  Center frequency in [Hz]; optional; If the value is not provided or set to 0, phase calculation
  in coefficients is disabled, i.e. that path length has not influence on the results. This can be
  used to calculate the antenna response for a specific angle and polarization. Scalar value

- **`use_absolute_delays`** [13] (optional)<br>
  If true, the LOS delay is included for all paths; Default is `false`, i.e. delays are normalized
  to the LOS delay.

- **`add_fake_los_path`** [14] (optional)<br>
  If true, adds a zero-power LOS path as the first path in case where no LOS path was present.
  Default: `false`

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
  Azimuth of Departure angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`eod`** (optional)<br>
  Elevation of Departure angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`aoa`** (optional)<br>
  Azimuth of Arrival angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`eoa`** (optional)<br>
  Elevation of Arrival angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`

## Caveat:
- Input data is directly accessed from MATLAB / Octave memory, without copying if it is provided in
  **double** precision.
- Other formats (e.g. single precision inputs) will be converted to double automatically, causing
  additional computation steps.
- To improve performance of repeated computations (e.g. in loops), consider preparing the data
  accordingly to avoid unecessary computations.
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 11)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs == 0)
        return;

    // Assemble TX array antenna object
    auto ant_tx = quadriga_lib::arrayant<double>();
    ant_tx.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_re"));
    ant_tx.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_im"));
    ant_tx.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_re"));
    ant_tx.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_im"));
    ant_tx.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "azimuth_grid"));
    ant_tx.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "elevation_grid"));
    if (qd_mex_has_field(prhs[0], "element_pos"))
        ant_tx.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "element_pos"));
    if (qd_mex_has_field(prhs[0], "coupling_re"))
        ant_tx.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_re"));
    if (qd_mex_has_field(prhs[0], "coupling_im"))
        ant_tx.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_im"));

    // Assemble RX array antenna object
    auto ant_rx = quadriga_lib::arrayant<double>();
    ant_rx.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_theta_re"));
    ant_rx.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_theta_im"));
    ant_rx.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_phi_re"));
    ant_rx.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_phi_im"));
    ant_rx.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[1], "azimuth_grid"));
    ant_rx.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[1], "elevation_grid"));
    if (qd_mex_has_field(prhs[1], "element_pos"))
        ant_rx.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "element_pos"));
    if (qd_mex_has_field(prhs[1], "coupling_re"))
        ant_rx.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "coupling_re"));
    if (qd_mex_has_field(prhs[1], "coupling_im"))
        ant_rx.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "coupling_im"));

    // Parse other arguments
    const auto fbs_pos = qd_mex_get_double_Mat(prhs[2]);
    const auto lbs_pos = qd_mex_get_double_Mat(prhs[3]);
    const auto path_gain = qd_mex_get_double_Col(prhs[4]);
    const auto path_length = qd_mex_get_double_Col(prhs[5]);
    const auto M = qd_mex_get_double_Mat(prhs[6]);
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[7], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[8], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[9], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[10], "rx_orientation", 3);
    double center_freq = (nrhs < 12) ? 0.0 : qd_mex_get_scalar<double>(prhs[11], "center_freq", 0.0);
    bool use_absolute_delays = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "use_absolute_delays", false);
    bool add_fake_los_path = (nrhs < 14) ? false : qd_mex_get_scalar<bool>(prhs[13], "add_fake_los_path", false);

    // Extract scalar values
    double Tx = tx_pos.at(0), Ty = tx_pos.at(1), Tz = tx_pos.at(2);
    double Tb = tx_orientation.at(0), Tt = tx_orientation.at(1), Th = tx_orientation.at(2);
    double Rx = rx_pos.at(0), Ry = rx_pos.at(1), Rz = rx_pos.at(2);
    double Rb = rx_orientation.at(0), Rt = rx_orientation.at(1), Rh = rx_orientation.at(2);

    // Derived inputs
    arma::uword n_ports_tx = ant_tx.n_ports();
    arma::uword n_ports_rx = ant_rx.n_ports();
    arma::uword n_path = fbs_pos.n_cols;

    // Initialize output memory
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    arma::cube *p_aod = nullptr, *p_eod = nullptr, *p_aoa = nullptr, *p_eoa = nullptr;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&coeff_re, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&coeff_im, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&delay, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&aod, n_ports_rx, n_ports_tx, n_path), p_aod = &aod;
    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&eod, n_ports_rx, n_ports_tx, n_path), p_eod = &eod;
    if (nlhs > 5)
        plhs[5] = qd_mex_init_output(&aoa, n_ports_rx, n_ports_tx, n_path), p_aoa = &aoa;
    if (nlhs > 6)
        plhs[6] = qd_mex_init_output(&eoa, n_ports_rx, n_ports_tx, n_path), p_eoa = &eoa;

    // Call member function
    CALL_QD(quadriga_lib::get_channels_spherical<double>(&ant_tx, &ant_rx,
                                                         Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, Rz, Rb, Rt, Rh,
                                                         &fbs_pos, &lbs_pos, &path_gain, &path_length, &M,
                                                         &coeff_re, &coeff_im, &delay,
                                                         center_freq, use_absolute_delays, add_fake_los_path,
                                                         p_aod, p_eod, p_aoa, p_eoa));
}
