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
Array antenna functions
SECTION!*/

/*!MD
# GET_CHANNELS_IRS
Calculate channel coefficients for intelligent reflective surfaces (IRS)

## Description:
- Calculates MIMO channel coefficients and delays for IRS-assisted communication using two channel segments:
  1. TX → IRS; 2. IRS → RX
- The IRS is modeled as a passive antenna array with phase shifts defined via its coupling matrix.
- IRS codebook entries can be selected via a port index (`i_irs`).
- Supports combining paths from both segments to form `n_path_irs` valid output paths, subject to a gain threshold.
- Optional second IRS array allows different antenna behavior for TX-IRS and IRS-RX directions.

## Usage:

```
[ coeff_re, coeff_im, delays, active_path, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, ...
    i_irs, threshold_dB, center_freq, use_absolute_delays, active_path,  ant_irs_2 );
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

- **`ant_irs`** [3] (required)<br>
  Struct containing the intelligent reflective surface (IRS) model:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation_irs, n_azimuth_irs, n_elements_irs]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation_irs, n_azimuth_irs, n_elements_irs]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation_irs, n_azimuth_irs, n_elements_irs]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation_irs, n_azimuth_irs, n_elements_irs]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth_irs]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation_irs]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements_irs]`
  `coupling_re`    | Coupling matrix, real part, optional                  | Size: `[n_elements_irs, n_ports_irs]`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Size: `[n_elements_irs, n_ports_irs]`

- **`fbs_pos_1`** [4] (required)<br>
  First-bounce scatterer positions of TX → IRS paths, Size: `[ 3, n_path_1 ]`.

- **`lbs_pos_1`** [5] (required)<br>
  Last-bounce scatterer positions of TX → IRS paths, Size `[3, n_path_1]`.

- **`path_gain_1`** [6] (required)<br>
  Path gains (linear) for TX → IRS paths, Length `n_path_1`.

- **`path_length_1`** [7] (required)<br>
  Path lengths for TX → IRS paths, Length `n_path_1`.

- **`M_1`** [8] (required)<br>
  Polarization transfer matrix for TX → IRS paths, Size `[8, n_path_1]`.

- **`fbs_pos_2`** [9] (required)<br>
  First-bounce scatterer positions of IRS → RX paths, Size: `[ 3, n_path_2 ]`

- **`lbs_pos_2`** [10] (required)<br>
  Last-bounce scatterer positions of IRS → RX paths, Size `[3, n_path_2]`

- **`path_gain_2`** [11] (required)<br>
  Path gains (linear) for IRS → RX paths, Length `n_path_2`.

- **`path_length_2`** [12] (required)<br>
  Path lengths for IRS → RX paths, Length `n_path_2`.

- **`M_2`** [13] (required)<br>
  Polarization transfer matrix for IRS → RX paths, Size `[8, n_path_2]`.

- **`tx_pos`** [14] (required)<br>
  Transmitter position in 3D Cartesian coordinates, Size: `[3,1]` or `[1,3]`

- **`tx_orientation`** [15] (required)<br>
  3-element vector describing the orientation of the transmit antenna in Euler angles (bank, tilt, heading),
  Size: `[3,1]` or `[1,3]`

- **`rx_pos`** [16] (required)<br>
  Receiver position in 3D Cartesian coordinates, Size: `[3,1]` or `[1,3]`

- **`rx_orientation`** [17] (required)<br>
  3-element vector describing the orientation of the receive antenna, Size: `[3,1]` or `[1,3]`

- **`irs_pos`** [18] (required)<br>
  IRS position in 3D Cartesian coordinates, Size: `[3,1]` or `[1,3]`

- **`irs_orientation`** [19] (required)<br>
  3-element (Euler) vector in Radians describing the orientation of the IRS, Size: `[3,1]` or `[1,3]`

- **`i_irs`** [20] (optional)<br>
  Index of IRS codebook entry (port number), Scalar,  Default: `0`.

- **`threshold_dB`** [21] (optional)<br>
  Threshold (in dB) below which paths are discarded, Scalar, Default: `-140.0`.

- **`center_freq`** [22] (optional)<br>
  Center frequency in [Hz]; optional; If the value is not provided or set to 0, phase calculation
  in coefficients is disabled, i.e. that path length has not influence on the results. This can be
  used to calculate the antenna response for a specific angle and polarization. Scalar value

- **`use_absolute_delays`** [23] (optional)<br>
  If true, the LOS delay is included for all paths; Default is `false`, i.e. delays are normalized
  to the LOS delay.

- **`active_path`** [24] (optional)<br>
  Optional bitmask for selecting active TX-IRS and IRS-RX path pairs. Ignores `threshold_dB` when provided.

- **`ant_irs_2`** [25] (optional)<br>
  Optional second IRS array (TX side for IRS → RX paths) for asymmetric IRS behavior. Same structure as for `ant_irs`

## Output Arguments:
- **`coeff_re`**<br>
  Channel coefficients, real part, Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`coeff_im`**<br>
  Channel coefficients, imaginary part, Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`delays`**<br>
  Propagation delay in seconds, Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`active_path`** (optional)<br>
  Boolean mask of length `n_path_1 * n_path_2`, indicating which path combinations were used.

- **`aod`** (optional)<br>
  Azimuth of Departure angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`eod`** (optional)<br>
  Elevation of Departure angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`aoa`** (optional)<br>
  Azimuth of Arrival angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`

- **`eoa`** (optional)<br>
  Elevation of Arrival angles in [rad], Size: `[ n_ports_rx, n_ports_tx, n_path ]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 19)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

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

    // Assemble IRS array antenna object
    auto ant_irs = quadriga_lib::arrayant<double>();
    ant_irs.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[2], "e_theta_re"));
    ant_irs.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[2], "e_theta_im"));
    ant_irs.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[2], "e_phi_re"));
    ant_irs.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[2], "e_phi_im"));
    ant_irs.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[2], "azimuth_grid"));
    ant_irs.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[2], "elevation_grid"));
    if (qd_mex_has_field(prhs[2], "element_pos"))
        ant_irs.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[2], "element_pos"));
    if (qd_mex_has_field(prhs[2], "coupling_re"))
        ant_irs.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[2], "coupling_re"));
    if (qd_mex_has_field(prhs[2], "coupling_im"))
        ant_irs.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[2], "coupling_im"));

    // Segment TX > IRS
    const auto fbs_pos_1 = qd_mex_get_double_Mat(prhs[3]);
    const auto lbs_pos_1 = qd_mex_get_double_Mat(prhs[4]);
    const auto path_gain_1 = qd_mex_get_double_Col(prhs[5]);
    const auto path_length_1 = qd_mex_get_double_Col(prhs[6]);
    const auto M_1 = qd_mex_get_double_Mat(prhs[7]);

    // Segment IRS > RX
    const auto fbs_pos_2 = qd_mex_get_double_Mat(prhs[8]);
    const auto lbs_pos_2 = qd_mex_get_double_Mat(prhs[9]);
    const auto path_gain_2 = qd_mex_get_double_Col(prhs[10]);
    const auto path_length_2 = qd_mex_get_double_Col(prhs[11]);
    const auto M_2 = qd_mex_get_double_Mat(prhs[12]);

    // Positions and Orientations
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[13], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[14], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[15], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[16], "rx_orientation", 3);
    const auto irs_pos = qd_mex_typecast_Col<double>(prhs[17], "irs_pos", 3);
    const auto irs_orientation = qd_mex_typecast_Col<double>(prhs[18], "irs_orientation", 3);

    // Configuration parameters
    arma::uword i_irs = (nrhs < 20) ? 0 : qd_mex_get_scalar<arma::uword>(prhs[19], "i_irs", 0);
    double threshold_dB = (nrhs < 21) ? 0.0 : qd_mex_get_scalar<double>(prhs[20], "threshold_dB", -140.0);
    double center_freq = (nrhs < 22) ? 0.0 : qd_mex_get_scalar<double>(prhs[21], "center_freq", 0.0);
    bool use_absolute_delays = (nrhs < 23) ? false : qd_mex_get_scalar<bool>(prhs[22], "use_absolute_delays", false);
    const auto active_path_in = (nrhs < 24) ? std::vector<bool>() : qd_mex_matlab2vector_Bool(prhs[23]);

    // Alternative IRS transmit array
    auto ant_irs_2 = quadriga_lib::arrayant<double>();
    bool use_alternative_irs = nrhs > 24 && mxIsStruct(prhs[24]);
    if (use_alternative_irs)
    {
        ant_irs_2.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[24], "e_theta_re"));
        ant_irs_2.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[24], "e_theta_im"));
        ant_irs_2.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[24], "e_phi_re"));
        ant_irs_2.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[24], "e_phi_im"));
        ant_irs_2.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[24], "azimuth_grid"));
        ant_irs_2.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[24], "elevation_grid"));
        if (qd_mex_has_field(prhs[24], "element_pos"))
            ant_irs_2.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[24], "element_pos"));
        if (qd_mex_has_field(prhs[24], "coupling_re"))
            ant_irs_2.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[24], "coupling_re"));
        if (qd_mex_has_field(prhs[24], "coupling_im"))
            ant_irs_2.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[24], "coupling_im"));
    }

    // Extract scalar values
    double Tx = tx_pos.at(0), Ty = tx_pos.at(1), Tz = tx_pos.at(2);
    double Tb = tx_orientation.at(0), Tt = tx_orientation.at(1), Th = tx_orientation.at(2);
    double Rx = rx_pos.at(0), Ry = rx_pos.at(1), Rz = rx_pos.at(2);
    double Rb = rx_orientation.at(0), Rt = rx_orientation.at(1), Rh = rx_orientation.at(2);
    double Ix = irs_pos.at(0), Iy = irs_pos.at(1), Iz = irs_pos.at(2);
    double Ib = irs_orientation.at(0), It = irs_orientation.at(1), Ih = irs_orientation.at(2);

    // Declare outputs
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    // Optionals
    arma::cube *p_aod = (nlhs > 4) ? &aod : nullptr;
    arma::cube *p_eod = (nlhs > 5) ? &eod : nullptr;
    arma::cube *p_aoa = (nlhs > 6) ? &aoa : nullptr;
    arma::cube *p_eoa = (nlhs > 7) ? &eoa : nullptr;
    const quadriga_lib::arrayant<double> *p_irs_array_2 = (use_alternative_irs) ? &ant_irs_2 : nullptr;
    const std::vector<bool> *p_active_path = (active_path_in.size() == 0) ? nullptr : &active_path_in;

    // Call library function
    std::vector<bool> active_path_out;
    CALL_QD(active_path_out = quadriga_lib::get_channels_irs<double>(&ant_tx, &ant_rx, &ant_irs,
                                                                     Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, Rz, Rb, Rt, Rh, Ix, Iy, Iz, Ib, It, Ih,
                                                                     &fbs_pos_1, &lbs_pos_1, &path_gain_1, &path_length_1, &M_1,
                                                                     &fbs_pos_2, &lbs_pos_2, &path_gain_2, &path_length_2, &M_2,
                                                                     &coeff_re, &coeff_im, &delay,
                                                                     i_irs, threshold_dB, center_freq, use_absolute_delays,
                                                                     p_aod, p_eod, p_aoa, p_eoa, p_irs_array_2, p_active_path));

    // Write to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&coeff_re);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&coeff_im);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&delay);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&active_path_out);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&aod);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&eod);
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&aoa);
    if (nlhs > 7)
        plhs[7] = qd_mex_copy2matlab(&eoa);
}