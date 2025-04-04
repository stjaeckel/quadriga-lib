// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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
[ coeff_re, coeff_im, delays, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( ...
    e_theta_re_tx, e_theta_im_tx, e_phi_re_tx, e_phi_im_tx, azimuth_grid_tx, elevation_grid_tx, element_pos_tx, coupling_re_tx, coupling_im_tx, ...
    e_theta_re_rx, e_theta_im_rx, e_phi_re_rx, e_phi_im_rx, azimuth_grid_rx, elevation_grid_rx, element_pos_rx, coupling_re_rx, coupling_im_rx, ...
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, center_freq, use_absolute_delays, add_fake_los_path );
```

## Input Arguments:
- **TX Antenna data:** (inputs 1-9, single or double)
  `e_theta_re_tx`     | Real part of e-theta field component             | Size: `[n_elevation_tx, n_azimuth_tx, n_tx_elements]`
  `e_theta_im_tx`     | Imaginary part of e-theta field component        | Size: `[n_elevation_tx, n_azimuth_tx, n_tx_elements]`
  `e_phi_re_tx`       | Real part of e-phi field component               | Size: `[n_elevation_tx, n_azimuth_tx, n_tx_elements]`
  `e_phi_im_tx`       | Imaginary part of e-phi field component          | Size: `[n_elevation_tx, n_azimuth_tx, n_tx_elements]`
  `azimuth_grid_tx`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth_tx]`
  `elevation_grid_tx` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation_tx]`
  `element_pos_tx`    | Antenna element (x,y,z) positions, optional      | Size: `[3, n_tx_elements]` or `[]`
  `coupling_re_tx`    | Real part of coupling matrix, optional           | Size: `[n_tx_elements, n_tx_ports]` or `[]`
  `coupling_im_tx`    | Imaginary part of coupling matrix, optional      | Size: `[n_tx_elements, n_tx_ports]` or `[]`

- **RX Antenna data:** (inputs 10-18, single or double)
  `e_theta_re_rx`     | Real part of e-theta field component             | Size: `[n_elevation_rx, n_azimuth_rx, n_rx_elements]`
  `e_theta_im_rx`     | Imaginary part of e-theta field component        | Size: `[n_elevation_rx, n_azimuth_rx, n_rx_elements]`
  `e_phi_re_rx`       | Real part of e-phi field component               | Size: `[n_elevation_rx, n_azimuth_rx, n_rx_elements]`
  `e_phi_im_rx`       | Imaginary part of e-phi field component          | Size: `[n_elevation_rx, n_azimuth_rx, n_rx_elements]`
  `azimuth_grid_rx`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth_rx]`
  `elevation_grid_rx` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation_rx]`
  `element_pos_rx`    | Antenna element (x,y,z) positions, optional      | Size: `[3, n_rx_elements]` or `[]`
  `coupling_re_rx`    | Real part of coupling matrix, optional           | Size: `[n_rx_elements, n_rx_ports]` or `[]`
  `coupling_im_rx`    | Imaginary part of coupling matrix, optional      | Size: `[n_rx_elements, n_rx_ports]` or `[]`

- **`fbs_pos`**<br>
  First interaction point of the rays and the environment; Size: `[ 3, n_path ]`

- **`lbs_pos`**<br>
  Last interaction point of the rays and the environment; For single-bounce models, this must be
  identical to `fbs_pos`. Size: `[ 3, n_path ]`

- **`path_gain`**<br>
  Path gain (linear scale); Size: `[ 1, n_path ]` or `[ n_path, 1 ]`

- **`path_length`**<br>
  Total path length in meters; If `path_length` is shorter than the shortest possible path from TX to
  FBS to LBS to RX, it is replaced by the shortest path length. <br>Size: `[ 1, n_path ]` or `[ n_path, 1 ]`

- **`M`**<br>
  Polarization transfer matrix; interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH);
  Size: `[ 8, n_path ]`

- **`tx_pos`**<br>
  Transmitter position in 3D Cartesian coordinates; Size: `[3,1]` or `[1,3]`

- **`tx_orientation`**<br>
  3-element vector describing the orientation of the transmit antenna. The The first value describes
  the ”bank angle”, the second value describes the  ”tilt angle”, (positive values point upwards),
  the third value describes the bearing or ”heading angle”, in mathematic sense. Values must be given
  in [rad]. East corresponds to 0, and the angles increase counter-clockwise, so north is pi/2, south
  is -pi/2, and west is equal to pi. Single or double precision, Size: `[3,1]` or `[1,3]`

- **`rx_pos`**<br>
  Receiver position in 3D Cartesian coordinates; Size: `[3,1]` or `[1,3]`

- **`rx_orientation`**<br>
  3-element vector describing the orientation of the receive antenna. Size: `[3,1]` or `[1,3]`

- **`center_freq`**<br>
  Center frequency in [Hz]; optional; If the value is not provided or set to 0, phase calculation
  in coefficients is disabled, i.e. that path length has not influence on the results. This can be
  used to calculate the antenna response for a specific angle and polarization. Scalar value

- **`use_absolute_delays`** (optional)<br>
  If true, the LOS delay is included for all paths; Default is `false`, i.e. delays are normalized
  to the LOS delay.

- **`add_fake_los_path`** (optional)<br>
  If true, adds a zero-power LOS path as the first path in case where no LOS path was present.
  Default: `false`

## Output Arguments:
- **`coeff_re`**<br>
  Channel coefficients, real part, Size: `[ n_rx_ports, n_tx_ports, n_path ]`

- **`coeff_im`**<br>
  Channel coefficients, imaginary part, Size: `[ n_rx_ports, n_tx_ports, n_path ]`

- **`delays`**<br>
  Propagation delay in seconds, Size: `[ n_rx_ports, n_tx_ports, n_path ]`

- **`aod`** (optional)<br>
  Azimuth of Departure angles in [rad], Size: `[ n_rx_ports, n_tx_ports, n_path ]`

- **`eod`** (optional)<br>
  Elevation of Departure angles in [rad], Size: `[ n_rx_ports, n_tx_ports, n_path ]`

- **`aoa`** (optional)<br>
  Azimuth of Arrival angles in [rad], Size: `[ n_rx_ports, n_tx_ports, n_path ]`

- **`eoa`** (optional)<br>
  Elevation of Arrival angles in [rad], Size: `[ n_rx_ports, n_tx_ports, n_path ]`

## Caveat:
- Antenna patterns, `fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, and `M` can be provided in
  single or double precision, but types must match. Outputs are returned in the same type.
- Input data is directly accessed from MATLAB / Octave memory, without copying. To improve performance
  of repeated computations (e.g. in loops), consider preparing the data accordingly to avoid unecessary
  computation.
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs - Transmit antenna:
    //  0 - e_theta_re      Vertical component of the electric field, real part,            Size [n_tx_elevation, n_tx_azimuth, n_tx_elements]
    //  1 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_tx_elevation, n_tx_azimuth, n_tx_elements]
    //  2 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_tx_elevation, n_tx_azimuth, n_tx_elements]
    //  3 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_tx_elevation, n_tx_azimuth, n_tx_elements]
    //  4 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_tx_azimuth"
    //  5 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_tx_elevation"
    //  6 - element_pos     Element positions, optional, Default: [0,0,0]                   Size [3, n_tx_elements] or []
    //  7 - coupling_re     Coupling matrix, real part, Default: Identity matrix            Size [n_tx_elements, n_tx_ports] or []
    //  8 - coupling_im     Coupling matrix, imaginary part, Default: Zero matrix           Size [n_tx_elements, n_tx_ports] or []

    // Inputs - Receive antenna:
    //  9 - e_theta_re      Vertical component of the electric field, real part,            Size [n_rx_elevation, n_rx_azimuth, n_rx_elements]
    // 10 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_rx_elevation, n_rx_azimuth, n_rx_elements]
    // 11 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_rx_elevation, n_rx_azimuth, n_rx_elements]
    // 12 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_rx_elevation, n_rx_azimuth, n_rx_elements]
    // 13 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_rx_azimuth"
    // 14 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_rx_elevation"
    // 15 - element_pos     Element positions, optional, Default: [0,0,0]                   Size [3, n_rx_elements] or []
    // 16 - coupling_re     Coupling matrix, real part, Default: Identity matrix            Size [n_rx_elements, n_rx_ports] or []
    // 17 - coupling_im     Coupling matrix, imaginary part, Default: Zero matrix           Size [n_rx_elements, n_rx_ports] or []

    // Inputs - Path data
    // 18 - fbs_pos         First-bounce scatterer positions                                Size [3, n_path]
    // 19 - lbs_pos         Last-bounce scatterer positions                                 Size [3, n_path]
    // 20 - path_gain       Path gain (linear scale)                                        Size [1, n_path]
    // 21 - path_length     Absolute path length from TX to RX phase center                 Size [1, n_path]
    // 22 - M               Polarization transfer matrix                                    Size [8, n_path]
    //                      interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH)

    // Inputs - Positions and orientations:
    // 23 - tx_pos          Transmitter position in Cartesian coordinates                   Size [3,1]
    // 24 - tx_orientation  Transmitter orientation (bank, tilt, head) in [rad]             Size [3,1]
    // 25 - rx_pos          Receiver position in Cartesian coordinates                      Size [3,1]
    // 26 - rx_orientation  Receiver orientation (bank, tilt, head) in [rad]                Size [3,1]

    // Inputs - Scalars and switches
    // 27 - center_freq          Center frequency in [Hz]
    // 28 - use_absolute_delays  If true, the LOS delay is included for all paths
    // 29 - add_fake_los_path    Adds a zero-power LOS path in case where no LOS path was present

    // Outputs:
    //  0 - coeff_re        Channel coefficients, real part                                 Size [n_rx_ports, n_tx_ports, n_path]
    //  1 - coeff_im        Channel coefficients, imaginary part                            Size [n_rx_ports, n_tx_ports, n_path]
    //  2 - delays          Propagation delay in seconds                                    Size [n_rx_ports, n_tx_ports, n_path]
    //  3 - aod             [OPTIONAL] Azimuth of Departure angles in [rad]                 Size [n_rx_ports, n_tx_ports, n_path]
    //  4 - eod             [OPTIONAL] Elevation of Departure angles in [rad]               Size [n_rx_ports, n_tx_ports, n_path]
    //  5 - aoa             [OPTIONAL] Azimuth of Arrival angles in [rad]                   Size [n_rx_ports, n_tx_ports, n_path]
    //  6 - eoa             [OPTIONAL] Elevation of Arrival angles in [rad]                 Size [n_rx_ports, n_tx_ports, n_path]

    if (nrhs < 27)
        mexErrMsgIdAndTxt("quadriga_lib:get_channels_spherical:no_input", "Incorrect number of input arguments.");

    if (nlhs < 3 || nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:get_channels_spherical:no_output", "Incorrect number of output arguments.");

    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:get_channels_spherical:wrong_type", "Inputs must be provided in 'single' or 'double' precision.");

    for (int i = 1; i < 23; i++)
        if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
            mexErrMsgIdAndTxt("quadriga_lib:get_channels_spherical:wrong_type", "All inputs must have the same type: 'single' or 'double' precision");

    // Transmit antenna
    quadriga_lib::arrayant<float> tx_array_single;
    quadriga_lib::arrayant<double> tx_array_double;
    if (use_single)
        tx_array_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[0]),
        tx_array_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[1]),
        tx_array_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[2]),
        tx_array_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[3]),
        tx_array_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[4]),
        tx_array_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[5]),
        tx_array_single.element_pos = qd_mex_reinterpret_Mat<float>(prhs[6]),
        tx_array_single.coupling_re = qd_mex_reinterpret_Mat<float>(prhs[7]),
        tx_array_single.coupling_im = qd_mex_reinterpret_Mat<float>(prhs[8]);
    else
        tx_array_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[0]),
        tx_array_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[1]),
        tx_array_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[2]),
        tx_array_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[3]),
        tx_array_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[4]),
        tx_array_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[5]),
        tx_array_double.element_pos = qd_mex_reinterpret_Mat<double>(prhs[6]),
        tx_array_double.coupling_re = qd_mex_reinterpret_Mat<double>(prhs[7]),
        tx_array_double.coupling_im = qd_mex_reinterpret_Mat<double>(prhs[8]);

    std::string error_message = use_single ? tx_array_single.validate() : tx_array_double.validate();
    if (!error_message.empty())
    {
        error_message = "Tx array: " + error_message;
        mexErrMsgIdAndTxt("quadriga_lib:get_channels_spherical:invalid_argument", error_message.c_str());
    }

    // Receive antenna
    quadriga_lib::arrayant<float> rx_array_single;
    quadriga_lib::arrayant<double> rx_array_double;
    if (use_single)
        rx_array_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[9]),
        rx_array_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[10]),
        rx_array_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[11]),
        rx_array_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[12]),
        rx_array_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[13]),
        rx_array_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[14]),
        rx_array_single.element_pos = qd_mex_reinterpret_Mat<float>(prhs[15]),
        rx_array_single.coupling_re = qd_mex_reinterpret_Mat<float>(prhs[16]),
        rx_array_single.coupling_im = qd_mex_reinterpret_Mat<float>(prhs[17]);
    else
        rx_array_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[9]),
        rx_array_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[10]),
        rx_array_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[11]),
        rx_array_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[12]),
        rx_array_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[13]),
        rx_array_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[14]),
        rx_array_double.element_pos = qd_mex_reinterpret_Mat<double>(prhs[15]),
        rx_array_double.coupling_re = qd_mex_reinterpret_Mat<double>(prhs[16]),
        rx_array_double.coupling_im = qd_mex_reinterpret_Mat<double>(prhs[17]);

    error_message = use_single ? rx_array_single.validate() : rx_array_double.validate();
    if (!error_message.empty())
    {
        error_message = "Rx array: " + error_message;
        mexErrMsgIdAndTxt("quadriga_lib:get_channels_spherical:invalid_argument", error_message.c_str());
    }

    // Path data
    arma::fmat fbs_pos_single, lbs_pos_single, M_single;
    arma::mat fbs_pos_double, lbs_pos_double, M_double;
    arma::fvec path_gain_single, path_length_single;
    arma::vec path_gain_double, path_length_double;
    if (use_single)
        fbs_pos_single = qd_mex_reinterpret_Mat<float>(prhs[18]),
        lbs_pos_single = qd_mex_reinterpret_Mat<float>(prhs[19]),
        path_gain_single = qd_mex_reinterpret_Col<float>(prhs[20]),
        path_length_single = qd_mex_reinterpret_Col<float>(prhs[21]),
        M_single = qd_mex_reinterpret_Mat<float>(prhs[22]);
    else
        fbs_pos_double = qd_mex_reinterpret_Mat<double>(prhs[18]),
        lbs_pos_double = qd_mex_reinterpret_Mat<double>(prhs[19]),
        path_gain_double = qd_mex_reinterpret_Col<double>(prhs[20]),
        path_length_double = qd_mex_reinterpret_Col<double>(prhs[21]),
        M_double = qd_mex_reinterpret_Mat<double>(prhs[22]);

    unsigned long long n_path = use_single ? fbs_pos_single.n_cols : fbs_pos_double.n_cols;
    if (n_path == 0ULL ||
        (use_single && (lbs_pos_single.n_cols != n_path || path_gain_single.n_elem != n_path || path_length_single.n_elem != n_path || M_single.n_cols != n_path)) ||
        (!use_single && (lbs_pos_double.n_cols != n_path || path_gain_double.n_elem != n_path || path_length_double.n_elem != n_path || M_double.n_cols != n_path)))
    {
        error_message = "Inconsistent number of paths in input arguments";
        mexErrMsgIdAndTxt("quadriga_lib:get_channels_spherical:invalid_argument", error_message.c_str());
    }

    // Positions and orientations
    arma::vec tmp = qd_mex_typecast_Col<double>(prhs[23], "tx_pos", 3);
    double Tx = tmp.at(0), Ty = tmp.at(1), Tz = tmp.at(2);
    tmp = qd_mex_typecast_Col<double>(prhs[24], "tx_orientation", 3);
    double Tb = tmp.at(0), Tt = tmp.at(1), Th = tmp.at(2);
    tmp = qd_mex_typecast_Col<double>(prhs[25], "rx_pos", 3);
    double Rx = tmp.at(0), Ry = tmp.at(1), Rz = tmp.at(2);
    tmp = qd_mex_typecast_Col<double>(prhs[26], "rx_orientation", 3);
    double Rb = tmp.at(0), Rt = tmp.at(1), Rh = tmp.at(2);

    // Scalars and switches
    double center_freq = (nrhs < 28) ? 0.0 : qd_mex_get_scalar<double>(prhs[27], "center_freq", 0.0);
    bool use_absolute_delays = (nrhs < 29) ? false : qd_mex_get_scalar<bool>(prhs[28], "use_absolute_delays", false);
    bool add_fake_los_path = (nrhs < 30) ? false : qd_mex_get_scalar<bool>(prhs[29], "add_fake_los_path", false);

    // Allocate output memory
    unsigned long long n_tx_ports = use_single ? tx_array_single.n_ports() : tx_array_double.n_ports();
    unsigned long long n_rx_ports = use_single ? rx_array_single.n_ports() : rx_array_double.n_ports();
    n_path += add_fake_los_path ? 1ULL : 0ULL;

    arma::fcube coeff_re_single, coeff_im_single, delay_single, aod_single, eod_single, aoa_single, eoa_single;
    arma::cube coeff_re_double, coeff_im_double, delay_double, aod_double, eod_double, aoa_double, eoa_double;

    arma::fcube *p_aod_single = (nlhs > 3) ? &aod_single : NULL;
    arma::fcube *p_eod_single = (nlhs > 4) ? &eod_single : NULL;
    arma::fcube *p_aoa_single = (nlhs > 5) ? &aoa_single : NULL;
    arma::fcube *p_eoa_single = (nlhs > 6) ? &eoa_single : NULL;

    arma::cube *p_aod_double = (nlhs > 3) ? &aod_double : NULL;
    arma::cube *p_eod_double = (nlhs > 4) ? &eod_double : NULL;
    arma::cube *p_aoa_double = (nlhs > 5) ? &aoa_double : NULL;
    arma::cube *p_eoa_double = (nlhs > 6) ? &eoa_double : NULL;

    if (use_single)
    {
        plhs[0] = qd_mex_init_output(&coeff_re_single, n_rx_ports, n_tx_ports, n_path);
        plhs[1] = qd_mex_init_output(&coeff_im_single, n_rx_ports, n_tx_ports, n_path);
        plhs[2] = qd_mex_init_output(&delay_single, n_rx_ports, n_tx_ports, n_path);
        if (nlhs > 3)
            plhs[3] = qd_mex_init_output(p_aod_single, n_rx_ports, n_tx_ports, n_path);
        if (nlhs > 4)
            plhs[4] = qd_mex_init_output(p_eod_single, n_rx_ports, n_tx_ports, n_path);
        if (nlhs > 5)
            plhs[5] = qd_mex_init_output(p_aoa_single, n_rx_ports, n_tx_ports, n_path);
        if (nlhs > 6)
            plhs[6] = qd_mex_init_output(p_eoa_single, n_rx_ports, n_tx_ports, n_path);
    }
    else
    {
        plhs[0] = qd_mex_init_output(&coeff_re_double, n_rx_ports, n_tx_ports, n_path);
        plhs[1] = qd_mex_init_output(&coeff_im_double, n_rx_ports, n_tx_ports, n_path);
        plhs[2] = qd_mex_init_output(&delay_double, n_rx_ports, n_tx_ports, n_path);
        if (nlhs > 3)
            plhs[3] = qd_mex_init_output(p_aod_double, n_rx_ports, n_tx_ports, n_path);
        if (nlhs > 4)
            plhs[4] = qd_mex_init_output(p_eod_double, n_rx_ports, n_tx_ports, n_path);
        if (nlhs > 5)
            plhs[5] = qd_mex_init_output(p_aoa_double, n_rx_ports, n_tx_ports, n_path);
        if (nlhs > 6)
            plhs[6] = qd_mex_init_output(p_eoa_double, n_rx_ports, n_tx_ports, n_path);
    }

    // Call library function
    try
    {
        if (use_single)
            quadriga_lib::get_channels_spherical<float>(&tx_array_single, &rx_array_single,
                                                        (float)Tx, (float)Ty, (float)Tz, (float)Tb, (float)Tt, (float)Th,
                                                        (float)Rx, (float)Ry, (float)Rz, (float)Rb, (float)Rt, (float)Rh,
                                                        &fbs_pos_single, &lbs_pos_single,
                                                        &path_gain_single, &path_length_single, &M_single,
                                                        &coeff_re_single, &coeff_im_single, &delay_single,
                                                        (float)center_freq, use_absolute_delays, add_fake_los_path,
                                                        p_aod_single, p_eod_single, p_aoa_single, p_eoa_single);
        else
            quadriga_lib::get_channels_spherical<double>(&tx_array_double, &rx_array_double,
                                                         Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, Rz, Rb, Rt, Rh,
                                                         &fbs_pos_double, &lbs_pos_double,
                                                         &path_gain_double, &path_length_double, &M_double,
                                                         &coeff_re_double, &coeff_im_double, &delay_double,
                                                         center_freq, use_absolute_delays, add_fake_los_path,
                                                         p_aod_double, p_eod_double, p_aoa_double, p_eoa_double);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}