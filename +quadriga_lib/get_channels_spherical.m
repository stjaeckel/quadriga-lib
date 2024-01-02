% GET_CHANNELS_SPHERICAL
%    Calculate channel coefficients from path data and antenna patterns
%    
% Description:
%    In this function, the wireless propagation channel between a transmitter and a receiver is calculated, 
%    based on a single transmit and receive position. Additionally, interaction points with the environment, 
%    which are derived from either Ray Tracing or Geometric Stochastic Models such as QuaDRiGa, are 
%    considered. The calculation is performed under the assumption of spherical wave propagation. For accurate 
%    execution of this process, several pieces of input data are required:
%    
%    - The 3D Cartesian (local) coordinates of both the transmitter and the receiver.
%    - The specific interaction positions of the propagation paths within the environment.
%    - The polarization transfer matrix for each propagation path.
%    - Antenna models for both the transmitter and the receiver.
%    - The orientations of the antennas.
%    
% Usage:
%    
%    [ coeff_re, coeff_im, delays, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( ...
%        e_theta_re_tx, e_theta_im_tx, e_phi_re_tx, e_phi_im_tx, azimuth_grid_tx, elevation_grid_tx, element_pos_tx, coupling_re_tx, coupling_im_tx, ...
%        e_theta_re_rx, e_theta_im_rx, e_phi_re_rx, e_phi_im_rx, azimuth_grid_rx, elevation_grid_rx, element_pos_rx, coupling_re_rx, coupling_im_rx, ...
%        fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, center_freq, use_absolute_delays, add_fake_los_path );
%    
% Input Arguments:
%    - TX Antenna data: (inputs 1-9, single or double)
%      e_theta_re_tx     | Real part of e-theta field component             | Size: [n_elevation_tx, n_azimuth_tx, n_tx_elements]
%      e_theta_im_tx     | Imaginary part of e-theta field component        | Size: [n_elevation_tx, n_azimuth_tx, n_tx_elements]
%      e_phi_re_tx       | Real part of e-phi field component               | Size: [n_elevation_tx, n_azimuth_tx, n_tx_elements]
%      e_phi_im_tx       | Imaginary part of e-phi field component          | Size: [n_elevation_tx, n_azimuth_tx, n_tx_elements]
%      azimuth_grid_tx   | Azimuth angles in [rad] -pi to pi, sorted        | Size: [n_azimuth_tx]
%      elevation_grid_tx | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: [n_elevation_tx]
%      element_pos_tx    | Antenna element (x,y,z) positions, optional      | Size: [3, n_tx_elements] or []
%      coupling_re_tx    | Real part of coupling matrix, optional           | Size: [n_tx_elements, n_tx_ports] or []
%      coupling_im_tx    | Imaginary part of coupling matrix, optional      | Size: [n_tx_elements, n_tx_ports] or []
%    
%    - RX Antenna data: (inputs 10-18, single or double)
%      e_theta_re_rx     | Real part of e-theta field component             | Size: [n_elevation_rx, n_azimuth_rx, n_rx_elements]
%      e_theta_im_rx     | Imaginary part of e-theta field component        | Size: [n_elevation_rx, n_azimuth_rx, n_rx_elements]
%      e_phi_re_rx       | Real part of e-phi field component               | Size: [n_elevation_rx, n_azimuth_rx, n_rx_elements]
%      e_phi_im_rx       | Imaginary part of e-phi field component          | Size: [n_elevation_rx, n_azimuth_rx, n_rx_elements]
%      azimuth_grid_rx   | Azimuth angles in [rad] -pi to pi, sorted        | Size: [n_azimuth_rx]
%      elevation_grid_rx | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: [n_elevation_rx]
%      element_pos_rx    | Antenna element (x,y,z) positions, optional      | Size: [3, n_rx_elements] or []
%      coupling_re_rx    | Real part of coupling matrix, optional           | Size: [n_rx_elements, n_rx_ports] or []
%      coupling_im_rx    | Imaginary part of coupling matrix, optional      | Size: [n_rx_elements, n_rx_ports] or []
%    
%    - fbs_pos
%      First interaction point of the rays and the environment; Size: [ 3, n_path ]
%    
%    - lbs_pos
%      Last interaction point of the rays and the environment; For single-bounce models, this must be 
%      identical to fbs_pos. Size: [ 3, n_path ]
%    
%    - path_gain
%      Path gain (linear scale); Size: [ 1, n_path ] or [ n_path, 1 ]
%    
%    - path_length
%      Total path length in meters; Size: [ 1, n_path ] or [ n_path, 1 ]
%    
%    - M
%      Polarization transfer matrix; interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH);
%      Size: [ 8, n_path ] 
%    
%    - tx_pos
%      Transmitter position in 3D Cartesian coordinates; Size: [3,1] or [1,3]
%    
%    - tx_orientation
%      3-element vector describing the orientation of the transmit antenna. The The first value describes 
%      the ”bank angle”, the second value describes the  ”tilt angle”, (positive values point upwards), 
%      the third value describes the bearing or ”heading angle”, in mathematic sense. Values must be given 
%      in [rad]. East corresponds to 0, and the angles increase counter-clockwise, so north is pi/2, south 
%      is -pi/2, and west is equal to pi. Single or double precision, Size: [3,1] or [1,3]
%    
%    - rx_pos
%      Receiver position in 3D Cartesian coordinates; Size: [3,1] or [1,3]
%    
%    - rx_orientation
%      3-element vector describing the orientation of the receive antenna. Size: [3,1] or [1,3]
%    
%    - center_freq
%      Center frequency in [Hz]; optional; If the value is not provided or set to 0, phase calculation 
%      in coefficients is disabled, i.e. that path length has not influence on the results. This can be 
%      used to calculate the antenna response for a specific angle and polarization. Scalar value
%    
%    - use_absolute_delays (optional)
%      If true, the LOS delay is included for all paths; Default is false, i.e. delays are normalized
%      to the LOS delay.
%    
%    - add_fake_los_path (optional)
%      If true, adds a zero-power LOS path as the first path in case where no LOS path was present.
%      Default: false
%    
% Output Arguments:
%    - coeff_re
%      Channel coefficients, real part, Size: [ n_rx_ports, n_tx_ports, n_path ]
%    
%    - coeff_im
%      Channel coefficients, imaginary part, Size: [ n_rx_ports, n_tx_ports, n_path ]
%    
%    - delays
%      Propagation delay in seconds, Size: [ n_rx_ports, n_tx_ports, n_path ]
%    
%    - aod (optional)
%      Azimuth of Departure angles in [rad], Size: [ n_rx_ports, n_tx_ports, n_path ]
%    
%    - eod (optional)
%      Elevation of Departure angles in [rad], Size: [ n_rx_ports, n_tx_ports, n_path ]
%    
%    - aoa (optional)
%      Azimuth of Arrival angles in [rad], Size: [ n_rx_ports, n_tx_ports, n_path ]
%    
%    - eoa (optional)
%      Elevation of Arrival angles in [rad], Size: [ n_rx_ports, n_tx_ports, n_path ]
%    
% Caveat:
%    - Antenna patterns, fbs_pos, lbs_pos, path_gain, path_length, and M can be provided in 
%      single or double precision, but types must match. Outputs are returned in the same type.
%    - Input data is directly accessed from MATLAB / Octave memory, without copying. To improve performance
%      of repeated computations (e.g. in loops), consider preparing the data accordingly to avoid unecessary 
%      computation.
%
%
% quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
% Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
% All rights reserved.
%
% e-mail: info@sjc-wireless.com
%
% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
% in compliance with the License. You may obtain a copy of the License at
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software distributed under the License
% is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
% or implied. See the License for the specific language governing permissions and limitations under
% the License.
    