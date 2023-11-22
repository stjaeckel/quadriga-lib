% ARRAYANT_ROTATE_PATTERN
%    Rotates antenna patterns
%    
% Description:
%    This MATLAB function transforms the radiation patterns of array antenna elements, allowing for
%    precise rotations around the three principal axes (x, y, z) of the local Cartesian coordinate system.
%    This is essential in antenna design and optimization, enabling engineers to tailor the radiation
%    pattern for enhanced performance. The function also adjusts the sampling grid for non-uniformly
%    sampled antennas, such as parabolic antennas with small apertures, ensuring accurate and efficient
%    computations. The 3 rotations are applies in the order: 1. rotation around the x-axis (bank angle);
%    2. rotation around the y-axis (tilt angle), 3. rotation around the z-axis (heading angle)
%    
% Usage:
%    
%    [ e_theta_re_r, e_theta_im_r, e_phi_re_r, e_phi_im_r, azimuth_grid_r, elevation_grid_r, element_pos_r ] = ...
%        quadriga_lib.arrayant_rotate_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, ...
%        elevation_grid, element_pos, x_deg, y_deg, z_deg, usage )
%    
% Input Arguments:
%    - Antenna data: (inputs 1-7, single or double)
%      e_theta_re     | Real part of e-theta field component             | Size: [n_elevation, n_azimuth, n_elements]
%      e_theta_im     | Imaginary part of e-theta field component        | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_re       | Real part of e-phi field component               | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_im       | Imaginary part of e-phi field component          | Size: [n_elevation, n_azimuth, n_elements]
%      azimuth_grid   | Azimuth angles in [rad] -pi to pi, sorted        | Size: [n_azimuth]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: [n_elevation]
%      element_pos    | Antenna element (x,y,z) positions, optional      | Size: [3, n_elements] or []
%    
%    - x_deg
%      The rotation angle around x-axis (bank angle) in [degrees]
%    
%    - y_deg
%      The rotation angle around y-axis (tilt angle) in [degrees]
%    
%    - z_deg
%      The rotation angle around z-axis (heading angle) in [degrees]
%    
%    - usage
%      The optional parameter 'usage' can limit the rotation procedure either to the pattern or polarization.
%      usage = 0 | Rotate both, pattern and polarization, adjusts sampling grid (default)
%      usage = 1 | Rotate only pattern, adjusts sampling grid
%      usage = 2 | Rotate only polarization
%      usage = 3 | Rotate both, but do not adjust the sampling grid
%    
% Output Arguments:
%    - Antenna data of the rotated antenna: (outputs 1-7, single or double)
%      e_theta_re_r     | Real part of e-theta field component             | Size: [n_elevation_r, n_azimuth_r, n_elements]
%      e_theta_im_r     | Imaginary part of e-theta field component        | Size: [n_elevation_r, n_azimuth_r, n_elements]
%      e_phi_re_r       | Real part of e-phi field component               | Size: [n_elevation_r, n_azimuth_r, n_elements]
%      e_phi_im_r       | Imaginary part of e-phi field component          | Size: [n_elevation_r, n_azimuth_r, n_elements]
%      azimuth_grid_r   | Azimuth angles in [rad] -pi to pi, sorted        | Size: [n_azimuth_r]
%      elevation_grid_r | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: [n_elevation_r]
%      element_pos_r    | Antenna element (x,y,z) positions, optional      | Size: [3, n_elements]
%
%
% quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
% Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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
    