% ARRAYANT_GENERATE
%    Generates predefined array antenna models
%    
% Description:
%    This functions can be used to generate a variety of pre-defined array antenna models, including 3GPP
%    array antennas used for 5G-NR simulations. The first argument is the array type. The following input
%    arguments are then specific to this type.
%    
% Usage:
%    
%    % Isotropic radiator, vertical polarization, 1 deg resolution
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('omni');
%    
%    % Short dipole radiating with vertical polarization, 1 deg resolution
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('dipole');
%    
%    % Half-wave dipole radiating with vertical polarization, 1 deg resolution
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('half-wave-dipole');
%    
%    % Cross-polarized isotropic radiator, 1 deg resolution
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('xpol');
%    
%    % An antenna with a custom 3dB beam with (in degree)
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, center_frequency, name] = ...
%        quadriga_lib.arrayant_generate('custom', az_3dB, el_3db, rear_gain_lin );
%    
%    % Antenna model for the 3GPP-NR channel model
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, center_frequency, name] = ...
%        quadriga_lib.arrayant_generate('3GPP', M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh );
%    
%    % Antenna model for the 3GPP-NR channel model with a custom pattern
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, center_frequency, name] = ...
%        quadriga_lib.arrayant_generate('3GPP', M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, ...
%        e_theta_re_c, e_theta_im_c, e_phi_re_c, e_phi_im_c, azimuth_grid_c, elevation_grid_c );
%    
% Input Arguments for type 'custom':
%    
%    - az_3dB
%      3dB beam width in azimuth direction in [deg], scalar
%    
%    - el_3db
%      3dB beam width in elevation direction in [deg], scalar
%    
%    - rear_gain_lin
%      Isotropic gain (linear scale) at the back of the antenna, scalar
%    
% Input Arguments for type '3GPP':
%    
%    - M
%      Number of vertically stacked elements, scalar, default = 1
%    
%    - N
%      Number of horizontally stacked elements, scalar, default = 1
%    
%    - center_freq
%      The center frequency in [Hz], scalar, default = 299792458 Hz
%    
%    - pol
%      Polarization indicator to be applied for each of the NM elements:
%      pol = 1 | vertical polarization (default value)
%      pol = 2 | H/V polarized elements, results in 2NM elements
%      pol = 3 | +/-45° polarized elements, results in 2NM elements
%      pol = 4 | vertical polarization, combines elements in vertical direction, results in N elements
%      pol = 5 | H/V polarization, combines elements in vertical direction, results in 2N elements
%      pol = 6 | +/-45° polarization, combines elements in vertical direction, results in 2N elements
%      Polarization indicator is ignored when a custom pattern is provided.
%    
%    - tilt
%      The electric downtilt angle in [deg], Only relevant for pol = 4/5/6, scalar, default = 0
%    
%    - spacing
%      Element spacing in [λ], scalar, default = 0.5
%      
%    - Mg
%      Number of nested panels in a column, scalar, default = 1
%      
%    - Ng
%      Number of nested panels in a row, scalar, default = 1
%    
%    - dgv
%      Panel spacing in vertical direction in [λ], scalar, default = 0.5
%    
%    - dgh
%      Panel spacing in vertical horizontal in [λ], scalar, default = 0.5
%    
%    - e_theta_re_c (optional)
%      Real part of the e-theta component of the custom pattern, double precision. 
%      Size: [n_elevation, n_azimuth, n_elements_c]
%      
%    - e_theta_im_c (optional)
%      Imaginary part of the e-theta component of the custom pattern, double precision. 
%      Size: [n_elevation, n_azimuth, n_elements_c]
%      
%    - e_phi_re_c (optional)
%      Real part of the e-phi component of the custom pattern, double precision. 
%      Size: [n_elevation, n_azimuth, n_elements_c]
%      
%    - e_phi_im_c (optional)
%      Imaginary part of the e-phi component of the custom pattern, double precision. 
%      Size: [n_elevation, n_azimuth, n_elements_c]
%      
%    - azimuth_grid_c (optional)
%      Azimuth angles (theta) in [rad], double precision, Size: [n_azimuth]
%      
%    - elevation_grid_c (optional)
%      Elevation angles (phi) in [rad], double precision, Size: [n_elevation]
%      
% Output Arguments:
%    
%    - e_theta_re
%      Real part of the e-theta component (vertical component) of the far field of each antenna element in 
%      the array antenna, double precision, Size: [n_elevation, n_azimuth, n_elements]
%      
%    - e_theta_im
%      Imaginary part of the e-theta component of the electric field, double precision, 
%      Size: [n_elevation, n_azimuth, n_elements]
%      
%    - e_phi_re
%      Real part of the e-phi component (horizontal component) of the far field of each antenna element in 
%      the array antenna, double precision, Size: [n_elevation, n_azimuth, n_elements]
%      
%    - e_phi_im
%      Imaginary part of the e-phi component of the electric field, double precision, 
%      Size: [n_elevation, n_azimuth, n_elements]
%      
%    - azimuth_grid
%      Azimuth angles (theta) in [rad] were samples of the field patterns are provided. Values are between 
%      -pi and pi, sorted in ascending order, double precision, Size: [n_azimuth]
%      
%    - elevation_grid
%      Elevation angles (phi) in [rad] where samples of the field patterns are provided. Values are between 
%      -pi/2 and pi/2, sorted in ascending order, double precision, Size: [n_elevation]
%    
%    - element_pos
%      Antenna element (x,y,z) positions relative to the array's phase-center in units of [m].
%      Size: [3, n_elements]
%    
%    - coupling_re
%      Real part of the array antenna coupling matrix. This matrix describes a pre- or post-processing
%      of the signals that are fed to or received by the antenna elements. The rows in the matrix
%      correspond to the antenna elements, the columns to the signal ports. 
%      Size: [n_elements, n_ports]
%    
%    - coupling_im
%      Real part of the array antenna coupling matrix, Size: [n_elements, n_ports]
%    
%    - center_frequency
%      Center frequency in [Hz] (optional). Default value is 299,792,458 Hz, Scalar
%    
%    - name
%      Name of the array antenna, string
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
    