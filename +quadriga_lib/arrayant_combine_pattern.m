% ARRAYANT_COMBINE_PATTERN
%    Calculate effective radiation patterns for array antennas
%    
% Description:
%    An array antenna consists of multiple individual elements. Each element occupies a specific position 
%    relative to the array's phase-center, its local origin. Elements can also be inter-coupled, 
%    represented by a coupling matrix. By integrating the element radiation patterns, their positions, 
%    and the coupling weights, one can determine an effective radiation pattern observable by a receiver 
%    in the antenna's far field. Leveraging these effective patterns is especially beneficial in antenna 
%    design, beamforming applications such as in 5G systems, and in planning wireless communication 
%    networks in complex environments like urban areas. This streamlined approach offers a significant 
%    boost in computation speed when calculating MIMO channel coefficients, as it reduces the number of 
%    necessary operations. The function arrayant_combine_pattern is designed to compute these effective 
%    radiation patterns.
%    
% Usage:
%    
%    [ e_theta_re_c, e_theta_im_c, e_phi_re_c, e_phi_im_c] = quadriga_lib.arrayant_combine_pattern( ...
%        e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, 
%        coupling_re, coupling_im, center_freq);
%    
% Examples:
%    
%    The following example creates a unified linear array of 4 dipoles, spaced at half-wavelength. The
%    elements are then coupled with each other (i.e., they receive the same signal). The effective pattern
%    is calculated using arrayant_combine_pattern.
%    
%    % Generate dipole pattern
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos] = ...
%        quadriga_lib.arrayant_generate('dipole');
%    
%    % Duplicate 4 times
%    e_theta_re  = repmat(e_theta_re, [1,1,4]);
%    e_theta_im  = repmat(e_theta_im, [1,1,4]);
%    e_phi_re    = repmat(e_phi_re, [1,1,4]);
%    e_phi_im    = repmat(e_phi_im, [1,1,4]);
%    element_pos = repmat(element_pos, [1,4]);
%    
%    % Set element positions and coupling matrix
%    element_pos(2,:) = [ -0.75, -0.25, 0.25, 0.75];  % lambda, along y-axis
%    coupling_re = [ 1 ; 1 ; 1 ; 1 ]/sqrt(4);
%    
%    % Calculate effective pattern
%    [ e_theta_re_c, e_theta_im_c, e_phi_re_c, e_phi_im_c] = quadriga_lib.arrayant_combine_pattern( ...
%        e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re);
%    
%    % Plot gain
%    plot( azimuth_grid180/pi, [ 10log10( e_theta_re(91,:,1).^2 ); 10log10( e_theta_re_c(91,:).^2 ) ]);
%    axis([-180 180 -20 15]); ylabel('Gain (dBi)'); xlabel('Azimth angle (deg)'); legend('Dipole','Array')
%    
%    
% Input Arguments:
%    - Antenna data: (inputs 1-10, single or double)
%      e_theta_re     | Real part of e-theta field component                  | Size: [n_elevation, n_azimuth, n_elements]
%      e_theta_im     | Imaginary part of e-theta field component             | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_re       | Real part of e-phi field component                    | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_im       | Imaginary part of e-phi field component               | Size: [n_elevation, n_azimuth, n_elements]
%      azimuth_grid   | Azimuth angles in [rad] -pi to pi, sorted             | Size: [n_azimuth]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation]
%      element_pos    | Antenna element (x,y,z) positions, optional           | Size: [3, n_elements] or []
%      coupling_re    | Real part of coupling matrix, optional                | Size: [n_elements, n_ports] or []
%      coupling_im    | Imaginary part of coupling matrix, optional           | Size: [n_elements, n_ports] or []
%      center_freq    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
%    
% Output Arguments:
%    - e_theta_re_c
%      Real part of the e-theta component (vertical component) of the effective array antenna.
%      Size: [n_elevation, n_azimuth, n_ports]
%    
%    - e_theta_im_c
%      Imaginary part of the e-theta component (vertical component) of the effective array antenna.
%      Size: [n_elevation, n_azimuth, n_ports]
%    
%    - e_phi_re_c
%      Real part of the e-phi component (horizontal component) of the effective array antenna.
%      Size: [n_elevation, n_azimuth, n_ports]
%    
%    - e_phi_im_c
%      Imaginary part of the e-phi component (horizontal component) of the effective array antenna.
%      Size: [n_elevation, n_azimuth, n_ports]
%    
% Caveat:
%    The effective antenna has all elements at the phase center [0,0,0]' and has perfect isolation
%    between its elements. Hence, no outputs for the effective element_pos and coupling are needed.
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
    