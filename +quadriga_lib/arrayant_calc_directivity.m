% ARRAYANT_CALC_DIRECTIVITY
%    Calculates the directivity (in dBi) of array antenna elements
%    
% Description:
%    Directivity is a parameter of an antenna or which measures the degree to which the radiation emitted 
%    is concentrated in a single direction. It is the ratio of the radiation intensity in a given direction 
%    from the antenna to the radiation intensity averaged over all directions. Therefore, the directivity 
%    of a hypothetical isotropic radiator is 1, or 0 dBi.
%    
% Usage:
%    
%    directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
%        e_phi_im, azimuth_grid, elevation_grid);
%    
%    directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
%        e_phi_im, azimuth_grid, elevation_grid, i_element);
%    
% Examples:
%    
%    % Generate dipole antenna
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid] = ...
%        quadriga_lib.arrayant_generate('dipole');
%    
%    % Calculate directivity 
%    directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
%        e_phi_im, azimuth_grid, elevation_grid);
%    
% Input Arguments:
%    - Antenna data: (inputs 1-6, single or double precision)
%      e_theta_re     | Real part of e-theta field component                  | Size: [n_elevation, n_azimuth, n_elements]
%      e_theta_im     | Imaginary part of e-theta field component             | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_re       | Real part of e-phi field component                    | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_im       | Imaginary part of e-phi field component               | Size: [n_elevation, n_azimuth, n_elements]
%      azimuth_grid   | Azimuth angles in [rad] -pi to pi, sorted             | Size: [n_azimuth]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation]
%    
%    - i_element (optional) 
%      Element index, 1-based. If not provided or empty, the directivity is calculated for all elements in the 
%      array antenna. Size: [n_out] or empty
%    
% Output Argument:
%    - directivity
%      Directivity of the antenna pattern in dBi, double precision, Size: [n_out] or [n_elements]
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
    