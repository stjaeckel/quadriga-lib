% ARRAYANT_INTERPOLATE
%    Interpolate array antenna field patterns
%    
% Description:
%    This function interpolates polarimetric antenna field patterns for a given set of azimuth and
%    elevation angles.
%    
% Usage:
%    
%    [V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma] = ...
%        quadriga_lib.arrayant_interpolate( e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
%        azimuth_grid, elevation_grid, azimuth, elevation, i_element, orientation, element_pos )
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
%    - azimuth
%      Azimuth angles in [rad] for which the field pattern should be interpolated. Values must be
%      between -pi and pi, single or double precision.
%      Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
%                 | Size: [1, n_ang]
%      Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
%                 | Size: [n_out, n_ang]
%      
%    - elevation
%      Elevation angles in [rad] for which the field pattern should be interpolated. Values must be
%      between -pi/2 and pi/2, single or double precision.
%      Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
%                 | Size: [1, n_ang]
%      Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
%                 | Size: [n_out, n_ang]
%      
%    - i_element
%      The element indices for which the interpolation should be done. Optional parameter. Values must
%      be between 1 and n_elements. It is possible to duplicate elements, i.e. by passing [1,1,2].
%      If this parameter is not provided (or an empty array is passed), i_element is initialized
%      to [1:n_elements]. In this case, n_out = n_elements. Allowed types: uint32 or double.
%      Size: [1, n_out] or [n_out, 1] or empty []
%      
%    - orientation
%      This (optional) 3-element vector describes the orientation of the array antenna or of individual
%      array elements. The The first value describes the ”bank angle”, the second value describes the
%      ”tilt angle”, (positive values point upwards), the third value describes the bearing or ”heading
%      angle”, in mathematic sense. Values must be given in [rad]. East corresponds to 0, and the
%      angles increase counter-clockwise, so north is pi/2, south is -pi/2, and west is equal to pi. By
%      default, the orientation is [0,0,0]', i.e. the broadside of the antenna points at the horizon
%      towards the East. Single or double precision
%      Size: [3, 1] or [3, n_out] or [3, 1, n_ang] or [3, n_out, n_ang] or empty []
%      
%    - element_pos
%      Positions of the array antenna elements in local cartesian coordinates (using units of [m]).
%      Optional parameter. If this parameter is not given, all elements are placed at the phase center
%      of the array at coordinates [0,0,0]'. Otherwise, positions are given for the elements in the
%      output of the interpolation function. For example, when duplicating the fist element by setting
%      i_element = [1,1], different element positions can be set for the two elements in the output.
%      Single or double precision, Size: [3, n_out] or empty []
%      
% Derived inputs:
%      n_azimuth      | Number of azimuth angles in the filed pattern 
%      n_elevation    | Number of elevation angles in the filed pattern 
%      n_elements     | Number of antenna elements filed pattern of the array antenna
%      n_ang          | Number of interpolation angles
%      n_out          | Number of antenna elements in the generated output (may differ from n_elements)
%    
% Output Arguments:
%    - V_re
%      Real part of the interpolated e-theta (vertical) field component.
%      Single or double precision (same as input), Size [n_out, n_ang]
%      
%    - V_im
%      Imaginary part of the interpolated e-theta (vertical) field component.
%      Single or double precision (same as input), Size [n_out, n_ang]
%      
%    - H_re
%      Real part of the interpolated e-phi (horizontal) field component.
%      Single or double precision (same as input), Size [n_out, n_ang]
%      
%    - H_im
%      Imaginary part of the interpolated e-phi (horizontal) field component.
%      Single or double precision (same as input), Size [n_out, n_ang]
%      
%    - dist
%      The effective distances between the antenna elements when seen from the direction of the
%      incident path. The distance is calculated by an projection of the array positions on the normal
%      plane of the incident path. This is needed for calculating the phase of the antenna response.
%      Single or double precision (same as input), Size [n_out, n_ang]
%      
%    - azimuth_loc
%      The azimuth angles in [rad] for the local antenna coordinate system, i.e., after applying the
%      'orientation'. If no orientation vector is given, these angles are identical to the input
%      azimuth angles. Optional output.
%      Single or double precision (same as input), Size [n_out, n_ang]
%      
%    - elevation_loc
%      The elevation angles in [rad] for the local antenna coordinate system, i.e., after applying the
%      'orientation'. If no orientation vector is given, these angles are identical to the input
%      elevation angles. Optional output.
%      Single or double precision (same as input), Size [n_out, n_ang]
%      
%    - gamma
%      Polarization rotation angles in [rad]
%      Single or double precision (same as input), Size [n_out, n_ang]
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
    