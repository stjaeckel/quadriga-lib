%ARRAYANT_CALC_DIRECTIVITY Calculates the directivity (in dBi) of array antenna elements
%
% Description:
%   Directivity is a parameter of an antenna or which measures the degree to which the radiation
%   emitted is concentrated in a single direction. It is the ratio of the radiation intensity in a
%   given direction from the antenna to the radiation intensity averaged over all directions.
%   Therefore, the directivity of a hypothetical isotropic radiator is 1, or 0 dBi. [Wikipedia]
%
% Inputs:
%   e_theta_re
%   Real part of the e-theta component (vertical component) of the far field of each antenna element
%   in the array antenna. Single or double precision, Size: [n_elevation, n_azimuth, n_elements]
%
%   e_theta_im
%   Imaginary part of the e-theta component of the electric field.
%   Single or double precision, Size: [n_elevation, n_azimuth, n_elements]
%
%   e_phi_re
%   Real part of the e-phi component (horizontal component) of the far field of each antenna element
%   in the array antenna. Single or double precision, Size: [n_elevation, n_azimuth, n_elements]
%
%   e_phi_im
%   Imaginary part of the e-phi component of the electric field.
%   Single or double precision, Size: [n_elevation, n_azimuth, n_elements]
%
%   azimuth_grid
%   Azimuth angles (theta) in [rad] were samples of the field patterns are provided. Values must be
%   between -pi and pi, sorted in ascending order. Single or double precision, Size: [n_azimuth]
%
%   elevation_grid
%   Elevation angles (phi) in [rad] where samples of the field patterns are provided. Values must be
%   between -pi/2 and pi/2, sorted in ascending order. Single or double precision, Size: [n_elevation]
%
%   i_element (optional)
%   Element index, Size: [n_out]
%
% Output:
%   directivity
%   Directivity of the antenna pattern in dBi, double precision, Size: [n_out]
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