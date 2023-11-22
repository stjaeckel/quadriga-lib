% ARRAYANT_QDANT_READ
%    Reads array antenna data from QDANT files
%    
% Description:
%    The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna pattern
%    data in XML. This function reads pattern data from the specified file.
%    
% Usage:
%    
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re,
%       coupling_im, center_freq, name, layout ] = quadriga_lib.arrayant_qdant_read( fn, id, use_single );
%    
% Input Arguments:
%    - fn
%      Filename of the QDANT file, string
%    
%    - id (optional)
%      ID of the antenna to be read from the file, optional, Default: Read first
%    
%    - use_single (optional)
%      Indicator if results should be returned in single precision, default = 0, returned in double precision
%    
% Output Arguments:
%    - Antenna data: (outputs 1-11, single or double)
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
%      name           | Name of the array antenna object                      | String
%    
%    - layout
%      Layout of multiple array antennas. Contain element ids that are present in the file
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
    