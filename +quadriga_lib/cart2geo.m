CART2GEO Transform Cartesian coordinates to geographic coordinates
%
% Description:
%   [azimuth,elevation,length] = arrayant_lib.cart2geo(cart) 
%   transforms corresponding elements of the Cartesian coordinate system (x, y, and z) to geographic coordinates azimuth, elevation, and length.
%
% Input:
%   cart
%   Cartesian coordinates (x,y,z)
%   Single or double precision, Size: [3, n_row, n_col]
%
% Outputs:
%   azimuth
%   Azimuth angles in [rad], values between -pi and pi.
%   Single or double precision (same as input), Size [n_row, n_col]
%
%   elevation
%   Elevation angles in [rad], values between -pi/2 and pi/2.
%   Single or double precision (same as input), Size [n_row, n_col]
%
%   length
%   Vector length, i.e. the distance from the origin to the point defined by x,y,z.
%   Single or double precision (same as input), Size [n_row, n_col]
%
%
% arrayant-lib c++/MEX Array antenna support library
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
