% INTERP
%    2D and 1D linear interpolation. 
%    
% Description:
%    This function implements 2D and 1D linear interpolation. 
%    
% Usage:
%    
%    dataI = quadriga_lib.interp( x, y, data, xI, yI );      % 2D case
%    
%    dataI = quadriga_lib.interp( x, [], data, xI );         % 1D case
%    
% Input Arguments:
%    - x
%      Vector of sample points in x direction for which data is provided; single or double; Length: [nx]
%    
%    - y
%      Vector of sample points in y direction for which data is provided; single or double; Length: [ny]
%      Must be an empty array [] in case of 1D interpolation.
%    
%    - data
%      The input data tensor; single or double; Size: [ny, nx, ne] or [1, nx, ne] for 1D case 
%      The 3rd dimension enables interpolation for mutiple datasets simultaneously. 
%    
%    - xI
%      Vector of sample points in x direction for which data should be interpolated; single or double; 
%      Length: [nxI]
%    
%    - yI
%      Vector of sample points in y direction for which data should be interpolated; single or double; 
%      Length: [nyI]
%    
% Output Arguments:
%    - dataI
%      The interpolated dat; single or double (same as data); 
%      Size: [nyI, nxI, ne] or [1, nxI, ne] for 1D case 
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
    