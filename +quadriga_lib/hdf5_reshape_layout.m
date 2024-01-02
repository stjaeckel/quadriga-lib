% HDF5_RESHAPE_LAYOUT
%    Reshapes the storage layout inside an existing HDF5 file
%    
% Description:
%    Quadriga-Lib provides an HDF5-based solution for the storage and organization of channel data. A 
%    notable feature of this library is its capacity to manage multiple channels within a single HDF5 
%    file. In this framework, channels can be arranged in a multi-dimensional array format.
%    Once an HDF5 file has been created, the number of channels in the storage layout is fixed. 
%    However, it is possible to reshape the layout using quadriga_lib.hdf5_reshape_layout. 
%    
% Usage:
%    
%    quadriga_lib.hdf5_reshape_layout( fn, storage_dims );
%    
% Input Arguments:
%    - fn
%      Filename of the HDF5 file, string
%    
%    - storage_dims
%      Size of the dimensions of the storage space, vector with 1-4 elements, i.e. [nx], [nx, ny], 
%      [nx,ny,nz] or [nx,ny,nz,nw]. By default, nx = 65536, ny = 1, nz = 1, nw = 1
%      An error is thrown if the number of elements in the file is different from the given size.
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
    