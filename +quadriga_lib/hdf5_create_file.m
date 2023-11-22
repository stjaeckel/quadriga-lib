% HDF5_CREATE_FILE
%    Create a new HDF5 channel file with a custom storage layout
%    
% Description:
%    Quadriga-Lib offers an HDF5-based method for storing and managing channel data. A key feature of this
%    library is its ability to organize multiple channels within a single HDF5 file while enabling access
%    to individual data sets without the need to read the entire file. In this system, channels can be
%    structured in a multi-dimensional array. For instance, the first dimension might represent the Base
%    Station (BS), the second the User Equipment (UE), and the third the frequency. However, it is important
%    to note that the dimensions of the storage layout must be defined when the file is initially created
%    and cannot be altered thereafter. The function quadriga_lib.hdf5_create_file is used to create an
%    empty file with a predetermined custom storage layout.
%    
% Usage:
%    
%    quadriga_lib.hdf5_create_file( fn, storage_dims );
%    
% Input Arguments:
%    - fn
%      Filename of the HDF5 file, string
%    
%    - storage_dims (optional)
%      Size of the dimensions of the storage space, vector with 1-4 elements, i.e. [nx], [nx, ny], 
%      [nx,ny,nz] or [nx,ny,nz,nw]. By default, nx = 65536, ny = 1, nz = 1, nw = 1
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
    