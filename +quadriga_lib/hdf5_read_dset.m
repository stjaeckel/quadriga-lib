% HDF5_READ_DSET
%    Read a single unstructured dataset from an HDF5 file
%    
% Description:
%    Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition 
%    to structured datasets, the library facilitates the inclusion of extra datasets of various types 
%    and shapes. This feature is particularly beneficial for integrating descriptive data or analysis 
%    results. The function quadriga_lib.hdf5_read_dset retrieves a single unstructured dataset. The
%    output type of the function is defined by the datatype in the file. An empty matrix is returned 
%    if the dataset does not exist in the file.
%    
% Usage:
%    
%    dset = quadriga_lib.hdf5_read_dset_names( fn, location, name );
%    
% Input Arguments:
%    - fn
%      Filename of the HDF5 file, string
%    
%    - location (optional)
%      Storage location inside the file; 1-based; vector with 1-4 elements, i.e. [ix], [ix, iy], 
%      [ix,iy,iz] or [ix,iy,iz,iw]; Default: ix = iy = iz = iw = 1
%    
%    - name
%      Name of the dataset; String
%    
% Output Argument:
%    - dset
%      Output data. Type and size is defined by the dataspace in the file
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
    