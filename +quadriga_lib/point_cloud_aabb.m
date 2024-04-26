%    POINT_CLOUD_AABB
%    Calculate the axis-aligned bounding box (AABB) of set of points in 3D-sapce
%    
% Description:
%    The axis-aligned minimum bounding box (or AABB) for a given set points is its minimum bounding 
%    box subject to the constraint that the edges of the box are parallel to the (Cartesian)
%    coordinate axes. Axis-aligned bounding boxes are used as an approximate location of the set of
%    points.
%    
% Usage:
%    
%    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_index, vec_size );
%    
% Input Arguments:
%    
%    - points
%      Points in 3D-Cartesian space; Size: [ n_points, 3 ]
%    
%    - sub_cloud_index (optional)
%      Start indices of the sub-clouds in 0-based notation. If this parameter is not given, the AABB of
%      the entire point cloud is returned. Type: uint32; Vector of length [ n_sub_cloud ]
%    
%    - vec_size (optional)
%      Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1. For values > 1,
%      the number of rows in the output is increased to a multiple of vec_size, padded with zeros.
%    
% Output Argument:
%    
%    - aabb
%      Axis-aligned bounding box of each sub-cloud. Each box is described by 6 values:
%      [ x_min, x_max, y_min, y_max, z_min, z_max ]; Size: [ n_sub_cloud, 6 ]
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
    