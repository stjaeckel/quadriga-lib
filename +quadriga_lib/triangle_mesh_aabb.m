% TRIANGLE_MESH_AABB
%    Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes
%    
% Description:
%    The axis-aligned minimum bounding box (or AABB) for a given set of triangles is its minimum 
%    bounding box subject to the constraint that the edges of the box are parallel to the (Cartesian) 
%    coordinate axes. Axis-aligned bounding boxes are used as an approximate location of the set of 
%    triangles. In order to find intersections with the triangles (e.g. using ray tracing), the 
%    initial check is the intersections between the rays and the AABBs. Since it is usually a much 
%    less expensive operation than the check of the actual intersection (because it only requires 
%    comparisons of coordinates), it allows quickly excluding checks of the pairs that are far apart. 
%    
% Usage:
%    
%    aabb = quadriga_lib.subdivide_triangles( triangle_mesh, sub_mesh_index, vec_size );
%    
% Input Arguments:
%    
%    - triangles
%      Vertices of the triangle mesh in global Cartesian coordinates. Each face is described by 3
%      points in 3D-space: [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; single or double precision;
%      Size: [ n_triangles, 9 ]
%    
%    - sub_mesh_index (optional)
%      Start indices of the sub-meshes in 0-based notation. If this parameter is not given, the AABB of 
%      the entire triangle mesh is returned. Type: uint32; Vector of length [ n_sub_mesh ]
%    
%    - vec_size (optional)
%      Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1. For values > 1,
%      the number of rows in the output is increased to a multiple of vec_size, padded with zeros. 
%    
% Output Argument:
%    
%    - aabb
%      Axis-aligned bounding box of each sub-mesh. Each box is described by 6 values: 
%      [ x_min, x_max, y_min, y_max, z_min, z_max ]; Size: [ n_sub_mesh, 6 ]
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
    