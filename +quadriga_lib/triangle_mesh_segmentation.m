% TRIANGLE_MESH_SEGMENTATION
%    Rearranges elements of a triangle mesh into smaller sub-meshes
%    
% Description:
%    This function processes the elements of a large triangle mesh by clustering those that are 
%    closely spaced. The resulting mesh retains the same elements but rearranges their order. 
%    The function aims to minimize the size of the axis-aligned bounding box around each cluster, 
%    referred to as a sub-mesh, while striving to maintain a specific number of elements within 
%    each cluster.
%    
%    This approach is particularly useful in computer graphics and simulation applications where 
%    managing computational resources efficiently is crucial. By organizing the mesh elements into 
%    compact clusters, the function enhances rendering performance and accelerates computational 
%    tasks, such as collision detection and physics simulations. It allows for quicker processing 
%    and reduced memory usage, making it an essential technique in both real-time graphics rendering 
%    and complex simulation environments.
%    
% Usage:
%    
%    [ triangles_out, sub_mesh_index, mesh_index ] = quadriga_lib.triangle_mesh_segmentation( ...
%        triangles_in, target_size, vec_size );
%    
%    [ triangles_out, sub_mesh_index, mesh_index, mtl_prop_out ] = ...
%         quadriga_lib.triangle_mesh_segmentation( triangles_in, target_size, vec_size, mtl_prop_in );
%    
%    
% Input Arguments:
%    
%    - triangles_in
%      Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
%      points in 3D-space: [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; single or double precision;
%      Size: [ n_triangles_in, 9 ]
%    
%    - target_size (optional)
%      The target number of elements of each sub-mesh. Default value = 1024. For best performance, the 
%      value should be around sgrt( n_triangles_in )
%    
%    - vec_size (optional)
%      Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1. 
%      For values > 1,the number of rows for each sub-mesh in the output is increased to a multiple 
%      of vec_size. For padding, zero-sized triangles are placed at the center of the AABB of 
%      the corresponding sub-mesh.
%      
%    - mtl_prop_in (optional)
%      Material properties of each mesh element; Size: [ n_triangles_in, 5 ]
%    
% Output Arguments:
%    
%    - triangles_out
%      Vertices of the clustered mesh in global Cartesian coordinates; singe or double precision;
%      Size: [ n_triangles_out, 9 ]
%    
%    - sub_mesh_index
%      Start indices of the sub-meshes in 0-based notation. Type: uint32; Vector of length [ n_sub_mesh ]
%    
%    - mesh_index
%      Indices for mapping elements of "triangles_in" to "triangles_out"; 1-based; 
%      Length: [ n_mesh_out ]; For vec_size > 1, the added elements not contained in the input
%      are indicated by zeros.
%    
%    - mtl_prop_out
%      Material properties for the sub-divided triangle mesh elements. The values for the new faces are 
%      copied from mtl_prop_in; Size: [ n_triangles_out, 5 ]; For vec_size > 1, the added elements
%      will contain the vacuum / air material.
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
    