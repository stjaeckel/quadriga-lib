% RAY_TRIANGLE_INTERSECT
%    Calculates the intersection of rays and triangles in three dimensions
%    
% Description:
%    - This function implements the Möller–Trumbore ray-triangle intersection algorithm, known for its
%      efficiency in calculating the intersection of a ray and a triangle in three-dimensional space.
%      This method achieves its speed by eliminating the need for precomputed plane equations of the plane
%      containing the triangle.
%    
%    - For further information, refer to [Wikipedia: Möller–Trumbore intersection algorithm].
%    
%    - The algorithm defines the ray using two points: an origin and a destination. Similarly, the triangle
%      is specified by its three vertices. 
%      
%    - To enhance performance, this implementation leverages AVX2 intrinsic functions and OpenMP, when 
%      available, to speed up the computational process.
%    
% Usage:
%    
%    [ fbs, sbs, no_interact, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( orig, dest, mesh );
%    
% Input Arguments:
%    - orig
%      Ray origins in 3D Cartesian coordinates; Size: [ no_ray, 3 ]
%    
%    - dest
%      Ray destinations in 3D Cartesian coordinates; Size: [ no_ray, 3 ]
%    
%    - mesh
%      Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3 points
%      in 3D-space. Hence, a face has 9 values in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; 
%      Size: [ no_mesh, 9 ]
%    
% Output Arguments:
%    - fbs
%      First interaction point between the rays and the triangular mesh. If no interaction was found, the
%      FBS location is equal to dest. Size: [ no_ray, 3 ]
%    
%    - sbs
%      Second interaction point between the rays and the triangular mesh. If no interaction was found, the
%      SBS location is equal to dest. Size: [ no_ray, 3 ]
%    
%    - no_interact
%      Total number of interactions between the origin point and the destination; uint32; Length: [ no_ray ]
%    
%    - fbs_ind
%      Index of the triangle that was hit by the ray at the FBS location; 1-based; uint32; Length: [ no_ray ]
%    
%    - sbs_ind
%      Index of the triangle that was hit by the ray at the SBS location; 1-based; uint32; Length: [ no_ray ]
%    
% Caveat:
%    - orig, dest, and mesh can be provided in single or double precision; fbs and lbs will have
%      the same type.
%    - All internal computation are done in single precision to achieve an additional 2x improvement in 
%      speed compared to double precision when using AVX2 intrinsic instructions
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
    