% ICOSPHERE
%    Construct a geodesic polyhedron (icosphere), a convex polyhedron made from triangles
%    
% Description:
%    An icosphere is constructed by subdividing faces of an icosahedron, a polyhedron with 20 faces,
%    12 vertices and 30 edges, and then projecting the new vertices onto the surface of a sphere. The
%    resulting mesh has 6 triangles at each vertex, except for 12 vertices which have 5 triangles.
%    The approximate equilateral triangles have roughly the same edge length and surface area.
%    
% Usage:
%    
%    [ center, length, vert, direction ] = quadriga_lib.icosphere( no_div, radius );
%    
% Input Arguments:
%    
%    - no_div
%      Number of divisions per edge of the generating icosahedron. The resulting number of faces is
%      equal to no_face = 20 · no_div^2
%    
%    - radius
%      Radius of the sphere in meters
%    
%    
% Output Arguments:
%    
%    - center
%      Position of the center point of each triangle; Size: [ no_face x 3 ]
%    
%    - length
%      Length of the vector pointing from the origin to the center point. This number is smaller than
%      1 since the triangles are located inside the unit sphere; Size: [ no_face x 1 ]
%    
%    - vert
%      The 3 vectors pointing from the center point to the vertices of each triangle; the values are
%      in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; Size: [ no_face x 9 ]
%    
%    - direction
%      The directions of the vertex-rays in geographic coordinates (azimuth and elevation angle in
%      rad); the values are in the order [ v1az, v1el, v2az, v2el, v3az, v3el ]; Size: [ no_face x 6 ]
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
    