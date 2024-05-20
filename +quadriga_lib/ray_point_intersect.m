% RAY_POINT_INTERSECT
%    Calculates the intersection of ray beams with points in three dimensions
%    
% Description:
%    Unlike traditional ray tracing, where rays do not have a physical size, beam tracing models rays as 
%    beams with volume. Beams are defined by triangles whose vertices diverge as the beam extends. This 
%    approach is used to simulate a kind of divergence or spread in the beam, reminiscent of how radio 
%    waves spreads as they travel from a point source. The volumetric nature of the beams allows for more 
%    realistic modeling of energy distribution. As beams widen, the energy they carry can be distributed 
%    across their cross-sectional area, affecting the intensity of the interaction with surfaces.
%    Unlike traditional ray tracing where intersections are line-to-geometry tests, beam tracing requires 
%    volumetric intersection tests.
%    
%    Ray beams are determined by an origin point, three vectors pointing from the origin to the three 
%    vertices of a triangle that defines the shape of the tube and the three direction of the rays at 
%    the vertices.
%    
% Usage:
%    
%    [ hit_count, ray_ind ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, ...
%        max_no_hit, sub_cloud_index, target_size );
%    
% Input Arguments:
%    - orig
%      Ray origins in 3D Cartesian coordinates; Size: [ no_ray, 3 ]
%    
%    - trivec
%      The 3 vectors pointing from the center point of the ray at the ray origin to the vertices of 
%      a triangular propagation tube (the beam), the values are in the order 
%      [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; Size: [ no_ray, 9 ]
%    
%    - tridir
%      The directions of the vertex-rays. Size: [ n_ray, 9 ], Values must be given in Cartesian 
%      coordinates in the order  [ d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z  ]; The vector does
%      not need to be normalized.
%    
%    - points
%      Points in 3D-Cartesian space; Size: [ n_points_in, 3 ]
%    
%    - max_no_hit (optional)
%      Max. number of hits in the output ray_ind, default = 32
%    
%    - sub_cloud_index (optional)
%      Start indices of the sub-clouds in 0-based notation. Type: uint32; Vector of length [ n_sub_cloud ] 
%      If this optional input is not given, the sub-could index is calculated automatically. Passing a
%      value of uint32(0) will disable the sub-cloud calculation. 
%    
%    - target_size (optional)
%      Target value for the sub-cloud size, only evaluated if 'sub_cloud_index' is not given.
%    
% Output Arguments:
%    
%    - hit_count
%      Number of rays that hit a point, unit32, Length: [ n_points ]
%    
%    - ray_ind
%      Ray indices that hit the points, 1-based, 0 = no hit, Size [ n_points, max_no_hit ]
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
    