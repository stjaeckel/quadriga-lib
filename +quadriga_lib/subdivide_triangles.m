% SUBDIVIDE_TRIANGLES
%    Subdivide the faces of a triangle mesh into smaller faces
%    
% Description:
%    This function splits the triangles of a mesh into smaller triangles by subdividing the edges
%    into n_div sub-edges. This creates n_div^2 sub-faces per face.
%    
% Usage:
%    
%    triangles_out = quadriga_lib.subdivide_triangles( triangles_in, no_div );
%    
% Input Arguments:
%    
%    - triangles_in
%      Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
%      points in 3D-space: [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; singe or double precision;
%      Size: [ n_triangles_in, 9 ]
%    
%    - no_div
%      Number of divisions per edge of the input mesh. The resulting number of faces is equal to
%      n_triangles_out = n_triangles_in  n_div^2
%    
% Output Argument:
%    
%    - triangles_out
%      Vertices of the sub-divided mesh in global Cartesian coordinates; singe or double precision;
%      Size: [ n_triangles_out, 9 ]
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
    