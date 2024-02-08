% RAY_MESH_INTERACT
%    Calculates interactions (reflection, transmission, refraction) of radio waves with objects
%    
% Description:
%    - Radio waves that interact with a building, or other objects in the environment, will produce losses
%      that depend on the electrical properties of the materials and material structure.
%    - This function calculates the interactions of radio-waves with objects in a 3D environment.
%    - It considers a plane wave incident upon a planar interface between two homogeneous and isotropic
%      media of differing electric properties. The media extend sufficiently far from the interface such
%      that the effect of any other interface is negligible.
%    - Air to media transition is assumed if front side of a face is hit and FBS != SBS
%    - Media to air transition is assumed if back side of a face is hit and FBS != SBS
%    - Media to media transition is assumed if FBS = SBS with opposing face orientations
%    - Order of the vertices determines side (front or back) of a mesh element
%    - Overlapping geometry in the triangle mesh must be avoided, since materials are transparent to radio
%      waves.
%    - Implementation is done according to ITU-R P.2040-1.
%    - Rays that do not interact with the environment (i.e. for which fbs_ind = 0) are omitted from
%      the output.
%    
% Material properties:
%    Each material is defined by its electrical properties. Radio waves that interact with a building will
%    produce losses that depend on the electrical properties of the building materials, the material
%    structure and the frequency of the radio wave. The fundamental quantities of interest are the electrical
%    permittivity (ϵ) and the conductivity (σ). The five parameters returned in mtl_prop then are:
%    
%    - Real part of relative permittivity at f = 1 GHz (a)
%    - Frequency dependence of rel. permittivity (b) such that ϵ = a · f^b
%    - Conductivity at f = 1 GHz (c)
%    - Frequency dependence of conductivity (d) such that σ = c· f^d
%    - Fixed attenuation in dB applied to each transition
%    
% Usage:
%    
%    [ origN, destN, gainN, xprmatN, trivecN, tridirN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, normal_vecN] = ...
%        quadriga_lib.ray_mesh_interact(  interaction_type, center_freq, orig, dest, fbs, sbs, mesh, mtl_prop,
%        fbs_ind, sbs_ind, trivec, tridir, orig_length );
%    
% Input Arguments:
%    - interaction_type
%      Interaction type: (0) Reflection, (1) Transmission, (2) Refraction
%    
%    - center_freq
%      Center frequency in [Hz]; Scalar value
%    
%    - orig
%      Ray origins in 3D Cartesian coordinates; Size: [ no_ray, 3 ]
%    
%    - dest
%      Ray destinations in 3D Cartesian coordinates; Size: [ no_ray, 3 ]
%    
%    - fbs
%      First interaction point between the rays and the triangular mesh. Size: [ no_ray, 3 ]
%    
%    - sbs
%      Second interaction point between the rays and the triangular mesh. Size: [ no_ray, 3 ]
%    
%    - mesh
%      Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3 points
%      in 3D-space. Hence, a face has 9 values in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ];
%      Size: [ no_mesh, 9 ]
%    
%    - mtl_prop
%      Material properties of each mesh element (see above); Size: [ no_mesh, 5 ]
%    
%    - fbs_ind
%      Index of the triangle that was hit by the ray at the FBS location; 1-based; Length: [ no_ray ]
%    
%    - sbs_ind
%      Index of the triangle that was hit by the ray at the SBS location; 1-based; Length: [ no_ray ]
%    
%    - trivec (optional)
%      The 3 vectors pointing from the center point of the ray at the origin to the vertices of a triangular
%      propagation tube, the values are in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ];
%      Size: [ no_ray, 9 ]
%    
%    - tridir (optional)
%      The directions of the vertex-rays in geographic coordinates (azimuth and elevation angle in rad);
%      the values are in the order [ v1az, v1el, v2az, v2el, v3az, v3el ]; Size: [ no_ray, 6 ]
%    
%    - orig_length (optional)
%      Path length at origin point, default is 0, Size: [ n_ray ]
%    
% Output Arguments:
%    - origN
%      New ray origins after the interaction with the medium, usually placed close to the FBS location.
%      A small offset of 0.001 m in the direction of travel after the interaction with the medium is added
%      to avoid getting stuct inside a mesh element. Size: [ no_rayN, 3 ]
%    
%    - destN
%      New ray destinaion after the interaction with the medium, taking the change of direction into account;
%      Size: [ no_rayN, 3 ]
%    
%    - gainN
%      Gain (negative loss) caused by the interaction with the medium, averaged over both polarization
%      directions. This value includes the in-medium attenuation, but does not account for FSPL. Linear scale.
%      Size: [ no_rayN ]
%    
%    - xprmatN
%      Polarization transfer matrix;
%      Interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH);
%      The values account for the following effects: (1) gain caused by the interaction with the medium,
%      (2) different reflection/transmission coefficients for transverse electric (TE) and transverse
%      magnetic (TM) polarisation, (3) orientation of the incidence plane, (4) in-medium attenuation.
%      FSPL is excluded.  Size: [ no_rayN, 8 ]
%    
%    - trivecN
%      The 3 vectors pointing from the center point of the ray at origN to the vertices of a triangular
%      propagation tube.  Size: [ no_rayN, 9 ]
%    
%    - tridirN
%      The directions of the vertex-rays after interaction with the medium in geographic coordinates
%      (azimuth and elevation angle in rad); Size: [ no_rayN, 6 ]
%    
%    - orig_lengthN
%      Length of the ray from orig to origN. If orig_length is given as input, its value is added.
%      Size: [ no_rayN ]
%    
%    - fbs_angleN
%      Angle between incoming ray and FBS in [rad], Size [ n_rayN ]
%    
%    - thicknessN
%      Material thickness in meters calculated from the difference between FBS and SBS, Size [ n_rayN ]
%    
%    - edge_lengthN
%      Max. edge length of the ray tube triangle at the new origin. A value of infinity indicates that only
%      a part of the ray tube hits the object. Size [ n_rayN, 3 ]
%    
%    - normal_vecN
%      Normal vector of FBS and SBS, Size [ n_rayN, 6 ]
%    
%    - out_typeN
%      A numeric indicator describing the type of the interaction. The total refection indicator is only 
%      set in refraction mode.
%      No | θF<0 | θS<0 | dFS=0 | TotRef | iSBS=0 | NF=-NS | NF=NS | startIn | endIn | Meaning
%      ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
%       0 |      |      |       |        |        |        |       |         |       | Undefined
%       1 |   no |  N/A |    no |    N/A |    yes |    N/A |   N/A |      no |   yes | Single Hit o-i
%       2 |  yes |  N/A |    no |     no |    yes |    N/A |   N/A |     yes |    no | Single Hit i-o
%       3 |  yes |  N/A |    no |    yes |    yes |    N/A |   N/A |     yes |    no | Single Hit i-o, TR
%      ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
%       4 |   no |  yes |   yes |     no |     no |    yes |    no |     yes |   yes | M2M, M2 hit first
%       5 |  yes |   no |   yes |     no |     no |    yes |    no |     yes |   yes | M2M, M1 hit first
%       6 |  yes |   no |   yes |    yes |     no |    yes |    no |     yes |   yes | M2M, M1 hit first, TR
%      ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
%       7 |   no |   no |   yes |    N/A |     no |     no |   yes |      no |   yes | Overlapping Faces, o-i
%       8 |  yes |  yes |   yes |     no |     no |     no |   yes |     yes |    no | Overlapping Faces, i-o
%       9 |  yes |  yes |   yes |    yes |     no |     no |   yes |     yes |    no | Overlapping Faces, i-o, TR
%      ---| -----|------|-------|--------|--------|--------|-------|---------|-------|----------------------------
%      10 |   no |  yes |   yes |    N/A |     no |     no |    no |      no |    no | Edge Hit, o-i-o
%      11 |  yes |   no |   yes |     no |     no |     no |    no |     yes |   yes | Edge Hit, i-o-i
%      12 |  yes |   no |   yes |    yes |     no |     no |    no |     yes |   yes | Edge Hit, i-o-i, TR
%      13 |   no |   no |   yes |    N/A |     no |     no |    no |      no |   yes | Edge Hit, o-i
%      14 |  yes |  yes |   yes |     no |     no |     no |    no |     yes |    no | Edge Hit, i-o
%      15 |  yes |  yes |   yes |    yes |     no |     no |    no |     yes |    no | Edge Hit, i-o, TR
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
    