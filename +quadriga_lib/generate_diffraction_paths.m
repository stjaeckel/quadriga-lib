% GENERATE_DIFFRACTION_PATHS
%    Generate propagation paths for estimating the diffraction gain
%    
% Description:
%    Diffraction refers to the phenomenon where waves bend or interfere around the edges of an obstacle,
%    extending into the region that would otherwise be in the obstacle's geometrical shadow. The object
%    causing the diffraction acts as a secondary source for the wave's propagation. A specific example of
%    this is the knife-edge effect, or knife-edge diffraction, where a sharp, well-defined obstacle—like
%    a mountain range or a building wall—partially truncates the incident radiation.
%    
%    To estimate the diffraction gain in a three-dimensional space, one can assess the extent to which the
%    Fresnel ellipsoid is obstructed by objects, and then evaluate the impact of this obstruction on the
%    received power. This method presupposes that diffracted waves travel along slightly varied paths
%    before arriving at a receiver. These waves may reach the receiver out of phase with the primary wave
%    due to their different travel lengths, leading to either constructive or destructive interference.
%    
%    The process of estimating the gain involves dividing the wave propagation from a transmitter to a
%    receiver into n_path paths. These paths are represented by elliptic arcs, which are further
%    approximated using n_seg line segments. Each segment can be individually blocked or attenuated
%    by environmental objects. To determine the overall diffraction gain, a weighted sum of these
%    individual path contributions is calculated. The weighting is adjusted to align with the uniform
%    theory of diffraction (UTD) coefficients in two dimensions, but the methodology is adapted for
%    any 3D object shape. This function generates the elliptic propagation paths and corresponding weights
%    necessary for this calculation.
%    
% Caveat:
%    - Each ellipsoid consists of n_path diffraction paths. The number of paths is determined by the
%      level of detail (lod).
%    - All diffraction paths of an ellipsoid originate at orig and arrive at dest
%    - Each diffraction path has n_seg segments
%    - Points orig and dest lay on the semi-major axis of the ellipsoid
%    - The generated rays sample the volume of the ellipsoid
%    - Weights are calculated from the Knife-edge diffraction model when parts of the ellipsoid are shadowed
%    - Initial weights are normalized such that sum(prod(weights,3),2) = 1
%    - Inputs orig and dest may be provided as double or single precision
%    
% Usage:
%    
%    [ rays, weights ] = quadriga_lib.generate_diffraction_paths( orig, dest, center_frequency, lod );
%    
% Input Arguments:
%    - orig
%      Origin point of the propagation ellipsoid (e.g. transmitter positions). Size: [ n_pos, 3 ]
%    
%    - dest
%      Destination point of the propagation ellipsoid (e.g. receiver positions). Size: [ n_pos, 3 ]
%    
%    - center_freq
%      The center frequency in [Hz], scalar, default = 299792458 Hz
%    
%    - lod
%      Level of detail, scalar value
%      lod = 1 | results in n_path = 7 and n_seg = 3
%      lod = 2 | results in n_path = 19 and n_seg = 3
%      lod = 3 | results in n_path = 37 and n_seg = 4
%      lod = 4 | results in n_path = 61 and n_seg = 5
%      lod = 5 | results in n_path = 1 and n_seg = 2 (for debugging)
%      lod = 6 | results in n_path = 2 and n_seg = 2 (for debugging)
%    
% Output Arguments:
%    - rays
%      Coordinates of the generated rays; Size: [ n_pos, n_path, n_seg-1, 3 ]
%    
%    - weights
%      Weights; Size: [ n_pos, n_path, n_seg ]
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
    