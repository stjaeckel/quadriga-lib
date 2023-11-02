% CALC_ROTATION_MATRIX 
%    Calculates a 3x3 rotation matrix from a 3-element orientation vector
%    
% Description:
%    
%    In linear algebra, a rotation matrix is a transformation matrix that is used to perform a rotation 
%    in Euclidean space. The rotation of a rigid body (or three-dimensional coordinate system with a 
%    fixed origin) is described by a single rotation about some axis. Such a rotation may be uniquely 
%    described by three real-valued parameters. The idea behind Euler rotations is to split the complete 
%    rotation of the coordinate system into three simpler constitutive rotations, called precession, 
%    nutation, and intrinsic rotation, being each one of them an increment on one of the Euler angles. In 
%    aviation orientation of the aircraft is usually expressed as intrinsic Tait-Bryan angles following 
%    the z-y-x convention, which are called heading, tilt, and bank (or synonymously, yaw, pitch, 
%    and roll). This function calculates the 3x3 rotation matrix R from the intrinsic Tait-Bryan 
%    (orientation) angles. 
%    
% Usage:
%    
%    rotation = quadriga_lib.calc_rotation_matrix( orientation, invert_y_axis, transpose )
%    
% Example:
%    
%    The following example obtains the 3x3 matrix R for a 45 degree rotation around the z-axis:
%    
%    bank    = 0;
%    tilt    = 0;
%    heading = 45  pi/180;
%    
%    orientation = [ bank; tilt; heading ];
%    rotation = quadriga_lib.calc_rotation_matrix( orientation );
%    R = reshape( rotation, 3, 3 );
%    
% Input Arguments:
%    
%    - orientation
%      This 3-element vector describes the orientation of the array antenna or of individual array elements.
%      The The first value describes the ”bank angle”, the second value describes the  ”tilt angle”, 
%      (positive values point upwards), the third value describes the bearing or ”heading angle”, in 
%      mathematic sense. Values must be given in [rad]. East corresponds to 0, and the angles increase 
%      counter-clockwise, so north is pi/2, south is -pi/2, and west is equal to pi. 
%      Single or double precision, Size: [3, n_row, n_col]
%    
%    - invert_y_axis
%      Optional parameter. If set to 1, the rotation around the y-axis is inverted.
%    
%    - transpose
%      Optional parameter. If set to 1, the output is transposed.
%    
% Output Argument:
%    
%    - rotation
%      The rotation matrix, i.e. a transformation matrix that is used to perform a rotation in 3D 
%      Euclidean space. The matrix produces the desired effect only if it is used to premultiply column 
%      vectors. The rotations are applies in the order: heading (around z axis), tilt (around y axis) 
%      and bank (around x axis). The 9 elements of the rotation matrix are returned in column-major 
%      order. Single or double precision (same as input), Size: [9, n_row, n_col]
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
    