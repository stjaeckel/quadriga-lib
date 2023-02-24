CALC_ROTATION_MATRIX Calculates a 3x3 rotation matrix from a 3-element orientation vector
%
% Input:
%   orientation
%   This (optional) 3-element vector describes the orientation of the array antenna or of individual
%   array elements. The The first value describes the ”bank angle”, the second value describes the
%   ”tilt angle”, (positive values point upwards), the third value describes the bearing or ”heading
%   angle”, in mathematic sense. Values must be given in [rad]. East corresponds to 0, and the
%   angles increase counter-clockwise, so north is pi/2, south is -pi/2, and west is equal to pi. 
%   Single or double precision, Size: [3, n_row, n_col]
%
%   invert_y_axis
%   Optional parameter. If set to 1, the rotation around the y-axis is inverted
%
%   transpose
%   Optional parameter. If set to 1, the outout is transposed.
%
% Derived input:
%   n_rot           Number of antenna elements
%
% Output:
%   rotation
%   The rotation matrix, i.e. a transformation matrix that is used to perform a rotation in 3D Euclidean 
%   space. The matrix produces the desired effect only if it is used to premultiply column vectors. 
%   The rotations are applies in the order: heading (around z axis), tilt (around y axis) and bank 
%   (around x axis). The 9 elements of the rotation matrix are returned in column-major order.
%   Single or double precision (same as input), Size: [9, n_row, n_col]
%
%
% arrayant-lib c++/MEX Array antenna support library
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
