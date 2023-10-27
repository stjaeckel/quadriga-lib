%ARRAYANT_COMBINE_PATTERN calculates effective radiation patterns for array antennas
%
% Description:
%   An array antenna consists of multiple individual elements. Each element occupies a specific
%   position relative to the array's phase-center, its local origin. Elements can also be
%   inter-coupled, represented by a coupling matrix. By integrating the element radiation patterns,
%   their positions, and the coupling weights, one can determine an effective radiation pattern
%   observable by a receiver in the antenna's far field. Leveraging these effective patterns is
%   especially beneficial in antenna design, beamforming applications such as in 5G systems, and
%   in planning wireless communication networks in complex environments like urban areas. This
%   streamlined approach offers a significant boost in computation speed when calculating MIMO
%   channel coefficients, as it reduces the number of necessary operations. The function
%   "arrayant_combine_pattern" is designed to compute these effective radiation patterns.
%
% Inputs:
%   e_theta_re
%   Real part of the e-theta component (vertical component) of the far field of each antenna element
%   in the array antenna. Single or double precision, Size: [n_elevation, n_azimuth, n_elements]
%
%   e_theta_im
%   Imaginary part of the e-theta component of the electric field.
%   Single or double precision, Size: [n_elevation, n_azimuth, n_elements]
%
%   e_phi_re
%   Real part of the e-phi component (horizontal component) of the far field of each antenna element
%   in the array antenna. Single or double precision, Size: [n_elevation, n_azimuth, n_elements]
%
%   e_phi_im
%   Imaginary part of the e-phi component of the electric field.
%   Single or double precision, Size: [n_elevation, n_azimuth, n_elements]
%
%   azimuth_grid
%   Azimuth angles (theta) in [rad] were samples of the field patterns are provided. Values must be
%   between -pi and pi, sorted in ascending order. Single or double precision, Size: [n_azimuth]
%
%   elevation_grid
%   Elevation angles (phi) in [rad] where samples of the field patterns are provided. Values must be
%   between -pi/2 and pi/2, sorted in ascending order. Single or double precision, Size: [n_elevation]
%
%   element_pos (optional)
%   Antenna element (x,y,z) positions relative to the array's phase-center in units of [m].
%   Size: [3, n_elements] or []; empty input assumes position [0;0;0] for all elements
%
%   coupling_re (optional)
%   Real part of the array antenna coupling matrix. This matrix describes a pre- or post-processing
%   of the signals that are fed to or received by the antenna elements. The rows in the matrix
%   correspond to the antenna elements, the columns to the signal ports. By default, coupling is set
%   to an identity matrix which indicates perfect isolation between the antenna elements.
%   Size: [n_elements, n_ports] or []; empty assumes perfect isolation
%
%   coupling_im (optional)
%   Real part of the array antenna coupling matrix. Must have identical size as "coupling_re".
%   Size: [n_elements, n_ports] or []
%
%   center_frequency (optional)
%   Center frequency in [Hz] (optional). Default value is 299,792,458 Hz, which corresponds to a
%   wavelength of 1 m. In this case, element positions "element_pos" can be provided as multiple of
%   the wavelength. Scalar
%
% Outputs:
%   e_theta_re_c
%   Real part of the e-theta component (vertical component) of the effective array antenna.
%   Size: [n_elevation, n_azimuth, n_ports]
%
%   e_theta_im_c
%   Imaginary part of the e-theta component (vertical component) of the effective array antenna.
%   Size: [n_elevation, n_azimuth, n_ports]
%
%   e_phi_re_c
%   Real part of the e-phi component (horizontal component) of the effective array antenna.
%   Size: [n_elevation, n_azimuth, n_ports]
%
%   e_phi_im_c
%   Imaginary part of the e-phi component (horizontal component) of the effective array antenna.
%   Size: [n_elevation, n_azimuth, n_ports]
%
% Note: The effective antenna has all elements at the phase center [0,0,0]' and has perfect isolation
% between its elements. Hence, no outputs for the effective "element_pos" and "coupling" are needed.
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