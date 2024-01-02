% HDF5_READ_CHANNEL
%    Read channel data from an HDF5 file
%    
% Description:
%    Quadriga-Lib provides an HDF5-based solution for storing and organizing channel data. This data 
%    comprises various well-defined sets, including channel coefficients, positions of transmitters and 
%    receivers, as well as path data that reflects the interaction of radio waves with the environment. 
%    Typically, these datasets are multi-dimensional, encompassing data for n_rx receive antennas, 
%    n_tx transmit antennas, n_path propagation paths, and n_snap snapshots. Snapshots are 
%    particularly useful for recording data across different locations (such as along a trajectory) or 
%    various frequencies. It is important to note that not all datasets include all these dimensions.
%    
%    The library also supports the addition of extra datasets of any type or shape, which can be useful 
%    for incorporating descriptive data or analysis results. To facilitate data access, the function 
%    quadriga_lib.hdf5_read_channel is designed to read both structured and unstructured data from the 
%    file.
%    
% Usage:
%    
%    [ par, rx_position, tx_position, coeff_re, coeff_im, delay, center_freq, name, initial_pos, ...
%       path_gain, path_length, path_polarization, path_angles, path_fbs_pos, path_lbs_pos, no_interact, ...
%       interact_coord, rx_orientation, tx_orientation ] = quadriga_lib.hdf5_read_channel( fn, location, snap );
%    
% Input Arguments:
%    - fn
%      Filename of the HDF5 file, string
%    
%    - location (optional)
%      Storage location inside the file; 1-based; vector with 1-4 elements, i.e. [ix], [ix, iy], 
%      [ix,iy,iz] or [ix,iy,iz,iw]; Default: ix = iy = iz = iw = 1
%    
%    - snap (optional)
%      Snapshot range; optional; vector, default = read all
%    
% Output Arguments:
%    - par
%      Unstructured data as struct, may be empty if no unstructured data is present
%    
%    - Structured data: (outputs 2-19, single precision)
%      rx_position    | Receiver positions                                       | [3, n_snap] or [3, 1]
%      tx_position    | Transmitter positions                                    | [3, n_snap] or [3, 1]
%      coeff_re       | Channel coefficients, real part                          | [n_rx, n_tx, n_path, n_snap]
%      coeff_im       | Channel coefficients, imaginary part                     | [n_rx, n_tx, n_path, n_snap]
%      delay          | Propagation delays in seconds                            | [n_rx, n_tx, n_path, n_snap] or [1, 1, n_path, n_snap]
%      center_freq    | Center frequency in [Hz]                                 | [n_snap, 1] or scalar
%      name           | Name of the channel                                      | String
%      initial_pos    | Index of reference position, 1-based                     | uint32, scalar
%      path_gain      | Path gain before antenna, linear scale                   | [n_path, n_snap]
%      path_length    | Path length from TX to RX phase center in m              | [n_path, n_snap]
%      polarization   | Polarization transfer function, interleaved complex      | [8, n_path, n_snap]
%      path_angles    | Departure and arrival angles {AOD, EOD, AOA, EOA} in rad | [n_path, 4, n_snap]
%      path_fbs_pos   | First-bounce scatterer positions                         | [3, n_path, n_snap]
%      path_lbs_pos   | Last-bounce scatterer positions                          | [3, n_path, n_snap]
%      no_interact    | Number interaction points of paths with the environment  | uint32, [n_path, n_snap]
%      interact_coord | Interaction coordinates                                  | [3, max(sum(no_interact)), n_snap]
%      rx_orientation | Transmitter orientation                                  | [3, n_snap] or [3, 1]
%      tx_orientation | Receiver orientation                                     | [3, n_snap] or [3, 1]
%      
% Caveat:
%    - Empty outputs are returned if data set does not exist in the file
%    - All structured data is stored in single precision. Hence, outputs are also in single precision.
%    - Unstructured datatypes are returned as stored in the HDF file (same type, dimensions and storage order)
%    - Typically, n_path may vary for each snapshot. In such cases, n_path is set to the maximum value found 
%      within the range of snapshots, and any missing paths are padded with zeroes.
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
    