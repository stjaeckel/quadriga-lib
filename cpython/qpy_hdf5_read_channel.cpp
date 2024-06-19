// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "quadriga_lib.hpp"

#include "python_helpers.cpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_CHANNEL
Reads channel data from HDF5 files

## Description:
Quadriga-Lib provides an HDF5-based solution for storing and organizing channel data. This data
comprises various well-defined sets, including channel coefficients, positions of transmitters and
receivers, as well as path data that reflects the interaction of radio waves with the environment.
Typically, these datasets are multi-dimensional, encompassing data for `n_rx` receive antennas,
`n_tx` transmit antennas, `n_path` propagation paths, and `n_snap` snapshots. Snapshots are
particularly useful for recording data across different locations (such as along a trajectory) or
various frequencies. It is important to note that not all datasets include all these dimensions.<br><br>

The library also supports the addition of extra datasets of any type or shape, which can be useful
for incorporating descriptive data or analysis results. To facilitate data access, the function
`quadriga_lib.hdf5_read_channel` is designed to read both structured and unstructured data from the
file.

## Usage:

```
data = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, snap );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`ix`**<br>
  Number of elements on the x-dimension, Default = 0

- **`iy`**<br>
  Number of elements on the x-dimension, Default = 0

- **`iz`**<br>
  Number of elements on the x-dimension, Default = 0

- **`iw`**<br>
  Number of elements on the x-dimension, Default = 0

- **`snap`** (optional)<br>
  Snapshot range, 0-based notation; optional; vector, default: empty = read all

## Output Arguments:
- **`data`**<br>
  Dictionary containing the data in the HDF file with the following keys:
  `par`            | Dictionary of unstructured data                          | Variable
  `rx_position`    | Receiver positions                                       | `[3, n_snap]` or `[3, 1]`
  `tx_position`    | Transmitter positions                                    | `[3, n_snap]` or `[3, 1]`
  `coeff`          | Channel coefficients, complex valued                     | `[n_rx, n_tx, n_path, n_snap]`
  `delay`          | Propagation delays in seconds                            | `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`
  `center_freq`    | Center frequency in [Hz]                                 | `[n_snap, 1]` or scalar
  `name`           | Name of the channel                                      | String
  `initial_pos`    | Index of reference position, 0-based                     | uint32, scalar
  `path_gain`      | Path gain before antenna, linear scale                   | `[n_path, n_snap]`
  `path_length`    | Path length from TX to RX phase center in m              | `[n_path, n_snap]`
  `polarization`   | Polarization transfer function, complex valued           | `[4, n_path, n_snap]`
  `path_angles`    | Departure and arrival angles {AOD, EOD, AOA, EOA} in rad | `[n_path, 4, n_snap]`
  `path_fbs_pos`   | First-bounce scatterer positions                         | `[3, n_path, n_snap]`
  `path_lbs_pos`   | Last-bounce scatterer positions                          | `[3, n_path, n_snap]`
  `no_interact`    | Number interaction points of paths with the environment  | uint32, `[n_path, n_snap]`
  `interact_coord` | Interaction coordinates                                  | `[3, max(sum(no_interact)), n_snap]`
  `rx_orientation` | Transmitter orientation                                  | `[3, n_snap]` or `[3, 1]`
  `tx_orientation` | Receiver orientation                                     | `[3, n_snap]` or `[3, 1]`

## Caveat:
- Only datasets that are present in the HDF file are returned in the dictionary.
- Although the data is stored in single precision, it is converted to double precision by default.
MD!*/

pybind11::dict hdf5_read_channel(const std::string fn,
                                 unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                 const pybind11::array_t<unsigned> snap)
{
    // Read data from file
    const auto channel = quadriga_lib::hdf5_read_channel<double>(fn, ix, iy, iz, iw);

    // Read snapshot range
    arma::uvec snap_arma; // Empty = read all
    if (snap.size() != 0)
    {
        pybind11::buffer_info buf = snap.request();
        size_t n_snap = (size_t)buf.size;
        snap_arma.set_size(n_snap);
        unsigned *python_ptr = reinterpret_cast<unsigned *>(buf.ptr);
        arma::uword *arma_ptr = snap_arma.memptr();
        for (size_t i = 0; i < n_snap; ++i)
            arma_ptr[i] = (arma::uword)python_ptr[i];
    }

    // Update snapshot index
    auto n_snap_channel = channel.n_snap();
    if (snap_arma.empty() && n_snap_channel != 0)
    {
        snap_arma.set_size(n_snap_channel);
        arma::uword *p = snap_arma.memptr();
        for (arma::uword s = 0; s < n_snap_channel; ++s)
            p[s] = s;
    }
    else if (n_snap_channel != 0) // Check bounds
    {
        for (auto &p : snap_arma)
            if (p >= n_snap_channel)
                throw std::invalid_argument("Snapshot index out of bound.");
    }
    else // n_snap_channel == 0ULL
        snap_arma.reset();

    arma::uword n_snap = snap_arma.n_elem;
    arma::uword *i_snap = snap_arma.memptr(); // Snapshot index

    // Get number of paths
    arma::uword n_path = 0;
    if (n_snap != 0 && n_snap_channel != 0)
    {
        arma::uvec n_path_vec = channel.n_path();
        arma::uword *p = n_path_vec.memptr();
        for (arma::uword i = 0; i < n_snap; ++i)
            n_path = (p[i_snap[i]] > n_path) ? p[i_snap[i]] : n_path;
    }

    // Initialize output
    pybind11::dict output;

    // Parse unstructured data
    if (!channel.par_names.empty())
    {
        pybind11::dict par;
        for (size_t n = 0; n < channel.par_names.size(); ++n)
        {
            unsigned long long dims[3];
            void *dataptr;
            int type_id = quadriga_lib::any_type_id(&channel.par_data.at(n), dims, &dataptr);

            if (type_id == 9) // Strings
            {
                auto data = std::any_cast<std::string>(channel.par_data.at(n));
                par[channel.par_names.at(n).c_str()] = data;
            }

            // Scalars
            else if (type_id == 10)
                par[channel.par_names.at(n).c_str()] = *(float *)dataptr;
            else if (type_id == 11)
                par[channel.par_names.at(n).c_str()] = *(double *)dataptr;
            else if (type_id == 12)
                par[channel.par_names.at(n).c_str()] = *(unsigned long long int *)dataptr;
            else if (type_id == 13)
                par[channel.par_names.at(n).c_str()] = *(long long int *)dataptr;
            else if (type_id == 14)
                par[channel.par_names.at(n).c_str()] = *(unsigned int *)dataptr;
            else if (type_id == 15)
                par[channel.par_names.at(n).c_str()] = *(int *)dataptr;

            // Matrices
            else if (type_id == 20)
            {
                auto data = std::any_cast<arma::Mat<float>>(channel.par_data.at(n));
                size_t strides[2] = {sizeof(float), data.n_rows * sizeof(float)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<float>({data.n_rows, data.n_cols}, strides, data.memptr());
            }
            else if (type_id == 21)
            {
                auto data = std::any_cast<arma::Mat<double>>(channel.par_data.at(n));
                size_t strides[2] = {sizeof(double), data.n_rows * sizeof(double)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<double>({data.n_rows, data.n_cols}, strides, data.memptr());
            }
            else if (type_id == 22)
            {
                auto data = std::any_cast<arma::Mat<unsigned long long>>(channel.par_data.at(n));
                size_t strides[2] = {sizeof(unsigned long long), data.n_rows * sizeof(unsigned long long)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<unsigned long long>({data.n_rows, data.n_cols}, strides, data.memptr());
            }
            else if (type_id == 23)
            {
                auto data = std::any_cast<arma::Mat<long long>>(channel.par_data.at(n));
                size_t strides[2] = {sizeof(long long), data.n_rows * sizeof(long long)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<long long>({data.n_rows, data.n_cols}, strides, data.memptr());
            }
            else if (type_id == 24)
            {
                auto data = std::any_cast<arma::Mat<unsigned>>(channel.par_data.at(n));
                size_t strides[2] = {sizeof(unsigned), data.n_rows * sizeof(unsigned)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<unsigned>({data.n_rows, data.n_cols}, strides, data.memptr());
            }
            else if (type_id == 25)
            {
                auto data = std::any_cast<arma::Mat<int>>(channel.par_data.at(n));
                size_t strides[2] = {sizeof(int), data.n_rows * sizeof(int)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<int>({data.n_rows, data.n_cols}, strides, data.memptr());
            }

            // Cubes
            else if (type_id == 30)
            {
                auto data = std::any_cast<arma::Cube<float>>(channel.par_data.at(n));
                size_t strides[3] = {sizeof(float), data.n_rows * sizeof(float), data.n_rows * data.n_cols * sizeof(float)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<float>({data.n_rows, data.n_cols, data.n_slices}, strides, data.memptr());
            }
            else if (type_id == 31)
            {
                auto data = std::any_cast<arma::Cube<double>>(channel.par_data.at(n));
                size_t strides[3] = {sizeof(double), data.n_rows * sizeof(double), data.n_rows * data.n_cols * sizeof(double)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<double>({data.n_rows, data.n_cols, data.n_slices}, strides, data.memptr());
            }
            else if (type_id == 32)
            {
                auto data = std::any_cast<arma::Cube<unsigned long long>>(channel.par_data.at(n));
                size_t strides[3] = {sizeof(unsigned long long), data.n_rows * sizeof(unsigned long long), data.n_rows * data.n_cols * sizeof(unsigned long long)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<unsigned long long>({data.n_rows, data.n_cols, data.n_slices}, strides, data.memptr());
            }
            else if (type_id == 33)
            {
                auto data = std::any_cast<arma::Cube<long long>>(channel.par_data.at(n));
                size_t strides[3] = {sizeof(long long), data.n_rows * sizeof(long long), data.n_rows * data.n_cols * sizeof(long long)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<long long>({data.n_rows, data.n_cols, data.n_slices}, strides, data.memptr());
            }
            else if (type_id == 34)
            {
                auto data = std::any_cast<arma::Cube<unsigned>>(channel.par_data.at(n));
                size_t strides[3] = {sizeof(unsigned), data.n_rows * sizeof(unsigned), data.n_rows * data.n_cols * sizeof(unsigned)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<unsigned>({data.n_rows, data.n_cols, data.n_slices}, strides, data.memptr());
            }
            else if (type_id == 35)
            {
                auto data = std::any_cast<arma::Cube<int>>(channel.par_data.at(n));
                size_t strides[3] = {sizeof(int), data.n_rows * sizeof(int), data.n_rows * data.n_cols * sizeof(int)};
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<int>({data.n_rows, data.n_cols, data.n_slices}, strides, data.memptr());
            }

            // Vectors (Columns only)
            else if (type_id == 40)
            {
                auto data = std::any_cast<arma::Col<float>>(channel.par_data.at(n));
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<float>(data.n_elem, data.memptr());
            }
            else if (type_id == 41)
            {
                auto data = std::any_cast<arma::Col<double>>(channel.par_data.at(n));
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<double>(data.n_elem, data.memptr());
            }
            else if (type_id == 42)
            {
                auto data = std::any_cast<arma::Col<unsigned long long>>(channel.par_data.at(n));
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<unsigned long long>(data.n_elem, data.memptr());
            }
            else if (type_id == 43)
            {
                auto data = std::any_cast<arma::Col<long long>>(channel.par_data.at(n));
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<long long>(data.n_elem, data.memptr());
            }
            else if (type_id == 44)
            {
                auto data = std::any_cast<arma::Col<unsigned>>(channel.par_data.at(n));
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<unsigned>(data.n_elem, data.memptr());
            }
            else if (type_id == 45)
            {
                auto data = std::any_cast<arma::Col<int>>(channel.par_data.at(n));
                par[channel.par_names.at(n).c_str()] = pybind11::array_t<int>(data.n_elem, data.memptr());
            }
        }
        output["par"] = par;
    }

    if (channel.rx_pos.n_cols == 1 || (channel.rx_pos.n_cols > 1 && snap.size() == 0)) // Return all
        output["rx_position"] = qd_python_Mat_to_NPArray(&channel.rx_pos);
    else if (snap.size() != 0) // Subset
        output["rx_position"] = qd_python_Mat_to_NPArray(&channel.rx_pos, n_snap, i_snap);

    if (channel.tx_pos.n_cols == 1 || (channel.tx_pos.n_cols > 1 && snap.size() == 0)) // Return all
        output["tx_position"] = qd_python_Mat_to_NPArray(&channel.tx_pos);
    else if (snap.size() != 0) // Subset
        output["tx_position"] = qd_python_Mat_to_NPArray(&channel.tx_pos, n_snap, i_snap);

    if (!channel.coeff_re.empty())
        output["coeff"] = qd_python_2vectorCubes_to_complexNPArray(&channel.coeff_re, &channel.coeff_im, n_snap, i_snap);

    if (!channel.delay.empty())
        output["delay"] = qd_python_vectorCube_to_NPArray(&channel.delay, n_snap, i_snap);

    if (channel.center_frequency.n_elem == 1 || (channel.center_frequency.n_elem > 1 && snap.size() == 0)) // Return all
        output["center_freq"] = qd_python_Col_to_NPArray(&channel.center_frequency);
    else if (snap.size() != 0) // Subset
        output["center_freq"] = qd_python_Col_to_NPArray(&channel.center_frequency, n_snap, i_snap);

    if (n_snap_channel > 0)
    {
        output["name"] = channel.name;
        output["initial_pos"] = channel.initial_position;
    }

    if (!channel.path_gain.empty())
        output["path_gain"] = qd_python_vectorCol_to_NPArray(&channel.path_gain, n_snap, i_snap);

    if (!channel.path_length.empty())
        output["path_length"] = qd_python_vectorCol_to_NPArray(&channel.path_length, n_snap, i_snap);

    if (!channel.path_polarization.empty())
        output["polarization"] = qd_python_vectorMat_to_complexNPArray(&channel.path_polarization, n_snap, i_snap);

    if (!channel.path_angles.empty())
        output["path_angles"] = qd_python_vectorMat_to_NPArray(&channel.path_angles, n_snap, i_snap);

    if (!channel.path_fbs_pos.empty())
        output["path_fbs_pos"] = qd_python_vectorMat_to_NPArray(&channel.path_fbs_pos, n_snap, i_snap);

    if (!channel.path_lbs_pos.empty())
        output["path_lbs_pos"] = qd_python_vectorMat_to_NPArray(&channel.path_lbs_pos, n_snap, i_snap);

    if (!channel.no_interact.empty())
        output["no_interact"] = qd_python_vectorCol_to_NPArray(&channel.no_interact, n_snap, i_snap);

    if (!channel.interact_coord.empty())
        output["interact_coord"] = qd_python_vectorMat_to_NPArray(&channel.interact_coord, n_snap, i_snap);

    if (channel.rx_orientation.n_cols == 1 || (channel.rx_orientation.n_cols > 1 && snap.size() == 0)) // Return all
        output["rx_orientation"] = qd_python_Mat_to_NPArray(&channel.rx_orientation);
    else if (snap.size() != 0) // Subset
        output["rx_orientation"] = qd_python_Mat_to_NPArray(&channel.rx_orientation, n_snap, i_snap);

    if (channel.tx_orientation.n_cols == 1 || (channel.tx_orientation.n_cols > 1 && snap.size() == 0)) // Return all
        output["tx_orientation"] = qd_python_Mat_to_NPArray(&channel.tx_orientation);
    else if (snap.size() != 0) // Subset
        output["tx_orientation"] = qd_python_Mat_to_NPArray(&channel.tx_orientation, n_snap, i_snap);

    return output;
}
