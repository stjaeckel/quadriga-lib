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

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_WRITE_CHANNEL
Writes channel data to HDF5 files

## Description:
Quadriga-Lib provides an HDF5-based solution for storing and organizing channel data. This function
can be used to write structured and unstructured data to an HDF5 file.

## Usage:

```
storage_dims = quadriga_lib.hdf5_write_channel( fn, ix, iy, iz, iw, rx_position, tx_position, ...
   coeff_re, coeff_im, delay, center_freq, name, initial_pos, path_gain, path_length, ...
   path_polarization, path_angles, path_fbs_pos, path_lbs_pos, no_interact, interact_coord, ...
   rx_orientation, tx_orientation )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`ix`**<br>
  Storage index for x-dimension, Default = 0

- **`iy`**<br>
  Storage index for y-dimension, Default = 0

- **`iz`**<br>
  Storage index for z-dimension, Default = 0

- **`iw`**<br>
  Storage index for w-dimension, Default = 0

- **`par`**<br>
  Dictionary of unstructured data, can be empty if no unstructured data should be written

- **Structured data:** (double precision)<br>
  Each snapshot may have a different number of paths `n_path_s`. Variable-length data is provided as lists.
  `rx_pos`         | Receiver positions                                       | `[3, n_snap]` or `[3, 1]`
  `tx_pos`         | Transmitter positions                                    | `[3, n_snap]` or `[3, 1]`
  `coeff`          | Channel coefficients, complex valued                     | list of `[n_rx, n_tx, n_path_s]`
  `delay`          | Propagation delays in seconds                            | list of `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`
  `center_freq`    | Center frequency in [Hz]                                 | `[n_snap]` or scalar
  `name`           | Name of the channel                                      | String
  `initial_pos`    | Index of reference position, 1-based                     | uint32, scalar
  `path_gain`      | Path gain before antenna, linear scale                   | list of `[n_path_s]`
  `path_length`    | Path length from TX to RX phase center in m              | list of `[n_path_s]`
  `polarization`   | Polarization transfer function, complex valued           | list of `[4, n_path_s]`
  `path_angles`    | Departure and arrival angles {AOD, EOD, AOA, EOA} in rad | list of `[n_path, 4_s]`
  `path_fbs_pos`   | First-bounce scatterer positions                         | list of `[3, n_path_s]`
  `path_lbs_pos`   | Last-bounce scatterer positions                          | list of `[3, n_path_s]`
  `no_interact`    | Number interaction points of paths with the environment  | uint32, list of `[n_path_s]`
  `interact_coord` | Interaction coordinates                                  | list of `[3, max(sum(no_interact))]`
  `rx_orientation` | Transmitter orientation                                  | `[3, n_snap]` or `[3]`
  `tx_orientation` | Receiver orientation                                     | `[3, n_snap]` or `[3]`

## Output Arguments:
- **`storage_dims`**<br>
  Size of the dimensions of the storage space, vector with 4 elements, i.e. `[nx,ny,nz,nw]`.

## Caveat:
- If the file exists already, the new data is added to the exisiting file
- If a new file is created, a storage layout is created to store the location of datasets in the file
- For `location = [ix]` storage layout is `[65536,1,1,1]` or `[ix,1,1,1]` if (`ix > 65536`)
- For `location = [ix,iy]` storage layout is `[1024,64,1,1]`
- For `location = [ix,iy,iz]` storage layout is `[256,16,16,1]`
- For `location = [ix,iy,iz,iw]` storage layout is `[128,8,8,8]`
- You can create a custom storage layout by creating the file first using "`hdf5_create_file`"
- You can reshape the storage layout by using "`hdf5_reshape_storage`", but the total number of elements must not change
- Inputs can be empty or missing.
- All structured data is written in single precision (but can can be provided as single or double)
- Unstructured datatypes are maintained in the HDF file
- Supported unstructured types: string, double, float, (u)int32, (u)int64
- Supported unstructured size: up to 3 dimensions
- Storage order of the unstructured data is maintained
MD!*/

py::array_t<unsigned> hdf5_write_channel(const std::string fn,
                                         unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                         const py::dict par,
                                         const py::array_t<double> rx_pos,
                                         const py::array_t<double> tx_pos,
                                         const py::list coeff,
                                         const py::list delay,
                                         const py::array_t<double> center_frequency,
                                         const std::string name,
                                         const int initial_position,
                                         const py::list path_gain,
                                         const py::list path_length,
                                         const py::list path_polarization,
                                         const py::list path_angles,
                                         const py::list path_fbs_pos,
                                         const py::list path_lbs_pos,
                                         const py::list no_interact,
                                         const py::list interact_coord,
                                         const py::array_t<double> rx_orientation,
                                         const py::array_t<double> tx_orientation)
{
    // Construct channel object from input data
    auto c = quadriga_lib::channel<double>();
    c.initial_position = initial_position;
    c.name = name;

    // Process the unstructured data
    for (auto item : par)
    {
        std::string fieldName = py::str(item.first);
        std::string fieldString = "par." + fieldName;
        c.par_names.push_back(std::string(fieldName));
        c.par_data.push_back(qd_python_anycast(item.second, fieldName));
    }

    if (rx_pos.size() != 0)
        c.rx_pos = qd_python_numpy2arma_Mat(rx_pos, true);

    if (tx_pos.size() != 0)
        c.tx_pos = qd_python_numpy2arma_Mat(tx_pos, true);

    if (coeff.size() != 0)
        qd_python_list2vector_Cube_Cplx(coeff, c.coeff_re, c.coeff_im);

    if (delay.size() != 0)
        c.delay = qd_python_list2vector_Cube<double>(delay);

    if (center_frequency.size() != 0)
        c.center_frequency = qd_python_numpy2arma_Col(center_frequency, true, true);

    if (path_gain.size() != 0)
        c.path_gain = qd_python_list2vector_Col<double>(path_gain);

    if (path_length.size() != 0)
        c.path_length = qd_python_list2vector_Col<double>(path_length);

    if (path_polarization.size() != 0)
        c.path_polarization = qd_python_Complex2Interleaved(qd_python_list2vector_Mat<std::complex<double>>(path_polarization));

    if (path_angles.size() != 0)
        c.path_angles = qd_python_list2vector_Mat<double>(path_angles);

    if (path_fbs_pos.size() != 0)
        c.path_fbs_pos = qd_python_list2vector_Mat<double>(path_fbs_pos);

    if (path_lbs_pos.size() != 0)
        c.path_lbs_pos = qd_python_list2vector_Mat<double>(path_lbs_pos);

    if (no_interact.size() != 0)
        c.no_interact = qd_python_list2vector_Col<unsigned>(no_interact);

    if (interact_coord.size() != 0)
        c.interact_coord = qd_python_list2vector_Mat<double>(interact_coord);

    if (rx_orientation.size() != 0)
        c.rx_orientation = qd_python_numpy2arma_Mat(rx_orientation, true);

    if (tx_orientation.size() != 0)
        c.tx_orientation = qd_python_numpy2arma_Mat(tx_orientation, true);

    // Create HDF File if it dies not already exist
    auto storage_space = quadriga_lib::hdf5_read_layout(fn);
    if (storage_space.at(0) == 0) // File does not exist
    {
        unsigned nx = 1, ny = 1, nz = 1, nw = 1;
        if (iy == 0 && iz == 0 && iw == 0)
            nx = ix > 65536U ? ix : 65536U;
        else if (iz == 0 && iw == 0)
            nx = ix > 1024U ? ix : 1024U,
            ny = iy > 64U ? iy : 64U;
        else if (iw == 0)
            nx = ix > 256U ? ix : 256U,
            ny = iy > 16U ? iy : 16U,
            nz = iz > 16U ? iz : 16U;
        else
            nx = ix > 128U ? ix : 128U,
            ny = iy > 8U ? iy : 8U,
            nz = iz > 8U ? iz : 8U,
            nw = iz > 8U ? iw : 8U;

        quadriga_lib::hdf5_create(fn, nx, ny, nz, nw);
        storage_space.at(0) = nx;
        storage_space.at(1) = ny;
        storage_space.at(2) = nz;
        storage_space.at(3) = nw;
    }

    // Throw error if location exceeds storage space
    if (ix > storage_space.at(0) || iy > storage_space.at(1) || iz > storage_space.at(2) || iw > storage_space.at(3))
        throw std::invalid_argument("Location exceeds storage space in HDF file");

    // Write data to file
    quadriga_lib::hdf5_write(&c, fn, ix, iy, iz, iw);

    return py::array_t<unsigned>(4ULL, storage_space.memptr());
}