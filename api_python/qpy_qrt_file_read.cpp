// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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
# qrt_file_read
Read ray-tracing data from QRT file

## Usage:
```
from quadriga_lib import channel
data = channel.qrt_file_read( fn, i_cir, i_orig, downlink )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the QRT file, string

- **`cir`**<br>
  Snapshot index in the file, Default = 0

- **`orig`**<br>
  Origin index (for downlink Origin = TX), Default = 0

- **`downlink`**<br>
  Switch for uplink / downlink direction, Default = true (downlink)

## Output Arguments:
- **`data`**<br>
  Dictionary containing the data in the HDF file with the following keys:
  `center_freq`    | Center frequency in [Hz]                                 | scalar
  `tx_pos`         | Transmitter position                                     | Length `[3]`
  `tx_orientation` | Transmitter orientation, Euler angles, rad               | Length `[3]`
  `rx_pos`         | Receiver position                                        | Length `[3]`
  `rx_orientation` | Receiver orientation, Euler angles, rad                  | Length `[3]`
  `fbs_pos`        | First-bounce scatterer positions                         | Size `[3, n_path]`
  `lbs_pos`        | Last-bounce scatterer positions                          | Size `[3, n_path]`
  `path_gain`      | Path gain before antenna, linear scale                   | Length `[n_path]`
  `path_length`    | Path length from TX to RX phase center in m              | Length `[n_path]`
  `M`              | Polarization transfer function, interleaved complex      | Size `[8, n_path]`
  `aod`            | Departure azimuth angles in [rad]                        | Length `[n_path]`
  `eod`            | Departure elevation angles in [rad]                      | Length `[n_path]`
  `aoa`            | Arrival azimuth angles in [rad]                          | Length `[n_path]`
  `eoa`            | Arrival elevation angles in [rad]                        | Length `[n_path]`
  `path_coord`     | Interaction coordinates                                  | List of `[3, n_int_s]`
MD!*/

py::dict qrt_file_read(const std::string &fn, arma::uword cir, arma::uword orig, bool downlink)
{
    double center_frequency;
    arma::vec tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa;
    arma::mat fbs_pos, lbs_pos, M;
    arma::vec path_gain, path_length;
    std::vector<arma::mat> path_coord;

    quadriga_lib::qrt_file_read<double>(fn, cir, orig, downlink, &center_frequency, &tx_pos, &tx_orientation,
                                        &rx_pos, &rx_orientation, &fbs_pos, &lbs_pos, &path_gain,
                                        &path_length, &M, &aod, &eod, &aoa, &eoa, &path_coord);

    py::dict output;
    output["center_freq"] = center_frequency;
    output["tx_pos"] = qd_python_copy2numpy(tx_pos);
    output["tx_orientation"] = qd_python_copy2numpy(tx_orientation);
    output["rx_pos"] = qd_python_copy2numpy(rx_pos);
    output["rx_orientation"] = qd_python_copy2numpy(rx_orientation);
    output["fbs_pos"] = qd_python_copy2numpy(fbs_pos);
    output["lbs_pos"] = qd_python_copy2numpy(lbs_pos);
    output["path_gain"] = qd_python_copy2numpy(path_gain);
    output["path_length"] = qd_python_copy2numpy(path_length);
    output["M"] = qd_python_copy2numpy(M);
    output["aod"] = qd_python_copy2numpy(aod);
    output["eod"] = qd_python_copy2numpy(eod);
    output["aoa"] = qd_python_copy2numpy(aoa);
    output["eoa"] = qd_python_copy2numpy(eoa);
    output["path_coord"] = qd_python_copy2numpy(path_coord);

    return output;
}