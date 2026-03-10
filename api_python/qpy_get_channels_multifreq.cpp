// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# get_channels_multifreq
Calculate channel coefficients for spherical waves across multiple frequencies

## Description:
- Extends `get_channels_spherical` to support frequency-dependent antenna patterns, path gains,
  and polarization transfer (Jones) matrices across multiple output frequencies.
- **Geometry is computed once**: departure angles, arrival angles, element-resolved path delays, and LOS
  path detection are frequency-independent and reused for all output frequencies. This avoids redundant
  trigonometry and distance calculations.
- **Four frequency grids** are aligned by interpolation:
  1. | TX array frequencies (defined by `center_freq` in the `ant_tx` dictionary)
  2. | RX array frequencies (defined by `center_freq` in the `ant_rx` dictionary)
  3. | Input sample frequencies (`freq_in`) at which `path_gain` and `M` are provided
  4. | Target output frequencies (`freq_out`) at which coefficients and delays are returned
- For each output frequency, TX and RX antenna patterns are interpolated from their respective
  multi-frequency entries using spherical interpolation (SLERP) with linear fallback, the same
  algorithm used in `arrayant_interpolate_multi`.
- Path gain is interpolated linearly across frequency. The Jones matrix `M` is interpolated using
  SLERP for each complex entry pair to preserve phase coherence.
- **Extrapolation** is handled by clamping to the nearest available frequency entry in all four grids.
- **Propagation speed** can be set to support both radio (speed of light, default) and acoustic
  (speed of sound, ~343 m/s) simulations. This affects wavelength, wave number, and delay calculations.
- The Jones matrix `M` supports two formats: 8 rows for full polarimetric
  (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH), or 2 rows for scalar pressure waves
  (ReVV, ImVV only), where VH, HV, and HH entries are implicitly zero.
- Antenna element coupling is applied using the coupling matrices from the antenna dictionary.
  If coupling varies with frequency, provide 3D coupling arrays `(n_elem, n_ports, n_freq)`.
- Antenna dictionaries accept both 3D patterns (single-frequency, clamped for all output
  frequencies) and 4D patterns (multi-frequency, 4th dimension = frequency).

## Usage:

```
from quadriga_lib import arrayant
import numpy as np

coeff_re, coeff_im, delays = arrayant.get_channels_multifreq( ant_tx, ant_rx,
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation,
    freq_in, freq_out, use_absolute_delays, add_fake_los_path, propagation_speed )
```

## Input Arguments:
- **`ant_tx`** (required)<br>
  Dictionary containing the transmit (TX) arrayant data. Pattern fields may be 3D
  (single-frequency) or 4D (multi-frequency, 4th dimension = frequency). The following keys
  are expected:
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_el, n_az, n_elem)` or `(n_el, n_az, n_elem, n_freq)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_el, n_az, n_elem)` or `(n_el, n_az, n_elem, n_freq)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_el, n_az, n_elem)` or `(n_el, n_az, n_elem, n_freq)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_el, n_az, n_elem)` or `(n_el, n_az, n_elem, n_freq)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Shape: `(3, n_elements)`
  `coupling_re`    | Coupling matrix, real part, optional                  | Shape: `(n_elem, n_ports)` or `(n_elem, n_ports, n_freq)`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Shape: `(n_elem, n_ports)` or `(n_elem, n_ports, n_freq)`
  `center_freq`    | Center frequency in [Hz], optional                    | Scalar or 1D array `(n_freq)`
  `name`           | Name of the array antenna object, optional            | String

- **`ant_rx`** (required)<br>
  Dictionary containing the receive (RX) arrayant data (same format as `ant_tx`).

- **`fbs_pos`** (required)<br>
  First-bounce scatterer positions. For single-bounce models, identical to `lbs_pos`.
  Shape: `( 3, n_path )`

- **`lbs_pos`** (required)<br>
  Last-bounce scatterer positions. For single-bounce models, identical to `fbs_pos`.
  Shape: `( 3, n_path )`

- **`path_gain`** (required)<br>
  Path gain in linear scale. Each column corresponds to one input frequency.
  Shape: `( n_path, n_freq_in )`

- **`path_length`** (required)<br>
  Absolute path lengths from TX to RX phase center in meters.
  Shape: `( n_path )`

- **`M`** (required)<br>
  Polarization transfer matrix in interleaved complex format. Each slice along the 3rd dimension
  corresponds to one input frequency. Full polarimetric format uses 8 rows:
  (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH). Scalar pressure format uses 2 rows:
  (ReVV, ImVV), with VH, HV, and HH implicitly zero.
  Shape: `( 8, n_path, n_freq_in )` or `( 2, n_path, n_freq_in )`

- **`tx_pos`** (required)<br>
  Transmitter position in 3D Cartesian coordinates [m], Shape: `(3)`

- **`tx_orientation`** (required)<br>
  Transmitter antenna orientation in Euler angles (bank, tilt, heading) [rad], Shape: `(3)`

- **`rx_pos`** (required)<br>
  Receiver position in 3D Cartesian coordinates [m], Shape: `(3)`

- **`rx_orientation`** (required)<br>
  Receiver antenna orientation in Euler angles (bank, tilt, heading) [rad], Shape: `(3)`

- **`freq_in`** (required)<br>
  Input sample frequencies in [Hz] at which `path_gain` and `M` are defined.
  Shape: `( n_freq_in )`

- **`freq_out`** (required)<br>
  Target frequencies in [Hz] at which to compute output coefficients and delays.
  Shape: `( n_freq_out )`

- **`use_absolute_delays`** (optional)<br>
  If true, the LOS delay is included in all path delays. Default: `False`, i.e. delays are
  normalized so that the LOS path has zero delay.

- **`add_fake_los_path`** (optional)<br>
  If true, adds a zero-power LOS path as the first path when no LOS path was detected.
  Default: `False`

- **`propagation_speed`** (optional)<br>
  Wave propagation speed in [m/s]. Default: `299792458.0` (speed of light for radio simulations).
  Set to ~`343.0` for acoustic simulations in air.

## Derived inputs:
  `n_freq_in`    | Number of input frequency samples (columns of `path_gain`, slices of `M`)
  `n_freq_out`   | Number of output frequency samples (length of `freq_out`)
  `n_path`       | Number of propagation paths (columns of `fbs_pos`)
  `n_ports_tx`   | Number of TX antenna ports after coupling
  `n_ports_rx`   | Number of RX antenna ports after coupling

## Output Arguments:
- **`coeff_re`**<br>
  Channel coefficients, real part, 4D array with the 4th dimension being frequency.
  Shape: `( n_ports_rx, n_ports_tx, n_path, n_freq_out )`

- **`coeff_im`**<br>
  Channel coefficients, imaginary part, 4D array with the 4th dimension being frequency.
  Shape: `( n_ports_rx, n_ports_tx, n_path, n_freq_out )`

- **`delays`**<br>
  Propagation delays in seconds, 4D array with the 4th dimension being frequency.
  Shape: `( n_ports_rx, n_ports_tx, n_path, n_freq_out )`

## Example:

```
from quadriga_lib import arrayant
import numpy as np

# Build a 2-way speaker as TX (source)
tx_woofer = arrayant.generate_speaker(
    driver_type='piston', radius=0.083,
    lower_cutoff=50.0, upper_cutoff=3000.0,
    lower_rolloff_slope=12.0, upper_rolloff_slope=24.0, sensitivity=87.0,
    radiation_type='hemisphere', baffle_width=0.20, baffle_height=0.30,
    frequencies=np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0]),
    angular_resolution=10.0)

# Omnidirectional microphone as RX (single-frequency, clamped for all output frequencies)
rx = arrayant.generate('omni')

# Simple LOS path setup
fbs_pos = np.array([[0.5], [0.0], [0.0]])     # Scatterer at RX position
lbs_pos = np.array([[0.5], [0.0], [0.0]])
path_length = np.array([1.0])                 # 1 meter distance

# Frequency-flat path gain and scalar Jones matrix at two input frequencies
freq_in = np.array([100.0, 10000.0])
path_gain = np.ones((1, 2))                   # Unit gain at both frequencies
M = np.zeros((2, 1, 2))                       # Scalar pressure (2 rows)
M[0, 0, 0] = 1.0; M[0, 0, 1] = 1.0            # ReVV = 1 at both frequencies

# Compute channel at 3 output frequencies using speed of sound
freq_out = np.array([200.0, 1000.0, 5000.0])
coeff_re, coeff_im, delays = arrayant.get_channels_multifreq(
    tx_woofer, rx, fbs_pos, lbs_pos, path_gain, path_length, M,
    np.zeros(3),                               # TX at origin
    np.zeros(3),                               # TX orientation (no rotation)
    np.array([1.0, 0.0, 0.0]),                 # RX at (1, 0, 0)
    np.zeros(3),                               # RX orientation (no rotation)
    freq_in, freq_out,
    propagation_speed=343.0)                   # Speed of sound for acoustics

# coeff_re.shape = (1, n_tx_ports, 1, 3) — one RX port, one path, 3 output frequencies
```

## Caveat:
- Input data is directly accessed from Python memory, without copying if it is provided in
  **double** precision and is in F-contiguous (column-major) order.
- Other formats (e.g. single precision inputs or C-contiguous (row-major) order) will be converted
  to double automatically, causing additional computation steps.
- To improve performance of repeated computations (e.g. in loops), consider preparing the data
  in F-contiguous double precision to avoid unnecessary copies.

## See also:
- [[get_channels_spherical]] — single-frequency version
- [[interpolate]] — multi-frequency antenna interpolation
- [[generate_speaker]] — parametric loudspeaker directivity model
MD!*/

py::tuple get_channels_multifreq(const py::dict &ant_tx,
                                 const py::dict &ant_rx,
                                 const py::array_t<double> &fbs_pos,
                                 const py::array_t<double> &lbs_pos,
                                 const py::array_t<double> &path_gain,
                                 const py::array_t<double> &path_length,
                                 const py::array_t<double> &M,
                                 const py::array_t<double> &tx_pos,
                                 const py::array_t<double> &tx_orientation,
                                 const py::array_t<double> &rx_pos,
                                 const py::array_t<double> &rx_orientation,
                                 const py::array_t<double> &freq_in,
                                 const py::array_t<double> &freq_out,
                                 const bool use_absolute_delays,
                                 const bool add_fake_los_path,
                                 const double propagation_speed)
{
    // Parse TX antenna: 3D (single-freq) or 4D (multi-freq)
    std::vector<quadriga_lib::arrayant<double>> ant_tx_vec;
    {
        py::array e_theta_re_arr = py::cast<py::array>(ant_tx["e_theta_re"]);
        int nd = (int)e_theta_re_arr.request().ndim;
        if (nd == 3)
            ant_tx_vec = {qd_python_dict2arrayant(ant_tx, true)};
        else
            ant_tx_vec = qd_python_dict2arrayant_multi(ant_tx, true);
    }

    // Parse RX antenna: 3D (single-freq) or 4D (multi-freq)
    std::vector<quadriga_lib::arrayant<double>> ant_rx_vec;
    {
        py::array e_theta_re_arr = py::cast<py::array>(ant_rx["e_theta_re"]);
        int nd = (int)e_theta_re_arr.request().ndim;
        if (nd == 3)
            ant_rx_vec = {qd_python_dict2arrayant(ant_rx, true)};
        else
            ant_rx_vec = qd_python_dict2arrayant_multi(ant_rx, true);
    }

    // Parse remaining input arguments
    const auto fbs_pos_a = qd_python_numpy2arma_Mat(fbs_pos, true);
    const auto lbs_pos_a = qd_python_numpy2arma_Mat(lbs_pos, true);
    const auto path_gain_a = qd_python_numpy2arma_Mat(path_gain, true);
    const auto path_length_a = qd_python_numpy2arma_Col(path_length, true);
    const auto M_a = qd_python_numpy2arma_Cube(M, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation, true, false, "rx_orientation", 3);
    const auto freq_in_a = qd_python_numpy2arma_Col(freq_in, true);
    const auto freq_out_a = qd_python_numpy2arma_Col(freq_out, true);

    // Extract scalar values
    double Tx = tx_pos_a.at(0), Ty = tx_pos_a.at(1), Tz = tx_pos_a.at(2);
    double Tb = tx_orientation_a.at(0), Tt = tx_orientation_a.at(1), Th = tx_orientation_a.at(2);
    double Rx = rx_pos_a.at(0), Ry = rx_pos_a.at(1), Rz = rx_pos_a.at(2);
    double Rb = rx_orientation_a.at(0), Rt = rx_orientation_a.at(1), Rh = rx_orientation_a.at(2);

    // Initialize output containers
    std::vector<arma::cube> coeff_re, coeff_im, delay;

    // Call C++ library function
    quadriga_lib::get_channels_multifreq<double>(ant_tx_vec, ant_rx_vec,
                                                 Tx, Ty, Tz, Tb, Tt, Th,
                                                 Rx, Ry, Rz, Rb, Rt, Rh,
                                                 fbs_pos_a, lbs_pos_a, path_gain_a, path_length_a, M_a,
                                                 freq_in_a, freq_out_a,
                                                 coeff_re, coeff_im, delay,
                                                 use_absolute_delays, add_fake_los_path, propagation_speed);

    // Convert outputs to 4D numpy arrays (4th dim = frequency)
    auto coeff_re_p = qd_python_copy2numpy_4d(coeff_re);
    auto coeff_im_p = qd_python_copy2numpy_4d(coeff_im);
    auto delay_p = qd_python_copy2numpy_4d(delay);

    return py::make_tuple(coeff_re_p, coeff_im_p, delay_p);
}
