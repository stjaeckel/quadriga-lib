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
# baseband_freq_response
Transforms the channel into frequency domain and returns the frequency response

## Usage:

```
from quadriga_lib import channel
hmat = channel.baseband_freq_response( coeff, delay, bandwidth, carriers, pilot_grid, snap );
```

## Input Arguments:
- **`coeff`**<br>
  Channel coefficients, complex-valued, List of length `n_snap`, 
  Each list item is an munpy.ndarray of Shape `( n_rx, n_tx, n_path )` where `n_path` can be different 
  for each snapshot.

- **`delay`**<br>
  Propagation delay in seconds, List of length `n_snap`, 
  Each list item is an munpy.ndarray of Shape `( n_rx, n_tx, n_path )` or `( 1, 1, n_path )` where 
  `n_path` can be different for each snapshot.

- **`bandwidth`**<br>
  The baseband bandwidth in [Hz], scalar

- **`carriers`** (optional)<br>
  Number of carriers, equally spaced across the bandwidth. The first entry of the generated spectrum
  is equal to the center frequency f0. The spectrum is generated from f0 to f0+bandwidth. This
  argument is only evaluated if `pilot_grid` is not provided. Default value = 128

- **`pilot_grid`** (optional)<br>
  Sub-carrier positions relative to the bandwidth. The carrier positions are given relative to the
  bandwidth where '0' is the begin of the spectrum (i.e., the center frequency f0) and '1' is
  equal to f0+bandwidth. To obtain the channel frequency response centered around f0, the
  input variable 'pilot_grid' must be set to '(-N/2:N/2)/N', where N is the number of sub-
  carriers. Vector of length: `( n_carriers )`

- **`snap`** (optional)<br>
  Snapshot indices for which the frequency response should be generated (1-based index). If this
  variable is not given, all snapshots are processed. Length: `( n_out )`

## Output Argument:
- **`hmat`**<br>
  Freq. domain channel matrices (H), complex-valued, Shape `( n_rx, n_tx, n_carriers, n_out )`
MD!*/

// #include <chrono>

// std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(); // Start time
// std::cout << "Start CPP" << std::endl;

// std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(); // Current time
// double ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
// std::cout << "Interface, t = " << std::round(ms) / 1000.0 << std::endl;

// std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now(); // Current time
// ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
// std::cout << "DFT, t = " << std::round(ms) / 1000.0 << std::endl;

py::array_t<std::complex<double>> baseband_freq_response(const py::list &coeff,
                                                         const py::list &delay,
                                                         const double bandwidth,
                                                         const size_t carriers,
                                                         const py::array_t<double> &pilot_grid,
                                                         const py::array_t<arma::uword> &snap)
{
    // Return empty array if input is empty
    if (coeff.size() == 0)
        return py::array_t<std::complex<double>>();

    // Check if sizes match
    if (coeff.size() != delay.size())
        throw std::invalid_argument("Size of 'delay' does not match size of 'coeff'.");

    // Generate pilot grid
    arma::vec pilot_grid_arma;
    if (pilot_grid.size() == 0)
    {
        if (carriers == 0)
            throw std::invalid_argument("Number of carriers cannot be 0.");
        pilot_grid_arma = arma::linspace<arma::vec>(0.0, 1.0, (arma::uword)carriers);
    }
    else
        pilot_grid_arma = qd_python_numpy2arma_Col(pilot_grid, true);

    // Get snapshot indices
    arma::uword n_coeff = coeff.size();
    arma::uvec snap_arma = (snap.size() == 0) ? arma::regspace<arma::uvec>(0, n_coeff - 1) : qd_python_numpy2arma_Col(snap, true);
    arma::uword n_snap = snap_arma.n_elem;

    // List data access
    std::vector<std::complex<double> *> coeff_pointers;
    std::vector<py::array_t<std::complex<double>>> coeff_arrays;

    std::vector<double *> delay_pointers;
    std::vector<py::array_t<double>> delay_arrays;

    auto coeff_shape = qd_python_get_list_shape(coeff, coeff_pointers, coeff_arrays);
    auto delay_shape = qd_python_get_list_shape(delay, delay_pointers, delay_arrays);

    arma::uword n_rx = coeff_shape[0][0];
    arma::uword n_tx = coeff_shape[0][1];
    arma::uword n_carrier = pilot_grid_arma.n_elem;
    arma::uword n_data = n_rx * n_tx * n_carrier;

    // Sanity check
    for (arma::uword i = 0; i < n_coeff; ++i)
    {
        if (coeff_shape[i][7] == 0)
            continue;

        if (coeff_shape[i][0] != n_rx || coeff_shape[i][1] != n_tx)
            throw std::invalid_argument("MIMO matrix dimensions do not match.");

        arma::uword n_path = coeff_shape[i][2];
        if (!(delay_shape[i][0] == n_rx && delay_shape[i][1] == n_tx && delay_shape[i][2] == n_path) &&
            !(delay_shape[i][0] == 1 && delay_shape[i][1] == 1 && delay_shape[i][2] == n_path))
            throw std::invalid_argument("Input 'delay' must match the size as 'coeff'");
    }

    // Allocate memory for the output
    auto output = qd_python_init_output<std::complex<double>>(n_rx, n_tx, n_carrier, n_snap);
    std::complex<double> *p_data = static_cast<std::complex<double> *>(output.mutable_data());

// Calculate DFT
#pragma omp parallel for
    for (arma::uword i_snap = 0; i_snap < n_snap; ++i_snap)
    {
        arma::uword j_snap = snap_arma[i_snap];

        arma::cx_cube hmat(&p_data[i_snap * n_data], n_rx, n_tx, n_carrier, false, true);
        if (coeff_shape[j_snap][7] == 0)
        {
            hmat.zeros();
            continue;
        }

        arma::cube coeff_re, coeff_im, delay_re;
        qd_python_copy2arma(coeff_pointers[j_snap], coeff_shape[j_snap], coeff_re, coeff_im);
        qd_python_copy2arma(delay_pointers[j_snap], delay_shape[j_snap], delay_re);

        quadriga_lib::baseband_freq_response<double>(&coeff_re, &coeff_im, &delay_re, &pilot_grid_arma,
                                                     bandwidth, nullptr, nullptr, &hmat);
    }

    return output;
}