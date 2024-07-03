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
# BASEBAND_FREQ_RESPONSE
Transforms the channel into frequency domain and returns the frequency response

## Usage:

```
hmat = quadriga_lib.baseband_freq_response( coeff, delay, bandwidth, carriers, pilot_grid, snap );
```

## Input Arguments:
- **`coeff`**<br>
  Channel coefficients, complex-valued, Size: `[ n_rx, n_tx, n_path, n_snap ]`

- **`delay`**<br>
  Propagation delay in seconds, Size: `[ n_rx, n_tx, n_path, n_snap ]` or `[ 1, 1, n_path, n_snap ]`

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
  carriers. Vector of length: `[ n_carriers ]`

- **`snap`** (optional)<br>
  Snapshot indices for which the frequency response should be generated (1-based index). If this
  variable is not given, all snapshots are processed. Length: `[ n_out ]`

## Output Argument:
- **`hmat`**<br>
  Freq. domain channel matrices (H), complex-valued, Size `[ n_rx, n_tx, n_carriers, n_out ]`
MD!*/



pybind11::array_t<std::complex<double>> baseband_freq_response(const pybind11::array_t<std::complex<double>> coeff,
                                                               const pybind11::array_t<double> delay,
                                                               const double bandwidth,
                                                               const size_t carriers,
                                                               const pybind11::array_t<double> pilot_grid,
                                                               const pybind11::array_t<unsigned> snap)
{
    std::vector<arma::cube> coeff_re, coeff_im, delay_arma, hmat_re, hmat_im;
    arma::vec pilot_grid_arma;
    arma::u32_vec i_snap_arma;

    if (coeff.size() != 0)
        qd_python_complexNPArray_to_2vectorCube(&coeff, 3, &coeff_re, &coeff_im);

    size_t n_snap = coeff_re.size();
    if (delay.size() != 0)
    {
        pybind11::buffer_info buf = delay.request();
        size_t n_dim = (size_t)buf.ndim;
        size_t n_cols = (n_dim < 2) ? 1 : (size_t)buf.shape[1];

        if (n_dim == 2 && n_cols == n_snap) // Compact mode
        {
            auto tmp = qd_python_NPArray_to_vectorCube(&delay, 1);
            for (auto &d : tmp)
                delay_arma.push_back(arma::cube(d.memptr(), 1, 1, d.n_elem, true));
        }
        else
            delay_arma = qd_python_NPArray_to_vectorCube(&delay, 3);
    }

    if (pilot_grid.size() == 0)
    {
        if (carriers == 0)
            throw std::invalid_argument("Number of carriers cannot be 0.");
        pilot_grid_arma = arma::linspace<arma::vec>(0.0, 1.0, (arma::uword)carriers);
    }
    else
        pilot_grid_arma = qd_python_NPArray_to_Col(&pilot_grid);

    if (snap.size() != 0)
        i_snap_arma = qd_python_NPArray_to_Col(&snap);

    quadriga_lib::baseband_freq_response_vec<double>(&coeff_re, &coeff_im, &delay_arma, &pilot_grid_arma,
                                                     bandwidth, &hmat_re, &hmat_im, &i_snap_arma);

    return qd_python_2vectorCubes_to_complexNPArray(&hmat_re, &hmat_im);
}