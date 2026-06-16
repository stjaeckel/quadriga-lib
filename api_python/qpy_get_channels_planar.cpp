// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_channels_planar
Calculate MIMO channel coefficients for planar wave paths

- Computes complex channel coefficients and delays for all TX/RX port pairs across `n_path` propagation
  paths under the planar wave assumption.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based; the input angles are not used for LOS detection.
- Polarization coupling is applied via the 8-row transfer matrix `M` (interleaved Re/Im for VV, VH, HV, HH).
- If `center_freq == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == False`, the straight-line TX-RX delay (LOS delay) is subtracted from all paths.
- If `add_fake_los_path == True`, a zero-power LOS path is prepended when none is present, making the
  output size `n_path + 1`.
- `complex=True` returns one combined complex coefficient array `coeff`; `complex=False` (default) returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path.

## Usage:
```
coeff_re, coeff_im, delays, rx_Doppler = quadriga_lib.arrayant.get_channels_planar( ant_tx, ant_rx, \
    aod, eod, aoa, eoa, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, \
    center_freq, use_absolute_delays, add_fake_los_path )

coeff, delays, rx_Doppler = quadriga_lib.arrayant.get_channels_planar( ..., complex=True )
```

## Inputs:
- **`ant_tx`** — Transmit arrayant dict; see [[generate]]
- **`ant_rx`** — Receive arrayant dict; see [[generate]]
- **`aod`** — Departure azimuth angles in rad; `(n_path,)`
- **`eod`** — Departure elevation angles in rad; `(n_path,)`
- **`aoa`** — Arrival azimuth angles in rad; `(n_path,)`
- **`eoa`** — Arrival elevation angles in rad; `(n_path,)`
- **`path_gain`** — Path gains in linear scale; `(n_path,)`
- **`path_length`** — Total path lengths from TX to RX phase center; `(n_path,)`
- **`M`** — Polarization transfer matrix, interleaved Re/Im; `(8, n_path)` (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH)
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`center_freq`** — Center frequency in Hz; set to `0` to skip phase computation; default: `0.0`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, prepends a zero-power LOS path when none is present; default: `False`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `False`

## Outputs:
- **`coeff_re`** — Real part of channel coefficients (`complex=False`); `(n_ports_rx, n_ports_tx, n_path)`
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape as `coeff_re`
- **`coeff`** — Complex channel coefficients (`complex=True`), replaces `coeff_re`/`coeff_im`; same shape
- **`delays`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`rx_Doppler`** — Doppler weights for moving RX; positive = toward path, negative = away; `(n_path,)`

## See also:
- [[get_channels_spherical]] (spherical wave variant)
- [[get_channels_ieee_indoor]] (IEEE compliant channels using `get_channels_planar` internally)
- [[generate]] (antenna array generator)
MD!*/

py::tuple get_channels_planar(const py::dict &ant_tx,
                              const py::dict &ant_rx,
                              const py::array_t<double> &aod,
                              const py::array_t<double> &eod,
                              const py::array_t<double> &aoa,
                              const py::array_t<double> &eoa,
                              const py::array_t<double> &path_gain,
                              const py::array_t<double> &path_length,
                              const py::array_t<double> &M,
                              const py::array_t<double> &tx_pos,
                              const py::array_t<double> &tx_orientation,
                              const py::array_t<double> &rx_pos,
                              const py::array_t<double> &rx_orientation,
                              const double center_freq,
                              const bool use_absolute_delays,
                              const bool add_fake_los_path,
                              const bool complex)
{
    // Parse input arguments
    const auto ant_tx_a = qd_python_dict2arrayant(ant_tx, true);
    const auto ant_rx_a = qd_python_dict2arrayant(ant_rx, true);
    const auto aod_a = qd_python_numpy2arma_Col(aod, true);
    const auto eod_a = qd_python_numpy2arma_Col(eod, true);
    const auto aoa_a = qd_python_numpy2arma_Col(aoa, true);
    const auto eoa_a = qd_python_numpy2arma_Col(eoa, true);
    const auto path_gain_a = qd_python_numpy2arma_Col(path_gain, true);
    const auto path_length_a = qd_python_numpy2arma_Col(path_length, true);
    const auto M_a = qd_python_numpy2arma_Mat(M, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation, true, false, "rx_orientation", 3);

    // Extract scalar values
    double Tx = tx_pos_a.at(0), Ty = tx_pos_a.at(1), Tz = tx_pos_a.at(2);
    double Tb = tx_orientation_a.at(0), Tt = tx_orientation_a.at(1), Th = tx_orientation_a.at(2);
    double Rx = rx_pos_a.at(0), Ry = rx_pos_a.at(1), Rz = rx_pos_a.at(2);
    double Rb = rx_orientation_a.at(0), Rt = rx_orientation_a.at(1), Rh = rx_orientation_a.at(2);

    // Derived inputs
    arma::uword n_ports_tx = ant_tx_a.n_ports();
    arma::uword n_ports_rx = ant_rx_a.n_ports();
    arma::uword n_path = add_fake_los_path ? aod_a.n_elem + 1 : aod_a.n_elem;

    // Initialize delay and Doppler outputs (always real, zero-copy)
    arma::cube coeff_re, coeff_im, delay;
    arma::vec rx_Doppler;
    auto delay_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &delay);
    auto rx_Doppler_p = qd_python_init_output<double>(n_path, &rx_Doppler);

    if (complex)
    {
        quadriga_lib::get_channels_planar<double>(&ant_tx_a, &ant_rx_a,
                                                  Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, Rz, Rb, Rt, Rh,
                                                  &aod_a, &eod_a, &aoa_a, &eoa_a, &path_gain_a, &path_length_a, &M_a,
                                                  &coeff_re, &coeff_im, &delay,
                                                  center_freq, use_absolute_delays, add_fake_los_path,
                                                  &rx_Doppler);

        auto coeff_p = qd_python_copy2numpy<double, std::complex<double>>(&coeff_re, &coeff_im);
        return py::make_tuple(coeff_p, delay_p, rx_Doppler_p);
    }

    // Real path: zero-copy coeff outputs written in place by C++
    auto coeff_re_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_re);
    auto coeff_im_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_im);

    quadriga_lib::get_channels_planar<double>(&ant_tx_a, &ant_rx_a,
                                              Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, Rz, Rb, Rt, Rh,
                                              &aod_a, &eod_a, &aoa_a, &eoa_a, &path_gain_a, &path_length_a, &M_a,
                                              &coeff_re, &coeff_im, &delay,
                                              center_freq, use_absolute_delays, add_fake_los_path,
                                              &rx_Doppler);

    return py::make_tuple(coeff_re_p, coeff_im_p, delay_p, rx_Doppler_p);
}

// pybind11 declaration:
// m.def("get_channels_planar", &get_channels_planar,
//       py::arg("ant_tx"),
//       py::arg("ant_rx"),
//       py::arg("aod"),
//       py::arg("eod"),
//       py::arg("aoa"),
//       py::arg("eoa"),
//       py::arg("path_gain"),
//       py::arg("path_length"),
//       py::arg("M"),
//       py::arg("tx_pos"),
//       py::arg("tx_orientation"),
//       py::arg("rx_pos"),
//       py::arg("rx_orientation"),
//       py::arg("center_freq") = 0.0,
//       py::arg("use_absolute_delays") = false,
//       py::arg("add_fake_los_path") = false,
//       py::arg("complex") = false);
