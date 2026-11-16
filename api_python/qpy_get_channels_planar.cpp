// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# channel.planar
Calculate MIMO channel coefficients for planar wave paths

- Computes complex channel coefficients and delays for all TX/RX port pairs across `n_path` propagation
  paths under the planar wave assumption.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based; the input angles are not used for LOS detection.
- Polarization coupling is applied via the 8-row transfer matrix `M`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- Path data is given for `n_freq` frequencies; single-frequency data uses `n_freq = 1`.
- The scalar argument `i_freq` selects the frequency index that is processed; a single frequency is returned per call.
- `center_freq = None` or a selected entry of `0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual.
- If `use_absolute_delays == False`, the straight-line TX-RX delay between the phase centres (LOS delay) is subtracted
  from all paths; per-element delays of pairs that are closer to each other than the phase centres are then
  negative, down to about -(half the TX array size + half the RX array size) / speed of light; [[quantize_delays]] rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.
- If `add_fake_los_path == True`, one output path is always added (output size `n_path + 1`): slot 0 holds the LOS
  path, with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the
  TX-RX distance; it is moved to slot 0 and its original slot (input index + 1) holds a zero-power path. All other
  paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0)`; with `add_fake_los_path = True` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- `complex=True` (default) returns one combined complex coefficient array `coeff`; `complex=False` returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path.
- The flag `complex` takes a bool or `numpy.bool_`; `None` selects the default, any other type raises `TypeError`.

## Usage:
```
coeff, delays, rx_Doppler = quadriga_lib.channel.planar( ant_tx, ant_rx, \
    aod, eod, aoa, eoa, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, \
    center_freq, use_absolute_delays, add_fake_los_path )

coeff_re, coeff_im, delays, rx_Doppler = quadriga_lib.channel.planar( ..., complex=False )

coeff, delays, rx_Doppler = quadriga_lib.channel.planar( ..., i_freq=1 )
```

## Inputs:
- **`ant_tx`** — Transmit arrayant dict; see [[generate]]
- **`ant_rx`** — Receive arrayant dict; see [[generate]]
- **`aod`** — Departure azimuth angles in rad; `(n_path,)`
- **`eod`** — Departure elevation angles in rad; `(n_path,)`
- **`aoa`** — Arrival azimuth angles in rad; `(n_path,)`
- **`eoa`** — Arrival elevation angles in rad; `(n_path,)`
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`
- **`path_length`** — Total path lengths from TX to RX phase center; `(n_path,)`
- **`M`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; an entry of `0` or `None` gives the
  coefficients at f = 0 (see above); default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `True`
- **`i_freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `M` and `center_freq` that is
  processed; default: `0`

## Outputs:
- **`coeff`** — Complex channel coefficients (`complex=True`); `(n_ports_rx, n_ports_tx, n_path)`
- **`coeff_re`** — Real part of channel coefficients (`complex=False`), replaces `coeff`; same shape
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape
- **`delays`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`rx_Doppler`** — Doppler weights for moving RX; positive = toward path, negative = away; `(n_path,)`

## See also:
- [[channel.spherical]] (spherical wave variant)
- [[channel.ieee_indoor]] (IEEE compliant channels using `channel.planar` internally)
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
                              py::handle center_freq,
                              const bool use_absolute_delays,
                              const bool add_fake_los_path,
                              py::handle complex,
                              const arma::uword i_freq)
{
    const bool want_complex = qd_python_flag(complex, true, "complex");

    // Parse input arguments
    const auto ant_tx_a = qd_python_dict2arrayant(ant_tx, true);
    const auto ant_rx_a = qd_python_dict2arrayant(ant_rx, true);
    const auto aod_a = qd_python_numpy2arma_Col(aod, true);
    const auto eod_a = qd_python_numpy2arma_Col(eod, true);
    const auto aoa_a = qd_python_numpy2arma_Col(aoa, true);
    const auto eoa_a = qd_python_numpy2arma_Col(eoa, true);
    const auto path_gain_a = qd_python_numpy2arma_Mat(path_gain, true);
    const auto path_length_a = qd_python_numpy2arma_Col(path_length, true);
    const auto M_a = qd_python_numpy2arma_Cube(M, true);
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation, true, false, "rx_orientation", 3);

    // Derived inputs
    arma::uword n_ports_tx = ant_tx_a.n_ports();
    arma::uword n_ports_rx = ant_rx_a.n_ports();
    arma::uword n_path_in = aod_a.n_elem;
    arma::uword n_freq = path_gain_a.n_cols;
    arma::uword n_path = add_fake_los_path ? n_path_in + 1 : n_path_in;

    // Check multi-frequency data consistency, path counts are validated by the C++ core
    if (M_a.n_slices != n_freq)
        throw std::invalid_argument("Inputs 'path_gain' and 'M' must have the same number of frequencies.");

    if (!center_freq_a.empty() && center_freq_a.n_elem != n_freq)
        throw std::invalid_argument("Input 'center_freq' must have 'n_freq' elements.");

    if (i_freq >= n_freq)
        throw std::out_of_range("Input 'i_freq' exceeds the number of frequencies.");

    // Select the frequency slice, aliased without copying the data
    const arma::Col<double> path_gain_f(const_cast<double *>(path_gain_a.colptr(i_freq)), path_gain_a.n_rows, false, true);
    const arma::Mat<double> M_f(const_cast<double *>(M_a.slice_memptr(i_freq)), M_a.n_rows, M_a.n_cols, false, true);
    const double center_freq_sel = center_freq_a.empty() ? 0.0 : center_freq_a.at(i_freq);

    // Initialize delay and Doppler outputs (always real, zero-copy)
    arma::cube coeff_re, coeff_im, delay;
    arma::vec rx_Doppler;
    auto delay_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &delay);
    auto rx_Doppler_p = qd_python_init_output<double>(n_path, &rx_Doppler);

    if (want_complex)
    {
        quadriga_lib::get_channels_planar<double>(ant_tx_a, ant_rx_a,
                                                  tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                                  aod_a, eod_a, aoa_a, eoa_a, path_gain_f, path_length_a, M_f,
                                                  coeff_re, coeff_im, delay,
                                                  center_freq_sel, use_absolute_delays, add_fake_los_path,
                                                  &rx_Doppler);

        auto coeff_p = qd_python_copy2numpy<double, std::complex<double>>(&coeff_re, &coeff_im);
        return py::make_tuple(coeff_p, delay_p, rx_Doppler_p);
    }

    // Real path: zero-copy coeff outputs written in place by C++
    auto coeff_re_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_re);
    auto coeff_im_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_im);

    quadriga_lib::get_channels_planar<double>(ant_tx_a, ant_rx_a,
                                              tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                              aod_a, eod_a, aoa_a, eoa_a, path_gain_f, path_length_a, M_f,
                                              coeff_re, coeff_im, delay,
                                              center_freq_sel, use_absolute_delays, add_fake_los_path,
                                              &rx_Doppler);

    return py::make_tuple(coeff_re_p, coeff_im_p, delay_p, rx_Doppler_p);
}
