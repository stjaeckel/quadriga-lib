// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# channel.multifreq
Compute channel coefficients for spherical waves across multiple frequencies

- Multi-frequency extension of [[channel.spherical]] with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) is computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (`tx_array['center_freq']`), RX array (`rx_array['center_freq']`), input samples (`freq`), and output (`freq_new`)
- Antenna pattern fields may be 3D (single-frequency, clamped for all output frequencies) or 4D (multi-frequency, 4th axis = frequency)
- TX/RX patterns are interpolated per output frequency via SLERP with linear fallback
- `path_gain` is interpolated linearly; `M` is interpolated via SLERP per complex entry pair to preserve phase
- Coupling matrices are interpolated across frequencies (SLERP for complex pairs); pass 3D coupling `(n_elem, n_ports, n_freq)` for per-frequency coupling
- Extrapolation clamps to the nearest frequency entry on all four grids
- `M` accepts 8 rows (full polarimetric: the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx),
  rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx)) or 2 rows (scalar pressure: Re, Im of the (V rx, V tx) entry only)
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS) is added to the
  FBS-LBS segment. Delays and phases follow this length; the per-element offsets follow the FBS / LBS geometry
- `propagation_speed` supports EM (speed of light, default) and acoustic (343 m/s) simulations
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`: with `add_fake_los_path`, slot 0 holds the LOS path
  (zero power if no LOS path was found; LOS = the path whose length is closest to the TX-RX distance, within 0.1 mm);
  a LOS path found in the input is moved to slot 0 and leaves a zero-power path in its original slot
  (input index + 1); all other paths keep their order, shifted by one slot
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0, n_freq_new)`; with
  `add_fake_los_path = True` they hold only this zero-power LOS path, with the LOS delays per antenna pair
- If `use_absolute_delays` is false, the LOS delay between the TX and RX phase centres is subtracted from all paths;
  per-element delays can then be negative, down to about -(half the TX array size + half the RX array size) /
  `propagation_speed` (both arrays endfire). [[quantize_delays]] rejects delays at or below `-tap_spacing / 2`. For
  acoustics (`propagation_speed` about 343 m/s) that happens for practically every array: at 48 kHz taps, half sizes
  adding up to more than about 3.6 mm (a 5 cm microphone pair already exceeds it); for radio at 5 ns taps, about
  0.75 m. Use `use_absolute_delays = true` in that case
- The coefficients always carry the phase of the absolute path length, also when `use_absolute_delays` is false.
  [[baseband_freq_response]] with multi-frequency input and `remove_delay_phase = True` removes the phase of `delay`
  before it interpolates across `freq`, so it needs channels generated with `use_absolute_delays = True`; with
  relative delays, the phase of the TX-RX distance stays in the coefficients, is interpolated across the frequency
  grid and gives wrong phases
- `complex=True` (default) returns one combined complex coefficient array `coeff`; `complex=False` returns
  separate real `coeff_re` and `coeff_im`

## Usage:
```
coeff, delay = quadriga_lib.channel.multifreq( tx_array, rx_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, M, \
    freq, freq_new, use_absolute_delays, add_fake_los_path, propagation_speed, use_kernel, complex=True )

coeff_re, coeff_im, delay = quadriga_lib.channel.multifreq( ..., complex=False )
```

## Inputs:
- **`tx_array`** — Multi-frequency TX arrayant dict; pattern fields 3D `(n_el, n_az, n_elem)` or 4D `(n_el, n_az, n_elem, n_freq)`; see [[generate]];
  4D input requires `center_freq` with one value per frequency entry `(n_freq,)`
- **`rx_array`** — RX arrayant dict; same format as `tx_array`
- **`tx_pos`** — Transmitter position, finite; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading), finite; `(3,)`
- **`rx_pos`** — Receiver position, finite; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading), finite; `(3,)`
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`
- **`path_gain`** — Linear-scale path gains, one column per input frequency; `(n_path, n_freq)`
- **`path_length`** — Absolute TX-to-RX path lengths (between the phase centres); values shorter than the
  TX-FBS-LBS-RX polyline (e.g. 0) use the polyline length; `(n_path,)`
- **`M`** — Polarization transfer matrix, row order as above; `(8, n_path, n_freq)` full pol or
  `(2, n_path, n_freq)` scalar pressure
- **`freq`** — Input sample frequencies for `path_gain` and `M`; `(n_freq,)`
- **`freq_new`** — Target output frequencies; `(n_freq_new,)`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`propagation_speed`** — Wave speed in m/s; use ~343.0 for acoustics; default: `299792458.0`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single precision; raises if
  unavailable); auto selects GENERIC in this version, see [[interpolate]]; `int`; default: 0
- **`complex`** — Keyword-only; if `True`, combine coefficients into a single complex array `coeff`; if `False`,
  return separate `coeff_re` and `coeff_im`; default: `True`

## Outputs:
- **`coeff`** — Complex channel coefficients (`complex=True`); `(n_ports_rx, n_ports_tx, n_path_out, n_freq_new)`
- **`coeff_re`** — Real part of channel coefficients (`complex=False`), replaces `coeff`; same shape
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape
- **`delay`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path_out, n_freq_new)`

## See also:
- [[channel.spherical]] (single-frequency equivalent)
- [[generate_speaker]] (acoustic source construction)
MD!*/

py::tuple get_channels_multifreq(const py::dict &tx_array,
                                 const py::dict &rx_array,
                                 const py::array_t<double> &tx_pos,
                                 const py::array_t<double> &tx_orientation,
                                 const py::array_t<double> &rx_pos,
                                 const py::array_t<double> &rx_orientation,
                                 const py::array_t<double> &fbs_pos,
                                 const py::array_t<double> &lbs_pos,
                                 const py::array_t<double> &path_gain,
                                 const py::array_t<double> &path_length,
                                 const py::array_t<double> &M,
                                 const py::array_t<double> &freq,
                                 const py::array_t<double> &freq_new,
                                 py::handle use_absolute_delays,
                                 py::handle add_fake_los_path,
                                 const double propagation_speed,
                                 const int use_kernel,
                                 py::handle complex)
{
    const bool use_absolute_delays_b = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool add_fake_los_path_b = qd_python_flag(add_fake_los_path, false, "add_fake_los_path");

    const bool want_complex = qd_python_flag(complex, true, "complex");

    // Parse antenna dictionaries (3D single-freq or 4D multi-freq handled internally)
    auto ant_tx_vec = qd_python_dict2arrayant_multi(tx_array, true);
    auto ant_rx_vec = qd_python_dict2arrayant_multi(rx_array, true);

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
    const auto freq_a = qd_python_numpy2arma_Col(freq, true);
    const auto freq_new_a = qd_python_numpy2arma_Col(freq_new, true);

    // Derived output sizes
    arma::uword n_ports_tx = ant_tx_vec[0].n_ports();
    arma::uword n_ports_rx = ant_rx_vec[0].n_ports();
    arma::uword n_path = add_fake_los_path_b ? fbs_pos_a.n_cols + 1 : fbs_pos_a.n_cols;
    arma::uword n_freq_new = freq_new_a.n_elem;

    std::vector<arma::cube> coeff_re, coeff_im, delay;

    if (want_complex)
    {
        // Complex output written in place by C++ (0 paths: the zero-power LOS path with add_fake_los_path, else
        // empty outputs)
        std::vector<arma::cx_cube> coeff;
        auto coeff_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_new, &coeff);
        auto delay_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_new, &delay);
        quadriga_lib::get_channels_multifreq<double>(ant_tx_vec, ant_rx_vec,
                                                     tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                                     fbs_pos_a, lbs_pos_a, path_gain_a, path_length_a, M_a,
                                                     freq_a, freq_new_a,
                                                     coeff_re, coeff_im, delay,
                                                     use_absolute_delays_b, add_fake_los_path_b, propagation_speed, &coeff,
                                                     use_kernel);
        return py::make_tuple(coeff_py, delay_py);
    }

    // Allocate 4D outputs, frequency on the 4th axis (zero-copy, written in place by C++)
    auto coeff_re_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_new, &coeff_re);
    auto coeff_im_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_new, &coeff_im);
    auto delay_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_new, &delay);

    // Call C++ library function (0 paths: the zero-power LOS path with add_fake_los_path, else empty outputs)
    quadriga_lib::get_channels_multifreq<double>(ant_tx_vec, ant_rx_vec,
                                                 tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                                 fbs_pos_a, lbs_pos_a, path_gain_a, path_length_a, M_a,
                                                 freq_a, freq_new_a,
                                                 coeff_re, coeff_im, delay,
                                                 use_absolute_delays_b, add_fake_los_path_b, propagation_speed, nullptr,
                                                 use_kernel);

    return py::make_tuple(coeff_re_py, coeff_im_py, delay_py);
}
