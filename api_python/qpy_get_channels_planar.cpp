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
- Errors in the path data of a snapshot start with `Snapshot <i>: ` (0-based), also for a single snapshot
  (`Snapshot 0: `); antenna and `use_kernel` errors have no prefix.

## Batch mode:
- If `aod` is a `list` of arrays, the function switches to batch mode and processes `n_snap` snapshots in one call,
  the snapshots in parallel (OpenMP in C++).
- The path inputs `aod`, `eod`, `aoa`, `eoa`, `path_gain`, `path_length` and `M` are then lists of length `n_snap`;
  list entries that are Fortran-contiguous float64 arrays are read in place, other entries are copied.
- The position and orientation inputs are `(3, n_snap)` arrays with one column per snapshot; a `(3,)` array or a
  single-column `(3, 1)` array is applied to all snapshots.
- All outputs are returned as lists of length `n_snap`, one entry per snapshot.
- The GIL is released while processing, so the function can be combined with Python threads. The function keeps
  references to all input arrays until it returns, but other threads must not write into them meanwhile.

## Usage:
```
coeff, delay, rx_Doppler = quadriga_lib.channel.planar( tx_array, rx_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa, path_gain, path_length, M, \
    center_freq, *, use_absolute_delays, add_fake_los_path, use_kernel, i_freq )
```

## Inputs:
- **`tx_array`** — Transmit arrayant dict with single-frequency (3D) patterns, see [[generate]]; a 4D dict raises
  `ValueError` (use [[channel.multifreq]])
- **`rx_array`** — Receive arrayant dict with single-frequency (3D) patterns; as `tx_array`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`, or `(3, n_snap)` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`, or `(3, n_snap)` in batch mode
- **`aod`** — Departure azimuth angles in rad; `(n_path,)`, or a list of length `n_snap` in batch mode
- **`eod`** — Departure elevation angles in rad; `(n_path,)`, or a list in batch mode
- **`aoa`** — Arrival azimuth angles in rad; `(n_path,)`, or a list in batch mode
- **`eoa`** — Arrival elevation angles in rad; `(n_path,)`, or a list in batch mode
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`, or a list in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; `(n_path,)`, or a list in batch mode
- **`M`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`, or a list in batch mode
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; an entry of `0` or `None` gives the
  coefficients at f = 0 (see above); default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single precision; raises if
  unavailable); auto selects GENERIC in this version, see [[interpolate]]; `int`; default: 0
- **`i_freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `M` and `center_freq` that is
  processed; Python only, since the C++ function takes single-frequency path data; default: `0`

## Outputs:
- **`coeff`** — Complex channel coefficients; `(n_ports_rx, n_ports_tx, n_path)`
- **`delay`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`rx_Doppler`** — Doppler weights for moving RX; positive = toward path, negative = away; `(n_path,)`

## See also:
- [[channel.spherical]] (spherical wave variant)
- [[channel.ieee_indoor]] (IEEE compliant channels using `channel.planar` internally)
- [[generate]] (antenna array generator)
MD!*/

py::tuple get_channels_planar(const py::dict &tx_array,
                              const py::dict &rx_array,
                              py::handle tx_pos,
                              py::handle tx_orientation,
                              py::handle rx_pos,
                              py::handle rx_orientation,
                              py::handle aod,
                              py::handle eod,
                              py::handle aoa,
                              py::handle eoa,
                              py::handle path_gain,
                              py::handle path_length,
                              py::handle M,
                              py::handle center_freq,
                              py::handle use_absolute_delays,
                              py::handle add_fake_los_path,
                              const int use_kernel,
                              const arma::uword i_freq)
{
    const bool use_absolute_delays_b = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool add_fake_los_path_b = qd_python_flag(add_fake_los_path, false, "add_fake_los_path");

    // References to every aliased Python array, held until the function returns (the GIL is released below)
    std::vector<py::object> keep_alive;
    qd_python_keep_dict_values(tx_array, keep_alive);
    qd_python_keep_dict_values(rx_array, keep_alive);

    const auto ant_tx_a = qd_python_dict2arrayant(tx_array, true);
    const auto ant_rx_a = qd_python_dict2arrayant(rx_array, true);
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const arma::uword n_ports_tx = ant_tx_a.n_ports(), n_ports_rx = ant_rx_a.n_ports();

    // Batch mode for lists of path arrays; a single snapshot is processed as a batch of one
    const bool batch = qd_python_is_array_list(aod);
    if (batch && (!qd_python_is_array_list(eod) || !qd_python_is_array_list(aoa) || !qd_python_is_array_list(eoa) ||
                  !qd_python_is_array_list(path_gain) || !qd_python_is_array_list(path_length) || !qd_python_is_array_list(M)))
        throw std::invalid_argument("Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length' and 'M' must all be lists.");

    // Alias the path data, no data is copied
    const auto aod_a = qd_python_alias_list<arma::Col<double>>(aod, batch, keep_alive);
    const auto eod_a = qd_python_alias_list<arma::Col<double>>(eod, batch, keep_alive);
    const auto aoa_a = qd_python_alias_list<arma::Col<double>>(aoa, batch, keep_alive);
    const auto eoa_a = qd_python_alias_list<arma::Col<double>>(eoa, batch, keep_alive);
    const auto path_gain_a = qd_python_alias_list<arma::Mat<double>>(path_gain, batch, keep_alive);
    const auto path_length_a = qd_python_alias_list<arma::Col<double>>(path_length, batch, keep_alive);
    const auto M_a = qd_python_alias_list<arma::Cube<double>>(M, batch, keep_alive);
    const arma::uword n_snap = (arma::uword)aod_a.size();

    // Needed before the frequency check below, which indexes the lists
    if (eod_a.size() != n_snap || aoa_a.size() != n_snap || eoa_a.size() != n_snap ||
        path_gain_a.size() != n_snap || path_length_a.size() != n_snap || M_a.size() != n_snap)
        throw std::invalid_argument("Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length' and 'M' must have the same length.");

    // Positions and orientations: one column per snapshot or one for all (checked by the C++ core); a single snapshot
    // takes any 3-element array
    auto position = [&](const py::handle &h, const char *name)
    { return batch ? qd_python_numpy2arma_Mat<double>(h, true) : arma::Mat<double>(qd_python_numpy2arma_Col<double>(h.cast<py::array_t<double>>(), false, false, name, 3)); };
    const arma::Mat<double> tx_pos_a = position(tx_pos, "tx_pos"), tx_orientation_a = position(tx_orientation, "tx_orientation");
    const arma::Mat<double> rx_pos_a = position(rx_pos, "rx_pos"), rx_orientation_a = position(rx_orientation, "rx_orientation");

    // Frequency slice of every snapshot, aliased without copying; path counts are validated by the C++ core
    std::vector<arma::vec> path_gain_f;
    std::vector<arma::mat> M_f;
    const double center_freq_sel = qd_python_select_freq(path_gain_a, M_a, center_freq_a, i_freq, path_gain_f, M_f);

    // Outputs: numpy memory is mapped to the Armadillo objects, which C++ writes in place
    std::vector<arma::cube> coeff_re, coeff_im, delay(n_snap);
    std::vector<arma::cx_cube> coeff(n_snap);
    std::vector<arma::vec> rx_Doppler(n_snap);
    py::list coeff_py, delay_py, rx_Doppler_py;
    for (arma::uword i = 0ULL; i < n_snap; ++i)
    {
        const arma::uword n_path = add_fake_los_path_b ? aod_a[i].n_elem + 1ULL : aod_a[i].n_elem;
        coeff_py.append(qd_python_init_output(n_ports_rx, n_ports_tx, n_path, &coeff[i]));
        delay_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &delay[i]));
        rx_Doppler_py.append(qd_python_init_output<double>(n_path, &rx_Doppler[i]));
    }

    {
        py::gil_scoped_release release;
        quadriga_lib::get_channels_planar_vec<double>(ant_tx_a, ant_rx_a, tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                                      aod_a, eod_a, aoa_a, eoa_a, path_gain_f, path_length_a, M_f,
                                                      coeff_re, coeff_im, delay,
                                                      center_freq_sel, use_absolute_delays_b, add_fake_los_path_b,
                                                      &rx_Doppler, &coeff, use_kernel);
    }

    // Lists in batch mode, the arrays of the one snapshot otherwise
    if (batch)
        return py::make_tuple(coeff_py, delay_py, rx_Doppler_py);
    return py::make_tuple(coeff_py[0], delay_py[0], rx_Doppler_py[0]);
}
