// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# channel.spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths,
  using the spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation
  (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `polmat`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx).
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS, e.g. paths of
  [[qrt_file_read]] with more than two interactions) is added to the FBS-LBS segment. Delays and phases follow this
  length; the per-element offsets follow the FBS / LBS geometry.
- Path data is given for `n_freq` frequencies; single-frequency data uses `n_freq = 1`.
- The scalar argument `i_freq` selects the frequency index that is processed; a single frequency is returned per call.
- `center_freq = None` or a selected entry of `0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual.
- If `use_absolute_delays == False`, the LOS delay between the TX and RX phase centres is subtracted from all paths.
  Delays are per element pair, so pairs that are closer to each other than the phase centres get negative delays,
  down to about -(half the TX array size + half the RX array size) / speed of light (both arrays endfire).
  [[quantize_delays]] maps delays above `-tap_spacing / 2` to tap 0 and rejects lower ones (at 5 ns taps: half sizes
  adding up to more than about 0.75 m); use `use_absolute_delays = true` in that case.
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm.
- If `add_fake_los_path == True`, one output path is always added (`n_path + 1` outputs): slot 0 holds the LOS path,
  with zero power if no LOS path was found. If a LOS path was found, it is moved to slot 0 and its original slot
  (input index + 1) holds a zero-power path. All other paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0)`; with `add_fake_los_path = True` they
  hold only this zero-power LOS path, with the LOS delays and angles per antenna pair (also per snapshot in batch
  mode).
- Errors in the path data of a snapshot start with `Snapshot <i>: ` (0-based), also for a single snapshot
  (`Snapshot 0: `); antenna and `use_kernel` errors have no prefix.

## Batch mode:
- If `fbs_pos` is a `list` of arrays, the function switches to batch mode and processes `n_snap` snapshots
  in one call, the snapshots in parallel (OpenMP in C++).
- The path inputs `fbs_pos`, `lbs_pos`, `path_gain`, `path_length` and `polmat` are then lists of length `n_snap`,
  matching the output format of [[qrt_file_read]]; list entries that are Fortran-contiguous float64 arrays (as
  returned by [[qrt_file_read]]) are read in place, other entries are copied.
- The position and orientation inputs are `(3, n_snap)` arrays with one column per snapshot; a `(3,)` array
  or a single-column `(3, 1)` array is applied to all snapshots.
- All outputs are returned as lists of length `n_snap`, one entry per snapshot.
- Snapshots without any path produce an empty output entry with `n_path = 0`, or only the zero-power LOS path
  when `add_fake_los_path` is set (as for a single snapshot).
- The GIL is released while processing, so the function can be combined with Python threads. The function keeps
  references to all input arrays until it returns, but other threads must not write into them meanwhile.

## Usage:
```
coeff, delay = quadriga_lib.channel.spherical( tx_array, rx_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, \
    center_freq, *, use_absolute_delays, add_fake_los_path, angles, use_kernel, i_freq )

# Departure and arrival angles as well
coeff, delay, aod, eod, aoa, eoa = quadriga_lib.channel.spherical( ..., angles=True )
```

## Inputs:
- **`tx_array`** — Transmit arrayant dict with single-frequency (3D) patterns, see [[generate]]; a 4D dict raises
  `ValueError` (use [[channel.multifreq]])
- **`rx_array`** — Receive arrayant dict with single-frequency (3D) patterns; as `tx_array`
- **`tx_pos`** — Transmitter position in Cartesian coordinates, finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading), finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_pos`** — Receiver position in Cartesian coordinates, finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading), finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`, or a list of length `n_snap` in batch mode
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`, or a list of length `n_snap` in batch mode
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`, or a list of length `n_snap` in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `(n_path,)`, or a list of length `n_snap` in batch mode
- **`polmat`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`, or a list of length `n_snap`
  in batch mode
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; an entry of `0` or `None` gives the coefficients at f = 0 (see above); default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`angles`** — If `True`, also return the departure and arrival angles; `False` saves their memory and, with element
  coupling, the time to convert them to the ports; default: `False`
- **`use_kernel`** — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto, 1 = GENERIC,
  2 = AVX2 (single precision; raises if unavailable); auto selects GENERIC in this version, see [[interpolate]];
  `int`; default: 0
- **`i_freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `polmat` and `center_freq` that is
  processed; Python only, since the C++ function takes single-frequency path data; default: `0`

## Outputs:
- **`coeff`** — Complex channel coefficients; `(n_ports_rx, n_ports_tx, n_path)`
- **`delay`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`aod`** — Azimuth of departure in the GCS, in rad; `(n_ports_rx, n_ports_tx, n_path)`; only with `angles=True`
- **`eod`** — Elevation of departure in the GCS, in rad; same shape; only with `angles=True`
- **`aoa`** — Azimuth of arrival in the GCS, in rad; same shape; only with `angles=True`
- **`eoa`** — Elevation of arrival in the GCS, in rad; same shape; only with `angles=True`
- In batch mode, every output is a list of length `n_snap` whose entries have the shapes given above.

## See also:
- [[channel.multifreq]] (multi-frequency extension)
- [[channel.planar]] (planar wave variant)
- [[qrt_file_read]] (produces the list format used by batch mode)
MD!*/

py::tuple get_channels_spherical(const py::dict &tx_array,
                                 const py::dict &rx_array,
                                 py::handle tx_pos,
                                 py::handle tx_orientation,
                                 py::handle rx_pos,
                                 py::handle rx_orientation,
                                 py::handle fbs_pos,
                                 py::handle lbs_pos,
                                 py::handle path_gain,
                                 py::handle path_length,
                                 py::handle polmat,
                                 py::handle center_freq,
                                 py::handle use_absolute_delays,
                                 py::handle add_fake_los_path,
                                 py::handle angles,
                                 const int use_kernel,
                                 const arma::uword i_freq)
{
    const bool use_absolute_delays_b = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool add_fake_los_path_b = qd_python_flag(add_fake_los_path, false, "add_fake_los_path");
    const bool angles_b = qd_python_flag(angles, false, "angles");

    // References to every aliased Python array, held until the function returns (the GIL is released below)
    std::vector<py::object> keep_alive;
    qd_python_keep_dict_values(tx_array, keep_alive);
    qd_python_keep_dict_values(rx_array, keep_alive);

    const auto ant_tx_a = qd_python_dict2arrayant(tx_array, true);
    const auto ant_rx_a = qd_python_dict2arrayant(rx_array, true);
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const arma::uword n_ports_tx = ant_tx_a.n_ports(), n_ports_rx = ant_rx_a.n_ports();

    // Batch mode for lists of path arrays; a single snapshot is processed as a batch of one
    const bool batch = qd_python_is_array_list(fbs_pos);
    if (batch && (!qd_python_is_array_list(lbs_pos) || !qd_python_is_array_list(path_gain) ||
                  !qd_python_is_array_list(path_length) || !qd_python_is_array_list(polmat)))
        throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'polmat' must all be lists.");

    // Alias the path data, no data is copied
    const auto fbs_pos_a = qd_python_alias_list<arma::Mat<double>>(fbs_pos, batch, keep_alive);
    const auto lbs_pos_a = qd_python_alias_list<arma::Mat<double>>(lbs_pos, batch, keep_alive);
    const auto path_gain_a = qd_python_alias_list<arma::Mat<double>>(path_gain, batch, keep_alive);
    const auto path_length_a = qd_python_alias_list<arma::Col<double>>(path_length, batch, keep_alive);
    const auto polmat_a = qd_python_alias_list<arma::Cube<double>>(polmat, batch, keep_alive);
    const arma::uword n_snap = (arma::uword)fbs_pos_a.size();

    // Needed before the frequency check below, which indexes all five lists
    if (lbs_pos_a.size() != n_snap || path_gain_a.size() != n_snap ||
        path_length_a.size() != n_snap || polmat_a.size() != n_snap)
        throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'polmat' must have the same length.");

    // Positions and orientations: one column per snapshot or one for all (checked by the C++ core); a single snapshot
    // takes any 3-element array
    auto position = [&](const py::handle &h, const char *name)
    { return batch ? qd_python_numpy2arma_Mat<double>(h, true) : arma::Mat<double>(qd_python_numpy2arma_Col<double>(h.cast<py::array_t<double>>(), false, false, name, 3)); };
    const arma::Mat<double> tx_pos_a = position(tx_pos, "tx_pos"), tx_orientation_a = position(tx_orientation, "tx_orientation");
    const arma::Mat<double> rx_pos_a = position(rx_pos, "rx_pos"), rx_orientation_a = position(rx_orientation, "rx_orientation");

    // Frequency slice of every snapshot, aliased without copying; path counts are validated by the C++ core
    std::vector<arma::vec> path_gain_f;
    std::vector<arma::mat> polmat_f;
    const double center_freq_sel = qd_python_select_freq(path_gain_a, polmat_a, center_freq_a, i_freq, path_gain_f, polmat_f);

    // Outputs: numpy memory is mapped to the Armadillo cubes, which C++ writes in place
    std::vector<arma::cube> coeff_re, coeff_im, delay(n_snap), aod(n_snap), eod(n_snap), aoa(n_snap), eoa(n_snap);
    std::vector<arma::cx_cube> coeff(n_snap);
    py::list coeff_py, delay_py, aod_py, eod_py, aoa_py, eoa_py;
    for (arma::uword i = 0ULL; i < n_snap; ++i)
    {
        const arma::uword n_path = add_fake_los_path_b ? fbs_pos_a[i].n_cols + 1ULL : fbs_pos_a[i].n_cols;
        coeff_py.append(qd_python_init_output(n_ports_rx, n_ports_tx, n_path, &coeff[i]));
        delay_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &delay[i]));
        if (angles_b) // Without angles, the core skips their conversion to the ports
        {
            aod_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aod[i]));
            eod_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eod[i]));
            aoa_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aoa[i]));
            eoa_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eoa[i]));
        }
    }

    {
        py::gil_scoped_release release;
        quadriga_lib::get_channels_spherical_vec<double>(ant_tx_a, ant_rx_a, tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                                         fbs_pos_a, lbs_pos_a, path_gain_f, path_length_a, polmat_f,
                                                         coeff_re, coeff_im, delay,
                                                         center_freq_sel, use_absolute_delays_b, add_fake_los_path_b,
                                                         angles_b ? &aod : nullptr, angles_b ? &eod : nullptr,
                                                         angles_b ? &aoa : nullptr, angles_b ? &eoa : nullptr, &coeff, use_kernel);
    }

    // Lists in batch mode, the arrays of the one snapshot otherwise
    py::list out;
    auto put = [&](const py::list &l)
    { out.append(batch ? py::object(l) : py::object(l[0])); };
    put(coeff_py), put(delay_py);
    if (angles_b)
        put(aod_py), put(eod_py), put(aoa_py), put(eoa_py);
    return py::tuple(out);
}
