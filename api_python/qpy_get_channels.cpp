// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

// Path inputs of the channel functions: true if 'h' is a list of numpy arrays (batch mode) or an empty list
static inline bool qd_python_is_array_list(const py::handle &h)
{
    if (!py::isinstance<py::list>(h))
        return false;
    py::list lst = py::reinterpret_borrow<py::list>(h);
    return lst.size() == 0 || py::isinstance<py::array>(lst[0]);
}

// Aliases the arrays of a list (batch mode), or a single input as a one-entry vector, without copying the data (arrays
// that are not Fortran-contiguous float64 are copied, as are lists and scalars of a single input). T is
// arma::Col<double>, arma::Mat<double> or arma::Cube<double>.
// Every entry is added to 'keep_alive': the aliased memory must stay valid while the GIL is released, even if another
// thread removes the entry from the list.
template <typename T>
static inline std::vector<T> qd_python_alias_list(const py::handle &h, bool batch, std::vector<py::object> &keep_alive)
{
    std::vector<T> out;
    auto add = [&](const py::handle &item)
    {
        keep_alive.push_back(py::reinterpret_borrow<py::object>(item));
        if constexpr (std::is_same_v<T, arma::Col<double>>)
            out.push_back(qd_python_numpy2arma_Col<double>(keep_alive.back(), true));
        else if constexpr (std::is_same_v<T, arma::Mat<double>>)
            out.push_back(qd_python_numpy2arma_Mat<double>(keep_alive.back(), true));
        else
            out.push_back(qd_python_numpy2arma_Cube<double>(keep_alive.back(), true));
    };
    if (batch)
    {
        py::list lst = py::reinterpret_borrow<py::list>(h);
        out.reserve(lst.size());
        for (auto item : lst)
            add(item);
    }
    else
        out.reserve(1), add(h);
    return out;
}

// Adds all values of a dict (e.g. an arrayant whose patterns are aliased) to 'keep_alive'
static inline void qd_python_keep_dict_values(const py::dict &dict, std::vector<py::object> &keep_alive)
{
    for (auto item : dict)
        keep_alive.push_back(py::reinterpret_borrow<py::object>(item.second));
}

// Frequency selection of the channel wrappers: all snapshots must have the same number of frequencies 'n_freq' in
// 'path_gain' [n_path, n_freq] and 'polmat' [8, n_path, n_freq], 'center_freq' is empty or has 'n_freq' entries, and
// 'i_freq' < n_freq (unless there are no snapshots). The slice 'i_freq' of every snapshot is aliased into 'path_gain_f'
// and 'polmat_f'; returns the selected center frequency (0 for an empty 'center_freq').
static inline double qd_python_select_freq(const std::vector<arma::Mat<double>> &path_gain, const std::vector<arma::Cube<double>> &polmat,
                                    const arma::Col<double> &center_freq, arma::uword i_freq,
                                    std::vector<arma::vec> &path_gain_f, std::vector<arma::mat> &polmat_f)
{
    const size_t n_snap = path_gain.size();
    arma::uword n_freq = 0ULL;
    for (size_t i = 0; i < n_snap; ++i)
    {
        const arma::uword n_freq_i = path_gain[i].n_cols;
        if (i == 0)
            n_freq = n_freq_i;
        else if (n_freq_i != n_freq)
            throw std::invalid_argument("All entries of 'path_gain' must have the same number of frequencies.");

        if (polmat[i].n_slices != n_freq_i)
            throw std::invalid_argument("Inputs 'path_gain' and 'polmat' must have the same number of frequencies.");
    }

    if (!center_freq.empty() && center_freq.n_elem != n_freq)
        throw std::invalid_argument("Input 'center_freq' must have 'n_freq' elements.");

    if (n_snap != 0 && i_freq >= n_freq)
        throw std::out_of_range("Input 'i_freq' exceeds the number of frequencies.");

    path_gain_f.clear(), polmat_f.clear();
    path_gain_f.reserve(n_snap), polmat_f.reserve(n_snap);
    for (size_t i = 0; i < n_snap; ++i)
    {
        path_gain_f.emplace_back(const_cast<double *>(path_gain[i].colptr(i_freq)), path_gain[i].n_rows, false, true);
        polmat_f.emplace_back(const_cast<double *>(polmat[i].slice_memptr(i_freq)), polmat[i].n_rows, polmat[i].n_cols, false, true);
    }
    return center_freq.empty() ? 0.0 : center_freq.at(i_freq);
}

// Position or orientation input of the planar and spherical wrappers: in batch mode a [3, n_snap] matrix or one column
// for all snapshots (checked by the C++ core), else any 3-element array, returned as [3, 1]
static inline arma::Mat<double> qd_python_get_xyz(const py::handle &h, const char *name, bool batch)
{
    return batch ? qd_python_numpy2arma_Mat<double>(h, true)
                 : arma::Mat<double>(qd_python_numpy2arma_Col<double>(h.cast<py::array_t<double>>(), false, false, name, 3));
}

// IRS arrayant dict: the coupling matrix holds the IRS weights; without one, C++ raises instead of using identity
static inline quadriga_lib::arrayant<double> qd_irs_dict2arrayant(const py::dict &irs)
{
    auto ant = qd_python_dict2arrayant(irs, true);
    auto given = [&](const char *key)
    { return irs.contains(key) && !irs[key].is_none() && py::array::ensure(irs[key]).size() != 0; };
    if (!given("coupling_re") && !given("coupling_im"))
        ant.coupling_re.reset(), ant.coupling_im.reset();
    return ant;
}

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
- Polarization coupling is applied via the 8-row transfer matrix `polmat`: the 2x2 Jones matrix (rows = RX polarisation
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
- The path inputs `aod`, `eod`, `aoa`, `eoa`, `path_gain`, `path_length` and `polmat` are then lists of length `n_snap`;
  list entries that are Fortran-contiguous float64 arrays are read in place, other entries are copied.
- The position and orientation inputs are `(3, n_snap)` arrays with one column per snapshot; a `(3,)` array or a
  single-column `(3, 1)` array is applied to all snapshots.
- All outputs are returned as lists of length `n_snap`, one entry per snapshot.
- The GIL is released while processing, so the function can be combined with Python threads. The function keeps
  references to all input arrays until it returns, but other threads must not write into them meanwhile.

## Usage:
```
coeff, delay, rx_Doppler = quadriga_lib.channel.planar( tx_array, rx_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa, path_gain, path_length, polmat, \
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
- **`polmat`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`, or a list in batch mode
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; an entry of `0` or `None` gives the
  coefficients at f = 0 (see above); default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single precision; raises if
  unavailable); auto selects GENERIC in this version, see [[interpolate]]; `int`; default: 0
- **`i_freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `polmat` and `center_freq` that is
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
                              py::handle polmat,
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
                  !qd_python_is_array_list(path_gain) || !qd_python_is_array_list(path_length) || !qd_python_is_array_list(polmat)))
        throw std::invalid_argument("Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length' and 'polmat' must all be lists.");

    // Alias the path data, no data is copied
    const auto aod_a = qd_python_alias_list<arma::Col<double>>(aod, batch, keep_alive);
    const auto eod_a = qd_python_alias_list<arma::Col<double>>(eod, batch, keep_alive);
    const auto aoa_a = qd_python_alias_list<arma::Col<double>>(aoa, batch, keep_alive);
    const auto eoa_a = qd_python_alias_list<arma::Col<double>>(eoa, batch, keep_alive);
    const auto path_gain_a = qd_python_alias_list<arma::Mat<double>>(path_gain, batch, keep_alive);
    const auto path_length_a = qd_python_alias_list<arma::Col<double>>(path_length, batch, keep_alive);
    const auto polmat_a = qd_python_alias_list<arma::Cube<double>>(polmat, batch, keep_alive);
    const arma::uword n_snap = (arma::uword)aod_a.size();

    // Needed before the frequency check below, which indexes the lists
    if (eod_a.size() != n_snap || aoa_a.size() != n_snap || eoa_a.size() != n_snap ||
        path_gain_a.size() != n_snap || path_length_a.size() != n_snap || polmat_a.size() != n_snap)
        throw std::invalid_argument("Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length' and 'polmat' must have the same length.");

    const arma::Mat<double> tx_pos_a = qd_python_get_xyz(tx_pos, "tx_pos", batch);
    const arma::Mat<double> tx_orientation_a = qd_python_get_xyz(tx_orientation, "tx_orientation", batch);
    const arma::Mat<double> rx_pos_a = qd_python_get_xyz(rx_pos, "rx_pos", batch);
    const arma::Mat<double> rx_orientation_a = qd_python_get_xyz(rx_orientation, "rx_orientation", batch);

    // Frequency slice of every snapshot, aliased without copying; path counts are validated by the C++ core
    std::vector<arma::vec> path_gain_f;
    std::vector<arma::mat> polmat_f;
    const double center_freq_sel = qd_python_select_freq(path_gain_a, polmat_a, center_freq_a, i_freq, path_gain_f, polmat_f);

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
                                                      aod_a, eod_a, aoa_a, eoa_a, path_gain_f, path_length_a, polmat_f,
                                                      coeff_re, coeff_im, delay,
                                                      center_freq_sel, use_absolute_delays_b, add_fake_los_path_b,
                                                      &rx_Doppler, &coeff, use_kernel);
    }

    // Lists in batch mode, the arrays of the one snapshot otherwise
    if (batch)
        return py::make_tuple(coeff_py, delay_py, rx_Doppler_py);
    return py::make_tuple(coeff_py[0], delay_py[0], rx_Doppler_py[0]);
}

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

    const arma::Mat<double> tx_pos_a = qd_python_get_xyz(tx_pos, "tx_pos", batch);
    const arma::Mat<double> tx_orientation_a = qd_python_get_xyz(tx_orientation, "tx_orientation", batch);
    const arma::Mat<double> rx_pos_a = qd_python_get_xyz(rx_pos, "rx_pos", batch);
    const arma::Mat<double> rx_orientation_a = qd_python_get_xyz(rx_orientation, "rx_orientation", batch);

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

/*!MD
# channel.multifreq
Compute channel coefficients for spherical waves across multiple frequencies

- Multi-frequency extension of [[channel.spherical]] with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) is computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (`tx_array['center_freq']`), RX array (`rx_array['center_freq']`), input samples (`freq`), and output (`freq_new`)
- Antenna pattern fields may be 3D (single-frequency, clamped for all output frequencies) or 4D (multi-frequency, 4th axis = frequency)
- TX/RX patterns are interpolated per output frequency via SLERP with linear fallback
- `path_gain` is interpolated linearly; `polmat` is interpolated via SLERP per complex entry pair to preserve phase
- Coupling matrices are interpolated across frequencies (SLERP for complex pairs); pass 3D coupling `(n_elem, n_ports, n_freq)` for per-frequency coupling
- Extrapolation clamps to the nearest frequency entry on all four grids
- `polmat` accepts 8 rows (full polarimetric: the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
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
  [[baseband_freq_response_multi]] with `remove_delay_phase = True` removes the phase of `delay`
  before it interpolates across `freq`, so it needs channels generated with `use_absolute_delays = True`; with
  relative delays, the phase of the TX-RX distance stays in the coefficients, is interpolated across the frequency
  grid and gives wrong phases

## Usage:
```
coeff, delay = quadriga_lib.channel.multifreq( tx_array, rx_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, \
    freq, freq_new, *, use_absolute_delays, add_fake_los_path, propagation_speed, use_kernel )
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
- **`polmat`** — Polarization transfer matrix, row order as above; `(8, n_path, n_freq)` full pol or
  `(2, n_path, n_freq)` scalar pressure
- **`freq`** — Input sample frequencies for `path_gain` and `polmat`; `(n_freq,)`
- **`freq_new`** — Target output frequencies; `(n_freq_new,)`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`propagation_speed`** — Wave speed in m/s; use ~343.0 for acoustics; default: `299792458.0`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single precision; raises if
  unavailable); auto selects GENERIC in this version, see [[interpolate]]; `int`; default: 0

## Outputs:
- **`coeff`** — Complex channel coefficients; `(n_ports_rx, n_ports_tx, n_path_out, n_freq_new)`
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
                                 const py::array_t<double> &polmat,
                                 const py::array_t<double> &freq,
                                 const py::array_t<double> &freq_new,
                                 py::handle use_absolute_delays,
                                 py::handle add_fake_los_path,
                                 const double propagation_speed,
                                 const int use_kernel)
{
    const bool use_absolute_delays_b = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool add_fake_los_path_b = qd_python_flag(add_fake_los_path, false, "add_fake_los_path");

    auto ant_tx_vec = qd_python_dict2arrayant_multi(tx_array, true);
    auto ant_rx_vec = qd_python_dict2arrayant_multi(rx_array, true);

    const auto fbs_pos_a = qd_python_numpy2arma_Mat(fbs_pos, true);
    const auto lbs_pos_a = qd_python_numpy2arma_Mat(lbs_pos, true);
    const auto path_gain_a = qd_python_numpy2arma_Mat(path_gain, true);
    const auto path_length_a = qd_python_numpy2arma_Col(path_length, true);
    const auto polmat_a = qd_python_numpy2arma_Cube(polmat, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation, true, false, "rx_orientation", 3);
    const auto freq_a = qd_python_numpy2arma_Col(freq, true);
    const auto freq_new_a = qd_python_numpy2arma_Col(freq_new, true);

    arma::uword n_ports_tx = ant_tx_vec[0].n_ports();
    arma::uword n_ports_rx = ant_rx_vec[0].n_ports();
    arma::uword n_path = add_fake_los_path_b ? fbs_pos_a.n_cols + 1 : fbs_pos_a.n_cols;
    arma::uword n_freq_new = freq_new_a.n_elem;

    // Complex output written in place by C++, coeff_re / coeff_im left empty (0 paths: the zero-power LOS path with
    // add_fake_los_path, else empty outputs)
    std::vector<arma::cube> coeff_re, coeff_im, delay;
    std::vector<arma::cx_cube> coeff;
    auto coeff_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_new, &coeff);
    auto delay_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_new, &delay);
    quadriga_lib::get_channels_multifreq<double>(ant_tx_vec, ant_rx_vec,
                                                 tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                                 fbs_pos_a, lbs_pos_a, path_gain_a, path_length_a, polmat_a,
                                                 freq_a, freq_new_a,
                                                 coeff_re, coeff_im, delay,
                                                 use_absolute_delays_b, add_fake_los_path_b, propagation_speed, &coeff,
                                                 use_kernel);
    return py::make_tuple(coeff_py, delay_py);
}

/*!MD
# channel.irs
Calculate MIMO channel coefficients for IRS-assisted communication

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- The IRS coupling matrix holds the IRS weights and must be given: an `irs_array` or `irs_array_2` dict without
  `coupling_re` / `coupling_im` raises, unlike in the other functions, where it stands for identity
- Polarization coupling is applied via the 8-row transfer matrices `polmat_1`, `polmat_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx)
- Each segment is computed by [[channel.spherical]]: NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths that exceed `threshold_dB`; the pair
  of segment-1 path `i_1` and segment-2 path `i_2` has index `i_1 · n_path_2 + i_2` in `active_path` and
  `active_path_out`, and the output keeps this order
- A segment without paths (all its path inputs empty) gives empty outputs `(n_rx, n_tx, 0)`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual
- If `use_absolute_delays == False`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

## Usage:
```
coeff, delay, active_path_out = quadriga_lib.channel.irs( tx_array, rx_array, irs_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, \
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1, \
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2, center_freq, \
    *, i_irs, threshold_dB, use_absolute_delays, angles, irs_array_2, active_path, use_kernel )

# Departure and arrival angles as well
coeff, delay, active_path_out, aod, eod, aoa, eoa = quadriga_lib.channel.irs( ..., angles=True )
```

## Inputs:
- **`tx_array`** — Transmit arrayant dict; `n_tx` = number of ports after element coupling; see [[generate]]
- **`rx_array`** — Receive arrayant dict; `n_rx` = number of ports after element coupling; see [[generate]]
- **`irs_array`** — IRS arrayant dict (TX-facing side) with `n_irs` elements; the coupling matrix holds the IRS
  weights, one codebook entry per column, `(n_irs, n_codebook)`; see [[generate]]
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`irs_pos`** — IRS position in Cartesian coordinates; `(3,)`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `(3, n_path_1)`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `(3, n_path_1)`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `(n_path_1,)`
- **`path_length_1`** — Total path lengths from TX to IRS phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `(n_path_1,)`
- **`polmat_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `(8, n_path_1)`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `(3, n_path_2)`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `(3, n_path_2)`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `(n_path_2,)`
- **`path_length_2`** — Total path lengths from IRS to RX phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `(n_path_2,)`
- **`polmat_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `(8, n_path_2)`
- **`center_freq`** — Center frequency in Hz; 0 gives the coefficients at f = 0 (see above); default: `0.0`
- **`i_irs`** — IRS codebook port index, 0-based (one port per column of the IRS coupling matrix); default: `0`
- **`threshold_dB`** — Gain threshold in dB; path combinations below this are discarded; default: `-140.0`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`angles`** — If `True`, also return the departure and arrival angles; `False` saves their memory; default: `False`
- **`irs_array_2`** — Second IRS arrayant dict for the RX-facing side (asymmetric IRS), with the same number of
  elements and the same coupling size as `irs_array`; default: `None` (`irs_array` for both sides)
- **`active_path`** — Mask selecting active path pairs, bool or numeric (nonzero = active); overrides `threshold_dB`
  when not empty; `(n_path_1 · n_path_2,)`; default: `None`
- **`use_kernel`** — Kernel of both segments, as in [[channel.spherical]]: 0 = auto, 1 = GENERIC, 2 = AVX2 (single
  precision; raises if unavailable); auto selects GENERIC in this version; `int`; default: 0

## Outputs:
- **`coeff`** — Complex channel coefficients; `(n_rx, n_tx, n_path_irs)`
- **`delay`** — Propagation delays in seconds; `(n_rx, n_tx, n_path_irs)`
- **`active_path_out`** — Mask of the path combinations in the output (can be passed as `active_path` to a later
  call); `bool`; `(n_path_1 · n_path_2,)`
- **`aod`** — Azimuth of departure in rad; `(n_rx, n_tx, n_path_irs)`; only with `angles=True`
- **`eod`** — Elevation of departure in rad; same shape; only with `angles=True`
- **`aoa`** — Azimuth of arrival in rad; same shape; only with `angles=True`
- **`eoa`** — Elevation of arrival in rad; same shape; only with `angles=True`

## See also:
- [[channel.spherical]] (single-segment spherical-wave channel)
- [[channel.planar]] (single-segment planar-wave channel)
- [[generate]] (antenna array generator)
MD!*/

py::tuple get_channels_irs(const py::dict &tx_array,
                           const py::dict &rx_array,
                           const py::dict &irs_array,
                           const py::array_t<double> &tx_pos,
                           const py::array_t<double> &tx_orientation,
                           const py::array_t<double> &rx_pos,
                           const py::array_t<double> &rx_orientation,
                           const py::array_t<double> &irs_pos,
                           const py::array_t<double> &irs_orientation,
                           const py::array_t<double> &fbs_pos_1,
                           const py::array_t<double> &lbs_pos_1,
                           const py::array_t<double> &path_gain_1,
                           const py::array_t<double> &path_length_1,
                           const py::array_t<double> &polmat_1,
                           const py::array_t<double> &fbs_pos_2,
                           const py::array_t<double> &lbs_pos_2,
                           const py::array_t<double> &path_gain_2,
                           const py::array_t<double> &path_length_2,
                           const py::array_t<double> &polmat_2,
                           const double center_freq,
                           const arma::uword i_irs,
                           const double threshold_dB,
                           py::handle use_absolute_delays,
                           py::handle angles,
                           py::handle irs_array_2,
                           py::handle active_path,
                           const int use_kernel)
{
    const bool want_absolute = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool want_angles = qd_python_flag(angles, false, "angles");

    const auto tx_array_a = qd_python_dict2arrayant(tx_array, true);
    const auto rx_array_a = qd_python_dict2arrayant(rx_array, true);
    const auto irs_array_a = qd_irs_dict2arrayant(irs_array);

    quadriga_lib::arrayant<double> irs_array_2_a;
    if (!irs_array_2.is_none())
    {
        if (!py::isinstance<py::dict>(irs_array_2))
            throw py::type_error("Input 'irs_array_2' must be a dict or None.");
        irs_array_2_a = qd_irs_dict2arrayant(py::reinterpret_borrow<py::dict>(irs_array_2));
    }

    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation, true, false, "rx_orientation", 3);
    const auto irs_pos_a = qd_python_numpy2arma_Col(irs_pos, true, false, "irs_pos", 3);
    const auto irs_orientation_a = qd_python_numpy2arma_Col(irs_orientation, true, false, "irs_orientation", 3);

    // Path data of both segments, sizes are validated by the C++ core
    const auto fbs_pos_1_a = qd_python_numpy2arma_Mat(fbs_pos_1, true);
    const auto lbs_pos_1_a = qd_python_numpy2arma_Mat(lbs_pos_1, true);
    const auto path_gain_1_a = qd_python_numpy2arma_Col(path_gain_1, true);
    const auto path_length_1_a = qd_python_numpy2arma_Col(path_length_1, true);
    const auto polmat_1_a = qd_python_numpy2arma_Mat(polmat_1, true);
    const auto fbs_pos_2_a = qd_python_numpy2arma_Mat(fbs_pos_2, true);
    const auto lbs_pos_2_a = qd_python_numpy2arma_Mat(lbs_pos_2, true);
    const auto path_gain_2_a = qd_python_numpy2arma_Col(path_gain_2, true);
    const auto path_length_2_a = qd_python_numpy2arma_Col(path_length_2, true);
    const auto polmat_2_a = qd_python_numpy2arma_Mat(polmat_2, true);

    // Active path mask: nonzero = active
    const auto active_path_in = qd_python_numpy2arma_Col<double>(active_path, true);
    std::vector<bool> active_path_a(active_path_in.n_elem);
    for (arma::uword i = 0; i < active_path_in.n_elem; ++i)
        active_path_a[i] = active_path_in[i] != 0.0;

    // Output sizes depend on the path selection: copied after the call
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    std::vector<bool> active_path_out;

    quadriga_lib::get_channels_irs<double>(tx_array_a, rx_array_a, irs_array_a,
                                           tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                           irs_pos_a, irs_orientation_a,
                                           fbs_pos_1_a, lbs_pos_1_a, path_gain_1_a, path_length_1_a, polmat_1_a,
                                           fbs_pos_2_a, lbs_pos_2_a, path_gain_2_a, path_length_2_a, polmat_2_a,
                                           coeff_re, coeff_im, delay,
                                           i_irs, threshold_dB, center_freq, want_absolute,
                                           want_angles ? &aod : nullptr, want_angles ? &eod : nullptr,
                                           want_angles ? &aoa : nullptr, want_angles ? &eoa : nullptr,
                                           irs_array_2_a, active_path_a, &active_path_out, use_kernel);

    py::array_t<bool> active_path_out_py((py::ssize_t)active_path_out.size());
    bool *p_out = active_path_out_py.mutable_data();
    for (size_t i = 0; i < active_path_out.size(); ++i)
        p_out[i] = active_path_out[i];

    py::list out;
    out.append(qd_python_copy2numpy<double, std::complex<double>>(&coeff_re, &coeff_im));
    out.append(qd_python_copy2numpy(&delay));
    out.append(active_path_out_py);
    if (want_angles)
        for (const auto *c : {&aod, &eod, &aoa, &eoa})
            out.append(qd_python_copy2numpy(c));

    return py::tuple(out);
}

/*!MD
# channel.ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions
- 2D model: azimuth angles and planar motion only, no elevation
- Supported channel types: `A, B, C, D, E, F` (TGn definitions)
- MU-MIMO supported (`n_users > 1`) with per-user distances/floors and optional angle offsets per TGac
- Time-evolving channels via `observation_time`, `update_interval`, and mobility parameters; `observation_time = 0.0` yields a static channel
- Floor penetration loss according to TGah for `center_freq` < 1 GHz and TGax for above 1 GHz; floors add attenuation
  (and the TGah shadow fading below 1 GHz) but no delay; the station is placed 3 m per floor above the AP
- Taps with zero power in the model tables are removed; the number of paths does not depend on distance, floors or walls
- Time evolution: each sub-path is rotated by `exp(+j 2 pi f_D t)`, i.e. a station moving towards the AP or a scatterer
  sees a positive Doppler shift `f_D`, as the channel of [[channel.spherical]] along the returned station positions
- Uplink: the coefficient and delay matrices are the transposes of the downlink ones (reciprocal channel, no conjugation),
  so both directions have the same Doppler shift; positions and orientations are swapped (the moving station is the transmitter)
- Override parameters (`kf`, `xpr_nlos`, `sf_std_los_dB`, `sf_std_nlos_dB`, `dist_breakpoint`): `NaN` selects the model default
- Per-snapshot data (`coeff`, `delay`, `path_gain`) is returned as a list of per-snapshot arrays, 
  or stacked into one array along an appended snapshot axis when `stack = True`

## Usage:
```
chan = quadriga_lib.channel.ieee_indoor( ap_array, sta_array, channel_type, center_freq, *, \
    tap_spacing, n_users, observation_time, update_interval, speed_station_kmh, speed_env_kmh, \
    dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect, seed, \
    kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint, n_walls, wall_loss_dB, stack )
```

## Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling, see arrayant-[[generate]]
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling, see arrayant-[[generate]]
- **`channel_type`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`center_freq`** *(optional)* — Carrier frequency in Hz (finite, above 100 MHz); default: 5.25e9
- **`tap_spacing`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output list length equals `n_users`; default: 1
- **`observation_time`** *(optional)* — Channel observation time in seconds; must be finite; default: 0
- **`update_interval`** *(optional)* — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** *(optional)* — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`dist`** *(optional)* — TX-to-RX distance(s) in meters (finite, > 0); `(n_users,)` or `(1,)`; default: 4.99
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `(n_users,)` or `(1,)`; default: 0
- **`uplink`** *(optional)* — Set `True` to generate uplink (reverse) direction; coefficients are the transpose of the downlink; default: False
- **`offset_angles_deg`** *(optional)* — Azimuth offset angles in degrees (finite); rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; `(4, n_users)`;
  default: TGac auto-defaults for `n_users > 1`
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`doppler_effect`** *(optional)* — Special Doppler: models D/E use the mains frequency in Hz, model F the vehicle speed in km/h (finite, >= 0); 0 disables; default: 50
- **`seed`** *(optional)* — RNG seed; any other value than `-1` gives the same channels on all platforms; `-1` uses the system random device; default: -1
- **`kf`** *(optional)* — Overrides the model KF (linear scale), finite and >= 0; default: A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0;
  applied to first tap only; with `kf` given, the LOS path is kept also beyond the breakpoint and on other floors
- **`xpr_nlos`** *(optional)* — Overrides the NLOS cross-polarization ratio (linear scale), finite and > 0; default: 2 (3 dB)
- **`sf_std_los_dB`** *(optional)* — Overrides the LOS shadow fading std in dB (applied when d < breakpoint), finite and >= 0; default: 3 dB, 2 dB below 1 GHz (TGah)
- **`sf_std_nlos_dB`** *(optional)* — Overrides the NLOS shadow fading std in dB (applied when d >= breakpoint), finite and >= 0;
  default: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB, each 1 dB lower below 1 GHz (TGah)
- **`dist_breakpoint`** *(optional)* — Overrides the breakpoint distance in meters, finite and > 0; default: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- **`n_walls`** *(optional)* — Number of walls per user TGax models; `(n_users,)` or `(1,)`; default: 0
- **`wall_loss_dB`** *(optional)* — Penetration loss for a single wall in dB (finite); TGax defines 5 or 7; default: 5
- **`stack`** *(optional)* — If `True`, per-snapshot data is stacked into a single array along an appended 
  snapshot axis instead of being returned as a list of per-snapshot arrays; default: False


## Output:
- **`chan`** — List of length `n_users` containing dictionaries of channel data. Only the keys
  listed below are populated; all other channel keys are omitted by this generator.<br><br>
  | Key                | Description                                                              | Shape `stack = False`            | Shape `stack = True`           |
  | ------------------ | ------------------------------------------------------------------------ | -------------------------------- | ------------------------------ |
  | `name`             | Channel name                                                             | str                              | str                            |
  | `tx_pos`           | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_pos`           | Receiver positions (STA for downlink, AP for uplink)                     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `coeff`            | Channel coefficients, complex valued                                     | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`            | `(n_path, n_snap)`             |
  | `center_freq` | Center Frequency in Hz                                                   | `(1,)`                           | `(1,)`                         |
  | `initial_position` | 0-based index of the reference snapshot                                  | int                              | int                            |

## See also:
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc">IEEE 802.11-03/940r4 - TGn Channel Models</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc">IEEE 802.11-09/0308r12 - TGac Channel Model Addendum</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx">IEEE 802.11-11/0968r4 - TGah Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx">IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf">S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025</a>
- [[hdf5_write_channel]] (for writing channel data to a HDF5 file)
- [[hdf5_read_channel]] (for reading channel data to a HDF5 file)
MD!*/

py::list get_channels_ieee_indoor(const py::dict &ap_array,
                                  const py::dict &sta_array,
                                  const std::string &channel_type,
                                  const double center_freq,
                                  const double tap_spacing,
                                  const arma::uword n_users,
                                  const double observation_time,
                                  const double update_interval,
                                  const double speed_station_kmh,
                                  const double speed_env_kmh,
                                  const py::handle &dist,
                                  const py::handle &n_floors,
                                  const py::handle &uplink,
                                  const py::handle &offset_angles_deg,
                                  const arma::uword n_subpath,
                                  const double doppler_effect,
                                  const arma::sword seed,
                                  const double kf,
                                  const double xpr_nlos,
                                  const double sf_std_los_dB,
                                  const double sf_std_nlos_dB,
                                  const double dist_breakpoint,
                                  const py::handle &n_walls,
                                  const double wall_loss_dB,
                                  const py::handle &stack)
{
    // None gives an empty container, which selects the C++ default
    const auto ant_tx_a = qd_python_dict2arrayant(ap_array, true);
    const auto ant_rx_a = qd_python_dict2arrayant(sta_array, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);
    const auto n_floors_a = qd_python_numpy2arma_Col<arma::uword>(n_floors, true, false, "n_floors");
    const bool uplink_b = qd_python_flag(uplink, false, "uplink");
    const auto offset_angles_deg_a = qd_python_numpy2arma_Mat<double>(offset_angles_deg, true);
    const auto n_walls_a = qd_python_numpy2arma_Col<arma::uword>(n_walls, true, false, "n_walls");
    const bool stack_b = qd_python_flag(stack, false, "stack");

    const auto chan = quadriga_lib::get_channels_ieee_indoor(ant_tx_a, ant_rx_a, channel_type, center_freq, tap_spacing, n_users,
                                                             observation_time, update_interval, speed_station_kmh, speed_env_kmh,
                                                             dist_a, n_floors_a, uplink_b, offset_angles_deg_a, n_subpath,
                                                             doppler_effect, seed, kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB,
                                                             dist_breakpoint, n_walls_a, wall_loss_dB);

    return qd_python_channel2list(chan, false, stack_b);
}
