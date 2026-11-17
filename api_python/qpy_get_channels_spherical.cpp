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
- Polarization coupling is applied via the 8-row transfer matrix `M`: the 2x2 Jones matrix (rows = RX polarisation
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
- `complex=True` (default) returns one combined complex coefficient array `coeff`; `complex=False` returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path

## Batch mode:
- If `fbs_pos` is a `list` of arrays, the function switches to batch mode and processes `n_snap` snapshots
  in one call, the snapshots in parallel (OpenMP in C++).
- The path inputs `fbs_pos`, `lbs_pos`, `path_gain`, `path_length` and `M` are then lists of length `n_snap`,
  matching the output format of [[qrt_file_read]]; list entries that are Fortran-contiguous float64 arrays (as
  returned by [[qrt_file_read]]) are read in place, other entries are copied.
- The position and orientation inputs are `(3, n_snap)` arrays with one column per snapshot; a `(3,)` array
  or a single-column `(3, 1)` array is applied to all snapshots.
- All outputs are returned as lists of length `n_snap`, one entry per snapshot.
- Snapshots without any path produce an empty output entry with `n_path = 0`, or only the zero-power LOS path
  when `add_fake_los_path` is set (as for a single snapshot).
- Batch mode releases the GIL while processing, so it can be combined with Python threads. The function keeps
  references to all input arrays until it returns, but other threads must not write into them meanwhile.

## Usage:
```
# Complex coefficients (default)
coeff, delays = quadriga_lib.channel.spherical( ant_tx, ant_rx, \
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, \
    center_freq, use_absolute_delays, add_fake_los_path, angles, complex, i_freq, use_kernel )

# Real and imaginary parts as separate arrays
coeff_re, coeff_im, delays = quadriga_lib.channel.spherical( ..., complex=False )

# Departure and arrival angles as well
coeff, delays, aod, eod, aoa, eoa = quadriga_lib.channel.spherical( ..., angles=True )
```

## Inputs:
- **`ant_tx`** — Transmit arrayant dict with single-frequency (3D) patterns, see [[generate]]; a 4D dict raises
  `ValueError` (use [[channel.multifreq]])
- **`ant_rx`** — Receive arrayant dict with single-frequency (3D) patterns; as `ant_tx`
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`, or a list of length `n_snap` in batch mode
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`, or a list of length `n_snap` in batch mode
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`, or a list of length `n_snap` in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `(n_path,)`, or a list of length `n_snap` in batch mode
- **`M`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`, or a list of length `n_snap`
  in batch mode
- **`tx_pos`** — Transmitter position in Cartesian coordinates, finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading), finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_pos`** — Receiver position in Cartesian coordinates, finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading), finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; an entry of `0` or `None` gives the coefficients at f = 0 (see above); default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`angles`** — If `True`, also return the departure and arrival angles; `False` saves their memory and, with element
  coupling, the time to convert them to the ports; default: `False`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `True`
- **`i_freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `M` and `center_freq` that is
  processed; default: `0`
- **`use_kernel`** — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto, 1 = GENERIC,
  2 = AVX2 (single precision; raises if unavailable); auto selects GENERIC in this version, see [[interpolate]];
  `int`; default: 0

## Outputs:
- **`coeff`** — Complex channel coefficients (`complex=True`); `(n_ports_rx, n_ports_tx, n_path)`
- **`coeff_re`** — Real part of channel coefficients (`complex=False`), replaces `coeff`; same shape
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape
- **`delays`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
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

// Batch mode is selected when the path inputs are given as a list of numpy arrays
static bool qd_is_array_list(const py::handle &h)
{
    if (!py::isinstance<py::list>(h))
        return false;

    py::list lst = py::reinterpret_borrow<py::list>(h);
    return lst.size() == 0 || py::isinstance<py::array>(lst[0]);
}

// Alias the entries of a list of numpy arrays, the array data itself is not copied (entries that are not
// Fortran-contiguous float64 arrays are copied). Every entry is added to 'keep_alive': the aliased memory must stay
// valid while the GIL is released, even if another thread removes the entry from the list.
static std::vector<arma::Mat<double>> qd_alias_list_Mat(const py::handle &h, std::vector<py::object> &keep_alive)
{
    py::list lst = py::reinterpret_borrow<py::list>(h);
    std::vector<arma::Mat<double>> out;
    out.reserve(lst.size());
    for (auto item : lst)
    {
        keep_alive.push_back(py::reinterpret_borrow<py::object>(item));
        out.push_back(qd_python_numpy2arma_Mat<double>(keep_alive.back(), true));
    }
    return out;
}

static std::vector<arma::Col<double>> qd_alias_list_Col(const py::handle &h, std::vector<py::object> &keep_alive)
{
    py::list lst = py::reinterpret_borrow<py::list>(h);
    std::vector<arma::Col<double>> out;
    out.reserve(lst.size());
    for (auto item : lst)
    {
        keep_alive.push_back(py::reinterpret_borrow<py::object>(item));
        out.push_back(qd_python_numpy2arma_Col<double>(keep_alive.back(), true));
    }
    return out;
}

static std::vector<arma::Cube<double>> qd_alias_list_Cube(const py::handle &h, std::vector<py::object> &keep_alive)
{
    py::list lst = py::reinterpret_borrow<py::list>(h);
    std::vector<arma::Cube<double>> out;
    out.reserve(lst.size());
    for (auto item : lst)
    {
        keep_alive.push_back(py::reinterpret_borrow<py::object>(item));
        out.push_back(qd_python_numpy2arma_Cube<double>(keep_alive.back(), true));
    }
    return out;
}

// Add all values of an arrayant dict to 'keep_alive' (the antenna patterns are aliased, see above)
static void qd_keep_dict_values(const py::dict &dict, std::vector<py::object> &keep_alive)
{
    for (auto item : dict)
        keep_alive.push_back(py::reinterpret_borrow<py::object>(item.second));
}

py::tuple get_channels_spherical(const py::dict &ant_tx,
                                 const py::dict &ant_rx,
                                 py::handle fbs_pos,
                                 py::handle lbs_pos,
                                 py::handle path_gain,
                                 py::handle path_length,
                                 py::handle M,
                                 py::handle tx_pos,
                                 py::handle tx_orientation,
                                 py::handle rx_pos,
                                 py::handle rx_orientation,
                                 py::handle center_freq,
                                 py::handle use_absolute_delays,
                                 py::handle add_fake_los_path,
                                 py::handle angles,
                                 py::handle complex,
                                 const arma::uword i_freq,
                                 const int use_kernel)
{
    const bool use_absolute_delays_b = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool add_fake_los_path_b = qd_python_flag(add_fake_los_path, false, "add_fake_los_path");
    const bool angles_b = qd_python_flag(angles, false, "angles");
    const bool want_complex = qd_python_flag(complex, true, "complex");

    // References to every aliased Python array, held until the function returns (the GIL is released below)
    std::vector<py::object> keep_alive;
    qd_keep_dict_values(ant_tx, keep_alive);
    qd_keep_dict_values(ant_rx, keep_alive);

    // Parse the antenna arrays and center frequencies, common to both code paths
    const auto ant_tx_a = qd_python_dict2arrayant(ant_tx, true);
    const auto ant_rx_a = qd_python_dict2arrayant(ant_rx, true);
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);

    arma::uword n_ports_tx = ant_tx_a.n_ports();
    arma::uword n_ports_rx = ant_rx_a.n_ports();

    // Batch mode: process a list of snapshots in parallel
    if (qd_is_array_list(fbs_pos))
    {
        if (!qd_is_array_list(lbs_pos) || !qd_is_array_list(path_gain) ||
            !qd_is_array_list(path_length) || !qd_is_array_list(M))
            throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'M' must all be lists.");

        // Alias the list entries, no path data is copied
        const auto fbs_pos_a = qd_alias_list_Mat(fbs_pos, keep_alive);
        const auto lbs_pos_a = qd_alias_list_Mat(lbs_pos, keep_alive);
        const auto path_gain_a = qd_alias_list_Mat(path_gain, keep_alive);
        const auto path_length_a = qd_alias_list_Col(path_length, keep_alive);
        const auto M_a = qd_alias_list_Cube(M, keep_alive);

        const arma::uword n_snap = (arma::uword)fbs_pos_a.size();

        // Needed before the frequency check below, which indexes all five lists
        if (lbs_pos_a.size() != n_snap || path_gain_a.size() != n_snap ||
            path_length_a.size() != n_snap || M_a.size() != n_snap)
            throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'M' must have the same length.");

        // Positions and orientations, one column per snapshot or one for all; checked by the C++ core
        const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
        const auto tx_orientation_a = qd_python_numpy2arma_Mat<double>(tx_orientation, true);
        const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
        const auto rx_orientation_a = qd_python_numpy2arma_Mat<double>(rx_orientation, true);

        // Check the frequency dimension, path counts are validated by the C++ core
        arma::uword n_freq = 0ULL;
        for (arma::uword i = 0ULL; i < n_snap; ++i)
        {
            arma::uword n_freq_i = path_gain_a[i].n_cols;
            if (i == 0ULL)
                n_freq = n_freq_i;
            else if (n_freq_i != n_freq)
                throw std::invalid_argument("All entries of 'path_gain' must have the same number of frequencies.");

            if (M_a[i].n_slices != n_freq_i)
                throw std::invalid_argument("Inputs 'path_gain' and 'M' must have the same number of frequencies.");
        }

        if (!center_freq_a.empty() && center_freq_a.n_elem != n_freq)
            throw std::invalid_argument("Input 'center_freq' must have 'n_freq' elements.");

        if (n_snap != 0ULL && i_freq >= n_freq)
            throw std::out_of_range("Input 'i_freq' exceeds the number of frequencies.");

        const double center_freq_sel = center_freq_a.empty() ? 0.0 : center_freq_a.at(i_freq);

        // Select the frequency slice of every snapshot, aliased without copying the data
        std::vector<arma::vec> path_gain_f;
        std::vector<arma::mat> M_f;
        path_gain_f.reserve(n_snap), M_f.reserve(n_snap);
        for (arma::uword i = 0ULL; i < n_snap; ++i)
        {
            path_gain_f.emplace_back(const_cast<double *>(path_gain_a[i].colptr(i_freq)), path_gain_a[i].n_rows, false, true);
            M_f.emplace_back(const_cast<double *>(M_a[i].slice_memptr(i_freq)), M_a[i].n_rows, M_a[i].n_cols, false, true);
        }

        // Allocate the outputs; numpy memory is mapped to the Armadillo cubes, which C++ writes in place
        std::vector<arma::cube> coeff_re(n_snap), coeff_im(n_snap), delay(n_snap);
        std::vector<arma::cube> aod(n_snap), eod(n_snap), aoa(n_snap), eoa(n_snap);
        py::list coeff_re_py, coeff_im_py, delay_py, aod_py, eod_py, aoa_py, eoa_py;

        for (arma::uword i = 0ULL; i < n_snap; ++i)
        {
            const arma::uword n_path = add_fake_los_path_b ? fbs_pos_a[i].n_cols + 1ULL : fbs_pos_a[i].n_cols;
            delay_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &delay[i]));
            if (angles_b) // Without angles, the core skips their conversion to the ports
            {
                aod_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aod[i]));
                eod_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eod[i]));
                aoa_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aoa[i]));
                eoa_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eoa[i]));
            }

            if (!want_complex) // Complex coefficients are combined after the C++ call, no zero-copy mapping possible
            {
                coeff_re_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_re[i]));
                coeff_im_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_im[i]));
            }
        }

        {
            py::gil_scoped_release release;
            quadriga_lib::get_channels_spherical_vec<double>(ant_tx_a, ant_rx_a, tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                                             fbs_pos_a, lbs_pos_a, path_gain_f, path_length_a, M_f,
                                                             coeff_re, coeff_im, delay,
                                                             center_freq_sel, use_absolute_delays_b, add_fake_los_path_b,
                                                             angles_b ? &aod : nullptr, angles_b ? &eod : nullptr,
                                                             angles_b ? &aoa : nullptr, angles_b ? &eoa : nullptr, use_kernel);
        }

        py::list out;
        if (want_complex)
        {
            py::list coeff_py;
            for (arma::uword i = 0ULL; i < n_snap; ++i)
                coeff_py.append(qd_python_copy2numpy<double, std::complex<double>>(&coeff_re[i], &coeff_im[i]));
            out.append(coeff_py);
        }
        else
            out.append(coeff_re_py), out.append(coeff_im_py);
        out.append(delay_py);
        if (angles_b)
            out.append(aod_py), out.append(eod_py), out.append(aoa_py), out.append(eoa_py);
        return py::tuple(out);
    }

    // Single-snapshot mode
    const auto fbs_pos_arr = fbs_pos.cast<py::array_t<double>>();
    const auto lbs_pos_arr = lbs_pos.cast<py::array_t<double>>();
    const auto path_gain_arr = path_gain.cast<py::array_t<double>>();
    const auto path_length_arr = path_length.cast<py::array_t<double>>();
    const auto M_arr = M.cast<py::array_t<double>>();
    const auto tx_pos_arr = tx_pos.cast<py::array_t<double>>();
    const auto tx_orientation_arr = tx_orientation.cast<py::array_t<double>>();
    const auto rx_pos_arr = rx_pos.cast<py::array_t<double>>();
    const auto rx_orientation_arr = rx_orientation.cast<py::array_t<double>>();

    const auto fbs_pos_a = qd_python_numpy2arma_Mat(fbs_pos_arr, true);
    const auto lbs_pos_a = qd_python_numpy2arma_Mat(lbs_pos_arr, true);
    const auto path_gain_a = qd_python_numpy2arma_Mat(path_gain_arr, true);
    const auto path_length_a = qd_python_numpy2arma_Col(path_length_arr, true);
    const auto M_a = qd_python_numpy2arma_Cube(M_arr, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos_arr, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation_arr, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos_arr, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation_arr, true, false, "rx_orientation", 3);

    // Derived inputs
    arma::uword n_path_in = fbs_pos_a.n_cols;
    arma::uword n_freq = path_gain_a.n_cols;
    arma::uword n_path = add_fake_los_path_b ? n_path_in + 1 : n_path_in;

    // Check the frequency dimension, path counts are validated by the C++ core
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

    // Outputs written in place by C++; split coefficients only for complex=False (combined after the call otherwise)
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    pybind11::array_t<double> coeff_re_py, coeff_im_py;
    if (!want_complex)
    {
        coeff_re_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_re);
        coeff_im_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_im);
    }
    auto delay_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &delay);
    pybind11::array_t<double> aod_py, eod_py, aoa_py, eoa_py;
    if (angles_b) // Without angles, the core skips their conversion to the ports
    {
        aod_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aod);
        eod_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eod);
        aoa_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aoa);
        eoa_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eoa);
    }

    {
        py::gil_scoped_release release;
        quadriga_lib::get_channels_spherical<double>(ant_tx_a, ant_rx_a,
                                                     tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                                     fbs_pos_a, lbs_pos_a, path_gain_f, path_length_a, M_f,
                                                     coeff_re, coeff_im, delay,
                                                     center_freq_sel, use_absolute_delays_b, add_fake_los_path_b,
                                                     angles_b ? &aod : nullptr, angles_b ? &eod : nullptr,
                                                     angles_b ? &aoa : nullptr, angles_b ? &eoa : nullptr, use_kernel);
    }

    py::list out;
    if (want_complex)
        out.append(qd_python_copy2numpy<double, std::complex<double>>(&coeff_re, &coeff_im));
    else
        out.append(coeff_re_py), out.append(coeff_im_py);
    out.append(delay_py);
    if (angles_b)
        out.append(aod_py), out.append(eod_py), out.append(aoa_py), out.append(eoa_py);
    return py::tuple(out);
}
