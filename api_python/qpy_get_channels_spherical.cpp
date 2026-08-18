// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths,
  using the spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation
  (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `M` (interleaved Re/Im for VV, VH, HV, HH components).
- Path data is given for `n_freq` frequencies; single-frequency data uses `n_freq = 1`.
- The scalar argument `freq` selects the frequency index that is processed; a single frequency is returned per call.
- If `center_freq` is `None` or the selected entry is `0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == False`, the minimum delay (LOS delay) is subtracted from all paths.
- If `add_fake_los_path == True`, a zero-power LOS path is prepended when no LOS path is detected.
- `complex=True` returns one combined complex coefficient array `coeff`; `complex=False` (default) returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path

## Batch mode:
- If `fbs_pos` is a `list` of arrays, the function switches to batch mode and processes `n_out` snapshots
  in one call, distributing the snapshots over all available cores (OpenMP).
- The path inputs `fbs_pos`, `lbs_pos`, `path_gain`, `path_length` and `M` are then lists of length `n_out`,
  matching the output format of [[qrt_file_read]]; list entries are aliased, not copied.
- The position and orientation inputs are `(3, n_out)` arrays with one column per snapshot; a `(3,)` array
  or a single-column `(3, 1)` array is applied to all snapshots.
- All outputs are returned as lists of length `n_out`, one entry per snapshot.
- Snapshots without any path produce an empty output entry with `n_path = 0`.
- Batch mode releases the GIL while processing, so it can be combined with Python threads.

## Usage:
```
coeff_re, coeff_im, delays = quadriga_lib.arrayant.get_channels_spherical( ant_tx, ant_rx, \
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, \
    center_freq, use_absolute_delays, add_fake_los_path )

coeff, delays = quadriga_lib.arrayant.get_channels_spherical( ..., complex=True )

coeff_re, coeff_im, delays, aod, eod, aoa, eoa = quadriga_lib.arrayant.get_channels_spherical( ..., angles=True )

coeff_re, coeff_im, delays = quadriga_lib.arrayant.get_channels_spherical( ..., freq=1 )
```

## Inputs:
- **`ant_tx`** — Transmit arrayant dict; see [[generate]]
- **`ant_rx`** — Receive arrayant dict; see [[generate]]
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`, or a list of length `n_out` in batch mode
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`, or a list of length `n_out` in batch mode
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`, or a list of length `n_out` in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; `(n_path,)`, or a list of length `n_out` in batch mode
- **`M`** — Polarization transfer matrix, interleaved Re/Im; `(8, n_path, n_freq)`, or a list of length
  `n_out` in batch mode; (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH)
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`, or `(3, n_out)` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`, or `(3, n_out)` in batch mode
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`, or `(3, n_out)` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`, or `(3, n_out)` in batch mode
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; set an entry to `0` or pass `None` to skip phase computation; default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, prepends a zero-power LOS path when none is present; default: `False`
- **`angles`** — If `True`, also return departure/arrival angles in antenna-local coordinates; default: `False`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `False`
- **`freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `M` and `center_freq` that is
  processed; default: `0`

## Outputs:
- **`coeff_re`** — Real part of channel coefficients (`complex=False`); `(n_ports_rx, n_ports_tx, n_path)`
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape as `coeff_re`
- **`coeff`** — Complex channel coefficients (`complex=True`), replaces `coeff_re`/`coeff_im`; same shape
- **`delays`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`aod`** — Azimuth of departure in rad; `(n_ports_rx, n_ports_tx, n_path)`; only when `angles=True`
- **`eod`** — Elevation of departure in rad; same shape; only when `angles=True`
- **`aoa`** — Azimuth of arrival in rad; same shape; only when `angles=True`
- **`eoa`** — Elevation of arrival in rad; same shape; only when `angles=True`
- In batch mode, every output is a list of length `n_out` whose entries have the shapes given above.

## See also:
- [[get_channels_multifreq]] (multi-frequency extension)
- [[get_channels_planar]] (planar wave variant)
- [[get_channels_irs]] (IRS-assisted communication)
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

// Alias the entries of a list of numpy arrays, the array data itself is not copied
static std::vector<arma::Mat<double>> qd_alias_list_Mat(const py::handle &h)
{
    py::list lst = py::reinterpret_borrow<py::list>(h);
    std::vector<arma::Mat<double>> out;
    out.reserve(lst.size());
    for (auto item : lst)
        out.push_back(qd_python_numpy2arma_Mat<double>(item, true));
    return out;
}

static std::vector<arma::Col<double>> qd_alias_list_Col(const py::handle &h)
{
    py::list lst = py::reinterpret_borrow<py::list>(h);
    std::vector<arma::Col<double>> out;
    out.reserve(lst.size());
    for (auto item : lst)
        out.push_back(qd_python_numpy2arma_Col<double>(item, true));
    return out;
}

static std::vector<arma::Cube<double>> qd_alias_list_Cube(const py::handle &h)
{
    py::list lst = py::reinterpret_borrow<py::list>(h);
    std::vector<arma::Cube<double>> out;
    out.reserve(lst.size());
    for (auto item : lst)
        out.push_back(qd_python_numpy2arma_Cube<double>(item, true));
    return out;
}

// Select the column of a (3, n_out) matrix belonging to a snapshot, a single column applies to all snapshots
static const double *qd_snapshot_col(const arma::Mat<double> &mat, arma::uword i_snap)
{
    return mat.colptr(mat.n_cols == 1ULL ? 0ULL : i_snap);
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
                                 const bool use_absolute_delays,
                                 const bool add_fake_los_path,
                                 const bool angles,
                                 const bool complex,
                                 const arma::uword freq)
{
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
        const auto fbs_pos_a = qd_alias_list_Mat(fbs_pos);
        const auto lbs_pos_a = qd_alias_list_Mat(lbs_pos);
        const auto path_gain_a = qd_alias_list_Mat(path_gain);
        const auto path_length_a = qd_alias_list_Col(path_length);
        const auto M_a = qd_alias_list_Cube(M);

        const arma::uword n_out = (arma::uword)fbs_pos_a.size();

        if (lbs_pos_a.size() != n_out || path_gain_a.size() != n_out ||
            path_length_a.size() != n_out || M_a.size() != n_out)
            throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'M' must have the same length.");

        // Positions and orientations, one column per snapshot
        const auto tx_pos_arr = tx_pos.cast<py::array_t<double>>();
        const auto tx_orientation_arr = tx_orientation.cast<py::array_t<double>>();
        const auto rx_pos_arr = rx_pos.cast<py::array_t<double>>();
        const auto rx_orientation_arr = rx_orientation.cast<py::array_t<double>>();

        const auto tx_pos_a = qd_python_numpy2arma_Mat(tx_pos_arr, true);
        const auto tx_orientation_a = qd_python_numpy2arma_Mat(tx_orientation_arr, true);
        const auto rx_pos_a = qd_python_numpy2arma_Mat(rx_pos_arr, true);
        const auto rx_orientation_a = qd_python_numpy2arma_Mat(rx_orientation_arr, true);

        if (tx_pos_a.n_rows != 3ULL || tx_orientation_a.n_rows != 3ULL ||
            rx_pos_a.n_rows != 3ULL || rx_orientation_a.n_rows != 3ULL)
            throw std::invalid_argument("Inputs 'tx_pos', 'tx_orientation', 'rx_pos' and 'rx_orientation' must have 3 rows.");

        if ((tx_pos_a.n_cols != 1ULL && tx_pos_a.n_cols != n_out) ||
            (tx_orientation_a.n_cols != 1ULL && tx_orientation_a.n_cols != n_out) ||
            (rx_pos_a.n_cols != 1ULL && rx_pos_a.n_cols != n_out) ||
            (rx_orientation_a.n_cols != 1ULL && rx_orientation_a.n_cols != n_out))
            throw std::invalid_argument("Inputs 'tx_pos', 'tx_orientation', 'rx_pos' and 'rx_orientation' must have 1 or 'n_out' columns.");

        // Check the frequency dimension, path counts are validated by the C++ core
        arma::uword n_freq = 0ULL;
        for (arma::uword i = 0ULL; i < n_out; ++i)
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

        if (n_out != 0ULL && freq >= n_freq)
            throw std::out_of_range("Input 'freq' exceeds the number of frequencies.");

        const double center_frequency = center_freq_a.empty() ? 0.0 : center_freq_a.at(freq);

        // Number of output paths per snapshot
        std::vector<arma::uword> n_path(n_out);
        for (arma::uword i = 0ULL; i < n_out; ++i)
        {
            arma::uword n_path_in = fbs_pos_a[i].n_cols;
            n_path[i] = (add_fake_los_path && n_path_in != 0ULL) ? n_path_in + 1ULL : n_path_in;
        }

        // Allocate all outputs before the parallel region, numpy memory is mapped to the Armadillo cubes
        std::vector<arma::cube> coeff_re(n_out), coeff_im(n_out), delay(n_out);
        std::vector<arma::cube> aod(n_out), eod(n_out), aoa(n_out), eoa(n_out);
        py::list coeff_re_py, coeff_im_py, delay_py, aod_py, eod_py, aoa_py, eoa_py;

        for (arma::uword i = 0ULL; i < n_out; ++i)
        {
            delay_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path[i], &delay[i]));

            if (angles)
            {
                aod_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path[i], &aod[i]));
                eod_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path[i], &eod[i]));
                aoa_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path[i], &aoa[i]));
                eoa_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path[i], &eoa[i]));
            }

            if (complex) // Coefficients are combined after the C++ call, no zero-copy mapping possible
            {
                coeff_re[i].zeros(n_ports_rx, n_ports_tx, n_path[i]);
                coeff_im[i].zeros(n_ports_rx, n_ports_tx, n_path[i]);
            }
            else
            {
                coeff_re_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path[i], &coeff_re[i]));
                coeff_im_py.append(qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path[i], &coeff_im[i]));
            }
        }

        // Process all snapshots in parallel, no Python object may be touched inside this region
        std::string error_message;
        {
            py::gil_scoped_release release;

#if defined(_OPENMP) && _OPENMP >= 200805
            omp_set_max_active_levels(1); // Parallelize over snapshots, the path loop in the core stays serial
#elif defined(_OPENMP)
            omp_set_nested(0); // OpenMP 2.0 (MSVC /openmp) has no max_active_levels
#endif

#pragma omp parallel for schedule(dynamic)
            for (long long i = 0LL; i < (long long)n_out; ++i)
            {
                try
                {
                    const arma::uword j = (arma::uword)i;
                    if (n_path[j] == 0ULL) // Snapshot without any path
                    {
                        coeff_re[j].zeros();
                        coeff_im[j].zeros();
                        delay[j].zeros();
                        if (angles)
                            aod[j].zeros(), eod[j].zeros(), aoa[j].zeros(), eoa[j].zeros();
                        continue;
                    }

                    // Select the frequency slice, aliased without copying the data
                    const arma::Col<double> path_gain_f(const_cast<double *>(path_gain_a[j].colptr(freq)),
                                                        path_gain_a[j].n_rows, false, true);
                    const arma::Mat<double> M_f(const_cast<double *>(M_a[j].slice_memptr(freq)),
                                                M_a[j].n_rows, M_a[j].n_cols, false, true);

                    const double *T = qd_snapshot_col(tx_pos_a, j);
                    const double *To = qd_snapshot_col(tx_orientation_a, j);
                    const double *R = qd_snapshot_col(rx_pos_a, j);
                    const double *Ro = qd_snapshot_col(rx_orientation_a, j);

                    quadriga_lib::get_channels_spherical<double>(&ant_tx_a, &ant_rx_a,
                                                                 T[0], T[1], T[2], To[0], To[1], To[2],
                                                                 R[0], R[1], R[2], Ro[0], Ro[1], Ro[2],
                                                                 &fbs_pos_a[j], &lbs_pos_a[j], &path_gain_f, &path_length_a[j], &M_f,
                                                                 &coeff_re[j], &coeff_im[j], &delay[j],
                                                                 center_frequency, use_absolute_delays, add_fake_los_path,
                                                                 angles ? &aod[j] : nullptr,
                                                                 angles ? &eod[j] : nullptr,
                                                                 angles ? &aoa[j] : nullptr,
                                                                 angles ? &eoa[j] : nullptr);
                }
                catch (const std::exception &e)
                {
#pragma omp critical
                    if (error_message.empty())
                        error_message = "Snapshot " + std::to_string(i) + ": " + e.what();
                }
            }
        }

        if (!error_message.empty())
            throw std::invalid_argument(error_message.c_str());

        if (complex)
        {
            py::list coeff_py;
            for (arma::uword i = 0ULL; i < n_out; ++i)
                coeff_py.append(qd_python_copy2numpy<double, std::complex<double>>(&coeff_re[i], &coeff_im[i]));

            if (angles)
                return py::make_tuple(coeff_py, delay_py, aod_py, eod_py, aoa_py, eoa_py);
            else
                return py::make_tuple(coeff_py, delay_py);
        }

        if (angles)
            return py::make_tuple(coeff_re_py, coeff_im_py, delay_py, aod_py, eod_py, aoa_py, eoa_py);
        else
            return py::make_tuple(coeff_re_py, coeff_im_py, delay_py);
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

    // Extract scalar values
    double Tx = tx_pos_a.at(0), Ty = tx_pos_a.at(1), Tz = tx_pos_a.at(2);
    double Tb = tx_orientation_a.at(0), Tt = tx_orientation_a.at(1), Th = tx_orientation_a.at(2);
    double Rx = rx_pos_a.at(0), Ry = rx_pos_a.at(1), Rz = rx_pos_a.at(2);
    double Rb = rx_orientation_a.at(0), Rt = rx_orientation_a.at(1), Rh = rx_orientation_a.at(2);

    // Derived inputs
    arma::uword n_path_in = fbs_pos_a.n_cols;
    arma::uword n_freq = path_gain_a.n_cols;
    arma::uword n_path = add_fake_los_path ? n_path_in + 1 : n_path_in;

    // Check the frequency dimension, path counts are validated by the C++ core
    if (M_a.n_slices != n_freq)
        throw std::invalid_argument("Inputs 'path_gain' and 'M' must have the same number of frequencies.");

    if (!center_freq_a.empty() && center_freq_a.n_elem != n_freq)
        throw std::invalid_argument("Input 'center_freq' must have 'n_freq' elements.");

    if (freq >= n_freq)
        throw std::out_of_range("Input 'freq' exceeds the number of frequencies.");

    // Select the frequency slice, aliased without copying the data
    const arma::Col<double> path_gain_f(const_cast<double *>(path_gain_a.colptr(freq)), path_gain_a.n_rows, false, true);
    const arma::Mat<double> M_f(const_cast<double *>(M_a.slice_memptr(freq)), M_a.n_rows, M_a.n_cols, false, true);
    const double center_frequency = center_freq_a.empty() ? 0.0 : center_freq_a.at(freq);

    // Initialize angles
    arma::cube aod, eod, aoa, eoa;
    pybind11::array_t<double> aod_py, eod_py, aoa_py, eoa_py;
    arma::cube *p_aod = nullptr, *p_eod = nullptr, *p_aoa = nullptr, *p_eoa = nullptr;
    if (angles)
    {
        aod_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aod);
        eod_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eod);
        aoa_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &aoa);
        eoa_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &eoa);
        p_aod = &aod;
        p_eod = &eod;
        p_aoa = &aoa;
        p_eoa = &eoa;
    }

    // Initialize delays
    arma::cube coeff_re, coeff_im, delay;
    auto delay_py = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &delay);

    if (complex)
    {
        {
            py::gil_scoped_release release;
            quadriga_lib::get_channels_spherical<double>(&ant_tx_a, &ant_rx_a,
                                                         Tx, Ty, Tz, Tb, Tt, Th,
                                                         Rx, Ry, Rz, Rb, Rt, Rh,
                                                         &fbs_pos_a, &lbs_pos_a, &path_gain_f, &path_length_a, &M_f,
                                                         &coeff_re, &coeff_im, &delay,
                                                         center_frequency, use_absolute_delays, add_fake_los_path,
                                                         p_aod, p_eod, p_aoa, p_eoa);
        }

        auto coeff_py = qd_python_copy2numpy<double, std::complex<double>>(&coeff_re, &coeff_im);

        if (angles)
            return py::make_tuple(coeff_py, delay_py, aod_py, eod_py, aoa_py, eoa_py);
        else
            return py::make_tuple(coeff_py, delay_py);
    }

    // Real path: zero-copy outputs written in place by C++
    auto coeff_re_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_re);
    auto coeff_im_p = qd_python_init_output<double>(n_ports_rx, n_ports_tx, n_path, &coeff_im);

    {
        py::gil_scoped_release release;
        quadriga_lib::get_channels_spherical<double>(&ant_tx_a, &ant_rx_a,
                                                     Tx, Ty, Tz, Tb, Tt, Th,
                                                     Rx, Ry, Rz, Rb, Rt, Rh,
                                                     &fbs_pos_a, &lbs_pos_a, &path_gain_f, &path_length_a, &M_f,
                                                     &coeff_re, &coeff_im, &delay,
                                                     center_frequency, use_absolute_delays, add_fake_los_path,
                                                     p_aod, p_eod, p_aoa, p_eoa);
    }

    if (angles)
        return py::make_tuple(coeff_re_p, coeff_im_p, delay_py, aod_py, eod_py, aoa_py, eoa_py);
    else
        return py::make_tuple(coeff_re_p, coeff_im_p, delay_py);
}

// pybind11 declaration:
// m.def("get_channels_spherical", &get_channels_spherical,
//       py::arg("ant_tx"),
//       py::arg("ant_rx"),
//       py::arg("fbs_pos"),
//       py::arg("lbs_pos"),
//       py::arg("path_gain"),
//       py::arg("path_length"),
//       py::arg("M"),
//       py::arg("tx_pos"),
//       py::arg("tx_orientation"),
//       py::arg("rx_pos"),
//       py::arg("rx_orientation"),
//       py::arg("center_freq") = py::none(),
//       py::arg("use_absolute_delays") = false,
//       py::arg("add_fake_los_path") = false,
//       py::arg("angles") = false,
//       py::arg("complex") = false,
//       py::arg("freq") = 0);