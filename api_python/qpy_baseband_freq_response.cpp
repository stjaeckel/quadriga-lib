// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# baseband_freq_response
Compute the baseband frequency response of a MIMO channel

- Transforms time-domain channel coefficients into the frequency-domain transfer function H(f)
- Snapshots are passed as Python lists; each list item is one snapshot, processed in parallel via OpenMP
- Two per-snapshot modes, selected by the ndim of the coefficient arrays:
  - single-freq: 3D items `(n_rx, n_tx, n_path)`; DFT over path delays at the requested carriers (AVX2)
  - multi-freq: 4D items `(n_rx, n_tx, n_path, n_freq_in)`; SLERP interpolation across `freq_in`,
    then delay-induced phase applied per output carrier
- All list items must share ndim, `n_rx`, `n_tx` (and `n_freq_in` for 4D); `n_path` may vary per snapshot
- Coefficients are given either as one complex array `coeff`, or as split `coeff_re` + `coeff_im`;
  supplying both forms is an error
- Output carriers are defined one of two mutually exclusive ways:
  - `bandwidth` paired with `pilot_grid` (or `carriers` for an evenly spaced grid) — single-freq only
  - `freq_in` + `freq_out` as absolute frequencies — required for multi-freq, optional for single-freq
- For single-freq with `freq_in` + `freq_out`, the baseband grid is derived internally; the phase is
  referenced to `freq_in[0]` when given, else to `min(freq_out)`
- `delay` may be broadcast: a `(1, 1, n_path)` item applies the same delays to every RX/TX pair;
  for multi-freq a 3D delay item is reused for all input frequencies
- `snap` selects a 0-based subset of snapshots; omitted processes all snapshots
- Internal arithmetic is single-precision; double inputs are narrowed to float, results widened back

## Usage:
```
# Single-freq, bandwidth + evenly spaced carriers
hmat = quadriga_lib.channel.baseband_freq_response( coeff=coeff, delay=delay, bandwidth=100e6, carriers=128 )

# Single-freq, absolute frequencies
hmat = quadriga_lib.channel.baseband_freq_response( coeff=coeff, delay=delay, freq_in=freq_in, freq_out=freq_out )

# Multi-freq, split real / imaginary parts
hmat = quadriga_lib.channel.baseband_freq_response( coeff_re=cre, coeff_im=cim, delay=delay, freq_in=freq_in, freq_out=freq_out )
```

## Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`
  (single-freq) or `(n_rx, n_tx, n_path, n_freq_in)` (multi-freq); mutually exclusive with
  `coeff_re` / `coeff_im`; default: `None`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item shaped like the matching
  `coeff` item or broadcast as `(1, 1, n_path)` (with an optional 4th dimension `n_freq_in`)
- **`bandwidth`** — Baseband bandwidth in Hz; paired with `pilot_grid` or `carriers`; single-freq only;
  cannot be combined with `freq_out`; default: `0.0`
- **`carriers`** — Number of evenly spaced carriers on `[0, 1]`; used only when both `pilot_grid` and
  `freq_out` are omitted; default: 128
- **`pilot_grid`** — Normalized sub-carrier positions, `0.0` = center, `1.0` = center + `bandwidth`;
  1D array of length `n_carrier`; default: `None`
- **`snap`** — 0-based snapshot indices to process; 1D array of length `n_out`; omitted processes
  all snapshots; default: `None`
- **`coeff_re`** — Real part of channel coefficients; same list/shape layout as `coeff`; must be
  paired with `coeff_im`; mutually exclusive with `coeff`; default: `None`
- **`coeff_im`** — Imaginary part of channel coefficients; same layout as `coeff_re`; default: `None`
- **`freq_in`** — Input sample frequencies in Hz; 1D array; required for multi-freq input; for
  single-freq input used together with `freq_out` as the phase reference; default: `None`
- **`freq_out`** — Absolute output carrier frequencies in Hz; 1D array of length `n_carrier`;
  required for multi-freq input; may replace `bandwidth` + `pilot_grid` for single-freq; default: `None`
- **`remove_delay_phase`** — If `True`, undo the delay-induced phase baked in by the channel
  generator before SLERP, then re-apply it analytically per output carrier; multi-freq only;
  default: `True`

## Outputs:
- **`hmat`** — Complex frequency-domain channel matrix; shape `(n_rx, n_tx, n_carrier, n_out)`,
  where `n_out` is the number of processed snapshots

## See also:
- [[get_channels_spherical]] (single-frequency channel generator)
- [[get_channels_multifreq]] (multi-frequency channel generator)
MD!*/

py::array_t<std::complex<double>> baseband_freq_response(const py::object &coeff,
                                                         const py::object &delay,
                                                         double bandwidth,
                                                         size_t carriers,
                                                         py::handle pilot_grid,
                                                         py::handle snap,
                                                         const py::object &coeff_re,
                                                         const py::object &coeff_im,
                                                         py::handle freq_in,
                                                         py::handle freq_out,
                                                         bool remove_delay_phase)
{
    // Input mode detection and basic validation
    bool have_coeff = !coeff.is_none();
    bool have_cre = !coeff_re.is_none();
    bool have_cim = !coeff_im.is_none();
    bool have_delay = !delay.is_none();

    // Convert optional frequency / pilot inputs up front (handle overloads accept None)
    arma::vec freq_in_arma = qd_python_numpy2arma_Col<double>(freq_in, true);
    arma::vec freq_out_arma = qd_python_numpy2arma_Col<double>(freq_out, true);
    arma::vec pilot_grid_arma_in = qd_python_numpy2arma_Col<double>(pilot_grid, true);
    bool have_freq_in = !freq_in_arma.is_empty();
    bool have_freq_out = !freq_out_arma.is_empty();
    bool have_pilot = !pilot_grid_arma_in.is_empty();
    bool have_bw = (bandwidth > 0.0);

    // Mutual exclusivity checks
    if (have_coeff && (have_cre || have_cim))
        throw std::invalid_argument("Cannot provide both 'coeff' and 'coeff_re'/'coeff_im'.");
    if (have_cre != have_cim)
        throw std::invalid_argument("'coeff_re' and 'coeff_im' must both be provided.");
    if (!have_coeff && !have_cre)
        throw std::invalid_argument("Must provide either 'coeff' or both 'coeff_re' and 'coeff_im'.");
    if (!have_delay)
        throw std::invalid_argument("'delay' must be provided.");

    bool use_complex = have_coeff;

    // Cast inputs to lists and validate types
    if (!py::isinstance<py::list>(delay))
        throw std::invalid_argument("'delay' must be a list.");
    py::list delay_list = py::cast<py::list>(delay);

    py::list coeff_list, cre_list, cim_list;
    if (use_complex)
    {
        if (!py::isinstance<py::list>(coeff))
            throw std::invalid_argument("'coeff' must be a list.");
        coeff_list = py::cast<py::list>(coeff);
    }
    else
    {
        if (!py::isinstance<py::list>(coeff_re))
            throw std::invalid_argument("'coeff_re' must be a list.");
        if (!py::isinstance<py::list>(coeff_im))
            throw std::invalid_argument("'coeff_im' must be a list.");
        cre_list = py::cast<py::list>(coeff_re);
        cim_list = py::cast<py::list>(coeff_im);
    }

    size_t n_snap_total = use_complex ? coeff_list.size() : cre_list.size();
    if (n_snap_total == 0) // Preserve the 4D output contract for an empty snapshot list
    {
        std::vector<arma::Cube<std::complex<double>>> hmat_empty;
        return qd_python_init_output<std::complex<double>>(0, 0, 0, 0, &hmat_empty);
    }

    if (delay_list.size() != n_snap_total)
        throw std::invalid_argument("Length of 'delay' list does not match coefficient list length.");
    if (!use_complex && cim_list.size() != n_snap_total)
        throw std::invalid_argument("Length of 'coeff_im' list does not match 'coeff_re' list length.");

    // Determine dimensionality from first list item
    py::array first_arr;
    if (use_complex)
        first_arr = py::cast<py::array>(coeff_list[0]);
    else
        first_arr = py::cast<py::array>(cre_list[0]);

    int input_ndim = (int)first_arr.ndim();
    if (input_ndim != 3 && input_ndim != 4)
        throw std::invalid_argument("Coefficient arrays must be 3D (single-freq) or 4D (multi-freq), got " +
                                    std::to_string(input_ndim) + "D.");

    bool is_multifreq = (input_ndim == 4);

    // Validate all list items have consistent ndim
    for (size_t i = 1; i < n_snap_total; ++i)
    {
        py::array arr;
        if (use_complex)
            arr = py::cast<py::array>(coeff_list[i]);
        else
            arr = py::cast<py::array>(cre_list[i]);

        if ((int)arr.ndim() != input_ndim)
            throw std::invalid_argument("All coefficient arrays must have the same dimensionality. Item 0 is " +
                                        std::to_string(input_ndim) + "D, but item " + std::to_string(i) +
                                        " is " + std::to_string((int)arr.ndim()) + "D.");
    }

    // Frequency grid validation
    if (is_multifreq)
    {
        if (!have_freq_in)
            throw std::invalid_argument("'freq_in' is required for multi-frequency (4D) inputs.");
        if (!have_freq_out)
            throw std::invalid_argument("'freq_out' is required for multi-frequency (4D) inputs.");
        if (have_bw)
            throw std::invalid_argument("'bandwidth' cannot be used with multi-frequency (4D) inputs; use 'freq_out'.");
        if (have_pilot)
            throw std::invalid_argument("'pilot_grid' cannot be used with multi-frequency (4D) inputs; use 'freq_out'.");
    }
    else
    {
        if (have_freq_out && have_bw)
            throw std::invalid_argument("Cannot provide both 'bandwidth' and 'freq_out'; use one or the other.");
        if (have_pilot && !have_bw)
            throw std::invalid_argument("'bandwidth' must be > 0 when 'pilot_grid' is given.");
        if (!have_bw && !have_freq_out)
            throw std::invalid_argument("Must provide 'bandwidth' (with 'pilot_grid' or 'carriers') or 'freq_out' to define the frequency grid.");
    }

    // Build pilot grid for single-freq mode
    arma::vec pilot_grid_arma;
    if (!is_multifreq)
    {
        if (have_freq_out)
        {
            // Derive bandwidth and pilot_grid from freq_out
            // When freq_in is provided (single carrier frequency at which coefficients were generated),
            // use it as the phase reference: pilot_grid[k] = (freq_out[k] - freq_in) / bandwidth.
            // This ensures the DFT phase offset correctly complements the baked-in delay phase.
            // When freq_in is not provided, use freq_out.min() as the reference (baseband convention).
            double f_ref = (have_freq_in && freq_in_arma.n_elem >= 1) ? freq_in_arma(0) : freq_out_arma.min();
            double f_min = freq_out_arma.min();
            double f_max = freq_out_arma.max();
            bandwidth = f_max - f_min;
            if (bandwidth <= 0.0 && freq_out_arma.n_elem > 1)
                throw std::invalid_argument("'freq_out' values must span a positive range.");
            if (bandwidth <= 0.0)
                bandwidth = 1.0; // Single carrier: avoid division by zero
            pilot_grid_arma.set_size(freq_out_arma.n_elem);
            for (arma::uword k = 0; k < freq_out_arma.n_elem; ++k)
                pilot_grid_arma(k) = (freq_out_arma(k) - f_ref) / bandwidth;
        }
        else if (have_pilot)
        {
            pilot_grid_arma = pilot_grid_arma_in;
        }
        else
        {
            if (carriers == 0)
                throw std::invalid_argument("Number of carriers cannot be 0.");
            pilot_grid_arma = arma::linspace<arma::vec>(0.0, 1.0, (arma::uword)carriers);
        }
    }

    arma::uword n_carrier = is_multifreq ? (arma::uword)freq_out_arma.n_elem : (arma::uword)pilot_grid_arma.n_elem;
    arma::uword n_freq_in_expected = is_multifreq ? (arma::uword)freq_in_arma.n_elem : 0;

    // Pre-parse list items (extract arrays, pointers, shapes; validate dimensions)
    // Must happen before OMP to avoid GIL issues.
    arma::uword n_rx = 0, n_tx = 0;

    // For 3D complex path
    std::vector<std::complex<double> *> coeff_ptrs_cx;
    std::vector<py::array_t<std::complex<double>>> coeff_owned_cx;
    std::vector<std::array<size_t, 9>> coeff_shapes_cx;

    // For 3D real path (coeff_re + coeff_im)
    std::vector<double *> cre_ptrs, cim_ptrs;
    std::vector<py::array_t<double>> cre_owned, cim_owned;
    std::vector<std::array<size_t, 9>> cre_shapes, cim_shapes;

    // For 3D delay (used in both 3D and as fallback in 4D mode)
    std::vector<double *> delay_ptrs;
    std::vector<py::array_t<double>> delay_owned;
    std::vector<std::array<size_t, 9>> delay_shapes;

    // --- 4D pre-parse: manual per-item arrays ---

    std::vector<py::array_t<std::complex<double>>> coeff_arrays_4d_cx;
    std::vector<const std::complex<double> *> coeff_ptrs_4d_cx;
    std::vector<std::array<size_t, 9>> coeff_shapes_4d_cx;
    std::vector<size_t> coeff_nframes_4d_cx, coeff_fstride_4d_cx;

    std::vector<py::array_t<double>> cre_arrays_4d, cim_arrays_4d;
    std::vector<const double *> cre_ptrs_4d, cim_ptrs_4d;
    std::vector<std::array<size_t, 9>> cre_shapes_4d, cim_shapes_4d;
    std::vector<size_t> cre_nframes_4d, cim_nframes_4d;
    std::vector<size_t> cre_fstride_4d, cim_fstride_4d;

    std::vector<py::array_t<double>> delay_arrays_4d;
    std::vector<const double *> delay_ptrs_4d;
    std::vector<std::array<size_t, 9>> delay_shapes_4d;
    std::vector<size_t> delay_nframes_4d, delay_fstride_4d;

    if (!is_multifreq) // ---- 3D single-freq pre-parse ----
    {
        if (use_complex)
            coeff_shapes_cx = qd_python_get_list_shape(coeff_list, coeff_ptrs_cx, coeff_owned_cx);
        else
        {
            cre_shapes = qd_python_get_list_shape(cre_list, cre_ptrs, cre_owned);
            cim_shapes = qd_python_get_list_shape(cim_list, cim_ptrs, cim_owned);
        }
        delay_shapes = qd_python_get_list_shape(delay_list, delay_ptrs, delay_owned);

        // Extract n_rx, n_tx from first item and validate consistency
        const auto &s0 = use_complex ? coeff_shapes_cx[0] : cre_shapes[0];
        n_rx = (arma::uword)s0[0];
        n_tx = (arma::uword)s0[1];

        for (size_t i = 0; i < n_snap_total; ++i)
        {
            const auto &sc = use_complex ? coeff_shapes_cx[i] : cre_shapes[i];

            if (sc[0] != n_rx || sc[1] != n_tx)
                throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                            ": coefficient dimensions (" + std::to_string(sc[0]) + ", " +
                                            std::to_string(sc[1]) + ") do not match expected (" +
                                            std::to_string(n_rx) + ", " + std::to_string(n_tx) + ").");

            if (!use_complex)
            {
                const auto &si = cim_shapes[i];
                if (si[0] != sc[0] || si[1] != sc[1] || si[2] != sc[2])
                    throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                                ": 'coeff_im' shape does not match 'coeff_re'.");
            }

            // Validate delay shape compatibility
            const auto &sd = delay_shapes[i];
            arma::uword n_path_i = (arma::uword)sc[2];
            if (!(sd[0] == n_rx && sd[1] == n_tx && sd[2] == n_path_i) &&
                !(sd[0] == 1 && sd[1] == 1 && sd[2] == n_path_i))
                throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                            ": 'delay' shape is incompatible with coefficient shape.");
        }
    }
    else // ---- 4D multi-freq pre-parse ----
    {
        auto alloc_4d = [&](size_t n)
        {
            if (use_complex)
            {
                coeff_arrays_4d_cx.resize(n);
                coeff_ptrs_4d_cx.resize(n);
                coeff_shapes_4d_cx.resize(n);
                coeff_nframes_4d_cx.resize(n);
                coeff_fstride_4d_cx.resize(n);
            }
            else
            {
                cre_arrays_4d.resize(n);
                cim_arrays_4d.resize(n);
                cre_ptrs_4d.resize(n);
                cim_ptrs_4d.resize(n);
                cre_shapes_4d.resize(n);
                cim_shapes_4d.resize(n);
                cre_nframes_4d.resize(n);
                cim_nframes_4d.resize(n);
                cre_fstride_4d.resize(n);
                cim_fstride_4d.resize(n);
            }
            delay_arrays_4d.resize(n);
            delay_ptrs_4d.resize(n);
            delay_shapes_4d.resize(n);
            delay_nframes_4d.resize(n, 1);
            delay_fstride_4d.resize(n, 0);
        };
        alloc_4d(n_snap_total);

        for (size_t i = 0; i < n_snap_total; ++i)
        {
            // --- Parse coefficient arrays ---
            if (use_complex)
            {
                py::array arr = py::cast<py::array>(coeff_list[i]);
                if (!py::detail::npy_format_descriptor<std::complex<double>>::dtype().is(arr.dtype()))
                    coeff_arrays_4d_cx[i] = py::array_t<std::complex<double>>(arr);
                else
                    coeff_arrays_4d_cx[i] = arr.cast<py::array_t<std::complex<double>>>();

                coeff_shapes_4d_cx[i] = qd_python_get_shape(coeff_arrays_4d_cx[i], false,
                                                            &coeff_nframes_4d_cx[i], &coeff_fstride_4d_cx[i]);
                coeff_ptrs_4d_cx[i] = coeff_arrays_4d_cx[i].data();

                if (i == 0)
                {
                    n_rx = (arma::uword)coeff_shapes_4d_cx[0][0];
                    n_tx = (arma::uword)coeff_shapes_4d_cx[0][1];
                }
                else
                {
                    if (coeff_shapes_4d_cx[i][0] != n_rx || coeff_shapes_4d_cx[i][1] != n_tx)
                        throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                                    ": coefficient dimensions do not match snapshot 0.");
                }
                if (coeff_nframes_4d_cx[i] != n_freq_in_expected)
                    throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                                ": 4th dimension (" + std::to_string(coeff_nframes_4d_cx[i]) +
                                                ") does not match freq_in length (" +
                                                std::to_string(n_freq_in_expected) + ").");
            }
            else
            {
                // coeff_re
                py::array arr_re = py::cast<py::array>(cre_list[i]);
                if (!py::detail::npy_format_descriptor<double>::dtype().is(arr_re.dtype()))
                    cre_arrays_4d[i] = py::array_t<double>(arr_re);
                else
                    cre_arrays_4d[i] = arr_re.cast<py::array_t<double>>();

                cre_shapes_4d[i] = qd_python_get_shape(cre_arrays_4d[i], false,
                                                       &cre_nframes_4d[i], &cre_fstride_4d[i]);
                cre_ptrs_4d[i] = cre_arrays_4d[i].data();

                // coeff_im
                py::array arr_im = py::cast<py::array>(cim_list[i]);
                if (!py::detail::npy_format_descriptor<double>::dtype().is(arr_im.dtype()))
                    cim_arrays_4d[i] = py::array_t<double>(arr_im);
                else
                    cim_arrays_4d[i] = arr_im.cast<py::array_t<double>>();

                cim_shapes_4d[i] = qd_python_get_shape(cim_arrays_4d[i], false,
                                                       &cim_nframes_4d[i], &cim_fstride_4d[i]);
                cim_ptrs_4d[i] = cim_arrays_4d[i].data();

                if (i == 0)
                {
                    n_rx = (arma::uword)cre_shapes_4d[0][0];
                    n_tx = (arma::uword)cre_shapes_4d[0][1];
                }
                else
                {
                    if (cre_shapes_4d[i][0] != n_rx || cre_shapes_4d[i][1] != n_tx)
                        throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                                    ": 'coeff_re' dimensions do not match snapshot 0.");
                }
                if (cre_nframes_4d[i] != n_freq_in_expected)
                    throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                                ": 'coeff_re' 4th dimension does not match freq_in length.");

                // Validate coeff_im matches coeff_re
                if (cim_shapes_4d[i][0] != cre_shapes_4d[i][0] ||
                    cim_shapes_4d[i][1] != cre_shapes_4d[i][1] ||
                    cim_shapes_4d[i][2] != cre_shapes_4d[i][2] ||
                    cim_nframes_4d[i] != cre_nframes_4d[i])
                    throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                                ": 'coeff_im' shape does not match 'coeff_re'.");
            }

            // --- Parse delay arrays (can be 3D or 4D) ---
            py::array arr_dl = py::cast<py::array>(delay_list[i]);
            if (!py::detail::npy_format_descriptor<double>::dtype().is(arr_dl.dtype()))
                delay_arrays_4d[i] = py::array_t<double>(arr_dl);
            else
                delay_arrays_4d[i] = arr_dl.cast<py::array_t<double>>();

            int dl_ndim = (int)delay_arrays_4d[i].ndim();
            if (dl_ndim == 3)
                delay_shapes_4d[i] = qd_python_get_shape(delay_arrays_4d[i], false);
            else if (dl_ndim == 4)
                delay_shapes_4d[i] = qd_python_get_shape(delay_arrays_4d[i], false,
                                                         &delay_nframes_4d[i], &delay_fstride_4d[i]);
            else
                throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                            ": 'delay' must be 3D or 4D, got " + std::to_string(dl_ndim) + "D.");

            delay_ptrs_4d[i] = delay_arrays_4d[i].data();

            // Validate delay shape compatibility with coefficients
            const auto &sc = use_complex ? coeff_shapes_4d_cx[i] : cre_shapes_4d[i];
            const auto &sd = delay_shapes_4d[i];
            arma::uword n_path_i = (arma::uword)sc[2];
            if (!(sd[0] == n_rx && sd[1] == n_tx && sd[2] == n_path_i) &&
                !(sd[0] == 1 && sd[1] == 1 && sd[2] == n_path_i))
                throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                            ": 'delay' shape is incompatible with coefficient shape.");

            if (dl_ndim == 4 && delay_nframes_4d[i] != n_freq_in_expected)
                throw std::invalid_argument("Snapshot " + std::to_string(i) +
                                            ": 'delay' 4th dimension does not match freq_in length.");
        }
    }

    // Process snapshot indices and allocate output
    arma::uvec snap_arma = qd_python_numpy2arma_Col<arma::uword>(snap, true);
    if (snap_arma.is_empty())
        snap_arma = arma::regspace<arma::uvec>(0, (arma::uword)n_snap_total - 1);
    else
    {
        for (arma::uword i = 0; i < snap_arma.n_elem; ++i)
            if (snap_arma(i) >= (arma::uword)n_snap_total)
                throw std::invalid_argument("Snapshot index " + std::to_string(snap_arma(i)) +
                                            " exceeds list length (" + std::to_string(n_snap_total) + ").");
    }
    arma::uword n_snap_out = snap_arma.n_elem;

    // Allocate 4D output: (n_rx, n_tx, n_carrier, n_snap_out)
    // Map each snapshot slice to a Cube for zero-copy writes
    std::vector<arma::Cube<std::complex<double>>> hmat_vec;
    auto output = qd_python_init_output<std::complex<double>>(n_rx, n_tx, n_carrier, n_snap_out, &hmat_vec);

    //  OMP parallel loop over snapshots
    int n_snap_omp = (int)n_snap_out;

    if (!is_multifreq) // ---- Single-freq (3D) path ----
    {
#pragma omp parallel
        {
            // Per-thread buffers — allocated once, reused across snapshots
            arma::Cube<double> cre_snap, cim_snap, dl_snap;

#pragma omp for schedule(static)
            for (int ii = 0; ii < n_snap_omp; ++ii)
            {
                arma::uword j = snap_arma[(arma::uword)ii]; // Source snapshot index

                if (use_complex)
                {
                    // Split complex → re/im
                    const auto &sc = coeff_shapes_cx[j];
                    if (sc[7] == 0) // Empty snapshot
                    {
                        hmat_vec[ii].zeros();
                        continue;
                    }
                    qd_python_copy2arma(coeff_ptrs_cx[j], sc, cre_snap, cim_snap);
                }
                else
                {
                    const auto &sc = cre_shapes[j];
                    if (sc[7] == 0) // Empty snapshot
                    {
                        hmat_vec[ii].zeros();
                        continue;
                    }
                    qd_python_copy2arma(cre_ptrs[j], sc, cre_snap);
                    qd_python_copy2arma(cim_ptrs[j], cim_shapes[j], cim_snap);
                }

                qd_python_copy2arma(delay_ptrs[j], delay_shapes[j], dl_snap);

                // Call single-freq DFT
                quadriga_lib::baseband_freq_response<double>(&cre_snap, &cim_snap, &dl_snap,
                                                             &pilot_grid_arma, bandwidth,
                                                             (arma::Cube<double> *)nullptr,
                                                             (arma::Cube<double> *)nullptr,
                                                             &hmat_vec[ii]);
            }
        }
    }
    else // ---- Multi-freq (4D) path ----
    {
#pragma omp parallel
        {
            // Per-thread buffers — allocated once, reused across snapshots
            std::vector<arma::Cube<double>> cre_vec(n_freq_in_expected);
            std::vector<arma::Cube<double>> cim_vec(n_freq_in_expected);
            std::vector<arma::Cube<double>> dl_vec(n_freq_in_expected);
            arma::Cube<double> dl_single; // For 3D delay broadcast

#pragma omp for schedule(static)
            for (int ii = 0; ii < n_snap_omp; ++ii)
            {
                arma::uword j = snap_arma[(arma::uword)ii]; // Source snapshot index

                // --- Extract coefficients per frequency ---
                if (use_complex)
                {
                    const auto &sc = coeff_shapes_4d_cx[j];
                    const std::complex<double> *base = coeff_ptrs_4d_cx[j];
                    size_t fstride = coeff_fstride_4d_cx[j];

                    if (sc[7] == 0) // Empty snapshot
                    {
                        hmat_vec[ii].zeros();
                        continue;
                    }

                    for (arma::uword f = 0; f < n_freq_in_expected; ++f)
                        qd_python_copy2arma(base + f * fstride, sc, cre_vec[f], cim_vec[f]);
                }
                else
                {
                    const auto &sc_re = cre_shapes_4d[j];
                    const auto &sc_im = cim_shapes_4d[j];
                    const double *base_re = cre_ptrs_4d[j];
                    const double *base_im = cim_ptrs_4d[j];
                    size_t fstride_re = cre_fstride_4d[j];
                    size_t fstride_im = cim_fstride_4d[j];

                    if (sc_re[7] == 0)
                    {
                        hmat_vec[ii].zeros();
                        continue;
                    }

                    for (arma::uword f = 0; f < n_freq_in_expected; ++f)
                    {
                        qd_python_copy2arma(base_re + f * fstride_re, sc_re, cre_vec[f]);
                        qd_python_copy2arma(base_im + f * fstride_im, sc_im, cim_vec[f]);
                    }
                }

                // --- Extract delay per frequency ---
                const double *dl_base = delay_ptrs_4d[j];
                const auto &sd = delay_shapes_4d[j];
                size_t dl_nf = delay_nframes_4d[j];
                size_t dl_fs = delay_fstride_4d[j];

                if (dl_nf == 1)
                {
                    // 3D delay: broadcast to all frequencies
                    qd_python_copy2arma(dl_base, sd, dl_single);
                    for (arma::uword f = 0; f < n_freq_in_expected; ++f)
                        dl_vec[f] = dl_single;
                }
                else
                {
                    for (arma::uword f = 0; f < n_freq_in_expected; ++f)
                        qd_python_copy2arma(dl_base + f * dl_fs, sd, dl_vec[f]);
                }

                // --- Call multi-freq SLERP + accumulate ---
                quadriga_lib::baseband_freq_response_multi<double>(
                    cre_vec, cim_vec, dl_vec,
                    freq_in_arma, freq_out_arma,
                    (arma::Cube<double> *)nullptr,
                    (arma::Cube<double> *)nullptr,
                    &hmat_vec[ii],
                    remove_delay_phase);
            }
        }
    }

    return output;
}

// pybind11 declaration (channel submodule):
// m.def("baseband_freq_response", &baseband_freq_response,
//       py::arg("coeff") = py::none(),
//       py::arg("delay") = py::none(),
//       py::arg("bandwidth") = 0.0,
//       py::arg("carriers") = 128,
//       py::arg("pilot_grid") = py::none(),
//       py::arg("snap") = py::none(),
//       py::arg("coeff_re") = py::none(),
//       py::arg("coeff_im") = py::none(),
//       py::arg("freq_in") = py::none(),
//       py::arg("freq_out") = py::none(),
//       py::arg("remove_delay_phase") = true);