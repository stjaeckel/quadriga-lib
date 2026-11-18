// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp" // qd_python_flag
#include "quadriga_lib.hpp"
#include "quadriga_lib_helper_functions.hpp" // qd_batch_loop

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# baseband_freq_response
Compute the baseband frequency response of a MIMO channel

- Transforms time-domain channel coefficients into the frequency-domain transfer function H(f)
- Snapshots are passed as Python lists; each list item is one snapshot, processed in parallel via OpenMP in C++
- Two per-snapshot modes, selected by the ndim of the coefficient arrays:
  - single-freq: 3D items `(n_rx, n_tx, n_path)`; DFT over path delays at the requested carriers (AVX2)
  - multi-freq: 4D items `(n_rx, n_tx, n_path, n_freq)`; SLERP interpolation across `freq`,
    then delay-induced phase applied per output carrier
- All list items must share ndim, `n_rx`, `n_tx` (and `n_freq` for 4D); `n_path` may vary per snapshot
- Output carriers are defined one of two mutually exclusive ways:
  - `bandwidth` paired with `pilot_grid` (or `carriers` for an evenly spaced grid) — single-freq only
  - `freq` + `freq_new` as absolute frequencies — required for multi-freq, optional for single-freq
- For single-freq with `freq` + `freq_new`, the baseband grid is derived internally; the phase is
  referenced to `freq[0]` when given, else to `min(freq_new)`
- `delay` may be broadcast: a `(1, 1, n_path)` item applies the same delays to every RX/TX pair;
  for multi-freq a 3D delay item is reused for all input frequencies
- single-freq: the phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to
  one period before sin / cos are evaluated in single precision; coefficients and sums are single precision, double
  inputs are narrowed to float, results widened back; multi-freq computes in double precision

## Usage:
```
hmat = quadriga_lib.channel.baseband_freq_response( coeff, delay, *, bandwidth, carriers, pilot_grid, \
    snap, freq, freq_new, remove_delay_phase, use_kernel )
```

## Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`
  (single-freq) or `(n_rx, n_tx, n_path, n_freq)` (multi-freq)
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item shaped like the matching
  `coeff` item or broadcast as `(1, 1, n_path)` (with an optional 4th dimension `n_freq`)
- **`bandwidth`** — Baseband bandwidth in Hz; paired with `pilot_grid` or `carriers`; single-freq only;
  cannot be combined with `freq_new`; default: `0.0`
- **`carriers`** — Number of evenly spaced carriers on `[0, 1]`; used only when both `pilot_grid` and
  `freq_new` are omitted; default: 128
- **`pilot_grid`** — Normalized sub-carrier positions, `0.0` = center, `1.0` = center + `bandwidth`;
  1D array of length `n_carrier`; default: `None`
- **`snap`** — 0-based snapshot indices to process; 1D array of length `n_out`; default: `None` (all snapshots)
- **`freq`** — Input sample frequencies in Hz; 1D array; required for multi-freq input; for
  single-freq input used together with `freq_new` as the phase reference; default: `None`
- **`freq_new`** — Absolute output carrier frequencies in Hz; 1D array of length `n_carrier`;
  required for multi-freq input; may replace `bandwidth` + `pilot_grid` for single-freq; default: `None`
- **`remove_delay_phase`** — If `True`, undo the delay-induced phase baked in by the channel generator before SLERP,
  then re-apply it analytically per output carrier; multi-freq only; needs channels generated with
  `use_absolute_delays = True`: the coefficients of the generators carry the phase of the absolute path length also
  when `delay` is relative, and with relative delays the phase of the TX-RX distance is interpolated across `freq`,
  which gives wrong phases (magnitudes stay right); default: `True`
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (raises if unavailable); default: 0

## Outputs:
- **`hmat`** — Complex frequency-domain channel matrix; shape `(n_rx, n_tx, n_carrier, n_out)`,
  where `n_out` is the number of processed snapshots

## See also:
- [[channel.spherical]] (single-frequency channel generator)
- [[channel.multifreq]] (multi-frequency channel generator)
MD!*/

py::array_t<std::complex<double>> baseband_freq_response(const py::handle &coeff,
                                                         const py::handle &delay,
                                                         double bandwidth,
                                                         size_t carriers,
                                                         py::handle pilot_grid,
                                                         py::handle snap,
                                                         py::handle freq,
                                                         py::handle freq_new,
                                                         py::handle remove_delay_phase,
                                                         int use_kernel)
{
    // Kernel, checked here because an empty snapshot list skips the C++ call
    if (use_kernel < 0 || use_kernel > 2)
        throw std::invalid_argument("Input 'use_kernel' must be 0, 1 or 2.");
    if (use_kernel == 2 && !quadriga_lib::quadriga_lib_has_AVX2())
        throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
    const bool remove_delay_phase_b = qd_python_flag(remove_delay_phase, true, "remove_delay_phase");

    // Optional frequency / pilot inputs (None gives an empty vector)
    arma::vec freq_arma = qd_python_numpy2arma_Col<double>(freq, true);
    arma::vec freq_new_arma = qd_python_numpy2arma_Col<double>(freq_new, true);
    arma::vec pilot_grid_arma_in = qd_python_numpy2arma_Col<double>(pilot_grid, true);
    bool have_freq = !freq_arma.is_empty();
    bool have_freq_new = !freq_new_arma.is_empty();
    bool have_pilot = !pilot_grid_arma_in.is_empty();
    bool have_bw = (bandwidth > 0.0);

    if (!py::isinstance<py::list>(coeff))
        throw std::invalid_argument("'coeff' must be a list.");
    if (!py::isinstance<py::list>(delay))
        throw std::invalid_argument("'delay' must be a list.");
    const py::list coeff_list = py::reinterpret_borrow<py::list>(coeff), delay_list = py::reinterpret_borrow<py::list>(delay);

    const size_t n_snap_total = coeff_list.size();
    if (n_snap_total == 0) // Preserve the 4D output contract for an empty snapshot list
    {
        std::vector<arma::Cube<std::complex<double>>> hmat_empty;
        return qd_python_init_output<std::complex<double>>(0, 0, 0, 0, &hmat_empty);
    }
    if (delay_list.size() != n_snap_total)
        throw std::invalid_argument("Length of 'delay' list does not match coefficient list length.");

    // Snapshot indices; only the selected snapshots are read and converted
    arma::uvec snap_arma = qd_python_numpy2arma_Col<arma::uword>(snap, true, false, "snap");
    if (snap_arma.is_empty())
        snap_arma = arma::regspace<arma::uvec>(0, (arma::uword)n_snap_total - 1);
    for (arma::uword j : snap_arma)
        if (j >= (arma::uword)n_snap_total)
            throw std::invalid_argument("Snapshot index " + std::to_string(j) + " exceeds list length (" + std::to_string(n_snap_total) + ").");
    const arma::uvec src = arma::unique(snap_arma);
    const size_t n_src = src.n_elem;

    // List items of the selected snapshots, read with the GIL
    std::vector<qd_python_item<std::complex<double>>> c_item(n_src);
    std::vector<qd_python_item<double>> dl_item(n_src);
    for (size_t k = 0; k < n_src; ++k)
        c_item[k] = qd_python_get_item<std::complex<double>>(coeff_list[src[k]]), dl_item[k] = qd_python_get_item<double>(delay_list[src[k]]);

    const int input_ndim = c_item[0].ndim;
    if (input_ndim != 3 && input_ndim != 4)
        throw std::invalid_argument("Coefficient arrays must be 3D (single-freq) or 4D (multi-freq), got " + std::to_string(input_ndim) + "D.");
    const bool is_multifreq = (input_ndim == 4);

    // Frequency grid validation
    if (is_multifreq)
    {
        if (!have_freq)
            throw std::invalid_argument("'freq' is required for multi-frequency (4D) inputs.");
        if (!have_freq_new)
            throw std::invalid_argument("'freq_new' is required for multi-frequency (4D) inputs.");
        if (have_bw)
            throw std::invalid_argument("'bandwidth' cannot be used with multi-frequency (4D) inputs; use 'freq_new'.");
        if (have_pilot)
            throw std::invalid_argument("'pilot_grid' cannot be used with multi-frequency (4D) inputs; use 'freq_new'.");
    }
    else
    {
        if (have_freq_new && have_bw)
            throw std::invalid_argument("Cannot provide both 'bandwidth' and 'freq_new'; use one or the other.");
        if (have_pilot && !have_bw)
            throw std::invalid_argument("'bandwidth' must be > 0 when 'pilot_grid' is given.");
        if (!have_bw && !have_freq_new)
            throw std::invalid_argument("Must provide 'bandwidth' (with 'pilot_grid' or 'carriers') or 'freq_new' to define the frequency grid.");
    }

    // Build pilot grid for single-freq mode
    arma::vec pilot_grid_arma;
    if (!is_multifreq)
    {
        if (have_freq_new)
        {
            // Derive bandwidth and pilot_grid from freq_new
            // When freq is provided (single carrier frequency at which coefficients were generated),
            // use it as the phase reference: pilot_grid[k] = (freq_new[k] - freq) / bandwidth.
            // This ensures the DFT phase offset correctly complements the baked-in delay phase.
            // When freq is not provided, use freq_new.min() as the reference (baseband convention).
            double f_ref = (have_freq && freq_arma.n_elem >= 1) ? freq_arma(0) : freq_new_arma.min();
            double f_min = freq_new_arma.min();
            double f_max = freq_new_arma.max();
            bandwidth = f_max - f_min;
            if (bandwidth <= 0.0 && freq_new_arma.n_elem > 1)
                throw std::invalid_argument("'freq_new' values must span a positive range.");
            if (bandwidth <= 0.0)
                bandwidth = 1.0; // Single carrier: avoid division by zero
            pilot_grid_arma.set_size(freq_new_arma.n_elem);
            for (arma::uword k = 0; k < freq_new_arma.n_elem; ++k)
                pilot_grid_arma(k) = (freq_new_arma(k) - f_ref) / bandwidth;
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

    // Output (n_rx, n_tx, n_carrier, n_snap_out); each snapshot slice is a Cube on the output memory, written by C++
    const arma::uword n_carrier = is_multifreq ? freq_new_arma.n_elem : pilot_grid_arma.n_elem;
    const arma::uword n_rx = (arma::uword)c_item[0].shape[0], n_tx = (arma::uword)c_item[0].shape[1];
    std::vector<arma::Cube<std::complex<double>>> hmat_vec;
    auto output = qd_python_init_output<std::complex<double>>(n_rx, n_tx, n_carrier, snap_arma.n_elem, &hmat_vec);

    // Split real / imaginary cubes of the selected snapshots, frames along the input frequencies; the others stay
    // empty and are not read by C++. A 3D delay applies to all frequencies; of a 4D delay, frame 0 is used (only
    // delay[0] is read). The checks here are those the 4D output needs; C++ checks the rest. Parallel from 2 MiB of
    // input (measured: 1.5 MiB is copied faster on one thread, 2 MiB 2.7x and 4 MiB 6x faster on 32 threads).
    const size_t n_frame = is_multifreq ? (size_t)freq_arma.n_elem : 1;
    std::vector<std::vector<arma::Cube<double>>> cre(n_snap_total), cim(n_snap_total), dl(n_snap_total);
    size_t n_bytes = 0;
    for (size_t k = 0; k < n_src; ++k)
        n_bytes += c_item[k].shape[8] * c_item[k].n_frames;

    qd_batch_loop(n_src, n_src > 1 && n_bytes >= (2ULL << 20), "Snapshot", [&](size_t k)
                  {
        const size_t j = src[k];
        const auto &item = c_item[k].shape;
        if (c_item[k].ndim != input_ndim)
            throw std::invalid_argument("All coefficient arrays must have the same number of dimensions.");
        if (item[0] != n_rx || item[1] != n_tx)
            throw std::invalid_argument("Coefficient dimensions do not match the first selected snapshot.");
        if (c_item[k].n_frames != n_frame)
            throw std::invalid_argument("4th dimension of the coefficients does not match the length of 'freq'.");
        if (dl_item[k].n_frames != 1 && dl_item[k].n_frames != n_frame)
            throw std::invalid_argument("4th dimension of 'delay' does not match the length of 'freq'.");

        cre[j].resize(n_frame), cim[j].resize(n_frame), dl[j].resize(n_frame);
        for (size_t f = 0; f < n_frame; ++f)
            qd_python_copy2arma(c_item[k].data + f * c_item[k].stride_frames, item, cre[j][f], cim[j][f]);
        qd_python_copy2arma(dl_item[k].data, dl_item[k].shape, dl[j][0]); }, &src);

    if (is_multifreq)
        quadriga_lib::baseband_freq_response_multi_vec<double>(cre, cim, dl, freq_arma, freq_new_arma, nullptr, nullptr,
                                                               &hmat_vec, remove_delay_phase_b, &snap_arma, use_kernel);
    else
    {
        std::vector<arma::Cube<double>> cre_3(n_snap_total), cim_3(n_snap_total), dl_3(n_snap_total);
        for (arma::uword j : src)
            cre_3[j] = std::move(cre[j][0]), cim_3[j] = std::move(cim[j][0]), dl_3[j] = std::move(dl[j][0]);
        quadriga_lib::baseband_freq_response_vec<double>(cre_3, cim_3, dl_3, pilot_grid_arma, bandwidth,
                                                         nullptr, nullptr, &hmat_vec, &snap_arma, use_kernel);
    }
    return output;
}
