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

- Computes the frequency-domain channel matrix at the carrier positions `pilot_grid` by a DFT over the time-domain
  path coefficients and delays
- Snapshots are passed as Python lists; each list item is one snapshot, processed in parallel; the number of paths
  may differ between snapshots, `n_rx` and `n_tx` may not
- Multi-frequency coefficients (e.g. from [[channel.multifreq]]) and absolute carrier frequencies are handled by
  [[baseband_freq_response_multi]]; the two functions take different arguments, so they are two functions,
  while [[interpolate]] serves both of its cases with one argument set
- `delay` may be broadcast: a `(1, 1, n_path)` item applies the same delays to every RX/TX pair
- The phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to one period
  before sin / cos are evaluated in single precision; coefficients and sums are single precision, double inputs are
  narrowed to float, results widened back

## Usage:
```
hmat = quadriga_lib.channel.baseband_freq_response( coeff, delay, pilot_grid, bandwidth, *, i_snap, use_kernel )
```

## Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item `(n_rx, n_tx, n_path)` or `(1, 1, n_path)`
- **`pilot_grid`** — Sub-carrier positions relative to the bandwidth, `0.0` = carrier, `1.0` = carrier + `bandwidth`;
  1D array of length `n_carrier`; `np.linspace(0, 1, n)` gives `n` evenly spaced carriers
- **`bandwidth`** — Baseband bandwidth in Hz
- **`i_snap`** — 0-based snapshot indices to process; 1D array of length `n_out`; default: `None` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (raises if unavailable); default: 0

## Outputs:
- **`hmat`** — Complex frequency-domain channel matrix; shape `(n_rx, n_tx, n_carrier, n_out)`

## See also:
- [[baseband_freq_response_multi]] (multi-frequency coefficients, absolute carrier frequencies)
- [[channel.spherical]] (single-frequency channel generator)
MD!*/

// Reads the snapshot lists 'coeff' and 'delay' of both baseband wrappers. Only the selected snapshots are read and
// converted, the others stay empty and are not read by C++. 'n_freq' = 0: 3D coefficient items (one cube each);
// else 4D items with n_freq frames, or 3D items for n_freq = 1. Of a 4D delay item only frame 0 is used (C++ reads
// only delay[0]). The checks here are those the 4D output needs; C++ checks the rest, also an empty list. Parallel
// from 2 MiB of input (measured: 1.5 MiB is copied faster on one thread, 2 MiB 2.7x and 4 MiB 6x faster on 32 threads).
static void qd_python_baseband_read(const py::handle &coeff, const py::handle &delay, const py::handle &i_snap,
                                    size_t n_freq, arma::uvec &snap,
                                    std::vector<std::vector<arma::Cube<double>>> &cre,
                                    std::vector<std::vector<arma::Cube<double>>> &cim,
                                    std::vector<std::vector<arma::Cube<double>>> &dl,
                                    arma::uword &n_rx, arma::uword &n_tx)
{
    if (!py::isinstance<py::list>(coeff))
        throw std::invalid_argument("'coeff' must be a list.");
    if (!py::isinstance<py::list>(delay))
        throw std::invalid_argument("'delay' must be a list.");
    const py::list coeff_list = py::reinterpret_borrow<py::list>(coeff), delay_list = py::reinterpret_borrow<py::list>(delay);

    const size_t n_snap_total = coeff_list.size();
    if (delay_list.size() != n_snap_total)
        throw std::invalid_argument("Length of 'delay' list does not match coefficient list length.");

    snap = qd_python_numpy2arma_Col<arma::uword>(i_snap, true, false, "i_snap");
    if (snap.is_empty() && n_snap_total != 0)
        snap = arma::regspace<arma::uvec>(0, (arma::uword)n_snap_total - 1);
    for (arma::uword j : snap)
        if (j >= (arma::uword)n_snap_total)
            throw std::invalid_argument("Snapshot index " + std::to_string(j) + " exceeds list length (" + std::to_string(n_snap_total) + ").");
    const arma::uvec src = arma::unique(snap);
    const size_t n_src = src.n_elem;

    // List items of the selected snapshots, read with the GIL
    std::vector<qd_python_item<std::complex<double>>> c_item(n_src);
    std::vector<qd_python_item<double>> dl_item(n_src);
    for (size_t k = 0; k < n_src; ++k)
        c_item[k] = qd_python_get_item<std::complex<double>>(coeff_list[src[k]]), dl_item[k] = qd_python_get_item<double>(delay_list[src[k]]);

    const int ndim = (n_freq == 0) ? 3 : 4;
    const size_t n_frame = (n_freq == 0) ? 1 : n_freq;
    n_rx = (n_src == 0) ? 0 : (arma::uword)c_item[0].shape[0], n_tx = (n_src == 0) ? 0 : (arma::uword)c_item[0].shape[1];
    cre.assign(n_snap_total, {}), cim.assign(n_snap_total, {}), dl.assign(n_snap_total, {});
    size_t n_bytes = 0;
    for (size_t k = 0; k < n_src; ++k)
        n_bytes += c_item[k].shape[8] * c_item[k].n_frames;

    qd_batch_loop(n_src, n_src > 1 && n_bytes >= (2ULL << 20), "Snapshot", [&](size_t k)
                  {
        const size_t j = src[k];
        const auto &item = c_item[k].shape;
        if (c_item[k].ndim != ndim && !(ndim == 4 && n_frame == 1 && c_item[k].ndim == 3))
            throw std::invalid_argument("Coefficient arrays must be " + std::to_string(ndim) + "D, got " + std::to_string(c_item[k].ndim) + "D.");
        if (item[0] != n_rx || item[1] != n_tx)
            throw std::invalid_argument("Coefficient dimensions do not match the first selected snapshot.");
        if (c_item[k].n_frames != n_frame)
            throw std::invalid_argument("4th dimension of the coefficients does not match the length of 'freq'.");
        if (dl_item[k].ndim > ndim)
            throw std::invalid_argument("Delay arrays must have at most " + std::to_string(ndim) + " dimensions, got " + std::to_string(dl_item[k].ndim) + "D.");
        if (dl_item[k].n_frames != 1 && dl_item[k].n_frames != n_frame)
            throw std::invalid_argument("4th dimension of 'delay' does not match the length of 'freq'.");

        cre[j].resize(n_frame), cim[j].resize(n_frame), dl[j].resize(n_frame);
        for (size_t f = 0; f < n_frame; ++f)
            qd_python_copy2arma(c_item[k].data + f * c_item[k].stride_frames, item, cre[j][f], cim[j][f]);
        qd_python_copy2arma(dl_item[k].data, dl_item[k].shape, dl[j][0]); }, &src);
}

py::array_t<std::complex<double>> baseband_freq_response(const py::handle &coeff,
                                                         const py::handle &delay,
                                                         const py::handle &pilot_grid,
                                                         double bandwidth,
                                                         const py::handle &i_snap,
                                                         int use_kernel)
{
    const arma::vec pilot_grid_arma = qd_python_numpy2arma_Col<double>(pilot_grid, true, false, "pilot_grid");

    arma::uvec snap;
    std::vector<std::vector<arma::Cube<double>>> cre, cim, dl;
    arma::uword n_rx = 0, n_tx = 0;
    qd_python_baseband_read(coeff, delay, i_snap, 0, snap, cre, cim, dl, n_rx, n_tx);

    // Output (n_rx, n_tx, n_carrier, n_out); each snapshot slice is a Cube on the output memory, written by C++
    std::vector<arma::Cube<std::complex<double>>> hmat_vec;
    auto output = qd_python_init_output<std::complex<double>>(n_rx, n_tx, pilot_grid_arma.n_elem, snap.n_elem, &hmat_vec);

    std::vector<arma::Cube<double>> cre_3(cre.size()), cim_3(cre.size()), dl_3(cre.size());
    for (size_t j = 0; j < cre.size(); ++j)
        if (!cre[j].empty())
            cre_3[j] = std::move(cre[j][0]), cim_3[j] = std::move(cim[j][0]), dl_3[j] = std::move(dl[j][0]);
    quadriga_lib::baseband_freq_response_vec<double>(cre_3, cim_3, dl_3, pilot_grid_arma, bandwidth,
                                                     nullptr, nullptr, &hmat_vec, snap, use_kernel);
    return output;
}

/*!MD
# baseband_freq_response_multi
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

- Interpolates the complex channel coefficients from the input frequencies `freq` to the output carrier
  frequencies `freq_new` using SLERP (magnitude and unwrapped phase each linear), then applies the delay-induced
  phase `exp(-j·2π·freq_new·delay)` per output carrier in double precision
- Snapshots are passed as Python lists; each list item is one snapshot, processed in parallel; the number of paths
  may differ between snapshots, `n_rx` and `n_tx` may not
- Single-frequency coefficients with absolute output carriers: items `(n_rx, n_tx, n_path)`, `freq` the frequency
  the coefficients were generated at; with `remove_delay_phase = True` (the default) and coefficients that carry the
  delay phase at `freq`, as the channel generators return them, this gives the result of [[baseband_freq_response]]
  with `pilot_grid · bandwidth = freq_new - freq`
- `delay` is frequency-independent: items `(n_rx, n_tx, n_path)` or `(1, 1, n_path)`; of a 4D item (as returned by
  [[channel.multifreq]]) only the first frequency is used
- With `remove_delay_phase`, the channels from [[channel.multifreq]] / [[channel.spherical]] must be generated with
  `use_absolute_delays = True`: their coefficients carry the phase of the absolute path length also when `delay` is
  relative, and with relative delays the phase of the TX-RX distance is interpolated across `freq`, which gives
  wrong phases (magnitudes stay right)
- Output frequencies outside the range of `freq` use constant extrapolation from the nearest endpoint

## Usage:
```
hmat = quadriga_lib.channel.baseband_freq_response_multi( coeff, delay, freq, freq_new, *, remove_delay_phase, \
    i_snap, use_kernel )
```

## Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path, n_freq)`; for
  `n_freq = 1` also `(n_rx, n_tx, n_path)`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item `(n_rx, n_tx, n_path)` or `(1, 1, n_path)`,
  with an optional 4th dimension of length `n_freq` (only the first frequency is used)
- **`freq`** — Input sample frequencies in Hz, sorted ascending; 1D array of length `n_freq`
- **`freq_new`** — Output carrier frequencies in Hz; 1D array of length `n_carrier`
- **`remove_delay_phase`** — Remove the delay phase `exp(-j·2π·freq·delay)` from the coefficients before the
  interpolation and apply it again at `freq_new`; `True` for the output of the channel generators, `False` for pure
  envelope coefficients; default: `True`
- **`i_snap`** — 0-based snapshot indices to process; 1D array of length `n_out`; default: `None` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (raises if unavailable); default: 0

## Outputs:
- **`hmat`** — Complex frequency-domain channel matrix; shape `(n_rx, n_tx, n_carrier, n_out)`

## See also:
- [[baseband_freq_response]] (single-frequency coefficients, relative carrier positions)
- [[channel.multifreq]] (multi-frequency channel generator)
MD!*/

py::array_t<std::complex<double>> baseband_freq_response_multi(const py::handle &coeff,
                                                               const py::handle &delay,
                                                               const py::handle &freq,
                                                               const py::handle &freq_new,
                                                               const py::handle &remove_delay_phase,
                                                               const py::handle &i_snap,
                                                               int use_kernel)
{
    const bool remove_delay_phase_b = qd_python_flag(remove_delay_phase, true, "remove_delay_phase");
    const arma::vec freq_arma = qd_python_numpy2arma_Col<double>(freq, true, false, "freq");
    const arma::vec freq_new_arma = qd_python_numpy2arma_Col<double>(freq_new, true, false, "freq_new");

    arma::uvec snap;
    std::vector<std::vector<arma::Cube<double>>> cre, cim, dl;
    arma::uword n_rx = 0, n_tx = 0;
    if (freq_arma.n_elem == 0)
        throw std::invalid_argument("Input 'freq' must have at least one element.");
    qd_python_baseband_read(coeff, delay, i_snap, freq_arma.n_elem, snap, cre, cim, dl, n_rx, n_tx);

    // Output (n_rx, n_tx, n_carrier, n_out); each snapshot slice is a Cube on the output memory, written by C++
    std::vector<arma::Cube<std::complex<double>>> hmat_vec;
    auto output = qd_python_init_output<std::complex<double>>(n_rx, n_tx, freq_new_arma.n_elem, snap.n_elem, &hmat_vec);
    quadriga_lib::baseband_freq_response_multi_vec<double>(cre, cim, dl, freq_arma, freq_new_arma, nullptr, nullptr,
                                                           &hmat_vec, remove_delay_phase_b, snap, use_kernel);
    return output;
}
