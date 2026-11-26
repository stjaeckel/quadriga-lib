// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"
#include "qd_helpers_shared.hpp" // qd_batch_loop, qd_snapshot_list

// One numpy array (e.g. a list item) as typed array with element strides, its data pointer and shape (see
// qd_python_get_shape; a 4th dimension gives n_frames frames, stride_frames elements apart). Reading it needs the GIL;
// copying from 'data' with qd_python_copy2arma does not, so the copies of many items can run in parallel.
namespace // Internal linkage, like the pybind11 types it holds
{
    template <typename dtype>
    struct qd_python_item
    {
        py::array_t<dtype> array; // Keeps the data alive
        const dtype *data = nullptr;
        std::array<size_t, 9> shape{};
        size_t n_frames = 1, stride_frames = 0;
        int ndim = 0;
    };
}

template <typename dtype>
static inline qd_python_item<dtype> qd_python_get_item(const py::handle &obj)
{
    qd_python_item<dtype> item;
    const py::array array = py::cast<py::array>(obj);
    if (py::detail::npy_format_descriptor<dtype>::dtype().is(array.dtype()))
        item.array = qd_python_element_strided(array.cast<py::array_t<dtype>>());
    else
        item.array = qd_python_element_strided(py::array_t<dtype>(array)); // Converting copy
    item.ndim = (int)item.array.ndim();
    item.shape = qd_python_get_shape(item.array, false, &item.n_frames, &item.stride_frames);
    item.data = item.array.data();
    return item;
}

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# freq_response
Frequency response of a MIMO channel at the given subcarriers

- The frequency response of one snapshot at the subcarriers $k$: $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$ with the
  offset $f_k$ = `bandwidth` · `pilot_grid[k]` from the carrier
- The snapshots are list items, computed in parallel; the number of paths may differ between them
- For a multi-frequency channel and absolute carrier frequencies, use [[freq_response_mf]]
- Precision: the phase is reduced to one period in double precision; sin / cos, coefficients and sums are single
  precision

## Usage:
```
hmat = quadriga_lib.channel.freq_response( coeff, delay, pilot_grid, bandwidth, *, i_snap, use_kernel )
```

## Inputs:
- **`coeff`** — Complex coefficients; list of `n_snap` arrays `(n_rx, n_tx, n_path)`
- **`delay`** — Delays in s; list of `n_snap` arrays `(n_rx, n_tx, n_path)` or `(1, 1, n_path)`
- **`pilot_grid`** — Subcarrier positions relative to the bandwidth, finite: 0 = carrier, 1 = carrier + `bandwidth`;
  `(n_carrier,)`; `np.linspace(0, 1, n)` gives `n` evenly spaced carriers
- **`bandwidth`** — Bandwidth in Hz, finite and >= 0
- **`i_snap`** — 0-based indices of the snapshots to compute; `(n_out,)`; default: `None` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (raises if
  unavailable); default: 0

## Output:
- **`hmat`** — Complex frequency response; complex128; `(n_rx, n_tx, n_carrier, n_out)`; the selected snapshots must
  have equal `n_rx` and `n_tx`

## See also:
- [[freq_response_mf]] (multi-frequency channel, absolute carrier frequencies)
- [[channel.spherical]] (coefficients and delays)
- [[quantize_delays]] (delays on a tap grid, for time-domain processing)
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
        throw std::invalid_argument("Input 'coeff' must be a list.");
    if (!py::isinstance<py::list>(delay))
        throw std::invalid_argument("Input 'delay' must be a list.");
    const py::list coeff_list = py::reinterpret_borrow<py::list>(coeff), delay_list = py::reinterpret_borrow<py::list>(delay);

    const size_t n_snap_total = coeff_list.size();
    if (delay_list.size() != n_snap_total)
        throw std::invalid_argument("Inputs 'coeff' and 'delay' must have the same length.");

    snap = qd_snapshot_list(qd_python_numpy2arma_Col<arma::uword>(i_snap, true, false, "i_snap"), n_snap_total);
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
            throw std::invalid_argument("Input 'coeff' must hold " + std::to_string(ndim) + "D arrays, got " + std::to_string(c_item[k].ndim) + "D.");
        if (item[0] != n_rx || item[1] != n_tx)
            throw std::invalid_argument("Input 'coeff' must have the n_rx and n_tx of the first selected snapshot.");
        if (c_item[k].n_frames != n_frame)
            throw std::invalid_argument("Input 'coeff' must have a 4th dimension of the length of 'freq'.");
        if (dl_item[k].ndim > ndim)
            throw std::invalid_argument("Input 'delay' must hold arrays of at most " + std::to_string(ndim) + " dimensions, got " + std::to_string(dl_item[k].ndim) + "D.");
        if (dl_item[k].n_frames != 1 && dl_item[k].n_frames != n_frame)
            throw std::invalid_argument("Input 'delay' must have a 4th dimension of 1 or the length of 'freq'.");

        cre[j].resize(n_frame), cim[j].resize(n_frame), dl[j].resize(n_frame);
        for (size_t f = 0; f < n_frame; ++f)
            qd_python_copy2arma(c_item[k].data + f * c_item[k].stride_frames, item, cre[j][f], cim[j][f]);
        qd_python_copy2arma(dl_item[k].data, dl_item[k].shape, dl[j][0]); }, &src);
}

py::array_t<std::complex<double>> freq_response(const py::handle &coeff,
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
    quadriga_lib::freq_response<double>(cre_3, cim_3, dl_3, pilot_grid_arma, bandwidth,
                                        nullptr, nullptr, &hmat_vec, snap, use_kernel);
    return output;
}

/*!MD
# freq_response_mf
Frequency response of a MIMO channel from coefficients sampled at several frequencies

- The frequency response of one snapshot of a multi-frequency channel at the absolute carrier frequencies
  $f_k$ = `freq_new[k]`: the coefficients of each path are interpolated from the input frequencies `freq` to $f_k$
  and the delay phase is applied there, $H_k = \sum_l \tilde a_l(f_k) \, e^{-j 2\pi f_k \tau_l}$
- Interpolation between the two neighbouring input frequencies: amplitude linear, phase along the shorter arc
  (SLERP); constant outside the range of `freq`
- `remove_delay_phase = True` interpolates the envelope $a_l(f_i) \, e^{j 2\pi f_i \tau_l}$ instead of $a_l(f_i)$, so
  that the delay phase the channel generators include is not interpolated
- The snapshots are list items, computed in parallel; the number of paths may differ between them

## Usage:
```
hmat = quadriga_lib.channel.freq_response_mf( coeff, delay, freq, freq_new, *, remove_delay_phase, \
    i_snap, use_kernel )
```

## Inputs:
- **`coeff`** — Complex coefficients; list of `n_snap` arrays `(n_rx, n_tx, n_path, n_freq)`, for `n_freq = 1` also
  `(n_rx, n_tx, n_path)`
- **`delay`** — Delays in s; list of `n_snap` arrays `(n_rx, n_tx, n_path)` or `(1, 1, n_path)` (a 4th dimension of
  length `n_freq` is allowed, only the first frequency is used)
- **`freq`** — Input frequencies in Hz, finite, strictly ascending; `(n_freq,)`
- **`freq_new`** — Output carrier frequencies in Hz, finite; `(n_carrier,)`
- **`remove_delay_phase`** — Remove the delay phase from the coefficients before the interpolation: `True` for the
  coefficients of [[channel.multifreq]] and [[channel.spherical]], `False` for coefficients without delay phase;
  default: `True`
- **`i_snap`** — 0-based indices of the snapshots to compute; `(n_out,)`; default: `None` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (raises if unavailable); default: 0

## Output:
- **`hmat`** — Complex frequency response; complex128; `(n_rx, n_tx, n_carrier, n_out)`; the selected snapshots must
  have equal `n_rx` and `n_tx`

## Notes:
- With `remove_delay_phase = True`, generate the channels with `use_absolute_delays = True`: their coefficients carry
  the phase of the absolute path length also when `delay` holds relative delays. With relative delays the phase of
  the Tx-Rx distance stays in the coefficients and is interpolated across `freq`, which gives wrong phases (the
  magnitudes stay right) unless the spacing of `freq` is far below $c$ / (Tx-Rx distance)
- One input frequency: the coefficients apply at all carriers; with `remove_delay_phase = True` the result is that of
  [[freq_response]] with `bandwidth` · `pilot_grid` = `freq_new - freq`
- Precision: the delay phase is reduced to one period in double precision (and removed in double precision with
  `remove_delay_phase`); sin / cos of the output phase and the interpolation are single
  precision, the sums double precision

## See also:
- [[freq_response]] (single-frequency coefficients, carriers relative to the bandwidth)
- [[channel.multifreq]] (multi-frequency channel)
MD!*/

py::array_t<std::complex<double>> freq_response_mf(const py::handle &coeff,
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
    quadriga_lib::freq_response_mf<double>(cre, cim, dl, freq_arma, freq_new_arma, nullptr, nullptr,
                                           &hmat_vec, remove_delay_phase_b, snap, use_kernel);
    return output;
}
