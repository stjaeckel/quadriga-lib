// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <cstdint>

#include "quadriga_lib.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"
#include "qd_helpers_channel.hpp"

// Bracketing indices i_lo, i_hi and weight w (0 = i_lo, 1 = i_hi) of freq in center_freqs (any order);
// exact is set when freq matches an entry within 1e-6 (relative) or lies outside the range; then only i_lo is used
template <typename dtype>
static inline void qd_find_freq_bracket(const arma::Col<dtype> &center_freqs, dtype freq,
                                     arma::uword &i_lo, arma::uword &i_hi, dtype &w, bool &exact)
{
    arma::uword n = center_freqs.n_elem;
    exact = false;
    i_lo = 0;
    i_hi = 0;
    w = dtype(0.0);

    if (n == 1)
    {
        exact = true;
        return;
    }

    // Order-independent bracket search: i_lo = largest entry <= freq,
    // i_hi = smallest entry >= freq. Track true min/max for extrapolation.
    bool have_lo = false, have_hi = false;
    arma::uword idx_min = 0, idx_max = 0;
    for (arma::uword i = 0; i < n; ++i)
    {
        dtype fi = center_freqs[i];
        if (fi < center_freqs[idx_min])
            idx_min = i;
        if (fi > center_freqs[idx_max])
            idx_max = i;
        if (fi <= freq && (!have_lo || fi > center_freqs[i_lo]))
            i_lo = i, have_lo = true;
        if (fi >= freq && (!have_hi || fi < center_freqs[i_hi]))
            i_hi = i, have_hi = true;
    }

    // Extrapolation below the smallest / above the largest entry: clamp
    if (!have_lo)
    {
        exact = true;
        i_lo = idx_min;
        return;
    }
    if (!have_hi)
    {
        exact = true;
        i_lo = idx_max;
        return;
    }

    // Exact match (bracket collapses to one entry, or coincides within tolerance)
    if (i_lo == i_hi ||
        std::abs((double)center_freqs[i_lo] - (double)freq) < 1.0e-6 * std::abs((double)center_freqs[i_lo]))
    {
        exact = true;
        return;
    }

    w = (freq - center_freqs[i_lo]) / (center_freqs[i_hi] - center_freqs[i_lo]);
}

// SLERP per element between two complex matrices given as real and imaginary parts; an empty imaginary part is zero
template <typename dtype>
static inline void qd_interpolate_complex_matrix(const arma::Mat<dtype> &re_lo, const arma::Mat<dtype> &im_lo,
                                              const arma::Mat<dtype> &re_hi, const arma::Mat<dtype> &im_hi,
                                              dtype w, arma::Mat<dtype> &re_out, arma::Mat<dtype> &im_out)
{
    arma::uword nR = re_lo.n_rows, nC = re_lo.n_cols, nE = nR * nC;
    re_out.set_size(nR, nC);
    im_out.set_size(nR, nC);

    const dtype *rlo = re_lo.memptr();
    const dtype *rhi = re_hi.memptr();
    bool has_im_lo = (im_lo.n_elem == nE);
    bool has_im_hi = (im_hi.n_elem == nE);
    const dtype *ilo = has_im_lo ? im_lo.memptr() : nullptr;
    const dtype *ihi = has_im_hi ? im_hi.memptr() : nullptr;
    dtype *ro = re_out.memptr();
    dtype *io = im_out.memptr();
    constexpr dtype zero = dtype(0.0);

    for (arma::uword k = 0; k < nE; ++k)
    {
        dtype ai = ilo ? ilo[k] : zero;
        dtype bi = ihi ? ihi[k] : zero;
        qd_slerp_complex(rlo[k], ai, rhi[k], bi, w, ro[k], io[k]);
    }
}

/*!SECTION
Channel generators
SECTION!*/

/*!MD
# get_channels_multifreq
Calculate MIMO channel coefficients and delays for spherical wave propagation at several frequencies

- As [[get_channels_spherical]], at the output frequencies `freq_new`, with multi-frequency arrays (frequency-dependent
  patterns and port weights), path gains and Jones matrices; for radio waves and, with `propagation_speed` = 343 m/s and
  scalar `polmat`, for sound
- The geometry (element-pair path lengths, angles, LOS path) is computed once; at each output frequency, the patterns
  and port weights of the arrays and `path_gain` and `polmat` are interpolated between their two neighbouring entries
- The coefficients follow the formula of [[get_channels_spherical]] with $f_c$ = `freq_new[k]` and $c$ =
  `propagation_speed`; the output has one cube per output frequency, a multi-frequency channel

## Declaration:
```
void quadriga_lib::get_channels_multifreq(
    const std::vector<arrayant<dtype>> &tx_array,
    const std::vector<arrayant<dtype>> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const arma::Mat<dtype> &fbs_pos,
    const arma::Mat<dtype> &lbs_pos,
    const arma::Mat<dtype> &path_gain,
    const arma::Col<dtype> &path_length,
    const arma::Cube<dtype> &polmat,
    const arma::Col<dtype> &freq,
    const arma::Col<dtype> &freq_new,
    std::vector<arma::Cube<dtype>> &coeff_re,
    std::vector<arma::Cube<dtype>> &coeff_im,
    std::vector<arma::Cube<dtype>> &delay,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    double propagation_speed = 299792458.0,
    std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr,
    int use_kernel = 0);
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx multi-frequency array, valid ([[is_valid]]), at least one entry, in any
  frequency order; `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; `[3, 1]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; `[3, 1]` or `[1, 3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `[3, n_path]`
- **`path_gain`** — Path gains at the frequencies `freq`, linear power; `[n_path, n_freq]`
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres, as in [[get_channels_spherical]];
  `[n_path]`
- **`polmat`** — Jones matrices at the frequencies `freq` ([Glossary](cpp_api.html#glossary), "Polarization"):
  `[8, n_path, n_freq]`, or `[2, n_path, n_freq]` for a scalar coefficient (real and imaginary part, sound)
- **`freq`** — Sample frequencies of `path_gain` and `polmat` in Hz, in any order; `[n_freq]`, at least one
- **`freq_new`** — Output frequencies in Hz; `[n_freq_new]`, at least one
- **`use_absolute_delays`** *(optional)* — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative
  to it
- **`add_fake_los_path`** *(optional)* — Put the LOS path into an added slot 0, with zero power if there is none
- **`propagation_speed`** *(optional)* — Wave speed in m/s, > 0
- **`use_kernel`** *(optional)* — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC,
  2 = AVX2 (single-precision arithmetic, also for `double`; throws if unavailable)

## Outputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_freq_new]` cubes
  `[n_rx, n_tx, n_out]`, `n_out = n_path`, or `n_path + 1` with `add_fake_los_path`
- **`delay`** — Delays in s; sizes as `coeff_re`
- **`coeff`** *(optional)* — Complex coefficients, replacing `coeff_re` / `coeff_im` (then left empty)

## Notes:
- **Interpolation**: `path_gain` linear; patterns, port weights and `polmat` per complex value, the amplitude linear and
  the phase along the shorter arc (linear for nearly opposite phases); constant outside the range of each grid
- **Scalar `polmat`**: only the $F_\theta$ patterns are used,
  $a_l = \sqrt{g_l} \, F_{\theta,r} \, m_l \, F_{\theta,t} \, e^{-j 2\pi f d / c}$ with the complex scalar $m_l$
- **Paths**: NLOS path length, LOS path, `add_fake_los_path` and 0 paths as in [[get_channels_spherical]]
- **Relative delays**: negative delays as in [[get_channels_spherical]], with $c$ = `propagation_speed`; for sound, this
  affects practically every array, so use absolute delays for [[quantize_delays]]
- **Delay phase**: the coefficients carry the phase of the absolute path length also with relative delays;
  [[freq_response_mf]] with `remove_delay_phase = true` therefore needs `use_absolute_delays = true`

## See also:
- [[get_channels_spherical]] (one frequency)
- [[freq_response_mf]] (frequency response of a multi-frequency channel)
- [[join_arrays]] (multi-frequency arrays)
- [[generate_speaker]] (loudspeaker arrays)
MD!*/

template <typename dtype>
void quadriga_lib::get_channels_multifreq(const std::vector<quadriga_lib::arrayant<dtype>> &tx_array,
                                          const std::vector<quadriga_lib::arrayant<dtype>> &rx_array,
                                          const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &tx_orientation,
                                          const arma::Mat<dtype> &rx_pos, const arma::Mat<dtype> &rx_orientation,
                                          const arma::Mat<dtype> &fbs_pos, const arma::Mat<dtype> &lbs_pos,
                                          const arma::Mat<dtype> &path_gain, const arma::Col<dtype> &path_length,
                                          const arma::Cube<dtype> &polmat,
                                          const arma::Col<dtype> &freq, const arma::Col<dtype> &freq_new,
                                          std::vector<arma::Cube<dtype>> &coeff_re, std::vector<arma::Cube<dtype>> &coeff_im,
                                          std::vector<arma::Cube<dtype>> &delay,
                                          bool use_absolute_delays, bool add_fake_los_path, double propagation_speed_in,
                                          std::vector<arma::Cube<std::complex<dtype>>> *coeff, int use_kernel)
{
    const dtype propagation_speed = dtype(propagation_speed_in);
    constexpr dtype zero = dtype(0.0);
    constexpr dtype one = dtype(1.0);
    const dtype rC = one / propagation_speed;
    const dtype two_pi = dtype(2.0) * dtype(arma::datum::pi);

    if (tx_array.empty())
        throw std::invalid_argument("Input 'tx_array' cannot be empty.");
    if (rx_array.empty())
        throw std::invalid_argument("Input 'rx_array' cannot be empty.");
    const dtype *p_tp = qd_xyz(tx_pos, "tx_pos"), *p_to = qd_xyz(tx_orientation, "tx_orientation");
    const dtype *p_rp = qd_xyz(rx_pos, "rx_pos"), *p_ro = qd_xyz(rx_orientation, "rx_orientation");
    const dtype Tx = p_tp[0], Ty = p_tp[1], Tz = p_tp[2];
    const dtype Rx = p_rp[0], Ry = p_rp[1], Rz = p_rp[2];

    // Kernel of the antenna interpolation, checked before any early return (the pattern size of AVX2 per call)
    const int kernel = qd_interpolate_pattern_kernel(use_kernel, 0);

    // Outputs: no cube may share memory with an input or with any other output cube
    const std::vector<arma::Cube<dtype>> *outs[] = {&coeff_re, &coeff_im, &delay};
    const char *names[] = {"coeff_re", "coeff_im", "delay"};
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            qd_check_alias_outputs(outs[i], names[i], outs[j], names[j]);
    {
        struct range
        {
            std::uintptr_t first, end; // Memory range [first, end) of one cube
            int name;                  // Index into names_all
        };
        const char *names_all[] = {"coeff_re", "coeff_im", "delay", "coeff"};
        std::vector<range> ranges;
        ranges.reserve(coeff_re.size() + coeff_im.size() + delay.size() + (coeff ? coeff->size() : 0));
        auto add = [&](const auto &vec, int name)
        {
            for (const auto &c : vec)
            {
                qd_check_alias(&c, names_all[name], &fbs_pos, &lbs_pos, &path_gain, &path_length, &polmat, &freq, &freq_new);
                if (c.n_elem != 0)
                    ranges.push_back({(std::uintptr_t)c.memptr(), (std::uintptr_t)(c.memptr() + c.n_elem), name});
            }
        };
        add(coeff_re, 0), add(coeff_im, 1), add(delay, 2);
        if (coeff != nullptr)
            add(*coeff, 3);
        std::sort(ranges.begin(), ranges.end(), [](const range &a, const range &b) { return a.first < b.first; });
        for (size_t i = 1, i_max = 0; i < ranges.size(); ++i)
        {
            if (ranges[i].first < ranges[i_max].end)
                throw std::invalid_argument(std::string("Outputs '") + names_all[ranges[i_max].name] + "' and '" +
                                            names_all[ranges[i].name] + "' cannot be the same object or share memory.");
            if (ranges[i].end > ranges[i_max].end)
                i_max = i;
        }
    }

    {
        std::string err = qd_arrayant_vec_check(tx_array);
        if (!err.empty())
            throw std::invalid_argument("Input 'tx_array' is not valid: " + err);
        err = qd_arrayant_vec_check(rx_array);
        if (!err.empty())
            throw std::invalid_argument("Input 'rx_array' is not valid: " + err);
    }

    // 0 paths: all path inputs empty; 'fbs_pos' of size [0, n] with n > 0 gives n paths without data (missing data)
    const bool no_paths = fbs_pos.n_elem == 0 && fbs_pos.n_cols == 0 && lbs_pos.n_elem == 0 && path_gain.n_elem == 0 &&
                          path_length.n_elem == 0 && polmat.n_elem == 0;

    // 0 paths with add_fake_los_path: the output holds only the zero-power LOS path. Its slot (delays from the TX-RX
    // geometry) does not depend on the path data, so a dummy NLOS path (1 m longer than the LOS path, never detected
    // as LOS) fills the input; its own slot is dropped.
    if (add_fake_los_path && no_paths)
    {
        const arma::Mat<dtype> pos_d = {(Tx + Rx) * (dtype)0.5, (Ty + Ry) * (dtype)0.5, (Tz + Rz) * (dtype)0.5};
        const arma::Mat<dtype> fbs_d = pos_d.t();
        const arma::Col<dtype> length_d = {qd_calc_length(Tx, Ty, Tz, Rx, Ry, Rz) + one};
        const arma::uword n_f = freq.n_elem;
        const arma::Mat<dtype> gain_d(1, n_f, arma::fill::ones);
        arma::Cube<dtype> M_d((polmat.n_rows == 2) ? 2 : 8, 1, n_f, arma::fill::zeros);
        for (arma::uword f = 0; f < n_f; ++f)
        {
            M_d(0, 0, f) = one;
            if (M_d.n_rows == 8)
                M_d(6, 0, f) = -one;
        }

        std::vector<arma::Cube<dtype>> cr, ci, dl;
        std::vector<arma::Cube<std::complex<dtype>>> cx;
        quadriga_lib::get_channels_multifreq<dtype>(tx_array, rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                    fbs_d, fbs_d, gain_d, length_d, M_d, freq, freq_new,
                                                    cr, ci, dl, use_absolute_delays, true, propagation_speed,
                                                    coeff ? &cx : nullptr, kernel);

        // Copy slot 0 to the outputs
        auto copy_slot = [](const auto &src, auto &dst)
        {
            if (dst.size() != src.size())
                dst.resize(src.size());
            for (size_t f = 0; f < src.size(); ++f)
            {
                dst[f].set_size(src[f].n_rows, src[f].n_cols, 1ULL);
                std::memcpy(dst[f].memptr(), src[f].memptr(), src[f].n_rows * src[f].n_cols * sizeof(*src[f].memptr()));
            }
        };
        copy_slot(cr, coeff_re), copy_slot(ci, coeff_im), copy_slot(dl, delay);
        if (coeff != nullptr)
            copy_slot(cx, *coeff);
        return;
    }

    // 0 paths without add_fake_los_path: the path inputs are not checked, the outputs are empty [n_rx, n_tx, 0]
    if (!no_paths && (fbs_pos.n_elem == 0 || lbs_pos.n_elem == 0 || path_gain.n_elem == 0 ||
                      path_length.n_elem == 0 || polmat.n_elem == 0))
        throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'polmat' cannot be empty.");

    if (!no_paths && (fbs_pos.n_rows != 3 || lbs_pos.n_rows != 3))
        throw std::invalid_argument("Inputs 'fbs_pos' and 'lbs_pos' must have 3 rows.");

    if (!no_paths && polmat.n_rows != 8 && polmat.n_rows != 2)
        throw std::invalid_argument("Input 'polmat' must have 8 rows (full polarimetric) or 2 rows (scalar pressure).");

    const bool scalar_M = (polmat.n_rows == 2);

    const arma::uword n_path = fbs_pos.n_cols;
    const arma::uword n_freq = freq.n_elem;
    const arma::uword n_freq_new = freq_new.n_elem;
    const arma::uword n_out = add_fake_los_path ? n_path + 1 : n_path;

    if (n_freq == 0)
        throw std::invalid_argument("Input 'freq' cannot be empty.");
    if (n_freq_new == 0)
        throw std::invalid_argument("Input 'freq_new' cannot be empty.");

    if (!no_paths && lbs_pos.n_cols != n_path)
        throw std::invalid_argument("Input 'lbs_pos' must have n_path columns.");
    if (!no_paths && (path_gain.n_rows != n_path || path_gain.n_cols != n_freq))
        throw std::invalid_argument("Input 'path_gain' must have size [n_path, n_freq].");
    if (!no_paths && path_length.n_elem != n_path)
        throw std::invalid_argument("Input 'path_length' must have n_path elements.");
    if (!no_paths && (polmat.n_cols != n_path || polmat.n_slices != n_freq))
        throw std::invalid_argument("Input 'polmat' must have size [8|2, n_path, n_freq].");
    if (propagation_speed <= zero)
        throw std::invalid_argument("Input 'propagation_speed' must be positive.");

    // Antenna dimensions (from first entry, consistent across all entries by validation)
    const arma::uword n_tx = tx_array[0].n_elements();
    const arma::uword n_rx = rx_array[0].n_elements();
    const arma::uword n_links = n_rx * n_tx;
    const arma::uword n_tx_ports = tx_array[0].n_ports();
    const arma::uword n_rx_ports = rx_array[0].n_ports();
    const arma::uword n_ports = n_tx_ports * n_rx_ports;

    // Size the outputs (the vectors only if their length differs, which keeps externally borrowed memory); a requested
    // complex output replaces coeff_re / coeff_im, which are left empty
    auto resize_output = [&](auto &out)
    {
        if (out.size() != n_freq_new)
            out.resize(n_freq_new);
        for (auto &c : out)
            c.set_size(n_rx_ports, n_tx_ports, n_out);
    };
    if (coeff != nullptr)
        coeff_re.clear(), coeff_im.clear(), resize_output(*coeff);
    else
        resize_output(coeff_re), resize_output(coeff_im);
    resize_output(delay);

    if (n_out == 0) // 0 paths without add_fake_los_path: empty outputs
        return;

    // Build center-frequency vectors for TX, RX, and freq grids
    arma::Col<dtype> tx_center_freqs((arma::uword)tx_array.size());
    arma::Col<dtype> rx_center_freqs((arma::uword)rx_array.size());
    for (arma::uword i = 0; i < tx_center_freqs.n_elem; ++i)
        tx_center_freqs[i] = (dtype)tx_array[i].center_freq;
    for (arma::uword i = 0; i < rx_center_freqs.n_elem; ++i)
        rx_center_freqs[i] = (dtype)rx_array[i].center_freq;

    // Phase 1: Geometry computation (frequency-independent, sequential)
    arma::Mat<dtype> AOD(n_links, n_out), EOD(n_links, n_out);
    arma::Mat<dtype> AOA(n_links, n_out), EOA(n_links, n_out);
    arma::Mat<dtype> path_delays_geom(n_links, n_out); // Element-resolved path lengths [m]
    dtype *p_aod = AOD.memptr(), *p_eod = EOD.memptr();
    dtype *p_aoa = AOA.memptr(), *p_eoa = EOA.memptr();
    dtype *p_path_dl = path_delays_geom.memptr();

    arma::uvec i_tx_element, i_rx_element;
    arma::Mat<dtype> tx_element_pos_interp, rx_element_pos_interp;
    dtype dist_rx_tx;
    const arma::uword true_los_path = qd_channels_spherical_geometry(tx_array[0].element_pos, rx_array[0].element_pos, n_tx, n_rx,
                                                                     p_tp, p_to, p_rp, p_ro, fbs_pos.memptr(), lbs_pos.memptr(),
                                                                     path_length.memptr(), n_out, add_fake_los_path,
                                                                     p_aod, p_eod, p_aoa, p_eoa, p_path_dl,
                                                                     i_tx_element, tx_element_pos_interp,
                                                                     i_rx_element, rx_element_pos_interp, dist_rx_tx);
    const arma::Cube<dtype> tx_ori(const_cast<dtype *>(p_to), 3, 1, 1), rx_ori(const_cast<dtype *>(p_ro), 3, 1, 1);

    // Port weights, skipped when it is the identity at all frequencies
    const bool different_output_size = (n_tx != n_tx_ports || n_rx != n_rx_ports);
    bool apply_element_port_weights = different_output_size;
    for (const auto *arr : {&tx_array, &rx_array})
        for (const auto &a : *arr)
            apply_element_port_weights = apply_element_port_weights || !qd_port_weights_is_identity(a.port_weights_re, a.port_weights_im);

    // Port weights of one array at an output frequency, interpolated between its bracket entries (SLERP per complex entry;
    // empty port_weights_re = identity, empty port_weights_im = zero), and its column-normalized power for the delays
    auto interpolate_port_weights = [](const std::vector<quadriga_lib::arrayant<dtype>> &arr, arma::uword lo, arma::uword hi,
                                   dtype w, bool exact, arma::uword n_el, arma::uword n_pt,
                                   arma::Mat<dtype> &cpl_re, arma::Mat<dtype> &cpl_im, arma::Mat<dtype> &cpl_pwr)
    {
        const arma::Mat<dtype> &re_lo = arr[lo].port_weights_re, &im_lo = arr[lo].port_weights_im;
        if (exact)
        {
            if (re_lo.n_elem != 0)
                cpl_re = re_lo;
            else
                cpl_re.eye(n_el, n_pt);
            if (im_lo.n_elem != 0)
                cpl_im = im_lo;
            else
                cpl_im.zeros(n_el, n_pt);
        }
        else
        {
            const arma::Mat<dtype> &re_hi = arr[hi].port_weights_re, &im_hi = arr[hi].port_weights_im;
            arma::Mat<dtype> eye;
            if (re_lo.n_elem == 0 || re_hi.n_elem == 0)
                eye.eye(n_el, n_pt);
            qd_interpolate_complex_matrix(re_lo.n_elem != 0 ? re_lo : eye, im_lo, re_hi.n_elem != 0 ? re_hi : eye, im_hi,
                                          w, cpl_re, cpl_im);
        }
        cpl_pwr.set_size(n_el, n_pt);
        qd_power_mat(n_el, n_pt, cpl_pwr.memptr(), true, cpl_re.memptr(), cpl_im.memptr());
    };

    // Phase 2: Per-frequency processing (parallelized over output frequencies)
    // The interpolation calls inside the parallel region do not throw for the validated arrays; should one, the error
    // is passed on after the region
    bool omp_failed = false;
    std::string omp_error;

    // Pattern of one array at the angles of one path, interpolated between its bracket entries (SLERP per value)
    auto interpolate_pattern = [&](const std::vector<quadriga_lib::arrayant<dtype>> &arr, arma::uword lo, arma::uword hi,
                                   dtype w, bool exact, const arma::Mat<dtype> &az, const arma::Mat<dtype> &el,
                                   const arma::uvec &i_element, const arma::Cube<dtype> &ori, const arma::Mat<dtype> &pos,
                                   arma::Mat<dtype> &V_re, arma::Mat<dtype> &V_im, arma::Mat<dtype> &H_re, arma::Mat<dtype> &H_im)
    {
        if (exact)
        {
            OMP_SAFE_CALL(omp_failed, omp_error,
                          qd_interpolate_pattern_dispatch<dtype>(arr[lo], az, el, i_element, ori, pos,
                                                                  V_re, V_im, H_re, H_im, nullptr, nullptr, nullptr, nullptr, kernel));
            return;
        }
        arma::Mat<dtype> Vr_lo(n_links, 1), Vi_lo(n_links, 1), Hr_lo(n_links, 1), Hi_lo(n_links, 1);
        arma::Mat<dtype> Vr_hi(n_links, 1), Vi_hi(n_links, 1), Hr_hi(n_links, 1), Hi_hi(n_links, 1);
        OMP_SAFE_CALL(omp_failed, omp_error,
                      qd_interpolate_pattern_dispatch<dtype>(arr[lo], az, el, i_element, ori, pos,
                                                              Vr_lo, Vi_lo, Hr_lo, Hi_lo, nullptr, nullptr, nullptr, nullptr, kernel));
        OMP_SAFE_CALL(omp_failed, omp_error,
                      qd_interpolate_pattern_dispatch<dtype>(arr[hi], az, el, i_element, ori, pos,
                                                              Vr_hi, Vi_hi, Hr_hi, Hi_hi, nullptr, nullptr, nullptr, nullptr, kernel));
        for (arma::uword k = 0; k < n_links; ++k)
        {
            qd_slerp_complex(Vr_lo[k], Vi_lo[k], Vr_hi[k], Vi_hi[k], w, V_re[k], V_im[k]);
            qd_slerp_complex(Hr_lo[k], Hi_lo[k], Hr_hi[k], Hi_hi[k], w, H_re[k], H_im[k]);
        }
    };

    const int n_freq_new_i = (int)n_freq_new;
#pragma omp parallel for schedule(static) if (n_freq_new >= 8)
    for (int f_i = 0; f_i < n_freq_new_i; ++f_i)
    {
        const arma::uword f = (arma::uword)f_i;
        const dtype f_new = freq_new[f];
        const dtype wavelength = (f_new > zero) ? propagation_speed / f_new : one;
        const dtype wave_number = (f_new > zero) ? two_pi * f_new / propagation_speed : zero;

        // Find frequency brackets for TX, RX, and freq grids
        arma::uword tx_lo, tx_hi, rx_lo, rx_hi, fin_lo, fin_hi;
        dtype tx_w, rx_w, fin_w;
        bool tx_exact, rx_exact, fin_exact;

        qd_find_freq_bracket(tx_center_freqs, f_new, tx_lo, tx_hi, tx_w, tx_exact);
        qd_find_freq_bracket(rx_center_freqs, f_new, rx_lo, rx_hi, rx_w, rx_exact);
        qd_find_freq_bracket(freq, f_new, fin_lo, fin_hi, fin_w, fin_exact);
        // Interpolate path gain at this frequency (linear interpolation)
        const dtype *p_gain_in = path_gain.memptr(); // Column-major [n_path, n_freq]
        arma::Col<dtype> gain_f(n_path);
        if (fin_exact)
        {
            std::memcpy(gain_f.memptr(), &p_gain_in[fin_lo * n_path], n_path * sizeof(dtype));
        }
        else
        {
            const dtype *col_lo = &p_gain_in[fin_lo * n_path];
            const dtype *col_hi = &p_gain_in[fin_hi * n_path];
            dtype wA = one - fin_w, wB = fin_w;
            dtype *pg = gain_f.memptr();
            for (arma::uword p = 0; p < n_path; ++p)
                pg[p] = wA * col_lo[p] + wB * col_hi[p];
        }

        // Interpolate polmat (Jones matrix) at this frequency (SLERP per complex entry)
        const arma::uword M_rows = polmat.n_rows; // 8 or 2
        arma::Mat<dtype> M_f(M_rows, n_path);
        if (fin_exact)
        {
            std::memcpy(M_f.memptr(), polmat.slice_memptr(fin_lo), M_rows * n_path * sizeof(dtype));
        }
        else
        {
            const dtype *M_lo = polmat.slice_memptr(fin_lo);
            const dtype *M_hi = polmat.slice_memptr(fin_hi);
            dtype *Mf = M_f.memptr();
            arma::uword n_pairs = M_rows / 2; // 4 complex pairs for full, 1 for scalar

            for (arma::uword p = 0; p < n_path; ++p)
            {
                arma::uword base = p * M_rows;
                for (arma::uword k = 0; k < n_pairs; ++k)
                {
                    arma::uword idx = base + 2 * k;
                    qd_slerp_complex(M_lo[idx], M_lo[idx + 1],
                                     M_hi[idx], M_hi[idx + 1],
                                     fin_w, Mf[idx], Mf[idx + 1]);
                }
            }
        }

        // Interpolate port weights at this frequency
        arma::Mat<dtype> tx_cpl_re_f, tx_cpl_im_f, rx_cpl_re_f, rx_cpl_im_f;
        arma::Mat<dtype> tx_port_weights_pwr_f, rx_port_weights_pwr_f;
        if (apply_element_port_weights)
        {
            interpolate_port_weights(tx_array, tx_lo, tx_hi, tx_w, tx_exact, n_tx, n_tx_ports, tx_cpl_re_f, tx_cpl_im_f, tx_port_weights_pwr_f);
            interpolate_port_weights(rx_array, rx_lo, rx_hi, rx_w, rx_exact, n_rx, n_rx_ports, rx_cpl_re_f, rx_cpl_im_f, rx_port_weights_pwr_f);
        }

        // Internal storage for this frequency (if output size differs from element size)
        arma::Cube<dtype> CR, CI, DL;
        if (different_output_size)
        {
            CR.set_size(n_rx, n_tx, n_out);
            CI.set_size(n_rx, n_tx, n_out);
            DL.set_size(n_rx, n_tx, n_out);
        }

        // Split coefficients in port space: the outputs, or scratch for the complex output
        arma::Cube<dtype> PR, PI;
        if (coeff != nullptr)
            PR.set_size(n_rx_ports, n_tx_ports, n_out), PI.set_size(n_rx_ports, n_tx_ports, n_out);
        arma::Cube<dtype> &out_re = coeff ? PR : coeff_re[f], &out_im = coeff ? PI : coeff_im[f];

        dtype *p_coeff_re = different_output_size ? CR.memptr() : out_re.memptr();
        dtype *p_coeff_im = different_output_size ? CI.memptr() : out_im.memptr();
        dtype *p_delays_f = different_output_size ? DL.memptr() : delay[f].memptr();

        // Copy geometric path delays (same for all frequencies, in meters)
        std::memcpy(p_delays_f, p_path_dl, n_links * n_out * sizeof(dtype));

        // MIMO coefficients, one path after the other
        for (arma::uword j = 0; j < n_out; ++j)
        {
            const arma::uword i = add_fake_los_path ? (j == 0 ? 0 : j - 1) : j;
            const arma::uword o_slice = j * n_links;

            // Interpolate the TX antenna at the departure angles and the RX antenna at the arrival angles
            const arma::Mat<dtype> AOD_j(&p_aod[o_slice], n_links, 1, false, true);
            const arma::Mat<dtype> EOD_j(&p_eod[o_slice], n_links, 1, false, true);
            const arma::Mat<dtype> AOA_j(&p_aoa[o_slice], n_links, 1, false, true);
            const arma::Mat<dtype> EOA_j(&p_eoa[o_slice], n_links, 1, false, true);

            arma::Mat<dtype> Vt_re(n_links, 1), Vt_im(n_links, 1), Ht_re(n_links, 1), Ht_im(n_links, 1);
            arma::Mat<dtype> Vr_re(n_links, 1), Vr_im(n_links, 1), Hr_re(n_links, 1), Hr_im(n_links, 1);
            interpolate_pattern(tx_array, tx_lo, tx_hi, tx_w, tx_exact, AOD_j, EOD_j, i_tx_element, tx_ori, tx_element_pos_interp,
                                Vt_re, Vt_im, Ht_re, Ht_im);
            interpolate_pattern(rx_array, rx_lo, rx_hi, rx_w, rx_exact, AOA_j, EOA_j, i_rx_element, rx_ori, rx_element_pos_interp,
                                Vr_re, Vr_im, Hr_re, Hr_im);

            // Compute MIMO coefficients for this path
            const dtype *pM = M_f.colptr(i);
            const dtype *pVrr = Vr_re.memptr(), *pVri = Vr_im.memptr();
            const dtype *pHrr = Hr_re.memptr(), *pHri = Hr_im.memptr();
            const dtype *pVtr = Vt_re.memptr(), *pVti = Vt_im.memptr();
            const dtype *pHtr = Ht_re.memptr(), *pHti = Ht_im.memptr();
            const dtype path_amplitude = (add_fake_los_path && j == 0) ? zero : std::sqrt(gain_f[i]);

            if (scalar_M) // Scalar pressure: only the (V rx, V tx) component, polmat = [Re, Im]
                for (arma::uword R = 0; R < n_links; ++R)
                {
                    const dtype re = pVrr[R] * pM[0] * pVtr[R] - pVri[R] * pM[1] * pVtr[R] - pVrr[R] * pM[1] * pVti[R] - pVri[R] * pM[0] * pVti[R];
                    const dtype im = pVrr[R] * pM[1] * pVtr[R] + pVri[R] * pM[0] * pVtr[R] + pVrr[R] * pM[0] * pVti[R] - pVri[R] * pM[1] * pVti[R];
                    const dtype phase = wave_number * std::fmod(p_delays_f[o_slice + R], wavelength);
                    const dtype cp = std::cos(phase), sp = std::sin(phase);
                    p_coeff_re[o_slice + R] = (re * cp + im * sp) * path_amplitude;
                    p_coeff_im[o_slice + R] = (-re * sp + im * cp) * path_amplitude;
                }
            else
                qd_coeff_combine(pVrr, pVri, pHrr, pHri, pVtr, pVti, pHtr, pHti,
                                 pM, &p_delays_f[o_slice], wave_number, wavelength, path_amplitude,
                                 &p_coeff_re[o_slice], &p_coeff_im[o_slice], n_links);

            // Convert path lengths [m] to propagation delays [s]
            for (arma::uword R = 0; R < n_links; ++R)
            {
                const dtype dl = use_absolute_delays ? p_delays_f[o_slice + R] : p_delays_f[o_slice + R] - dist_rx_tx;
                p_delays_f[o_slice + R] = dl * rC;
            }

        }

        // Apply port weights (using interpolated port weights); without different_output_size the
        // data is in the outputs and the port weights run in place
        if (apply_element_port_weights)
        {
            std::vector<dtype> temp;
            for (arma::uword j = 0; j < n_out; ++j)
            {
                const arma::uword o_slice = j * n_links;
                qd_port_weights_apply_slot(rx_cpl_re_f.memptr(), rx_cpl_im_f.memptr(), tx_cpl_re_f.memptr(), tx_cpl_im_f.memptr(),
                                       rx_port_weights_pwr_f.memptr(), tx_port_weights_pwr_f.memptr(), n_rx, n_rx_ports, n_tx, n_tx_ports,
                                       &p_coeff_re[o_slice], &p_coeff_im[o_slice], &p_delays_f[o_slice],
                                       different_output_size ? out_re.slice_memptr(j) : &p_coeff_re[o_slice],
                                       different_output_size ? out_im.slice_memptr(j) : &p_coeff_im[o_slice],
                                       different_output_size ? delay[f].slice_memptr(j) : &p_delays_f[o_slice], temp);
            }
        }

        // Move true LOS path to slot 0 if fake LOS was added
        if (add_fake_los_path && true_los_path != 0)
        {
            dtype *ptrR = out_re.slice_memptr(true_los_path);
            dtype *ptrI = out_im.slice_memptr(true_los_path);
            dtype *dstR = out_re.slice_memptr(0);
            dtype *dstI = out_im.slice_memptr(0);

            std::memcpy(dstR, ptrR, n_ports * sizeof(dtype));
            std::memcpy(dstI, ptrI, n_ports * sizeof(dtype));

            dtype *ptrD = delay[f].slice_memptr(true_los_path);
            dtype *dstD = delay[f].slice_memptr(0);
            std::memcpy(dstD, ptrD, n_ports * sizeof(dtype));

            for (arma::uword k = 0; k < n_ports; ++k)
                ptrR[k] = zero, ptrI[k] = zero;
        }

        // Interleave into the complex output
        if (coeff != nullptr)
        {
            dtype *dst = (dtype *)(*coeff)[f].memptr();
            const dtype *src_re = PR.memptr(), *src_im = PI.memptr();
            for (arma::uword k = 0; k < PR.n_elem; ++k)
                dst[2 * k] = src_re[k], dst[2 * k + 1] = src_im[k];
        }

    }

    if (omp_failed)
        throw std::runtime_error(omp_error);
}

template void quadriga_lib::get_channels_multifreq(
    const std::vector<quadriga_lib::arrayant<float>> &, const std::vector<quadriga_lib::arrayant<float>> &,
    const arma::Mat<float> &, const arma::Mat<float> &, const arma::Mat<float> &, const arma::Mat<float> &,
    const arma::Mat<float> &, const arma::Mat<float> &, const arma::Mat<float> &,
    const arma::Col<float> &, const arma::Cube<float> &, const arma::Col<float> &, const arma::Col<float> &,
    std::vector<arma::Cube<float>> &, std::vector<arma::Cube<float>> &, std::vector<arma::Cube<float>> &,
    bool, bool, double, std::vector<arma::Cube<std::complex<float>>> *, int);

template void quadriga_lib::get_channels_multifreq(
    const std::vector<quadriga_lib::arrayant<double>> &, const std::vector<quadriga_lib::arrayant<double>> &,
    const arma::Mat<double> &, const arma::Mat<double> &, const arma::Mat<double> &, const arma::Mat<double> &,
    const arma::Mat<double> &, const arma::Mat<double> &, const arma::Mat<double> &,
    const arma::Col<double> &, const arma::Cube<double> &, const arma::Col<double> &, const arma::Col<double> &,
    std::vector<arma::Cube<double>> &, std::vector<arma::Cube<double>> &, std::vector<arma::Cube<double>> &,
    bool, bool, double, std::vector<arma::Cube<std::complex<double>>> *, int);
