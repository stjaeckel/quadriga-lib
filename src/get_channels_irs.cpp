// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <utility>

#include "quadriga_channel.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"
#include "qd_helpers_channel.hpp"

/*!SECTION
Channel generators
SECTION!*/

/*!MD
# get_channels_irs
Calculate MIMO channel coefficients and delays for paths over an intelligent reflecting surface

- Paths of one link over an intelligent reflecting surface (IRS) with `n_irs` elements: segment 1 from the Tx to the
  IRS and segment 2 from the IRS to the Rx, each computed by [[get_channels_spherical]] with the IRS elements as Rx and
  Tx elements; each output path combines one path of each segment
- Each IRS element $k$ reflects once with its weight $w_k$, column `i_irs` of the IRS port weights (a codebook entry):
  per path pair, $\mathbf{H} = \mathbf{B} \, \mathrm{diag}(\mathbf{w}) \, \mathbf{A}$ with the element channels
  $\mathbf{A}$ of segment 1 `[n_irs, n_tx]` and $\mathbf{B}$ of segment 2 `[n_rx, n_irs]`
- The delay of a pair is the sum of its segment delays, each the $|w_k|^2$-weighted mean over the IRS elements; the
  departure angles come from segment 1, the arrival angles from segment 2, as $|w_k|^2$-weighted mean directions
- Kept are the path pairs whose strongest coefficient has a power above `threshold_dB`, or those set in `active_path`;
  the pair of segment-1 path $i_1$ and segment-2 path $i_2$ has the index $i_1 n_{\mathrm{path},2} + i_2$ in
  `active_path` and `active_path_out`, and the output keeps this order

## Declaration:
```
void quadriga_lib::get_channels_irs(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const quadriga_lib::arrayant<dtype> &irs_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const arma::Mat<dtype> &irs_pos,
    const arma::Mat<dtype> &irs_orientation,
    const arma::Mat<dtype> &fbs_pos_1,
    const arma::Mat<dtype> &lbs_pos_1,
    const arma::Col<dtype> &path_gain_1,
    const arma::Col<dtype> &path_length_1,
    const arma::Mat<dtype> &polmat_1,
    const arma::Mat<dtype> &fbs_pos_2,
    const arma::Mat<dtype> &lbs_pos_2,
    const arma::Col<dtype> &path_gain_2,
    const arma::Col<dtype> &path_length_2,
    const arma::Mat<dtype> &polmat_2,
    arma::Cube<dtype> &coeff_re,
    arma::Cube<dtype> &coeff_im,
    arma::Cube<dtype> &delay,
    arma::uword i_irs = 0,
    double threshold_dB = -140.0,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    arma::Cube<dtype> *aod = nullptr,
    arma::Cube<dtype> *eod = nullptr,
    arma::Cube<dtype> *aoa = nullptr,
    arma::Cube<dtype> *eoa = nullptr,
    const quadriga_lib::arrayant<dtype> &irs_array_2 = quadriga_lib::arrayant<dtype>(),
    const std::vector<bool> &active_path = std::vector<bool>(),
    std::vector<bool> *active_path_out = nullptr,
    int use_kernel = 0);
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, valid ([[arrayant]]); `n_tx` and `n_rx` ports
- **`irs_array`** — IRS array with `n_irs` elements, valid, with port weights `[n_irs, n_codebook]` that hold the
  reflection weights; its patterns serve both segments unless `irs_array_2` is given
- **`tx_pos`**, **`rx_pos`**, **`irs_pos`** — Tx, Rx and IRS position in m, finite; `[3, 1]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`**, **`irs_orientation`** — Orientations, Euler angles (bank, tilt, heading)
  in rad, finite; `[3, 1]` or `[1, 3]`
- **`fbs_pos_1`**, **`lbs_pos_1`** — First- and last-bounce scatterer positions of the segment-1 paths in m;
  `[3, n_path_1]`
- **`path_gain_1`** — Path gains of segment 1, linear power; `[n_path_1]`
- **`path_length_1`** — Path lengths of segment 1 in m between the Tx and IRS phase centres, as in
  [[get_channels_spherical]]; `[n_path_1]`
- **`polmat_1`** — Jones matrices of segment 1, the IRS receiving ([Glossary](cpp_api.html#glossary), "Polarization");
  `[8, n_path_1]`
- **`fbs_pos_2`**, **`lbs_pos_2`**, **`path_gain_2`**, **`path_length_2`**, **`polmat_2`** — The same for segment 2
  from the IRS to the Rx, the IRS transmitting; `n_path_2` paths
- **`i_irs`** *(optional)* — Codebook entry, 0-based column of the IRS port weights, below `n_codebook`
- **`threshold_dB`** *(optional)* — Minimum power in dB of the strongest coefficient of a kept path pair; not used with
  `active_path`
- **`center_freq`** *(optional)* — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0, as in
  [[get_channels_spherical]]
- **`use_absolute_delays`** *(optional)* — Keep the delay of the direct Tx-Rx distance in all delays; `false`: delays
  relative to it (Notes)
- **`irs_array_2`** *(optional)* — Patterns of the IRS for segment 2 (asymmetric IRS), valid, `n_irs` elements; its
  port weights and element positions are not used; empty: `irs_array`
- **`active_path`** *(optional)* — Path pairs to keep, replacing `threshold_dB`; empty: `threshold_dB`;
  `[n_path_1 * n_path_2]`
- **`use_kernel`** *(optional)* — Kernel of both segments, as in [[get_channels_spherical]]

## Outputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_rx, n_tx, n_path_irs]` with the
  number of kept path pairs `n_path_irs`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path_irs]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Azimuth and elevation of departure and arrival in rad, GCS;
  `[n_rx, n_tx, n_path_irs]`
- **`active_path_out`** *(optional)* — Kept path pairs, for `active_path` of a later call or [[combine_irs_coord]];
  `[n_path_1 * n_path_2]`

## Notes:
- **IRS port weights**: empty port weights throw here, unlike in the other functions, where they mean identity; the
  IRS elements are not combined into ports
- **Segments**: path lengths, LOS paths and the per-element geometry as in [[get_channels_spherical]]; a segment
  without paths (all its path inputs empty) gives the outputs `[n_rx, n_tx, 0]`
- **Relative delays**: the delay of the direct Tx-Rx distance (not via the IRS) is subtracted; delays can be slightly
  negative when the IRS is close to the Tx-Rx line

## See also:
- [[get_channels_spherical]] (the channel of each segment)
- [[combine_irs_coord]] (interaction coordinates of the kept path pairs)
MD!*/

template <typename dtype>
void quadriga_lib::get_channels_irs(const arrayant<dtype> &tx_array, const arrayant<dtype> &rx_array, const arrayant<dtype> &irs_array,
                                    const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &tx_orientation,
                                    const arma::Mat<dtype> &rx_pos, const arma::Mat<dtype> &rx_orientation,
                                    const arma::Mat<dtype> &irs_pos, const arma::Mat<dtype> &irs_orientation,
                                    const arma::Mat<dtype> &fbs_pos_1, const arma::Mat<dtype> &lbs_pos_1,
                                    const arma::Col<dtype> &path_gain_1, const arma::Col<dtype> &path_length_1, const arma::Mat<dtype> &polmat_1,
                                    const arma::Mat<dtype> &fbs_pos_2, const arma::Mat<dtype> &lbs_pos_2,
                                    const arma::Col<dtype> &path_gain_2, const arma::Col<dtype> &path_length_2, const arma::Mat<dtype> &polmat_2,
                                    arma::Cube<dtype> &coeff_re, arma::Cube<dtype> &coeff_im, arma::Cube<dtype> &delay,
                                    arma::uword i_irs, double threshold_dB_in, double center_freq_in, bool use_absolute_delays,
                                    arma::Cube<dtype> *aod, arma::Cube<dtype> *eod, arma::Cube<dtype> *aoa, arma::Cube<dtype> *eoa,
                                    const arrayant<dtype> &irs_array_2, const std::vector<bool> &active_path,
                                    std::vector<bool> *active_path_out, int use_kernel)
{
    if (!std::isfinite(center_freq_in) || center_freq_in < 0.0 || !std::isfinite((dtype)center_freq_in))
        throw std::invalid_argument("Input 'center_freq' must be finite and >= 0.");
    const dtype threshold_dB = dtype(threshold_dB_in);
    const dtype center_freq = dtype(center_freq_in);
    // Positions and orientations, checked here so that an error names the argument (the segment calls of
    // get_channels_spherical would report 'irs_pos' as 'rx_pos' or 'tx_pos')
    const dtype *p_tp = qd_xyz(tx_pos, "tx_pos"), *p_rp = qd_xyz(rx_pos, "rx_pos");
    qd_xyz(tx_orientation, "tx_orientation"), qd_xyz(rx_orientation, "rx_orientation");
    qd_xyz(irs_pos, "irs_pos"), qd_xyz(irs_orientation, "irs_orientation");

    qd_check_alias_list({&coeff_re, &coeff_im, &delay, aod, eod, aoa, eoa},
                        {"coeff_re", "coeff_im", "delay", "aod", "eod", "aoa", "eoa"},
                        &fbs_pos_1, &lbs_pos_1, &path_gain_1, &path_length_1, &polmat_1,
                        &fbs_pos_2, &lbs_pos_2, &path_gain_2, &path_length_2, &polmat_2);

    std::string error_message = qd_arrayant_is_valid_quick(irs_array);
    if (error_message.length() != 0)
    {
        error_message = "Input 'irs_array' is not valid: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }

    if (irs_array.port_weights_re.n_cols == 0ULL)
        throw std::invalid_argument("Input 'irs_array' must have port weights holding the IRS weights (phase shifts) of the elements.");

    if (i_irs >= irs_array.port_weights_re.n_cols)
        throw std::invalid_argument("Input 'i_irs' is out of range; the IRS port weights have " + std::to_string(irs_array.port_weights_re.n_cols) + " columns.");

    bool calc_angles = aod != nullptr || eod != nullptr || aoa != nullptr || eoa != nullptr;

    // Optional second IRS array for the RX-facing side
    const arma::uword n_elements_irs = irs_array.e_theta_re.n_slices;
    if (irs_array_2.n_elements() != 0)
    {
        error_message = qd_arrayant_is_valid_quick(irs_array_2);
        if (error_message.length() != 0)
        {
            error_message = "Input 'irs_array_2' is not valid: " + error_message;
            throw std::invalid_argument(error_message.c_str());
        }

        if (irs_array_2.e_theta_re.n_slices != n_elements_irs)
            throw std::invalid_argument("Input 'irs_array_2' must have the same number of elements as 'irs_array'.");
        // Its port weights and element_pos are not used: the weights and element positions come from irs_array
    }

    // IRS arrays without port weights (one port per element), as read-only views of the pattern data of a valid array.
    // Each segment gets its own object, since assigning to a view copies into the viewed data. An empty element_pos
    // has no data to view and is replaced by zeros.
    auto irs_view = [&](arrayant<dtype> &dst, const arrayant<dtype> &src)
    {
        const arma::uword n_az = src.e_theta_re.n_cols, n_el = src.e_theta_re.n_rows;
        auto cube = [&](const arma::Cube<dtype> &c)
        { return arma::Cube<dtype>(const_cast<dtype *>(c.memptr()), n_el, n_az, n_elements_irs, false, true); };
        dst.e_theta_re = cube(src.e_theta_re), dst.e_theta_im = cube(src.e_theta_im);
        dst.e_phi_re = cube(src.e_phi_re), dst.e_phi_im = cube(src.e_phi_im);
        dst.azimuth_grid = arma::Col<dtype>(const_cast<dtype *>(src.azimuth_grid.memptr()), n_az, false, true);
        dst.elevation_grid = arma::Col<dtype>(const_cast<dtype *>(src.elevation_grid.memptr()), n_el, false, true);

        if (irs_array.element_pos.is_empty())
            dst.element_pos.zeros(3ULL, n_elements_irs);
        else
            dst.element_pos = arma::Mat<dtype>(const_cast<dtype *>(irs_array.element_pos.memptr()), 3ULL, n_elements_irs, false, true);

        if (center_freq > dtype(1.0e-5))
            dst.center_freq = center_freq;
    };
    quadriga_lib::arrayant<dtype> irs_1, irs_2;
    irs_view(irs_1, irs_array);
    if (irs_array_2.n_elements() != 0)
        irs_view(irs_2, irs_array_2);
    const quadriga_lib::arrayant<dtype> &irs = irs_1, &irs_rx = irs_array_2.n_elements() != 0 ? irs_2 : irs_1;

    // Per-element channels of the first segment from the TX to the IRS, [n_irs, n_tx, n_path_1]
    arma::Cube<dtype> coeff_re_1, coeff_im_1, delay_1, aod_1, eod_1;
    quadriga_lib::get_channels_spherical<dtype>(tx_array, irs, tx_pos, tx_orientation, irs_pos, irs_orientation, fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1,
                                                coeff_re_1, coeff_im_1, delay_1, center_freq, true, false,
                                                calc_angles ? &aod_1 : nullptr, calc_angles ? &eod_1 : nullptr, nullptr, nullptr, use_kernel);

    // Per-element channels of the second segment from the IRS to the RX, [n_rx, n_irs, n_path_2]
    arma::Cube<dtype> coeff_re_2, coeff_im_2, delay_2, aoa_2, eoa_2;
    quadriga_lib::get_channels_spherical<dtype>(irs_rx, rx_array, irs_pos, irs_orientation, rx_pos, rx_orientation, fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2,
                                                coeff_re_2, coeff_im_2, delay_2, center_freq, true, false,
                                                nullptr, nullptr, calc_angles ? &aoa_2 : nullptr, calc_angles ? &eoa_2 : nullptr, use_kernel);

    // Calculate the LOS delay between TX and RX
    dtype los_delay = dtype(0.0);
    if (!use_absolute_delays)
        los_delay = qd_calc_length(p_tp[0], p_tp[1], p_tp[2], p_rp[0], p_rp[1], p_rp[2]) * dtype(1.0 / qd_speed_of_light);

    const arma::uword n_path_1 = coeff_re_1.n_slices;
    const arma::uword n_path_2 = coeff_re_2.n_slices;
    const arma::uword n_path = n_path_1 * n_path_2;
    const arma::uword n_tx = coeff_re_1.n_cols;
    const arma::uword n_rx = coeff_re_2.n_rows;
    const arma::uword n_irs = n_elements_irs;

    // IRS weights w of the codebook entry and their powers |w|^2, normalized to sum 1 (uniform if all are 0)
    const dtype *w_re = irs_array.port_weights_re.colptr(i_irs);
    const dtype *w_im = irs_array.port_weights_im.is_empty() ? nullptr : irs_array.port_weights_im.colptr(i_irs);
    std::vector<dtype> w_pwr(n_irs);
    qd_power_mat(n_irs, 1, w_pwr.data(), true, w_re, w_im);

    // Delays and angles of each segment path: |w|^2-weighted means over the IRS elements (angles from the mean of the
    // unit direction vectors); the elements are the rows (segment 1) or the columns (segment 2)
    auto reduce = [&](const arma::Cube<dtype> &dl, const arma::Cube<dtype> &az, const arma::Cube<dtype> &el, bool irs_rows,
                      arma::Mat<dtype> &dl_out, arma::Mat<dtype> &az_out, arma::Mat<dtype> &el_out)
    {
        const arma::uword n_ant = irs_rows ? dl.n_cols : dl.n_rows, n_p = dl.n_slices;
        dl_out.zeros(n_ant, n_p);
        if (calc_angles)
            az_out.set_size(n_ant, n_p), el_out.set_size(n_ant, n_p);
        for (arma::uword p = 0; p < n_p; ++p)
            for (arma::uword a = 0; a < n_ant; ++a)
            {
                double d = 0.0, x = 0.0, y = 0.0, z = 0.0;
                for (arma::uword k = 0; k < n_irs; ++k)
                {
                    const arma::uword i = irs_rows ? k + a * n_irs : a + k * n_ant;
                    const double wk = (double)w_pwr[k];
                    d += wk * (double)dl.slice_memptr(p)[i];
                    if (calc_angles)
                    {
                        const double ca = std::cos((double)az.slice_memptr(p)[i]), sa = std::sin((double)az.slice_memptr(p)[i]);
                        const double ce = std::cos((double)el.slice_memptr(p)[i]), se = std::sin((double)el.slice_memptr(p)[i]);
                        x += wk * ce * ca, y += wk * ce * sa, z += wk * se;
                    }
                }
                dl_out(a, p) = (dtype)d;
                if (calc_angles)
                    az_out(a, p) = (dtype)std::atan2(y, x), el_out(a, p) = (dtype)std::atan2(z, std::sqrt(x * x + y * y));
            }
    };
    arma::Mat<dtype> dl_1, az_1, el_1, dl_2, az_2, el_2;
    reduce(delay_1, aod_1, eod_1, true, dl_1, az_1, el_1);
    reduce(delay_2, aoa_2, eoa_2, false, dl_2, az_2, el_2);

    // Apply the weights to the first segment once: c1(k, tx) := w(k) * c1(k, tx)
    for (arma::uword i = 0; i < coeff_re_1.n_elem; ++i)
    {
        const arma::uword k = i % n_irs;
        const dtype a = coeff_re_1[i], b = coeff_im_1[i], c = w_re[k], d = w_im ? w_im[k] : dtype(0.0);
        coeff_re_1[i] = a * c - b * d, coeff_im_1[i] = a * d + b * c;
    }

    // Combine the two segments: each element reflects once, h(rx, tx) = sum_k c2(rx, k) * w(k) * c1(k, tx)
    arma::Cube<dtype> coeff_re_local(n_rx, n_tx, n_path, arma::fill::none);
    arma::Cube<dtype> coeff_im_local(n_rx, n_tx, n_path, arma::fill::none);
    arma::Cube<dtype> delay_local(n_rx, n_tx, n_path, arma::fill::none);

    arma::Cube<dtype> aod_local, eod_local, aoa_local, eoa_local;
    if (calc_angles)
    {
        aod_local.set_size(n_rx, n_tx, n_path);
        eod_local.set_size(n_rx, n_tx, n_path);
        aoa_local.set_size(n_rx, n_tx, n_path);
        eoa_local.set_size(n_rx, n_tx, n_path);
    }

    dtype threshold = (dtype)std::pow<double>(10.0, 0.1 * (double)threshold_dB);
    std::vector<bool> keep_path(n_path, false);

    bool use_threshold = active_path.empty();
    if (!use_threshold && active_path.size() != n_path)
        throw std::invalid_argument("Input 'active_path' must have n_path_1 * n_path_2 elements.");

    arma::uword i_path = 0ULL, j_path = 0ULL;
    for (arma::uword i_path_1 = 0ULL; i_path_1 < n_path_1; ++i_path_1)
    {
        const dtype *c1r = coeff_re_1.slice_memptr(i_path_1), *c1i = coeff_im_1.slice_memptr(i_path_1);
        const dtype *d1 = dl_1.colptr(i_path_1);

        for (arma::uword i_path_2 = 0ULL; i_path_2 < n_path_2; ++i_path_2)
        {
            const dtype *c2r = coeff_re_2.slice_memptr(i_path_2), *c2i = coeff_im_2.slice_memptr(i_path_2);
            const dtype *d2 = dl_2.colptr(i_path_2);

            dtype *cr = coeff_re_local.slice_memptr(i_path);
            dtype *ci = coeff_im_local.slice_memptr(i_path);
            dtype *dl = delay_local.slice_memptr(i_path);

            dtype pow = dtype(0.0);
            for (arma::uword i_tx = 0ULL; i_tx < n_tx; ++i_tx)
            {
                const dtype *ar = &c1r[i_tx * n_irs], *ai = &c1i[i_tx * n_irs];
                for (arma::uword i_rx = 0ULL; i_rx < n_rx; ++i_rx)
                {
                    dtype re = dtype(0.0), im = dtype(0.0);
                    for (arma::uword k = 0ULL; k < n_irs; ++k)
                    {
                        const dtype c = c2r[i_rx + k * n_rx], d = c2i[i_rx + k * n_rx];
                        re += ar[k] * c - ai[k] * d;
                        im += ar[k] * d + ai[k] * c;
                    }

                    arma::uword i = i_tx * n_rx + i_rx;
                    cr[i] = re, ci[i] = im;
                    dl[i] = d1[i_tx] + d2[i_rx] - los_delay;

                    re = re * re + im * im;
                    pow = (re > pow) ? re : pow;

                    if (calc_angles)
                    {
                        aod_local.slice_memptr(i_path)[i] = az_1(i_tx, i_path_1);
                        eod_local.slice_memptr(i_path)[i] = el_1(i_tx, i_path_1);
                        aoa_local.slice_memptr(i_path)[i] = az_2(i_rx, i_path_2);
                        eoa_local.slice_memptr(i_path)[i] = el_2(i_rx, i_path_2);
                    }
                }
            }

            bool keep_this_path = use_threshold ? pow > threshold : active_path.at(j_path);
            i_path = keep_this_path ? i_path + 1ULL : i_path;
            keep_path[j_path] = keep_this_path;
            ++j_path;
        }
    }

    const arma::uword n_bytes = n_rx * n_tx * i_path * sizeof(dtype);
    const std::pair<arma::Cube<dtype> *, const arma::Cube<dtype> *> outputs[] = {
        {&coeff_re, &coeff_re_local}, {&coeff_im, &coeff_im_local}, {&delay, &delay_local},
        {aod, &aod_local}, {eod, &eod_local}, {aoa, &aoa_local}, {eoa, &eoa_local}};
    for (const auto &o : outputs)
        if (o.first != nullptr)
        {
            o.first->set_size(n_rx, n_tx, i_path);
            if (n_bytes) // No memcpy for empty outputs (null data pointer)
                std::memcpy(o.first->memptr(), o.second->memptr(), n_bytes);
        }

    if (active_path_out != nullptr)
        *active_path_out = std::move(keep_path);
}

template void quadriga_lib::get_channels_irs(const arrayant<float> &tx_array, const arrayant<float> &rx_array, const arrayant<float> &irs_array,
                                             const arma::Mat<float> &tx_pos, const arma::Mat<float> &tx_orientation,
                                             const arma::Mat<float> &rx_pos, const arma::Mat<float> &rx_orientation,
                                             const arma::Mat<float> &irs_pos, const arma::Mat<float> &irs_orientation,
                                             const arma::Mat<float> &fbs_pos_1, const arma::Mat<float> &lbs_pos_1,
                                             const arma::Col<float> &path_gain_1, const arma::Col<float> &path_length_1, const arma::Mat<float> &polmat_1,
                                             const arma::Mat<float> &fbs_pos_2, const arma::Mat<float> &lbs_pos_2,
                                             const arma::Col<float> &path_gain_2, const arma::Col<float> &path_length_2, const arma::Mat<float> &polmat_2,
                                             arma::Cube<float> &coeff_re, arma::Cube<float> &coeff_im, arma::Cube<float> &delay,
                                             arma::uword i_irs, double threshold_dB, double center_freq, bool use_absolute_delays,
                                             arma::Cube<float> *aod, arma::Cube<float> *eod, arma::Cube<float> *aoa, arma::Cube<float> *eoa,
                                             const arrayant<float> &irs_array_2, const std::vector<bool> &active_path, std::vector<bool> *active_path_out, int use_kernel);

template void quadriga_lib::get_channels_irs(const arrayant<double> &tx_array, const arrayant<double> &rx_array, const arrayant<double> &irs_array,
                                             const arma::Mat<double> &tx_pos, const arma::Mat<double> &tx_orientation,
                                             const arma::Mat<double> &rx_pos, const arma::Mat<double> &rx_orientation,
                                             const arma::Mat<double> &irs_pos, const arma::Mat<double> &irs_orientation,
                                             const arma::Mat<double> &fbs_pos_1, const arma::Mat<double> &lbs_pos_1,
                                             const arma::Col<double> &path_gain_1, const arma::Col<double> &path_length_1, const arma::Mat<double> &polmat_1,
                                             const arma::Mat<double> &fbs_pos_2, const arma::Mat<double> &lbs_pos_2,
                                             const arma::Col<double> &path_gain_2, const arma::Col<double> &path_length_2, const arma::Mat<double> &polmat_2,
                                             arma::Cube<double> &coeff_re, arma::Cube<double> &coeff_im, arma::Cube<double> &delay,
                                             arma::uword i_irs, double threshold_dB, double center_freq, bool use_absolute_delays,
                                             arma::Cube<double> *aod, arma::Cube<double> *eod, arma::Cube<double> *aoa, arma::Cube<double> *eoa,
                                             const arrayant<double> &irs_array_2, const std::vector<bool> &active_path, std::vector<bool> *active_path_out, int use_kernel);
