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

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_channels_irs
Calculate MIMO channel coefficients for IRS-assisted communication

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- The IRS coupling matrix holds the IRS weights and must be given: an empty coupling of `irs_array` throws, unlike
  in the other functions, where it stands for identity
- Polarization coupling is applied via the 8-row transfer matrices `polmat_1`, `polmat_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx)
- Each segment is computed by [[get_channels_spherical]]: NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths that exceed `threshold_dB`; the pair
  of segment-1 path `i_1` and segment-2 path `i_2` has index `i_1 * n_path_2 + i_2` in `active_path` and
  `active_path_out`, and the output keeps this order
- A segment without paths (all its path inputs empty) gives empty outputs `[n_rx, n_tx, 0]`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- Optional `irs_array_2` provides a separate IRS antenna pattern for the RX-facing side (asymmetric IRS)
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

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
    dtype threshold_dB = -140.0,
    dtype center_freq = 0.0,
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
- **`tx_array`** — Transmit antenna array with `n_tx` elements; see [[arrayant]]
- **`rx_array`** — Receive antenna array with `n_rx` elements; see [[arrayant]]
- **`irs_array`** — IRS antenna array (TX-facing side) with `n_irs` elements; see [[arrayant]]
- **`tx_pos`** — Transmitter position in Cartesian coordinates in [m]; `[3, 1]` or `[1, 3]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading) in [rad]; `[3, 1]` or `[1, 3]`
- **`rx_pos`** — Receiver position in Cartesian coordinates in [m]; `[3, 1]` or `[1, 3]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading) in [rad]; `[3, 1]` or `[1, 3]`
- **`irs_pos`** — IRS position in Cartesian coordinates in [m]; `[3, 1]` or `[1, 3]`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading) in [rad]; `[3, 1]` or `[1, 3]`
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `[n_path_1]`
- **`path_length_1`** — Total path lengths from TX to IRS phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `[n_path_1]`
- **`polmat_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `[n_path_2]`
- **`polmat_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `[8, n_path_2]`
- **`i_irs`** *(optional)* — IRS codebook port index
- **`threshold_dB`** *(optional)* — Gain threshold in dB; path combinations below this are discarded
- **`center_freq`** *(optional)* — Center frequency; 0 gives the coefficients at f = 0 (see above)
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component
- **`irs_array_2`** *(optional)* — Second IRS antenna array for the RX-facing side; enables asymmetric IRS patterns; empty (no elements): `irs_array` on both sides; see [[arrayant]]
- **`active_path`** *(optional)* — Bitmask selecting active path pairs; overrides `threshold_dB`; `[n_path_1 * n_path_2]`
- **`use_kernel`** *(optional)* — Kernel of both segments, passed to [[get_channels_spherical]]: 0 = auto, 1 = GENERIC,
  2 = AVX2 (single precision, also for `double`; throws if unavailable); other values throw; auto selects GENERIC in
  this version

## Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_irs]`
- **`aod`** *(optional)* — Azimuth of departure; `[n_rx, n_tx, n_path_irs]`
- **`eod`** *(optional)* — Elevation of departure; `[n_rx, n_tx, n_path_irs]`
- **`aoa`** *(optional)* — Azimuth of arrival; `[n_rx, n_tx, n_path_irs]`
- **`eoa`** *(optional)* — Elevation of arrival; `[n_rx, n_tx, n_path_irs]`
- **`active_path_out`** *(optional)* — Boolean mask indicating which path combinations are in the output (can be passed
  as `active_path` to a later call or to [[combine_irs_coord]]); `[n_path_1 * n_path_2]`

## See also:
- [[combine_irs_coord]] (coordinate setup for IRS geometry)
- [[get_channels_spherical]] (single-segment spherical-wave channel)
- [[get_channels_planar]] (single-segment planar-wave channel)
- [[arrayant]] (antenna array class)
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
                                    arma::uword i_irs, dtype threshold_dB, dtype center_freq, bool use_absolute_delays,
                                    arma::Cube<dtype> *aod, arma::Cube<dtype> *eod, arma::Cube<dtype> *aoa, arma::Cube<dtype> *eoa,
                                    const arrayant<dtype> &irs_array_2, const std::vector<bool> &active_path,
                                    std::vector<bool> *active_path_out, int use_kernel)
{
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
        error_message = "IRS array: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }

    if (irs_array.coupling_re.n_cols == 0ULL)
        throw std::invalid_argument("IRS must have a Coupling matrix containing the IRS weights (phase shifts) for each element.");

    if (i_irs >= irs_array.coupling_re.n_cols)
        throw std::invalid_argument("IRS codebook index out of bound.");

    bool calc_angles = aod != nullptr || eod != nullptr || aoa != nullptr || eoa != nullptr;

    // Optional second IRS array for the RX-facing side
    const arma::uword n_elements_irs = irs_array.e_theta_re.n_slices;
    if (irs_array_2.n_elements() != 0)
    {
        error_message = qd_arrayant_is_valid_quick(irs_array_2);
        if (error_message.length() != 0)
        {
            error_message = "IRS array 2: " + error_message;
            throw std::invalid_argument(error_message.c_str());
        }

        if (irs_array_2.e_theta_re.n_slices != n_elements_irs)
            throw std::invalid_argument("IRS 2 must have the same number of elements as IRS 1");

        if (irs_array_2.coupling_re.n_cols != irs_array.coupling_re.n_cols)
            throw std::invalid_argument("Input 'irs_array_2' must have a coupling matrix (the IRS weights) of the same size as 'irs_array'.");
    }

    // IRS arrays with only the desired weight vector: read-only views of the pattern data of a valid array. Each
    // segment gets its own object, since assigning to a view copies into the viewed data. An empty element_pos or
    // coupling_im (of any shape) has no data to view and is replaced by zeros.
    auto irs_view = [&](arrayant<dtype> &dst, const arrayant<dtype> &src)
    {
        const arma::uword n_az = src.e_theta_re.n_cols, n_el = src.e_theta_re.n_rows;
        auto cube = [&](const arma::Cube<dtype> &c)
        { return arma::Cube<dtype>(const_cast<dtype *>(c.memptr()), n_el, n_az, n_elements_irs, false, true); };
        dst.e_theta_re = cube(src.e_theta_re), dst.e_theta_im = cube(src.e_theta_im);
        dst.e_phi_re = cube(src.e_phi_re), dst.e_phi_im = cube(src.e_phi_im);
        dst.azimuth_grid = arma::Col<dtype>(const_cast<dtype *>(src.azimuth_grid.memptr()), n_az, false, true);
        dst.elevation_grid = arma::Col<dtype>(const_cast<dtype *>(src.elevation_grid.memptr()), n_el, false, true);

        if (src.element_pos.is_empty())
            dst.element_pos.zeros(3ULL, n_elements_irs);
        else
            dst.element_pos = arma::Mat<dtype>(const_cast<dtype *>(src.element_pos.memptr()), 3ULL, n_elements_irs, false, true);

        dst.coupling_re = arma::Mat<dtype>(const_cast<dtype *>(src.coupling_re.colptr(i_irs)), n_elements_irs, 1ULL, false, true);
        if (src.coupling_im.is_empty())
            dst.coupling_im.zeros(n_elements_irs, 1ULL);
        else
            dst.coupling_im = arma::Mat<dtype>(const_cast<dtype *>(src.coupling_im.colptr(i_irs)), n_elements_irs, 1ULL, false, true);

        if (center_freq > dtype(1.0e-5))
            dst.center_freq = center_freq;
    };
    quadriga_lib::arrayant<dtype> irs_1, irs_2;
    irs_view(irs_1, irs_array);
    if (irs_array_2.n_elements() != 0)
        irs_view(irs_2, irs_array_2);
    const quadriga_lib::arrayant<dtype> &irs = irs_1, &irs_rx = irs_array_2.n_elements() != 0 ? irs_2 : irs_1;

    // Calculate the channel coefficients of the first segment from the TX to the IRS
    arma::Cube<dtype> coeff_re_1, coeff_im_1, delay_1, aod_1, eod_1;
    quadriga_lib::get_channels_spherical<dtype>(tx_array, irs, tx_pos, tx_orientation, irs_pos, irs_orientation, fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1,
                                                coeff_re_1, coeff_im_1, delay_1, center_freq, true, false,
                                                calc_angles ? &aod_1 : nullptr, calc_angles ? &eod_1 : nullptr, nullptr, nullptr, use_kernel);

    // Calculate channel coefficients of the second segment from the IRS to the RX
    arma::Cube<dtype> coeff_re_2, coeff_im_2, delay_2, aoa_2, eoa_2;
    quadriga_lib::get_channels_spherical<dtype>(irs_rx, rx_array, irs_pos, irs_orientation, rx_pos, rx_orientation, fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2,
                                                coeff_re_2, coeff_im_2, delay_2, center_freq, true, false,
                                                nullptr, nullptr, calc_angles ? &aoa_2 : nullptr, calc_angles ? &eoa_2 : nullptr, use_kernel);

    // Calculate the LOS delay between TX and RX
    dtype los_delay = dtype(0.0);
    if (!use_absolute_delays)
        los_delay = qd_calc_length(p_tp[0], p_tp[1], p_tp[2], p_rp[0], p_rp[1], p_rp[2]) * dtype(1.0 / qd_speed_of_light);

    // Combine the two channel segments
    arma::uword n_path_1 = coeff_re_1.n_slices;
    arma::uword n_path_2 = coeff_re_2.n_slices;
    arma::uword n_path = n_path_1 * n_path_2;
    arma::uword n_tx = coeff_re_1.n_cols;
    arma::uword n_rx = coeff_re_2.n_rows;

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
        throw std::invalid_argument("Optional input 'active_path' does not have the correct number of elements.");

    arma::uword i_path = 0ULL, j_path = 0ULL;
    for (arma::uword i_path_1 = 0ULL; i_path_1 < n_path_1; ++i_path_1)
    {
        dtype *c1r = coeff_re_1.slice_memptr(i_path_1);
        dtype *c1i = coeff_im_1.slice_memptr(i_path_1);
        dtype *d1 = delay_1.slice_memptr(i_path_1);
        dtype *p_aod_1 = calc_angles ? aod_1.slice_memptr(i_path_1) : nullptr;
        dtype *p_eod_1 = calc_angles ? eod_1.slice_memptr(i_path_1) : nullptr;

        for (arma::uword i_path_2 = 0ULL; i_path_2 < n_path_2; ++i_path_2)
        {
            dtype *c2r = coeff_re_2.slice_memptr(i_path_2);
            dtype *c2i = coeff_im_2.slice_memptr(i_path_2);
            dtype *d2 = delay_2.slice_memptr(i_path_2);
            dtype *p_aoa_2 = calc_angles ? aoa_2.slice_memptr(i_path_2) : nullptr;
            dtype *p_eoa_2 = calc_angles ? eoa_2.slice_memptr(i_path_2) : nullptr;

            dtype *cr = coeff_re_local.slice_memptr(i_path);
            dtype *ci = coeff_im_local.slice_memptr(i_path);
            dtype *dl = delay_local.slice_memptr(i_path);
            dtype *p_aod = calc_angles ? aod_local.slice_memptr(i_path) : nullptr;
            dtype *p_eod = calc_angles ? eod_local.slice_memptr(i_path) : nullptr;
            dtype *p_aoa = calc_angles ? aoa_local.slice_memptr(i_path) : nullptr;
            dtype *p_eoa = calc_angles ? eoa_local.slice_memptr(i_path) : nullptr;

            dtype pow = dtype(0.0);
            for (arma::uword i_tx = 0ULL; i_tx < n_tx; ++i_tx)
            {
                dtype a = c1r[i_tx], b = c1i[i_tx], dd1 = d1[i_tx];
                for (arma::uword i_rx = 0ULL; i_rx < n_rx; ++i_rx)
                {
                    dtype c = c2r[i_rx], d = c2i[i_rx];
                    dtype re = a * c - b * d;
                    dtype im = a * d + b * c;

                    arma::uword i = i_tx * n_rx + i_rx;
                    cr[i] = re, ci[i] = im;
                    dl[i] = dd1 + d2[i_rx] - los_delay;

                    re = re * re + im * im;
                    pow = (re > pow) ? re : pow;

                    if (calc_angles)
                    {
                        p_aod[i] = p_aod_1[i_tx];
                        p_eod[i] = p_eod_1[i_tx];
                        p_aoa[i] = p_aoa_2[i_rx];
                        p_eoa[i] = p_eoa_2[i_rx];
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
                                             arma::uword i_irs, float threshold_dB, float center_freq, bool use_absolute_delays,
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
