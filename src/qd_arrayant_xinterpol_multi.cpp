// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "qd_arrayant_interpolate.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "slerp.h"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# arrayant_interpolate_multi
Interpolate multi-frequency arrayant patterns at arbitrary angles and frequencies

- For each requested frequency, finds the two bracketing `center_frequency` entries, runs spatial interpolation on both via `qd_arrayant_interpolate`, then blends results in the frequency dimension.
- Frequency blending uses SLERP of complex field values with automatic fallback to linear interpolation when phase difference exceeds a threshold.
- Out-of-range frequencies are clamped to the nearest entry (no extrapolation).
- Consecutive frequency requests sharing the same bracketing entries reuse cached spatial interpolation results; sort `frequency` ascending or descending for best cache utilization.
- If `validate_input` is true, calls [[arrayant_is_valid_multi]] once before processing; set to `false` in performance-critical loops after initial validation.

## Declaration:
```
void quadriga_lib::arrayant_interpolate_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> *azimuth,
        const arma::Mat<dtype> *elevation,
        const arma::Col<dtype> *frequency,
        arma::Cube<dtype> *V_re,
        arma::Cube<dtype> *V_im,
        arma::Cube<dtype> *H_re,
        arma::Cube<dtype> *H_im,
        arma::uvec i_element = {},
        const arma::Cube<dtype> *orientation = nullptr,
        const arma::Mat<dtype> *element_pos_i = nullptr,
        bool validate_input = true);
```

## Inputs:
- **`arrayant_vec`** — Multi-frequency arrayant vector; entries need not be sorted by frequency
- **`azimuth`** — Azimuth angles in rad; must not be NULL, `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in rad; must not be NULL; size must match `azimuth`
- **`frequency`** — Target frequencies; must not be NULL or empty; `[n_freq]`
- **`i_element`** *(optional)* — 0-based element indices to interpolate; if empty, all elements are used (`n_out = n_elements`)
- **`orientation`** *(optional)* — Antenna orientation (bank, tilt, heading) in rad, applied at all frequencies; `[3,1,1]`; `[3,n_out,1]`; `[3,1,n_ang]`, or `[3,n_out,n_ang]`
- **`element_pos_i`** *(optional)* — Override element positions in m; if `nullptr`, positions from entry 0 are used; `[3, n_out]`
- **`validate_input`** *(optional)* — If `true`, validates `arrayant_vec` with [[arrayant_is_valid_multi]] before processing

## Outputs:
- **`V_re`** — Real part of interpolated e-theta field; must not be NULL; `[n_out, n_ang, n_freq]`
- **`V_im`** — Imaginary part of interpolated e-theta field; must not be NULL; `[n_out, n_ang, n_freq]`
- **`H_re`** — Real part of interpolated e-phi field; must not be NULL; `[n_out, n_ang, n_freq]`
- **`H_im`** — Imaginary part of interpolated e-phi field; must not be NULL; `[n_out, n_ang, n_freq]`

## Example:
```
auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);
arma::mat az = {0.0, 1.5708, -1.5708, 3.14159};
arma::mat el(1, 4, arma::fill::zeros);
arma::vec qf = {250.0, 1500.0, 8000.0};
arma::cube V_re, V_im, H_re, H_im;
quadriga_lib::arrayant_interpolate_multi(speaker, &az, &el, &qf, &V_re, &V_im, &H_re, &H_im);
```

## See also:
- .[[interpolate]] (single-frequency spatial interpolation)
- [[arrayant_concat_multi]] (build multi-element/multi-frequency models)
- [[arrayant_is_valid_multi]] (validation called when validate_input is true)
- [[generate_speaker]] (typical source of multi-frequency arrayant vectors)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_interpolate_multi(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
                                              const arma::Mat<dtype> *azimuth,
                                              const arma::Mat<dtype> *elevation,
                                              const arma::Col<dtype> *frequency,
                                              arma::Cube<dtype> *V_re, arma::Cube<dtype> *V_im,
                                              arma::Cube<dtype> *H_re, arma::Cube<dtype> *H_im,
                                              arma::uvec i_element,
                                              const arma::Cube<dtype> *orientation,
                                              const arma::Mat<dtype> *element_pos_i,
                                              bool validate_input)
{
    // --- Input validation ---
    if (arrayant_vec.empty())
        throw std::invalid_argument("arrayant_interpolate_multi: Input vector is empty.");
    if (azimuth == nullptr)
        throw std::invalid_argument("arrayant_interpolate_multi: Input 'azimuth' cannot be NULL.");
    if (elevation == nullptr)
        throw std::invalid_argument("arrayant_interpolate_multi: Input 'elevation' cannot be NULL.");
    if (frequency == nullptr || frequency->n_elem == 0)
        throw std::invalid_argument("arrayant_interpolate_multi: Input 'frequency' cannot be NULL or empty.");
    if (V_re == nullptr || V_im == nullptr || H_re == nullptr || H_im == nullptr)
        throw std::invalid_argument("arrayant_interpolate_multi: Outputs 'V_re', 'V_im', 'H_re', 'H_im' cannot be NULL.");

    if (validate_input)
    {
        std::string err = quadriga_lib::arrayant_is_valid_multi(arrayant_vec, true);
        if (!err.empty())
            throw std::invalid_argument("arrayant_interpolate_multi: Input validation failed: " + err);
    }

    arma::uword n_entries = (arma::uword)arrayant_vec.size();
    arma::uword n_ang = azimuth->n_cols;
    arma::uword n_freq = frequency->n_elem;

    // Determine n_elements and n_out
    arma::uword n_elements = arrayant_vec[0].e_theta_re.n_slices;
    if (i_element.is_empty())
        i_element = arma::regspace<arma::uvec>(0, n_elements - 1);
    else
    {
        if (arma::any(i_element >= n_elements))
            throw std::invalid_argument("arrayant_interpolate_multi: Element indices exceed array antenna size.");
    }

    arma::uword n_out = i_element.n_elem;

    if (azimuth->n_rows != 1 && azimuth->n_rows != n_out)
        throw std::invalid_argument("arrayant_interpolate_multi: Number of rows in 'azimuth' must be 1 or n_out.");
    if (elevation->n_rows != azimuth->n_rows || elevation->n_cols != n_ang)
        throw std::invalid_argument("arrayant_interpolate_multi: Sizes of 'azimuth' and 'elevation' do not match.");

    // Validate orientation
    if (orientation != nullptr && orientation->n_elem != 0)
    {
        if (orientation->n_rows != 3)
            throw std::invalid_argument("arrayant_interpolate_multi: 'orientation' must have 3 rows.");
        if (orientation->n_cols != 1 && orientation->n_cols != n_out)
            throw std::invalid_argument("arrayant_interpolate_multi: 'orientation' must have 1 or n_out columns.");
        if (orientation->n_slices != 1 && orientation->n_slices != n_ang)
            throw std::invalid_argument("arrayant_interpolate_multi: 'orientation' must have 1 or n_ang slices.");
    }

    // Validate alternative element positions
    if (element_pos_i != nullptr && element_pos_i->n_elem != 0)
    {
        if (element_pos_i->n_rows != 3 || element_pos_i->n_cols != n_out)
            throw std::invalid_argument("arrayant_interpolate_multi: 'element_pos_i' must have 3 rows and n_out columns.");
    }

    // --- Resize output cubes ---
    if (V_re->n_rows != n_out || V_re->n_cols != n_ang || V_re->n_slices != n_freq)
        V_re->set_size(n_out, n_ang, n_freq);
    if (V_im->n_rows != n_out || V_im->n_cols != n_ang || V_im->n_slices != n_freq)
        V_im->set_size(n_out, n_ang, n_freq);
    if (H_re->n_rows != n_out || H_re->n_cols != n_ang || H_re->n_slices != n_freq)
        H_re->set_size(n_out, n_ang, n_freq);
    if (H_im->n_rows != n_out || H_im->n_cols != n_ang || H_im->n_slices != n_freq)
        H_im->set_size(n_out, n_ang, n_freq);

    // --- Build sorted frequency index from arrayant center_frequencies ---
    arma::Col<dtype> cf(n_entries);
    for (arma::uword i = 0; i < n_entries; ++i)
        cf[i] = (dtype)arrayant_vec[i].center_frequency;

    // Build sort index (center frequencies should already be sorted, but be safe)
    arma::uvec sort_idx = arma::sort_index(cf);
    arma::Col<dtype> cf_sorted(n_entries);
    for (arma::uword i = 0; i < n_entries; ++i)
        cf_sorted[i] = cf[sort_idx[i]];

    // --- Prepare element index in 1-based unsigned format for qd_arrayant_interpolate ---
    arma::Col<unsigned> i_element_1based(n_out);
    {
        unsigned *pi = i_element_1based.memptr();
        arma::uword *pu = i_element.memptr();
        for (arma::uword i = 0; i < n_out; ++i)
            pi[i] = (unsigned)pu[i] + 1;
    }

    // --- Prepare element positions ---
    arma::Mat<dtype> element_pos_local(3, n_out, arma::fill::zeros);
    if (element_pos_i != nullptr && element_pos_i->n_elem != 0)
    {
        std::memcpy(element_pos_local.memptr(), element_pos_i->memptr(), 3 * n_out * sizeof(dtype));
    }
    else if (!arrayant_vec[0].element_pos.is_empty())
    {
        const dtype *ptrI = arrayant_vec[0].element_pos.memptr();
        dtype *ptrO = element_pos_local.memptr();
        for (arma::uword i = 0; i < n_out; ++i)
            std::memcpy(&ptrO[3 * i], &ptrI[3 * i_element[i]], 3 * sizeof(dtype));
    }

    // --- Prepare orientation ---
    arma::Cube<dtype> orientation_empty(3, 1, 1, arma::fill::zeros);
    const arma::Cube<dtype> *orientation_local = (orientation == nullptr || orientation->n_elem == 0)
                                                     ? &orientation_empty
                                                     : orientation;

    // --- Allocate two interchangeable interpolation buffers; lo/hi assigned via pointers ---
    arma::uword n_vals = n_out * n_ang;

    arma::Mat<dtype> buf_A_Vr(n_out, n_ang), buf_A_Vi(n_out, n_ang);
    arma::Mat<dtype> buf_A_Hr(n_out, n_ang), buf_A_Hi(n_out, n_ang);
    arma::Mat<dtype> buf_B_Vr(n_out, n_ang), buf_B_Vi(n_out, n_ang);
    arma::Mat<dtype> buf_B_Hr(n_out, n_ang), buf_B_Hi(n_out, n_ang);

    arma::Mat<dtype> *p_Vr_lo = &buf_A_Vr, *p_Vi_lo = &buf_A_Vi;
    arma::Mat<dtype> *p_Hr_lo = &buf_A_Hr, *p_Hi_lo = &buf_A_Hi;
    arma::Mat<dtype> *p_Vr_hi = &buf_B_Vr, *p_Vi_hi = &buf_B_Vi;
    arma::Mat<dtype> *p_Hr_hi = &buf_B_Hr, *p_Hi_hi = &buf_B_Hi;

    arma::sword cached_lo = -1, cached_hi = -1;

    auto swap_buffers = [&]()
    {
        std::swap(p_Vr_lo, p_Vr_hi);
        std::swap(p_Vi_lo, p_Vi_hi);
        std::swap(p_Hr_lo, p_Hr_hi);
        std::swap(p_Hi_lo, p_Hi_hi);
        std::swap(cached_lo, cached_hi);
    };

    auto interpolate_lo = [&](arma::uword idx)
    {
        const auto &a = arrayant_vec[idx];
        qd_arrayant_interpolate<dtype>(a.e_theta_re, a.e_theta_im, a.e_phi_re, a.e_phi_im,
                                       a.azimuth_grid, a.elevation_grid,
                                       *azimuth, *elevation,
                                       i_element_1based, *orientation_local, element_pos_local,
                                       *p_Vr_lo, *p_Vi_lo, *p_Hr_lo, *p_Hi_lo);
        cached_lo = (arma::sword)idx;
    };

    auto interpolate_hi = [&](arma::uword idx)
    {
        const auto &a = arrayant_vec[idx];
        qd_arrayant_interpolate<dtype>(a.e_theta_re, a.e_theta_im, a.e_phi_re, a.e_phi_im,
                                       a.azimuth_grid, a.elevation_grid,
                                       *azimuth, *elevation,
                                       i_element_1based, *orientation_local, element_pos_local,
                                       *p_Vr_hi, *p_Vi_hi, *p_Hr_hi, *p_Hi_hi);
        cached_hi = (arma::sword)idx;
    };

    // --- Process each requested frequency ---
    const dtype *p_freq = frequency->memptr();
    const dtype eps = std::numeric_limits<dtype>::epsilon();

    for (arma::uword fi = 0; fi < n_freq; ++fi)
    {
        dtype f = p_freq[fi];

        // Find bracketing indices in sorted center_frequency list
        arma::uword idx_lo, idx_hi;
        dtype w;

        if (n_entries == 1)
        {
            idx_lo = sort_idx[0];
            idx_hi = idx_lo;
            w = dtype(0.0);
        }
        else if (f <= cf_sorted[0])
        {
            idx_lo = sort_idx[0];
            idx_hi = idx_lo;
            w = dtype(0.0);
        }
        else if (f >= cf_sorted[n_entries - 1])
        {
            idx_lo = sort_idx[n_entries - 1];
            idx_hi = idx_lo;
            w = dtype(0.0);
        }
        else
        {
            arma::uword k = 0;
            while (k + 1 < n_entries && cf_sorted[k + 1] <= f)
                ++k;
            idx_lo = sort_idx[k];
            idx_hi = sort_idx[k + 1];
            dtype f_lo = cf_sorted[k], f_hi = cf_sorted[k + 1];
            w = (f - f_lo) / (f_hi - f_lo);
        }

        bool need_blend = (idx_lo != idx_hi) && (w >= eps);

        // --- Resolve lo buffer ---
        if ((arma::sword)idx_lo == cached_hi)
        {
            // Ascending case: new lo was old hi → swap buffers, no recomputation
            swap_buffers();
        }
        else if (need_blend && (arma::sword)idx_hi == cached_lo)
        {
            // Descending case: new hi is in lo buffer, but lo needs fresh compute.
            // Swap first to preserve hi data, then compute lo fresh.
            swap_buffers();
            interpolate_lo(idx_lo);
        }

        if ((arma::sword)idx_lo != cached_lo)
            interpolate_lo(idx_lo);

        // --- Copy or blend ---
        if (!need_blend)
        {
            std::memcpy(V_re->slice_memptr(fi), p_Vr_lo->memptr(), n_vals * sizeof(dtype));
            std::memcpy(V_im->slice_memptr(fi), p_Vi_lo->memptr(), n_vals * sizeof(dtype));
            std::memcpy(H_re->slice_memptr(fi), p_Hr_lo->memptr(), n_vals * sizeof(dtype));
            std::memcpy(H_im->slice_memptr(fi), p_Hi_lo->memptr(), n_vals * sizeof(dtype));
        }
        else
        {
            if ((arma::sword)idx_hi != cached_hi)
                interpolate_hi(idx_hi);

            const dtype *pVrl = p_Vr_lo->memptr(), *pVil = p_Vi_lo->memptr();
            const dtype *pHrl = p_Hr_lo->memptr(), *pHil = p_Hi_lo->memptr();
            const dtype *pVrh = p_Vr_hi->memptr(), *pVih = p_Vi_hi->memptr();
            const dtype *pHrh = p_Hr_hi->memptr(), *pHih = p_Hi_hi->memptr();
            dtype *pVro = V_re->slice_memptr(fi), *pVio = V_im->slice_memptr(fi);
            dtype *pHro = H_re->slice_memptr(fi), *pHio = H_im->slice_memptr(fi);

            for (arma::uword j = 0; j < n_vals; ++j)
            {
                slerp_complex_mf(pVrl[j], pVil[j], pVrh[j], pVih[j], w, pVro[j], pVio[j]);
                slerp_complex_mf(pHrl[j], pHil[j], pHrh[j], pHih[j], w, pHro[j], pHio[j]);
            }
        }
    }
}

template void quadriga_lib::arrayant_interpolate_multi(
    const std::vector<quadriga_lib::arrayant<float>> &, const arma::Mat<float> *, const arma::Mat<float> *,
    const arma::Col<float> *, arma::Cube<float> *, arma::Cube<float> *, arma::Cube<float> *, arma::Cube<float> *,
    arma::uvec, const arma::Cube<float> *, const arma::Mat<float> *, bool);

template void quadriga_lib::arrayant_interpolate_multi(
    const std::vector<quadriga_lib::arrayant<double>> &, const arma::Mat<double> *, const arma::Mat<double> *,
    const arma::Col<double> *, arma::Cube<double> *, arma::Cube<double> *, arma::Cube<double> *, arma::Cube<double> *,
    arma::uvec, const arma::Cube<double> *, const arma::Mat<double> *, bool);