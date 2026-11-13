// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "qd_arrayant_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"
#include "quadriga_lib_generic_functions.hpp"
#include "slerp.h"

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#endif

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Throws std::invalid_argument if one of the outputs out[0 .. N-1] is a data member of 'ant' or shares memory with
// one; nullptr outputs are skipped
template <typename T, std::size_t N, typename dtype>
static inline void qd_check_alias_arrayant(const T *const (&out)[N], const char *const (&name)[N],
                                           const quadriga_lib::arrayant<dtype> &ant)
{
    for (std::size_t i = 0; i < N; ++i)
        qd_check_alias(out[i], name[i], &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im, &ant.azimuth_grid,
                       &ant.elevation_grid, &ant.element_pos, &ant.coupling_re, &ant.coupling_im);
}

// Checks the shapes of the angles and of the optional inputs of the interpolation functions (nullptr or empty
// optional inputs pass); n_out and n_ang as in arrayant_interpolate
template <typename dtype>
static inline void qd_check_interpolation_inputs(const arma::Mat<dtype> &azimuth, const arma::Mat<dtype> &elevation,
                                                 const arma::Cube<dtype> *orientation, const arma::Mat<dtype> *element_pos,
                                                 arma::uword n_out)
{
    const arma::uword n_ang = azimuth.n_cols;
    if (n_ang > (arma::uword)INT_MAX)
        throw std::invalid_argument("Number of angles cannot exceed INT_MAX.");
    if (azimuth.n_rows != 1 && azimuth.n_rows != n_out)
        throw std::invalid_argument("Input 'azimuth' must have 1 or n_out rows.");
    if (elevation.n_rows != azimuth.n_rows || elevation.n_cols != n_ang)
        throw std::invalid_argument("Input 'elevation' must have the same size as 'azimuth'.");
    if (orientation != nullptr && orientation->n_elem != 0 &&
        (orientation->n_rows != 3 || (orientation->n_cols != 1 && orientation->n_cols != n_out) ||
         (orientation->n_slices != 1 && orientation->n_slices != n_ang)))
        throw std::invalid_argument("Input 'orientation' must have size [3, 1, 1], [3, n_out, 1], [3, 1, n_ang] or [3, n_out, n_ang].");
    if (element_pos != nullptr && element_pos->n_elem != 0 && (element_pos->n_rows != 3 || element_pos->n_cols != n_out))
        throw std::invalid_argument("Input 'element_pos' must have size [3, n_out].");
}

// Internal dispatch, declared in qd_arrayant_functions.hpp
template <typename dtype>
void qd_arrayant_interpolate_dispatch(const quadriga_lib::arrayant<dtype> &ant,
                                      const arma::Mat<dtype> &azimuth, const arma::Mat<dtype> &elevation,
                                      const arma::uvec &i_element, const arma::Cube<dtype> &orientation,
                                      const arma::Mat<dtype> &element_pos,
                                      arma::Mat<dtype> &V_re, arma::Mat<dtype> &V_im, arma::Mat<dtype> &H_re, arma::Mat<dtype> &H_im,
                                      arma::Mat<dtype> *dist, arma::Mat<dtype> *azimuth_loc, arma::Mat<dtype> *elevation_loc,
                                      arma::Mat<dtype> *gamma, int kernel)
{
    const arma::uword n_out = i_element.n_elem, n_ang = azimuth.n_cols;

    if (n_out == 0)
        throw std::invalid_argument("Input 'i_element' cannot be empty.");
    if (arma::any(i_element >= ant.e_theta_re.n_slices))
        throw std::invalid_argument("Input 'i_element' contains indices that exceed the number of elements.");
    if (orientation.n_elem == 0 || element_pos.n_elem == 0)
        throw std::invalid_argument("Inputs 'orientation' and 'element_pos' cannot be empty.");
    qd_check_interpolation_inputs(azimuth, elevation, &orientation, &element_pos, n_out);
    for (const arma::Mat<dtype> *out : {&V_re, &V_im, &H_re, &H_im, dist, azimuth_loc, elevation_loc, gamma})
        if (out != nullptr && (out->n_rows != n_out || out->n_cols != n_ang))
            throw std::invalid_argument("Outputs must have size [n_out, n_ang].");
    if (kernel != 1 && kernel != 2)
        throw std::invalid_argument("Kernel must be 1 (GENERIC) or 2 (AVX2).");
    if (kernel == 2)
        qd_arrayant_interpolate_kernel(2, ant.e_theta_re.n_elem);

    if (n_ang == 0)
        return;

    // Transposed rotation matrices, 9 values per orientation column and slice
    const arma::uword n_rot = orientation.n_cols * orientation.n_slices;
    arma::Col<dtype> rotation(9 * n_rot, arma::fill::none);
    qd_rotation_matrix(orientation.memptr(), rotation.memptr(), n_rot, true);

    dtype *p_dist = dist == nullptr ? nullptr : dist->memptr();
    dtype *p_azimuth_loc = azimuth_loc == nullptr ? nullptr : azimuth_loc->memptr();
    dtype *p_elevation_loc = elevation_loc == nullptr ? nullptr : elevation_loc->memptr();
    dtype *p_gamma = gamma == nullptr ? nullptr : gamma->memptr();

    if (kernel == 2)
    {
#if BUILD_WITH_AVX2
        qd_arrayant_interpolate_AVX2<dtype>(ant.e_theta_re.memptr(), ant.e_theta_im.memptr(), ant.e_phi_re.memptr(), ant.e_phi_im.memptr(),
                                            ant.e_theta_re.n_rows, ant.e_theta_re.n_cols,
                                            ant.azimuth_grid.memptr(), ant.elevation_grid.memptr(),
                                            azimuth.memptr(), elevation.memptr(), azimuth.n_rows, n_ang,
                                            rotation.memptr(), orientation.n_cols, orientation.n_slices,
                                            element_pos.memptr(), i_element.memptr(), n_out,
                                            V_re.memptr(), V_im.memptr(), H_re.memptr(), H_im.memptr(),
                                            p_dist, p_azimuth_loc, p_elevation_loc, p_gamma);
#endif
    }
    else
        qd_arrayant_interpolate_GENERIC<dtype>(ant.e_theta_re.memptr(), ant.e_theta_im.memptr(), ant.e_phi_re.memptr(), ant.e_phi_im.memptr(),
                                               ant.e_theta_re.n_rows, ant.e_theta_re.n_cols,
                                               ant.azimuth_grid.memptr(), ant.elevation_grid.memptr(),
                                               azimuth.memptr(), elevation.memptr(), azimuth.n_rows, n_ang,
                                               rotation.memptr(), orientation.n_cols, orientation.n_slices,
                                               element_pos.memptr(), i_element.memptr(), n_out,
                                               V_re.memptr(), V_im.memptr(), H_re.memptr(), H_im.memptr(),
                                               p_dist, p_azimuth_loc, p_elevation_loc, p_gamma);
}

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# arrayant_interpolate
Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the antenna and the resulting polarization rotation applied
- Between the samples of the pattern, amplitude and phase are interpolated separately (spherical interpolation),
  with linear interpolation where neighbouring samples have nearly opposite phases; the azimuth grid wraps around,
  elevations outside the grid take the values of the first or last grid row
- `azimuth` with one row gives the same angles for all outputs (planar wave), `n_out` rows one angle per output
  (spherical wave); `n_ang = 0` gives outputs of size `[n_out, 0]`
- Angles are not range-checked; values outside [-pi, pi] and [-pi/2, pi/2] denote the corresponding direction. NaN or
  infinite angles give NaN in all outputs of that angle and output, without an exception
- Outputs are resized only when their size differs, so pre-allocated outputs are written in place
- Kernels: GENERIC computes in the precision of `dtype`; AVX2 computes in single precision also for `double`, with
  errors of up to about 2e-6 of the peak field value and 2e-6 rad in the angles; at a distance d (in rad) from a pole
  of the local frame up to about 2e-7 / d rad in the local azimuth and the polarization angle and 1e-7 / d rad (at most
  4e-4 rad) in the local elevation; auto selects GENERIC in this version, which may change in a later version
- Throws `std::invalid_argument` for an invalid `ant`, sizes that do not match, element indices `>= n_elements`, an
  unknown or unavailable kernel, and outputs that are an input, a data member of `ant` or another output, or share
  memory with one

## Declaration:
```
void quadriga_lib::arrayant_interpolate(
    const arrayant<dtype> &ant,
    const arma::Mat<dtype> &azimuth,
    const arma::Mat<dtype> &elevation,
    arma::Mat<dtype> &V_re,
    arma::Mat<dtype> &V_im,
    arma::Mat<dtype> &H_re,
    arma::Mat<dtype> &H_im,
    const arma::uvec *i_element = nullptr,
    const arma::Cube<dtype> *orientation = nullptr,
    const arma::Mat<dtype> *element_pos = nullptr,
    arma::Mat<dtype> *dist = nullptr,
    arma::Mat<dtype> *azimuth_loc = nullptr,
    arma::Mat<dtype> *elevation_loc = nullptr,
    arma::Mat<dtype> *gamma = nullptr,
    int use_kernel = 0);
```

## Inputs:
- **`ant`** — Array antenna with `n_elements` elements; see [[arrayant]]
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; `nullptr` or empty: all elements in
  order (`n_out = n_elements`); `[n_out]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) of the antenna, for all outputs and angles,
  per output, per angle, or both; `nullptr` or empty: no rotation; `[3, 1, 1]`, `[3, n_out, 1]`, `[3, 1, n_ang]` or
  `[3, n_out, n_ang]`
- **`element_pos`** *(optional)* — Element positions in local antenna coordinates, used for `dist`; `nullptr` or
  empty: the columns `i_element` of `ant.element_pos`, zeros if that is empty; `[3, n_out]`
- **`use_kernel`** *(optional)* — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (throws if unavailable or if the pattern
  has more than `INT_MAX` samples); other values throw

## Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; `[n_out, n_ang]`
- **`dist`** *(optional)* — Distance of each element from the wavefront plane through the origin of the antenna;
  negative for elements closer to the source; `[n_out, n_ang]`
- **`azimuth_loc`** / **`elevation_loc`** *(optional)* — Azimuth and elevation in the local antenna frame;
  `[n_out, n_ang]`
- **`gamma`** *(optional)* — Polarization rotation angle from the global to the local frame; `[n_out, n_ang]`

## Example:
```
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
arma::mat azimuth = {0.0, 0.5 * arma::datum::pi, -0.5 * arma::datum::pi, arma::datum::pi};
arma::mat elevation(1, azimuth.n_elem, arma::fill::zeros);
arma::mat V_re, V_im, H_re, H_im;
quadriga_lib::arrayant_interpolate(ant, azimuth, elevation, V_re, V_im, H_re, H_im);
```

## See also:
- [[arrayant_interpolate_multi]] (frequency-dependent array antennas)
- [[arrayant]] (array antenna class)
- [[get_channels_spherical]] (channel coefficients from the interpolated patterns)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_interpolate(const arrayant<dtype> &ant,
                                        const arma::Mat<dtype> &azimuth,
                                        const arma::Mat<dtype> &elevation,
                                        arma::Mat<dtype> &V_re,
                                        arma::Mat<dtype> &V_im,
                                        arma::Mat<dtype> &H_re,
                                        arma::Mat<dtype> &H_im,
                                        const arma::uvec *i_element,
                                        const arma::Cube<dtype> *orientation,
                                        const arma::Mat<dtype> *element_pos,
                                        arma::Mat<dtype> *dist,
                                        arma::Mat<dtype> *azimuth_loc,
                                        arma::Mat<dtype> *elevation_loc,
                                        arma::Mat<dtype> *gamma,
                                        int use_kernel)
{
    const std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (!error_message.empty())
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);

    const arma::uword n_elements = ant.e_theta_re.n_slices;
    const bool has_i_element = i_element != nullptr && i_element->n_elem != 0;
    if (has_i_element && arma::any(*i_element >= n_elements))
        throw std::invalid_argument("Input 'i_element' contains indices that exceed the number of elements.");
    const arma::uword n_out = has_i_element ? i_element->n_elem : n_elements;
    const arma::uword n_ang = azimuth.n_cols;

    qd_check_interpolation_inputs(azimuth, elevation, orientation, element_pos, n_out);
    const int kernel = qd_arrayant_interpolate_kernel(use_kernel, ant.e_theta_re.n_elem);

    const arma::Mat<dtype> *outs[] = {&V_re, &V_im, &H_re, &H_im, dist, azimuth_loc, elevation_loc, gamma};
    const char *names[] = {"V_re", "V_im", "H_re", "H_im", "dist", "azimuth_loc", "elevation_loc", "gamma"};
    qd_check_alias_list(outs, names, &azimuth, &elevation, i_element, orientation, element_pos);
    qd_check_alias_arrayant(outs, names, ant);

    for (arma::Mat<dtype> *out : {&V_re, &V_im, &H_re, &H_im, dist, azimuth_loc, elevation_loc, gamma})
        if (out != nullptr && (out->n_rows != n_out || out->n_cols != n_ang))
            out->set_size(n_out, n_ang);

    // Defaults of the optional inputs
    arma::uvec i_element_all;
    if (!has_i_element)
        i_element_all = arma::regspace<arma::uvec>(0, n_elements - 1);

    const arma::Cube<dtype> no_rotation(3, 1, 1, arma::fill::zeros);
    const bool has_orientation = orientation != nullptr && orientation->n_elem != 0;

    arma::Mat<dtype> pos_local;
    const arma::Mat<dtype> *pos = element_pos;
    if (element_pos == nullptr || element_pos->n_elem == 0)
    {
        if (ant.element_pos.n_elem == 0)
            pos_local.zeros(3, n_out), pos = &pos_local;
        else if (has_i_element)
            pos_local = ant.element_pos.cols(*i_element), pos = &pos_local;
        else
            pos = &ant.element_pos;
    }

    qd_arrayant_interpolate_dispatch<dtype>(ant, azimuth, elevation, has_i_element ? *i_element : i_element_all,
                                            has_orientation ? *orientation : no_rotation, *pos,
                                            V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma, kernel);
}

/*!MD
# arrayant_interpolate_multi
Interpolate a multi-frequency array antenna at given angles and frequencies

- For each target frequency, interpolates the two entries whose center frequencies bracket it, each as
  [[arrayant_interpolate]], and blends their fields: spherical interpolation of the complex values over frequency,
  linear where the two values have nearly opposite phases or one of them is zero
- Target frequencies outside the range of the center frequencies take the nearest entry (no extrapolation)
- Consecutive target frequencies with the same bracketing entries reuse the spatial results; target frequencies in
  ascending or descending order are fastest
- The entries must pass [[arrayant_is_valid_multi]]: same grids, number of elements and element positions
- Angles, element indices, orientation and kernels as in [[arrayant_interpolate]]; `n_ang = 0` gives outputs of size
  `[n_out, 0, n_freq]`
- Throws `std::invalid_argument` for an empty or invalid `arrayant_vec`, target frequencies that are not finite, and
  for the errors of [[arrayant_interpolate]]

## Declaration:
```
void quadriga_lib::arrayant_interpolate_multi(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Mat<dtype> &azimuth,
    const arma::Mat<dtype> &elevation,
    arma::Cube<dtype> &V_re,
    arma::Cube<dtype> &V_im,
    arma::Cube<dtype> &H_re,
    arma::Cube<dtype> &H_im,
    const arma::Col<dtype> *frequency = nullptr,
    const arma::uvec *i_element = nullptr,
    const arma::Cube<dtype> *orientation = nullptr,
    const arma::Mat<dtype> *element_pos = nullptr,
    int use_kernel = 0);
```

## Inputs:
- **`arrayant_vec`** — Multi-frequency array antenna, one entry per center frequency; entries need not be sorted by
  frequency
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`frequency`** *(optional)* — Target frequencies in Hz; `nullptr` or empty: the center frequencies of the entries
  in vector order; `[n_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; `nullptr` or empty: all elements;
  `[n_out]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading), applied at all frequencies; shapes as in
  [[arrayant_interpolate]]
- **`element_pos`** *(optional)* — Element positions in m; checked for its size only, since no output depends on the
  positions; `[3, n_out]`
- **`use_kernel`** *(optional)* — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2; as in [[arrayant_interpolate]]

## Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang, n_freq]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; `[n_out, n_ang, n_freq]`

## Example:
```
auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);
arma::mat az = {0.0, 1.5708, -1.5708, 3.14159};
arma::mat el(1, 4, arma::fill::zeros);
arma::vec freq = {250.0, 1500.0, 8000.0};
arma::cube V_re, V_im, H_re, H_im;
quadriga_lib::arrayant_interpolate_multi(speaker, az, el, V_re, V_im, H_re, H_im, &freq);
```

## See also:
- [[arrayant_interpolate]] (single-frequency interpolation)
- [[arrayant_concat_multi]] (build multi-element, multi-frequency models)
- [[arrayant_is_valid_multi]] (validation of the entries)
- [[generate_speaker]] (typical source of multi-frequency array antennas)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_interpolate_multi(const std::vector<arrayant<dtype>> &arrayant_vec,
                                              const arma::Mat<dtype> &azimuth,
                                              const arma::Mat<dtype> &elevation,
                                              arma::Cube<dtype> &V_re,
                                              arma::Cube<dtype> &V_im,
                                              arma::Cube<dtype> &H_re,
                                              arma::Cube<dtype> &H_im,
                                              const arma::Col<dtype> *frequency,
                                              const arma::uvec *i_element,
                                              const arma::Cube<dtype> *orientation,
                                              const arma::Mat<dtype> *element_pos,
                                              int use_kernel)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("Input 'arrayant_vec' cannot be empty.");
    const std::string error_message = qd_arrayant_is_valid_multi_quick(arrayant_vec);
    if (!error_message.empty())
        throw std::invalid_argument("Input 'arrayant_vec' is not valid: " + error_message);

    const quadriga_lib::arrayant<dtype> &ant0 = arrayant_vec[0];
    const arma::uword n_entries = (arma::uword)arrayant_vec.size();
    const arma::uword n_elements = ant0.e_theta_re.n_slices;
    const bool has_i_element = i_element != nullptr && i_element->n_elem != 0;
    if (has_i_element && arma::any(*i_element >= n_elements))
        throw std::invalid_argument("Input 'i_element' contains indices that exceed the number of elements.");
    const arma::uword n_out = has_i_element ? i_element->n_elem : n_elements;
    const arma::uword n_ang = azimuth.n_cols;

    qd_check_interpolation_inputs(azimuth, elevation, orientation, element_pos, n_out);

    const bool has_frequency = frequency != nullptr && frequency->n_elem != 0;
    if (has_frequency && !frequency->is_finite())
        throw std::invalid_argument("Input 'frequency' must contain finite values.");

    unsigned long long n_pattern_values = 0;
    for (const auto &a : arrayant_vec)
        n_pattern_values = std::max(n_pattern_values, (unsigned long long)a.e_theta_re.n_elem);
    const int kernel = qd_arrayant_interpolate_kernel(use_kernel, n_pattern_values);

    const arma::Cube<dtype> *outs[] = {&V_re, &V_im, &H_re, &H_im};
    const char *names[] = {"V_re", "V_im", "H_re", "H_im"};
    qd_check_alias_list(outs, names, &azimuth, &elevation, frequency, i_element, orientation, element_pos);
    for (const auto &a : arrayant_vec)
        qd_check_alias_arrayant(outs, names, a);

    // Center frequencies in vector order: default target frequencies and frequency index
    arma::Col<dtype> cf(n_entries);
    for (arma::uword i = 0; i < n_entries; ++i)
        cf[i] = arrayant_vec[i].center_frequency;
    const arma::Col<dtype> &freq = has_frequency ? *frequency : cf;
    const arma::uword n_freq = freq.n_elem;

    for (arma::Cube<dtype> *out : {&V_re, &V_im, &H_re, &H_im})
        if (out->n_rows != n_out || out->n_cols != n_ang || out->n_slices != n_freq)
            out->set_size(n_out, n_ang, n_freq);

    if (n_ang == 0)
        return;

    // Sort index of the center frequencies (entries need not be sorted)
    const arma::uvec sort_idx = arma::sort_index(cf);
    arma::Col<dtype> cf_sorted(n_entries);
    for (arma::uword i = 0; i < n_entries; ++i)
        cf_sorted[i] = cf[sort_idx[i]];

    // Defaults of the optional inputs; the positions are those of entry 0 (the entries have the same positions)
    arma::uvec i_element_all;
    if (!has_i_element)
        i_element_all = arma::regspace<arma::uvec>(0, n_elements - 1);
    const arma::uvec &i_el = has_i_element ? *i_element : i_element_all;

    const arma::Cube<dtype> no_rotation(3, 1, 1, arma::fill::zeros);
    const arma::Cube<dtype> &ori = (orientation == nullptr || orientation->n_elem == 0) ? no_rotation : *orientation;

    arma::Mat<dtype> pos(3, n_out, arma::fill::zeros);
    if (element_pos != nullptr && element_pos->n_elem != 0)
        pos = *element_pos;
    else if (!ant0.element_pos.is_empty())
        pos = ant0.element_pos.cols(i_el);

    // Two interchangeable buffers of spatial results; lo / hi assigned via pointers
    const arma::uword n_vals = n_out * n_ang;

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

    // Spatial interpolation of entry 'idx' into the lo or the hi buffers
    auto interpolate_into = [&](arma::uword idx, bool hi)
    {
        qd_arrayant_interpolate_dispatch<dtype>(arrayant_vec[idx], azimuth, elevation, i_el, ori, pos,
                                                hi ? *p_Vr_hi : *p_Vr_lo, hi ? *p_Vi_hi : *p_Vi_lo,
                                                hi ? *p_Hr_hi : *p_Hr_lo, hi ? *p_Hi_hi : *p_Hi_lo,
                                                nullptr, nullptr, nullptr, nullptr, kernel);
        (hi ? cached_hi : cached_lo) = (arma::sword)idx;
    };

    // Process each target frequency
    const dtype *p_freq = freq.memptr();
    const dtype eps = std::numeric_limits<dtype>::epsilon();

    for (arma::uword fi = 0; fi < n_freq; ++fi)
    {
        dtype f = p_freq[fi];

        // Find bracketing indices in sorted center_frequency list
        arma::uword idx_lo, idx_hi;
        dtype w;

        if (f <= cf_sorted[0])
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

        // Resolve the lo buffer
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
            interpolate_into(idx_lo, false);
        }

        if ((arma::sword)idx_lo != cached_lo)
            interpolate_into(idx_lo, false);

        // Copy or blend
        if (!need_blend)
        {
            std::memcpy(V_re.slice_memptr(fi), p_Vr_lo->memptr(), n_vals * sizeof(dtype));
            std::memcpy(V_im.slice_memptr(fi), p_Vi_lo->memptr(), n_vals * sizeof(dtype));
            std::memcpy(H_re.slice_memptr(fi), p_Hr_lo->memptr(), n_vals * sizeof(dtype));
            std::memcpy(H_im.slice_memptr(fi), p_Hi_lo->memptr(), n_vals * sizeof(dtype));
        }
        else
        {
            if ((arma::sword)idx_hi != cached_hi)
                interpolate_into(idx_hi, true);

            const dtype *pVrl = p_Vr_lo->memptr(), *pVil = p_Vi_lo->memptr();
            const dtype *pHrl = p_Hr_lo->memptr(), *pHil = p_Hi_lo->memptr();
            const dtype *pVrh = p_Vr_hi->memptr(), *pVih = p_Vi_hi->memptr();
            const dtype *pHrh = p_Hr_hi->memptr(), *pHih = p_Hi_hi->memptr();
            dtype *pVro = V_re.slice_memptr(fi), *pVio = V_im.slice_memptr(fi);
            dtype *pHro = H_re.slice_memptr(fi), *pHio = H_im.slice_memptr(fi);

            for (arma::uword j = 0; j < n_vals; ++j)
            {
                slerp_complex_mf(pVrl[j], pVil[j], pVrh[j], pVih[j], w, pVro[j], pVio[j]);
                slerp_complex_mf(pHrl[j], pHil[j], pHrh[j], pHih[j], w, pHro[j], pHio[j]);
            }
        }
    }
}

template void qd_arrayant_interpolate_dispatch(const quadriga_lib::arrayant<float> &ant,
                                               const arma::Mat<float> &azimuth, const arma::Mat<float> &elevation,
                                               const arma::uvec &i_element, const arma::Cube<float> &orientation,
                                               const arma::Mat<float> &element_pos,
                                               arma::Mat<float> &V_re, arma::Mat<float> &V_im, arma::Mat<float> &H_re, arma::Mat<float> &H_im,
                                               arma::Mat<float> *dist, arma::Mat<float> *azimuth_loc, arma::Mat<float> *elevation_loc,
                                               arma::Mat<float> *gamma, int kernel);

template void qd_arrayant_interpolate_dispatch(const quadriga_lib::arrayant<double> &ant,
                                               const arma::Mat<double> &azimuth, const arma::Mat<double> &elevation,
                                               const arma::uvec &i_element, const arma::Cube<double> &orientation,
                                               const arma::Mat<double> &element_pos,
                                               arma::Mat<double> &V_re, arma::Mat<double> &V_im, arma::Mat<double> &H_re, arma::Mat<double> &H_im,
                                               arma::Mat<double> *dist, arma::Mat<double> *azimuth_loc, arma::Mat<double> *elevation_loc,
                                               arma::Mat<double> *gamma, int kernel);

template void quadriga_lib::arrayant_interpolate(const arrayant<float> &ant, const arma::Mat<float> &azimuth, const arma::Mat<float> &elevation,
                                                 arma::Mat<float> &V_re, arma::Mat<float> &V_im, arma::Mat<float> &H_re, arma::Mat<float> &H_im,
                                                 const arma::uvec *i_element, const arma::Cube<float> *orientation, const arma::Mat<float> *element_pos,
                                                 arma::Mat<float> *dist, arma::Mat<float> *azimuth_loc, arma::Mat<float> *elevation_loc, arma::Mat<float> *gamma,
                                                 int use_kernel);

template void quadriga_lib::arrayant_interpolate(const arrayant<double> &ant, const arma::Mat<double> &azimuth, const arma::Mat<double> &elevation,
                                                 arma::Mat<double> &V_re, arma::Mat<double> &V_im, arma::Mat<double> &H_re, arma::Mat<double> &H_im,
                                                 const arma::uvec *i_element, const arma::Cube<double> *orientation, const arma::Mat<double> *element_pos,
                                                 arma::Mat<double> *dist, arma::Mat<double> *azimuth_loc, arma::Mat<double> *elevation_loc, arma::Mat<double> *gamma,
                                                 int use_kernel);

template void quadriga_lib::arrayant_interpolate_multi(const std::vector<arrayant<float>> &arrayant_vec, const arma::Mat<float> &azimuth, const arma::Mat<float> &elevation,
                                                       arma::Cube<float> &V_re, arma::Cube<float> &V_im, arma::Cube<float> &H_re, arma::Cube<float> &H_im,
                                                       const arma::Col<float> *frequency, const arma::uvec *i_element,
                                                       const arma::Cube<float> *orientation, const arma::Mat<float> *element_pos, int use_kernel);

template void quadriga_lib::arrayant_interpolate_multi(const std::vector<arrayant<double>> &arrayant_vec, const arma::Mat<double> &azimuth, const arma::Mat<double> &elevation,
                                                       arma::Cube<double> &V_re, arma::Cube<double> &V_im, arma::Cube<double> &H_re, arma::Cube<double> &H_im,
                                                       const arma::Col<double> *frequency, const arma::uvec *i_element,
                                                       const arma::Cube<double> *orientation, const arma::Mat<double> *element_pos, int use_kernel);
