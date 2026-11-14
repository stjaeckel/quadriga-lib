// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <stdexcept>
#include <cstring>   // std::memcpy
#include <iomanip>   // std::setprecision
#include <algorithm> // std::replace
#include <fstream>
#include <limits> // std::numeric_limits

#include "quadriga_arrayant.hpp"
#include "quadriga_tools.hpp"
#include "quadriga_math.hpp"
#include "qd_arrayant_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

// Beamwidth and pointing angles of one element in degree (main lobe search on interpolated cuts); valid arrayant
template <typename dtype>
void qd_arrayant_calc_beamwidth_element(const quadriga_lib::arrayant<dtype> &ant, arma::uword i_element,
                                               dtype threshold_dB, dtype *beamwidth_az, dtype *beamwidth_el,
                                               dtype *az_point_ang, dtype *el_point_ang)
{
    const bool need_az = (beamwidth_az != nullptr) || (az_point_ang != nullptr);
    const bool need_el = (beamwidth_el != nullptr) || (el_point_ang != nullptr);
    if (!need_az && !need_el)
        return;

    const double rad2deg = 180.0 / arma::datum::pi;
    const double thres_lin = std::pow(10.0, -0.1 * double(threshold_dB));

    const arma::uword nEl = ant.e_theta_re.n_rows;
    const arma::uword nAz = ant.e_theta_re.n_cols;

    // Find (iEl, iAz) of the maximum on the coarse grid
    const dtype *p_tre = ant.e_theta_re.slice_memptr(i_element);
    const dtype *p_tim = ant.e_theta_im.slice_memptr(i_element);
    const dtype *p_pre = ant.e_phi_re.slice_memptr(i_element);
    const dtype *p_pim = ant.e_phi_im.slice_memptr(i_element);

    double p_max_coarse = -1.0;
    arma::uword i_max = 0;
    for (arma::uword i = 0; i < nEl * nAz; ++i)
    {
        double a = double(p_tre[i]), b = double(p_tim[i]);
        double c = double(p_pre[i]), d = double(p_pim[i]);
        double v = a * a + b * b + c * c + d * d;
        if (v > p_max_coarse)
        {
            p_max_coarse = v;
            i_max = i;
        }
    }
    const arma::uword iEl_max = i_max % nEl; // column-major: idx = iEl + iAz*nEl
    const arma::uword iAz_max = i_max / nEl;

    arma::uvec i_el_vec(1);
    i_el_vec(0) = i_element;

    // Width of the main lobe in samples of a cut: the samples above the threshold that are connected to the
    // maximum, searched outwards from the maximum (wrapping around for a circular cut). Returns n for a circular
    // cut that is above the threshold everywhere.
    auto lobe_width = [](const arma::Col<double> &P, arma::uword iM, double thres, bool circular) -> arma::uword
    {
        const arma::uword n = P.n_elem;
        arma::uword n_left = 0, n_right = 0;
        if (circular)
        {
            while (n_left + n_right + 1 < n && P(iM >= n_left + 1 ? iM - n_left - 1 : iM + n - n_left - 1) > thres)
                ++n_left;
            while (n_left + n_right + 1 < n && P((iM + n_right + 1) % n) > thres)
                ++n_right;
            if (n_left + n_right + 1 == n && P(iM) > thres)
                return n;
        }
        else
        {
            while (iM >= n_left + 1 && P(iM - n_left - 1) > thres)
                ++n_left;
            while (iM + n_right + 1 < n && P(iM + n_right + 1) > thres)
                ++n_right;
        }
        return n_left + n_right;
    };

    // Power of the interpolated cut and the index of its maximum
    auto cut_power = [&](const arma::Mat<dtype> &az_q, const arma::Mat<dtype> &el_q, arma::Col<double> &P) -> arma::uword
    {
        arma::Mat<dtype> V_re, V_im, H_re, H_im;
        quadriga_lib::arrayant_interpolate<dtype>(ant, az_q, el_q, V_re, V_im, H_re, H_im, &i_el_vec,
                                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
        P.set_size(az_q.n_elem);
        double P_max = 0.0;
        arma::uword iM = 0;
        for (arma::uword i = 0; i < az_q.n_elem; ++i)
        {
            double a = double(V_re(0, i)), b = double(V_im(0, i));
            double c = double(H_re(0, i)), d = double(H_im(0, i));
            double v = a * a + b * b + c * c + d * d;
            P(i) = v;
            if (v > P_max)
                P_max = v, iM = i;
        }
        return iM;
    };

    const double pi = arma::datum::pi, tau = 2.0 * pi, pi_half = 0.5 * pi, tol = 1.0e-4;

    // Azimuth pass: cut at the elevation of the maximum. When the grid covers the full circle (the gap between the
    // last and the first grid point is not larger than the largest grid step), the cut is circular.
    if (need_az)
    {
        const double az_first = double(ant.azimuth_grid(0));
        const double az_last = double(ant.azimuth_grid(nAz - 1));
        double step_max = 0.0;
        for (arma::uword i = 1; i < nAz; ++i)
            step_max = std::max(step_max, double(ant.azimuth_grid(i) - ant.azimuth_grid(i - 1)));
        const bool circular = nAz == 1 || az_first + tau - az_last <= step_max + tol;

        const arma::uword nAzi = circular ? 100 * nAz : 100 * nAz + 1;
        const double az_step = circular ? tau / double(nAzi) : (az_last - az_first) / double(100 * nAz);

        arma::Mat<dtype> az_q(1, nAzi), el_q(1, nAzi);
        for (arma::uword i = 0; i < nAzi; ++i)
        {
            double az = az_first + double(i) * az_step;
            az_q(0, i) = dtype(az > pi ? az - tau : az);
        }
        if (!circular)
            az_q(0, nAzi - 1) = dtype(az_last);
        el_q.fill(ant.elevation_grid(iEl_max));

        arma::Col<double> Pi;
        const arma::uword iM = cut_power(az_q, el_q, Pi);
        const arma::uword width = lobe_width(Pi, iM, Pi(iM) * thres_lin, circular);

        if (beamwidth_az != nullptr)
            *beamwidth_az = dtype(width == nAzi ? 360.0 : double(width) * az_step * rad2deg);
        if (az_point_ang != nullptr)
            *az_point_ang = dtype(double(az_q(0, iM)) * rad2deg);
    }

    // Elevation pass: cut along the great circle through the maximum (azimuth of the maximum and the opposite
    // azimuth), so that a main lobe can extend across the poles. Without samples at a pole the cut ends there.
    // The result is at most 180 degree.
    if (need_el)
    {
        const double el_first = double(ant.elevation_grid(0));
        const double el_last = double(ant.elevation_grid(nEl - 1));
        const bool full_el = el_first <= -pi_half + tol && el_last >= pi_half - tol;

        const double az_front = double(ant.azimuth_grid(iAz_max));
        const double az_back = az_front > 0.0 ? az_front - pi : az_front + pi;

        arma::uword nEli;
        double el_step;
        if (full_el) // Full great circle, parameter t in [el_first, el_first + 2 pi)
            nEli = 200 * nEl, el_step = tau / double(nEli);
        else
            nEli = 100 * nEl + 1, el_step = (el_last - el_first) / double(100 * nEl);

        arma::Mat<dtype> az_q(1, nEli), el_q(1, nEli);
        for (arma::uword i = 0; i < nEli; ++i)
        {
            double t = el_first + double(i) * el_step;
            if (!full_el || t <= pi_half) // Front half
                az_q(0, i) = dtype(az_front), el_q(0, i) = dtype(t);
            else if (t <= 3.0 * pi_half) // Back half, across the north pole
                az_q(0, i) = dtype(az_back), el_q(0, i) = dtype(pi - t);
            else // Across the south pole, back to the front half
                az_q(0, i) = dtype(az_front), el_q(0, i) = dtype(t - tau);
        }
        if (!full_el)
            el_q(0, nEli - 1) = dtype(el_last);

        arma::Col<double> Pi;
        const arma::uword iM = cut_power(az_q, el_q, Pi);
        const arma::uword width = lobe_width(Pi, iM, Pi(iM) * thres_lin, full_el);

        if (beamwidth_el != nullptr)
            *beamwidth_el = dtype(std::min(double(width) * el_step * rad2deg, 180.0));
        if (el_point_ang != nullptr)
            *el_point_ang = dtype(double(el_q(0, iM)) * rad2deg);
    }
}

// Directivity of one element in dBi; valid arrayant
template <typename dtype>
dtype qd_arrayant_calc_directivity_element(const quadriga_lib::arrayant<dtype> &ant, arma::uword i_element)
{
    // Constants
    double pi2 = 2.0 * arma::datum::pi, pi_half = 0.5 * arma::datum::pi;
    arma::uword naz = ant.azimuth_grid.n_elem, nel = ant.elevation_grid.n_elem;

    // Az and El grid as double
    arma::vec az = arma::conv_to<arma::vec>::from(ant.azimuth_grid);
    arma::vec el = arma::conv_to<arma::vec>::from(ant.elevation_grid);

    // Calculate the azimuth weights
    arma::vec waz(naz, arma::fill::none);
    double *ptr = waz.memptr();
    for (arma::uword i = 0; i < naz; ++i)
    {
        double x = i == 0 ? az.at(naz - 1) - pi2 : az.at(i - 1);
        double y = i == naz - 1 ? az.at(0) + pi2 : az.at(i + 1);
        ptr[i] = y - x;
    }

    // Calculate the elevation weights
    arma::vec wel(nel, arma::fill::none);
    ptr = wel.memptr();
    for (arma::uword i = 0; i < nel; ++i)
    {
        double x = i == 0 ? -pi_half : 0.5 * el.at(i - 1) + 0.5 * el.at(i);
        double y = i == nel - 1 ? pi_half : 0.5 * el.at(i) + 0.5 * el.at(i + 1);
        arma::vec tmp = arma::linspace<arma::vec>(x, y, 21);
        double val = arma::accu(arma::cos(tmp));
        ptr[i] = val * (y - x);
    }

    // Combine weights
    arma::mat W(nel, naz, arma::fill::none);
    ptr = W.memptr();
    double norm = 0.0;
    for (double *col = waz.begin(); col != waz.end(); ++col)
        for (double *row = wel.begin(); row != wel.end(); ++row)
            *ptr = *row * *col, norm += *ptr++;
    ptr = W.memptr();
    norm = 1.0 / norm;
    for (arma::uword i = 0; i < naz * nel; ++i)
        ptr[i] *= norm;

    // Calculate the directivity
    double p_sum = 0.0, p_max = 0.0;
    const dtype *p_theta_re = ant.e_theta_re.slice_memptr(i_element), *p_theta_im = ant.e_theta_im.slice_memptr(i_element);
    const dtype *p_phi_re = ant.e_phi_re.slice_memptr(i_element), *p_phi_im = ant.e_phi_im.slice_memptr(i_element);
    auto sum_power = [&](double s)
    {
        p_sum = 0.0, p_max = 0.0;
        for (arma::uword i = 0; i < naz * nel; ++i)
        {
            double a = s * double(p_theta_re[i]), b = s * double(p_theta_im[i]), c = s * double(p_phi_re[i]), d = s * double(p_phi_im[i]);
            double pow = a * a + b * b + c * c + d * d;
            p_max = pow > p_max ? pow : p_max;
            p_sum += pow * ptr[i];
        }
    };
    sum_power(1.0);
    if (std::isinf(p_max) || std::isinf(p_sum)) // Power above the double range
        sum_power(qd_inverse_max_field(p_theta_re, p_theta_im, p_phi_re, p_phi_im, naz * nel));

    // An all-zero element has no directivity; 0 dBi is returned
    double directivity = (p_max > 0.0 && p_sum > 0.0) ? 10.0 * std::log10(p_max / p_sum) : 0.0;
    return dtype(directivity);
}

template void qd_arrayant_calc_beamwidth_element(const quadriga_lib::arrayant<float> &, arma::uword, float, float *, float *, float *, float *);
template void qd_arrayant_calc_beamwidth_element(const quadriga_lib::arrayant<double> &, arma::uword, double, double *, double *, double *, double *);
template float qd_arrayant_calc_directivity_element(const quadriga_lib::arrayant<float> &, arma::uword);
template double qd_arrayant_calc_directivity_element(const quadriga_lib::arrayant<double> &, arma::uword);

/*!MD
# arrayant_calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in each direction than the antenna sampling grid)
- The width is that of the main lobe: starting at the maximum, the search goes outwards until the power drops below the
  threshold; side lobes above the threshold are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360° when
  the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°
- Calculated per element, not per port; ignores element coupling

## Declaration:
```
void quadriga_lib::arrayant_calc_beamwidth(
    const arrayant<dtype> &ant,
    const arma::uvec *i_element = nullptr,
    dtype threshold_dB = 3.0,
    arma::Col<dtype> *beamwidth_az_deg = nullptr,
    arma::Col<dtype> *beamwidth_el_deg = nullptr,
    arma::Col<dtype> *az_point_ang_deg = nullptr,
    arma::Col<dtype> *el_point_ang_deg = nullptr);
```

## Inputs:
- **`ant`** — Valid arrayant
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; `nullptr` or empty: all elements; `[n_out]`
- **`threshold_dB`** *(optional)* — Threshold in dB, positive and finite; 3 dB = FWHM

## Outputs:
- **`beamwidth_az_deg`** *(optional)* — Azimuth beamwidth in degree; `[n_out]`
- **`beamwidth_el_deg`** *(optional)* — Elevation beamwidth in degree; `[n_out]`
- **`az_point_ang_deg`** *(optional)* — Azimuth pointing angle of the main beam in degree; `[n_out]`
- **`el_point_ang_deg`** *(optional)* — Elevation pointing angle of the main beam in degree; `[n_out]`

## Example:
```
auto ant = quadriga_lib::generate_arrayant_custom<double>(30.0, 20.0);
arma::vec bw_az, bw_el;
quadriga_lib::arrayant_calc_beamwidth(ant, nullptr, 3.0, &bw_az, &bw_el); // bw_az(0) = 30, bw_el(0) = 20
```

## See also:
- [[arrayant_calc_beamwidth_multi]] (multi-freq counterpart)
- [[arrayant_calc_directivity]] (directivity in dBi of array antenna elements)
- [[arrayant_combine_pattern]] (apply element coupling before calculating the beamwidth)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_calc_beamwidth(const arrayant<dtype> &ant, const arma::uvec *i_element, dtype threshold_dB,
                                           arma::Col<dtype> *beamwidth_az_deg, arma::Col<dtype> *beamwidth_el_deg,
                                           arma::Col<dtype> *az_point_ang_deg, arma::Col<dtype> *el_point_ang_deg)
{
    const std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (!error_message.empty())
        throw std::invalid_argument(error_message);
    if (!std::isfinite(threshold_dB) || threshold_dB <= dtype(0.0))
        throw std::invalid_argument("Input 'threshold_dB' must be positive and finite.");

    const arma::Col<dtype> *outs[] = {beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg};
    const char *names[] = {"beamwidth_az_deg", "beamwidth_el_deg", "az_point_ang_deg", "el_point_ang_deg"};
    qd_check_alias_list(outs, names, i_element);
    qd_check_alias_arrayant(outs, names, ant);

    const arma::uvec i_out = qd_arrayant_calc_element_list(i_element, ant.e_theta_re.n_slices);
    const arma::uword n_out = i_out.n_elem;
    for (arma::Col<dtype> *out : {beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg})
        if (out != nullptr)
            out->set_size(n_out);

    for (arma::uword i = 0; i < n_out; ++i)
        qd_arrayant_calc_beamwidth_element(ant, i_out[i], threshold_dB,
                                           beamwidth_az_deg == nullptr ? nullptr : beamwidth_az_deg->memptr() + i,
                                           beamwidth_el_deg == nullptr ? nullptr : beamwidth_el_deg->memptr() + i,
                                           az_point_ang_deg == nullptr ? nullptr : az_point_ang_deg->memptr() + i,
                                           el_point_ang_deg == nullptr ? nullptr : el_point_ang_deg->memptr() + i);
}

template void quadriga_lib::arrayant_calc_beamwidth(const arrayant<float> &, const arma::uvec *, float, arma::Col<float> *,
                                                    arma::Col<float> *, arma::Col<float> *, arma::Col<float> *);
template void quadriga_lib::arrayant_calc_beamwidth(const arrayant<double> &, const arma::uvec *, double, arma::Col<double> *,
                                                    arma::Col<double> *, arma::Col<double> *, arma::Col<double> *);

/*!MD
# arrayant_calc_directivity
Calculate the directivity in dBi of array antenna elements

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi
- Calculated per element, not per port; ignores element coupling

## Declaration:
```
void quadriga_lib::arrayant_calc_directivity(
    const arrayant<dtype> &ant,
    arma::Col<dtype> &directivity_dBi,
    const arma::uvec *i_element = nullptr);
```

## Inputs:
- **`ant`** — Valid arrayant
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; `nullptr` or empty: all elements; `[n_out]`

## Outputs:
- **`directivity_dBi`** — Directivity in dBi; `[n_out]`

## Example:
```
auto ant = quadriga_lib::generate_arrayant_dipole<double>();
arma::vec directivity_dBi;
quadriga_lib::arrayant_calc_directivity(ant, directivity_dBi); // about 1.76 dBi
```

## See also:
- [[arrayant_calc_directivity_multi]] (multi-freq counterpart)
- [[arrayant_calc_beamwidth]] (beamwidth and pointing angles of array antenna elements)
- [[arrayant_combine_pattern]] (apply element coupling before calculating the directivity)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_calc_directivity(const arrayant<dtype> &ant, arma::Col<dtype> &directivity_dBi,
                                             const arma::uvec *i_element)
{
    const std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (!error_message.empty())
        throw std::invalid_argument(error_message);

    const arma::Col<dtype> *outs[] = {&directivity_dBi};
    const char *names[] = {"directivity_dBi"};
    qd_check_alias_list(outs, names, i_element);
    qd_check_alias_arrayant(outs, names, ant);

    const arma::uvec i_out = qd_arrayant_calc_element_list(i_element, ant.e_theta_re.n_slices);
    directivity_dBi.set_size(i_out.n_elem);
    for (arma::uword i = 0; i < i_out.n_elem; ++i)
        directivity_dBi[i] = qd_arrayant_calc_directivity_element(ant, i_out[i]);
}

template void quadriga_lib::arrayant_calc_directivity(const arrayant<float> &, arma::Col<float> &, const arma::uvec *);
template void quadriga_lib::arrayant_calc_directivity(const arrayant<double> &, arma::Col<double> &, const arma::uvec *);

/*!MD
# arrayant_combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` to produce one output element per port (column) of the coupling matrix
- The element positions add the phase `-2 pi f d / propagation_speed` with `f = center_frequency` and `d` the distance
  of the element from the wavefront plane through the origin; the default speed of light applies to radio waves, use
  about 343 m/s for acoustic arrays (e.g. from [[generate_speaker]])
- Empty `element_pos` counts as zeros, empty coupling as identity
- Useful for beamforming and MIMO channel computation speedup

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant_combine_pattern(
    const arrayant<dtype> &ant,
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr,
    dtype propagation_speed = 299792458.0);
```

## Inputs:
- **`ant`** — Valid arrayant
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to input grid
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to input grid
- **`propagation_speed`** *(optional)* — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics

## Returns:
- New `arrayant` with `n_ports` elements (= number of columns in `coupling_re/im`), each holding the combined effective pattern for that port

## See also:
- [[arrayant_combine_pattern_multi]] (multi-freq counterpart)
- [[arrayant_interpolate]] (used internally to compute effective radiation patterns)
- [[arrayant_rotate_pattern]] (useful for orienting array antenna patterns)
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant_combine_pattern(const arrayant<dtype> &ant,
                                                                     const arma::Col<dtype> *azimuth_grid_new,
                                                                     const arma::Col<dtype> *elevation_grid_new,
                                                                     dtype propagation_speed)
{
    // Check if the arrayant object is valid
    std::string error_message = qd_arrayant_check(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (!(propagation_speed > dtype(0.0)) || !std::isfinite(propagation_speed))
        throw std::invalid_argument("Input 'propagation_speed' must be positive and finite.");

    // Get output angular grid
    bool new_azgrid = azimuth_grid_new != nullptr && azimuth_grid_new->n_elem != 0;
    bool new_elgrid = elevation_grid_new != nullptr && elevation_grid_new->n_elem != 0;

    if (new_azgrid && !qd_in_range(azimuth_grid_new->memptr(), azimuth_grid_new->n_elem, dtype(-3.1415930), dtype(3.1415930), true, true))
        throw std::invalid_argument("Values of 'azimuth_grid_new' must be sorted and in between -pi and pi (equivalent to -180 to 180 degree).");

    if (new_elgrid && !qd_in_range(elevation_grid_new->memptr(), elevation_grid_new->n_elem, dtype(-1.5707965), dtype(1.5707965), true, true))
        throw std::invalid_argument("Values of 'elevation_grid_new' must be sorted and in between -pi/2 and pi/2 (equivalent to -90 to 90 degree).");

    const dtype *p_azimuth_grid = (new_azgrid) ? azimuth_grid_new->memptr() : ant.azimuth_grid.memptr();
    const dtype *p_elevation_grid = (new_elgrid) ? elevation_grid_new->memptr() : ant.elevation_grid.memptr();

    arma::uword n_azimuth_out = (new_azgrid) ? azimuth_grid_new->n_elem : ant.azimuth_grid.n_elem;
    arma::uword n_elevation_out = (new_elgrid) ? elevation_grid_new->n_elem : ant.elevation_grid.n_elem;
    arma::uword n_ang = n_azimuth_out * n_elevation_out;

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    {
        dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr();
        for (arma::uword ia = 0ULL; ia < n_azimuth_out; ++ia)
            for (arma::uword ie = 0ULL; ie < n_elevation_out; ++ie)
                *p_azimuth++ = p_azimuth_grid[ia], *p_elevation++ = p_elevation_grid[ie];
    }

    // Get element positions
    arma::uword n_elements = ant.e_theta_re.n_slices;
    arma::Mat<dtype> element_pos_empty(3, n_elements, arma::fill::zeros);
    const auto p_element_pos = ant.element_pos.empty() ? &element_pos_empty : &ant.element_pos;

    // Interpolate the pattern data
    const arma::uvec i_element = arma::regspace<arma::uvec>(0, n_elements - 1);
    const arma::Cube<dtype> orientation(3, 1, 1, arma::fill::zeros);
    arma::Mat<dtype> V_re(n_elements, n_ang), V_im(n_elements, n_ang), H_re(n_elements, n_ang), H_im(n_elements, n_ang), dist(n_elements, n_ang);

    qd_arrayant_interpolate_dispatch<dtype>(ant, azimuth, elevation, i_element, orientation, *p_element_pos,
                                            V_re, V_im, H_re, H_im, &dist, nullptr, nullptr, nullptr, 1);

    // Apply phase shift caused by element positions
    double lambda = (double)propagation_speed / (double)ant.center_frequency;
    dtype wave_no = dtype(2.0 * arma::datum::pi / lambda);
    arma::Mat<std::complex<dtype>> phase(arma::cos(wave_no * dist), arma::sin(-wave_no * dist));
    arma::Mat<std::complex<dtype>> Vi(V_re, V_im), Hi(H_re, H_im);
    Vi = Vi % phase, Hi = Hi % phase;

    // Apply coupling
    arma::uword n_ports_out = ant.coupling_re.empty() ? n_elements : ant.coupling_re.n_cols;
    arma::Mat<std::complex<dtype>> Vo(n_ang, n_ports_out), Ho(n_ang, n_ports_out);
    if (!ant.coupling_re.empty())
    {
        arma::Mat<dtype> coupling_im_empty(n_elements, n_ports_out, arma::fill::zeros);
        const auto p_coupling_im = ant.coupling_im.empty() ? &coupling_im_empty : &ant.coupling_im;
        arma::Mat<std::complex<dtype>> coupling(ant.coupling_re, *p_coupling_im);

        for (arma::uword i = 0ULL; i < n_elements; ++i)
        {
            arma::Col<std::complex<dtype>> vi = Vi.row(i).as_col(), hi = Hi.row(i).as_col();
            for (arma::uword o = 0ULL; o < n_ports_out; ++o)
            {
                std::complex<dtype> cpl = coupling.at(i, o);
                Vo.col(o) += vi * cpl, Ho.col(o) += hi * cpl;
            }
        }
    }
    else // Identity coupling: one output per element, the angles along the rows
        Vo = Vi.st(), Ho = Hi.st();

    // Write output data
    quadriga_lib::arrayant<dtype> output;
    {
        output.set_size(n_elevation_out, n_azimuth_out, n_ports_out, n_ports_out);

        std::memcpy(output.azimuth_grid.memptr(), p_azimuth_grid, n_azimuth_out * sizeof(dtype));
        std::memcpy(output.elevation_grid.memptr(), p_elevation_grid, n_elevation_out * sizeof(dtype));

        arma::Mat<dtype> cpy = arma::real(Vo);
        arma::uword n_bytes = n_elevation_out * n_azimuth_out * n_ports_out * sizeof(dtype);
        std::memcpy(output.e_theta_re.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::imag(Vo);
        std::memcpy(output.e_theta_im.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::real(Ho);
        std::memcpy(output.e_phi_re.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::imag(Ho);
        std::memcpy(output.e_phi_im.memptr(), cpy.memptr(), n_bytes);

        output.element_pos.zeros();
        output.coupling_re.eye();
        output.coupling_im.zeros();
        output.center_frequency = ant.center_frequency;
        output.name = ant.name;

        qd_arrayant_mark_valid(output);
    }

    return output;
}

template quadriga_lib::arrayant<float> quadriga_lib::arrayant_combine_pattern(const arrayant<float> &, const arma::Col<float> *, const arma::Col<float> *, float);
template quadriga_lib::arrayant<double> quadriga_lib::arrayant_combine_pattern(const arrayant<double> &, const arma::Col<double> *, const arma::Col<double> *, double);

/*!MD
# arrayant_concat
Concatenate the elements of two arrayants into a new arrayant

- Both arrays must share identical sampling grids; throws otherwise
- Coupling is block-diagonal (see [[arrayant_concat_multi]] diagram); name and center_frequency are taken from `ant_1`
- An empty `element_pos` counts as zeros, an empty `coupling_re` as identity (one port per element), an empty
  `coupling_im` as zeros

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant_concat(
    const arrayant<dtype> &ant_1,
    const arrayant<dtype> &ant_2);
```

## Inputs:
- **`ant_1`** — First arrayant; its elements come first in the output
- **`ant_2`** — Second arrayant; sampling grids must match `ant_1`

## Returns:
- New `arrayant` containing all elements of both arrays

## See also:
- [[arrayant_concat_multi]] (multi-freq counterpart)
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant_concat(const arrayant<dtype> &ant_1, const arrayant<dtype> &ant_2)
{
    // Check if arrayant objects are valid
    std::string error_message = qd_arrayant_is_valid_quick(ant_1);
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());
    error_message = qd_arrayant_is_valid_quick(ant_2);
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Check if grids are identical
    if (!arma::approx_equal(ant_1.azimuth_grid, ant_2.azimuth_grid, "absdiff", (dtype)1.0e-6))
        throw std::invalid_argument("Azimuth sampling grids don't match.");
    if (!arma::approx_equal(ant_1.elevation_grid, ant_2.elevation_grid, "absdiff", (dtype)1.0e-6))
        throw std::invalid_argument("Elevation sampling grids don't match.");

    arma::uword n_elevation = ant_1.n_elevation();
    arma::uword n_azimuth = ant_1.n_azimuth();
    arma::uword n_elements_1 = ant_1.n_elements();
    arma::uword n_elements_2 = ant_2.n_elements();
    arma::uword n_ports_1 = ant_1.n_ports();
    arma::uword n_ports_2 = ant_2.n_ports();
    arma::uword n_ang = n_azimuth * n_elevation;

    quadriga_lib::arrayant<dtype> output;
    output.set_size(n_elevation, n_azimuth, n_elements_1 + n_elements_2, n_ports_1 + n_ports_2);

    // Copy data from first antenna
    std::memcpy(output.azimuth_grid.memptr(), ant_1.azimuth_grid.memptr(), n_azimuth * sizeof(dtype));
    std::memcpy(output.elevation_grid.memptr(), ant_1.elevation_grid.memptr(), n_elevation * sizeof(dtype));
    std::memcpy(output.e_theta_re.slice_memptr(0), ant_1.e_theta_re.memptr(), n_ang * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_theta_im.slice_memptr(0), ant_1.e_theta_im.memptr(), n_ang * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_phi_re.slice_memptr(0), ant_1.e_phi_re.memptr(), n_ang * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_phi_im.slice_memptr(0), ant_1.e_phi_im.memptr(), n_ang * n_elements_1 * sizeof(dtype));
    if (!ant_1.element_pos.empty())
        std::memcpy(output.element_pos.colptr(0), ant_1.element_pos.memptr(), 3 * n_elements_1 * sizeof(dtype));

    // Inherit scalar/string metadata from ant_1
    output.center_frequency = ant_1.center_frequency;
    output.name = ant_1.name;

    // Copy data from second antenna
    std::memcpy(output.e_theta_re.slice_memptr(n_elements_1), ant_2.e_theta_re.memptr(), n_ang * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_theta_im.slice_memptr(n_elements_1), ant_2.e_theta_im.memptr(), n_ang * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_phi_re.slice_memptr(n_elements_1), ant_2.e_phi_re.memptr(), n_ang * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_phi_im.slice_memptr(n_elements_1), ant_2.e_phi_im.memptr(), n_ang * n_elements_2 * sizeof(dtype));
    if (!ant_2.element_pos.empty())
        std::memcpy(output.element_pos.colptr(n_elements_1), ant_2.element_pos.memptr(), 3 * n_elements_2 * sizeof(dtype));

    // Block-diagonal coupling (empty coupling_re: identity block, empty coupling_im: zeros)
    qd_arrayant_coupling_block_diag(ant_1.coupling_re, ant_1.coupling_im, n_elements_1,
                                    ant_2.coupling_re, ant_2.coupling_im, n_elements_2,
                                    output.coupling_re, output.coupling_im);

    return output;
}

template quadriga_lib::arrayant<float> quadriga_lib::arrayant_concat(const arrayant<float> &, const arrayant<float> &);
template quadriga_lib::arrayant<double> quadriga_lib::arrayant_concat(const arrayant<double> &, const arrayant<double> &);

/*!MD
# arrayant_copy_element
Copy a single antenna element to one or more destination slots

- Works in place; the array is enlarged if any destination index exceeds the current number of elements
- Added elements get their own port with identity coupling; positions of added elements that are not copied are zero
- When the array grows, an empty `element_pos` and an empty coupling are written out first (zeros, identity), so
  that all elements can be edited afterwards; otherwise empty members stay empty
- Throws `std::invalid_argument` for a read-only object, an empty `destination`, a source index out of bound, or a
  destination index equal to the largest `arma::uword` (e.g. `-1` cast to unsigned)

## Declaration:
```
void quadriga_lib::arrayant_copy_element(
    arrayant<dtype> &ant,
    arma::uword source,
    const arma::uvec &destination);
```

## Inputs:
- **`ant`** — Valid arrayant; updated in-place
- **`source`** — Index of the element to copy, 0-based
- **`destination`** — Target indices, 0-based; a single index is passed as `{i}`

## Example:
```
auto ant = quadriga_lib::generate_arrayant_dipole<double>();
quadriga_lib::arrayant_copy_element(ant, 0, {1, 2, 3}); // 4 identical elements
ant.element_pos.row(1) = arma::rowvec({-0.75, -0.25, 0.25, 0.75});
```

## See also:
- [[arrayant_copy_element_multi]] (multi-freq counterpart)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_copy_element(arrayant<dtype> &ant, arma::uword source, const arma::uvec &destination)
{
    if (ant.read_only)
        throw std::invalid_argument("Cannot copy elements of a read-only array antenna object.");

    if (destination.n_elem == 0)
        throw std::invalid_argument("Input 'destination' is empty.");

    // The largest index cannot be reached by resizing: index + 1 would wrap to 0
    if (destination.max() == std::numeric_limits<arma::uword>::max())
        throw std::invalid_argument("Destination index out of bound.");

    // Check if arrayant object is valid
    std::string error_message = qd_arrayant_check(ant); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    arma::uword n_el = ant.e_theta_re.n_rows;
    arma::uword n_az = ant.e_theta_re.n_cols;
    arma::uword n_ang = n_el * n_az;
    arma::uword n_elements = ant.e_theta_re.n_slices;
    arma::uword n_element_max = destination.max() + 1;

    if (source >= n_elements)
        throw std::invalid_argument("Source index out of bound");

    // Enlarge the arrayant; empty positions and coupling are written out first (zeros, identity), so that the new
    // elements can be edited; resize keeps the values and sets new ones to zero
    if (n_element_max > n_elements)
    {
        if (ant.element_pos.empty())
            ant.element_pos.zeros(3, n_elements);
        if (ant.coupling_re.empty())
            ant.coupling_re.eye(n_elements, n_elements);
        if (ant.coupling_im.empty())
            ant.coupling_im.zeros(arma::size(ant.coupling_re));

        arma::uword added_elements = n_element_max - n_elements;
        arma::uword n_ports = ant.coupling_re.n_cols;
        ant.e_theta_re.resize(n_el, n_az, n_element_max);
        ant.e_theta_im.resize(n_el, n_az, n_element_max);
        ant.e_phi_re.resize(n_el, n_az, n_element_max);
        ant.e_phi_im.resize(n_el, n_az, n_element_max);
        ant.element_pos.resize(3, n_element_max);
        ant.coupling_re.resize(n_element_max, n_ports + added_elements);
        ant.coupling_im.resize(n_element_max, n_ports + added_elements);
        for (arma::uword i = 0; i < added_elements; ++i)
            ant.coupling_re.at(n_elements + i, n_ports + i) = (dtype)1.0;
    }

    // Copy data from source to destination
    for (auto dest = destination.begin(); dest != destination.end(); ++dest)
        if (source != *dest)
        {
            std::memcpy(ant.e_theta_re.slice_memptr(*dest), ant.e_theta_re.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(ant.e_theta_im.slice_memptr(*dest), ant.e_theta_im.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(ant.e_phi_re.slice_memptr(*dest), ant.e_phi_re.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(ant.e_phi_im.slice_memptr(*dest), ant.e_phi_im.slice_memptr(source), n_ang * sizeof(dtype));
            if (!ant.element_pos.empty()) // Empty only when the array did not grow
                std::memcpy(ant.element_pos.colptr(*dest), ant.element_pos.colptr(source), 3 * sizeof(dtype));
        }

    qd_arrayant_mark_valid(ant);
}

template void quadriga_lib::arrayant_copy_element(arrayant<float> &, arma::uword, const arma::uvec &);
template void quadriga_lib::arrayant_copy_element(arrayant<double> &, arma::uword, const arma::uvec &);

/*!MD
# arrayant_export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives finer mesh
- Each element is placed at its position in `element_pos` (empty: at the origin)

## Declaration:
```
void quadriga_lib::arrayant_export_obj_file(
    const std::string &fn,
    const arrayant<dtype> &ant,
    dtype directivity_range = 30.0,
    const std::string &colormap = "jet",
    dtype object_radius = 1.0,
    arma::uword icosphere_n_div = 4,
    const arma::uvec *i_element = nullptr);
```

## Inputs:
- **`fn`** — Output OBJ filename; must end in .obj; the colormap is written to a .mtl file of the same name
- **`ant`** — Valid arrayant
- **`directivity_range`** *(optional)* — Dynamic range of the visualized directivity pattern in dB
- **`colormap`** *(optional)* — Colormap name; see [[colormap]] for supported options
- **`object_radius`** *(optional)* — Radius of the exported object
- **`icosphere_n_div`** *(optional)* — Icosphere subdivision count; higher = finer mesh, see [[icosphere]]
- **`i_element`** *(optional)* — 0-based element indices to export; `nullptr` or empty exports all elements

## See also:
- [[colormap]] (Used for setting the colormap)
- [[icosphere]] (Used internally to generate icosphere primitive)
- .[[write_paths_to_obj_file]] (function of Channel class to export propagation paths to OBJ 3D visualization)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_export_obj_file(const std::string &fn, const arrayant<dtype> &ant, dtype directivity_range,
                                            const std::string &colormap, dtype object_radius,
                                            arma::uword icosphere_n_div, const arma::uvec *i_element_in)
{
    // Check if arrayant object is valid
    std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Input validation
    std::string fn_suffix = ".obj";
    std::string fn_mtl;

    if (fn.size() >= fn_suffix.size() &&
        fn.compare(fn.size() - fn_suffix.size(), fn_suffix.size(), fn_suffix) == 0)
    {
        fn_mtl = fn.substr(0, fn.size() - fn_suffix.size()) + ".mtl";
    }
    else
        throw std::invalid_argument("OBJ-File name must end with .obj");

    // Replace spaces in ant_name
    std::string ant_name = ant.name;
    std::replace(ant_name.begin(), ant_name.end(), ' ', '_');

    // Extract the file name from the path
    std::string fn_mtl_base;
    size_t pos = fn_mtl.find_last_of("/");
    if (pos != std::string::npos)
        fn_mtl_base = fn_mtl.substr(pos + 1);
    else
        fn_mtl_base = fn_mtl;

    double range = -std::abs((double)directivity_range);
    if (range == 0.0 || !std::isfinite(range))
        throw std::invalid_argument("Directivity range must be finite and not 0.");

    double radius = (double)object_radius;
    if (!(radius > 0.0) || !std::isfinite(radius))
        throw std::invalid_argument("Object radius must be finite and larger than 0.");

    arma::uword no_elements = ant.n_elements();

    const arma::uvec i_element = (i_element_in == nullptr || i_element_in->n_elem == 0)
                                     ? arma::regspace<arma::uvec>(0, no_elements - 1)
                                     : *i_element_in;

    if (arma::any(i_element >= no_elements))
        throw std::invalid_argument("Element indices 'i_element' cannot exceed the array antenna size.");

    size_t no_elements_t = (size_t)i_element.n_elem;

    // Colormap
    arma::uchar_mat cmap = quadriga_lib::colormap(colormap);
    size_t n_cmap = (size_t)cmap.n_rows;

    // Export colormap to material file
    std::ofstream outFile(fn_mtl);
    if (outFile.is_open())
    {
        // Write some text to the file
        outFile << "# QuaDRiGa " << "antenna pattern colormap\n\n";
        for (size_t i = 0; i < n_cmap; ++i)
        {
            double R = (double)cmap(i, 0) / 255.0;
            double G = (double)cmap(i, 1) / 255.0;
            double B = (double)cmap(i, 2) / 255.0;
            outFile << "newmtl QuaDRiGa_ANT_" << colormap << "_" << std::setfill('0') << std::setw(2) << i << "\n";
            outFile << std::fixed << std::setprecision(6) << "Kd " << R << " " << G << " " << B << "\n\n";
        }
        outFile.close();
    }

    // Generate vertices and faces
    arma::mat vertices_icosphere; // double precision
    arma::u32_mat faces_icosphere;
    size_t no_faces, no_vert;
    {
        arma::mat center, vert;
        no_faces = quadriga_lib::icosphere<double>(icosphere_n_div, 1.0, &center, nullptr, &vert);

        vertices_icosphere.set_size(3 * no_faces, 3);
        faces_icosphere.set_size(no_faces, 3);

        double *pc = center.memptr(), *pv = vert.memptr(), *v = vertices_icosphere.memptr();
        unsigned *f_ico = faces_icosphere.memptr();

        for (size_t i_face = 0; i_face < no_faces; ++i_face)
        {
            double x, y, z;

            x = pc[i_face] + pv[i_face];
            y = pc[i_face + no_faces] + pv[i_face + no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 2 * no_faces];

            v[3 * i_face] = x;
            v[3 * i_face + 3 * no_faces] = y;
            v[3 * i_face + 6 * no_faces] = z;
            f_ico[i_face] = 3 * (unsigned)i_face + 1;

            x = pc[i_face] + pv[i_face + 3 * no_faces];
            y = pc[i_face + no_faces] + pv[i_face + 4 * no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 5 * no_faces];

            v[3 * i_face + 1] = x;
            v[3 * i_face + 3 * no_faces + 1] = y;
            v[3 * i_face + 6 * no_faces + 1] = z;
            f_ico[i_face + no_faces] = 3 * (unsigned)i_face + 2;

            x = pc[i_face] + pv[i_face + 6 * no_faces];
            y = pc[i_face + no_faces] + pv[i_face + 7 * no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 8 * no_faces];

            v[3 * i_face + 2] = x;
            v[3 * i_face + 3 * no_faces + 2] = y;
            v[3 * i_face + 6 * no_faces + 2] = z;
            f_ico[i_face + 2 * no_faces] = 3 * (unsigned)i_face + 3;
        }
    }
    no_vert = 3 * no_faces;

    // Convert vertices to angles
    arma::Mat<dtype> azimuth, elevation;
    azimuth.set_size(1, no_vert);
    elevation.set_size(1, no_vert);
    {
        double *v = vertices_icosphere.memptr();
        dtype *az = azimuth.memptr(), *el = elevation.memptr();
        for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
        {
            double x = v[i_vert];
            double y = v[i_vert + no_vert];
            double z = v[i_vert + 2 * no_vert];
            x = x > 1.0 ? 1.0 : x, y = y > 1.0 ? 1.0 : y, z = z > 1.0 ? 1.0 : z;
            x = x < -1.0 ? -1.0 : x, y = y < -1.0 ? -1.0 : y, z = z < -1.0 ? -1.0 : z;
            az[i_vert] = (dtype)std::atan2(y, x);
            el[i_vert] = (dtype)std::asin(z);
        }
    }

    // Interpolate antenna patterns (for all elements in the array)
    arma::Mat<dtype> V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate<dtype>(ant, azimuth, elevation, V_re, V_im, H_re, H_im, &i_element,
                                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
    dtype *vr = V_re.memptr(), *vi = V_im.memptr(), *hr = H_re.memptr(), *hi = H_im.memptr();

    // Calculate maximum Power in the Pattern
    double max_pow = 0.0;
    bool has_nan = false;
    auto find_max_pow = [&]()
    {
        max_pow = 0.0, has_nan = false;
        for (size_t i = 0; i < no_vert * no_elements_t; ++i)
        {
            double a = (double)vr[i], b = (double)vi[i], c = (double)hr[i], d = (double)hi[i];
            double p = a * a + b * b + c * c + d * d;
            max_pow = (p > max_pow) ? p : max_pow;
            has_nan = has_nan || std::isnan(p);
        }
    };
    find_max_pow();
    if (std::isinf(max_pow)) // Power above the double range: rescale the interpolated fields
    {
        dtype s = (dtype)qd_inverse_max_field(vr, vi, hr, hi, no_vert * no_elements_t);
        V_re *= s, V_im *= s, H_re *= s, H_im *= s;
        find_max_pow();
    }
    if (has_nan) // NaN field values, or infinite ones (NaN after the rescaling)
        throw std::invalid_argument("Field values must be finite.");
    if (!(max_pow > 0.0))
        throw std::invalid_argument("Pattern is zero in all directions for the exported elements.");

    // Write OBJ File
    outFile = std::ofstream(fn);
    if (outFile.is_open())
    {
        outFile << "# QuaDRiGa " << "antenna pattern\n";
        outFile << "mtllib " << fn_mtl_base << "\n";

        // Face offset for previous element in the OBJ file
        unsigned face_offset = 1; // OBJ starts counting at 1

        // Export each element to the OBJ file
        for (size_t i_el = 0; i_el < no_elements_t; ++i_el)
        {
            arma::uword i_array_element = i_element(i_el);
            outFile << "\no " << "QuaDRiGa_ANT_" << ant_name << "_" << std::setfill('0') << std::setw(3) << i_array_element << "\n";

            // Calculate Shape
            arma::vec pow(no_vert);
            double *pp = pow.memptr();
            for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
            {
                size_t i = i_vert * no_elements_t + i_el;
                double a = (double)vr[i], b = (double)vi[i], c = (double)hr[i], d = (double)hi[i];
                double p = a * a + b * b + c * c + d * d;
                p /= max_pow;                // Scale by maximum power (division: no overflow for tiny powers)
                p = 10.0 * std::log10(p);    // Convert to dB
                p = (p < range) ? range : p; // Set minimum
                p = (p - range) / (-range);  // Set to range 0 ... 1 (division: no overflow for tiny ranges)
                pp[i_vert] = p;
            }

            // Scale vertices
            arma::mat scaled_vertices(no_vert, 3);
            double *v_scale = scaled_vertices.memptr();  // Scaled vertices
            double *v_ico = vertices_icosphere.memptr(); // Icosphere vertices
            for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
            {
                v_scale[i_vert] = v_ico[i_vert] * pp[i_vert];                             // x
                v_scale[i_vert + no_vert] = v_ico[i_vert + no_vert] * pp[i_vert];         // y
                v_scale[i_vert + 2 * no_vert] = v_ico[i_vert + 2 * no_vert] * pp[i_vert]; // z
            }

            // Process faces
            double eps = 0.001; // Co-location tolerance in [m], relative to 1 m radius
            eps *= eps;         // Squared value

            arma::mat final_vertices(no_vert, 3);
            arma::u32_mat vert_index(no_faces, 3);
            arma::u32_vec cmap_index(no_faces);

            double *v_final = final_vertices.memptr(); // Final vertices
            unsigned *f_final = vert_index.memptr();   // Final face indices
            unsigned *i_cmap = cmap_index.memptr();    // Colormap index

            size_t no_final_vert = 0; // Number of final vertices
            size_t no_final_face = 0; // Number of final faces

            for (size_t i_face = 0; i_face < no_faces; ++i_face)
            {
                // Load the 3 scaled vertices of the face
                double x0 = v_scale[3 * i_face];
                double y0 = v_scale[3 * i_face + no_vert];
                double z0 = v_scale[3 * i_face + 2 * no_vert];

                double x1 = v_scale[3 * i_face + 1];
                double y1 = v_scale[3 * i_face + no_vert + 1];
                double z1 = v_scale[3 * i_face + 2 * no_vert + 1];

                double x2 = v_scale[3 * i_face + 2];
                double y2 = v_scale[3 * i_face + no_vert + 2];
                double z2 = v_scale[3 * i_face + 2 * no_vert + 2];

                // Check if face vertices collapse to a point or line
                double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                double dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                dx = x2 - x0, dy = y2 - y0, dz = z2 - z0;
                dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
                dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                // Do for each of the 3 vertices
                for (size_t i3 = 0; i3 < 3; ++i3)
                {
                    double x = x0, y = y0, z = z0;
                    if (i3 == 1)
                        x = x1, y = y1, z = z1;
                    else if (i3 == 2)
                        x = x2, y = y2, z = z2;

                    // Add the vertices to the final list
                    size_t i_vert_final = 0; // Face index
                    bool found = false;

                    // Get vertex ID
                    for (size_t i = 0; i < no_final_vert; ++i) // Search existing vertex
                    {
                        dx = v_final[i] - x;
                        dy = v_final[i + no_vert] - y;
                        dz = v_final[i + 2 * no_vert] - z;
                        dd = dx * dx + dy * dy + dz * dz;
                        if (dd < eps)
                        {
                            i_vert_final = i;
                            found = true;
                            break;
                        }
                    }
                    if (!found) // Add new vertex to list
                    {
                        i_vert_final = no_final_vert;
                        v_final[i_vert_final] = x;
                        v_final[i_vert_final + no_vert] = y;
                        v_final[i_vert_final + 2 * no_vert] = z;
                        ++no_final_vert;
                    }

                    // Write face ID
                    f_final[no_final_face + i3 * no_faces] = (unsigned)i_vert_final;
                }

                // Get the colormap index
                double pf = pp[3 * i_face] + pp[3 * i_face + 1] + pp[3 * i_face + 2];
                pf *= 21.0; // 63 * 0.33
                pf = std::round(pf);
                unsigned c = (unsigned)pf;
                c = (c > 63) ? 63 : c;
                i_cmap[no_final_face] = c;

                ++no_final_face;
            }

            // Write vertices to file
            for (size_t i_vert = 0; i_vert < no_final_vert; ++i_vert)
            {
                double x = v_final[i_vert];
                double y = v_final[i_vert + no_vert];
                double z = v_final[i_vert + 2 * no_vert];

                x *= radius;
                y *= radius;
                z *= radius;

                if (ant.element_pos.n_elem != 0)
                {
                    x += ant.element_pos(0, i_array_element);
                    y += ant.element_pos(1, i_array_element);
                    z += ant.element_pos(2, i_array_element);
                }

                outFile << "v " << x << " " << y << " " << z << "\n";
            }

            // Write faces to file, ordered by color
            for (unsigned i_color = 0; i_color < 64; ++i_color)
            {
                bool color_not_used = true;
                for (size_t i_face = 0; i_face < no_final_face; ++i_face)
                {
                    if (i_cmap[i_face] == i_color)
                    {
                        if (color_not_used)
                        {
                            outFile << "usemtl QuaDRiGa_ANT_" << colormap << "_" << std::setfill('0') << std::setw(2) << i_color << "\n";
                            color_not_used = false;
                        }
                        outFile << "f " << f_final[i_face] + face_offset << " " << f_final[i_face + no_faces] + face_offset << " " << f_final[i_face + 2 * no_faces] + face_offset << "\n";
                    }
                }
            }
            face_offset += (unsigned)no_final_vert;
        }

        outFile.close();
    }
}

template void quadriga_lib::arrayant_export_obj_file(const std::string &, const arrayant<float> &, float, const std::string &, float, arma::uword, const arma::uvec *);
template void quadriga_lib::arrayant_export_obj_file(const std::string &, const arrayant<double> &, double, const std::string &, double, arma::uword, const arma::uvec *);

/*!MD
# arrayant_remove_zeros
Remove zero-valued entries from antenna pattern data, reducing its size

- Works in place on the sampling grid and the pattern cubes; positions, coupling, name and frequency are unchanged
- A column or row is removed if the total power summed over all field components and array elements does not exceed
  1e-12 times the peak power of a single sample (summed over field components and elements); the result does not
  depend on the scale of the pattern; an all-zero pattern is left unchanged
- Immediate neighbors of retained entries are also kept unless the local grid spacing is strongly asymmetric (factor > 1.5)
- To keep the original, call it on a `.copy()`

## Declaration:
```
void quadriga_lib::arrayant_remove_zeros(arrayant<dtype> &ant);
```

## Inputs:
- **`ant`** — Valid arrayant; updated in-place
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_remove_zeros(arrayant<dtype> &ant)
{
    // Check if arrayant object is valid
    std::string error_message = qd_arrayant_check(ant); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (ant.read_only)
        throw std::invalid_argument("Cannot update read-only array antenna object inplace.");

    unsigned long long n_el = ant.e_theta_re.n_rows;
    unsigned long long n_az = ant.e_theta_re.n_cols;
    unsigned long long n_slices = ant.e_theta_re.n_slices;

    // Calculate the power pattern (in double precision, so that low-level float patterns do not underflow)
    arma::mat pow(n_el, n_az, arma::fill::zeros);
    double *pw = pow.memptr();
    dtype pi_half = dtype(arma::datum::pi / 2.0);
    dtype two_pi = dtype(2.0 * arma::datum::pi);

    auto sum_power = [&](double s)
    {
        for (auto is = 0ULL; is < n_slices; ++is)
        {
            const dtype *pa = ant.e_theta_re.slice_memptr(is), *pb = ant.e_theta_im.slice_memptr(is),
                        *pc = ant.e_phi_re.slice_memptr(is), *pd = ant.e_phi_im.slice_memptr(is);
            for (auto j = 0ULL; j < n_el * n_az; ++j)
            {
                double a = s * double(pa[j]), b = s * double(pb[j]), c = s * double(pc[j]), d = s * double(pd[j]);
                pw[j] += a * a + b * b + c * c + d * d;
            }
        }
    };
    sum_power(1.0);

    // Threshold relative to the peak power
    double p_peak = pow.max();
    if (std::isinf(p_peak)) // Power above the double range
    {
        pow.zeros();
        sum_power(qd_inverse_max_field(ant.e_theta_re.memptr(), ant.e_theta_im.memptr(), ant.e_phi_re.memptr(),
                                       ant.e_phi_im.memptr(), ant.e_theta_re.n_elem));
        p_peak = pow.max();
    }
    const double limit = 1.0e-12 * p_peak;

    arma::rowvec az_sum = arma::sum(pow, 0);
    arma::vec el_sum = arma::sum(pow, 1);
    pow.reset();
    if (az_sum.has_nan()) // NaN or Inf in the field values
        throw std::invalid_argument("Field values must be finite.");

    // Determine which parts of the pattern should be kept; an all-zero pattern is kept unchanged
    arma::uvec keep_az(n_az, arma::fill::zeros), keep_el(n_el, arma::fill::zeros);
    if (!(p_peak > 0.0))
        keep_az.ones(), keep_el.ones();
    unsigned long long *keep_az_ptr = keep_az.memptr();
    unsigned long long *keep_el_ptr = keep_el.memptr();

    const double *ps = az_sum.memptr();
    const dtype *grid_ptr = ant.azimuth_grid.memptr();
    for (auto ia = 0ULL; ia < n_az; ++ia)
    {
        // Spacing to the previous and the next grid point; the first and the last point are neighbours across +/- 180 deg
        dtype p = ia == 0 ? *grid_ptr - grid_ptr[n_az - 1] + two_pi : grid_ptr[ia] - grid_ptr[ia - 1];
        dtype n = ia == n_az - 1 ? *grid_ptr + two_pi - grid_ptr[ia] : grid_ptr[ia + 1] - grid_ptr[ia];

        unsigned long long ip = ia == 0ULL ? n_az - 1ULL : ia - 1ULL;
        unsigned long long in = ia == n_az - 1ULL ? 0ULL : ia + 1ULL;

        if (ps[ia] > limit)
            keep_az_ptr[ip] = n > 0.001 * p && p > 1.5 * n ? keep_az_ptr[ip] : 1,
            keep_az_ptr[ia] = 1,
            keep_az_ptr[in] = p > 0.001 * n && n > 1.5 * p ? keep_az_ptr[in] : 1;
    }

    ps = el_sum.memptr(), grid_ptr = ant.elevation_grid.memptr();
    for (auto ie = 0ULL; ie < n_el; ++ie)
    {
        dtype p = ie == 0 ? *grid_ptr + pi_half : grid_ptr[ie] - grid_ptr[ie - 1];
        dtype n = ie == n_el - 1 ? pi_half - grid_ptr[ie] : grid_ptr[ie + 1] - grid_ptr[ie];

        unsigned long long ip = ie == 0ULL ? 0ULL : ie - 1ULL;
        unsigned long long in = ie == n_el - 1ULL ? n_el - 1ULL : ie + 1ULL;

        if (ps[ie] > limit)
            keep_el_ptr[ip] = n > 0.001 * p && p > 1.5 * n ? keep_el_ptr[ip] : 1ULL,
            keep_el_ptr[ie] = 1,
            keep_el_ptr[in] = p > 0.001 * n && n > 1.5 * p ? keep_el_ptr[in] : 1ULL;
    }

    // Copy the kept part of the grids and patterns
    const arma::uvec i_az = arma::find(keep_az), i_el = arma::find(keep_el);
    if (i_az.n_elem == n_az && i_el.n_elem == n_el)
        return;

    auto shrink = [&](arma::Cube<dtype> &c)
    {
        arma::Cube<dtype> tmp(i_el.n_elem, i_az.n_elem, n_slices, arma::fill::none);
        for (auto is = 0ULL; is < n_slices; ++is)
            tmp.slice(is) = c.slice(is).submat(i_el, i_az);
        c = std::move(tmp);
    };
    shrink(ant.e_theta_re), shrink(ant.e_theta_im), shrink(ant.e_phi_re), shrink(ant.e_phi_im);
    ant.azimuth_grid = arma::Col<dtype>(ant.azimuth_grid.elem(i_az));
    ant.elevation_grid = arma::Col<dtype>(ant.elevation_grid.elem(i_el));

    qd_arrayant_mark_valid(ant);
}

template void quadriga_lib::arrayant_remove_zeros(arrayant<float> &);
template void quadriga_lib::arrayant_remove_zeros(arrayant<double> &);

/*!MD
# arrayant_rotate_pattern
Rotate antenna radiation patterns using Euler rotations, in place

- Rotates the pattern and/or polarization by the orientation (bank, tilt, heading); the rotated pattern equals what
  [[arrayant_interpolate]] returns for the unrotated pattern with the same `orientation`
- A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
- Element positions are rotated with the pattern (usage 0, 1, 3 and 4); coupling, `center_frequency` and `name` are kept
- Usage 0 and 1 adjust the sampling grid for non-uniformly sampled antennas; this needs all elements, so a subset
  in `i_element` throws when the grid must be adjusted
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid spurious polarization effects

## Declaration:
```
void quadriga_lib::arrayant_rotate_pattern(
    arrayant<dtype> &ant,
    const arma::Mat<dtype> &orientation,
    unsigned usage = 0,
    const arma::uvec *i_element = nullptr);
```

## Inputs:
- **`ant`** — Valid arrayant; updated in place
- **`orientation`** — Euler angles (bank, tilt, heading) in rad, finite; `[3, 1]`
- **`usage`** *(optional)* — Rotation mode:<br><br>
   | Mode | Pattern | Polarization | Grid adjustment |
   | ---- | ------- | ------------ | --------------- |
   | 0    | Yes     | Yes          | Yes             |
   | 1    | Yes     | No           | Yes             |
   | 2    | No      | Yes          | No              |
   | 3    | Yes     | Yes          | No              |
   | 4    | Yes     | No           | No              |
- **`i_element`** *(optional)* — Element indices to rotate, 0-based, no duplicates; `nullptr` or empty: all elements

## Example:
```
auto ant = quadriga_lib::generate_arrayant_3GPP<double>(4, 1);
arma::vec downtilt = {0.0, 10.0 * arma::datum::pi / 180.0, 0.0}; // 10 degree mechanical downtilt
quadriga_lib::arrayant_rotate_pattern(ant, downtilt);
```

## See also:
- [[arrayant_rotate_pattern_multi]] (multi-freq counterpart)
- [[arrayant_interpolate]] (same orientation convention)
- [[calc_rotation_matrix]] (rotation matrix of the orientation)
MD!*/

template <typename dtype>
void quadriga_lib::arrayant_rotate_pattern(arrayant<dtype> &ant, const arma::Mat<dtype> &orientation,
                                           unsigned usage, const arma::uvec *i_element)
{
    // Check if arrayant object is valid
    std::string error_message = qd_arrayant_check(ant); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (orientation.n_rows != 3 || orientation.n_cols != 1)
        throw std::invalid_argument("Input 'orientation' must have size [3, 1].");
    if (!orientation.is_finite())
        throw std::invalid_argument("Input 'orientation' must be finite.");
    if (usage > 4)
        throw std::invalid_argument("Input parameter 'usage' must be 0, 1, 2, 3 or 4.");
    if (ant.read_only)
        throw std::invalid_argument("Cannot update read-only array antenna object inplace.");

    // Set element indices; a subset without duplicates that holds every element counts as all elements
    const arma::uword n_elements = ant.e_theta_re.n_slices;
    arma::uvec i_out;
    if (i_element == nullptr || i_element->n_elem == 0)
        i_out = arma::regspace<arma::uvec>(0, n_elements - 1);
    else
    {
        if (arma::any(*i_element >= n_elements))
            throw std::invalid_argument("Input 'i_element' contains indices that exceed the number of elements.");
        i_out = *i_element;
        if (arma::find_unique(i_out).eval().n_elem != i_out.n_elem)
            throw std::invalid_argument("Input 'i_element' must not contain duplicates.");
    }
    const bool use_all_elements = i_out.n_elem == n_elements;

    // Extract the element positions
    arma::Mat<dtype> element_pos_update;
    if (ant.element_pos.n_elem == 0)
        element_pos_update.zeros(3, n_elements);
    else
        element_pos_update = ant.element_pos;
    const arma::Mat<dtype> element_pos_interp = element_pos_update.cols(i_out);

    unsigned long long n_el = ant.e_theta_re.n_rows;
    unsigned long long n_az = ant.e_theta_re.n_cols;
    unsigned long long n_out = i_out.n_elem;
    unsigned long long n_ang = n_el * n_az;

    dtype tau = dtype(arma::datum::tau), pi_half = dtype(arma::datum::pi / 2.0), pi = dtype(arma::datum::pi),
          deg2rad = dtype(arma::datum::pi / 180.0), limit = dtype(1.0e-6);

    // Calculate the coverage range for the angle sampling grid
    dtype zero = dtype(0.0), az_step_min = dtype(1e38), az_step_max = zero, el_step_min = az_step_min,
          el_step_max = zero, step = zero;
    const dtype *ptr = ant.azimuth_grid.memptr();
    for (auto i = 0ULL; i < n_az; ++i)
    {
        step = i == 0 ? *ptr - ptr[n_az - 1] + tau : ptr[i] - ptr[i - 1];
        az_step_min = step < az_step_min && step > limit ? step : az_step_min;
        az_step_max = step > az_step_max ? step : az_step_max;
    }
    ptr = ant.elevation_grid.memptr();
    for (auto i = 0ULL; i <= n_el; ++i)
    {
        if (i == 0)
            step = *ptr + pi_half;
        else if (i == n_el)
            step = pi_half - ptr[i - 1];
        else
            step = ptr[i] - ptr[i - 1];

        el_step_min = step < el_step_min && step > limit ? step : el_step_min;
        el_step_max = step > el_step_max ? step : el_step_max;
    }
    step = pi_half - ptr[n_el - 1];
    el_step_min = step < el_step_min && step > limit ? step : el_step_min;
    el_step_max = step > el_step_max ? step : el_step_max;
    step = el_step_min < az_step_min ? el_step_min : az_step_min;

    const bool update_grid = (usage == 0 || usage == 1) && (az_step_max - az_step_min > limit ||
                                                            el_step_max - el_step_min > limit ||
                                                            std::abs(az_step_min - el_step_min) > limit ||
                                                            n_az == 1 || n_el == 1);

    // Obtain interpolation angles
    arma::Col<dtype> azimuth_grid_update, elevation_grid_update;
    if (update_grid)
    {
        if (!use_all_elements)
            throw std::invalid_argument("Update of sampling grid cannot be done for single elements of an array antenna!");

        if (step > dtype(0.0017)) // >= 0.1 degree --> Use entire sphere
        {
            unsigned long long N = (unsigned long long)std::round(pi / step);
            N = N < 4ULL ? 4ULL : N;

            azimuth_grid_update = arma::linspace<arma::Col<dtype>>(-pi, pi, 2ULL * N + 1ULL);

            if (std::abs(orientation.at(0)) < limit && std::abs(orientation.at(1)) < limit)
                elevation_grid_update = ant.elevation_grid;
            else
                elevation_grid_update = arma::linspace<arma::Col<dtype>>(-pi_half + step, pi_half - step, N - 1ULL);
        }
        else // step < 0.1 degree --> Use subsampled sphere
        {
            auto sub = quadriga_lib::generate_arrayant_omni<dtype>();
            sub.e_theta_re.zeros();
            arma::Col<unsigned> has_az(361), has_el(181);

            // Select azimuth angles
            const dtype *p = ant.azimuth_grid.memptr();
            dtype *q = sub.azimuth_grid.memptr();
            unsigned *ra = has_az.memptr();
            for (unsigned i = 0; i < 360; ++i)
            {
                while (p != ant.azimuth_grid.end() && *p <= q[i + 1])
                {
                    ra[i] = *p >= q[i] && *p <= q[i + 1] ? 1 : ra[i];
                    p++;
                }
                ra[i + 1] = ra[i];
                if (p == ant.azimuth_grid.end())
                    break;
            }

            // Select elevation angles
            p = ant.elevation_grid.memptr(), q = sub.elevation_grid.memptr();
            unsigned *re = has_el.memptr();
            for (unsigned i = 0; i < 180; ++i)
            {
                while (p != ant.elevation_grid.end() && *p <= q[i + 1])
                {
                    re[i] = *p >= q[i] && *p <= q[i + 1] ? 1 : re[i];
                    p++;
                }
                re[i + 1] = re[i];
                if (p == ant.elevation_grid.end())
                    break;
            }

            // Set interpolation target
            dtype *pt = sub.e_theta_re.memptr();
            for (auto i = 0ULL; i < sub.e_theta_re.n_elem; ++i)
                pt[i] = ra[i / 181] == 1 && re[i % 181] == 1 ? 1.0 : 0.0;

            // Find target area for interpolation
            quadriga_lib::arrayant_rotate_pattern(sub, orientation, 1);
            quadriga_lib::arrayant_remove_zeros(sub);

            // Subdivide target grid
            dtype sdiv = std::fmod(deg2rad, step) < 0.1 * step ? std::floor(deg2rad / step) : std::ceil(deg2rad / step);
            dtype stp = deg2rad / sdiv;
            unsigned long long ndiv = (unsigned long long)sdiv;

            bool wrap = sub.azimuth_grid.at(sub.azimuth_grid.n_elem - 1) + stp > pi;
            unsigned long long N = wrap ? ndiv * (sub.azimuth_grid.n_elem - 1) + 1 : ndiv * sub.azimuth_grid.n_elem;
            azimuth_grid_update.set_size(N);
            pt = azimuth_grid_update.memptr(), q = sub.azimuth_grid.memptr();
            for (auto i = 0ULL; i < sub.azimuth_grid.n_elem; ++i)
                if (wrap && i == sub.azimuth_grid.n_elem - 1ULL)
                    *pt = q[i];
                else
                    for (auto j = 0ULL; j < ndiv; ++j)
                        *pt++ = q[i] + dtype(j) * stp;

            wrap = sub.elevation_grid.at(sub.elevation_grid.n_elem - 1) + stp > pi_half;
            N = wrap ? ndiv * (sub.elevation_grid.n_elem - 1) + 1 : ndiv * sub.elevation_grid.n_elem;
            elevation_grid_update.set_size(N);
            pt = elevation_grid_update.memptr(), q = sub.elevation_grid.memptr();
            for (auto i = 0ULL; i < sub.elevation_grid.n_elem; ++i)
                if (wrap && i == sub.elevation_grid.n_elem - 1)
                    *pt = q[i];
                else
                    for (auto j = 0ULL; j < ndiv; ++j)
                        *pt++ = q[i] + dtype(j) * stp;
        }

        n_az = azimuth_grid_update.n_elem;
        n_el = elevation_grid_update.n_elem;
        n_ang = n_az * n_el;
    }
    else
    {
        azimuth_grid_update = ant.azimuth_grid;
        elevation_grid_update = ant.elevation_grid;
    }

    // Initiate output variables
    arma::Mat<dtype> V_re(n_out, n_ang, arma::fill::none), V_im(n_out, n_ang, arma::fill::none);
    arma::Mat<dtype> H_re(n_out, n_ang, arma::fill::none), H_im(n_out, n_ang, arma::fill::none);
    arma::Mat<dtype> azimuth_loc, elevation_loc, gamma;

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr(),
          *p_phi = azimuth_grid_update.memptr(), *p_theta = elevation_grid_update.memptr();

    for (auto ia = 0ULL; ia < n_az; ++ia)
        for (auto ie = 0ULL; ie < n_el; ++ie)
            *p_azimuth++ = p_phi[ia], *p_elevation++ = p_theta[ie];

    // The rotated pattern is the unrotated pattern seen by an antenna with this orientation
    arma::Cube<dtype> orientation_cube(orientation.memptr(), 3, 1, 1);

    // Rotation matrix for the element positions
    const arma::Col<dtype> R = quadriga_lib::calc_rotation_matrix(arma::Col<dtype>(orientation_cube.memptr(), 3));

    if (usage == 1 || usage == 4)
        azimuth_loc.set_size(n_out, n_ang), elevation_loc.set_size(n_out, n_ang);
    else if (usage == 2)
        gamma.set_size(n_out, n_ang);

    const bool loc_angles = usage == 1 || usage == 4;
    qd_arrayant_interpolate_dispatch<dtype>(ant, azimuth, elevation, i_out, orientation_cube, element_pos_interp,
                                            V_re, V_im, H_re, H_im, nullptr, loc_angles ? &azimuth_loc : nullptr,
                                            loc_angles ? &elevation_loc : nullptr, usage == 2 ? &gamma : nullptr, 1);
    azimuth.reset(), elevation.reset();

    orientation_cube.zeros();
    if (loc_angles) // Only return interpolated pattern (ignore polarization)
        qd_arrayant_interpolate_dispatch<dtype>(ant, azimuth_loc, elevation_loc, i_out, orientation_cube, element_pos_interp,
                                                V_re, V_im, H_re, H_im, nullptr, nullptr, nullptr, nullptr, 1);
    azimuth_loc.reset(), elevation_loc.reset();

    if (usage == 2) // Only adjust the polarization
    {
        gamma = gamma.t();
        for (auto i = 0ULL; i < n_out; ++i)
        {
            const dtype *p_gamma = gamma.colptr(i);
            dtype *p_theta_re = ant.e_theta_re.slice_memptr(i_out.at(i)), *p_theta_im = ant.e_theta_im.slice_memptr(i_out.at(i));
            dtype *p_phi_re = ant.e_phi_re.slice_memptr(i_out.at(i)), *p_phi_im = ant.e_phi_im.slice_memptr(i_out.at(i));

            for (auto j = 0ULL; j < n_ang; ++j)
            {
                dtype sin_gamma = std::sin(p_gamma[j]), cos_gamma = std::cos(p_gamma[j]);
                dtype tmp = sin_gamma * p_theta_re[j];
                p_theta_re[j] = cos_gamma * p_theta_re[j] - sin_gamma * p_phi_re[j];
                p_phi_re[j] = tmp + cos_gamma * p_phi_re[j];
                tmp = sin_gamma * p_theta_im[j];
                p_theta_im[j] = cos_gamma * p_theta_im[j] - sin_gamma * p_phi_im[j];
                p_phi_im[j] = tmp + cos_gamma * p_phi_im[j];
            }
        }
        return;
    }

    // Usage 0, 1, 3 or 4: write the interpolated patterns, [n_ang, n_out] after the transpose
    if (n_out > 1)
        V_re = V_re.t(), V_im = V_im.t(), H_re = H_re.t(), H_im = H_im.t();

    if (update_grid) // Set new array size, keep the coupling (all elements here)
    {
        ant.set_size(n_el, n_az, n_out, ant.n_ports()); // Writes out empty positions and coupling (zeros, identity)
        ant.azimuth_grid = azimuth_grid_update;
        ant.elevation_grid = elevation_grid_update;
    }

    for (auto i = 0ULL; i < n_out; ++i)
    {
        const arma::uword s = i_out.at(i);
        std::memcpy(ant.e_theta_re.slice_memptr(s), V_re.colptr(0) + i * n_ang, n_ang * sizeof(dtype));
        std::memcpy(ant.e_theta_im.slice_memptr(s), V_im.colptr(0) + i * n_ang, n_ang * sizeof(dtype));
        std::memcpy(ant.e_phi_re.slice_memptr(s), H_re.colptr(0) + i * n_ang, n_ang * sizeof(dtype));
        std::memcpy(ant.e_phi_im.slice_memptr(s), H_im.colptr(0) + i * n_ang, n_ang * sizeof(dtype));
    }

    // Rotate the element positions; empty positions stay empty unless set_size above wrote zeros (grid update)
    if (ant.element_pos.n_elem != 0)
    {
        const dtype *R_ptr = R.memptr(), *pI = element_pos_update.memptr();
        for (auto i = 0ULL; i < n_out; ++i)
        {
            const arma::uword j = 3 * i_out.at(i);
            dtype *pO = ant.element_pos.colptr(i_out.at(i));
            pO[0] = R_ptr[0] * pI[j] + R_ptr[3] * pI[j + 1] + R_ptr[6] * pI[j + 2];
            pO[1] = R_ptr[1] * pI[j] + R_ptr[4] * pI[j + 1] + R_ptr[7] * pI[j + 2];
            pO[2] = R_ptr[2] * pI[j] + R_ptr[5] * pI[j + 1] + R_ptr[8] * pI[j + 2];
        }
    }

    if (update_grid)
        quadriga_lib::arrayant_remove_zeros(ant);
}

template void quadriga_lib::arrayant_rotate_pattern(arrayant<float> &, const arma::Mat<float> &, unsigned, const arma::uvec *);
template void quadriga_lib::arrayant_rotate_pattern(arrayant<double> &, const arma::Mat<double> &, unsigned, const arma::uvec *);

/*!MD
# qdant_write
Write arrayant data to a QDANT (XML) file

- Multiple antennas can be stored in the same file using distinct `id` values
- If `id = 0` and the file exists, the antenna is appended with `id = max_existing_id + 1`; if file does not exist, `id = 1` gets assigned
- Without `layout` (`nullptr` or empty), the layout in the file is replaced by a `[1, n]` row of all IDs in the file
  in file order; a custom layout written before is lost. Pass the layout again with every write that should keep it.
- The file has single-precision-equivalent accuracy relative to the element peak, also for double-precision data:
  pattern magnitudes (in dB) and phases (in degree) get the fewest decimals (0 to 6) that keep the error of each
  complex sample at or below 6e-8 times the peak amplitude of its element (over e_theta and e_phi). Samples near
  the peak are accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per 20 dB below the
  peak (about 6e-4 at -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak. An element at
  a low level is quantized relative to its own peak. Element positions and coupling (magnitude, phase) are written
  as the shortest text that reads back as the same float (at most 9 significant digits); sampling grids are rounded
  to 0.001 degree
- An empty `element_pos` or coupling is not written; [[qdant_read]] returns zeros and identity for it. Above 4096
  elements the identity coupling is written, because the reader requires it in the file
- All non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element that is
  zero in both field components is written as an all `-inf` `EthetaMag` node.

## Declaration:
```
unsigned quadriga_lib::qdant_write(
    const std::string &fn,
    const arrayant<dtype> &ant,
    unsigned id = 0,
    const arma::u32_mat *layout = nullptr);
```

## Inputs:
- **`fn`** — Output QDANT filename; must not be empty
- **`ant`** — Valid arrayant
- **`id`** *(optional)* — Target ID in file; `0` appends with auto-assigned ID
- **`layout`** *(optional)* — Matrix organizing multiple antenna IDs within the file; must reference only IDs present in the file

## Returns:
- ID assigned to the written antenna within the file

## See also:
- [[qdant_read]] (read back QDANT files)
- [[qdant_write_multi]] (multi-freq counterpart)
MD!*/

template <typename dtype>
unsigned quadriga_lib::qdant_write(const std::string &fn, const arrayant<dtype> &ant, unsigned id, const arma::u32_mat *layout)
{
    // Check if arrayant object is valid
    std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    const arma::u32_mat layout_empty;
    unsigned id_in_file = 0;
    error_message = qd_arrayant_qdant_write(fn, id, &ant.name, &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                            &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                            &ant.coupling_re, &ant.coupling_im, &ant.center_frequency,
                                            layout == nullptr ? &layout_empty : layout, &id_in_file);

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    return id_in_file;
}

template unsigned quadriga_lib::qdant_write(const std::string &, const arrayant<float> &, unsigned, const arma::u32_mat *);
template unsigned quadriga_lib::qdant_write(const std::string &, const arrayant<double> &, unsigned, const arma::u32_mat *);
