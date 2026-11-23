// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// Element indices of the calc functions: empty = all elements, duplicates allowed, out of range throws
static arma::uvec qd_calc_element_list(const arma::uvec &i_element, arma::uword n_elements)
{
    if (i_element.n_elem == 0) // A valid arrayant has at least one element
        return arma::regspace<arma::uvec>(0, n_elements - 1);
    if (arma::any(i_element >= n_elements))
        throw std::invalid_argument("Input 'i_element' contains an index that is out of range.");
    return i_element;
}

// Beamwidth and pointing angles of one element in degree (main lobe search on interpolated cuts); valid arrayant
template <typename dtype>
static void qd_calc_beamwidth_element(const quadriga_lib::arrayant<dtype> &ant, arma::uword i_element,
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
        quadriga_lib::interpolate_pattern<dtype>(ant, az_q, el_q, V_re, V_im, H_re, H_im, i_el_vec,
                                                  {}, {}, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
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
static dtype qd_calc_directivity_element(const quadriga_lib::arrayant<dtype> &ant, arma::uword i_element)
{
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

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# calc_beamwidth
Calculate the beamwidth and pointing angles of the elements of an antenna array in degrees

- Measures the width of the main lobe of each element in azimuth and elevation at a threshold below its peak, and the
  direction of the peak
- Calculated per element, not per port: the port weights are ignored

## Declarations:
```
void quadriga_lib::calc_beamwidth(
    const arrayant<dtype> &ant,
    const arma::uvec &i_element = arma::uvec(),
    double threshold_dB = 3.0,
    arma::Col<dtype> *beamwidth_az_deg = nullptr,
    arma::Col<dtype> *beamwidth_el_deg = nullptr,
    arma::Col<dtype> *az_point_ang_deg = nullptr,
    arma::Col<dtype> *el_point_ang_deg = nullptr);

void quadriga_lib::calc_beamwidth(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::uvec &i_element = arma::uvec(),
    double threshold_dB = 3.0,
    arma::Mat<dtype> *beamwidth_az_deg = nullptr,
    arma::Mat<dtype> *beamwidth_el_deg = nullptr,
    arma::Mat<dtype> *az_point_ang_deg = nullptr,
    arma::Mat<dtype> *el_point_ang_deg = nullptr);
```

## Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of `n_freq` arrayant objects that passes [[is_valid]]; column
  `i_freq` of each output holds the results of `arrayant_vec[i_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`
- **`threshold_dB`** *(optional)* — Threshold below the peak in dB, positive and finite; 3 dB is the half-power
  beamwidth; loudspeakers state the coverage angle between the -6 dB points, use 6

## Outputs:
- **`beamwidth_az_deg`** *(optional)* — Azimuth beamwidth in degrees; `[n_out]`, vector form `[n_out, n_freq]`
- **`beamwidth_el_deg`** *(optional)* — Elevation beamwidth in degrees; `[n_out]`, vector form `[n_out, n_freq]`
- **`az_point_ang_deg`** *(optional)* — Azimuth pointing angle of the main beam in degrees; `[n_out]`, vector form
  `[n_out, n_freq]`
- **`el_point_ang_deg`** *(optional)* — Elevation pointing angle of the main beam in degrees; `[n_out]`, vector form
  `[n_out, n_freq]`

## Example:
```
auto ant = quadriga_lib::generate_antenna<double>("custom", 1.0, NAN, {30.0, 20.0});
arma::vec bw_az, bw_el;
quadriga_lib::calc_beamwidth(ant, {}, 3.0, &bw_az, &bw_el); // about 30 and 20 degrees
```

## Notes:
- **Main lobe**: starting at the maximum, the search goes outward until the power drops below the threshold; side lobes
  above the threshold are not included
- **Resolution**: the cuts are interpolated as in [[interpolate_pattern]], about 100 times finer than the sampling grid
- **Azimuth cut**: at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360°
  when the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- **Elevation cut**: along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°

## See also:
- [[calc_directivity]] (directivity in dBi of the elements)
- [[apply_port_weights]] (combines the elements into ports, to measure the beams of the ports)
- [[generate_antenna]] (creates elements with a given beamwidth)
MD!*/

template <typename dtype>
void quadriga_lib::calc_beamwidth(const arrayant<dtype> &ant, const arma::uvec &i_element, double threshold_dB_in,
                                           arma::Col<dtype> *beamwidth_az_deg, arma::Col<dtype> *beamwidth_el_deg,
                                           arma::Col<dtype> *az_point_ang_deg, arma::Col<dtype> *el_point_ang_deg)
{
    const dtype threshold_dB = dtype(threshold_dB_in);
    const std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (!error_message.empty())
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);
    if (!std::isfinite(threshold_dB) || threshold_dB <= dtype(0.0))
        throw std::invalid_argument("Input 'threshold_dB' must be positive and finite.");

    const arma::Col<dtype> *outs[] = {beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg};
    const char *names[] = {"beamwidth_az_deg", "beamwidth_el_deg", "az_point_ang_deg", "el_point_ang_deg"};
    qd_check_alias_list(outs, names, &i_element);
    qd_check_alias_arrayant(outs, names, ant);

    const arma::uvec i_out = qd_calc_element_list(i_element, ant.e_theta_re.n_slices);
    const arma::uword n_out = i_out.n_elem;
    for (arma::Col<dtype> *out : {beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg})
        if (out != nullptr)
            out->set_size(n_out);

    for (arma::uword i = 0; i < n_out; ++i)
        qd_calc_beamwidth_element(ant, i_out[i], threshold_dB,
                                           beamwidth_az_deg == nullptr ? nullptr : beamwidth_az_deg->memptr() + i,
                                           beamwidth_el_deg == nullptr ? nullptr : beamwidth_el_deg->memptr() + i,
                                           az_point_ang_deg == nullptr ? nullptr : az_point_ang_deg->memptr() + i,
                                           el_point_ang_deg == nullptr ? nullptr : el_point_ang_deg->memptr() + i);
}

template void quadriga_lib::calc_beamwidth(const arrayant<float> &, const arma::uvec &, double, arma::Col<float> *,
                                                    arma::Col<float> *, arma::Col<float> *, arma::Col<float> *);
template void quadriga_lib::calc_beamwidth(const arrayant<double> &, const arma::uvec &, double, arma::Col<double> *,
                                                    arma::Col<double> *, arma::Col<double> *, arma::Col<double> *);

template <typename dtype>
void quadriga_lib::calc_beamwidth(const std::vector<arrayant<dtype>> &arrayant_vec,
                                                 const arma::uvec &i_element, double threshold_dB_in,
                                                 arma::Mat<dtype> *beamwidth_az_deg, arma::Mat<dtype> *beamwidth_el_deg,
                                                 arma::Mat<dtype> *az_point_ang_deg, arma::Mat<dtype> *el_point_ang_deg)
{
    const dtype threshold_dB = dtype(threshold_dB_in);
    if (arrayant_vec.empty())
        throw std::invalid_argument("Input 'arrayant_vec' cannot be empty.");
    const std::string error_message = qd_arrayant_vec_is_valid_quick(arrayant_vec);
    if (!error_message.empty())
        throw std::invalid_argument("Input 'arrayant_vec' is not valid: " + error_message);

    arma::Mat<dtype> *const outs[] = {beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg};
    const arma::Mat<dtype> *outs_c[] = {beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg};
    const char *names[] = {"beamwidth_az_deg", "beamwidth_el_deg", "az_point_ang_deg", "el_point_ang_deg"};
    qd_check_alias_list(outs_c, names, &i_element);
    for (const auto &a : arrayant_vec)
        qd_check_alias_arrayant(outs_c, names, a);

    if (!std::isfinite(threshold_dB) || threshold_dB <= dtype(0.0))
        throw std::invalid_argument("Input 'threshold_dB' must be positive and finite.");
    const arma::uvec i_out = qd_calc_element_list(i_element, arrayant_vec[0].e_theta_re.n_slices);

    // The entries share the element count (checked above)
    const arma::uword n_out = i_out.n_elem, n_freq = (arma::uword)arrayant_vec.size();
    for (arma::Mat<dtype> *out : outs)
        if (out != nullptr)
            out->set_size(n_out, n_freq);
    auto at = [&](int k, arma::uword i, arma::uword f) { return outs[k] == nullptr ? nullptr : &outs[k]->at(i, f); };
    for (arma::uword f = 0; f < n_freq; ++f)
        for (arma::uword i = 0; i < n_out; ++i)
            qd_calc_beamwidth_element(arrayant_vec[f], i_out[i], threshold_dB, at(0, i, f), at(1, i, f),
                                               at(2, i, f), at(3, i, f));
}

template void quadriga_lib::calc_beamwidth(const std::vector<arrayant<float>> &, const arma::uvec &, double,
                                                          arma::Mat<float> *, arma::Mat<float> *, arma::Mat<float> *, arma::Mat<float> *);
template void quadriga_lib::calc_beamwidth(const std::vector<arrayant<double>> &, const arma::uvec &, double,
                                                          arma::Mat<double> *, arma::Mat<double> *, arma::Mat<double> *, arma::Mat<double> *);

/*!MD
# calc_directivity
Calculate the directivity in dBi of the elements of an antenna array

- Measures how strongly an element concentrates its power: $D = 10 \log_{10}(P_\mathrm{max} / \bar{P})$ with the peak
  radiation intensity $P_\mathrm{max}$ and its mean $\bar{P}$ over the sphere; an isotropic radiator has 0 dBi
- Calculated per element, not per port: the port weights are ignored

## Declarations:
```
arma::Col<dtype> quadriga_lib::calc_directivity(
    const arrayant<dtype> &ant,
    const arma::uvec &i_element = arma::uvec());

arma::Mat<dtype> quadriga_lib::calc_directivity(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::uvec &i_element = arma::uvec());
```

## Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of `n_freq` arrayant objects that passes [[is_valid]]; column
  `i_freq` of the output holds the results of `arrayant_vec[i_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`

## Returns:
- Directivity in dBi; `[n_out]`, vector form `[n_out, n_freq]`

## Example:
```
auto ant = quadriga_lib::generate_antenna<double>("dipole");
arma::vec directivity_dBi = quadriga_lib::calc_directivity(ant); // about 1.76 dBi
```

## Notes:
- **Scale**: the result does not depend on the scale of the pattern; an element whose pattern is zero everywhere gives 0
  dBi

## See also:
- [[calc_beamwidth]] (beamwidth and pointing angles of the elements)
- [[apply_port_weights]] (combines the elements into ports, to measure the directivity of the ports)
MD!*/

template <typename dtype>
arma::Col<dtype> quadriga_lib::calc_directivity(const arrayant<dtype> &ant, const arma::uvec &i_element)
{
    const std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (!error_message.empty())
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);

    const arma::uvec i_out = qd_calc_element_list(i_element, ant.e_theta_re.n_slices);
    arma::Col<dtype> directivity_dBi(i_out.n_elem, arma::fill::none);
    for (arma::uword i = 0; i < i_out.n_elem; ++i)
        directivity_dBi[i] = qd_calc_directivity_element(ant, i_out[i]);
    return directivity_dBi;
}

template arma::Col<float> quadriga_lib::calc_directivity(const arrayant<float> &, const arma::uvec &);
template arma::Col<double> quadriga_lib::calc_directivity(const arrayant<double> &, const arma::uvec &);

template <typename dtype>
arma::Mat<dtype> quadriga_lib::calc_directivity(const std::vector<arrayant<dtype>> &arrayant_vec,
                                                               const arma::uvec &i_element)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("Input 'arrayant_vec' cannot be empty.");
    const std::string error_message = qd_arrayant_vec_is_valid_quick(arrayant_vec);
    if (!error_message.empty())
        throw std::invalid_argument("Input 'arrayant_vec' is not valid: " + error_message);

    const arma::uvec i_out = qd_calc_element_list(i_element, arrayant_vec[0].e_theta_re.n_slices);
    const arma::uword n_out = i_out.n_elem, n_freq = (arma::uword)arrayant_vec.size();
    arma::Mat<dtype> directivity_dBi(n_out, n_freq, arma::fill::none);
    for (arma::uword f = 0; f < n_freq; ++f)
        for (arma::uword i = 0; i < n_out; ++i)
            directivity_dBi.at(i, f) = qd_calc_directivity_element(arrayant_vec[f], i_out[i]);
    return directivity_dBi;
}

template arma::Mat<float> quadriga_lib::calc_directivity(const std::vector<arrayant<float>> &, const arma::uvec &);
template arma::Mat<double> quadriga_lib::calc_directivity(const std::vector<arrayant<double>> &, const arma::uvec &);