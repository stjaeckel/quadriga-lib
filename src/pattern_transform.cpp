// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "quadriga_math.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# rotate_pattern
Rotate the patterns of an antenna array using Euler rotations, in place

- Turns the patterns, the polarization and the element positions of an array, e.g. for a mechanical downtilt. The
  rotated pattern equals, at the grid points, what [[interpolate_pattern]] returns for the unrotated pattern with the
  same `orientation`
- Port weights, `center_freq` and `name` are kept

## Declarations:
```
void quadriga_lib::rotate_pattern(
    arrayant<dtype> &ant,
    const arma::Mat<dtype> &orientation,
    unsigned usage = 0,
    const arma::uvec &i_element = arma::uvec());

void quadriga_lib::rotate_pattern(
    std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Mat<dtype> &orientation,
    unsigned usage = 0,
    const arma::uvec &i_element = arma::uvec());
```

## Inputs:
- **`ant`** — Valid arrayant; updated in place
- **`arrayant_vec`** — Vector form: non-empty vector of arrayant objects; updated in place; the grid is never adjusted,
  so that all entries keep the same grid
- **`orientation`** — Euler angles (bank, tilt, heading) in rad, finite; a positive tilt turns the main beam down, as in
  3GPP TR 38.901; `[3, 1]` or `[1, 3]`
- **`usage`** *(optional)* — Rotation mode; vector form: 0 to 2, where 0 and 1 act as 3 and 4. In the single form, modes
  0 and 1 adjust the sampling grid of non-uniformly sampled patterns, which needs all elements: a subset in `i_element`
  then throws. Element positions turn in all modes except 2. For scalar acoustic fields (pressure in `e_theta_re` only),
  use 1 to avoid spurious polarization effects:<br><br>
   | Mode | Pattern | Polarization | Grid adjustment |
   | ---- | ------- | ------------ | --------------- |
   | 0    | Yes     | Yes          | Yes             |
   | 1    | Yes     | No           | Yes             |
   | 2    | No      | Yes          | No              |
   | 3    | Yes     | Yes          | No              |
   | 4    | Yes     | No           | No              |
- **`i_element`** *(optional)* — Element indices to rotate, 0-based, no duplicates; empty: all elements

## Example:
```
auto ant = quadriga_lib::generate_antenna<double>("3GPP", 1.0, NAN, {}, {4});
arma::vec downtilt = {0.0, 10.0 * arma::datum::pi / 180.0, 0.0}; // 10 degree mechanical downtilt
quadriga_lib::rotate_pattern(ant, downtilt);
```

## See also:
- [[interpolate_pattern]] (evaluates a pattern for an orientation without changing it)
- [[calc_rotation_matrix]] (rotation matrix of the orientation)
- [[generate_antenna]] (electrical tilt through the port weights)
- [[join_arrays]] (joins arrays that are then rotated together)
MD!*/

template <typename dtype>
void quadriga_lib::rotate_pattern(arrayant<dtype> &ant, const arma::Mat<dtype> &orientation,
                                           unsigned usage, const arma::uvec &i_element)
{
    std::string error_message = qd_arrayant_check(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);

    qd_xyz(orientation, "orientation");
    if (usage > 4)
        throw std::invalid_argument("Input 'usage' must be 0, 1, 2, 3 or 4.");
    // Set element indices; a subset without duplicates that holds every element counts as all elements
    const arma::uword n_elements = ant.e_theta_re.n_slices;
    arma::uvec i_out;
    if (i_element.n_elem == 0)
        i_out = arma::regspace<arma::uvec>(0, n_elements - 1);
    else
    {
        if (arma::any(i_element >= n_elements))
            throw std::invalid_argument("Input 'i_element' contains an index that is out of range.");
        i_out = i_element;
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
            throw std::invalid_argument("Input 'i_element' must select all elements when 'usage' 0 or 1 adjusts the sampling grid.");

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
            auto sub = quadriga_lib::generate_antenna<dtype>("omni", 1.0);
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
            quadriga_lib::rotate_pattern(sub, orientation, 1);
            quadriga_lib::trim_pattern(sub);

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
    arma::Cube<dtype> R;
    quadriga_lib::calc_rotation_matrix(orientation_cube, R);

    if (usage == 1 || usage == 4)
        azimuth_loc.set_size(n_out, n_ang), elevation_loc.set_size(n_out, n_ang);
    else if (usage == 2)
        gamma.set_size(n_out, n_ang);

    const bool loc_angles = usage == 1 || usage == 4;
    qd_interpolate_pattern_dispatch<dtype>(ant, azimuth, elevation, i_out, orientation_cube, element_pos_interp,
                                            V_re, V_im, H_re, H_im, nullptr, loc_angles ? &azimuth_loc : nullptr,
                                            loc_angles ? &elevation_loc : nullptr, usage == 2 ? &gamma : nullptr, 1);
    azimuth.reset(), elevation.reset();

    orientation_cube.zeros();
    if (loc_angles) // Only return interpolated pattern (ignore polarization)
        qd_interpolate_pattern_dispatch<dtype>(ant, azimuth_loc, elevation_loc, i_out, orientation_cube, element_pos_interp,
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

    if (update_grid) // Set new array size, keep the port weights (all elements here)
    {
        ant.set_size(n_el, n_az, n_out, ant.n_ports()); // Writes out empty positions and port weights (zeros, identity)
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
        quadriga_lib::trim_pattern(ant);
}

template void quadriga_lib::rotate_pattern(arrayant<float> &, const arma::Mat<float> &, unsigned, const arma::uvec &);
template void quadriga_lib::rotate_pattern(arrayant<double> &, const arma::Mat<double> &, unsigned, const arma::uvec &);

template <typename dtype>
void quadriga_lib::rotate_pattern(std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
                                                 const arma::Mat<dtype> &orientation, unsigned usage,
                                                 const arma::uvec &i_element)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("Input 'arrayant_vec' cannot be empty.");

    if (usage > 2)
        throw std::invalid_argument("Input 'usage' must be 0, 1 or 2.");

    // Validate input vector before modifying any entries; the entries share the element count, so orientation and
    // index errors surface at entry 0 before anything is changed
    std::string err = qd_arrayant_vec_is_valid_quick(arrayant_vec);
    if (!err.empty())
        throw std::invalid_argument("Input 'arrayant_vec' is not valid: " + err);

    if (orientation.n_elem != 3) // Checked once here, so that the error carries no entry prefix
        throw std::invalid_argument("Input 'orientation' must have 3 elements.");

    // Map usage to no-grid-adjustment modes: 0→3, 1→4, 2→2
    const unsigned internal_usage = usage == 0 ? 3 : (usage == 1 ? 4 : usage);

    for (size_t i = 0; i < arrayant_vec.size(); ++i)
    {
        try
        {
            quadriga_lib::rotate_pattern(arrayant_vec[i], orientation, internal_usage, i_element);
        }
        catch (const std::invalid_argument &e)
        {
            throw std::invalid_argument("Entry " + std::to_string(i) + ": " + e.what());
        }
    }
}

template void quadriga_lib::rotate_pattern(std::vector<quadriga_lib::arrayant<float>> &, const arma::Mat<float> &, unsigned, const arma::uvec &);
template void quadriga_lib::rotate_pattern(std::vector<quadriga_lib::arrayant<double>> &, const arma::Mat<double> &, unsigned, const arma::uvec &);

/*!MD
# trim_pattern
Remove zero-valued entries from antenna pattern data, reducing its size

- Removes the azimuth columns and elevation rows without power from the sampling grid and the patterns, in place
- Positions, port weights, name and frequency are unchanged

## Declaration:
```
void quadriga_lib::trim_pattern(arrayant<dtype> &ant);
```

## Input:
- **`ant`** — Valid arrayant; updated in place; to keep the original, call it on a copy (`auto b = ant;`)

## Notes:
- **Threshold**: a column or row is removed if its total power, summed over all field components and elements, does not
  exceed $10^{-12}$ times the peak power of a single sample (summed over field components and elements). The result does
  not depend on the scale of the pattern; an all-zero pattern is left unchanged
- **Neighbors**: the immediate neighbors of retained entries are also kept, unless the local grid spacing is strongly
  asymmetric (by a factor above 1.5)

## See also:
- [[rotate_pattern]] (trims the grid it adjusts)
- [[qdant_write]] (stores the reduced pattern)
MD!*/

template <typename dtype>
void quadriga_lib::trim_pattern(arrayant<dtype> &ant)
{
    std::string error_message = qd_arrayant_check(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);

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
        throw std::invalid_argument("Input 'ant' must have finite field values.");

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

template void quadriga_lib::trim_pattern(arrayant<float> &);
template void quadriga_lib::trim_pattern(arrayant<double> &);