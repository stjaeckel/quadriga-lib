// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <stdexcept>
#include <cstring>   // std::memcopy
#include <iomanip>   // std::setprecision
#include <algorithm> // std::replace
#include <iostream>
#include <limits>

#include "quadriga_lib.hpp"
#include "qd_arrayant_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# generate_arrayant_omni
Generate an isotropic radiator with vertical polarization

- Returns a single-element antenna array with omnidirectional pattern and vertical polarization.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni(dtype res = 1.0);
```

## Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

## Returns:
- `quadriga_lib::arrayant<dtype>` — Isotropic radiator antenna object
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni(dtype res)
{
    quadriga_lib::arrayant<dtype> ant;

    res = (res <= (dtype)0.0) ? (dtype)1.0 : (res >= (dtype)90.0 ? (dtype)90.0 : res);

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    arma::uword no_az = arma::uword(360.0 / (double)res) + 1ULL;
    arma::uword no_el = arma::uword(180.0 / (double)res) + 1ULL;

    ant.name = "omni";
    ant.e_theta_re.ones(no_el, no_az, 1);
    ant.e_theta_im.zeros(no_el, no_az, 1);
    ant.e_phi_re.zeros(no_el, no_az, 1);
    ant.e_phi_im.zeros(no_el, no_az, 1);
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, no_az);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, no_el);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_omni(float res);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_omni(double res);

/*!MD
# generate_arrayant_xpol
Generate a cross-polarized isotropic radiator

- Returns a two-element antenna array with omnidirectional patterns in vertical and horizontal polarization.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol(dtype res = 1.0);
```

## Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

## Returns:
- `quadriga_lib::arrayant<dtype>` — Cross-polarized isotropic radiator antenna object
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol(dtype res)
{
    quadriga_lib::arrayant<dtype> ant;

    res = (res <= (dtype)0.0) ? (dtype)1.0 : (res >= (dtype)90.0 ? (dtype)90.0 : res);

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    arma::uword no_az = arma::uword(360.0 / (double)res) + 1ULL;
    arma::uword no_el = arma::uword(180.0 / (double)res) + 1ULL;
    arma::uword no_val = no_az * no_el;

    ant.name = "xpol";
    ant.e_theta_re.set_size(no_el, no_az, 2);
    ant.e_theta_im.zeros(no_el, no_az, 2);
    ant.e_phi_re.set_size(no_el, no_az, 2);
    ant.e_phi_im.zeros(no_el, no_az, 2);

    dtype one = (dtype)1.0, zero = (dtype)0.0;
    dtype *ptr0 = ant.e_phi_re.slice_memptr(0);
    dtype *ptr1 = ant.e_theta_re.slice_memptr(1);
    dtype *ptr2 = ant.e_phi_re.slice_memptr(1);
    dtype *ptr3 = ant.e_theta_re.slice_memptr(0);
    for (arma::uword i = 0ULL; i < no_val; ++i)
        ptr0[i] = zero, ptr1[i] = zero, ptr2[i] = one, ptr3[i] = one;

    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, no_az);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, no_el);
    ant.element_pos.zeros(3, 2);
    ant.coupling_re.eye(2, 2);
    ant.coupling_im.zeros(2, 2);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_xpol(float res);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_xpol(double res);

/*!MD
# generate_arrayant_dipole
Generate a short dipole antenna with vertical polarization

- Returns a single-element short dipole antenna pattern with vertical polarization.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole(dtype res = 1.0);
```

## Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

## Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized short dipole antenna object
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole(dtype res)
{
    quadriga_lib::arrayant<dtype> ant;

    res = (res <= (dtype)0.0) ? (dtype)1.0 : (res >= (dtype)90.0 ? (dtype)90.0 : res);

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    arma::uword no_az = arma::uword(360.0 / (double)res) + 1ULL;
    arma::uword no_el = arma::uword(180.0 / (double)res) + 1ULL;

    ant.name = "dipole";
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, no_az);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, no_el);
    ant.e_theta_re.zeros(no_el, no_az, 1);
    ant.e_theta_im.zeros(no_el, no_az, 1);
    ant.e_phi_re.zeros(no_el, no_az, 1);
    ant.e_phi_im.zeros(no_el, no_az, 1);

    arma::Mat<dtype> tmp = arma::repmat(ant.elevation_grid, 1, no_az);
    tmp = arma::cos(dtype(0.999999) * tmp) * dtype(std::sqrt(1.499961));
    std::memcpy(ant.e_theta_re.slice_memptr(0), tmp.memptr(), tmp.n_elem * sizeof(dtype));

    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_dipole(float res);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_dipole(double res);

/*!MD
# generate_arrayant_half_wave_dipole
Generate a half-wave dipole antenna with vertical polarization

- Returns a single-element half-wave dipole antenna pattern with vertical polarization.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole(dtype res = 1.0);
```

## Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

## Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized half-wave dipole antenna object
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole(dtype res)
{
    quadriga_lib::arrayant<dtype> ant;

    res = (res <= (dtype)0.0) ? (dtype)1.0 : (res >= (dtype)90.0 ? (dtype)90.0 : res);

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    arma::uword no_az = arma::uword(360.0 / (double)res) + 1ULL;
    arma::uword no_el = arma::uword(180.0 / (double)res) + 1ULL;

    ant.name = "half-wave-dipole";
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, no_az);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, no_el);
    ant.e_theta_re.zeros(no_el, no_az, 1);
    ant.e_theta_im.zeros(no_el, no_az, 1);
    ant.e_phi_re.zeros(no_el, no_az, 1);
    ant.e_phi_im.zeros(no_el, no_az, 1);

    arma::Mat<dtype> tmp = dtype(0.999999) * arma::repmat(ant.elevation_grid, 1, no_az);
    tmp = arma::cos(pih * arma::sin(tmp)) / arma::cos(tmp);
    tmp = tmp * dtype(1.280968208215292);

    std::memcpy(ant.e_theta_re.slice_memptr(0), tmp.memptr(), tmp.n_elem * sizeof(dtype));

    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_half_wave_dipole(float res);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_half_wave_dipole(double res);

/*!MD
# generate_arrayant_custom
Generate an antenna with custom 3dB beamwidth

- Returns a single-element antenna with independently configurable azimuth and elevation 3dB (FWHM) beamwidths.
- Rear-side gain is controlled by a linear front-to-back ratio; `0.0` means no rear radiation.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(
    dtype az_3dB = 90.0,
    dtype el_3dB = 90.0, 
    dtype rear_gain_lin = 0.0, 
    dtype res = 1.0);
```

## Inputs:
- **`az_3dB`** *(optional)* — Azimuth 3dB beamwidth in degrees
- **`el_3dB`** *(optional)* — Elevation 3dB beamwidth in degrees
- **`rear_gain_lin`** *(optional)* — Front-to-back gain ratio (linear)
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

## Returns:
- `quadriga_lib::arrayant<dtype>` — Antenna object with specified beamwidth and rear gain
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(dtype az_3dB, dtype el_3dB, dtype rear_gain_lin, dtype res)
{
    // Fix input ranges
    az_3dB = (az_3dB <= (dtype)0.0) ? (dtype)90.0 : az_3dB;
    el_3dB = (el_3dB <= (dtype)0.0) ? (dtype)90.0 : el_3dB;
    rear_gain_lin = (rear_gain_lin <= (dtype)0.0) ? (dtype)0.0 : rear_gain_lin;
    res = (res <= (dtype)0.0) ? (dtype)1.0 : (res >= (dtype)90.0 ? (dtype)90.0 : res);

    constexpr dtype zero = dtype(0.0), one = dtype(1.0), half = dtype(0.5),
                    limit = dtype(1e-7), step = dtype(-0.382), limit_inf = dtype(1e38);
    const dtype pi = dtype(arma::datum::pi), deg2rad = dtype(arma::datum::pi / 360.0);

    arma::uword no_az = arma::uword(360.0 / (double)res) + 1ULL;
    arma::uword no_el = arma::uword(180.0 / (double)res) + 1ULL;

    quadriga_lib::arrayant<dtype> ant;

    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, no_az);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pi * half, pi * half, no_el);
    arma::Col<dtype> phi_sq = ant.azimuth_grid % ant.azimuth_grid;
    arma::Col<dtype> cos_theta = arma::cos(ant.elevation_grid);
    cos_theta.at(0) = zero, cos_theta.at(no_el - 1) = zero;
    arma::Col<dtype> az_3dB_rad(1), el_3dB_rad(1);
    az_3dB_rad.at(0) = az_3dB * deg2rad;
    el_3dB_rad.at(0) = el_3dB * deg2rad;

    // Calculate azimuth pattern cut
    dtype a = one, d = half, x = limit_inf, delta = limit_inf;
    arma::Col<dtype> C(no_az), D(no_el);
    for (unsigned lp = 0; lp < 5000; ++lp)
    {
        dtype an = (lp == 0) ? a : a + d;
        delta = (lp == 0) ? limit_inf : std::abs(a - an);
        C = rear_gain_lin + (one - rear_gain_lin) * arma::exp(-an * phi_sq);
        auto xn = quadriga_lib::interp_1D(C, ant.azimuth_grid, az_3dB_rad);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    C = arma::exp(-a * phi_sq);

    // Calculate elevation pattern cut
    a = one, d = half, x = limit_inf, delta = limit_inf;
    for (unsigned lp = 0; lp < 5000; ++lp)
    {
        dtype an = (lp == 0) ? a : a + d;
        delta = (lp == 0) ? limit_inf : std::abs(a - an);
        D = arma::pow(cos_theta, an);
        auto xn = quadriga_lib::interp_1D(D, ant.elevation_grid, el_3dB_rad);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    D = arma::pow(cos_theta, a);

    // Combined pattern
    ant.e_theta_re.zeros(no_el, no_az, 1);
    dtype *ptr = ant.e_theta_re.memptr();
    for (dtype *col = C.begin(); col != C.end(); ++col)
        for (dtype *row = D.begin(); row != D.end(); ++row)
            *ptr++ = std::sqrt(rear_gain_lin + (one - rear_gain_lin) * *row * *col);

    ant.e_theta_im.zeros(no_el, no_az, 1);
    ant.e_phi_re.zeros(no_el, no_az, 1);
    ant.e_phi_im.zeros(no_el, no_az, 1);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);
    ant.name = "custom";

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    // Normalize to Gain
    dtype directivity = ant.calc_directivity_dBi(0);
    directivity = dtype(std::pow(10.0, 0.1 * double(directivity)));
    dtype p_max = ant.e_theta_re.max();
    p_max *= p_max;
    ant.e_theta_re *= std::sqrt(directivity / p_max);

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_custom(float az_3dB, float el_3dB, float rear_gain_lin, float res);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_custom(double az_3dB, double el_3dB, double rear_gain_lin, double res);

/*!MD
# generate_arrayant_ula
Generate a uniform linear array (ULA)

- Returns a horizontally stacked linear array of N elements with half-wavelength spacing by default.
- Default per-element pattern is a vertically polarized isotropic radiator.
- If `pattern` is provided, its radiation pattern is used for each element; element positions, coupling, and center frequency from `pattern` are ignored.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_ula(
    arma::uword N = 1, 
    dtype center_freq = 299792458.0, 
    dtype spacing = 0.5,
    const quadriga_lib::arrayant<dtype> *pattern = nullptr, 
    dtype res = 1.0);
```

## Inputs:
- **`N`** *(optional)* — Number of elements
- **`center_freq`** *(optional)* — Center frequency
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`pattern`** *(optional)* — Custom per-element antenna pattern; overrides default isotropic pattern
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; ignored if `pattern` is provided

## Returns:
- `quadriga_lib::arrayant<dtype>` — ULA antenna array object
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_ula(arma::uword N, dtype center_freq, dtype spacing,
                                                                  const arrayant<dtype> *pattern, dtype res)
{
    // Fix input ranges
    N = (N == 0ULL) ? 1ULL : N;
    center_freq = (center_freq <= (dtype)0.0) ? (dtype)299792458.0 : center_freq;
    spacing = (spacing < (dtype)0.0) ? (dtype)0.5 : spacing;
    res = (res <= (dtype)0.0) ? (dtype)1.0 : (res >= (dtype)90.0 ? (dtype)90.0 : res);

    double wavelength = 299792458.0 / double(center_freq);
    constexpr dtype zero = dtype(0.0);

    // Initialize pattern
    quadriga_lib::arrayant<dtype> ant = (pattern == nullptr) ? quadriga_lib::generate_arrayant_omni<dtype>(res) : pattern->copy();

    if (pattern != nullptr)
    {
        std::string error_message = ant.validate(); // Deep check
        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());
    }

    ant.center_frequency = center_freq;

    // Duplicate the existing elements in y-direction (horizontal stacking)
    arma::uword n_elements = ant.n_elements();
    if (N > 1ULL)
    {
        for (arma::uword source = n_elements; source > 0ULL; source--)
        {
            arma::uword i_start = n_elements + source - 1ULL;
            arma::uword i_end = N * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> y_position = arma::linspace<arma::Col<dtype>>(zero, dtype(N - 1ULL) * spacing * dtype(wavelength), N);
        y_position = y_position - arma::mean(y_position);

        for (arma::uword m = 0ULL; m < N; ++m)
            for (arma::uword n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(1ULL, m * n_elements + n) = y_position.at(m);
    }

    ant.name = "ula";
    return ant;
}

template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_ula(arma::uword N, float center_freq, float spacing,
                                                                           const arrayant<float> *pattern, float res);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_ula(arma::uword N, double center_freq, double spacing,
                                                                            const arrayant<double> *pattern, double res);

/*!MD
# generate_arrayant_3GPP
Generate a 3GPP-NR compliant antenna array model

- Supports vertical (M) and horizontal (N) element stacking within panels, and multi-panel arrays (Mg × Ng).
- If `pattern` is provided, its radiation pattern is used for each element; element positions, coupling, and center frequency from `pattern` are ignored.
- Electrical downtilt (`tilt`) applies only to `pol` modes 4, 5, and 6.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(
    arma::uword M = 1, 
    arma::uword N = 1, 
    dtype center_freq = 299792458.0,
    unsigned pol = 1, 
    dtype tilt = 0.0, 
    dtype spacing = 0.5, 
    arma::uword Mg = 1,
    arma::uword Ng = 1, 
    dtype dgv = 0.5, 
    dtype dgh = 0.5,
    const quadriga_lib::arrayant<dtype> *pattern = nullptr, 
    dtype res = 1.0);
```

## Inputs:
- **`M`** *(optional)* — Number of vertical elements per panel
- **`N`** *(optional)* — Number of horizontal elements per panel
- **`center_freq`** *(optional)* — Center frequency
- **`pol`** *(optional)* — Polarization mode:<br><br>
   `pol` | Description | Elements 
  -------|-------------|----------
   1 | Vertical polarization | NM 
   2 | H/V polarization | 2NM 
   3 | ±45° polarization | 2NM 
   4 | Vertical, vertical elements combined | N 
   5 | H/V, vertical elements combined | 2N 
   6 | ±45°, vertical elements combined | 2N 
- **`tilt`** *(optional)* — Electrical downtilt in degrees; applies to `pol` 4–6
- **`spacing`** *(optional)* — Inter-element spacing within a panel in wavelengths
- **`Mg`** *(optional)* — Number of vertically stacked panels
- **`Ng`** *(optional)* — Number of horizontally stacked panels
- **`dgv`** *(optional)* — Panel spacing in vertical direction in wavelengths
- **`dgh`** *(optional)* — Panel spacing in horizontal direction in wavelengths
- **`pattern`** *(optional)* — Custom per-element antenna pattern; overrides default 3GPP element pattern
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; ignored if `pattern` is provided

## Returns:
- `quadriga_lib::arrayant<dtype>` — 3GPP-NR antenna array object
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(arma::uword M, arma::uword N, dtype center_freq,
                                                                   unsigned pol, dtype tilt, dtype spacing,
                                                                   arma::uword Mg, arma::uword Ng, dtype dgv, dtype dgh,
                                                                   const arrayant<dtype> *pattern, dtype res)
{
    // Fix input ranges
    M = (M == 0ULL) ? 1ULL : M;
    N = (N == 0ULL) ? 1ULL : N;
    center_freq = (center_freq <= (dtype)0.0) ? (dtype)299792458.0 : center_freq;
    pol = (pol == 0 || pol > 6) ? 1 : pol;
    spacing = (spacing < (dtype)0.0) ? (dtype)0.5 : spacing;
    Mg = (Mg == 0ULL) ? 1ULL : Mg;
    Ng = (Ng == 0ULL) ? 1ULL : Ng;
    dgv = (dgv < (dtype)0.0) ? (dtype)0.5 : dgv;
    dgh = (dgh < (dtype)0.0) ? (dtype)0.5 : dgh;
    res = (res <= (dtype)0.0) ? (dtype)1.0 : (res >= (dtype)90.0 ? (dtype)90.0 : res);

    double pi = arma::datum::pi, rad2deg = 180.0 / pi, deg2rad = pi / 180.0;
    double wavelength = 299792458.0 / double(center_freq);
    constexpr dtype zero = dtype(0.0);

    // Initialize pattern
    quadriga_lib::arrayant<dtype> ant = (pattern == nullptr) ? quadriga_lib::generate_arrayant_omni<dtype>(res) : pattern->copy();

    if (pattern != nullptr)
    {
        std::string error_message = ant.validate(); // Deep check
        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());
    }

    ant.center_frequency = center_freq;
    arma::uword n_az = ant.n_azimuth(), n_el = ant.n_elevation();

    if (pattern == nullptr) // Generate 3GPP default radiation pattern
    {
        // Single antenna element vertical radiation pattern cut in dB
        arma::Col<dtype> Y = ant.elevation_grid;
        for (dtype *py = Y.begin(); py < Y.end(); ++py)
        {
            double y = double(*py) * rad2deg / 65.0;
            y = 12.0 * y * y;
            *py = (y > 30.0) ? dtype(30.0) : dtype(y);
        }

        // Full pattern (normalized to 8 dBi gain using factor 2.51..)
        dtype *ptr = ant.e_theta_re.memptr(), *py = Y.memptr(), *px = ant.azimuth_grid.memptr();
        for (arma::uword ia = 0ULL; ia < n_az; ++ia)
        {
            double x = double(*px++) * rad2deg / 65.0;
            x = 12.0 * x * x;
            x = (x > 30.0) ? 30.0 : x;

            for (arma::uword ie = 0ULL; ie < n_el; ++ie)
            {
                double z = double(py[ie]) + x;
                z = z > 30.0 ? -30.0 : -z;
                *ptr++ = dtype(2.511886431509580 * std::sqrt(std::pow(10.0, 0.1 * z))); // 8dBi gain
            }
        }
        Y.reset();

        // Adjust polarization
        if (pol == 2 || pol == 5)
        {
            ant.copy_element(0, 1);
            ant.rotate_pattern(90.0, zero, zero, 2, 1);
        }
        else if (pol == 3 || pol == 6)
        {
            ant.copy_element(0, 1);
            ant.rotate_pattern(dtype(45.0), zero, zero, 2, 0);
            ant.rotate_pattern(dtype(-45.0), zero, zero, 2, 1);
        }
    }

    // Duplicate the existing elements in z-direction (vertical stacking)
    arma::uword n_elements = ant.n_elements();
    if (M > 1ULL)
        for (arma::uword source = n_elements; source > 0ULL; source--)
        {
            arma::uword i_start = n_elements + source - 1ULL;
            arma::uword i_end = M * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

    // Calculate the element z-position
    arma::Col<dtype> z_position(M);
    if (M > 1ULL)
    {
        z_position = arma::linspace<arma::Col<dtype>>(zero, dtype(M - 1ULL) * spacing * dtype(wavelength), M);
        z_position = z_position - arma::mean(z_position);

        for (arma::uword m = 0ULL; m < M; ++m)
            for (arma::uword n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(2ULL, m * n_elements + n) = z_position.at(m);
    }

    // Apply element coupling for polarization indicators 4, 5, and 6
    if (pol > 3 && M > 1ULL)
    {
        double tmp = 2.0 * pi * std::sin(double(tilt) * deg2rad) / wavelength;
        arma::Col<dtype> cpl_re = z_position * dtype(tmp);
        tmp = 1.0 / std::sqrt(double(M));
        arma::Col<dtype> cpl_im = arma::sin(cpl_re) * dtype(tmp);
        cpl_re = arma::cos(cpl_re) * dtype(tmp);

        ant.coupling_re.zeros(n_elements * M, n_elements);
        ant.coupling_im.zeros(n_elements * M, n_elements);

        for (arma::uword m = 0ULL; m < M; ++m)
            for (arma::uword n = 0ULL; n < n_elements; ++n)
            {
                ant.coupling_re.at(m * n_elements + n, n) = cpl_re.at(m);
                ant.coupling_im.at(m * n_elements + n, n) = cpl_im.at(m);
            }

        ant = ant.combine_pattern();
        M = 1ULL;
    }

    // Duplicate the existing elements in y-direction (horizontal stacking)
    n_elements = ant.n_elements();
    if (N > 1ULL)
    {
        for (arma::uword source = n_elements; source > 0ULL; source--)
        {
            arma::uword i_start = n_elements + source - 1ULL;
            arma::uword i_end = N * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> y_position = arma::linspace<arma::Col<dtype>>(zero, dtype(N - 1ULL) * spacing * dtype(wavelength), N);
        y_position = y_position - arma::mean(y_position);

        for (arma::uword m = 0ULL; m < N; ++m)
            for (arma::uword n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(1ULL, m * n_elements + n) = y_position.at(m);
    }

    // Duplicate panels in z-direction (vertical panel stacking)
    n_elements = ant.n_elements();
    if (Mg > 1ULL)
    {
        for (arma::uword source = n_elements; source > 0ULL; source--)
        {
            arma::uword i_start = n_elements + source - 1ULL;
            arma::uword i_end = Mg * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> zg_position = arma::linspace<arma::Col<dtype>>(zero, dtype(Mg - 1ULL) * dgv * dtype(wavelength), Mg);
        zg_position = zg_position - arma::mean(zg_position);

        for (arma::uword mg = 0ULL; mg < Mg; ++mg)
            for (arma::uword n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(2ULL, mg * n_elements + n) += zg_position.at(mg);
    }

    // Duplicate panels in y-direction (horizontal panel stacking)
    n_elements = ant.n_elements();
    if (Ng > 1)
    {
        for (arma::uword source = n_elements; source > 0; source--)
        {
            arma::uword i_start = n_elements + source - 1ULL;
            arma::uword i_end = Ng * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> yg_position = arma::linspace<arma::Col<dtype>>(zero, dtype(Ng - 1ULL) * dgh * dtype(wavelength), Ng);
        yg_position = yg_position - arma::mean(yg_position);

        for (arma::uword mg = 0ULL; mg < Ng; ++mg)
            for (arma::uword n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(1ULL, mg * n_elements + n) += yg_position.at(mg);
    }

    ant.name = "3gpp";
    return ant;
}

template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_3GPP(arma::uword M, arma::uword N, float center_freq,
                                                                            unsigned pol, float tilt, float spacing,
                                                                            arma::uword Mg, arma::uword Ng, float dgv, float dgh,
                                                                            const quadriga_lib::arrayant<float> *pattern, float res);

template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_3GPP(arma::uword M, arma::uword N, double center_freq,
                                                                             unsigned pol, double tilt, double spacing,
                                                                             arma::uword Mg, arma::uword Ng, double dgv, double dgh,
                                                                             const quadriga_lib::arrayant<double> *pattern, double res);
/*!MD
# generate_arrayant_multibeam
Generate a planar multi-element antenna array with multiple beam directions

- Returns an M×N planar array with beamforming weights computed via maximum-ratio transmission (MRT).
- MRT is optimal for a single beam; approximate when multiple beams are specified.
- Weights control relative beam contribution; only their ratios matter, not absolute values.
- If `separate_beams = true`, each angle pair produces an independent beam (weights ignored).
- If `apply_weights = true`, beamforming weights are baked into the element coupling matrix.
- Per-element pattern shape is controlled by `az_3dB`, `el_3dB`, and `rear_gain_lin`; see [[generate_arrayant_custom]].

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_multibeam(
    arma::uword M = 1,
    arma::uword N = 1,
    arma::Col<dtype> az = {0.0},
    arma::Col<dtype> el = {0.0},
    arma::Col<dtype> weight = {1.0},
    dtype center_freq = 299792458.0,
    unsigned pol = 1,
    dtype spacing = 0.5,
    dtype az_3dB = 120.0,
    dtype el_3dB = 120.0,
    dtype rear_gain_lin = 0.0,
    dtype res = 1.0,
    bool separate_beams = false,
    bool apply_weights = false);
```

## Inputs:
- **`M`** *(optional)* — Number of vertical (row) elements
- **`N`** *(optional)* — Number of horizontal (column) elements
- **`az`** *(optional)* — Azimuth beam angles in degrees; `[n_beams]`
- **`el`** *(optional)* — Elevation beam angles in degrees; `[n_beams]`
- **`weight`** *(optional)* — Per-beam scaling factors (normalized to sum = 1); `[n_beams]`
- **`center_freq`** *(optional)* — Center frequency
- **`pol`** *(optional)* — Polarization mode:<br><br>
   `pol` | Description | Elements 
  -------|-------------|----------
   1 | Vertical polarization | NM 
   2 | H/V polarization | 2NM 
   3 | ±45° polarization | 2NM 
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`az_3dB`** *(optional)* — Per-element azimuth 3dB beamwidth in degrees
- **`el_3dB`** *(optional)* — Per-element elevation 3dB beamwidth in degrees
- **`rear_gain_lin`** *(optional)* — Per-element front-to-back gain ratio (linear)
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees
- **`separate_beams`** *(optional)* — If `true`, generate one independent beam per angle pair
- **`apply_weights`** *(optional)* — If `true`, bake beamforming weights into the coupling matrix

## Returns:
- `quadriga_lib::arrayant<dtype>` — Multibeam planar array antenna object
MD!*/

// Generate multi-beam antenna
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_multibeam(arma::uword M, arma::uword N, arma::Col<dtype> az, arma::Col<dtype> el, arma::Col<dtype> weight,
                                                                        dtype center_freq, unsigned pol, dtype spacing, dtype az_3dB, dtype el_3dB,
                                                                        dtype rear_gain_lin, dtype res, bool separate_beams, bool apply_weights)
{
    dtype zero = (dtype)0.0;

    if (az.n_elem == 0)
        throw std::invalid_argument("Must have at least one beam direction.");

    arma::uword n_beams = az.n_elem;

    if (el.n_elem != n_beams)
        throw std::invalid_argument("Azimuth and elevation must have the same number of elements.");

    if (weight.n_elem != 0 && weight.n_elem != n_beams)
        throw std::invalid_argument("Beam weights must be empty or match the number of beams.");

    if (weight.n_elem == 0)
        weight.ones(n_beams);

    // Fix input ranges
    M = (M == 0) ? 1 : M;
    N = (N == 0) ? 1 : N;
    center_freq = (center_freq <= zero) ? (dtype)299792458.0 : center_freq;
    pol = (pol == 0 || pol > 3) ? 1 : pol;
    spacing = (spacing < zero) ? (dtype)0.5 : spacing;
    az_3dB = (az_3dB <= zero) ? (dtype)90.0 : az_3dB;
    el_3dB = (el_3dB <= zero) ? (dtype)90.0 : el_3dB;
    rear_gain_lin = (rear_gain_lin < zero) ? zero : rear_gain_lin;

    // Normalized distance = 5000 wavelengths
    dtype dist = dtype(5000.0 * 299792458.0) / center_freq;

    // Convert from DEG to RAD
    qd_multiply_scalar((dtype)0.017453292519943, az.memptr(), n_beams);
    qd_multiply_scalar((dtype)0.017453292519943, el.memptr(), n_beams);

    // Convert angles to cartesian
    arma::Mat<dtype> cart(3, n_beams, arma::fill::none);
    qd_geo2cart_interleaved<dtype>(n_beams, cart.memptr(), az.memptr(), el.memptr(), nullptr);

    // Build pattern
    auto pattern = quadriga_lib::generate_arrayant_custom<dtype>(az_3dB, el_3dB, rear_gain_lin, res);

    // Assemble array antenna
    quadriga_lib::arrayant<dtype> ant = quadriga_lib::generate_arrayant_3GPP<dtype>(M, N, center_freq, pol, zero, spacing, 1, 1, 0.5, 0.5, &pattern);

    arma::uword n_elements = ant.n_elements();
    arma::Mat<dtype> cpl_eye = ant.coupling_re;
    arma::Mat<dtype> cpl_zero = ant.coupling_im;

    // Probe array
    auto probe = quadriga_lib::generate_arrayant_omni<dtype>(30.0);

    // Phase shifters
    arma::Mat<dtype> cpl_re, cpl_im;

    // Get MRT weights
    const arma::Col<dtype> path_gain = {1.0};
    const arma::Col<dtype> path_length = {1000.0};
    const arma::Col<dtype> path_angle = {0.0};
    const arma::Col<dtype> pilot_grid = {0.0};
    arma::Mat<dtype> path_pol = {1.0, zero, zero, zero, zero, zero, -1.0, zero};
    path_pol = path_pol.t();

    // Storage for the coefficients
    arma::Cube<dtype> coeff_re(1, n_elements, n_beams);
    arma::Cube<dtype> coeff_im(1, n_elements, n_beams);
    arma::Cube<dtype> delay(1, n_elements, n_beams);

    // Variables for the optimization loop
    arma::Col<dtype> gain(n_beams);                                                // zeros
    arma::Mat<dtype> search_direction(2, n_beams, arma::fill::value((dtype)0.05)); // rad
    arma::Mat<dtype> angles(2, n_beams);
    for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
        angles(0, i_beam) = az[i_beam], angles(1, i_beam) = el[i_beam];

    // Beam pointing optimization loop
    arma::uword lp_max = separate_beams ? 7 : 25;
    for (arma::uword lp = 0; lp < lp_max; ++lp)
    {
        // Set the angles for testing
        arma::Mat<dtype> test_angles = angles;
        if (lp % 2 == 1) // Test azimuth
            for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
                test_angles(0, i_beam) += search_direction(0, i_beam);
        else if (lp != 0) // Test elevation
            for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
                test_angles(1, i_beam) += search_direction(1, i_beam);

        // Calculate the coefficients for each beam
        ant.coupling_re = cpl_eye;
        ant.coupling_im = cpl_zero;
        for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
        {
            arma::Col<dtype> aod = {test_angles(0, i_beam)};
            arma::Col<dtype> eod = {test_angles(1, i_beam)};

            // Temporary storage for the coefficients
            arma::Cube<dtype> coeff_re_tmp, coeff_im_tmp, delay_tmp;

            // Receiver positions
            dtype R[3];
            qd_geo2cart_interleaved<dtype>(1, R, aod.memptr(), eod.memptr(), nullptr);
            qd_multiply_scalar(dist, R, 3);

            quadriga_lib::get_channels_planar(&ant, &probe, zero, zero, zero, zero, zero, zero,
                                              R[0], R[1], R[2], zero, zero, zero, &aod, &eod,
                                              &path_angle, &path_angle, &path_gain, &path_length, &path_pol,
                                              &coeff_re_tmp, &coeff_im_tmp, &delay_tmp, center_freq);

            for (arma::uword i_elem = 0; i_elem < n_elements; ++i_elem)
            {
                coeff_re(0, i_elem, i_beam) = coeff_re_tmp[i_elem] * weight[i_beam];
                coeff_im(0, i_elem, i_beam) = coeff_im_tmp[i_elem] * weight[i_beam];
            }
        }

        // Calculate phase shifts
        if (separate_beams)
        {
            ant.coupling_re.set_size(n_elements, n_beams);
            ant.coupling_im.set_size(n_elements, n_beams);
            for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
                for (arma::uword i_elem = 0; i_elem < n_elements; ++i_elem)
                {
                    std::complex<dtype> w(coeff_re(0, i_elem, i_beam), coeff_im(0, i_elem, i_beam));

                    dtype mag = std::abs(w);
                    if (mag > std::numeric_limits<dtype>::epsilon() * dtype(1e3))
                        w /= mag;
                    else
                        w = std::complex<dtype>(dtype(1.0), dtype(0.0));

                    ant.coupling_re(i_elem, i_beam) = w.real();
                    ant.coupling_im(i_elem, i_beam) = -w.imag(); // Use conjugate
                }
        }
        else // Combine the coefficients into a single beam
        {
            arma::Cube<std::complex<dtype>> hmat;
            quadriga_lib::baseband_freq_response<dtype>(&coeff_re, &coeff_im, &delay, &pilot_grid, 1.0, nullptr, nullptr, &hmat);

            ant.coupling_re.set_size(n_elements, 1);
            ant.coupling_im.set_size(n_elements, 1);
            for (arma::uword i_elem = 0; i_elem < n_elements; ++i_elem)
            {
                std::complex<dtype> w = hmat[i_elem];

                dtype mag = std::abs(w);
                if (mag > std::numeric_limits<dtype>::epsilon() * dtype(1e3))
                    w /= mag;
                else
                    w = std::complex<dtype>(dtype(1.0), dtype(0.0));

                ant.coupling_re[i_elem] = w.real();
                ant.coupling_im[i_elem] = -w.imag(); // Use conjugate
            }
        }

        // Calculate gain per beam
        for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
        {
            arma::Col<dtype> aod = {az[i_beam]};
            arma::Col<dtype> eod = {el[i_beam]};
            arma::Cube<dtype> coeff_re_tmp, coeff_im_tmp, delay_tmp;

            dtype Rx = cart.at(0, i_beam) * dist;
            dtype Ry = cart.at(1, i_beam) * dist;
            dtype Rz = cart.at(2, i_beam) * dist;

            quadriga_lib::get_channels_planar(&ant, &probe, zero, zero, zero, zero, zero, zero,
                                              Rx, Ry, Rz, zero, zero, zero, &aod, &eod,
                                              &path_angle, &path_angle, &path_gain, &path_length, &path_pol,
                                              &coeff_re_tmp, &coeff_im_tmp, &delay_tmp, center_freq);

            dtype a = separate_beams ? coeff_re_tmp[i_beam] : coeff_re_tmp[0];
            dtype b = separate_beams ? coeff_im_tmp[i_beam] : coeff_im_tmp[0];
            dtype gain_new = a * a + b * b;

            // auto tmp = test_angles * 57.29577951308232;
            // tmp.print("test angles");
            // std::cout << "IT = " << lp << ", B = " << i_beam << ", Gain = " << 10.0*std::log10( gain[i_beam]) << " > " << 10.0*std::log10(gain_new) << std::endl;

            // Update search direction for next iteration
            if (gain_new > gain[i_beam]) // Update gain and angles, keep search direction
            {
                gain[i_beam] = gain_new;
                angles(0, i_beam) = test_angles(0, i_beam);
                angles(1, i_beam) = test_angles(1, i_beam);
                cpl_re = ant.coupling_re;
                cpl_im = ant.coupling_im;
            }
            else // Discard new angles, change search direction and step size
            {
                if (lp % 2 == 1) // azimuth
                    search_direction(0, i_beam) *= (dtype)-0.382;
                else // elevation
                    search_direction(1, i_beam) *= (dtype)-0.382;
            }
        }
    }

    // auto tmp = angles * 57.29577951308232;
    // tmp.print("final angles");

    // arma::Col<dtype> tmp2 = arma::log10(gain);
    // tmp2 *= 10.0;
    // tmp2.print("final gain");

    // Apply polarization
    if (pol > 1)
    {
        auto element_pos = ant.element_pos;

        // Extend array for second polarization
        arma::uvec dest = arma::regspace<arma::uvec>(n_elements, 2 * n_elements - 1);
        ant.copy_element(0, dest);

        // Update polarization
        if (pol == 2)
        {
            ant.rotate_pattern(90.0, 0.0, 0.0, 2, (unsigned)n_elements);
            dest = arma::regspace<arma::uvec>(n_elements + 1, 2 * n_elements - 1);
            ant.copy_element(n_elements, dest);
        }
        else if (pol == 3)
        {
            ant.rotate_pattern(45.0, 0.0, 0.0, 2, 0);
            dest = arma::regspace<arma::uvec>(1, n_elements - 1);
            ant.copy_element(0, dest);

            ant.rotate_pattern(-45.0, 0.0, 0.0, 2, (unsigned)n_elements);
            dest = arma::regspace<arma::uvec>(n_elements + 1, 2 * n_elements - 1);
            ant.copy_element(n_elements, dest);

            std::memcpy(ant.element_pos.memptr(), element_pos.memptr(), 3 * n_elements * sizeof(dtype));
        }

        // Update element positions
        std::memcpy(ant.element_pos.colptr(n_elements), element_pos.memptr(), 3 * n_elements * sizeof(dtype));

        // Set coupling weights
        if (separate_beams)
        {
            ant.coupling_re.zeros(2 * n_elements, 2 * n_beams);
            ant.coupling_im.zeros(2 * n_elements, 2 * n_beams);

            for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
            {
                std::memcpy(ant.coupling_re.colptr(2 * i_beam), cpl_re.colptr(i_beam), n_elements * sizeof(dtype));
                std::memcpy(ant.coupling_im.colptr(2 * i_beam), cpl_im.colptr(i_beam), n_elements * sizeof(dtype));
                std::memcpy(ant.coupling_re.colptr(2 * i_beam + 1) + n_elements, cpl_re.colptr(i_beam), n_elements * sizeof(dtype));
                std::memcpy(ant.coupling_im.colptr(2 * i_beam + 1) + n_elements, cpl_im.colptr(i_beam), n_elements * sizeof(dtype));
            }
        }
        else
        {
            ant.coupling_re.zeros(2 * n_elements, 2);
            ant.coupling_im.zeros(2 * n_elements, 2);

            std::memcpy(ant.coupling_re.memptr(), cpl_re.memptr(), n_elements * sizeof(dtype));
            std::memcpy(ant.coupling_im.memptr(), cpl_im.memptr(), n_elements * sizeof(dtype));
            std::memcpy(ant.coupling_re.colptr(1) + n_elements, cpl_re.memptr(), n_elements * sizeof(dtype));
            std::memcpy(ant.coupling_im.colptr(1) + n_elements, cpl_im.memptr(), n_elements * sizeof(dtype));
        }
    }
    else // V-Pol only
    {
        ant.coupling_re = cpl_re;
        ant.coupling_im = cpl_im;
    }

    // Make sure the generated antenna is valid
    auto err = ant.is_valid(false);
    if (!err.empty())
        throw std::invalid_argument(err);

    if (apply_weights)
        ant = ant.combine_pattern();

    return ant;
}

template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_multibeam(arma::uword M, arma::uword N, arma::Col<float> az, arma::Col<float> el, arma::Col<float> weight,
                                                                                 float center_freq, unsigned pol, float spacing, float az_3dB, float el_3dB,
                                                                                 float rear_gain_lin, float res, bool separate_beams, bool apply_weights);

template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_multibeam(arma::uword M, arma::uword N, arma::Col<double> az, arma::Col<double> el, arma::Col<double> weight,
                                                                                  double center_freq, unsigned pol, double spacing, double az_3dB, double el_3dB,
                                                                                  double rear_gain_lin, double res, bool separate_beams, bool apply_weights);