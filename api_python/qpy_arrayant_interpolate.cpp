// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# interpolate
Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the antenna and the resulting polarization rotation applied
- Single-frequency mode (3D pattern arrays and no `frequency`): outputs `(n_out, n_ang)`; `dist` and `local_angles`
  add outputs. Multi-frequency mode (4D pattern arrays, or `frequency` given): for each target frequency, the two
  entries whose center frequencies bracket it are interpolated and blended; frequencies outside the entries take the
  nearest entry; outputs `(n_out, n_ang, n_freq)`; a 3D pattern is the same at every frequency
- `azimuth` of shape `(1, 0)` or `(n_out, 0)` gives empty outputs; `np.array([])` raises `ValueError`
- NaN or infinite angles give NaN in all outputs of that angle and output, without an error
- Kernels: GENERIC computes in double precision; AVX2 in single precision, with errors of up to about 2e-6 of the peak
  field value and 2e-6 rad in the angles; at a distance d (in rad) from a pole of the local frame up to about
  2e-7 / d rad in the local azimuth and the polarization angle and 1e-7 / d rad (at most 4e-4 rad) in the local
  elevation; auto selects GENERIC in this version, which may change in a later version
- The flags `dist`, `local_angles`, `fast_access` and `complex` take a bool or `numpy.bool_`; `None` selects the
  default, any other type raises `TypeError`

## Usage:
```
# Single-frequency mode
V_re, V_im, H_re, H_im = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, element, \
    orientation, element_pos, use_kernel=0 )

# Single-frequency mode with the projected distance and the local angles
V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma = quadriga_lib.arrayant.interpolate( \
    arrayant, azimuth, elevation, ..., dist=True, local_angles=True )

# Multi-frequency mode
V_re, V_im, H_re, H_im = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, ..., frequency=freqs )

# Complex output
V, H = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, ..., complex=True )
```

## Inputs:
- **`arrayant`** — Dict containing the arrayant data; layout as documented in [[generate]]; 4D pattern arrays contain
  a frequency-dependent model
- **`azimuth`** — Azimuth angles in GCS; `(1, n_ang)` (same angles for all outputs) or `(n_out, n_ang)` (one angle
  per output)
- **`elevation`** — Elevation angles in GCS; same shape as `azimuth`
- **`element`** — Element indices, 0-based, duplicates allowed; `(n_out,)`; default: `None` (all elements)
- **`orientation`** — Euler angles (bank, tilt, heading) of the antenna; `(3, 1)`, `(3, n_out)`, `(3, 1, n_ang)` or
  `(3, n_out, n_ang)`; default: `None` (no rotation)
- **`element_pos`** — Element positions in local antenna coordinates, used for `dist`; `(3, n_out)`; default: `None`
  (the positions of the arrayant)
- **`frequency`** — Target frequencies in Hz; selects the multi-frequency mode; `(n_freq,)`; default: `None` (the
  `center_freq` of the entries)
- **`dist`** — Also return the projected distance; single-frequency mode only; default: `False`
- **`local_angles`** — Also return the local angles and the polarization rotation angle; single-frequency mode only;
  default: `False`
- **`fast_access`** — Require the pattern arrays to be float64 and Fortran-contiguous, so that they are read without
  a copy; raises otherwise. With `False`, such arrays are read without a copy as well and others are copied;
  default: `False`
- **`use_kernel`** — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (raises if unavailable); `int`; default: 0
- **`complex`** — Return `V` and `H` as complex arrays instead of `V_re, V_im, H_re, H_im`; default: `False`

## Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `(n_out, n_ang)`; multi-frequency mode:
  `(n_out, n_ang, n_freq)`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; same shape as `V_re`
- **`V`** / **`H`** — e-theta and e-phi field, complex; returned instead of the real parts when `complex` is `True`
- **`dist`** — Distance of each element from the wavefront plane through the origin of the antenna; negative for
  elements closer to the source; `(n_out, n_ang)`; returned when `dist` is `True`
- **`azimuth_loc`** / **`elevation_loc`** — Azimuth and elevation in the local antenna frame; `(n_out, n_ang)`;
  returned when `local_angles` is `True`
- **`gamma`** — Polarization rotation angle from the global to the local frame; `(n_out, n_ang)`; returned when
  `local_angles` is `True`

## See also:
- [[generate]] (arrayant dict layout)
- [[qdant_read]] (load arrayant data)
- [[generate_speaker]] (typical source of multi-frequency arrayants)
MD!*/

py::tuple arrayant_interpolate(const py::dict &arrayant,
                               const py::array_t<double> &azimuth,
                               const py::array_t<double> &elevation,
                               py::handle element,
                               py::handle orientation,
                               py::handle element_pos,
                               py::handle frequency,
                               py::handle dist,
                               py::handle local_angles,
                               py::handle fast_access,
                               int use_kernel,
                               py::handle complex)
{
    const bool want_dist = qd_python_flag(dist, false, "dist");
    const bool want_local_angles = qd_python_flag(local_angles, false, "local_angles");
    const bool strict = qd_python_flag(fast_access, false, "fast_access");
    const bool want_complex = qd_python_flag(complex, false, "complex");

    const auto azimuth_a = qd_python_numpy2arma_Mat(azimuth, true);
    const auto elevation_a = qd_python_numpy2arma_Mat(elevation, true);
    const auto element_a = qd_python_numpy2arma_Col<arma::uword>(element, true);
    const auto orientation_a = qd_python_numpy2arma_Cube<double>(orientation, true);
    const auto element_pos_a = qd_python_numpy2arma_Mat<double>(element_pos, true);
    const auto frequency_a = qd_python_numpy2arma_Col<double>(frequency, true);

    const arma::uvec *p_element = element_a.is_empty() ? nullptr : &element_a;
    const arma::cube *p_orientation = orientation_a.is_empty() ? nullptr : &orientation_a;
    const arma::mat *p_element_pos = element_pos_a.is_empty() ? nullptr : &element_pos_a;

    const auto ant = qd_python_dict2arrayant_multi(arrayant, true, strict);
    const arma::uword n_out = element_a.is_empty() ? ant[0].n_elements() : element_a.n_elem;
    const arma::uword n_ang = azimuth_a.n_cols;

    if (ant.size() > 1 || !frequency_a.is_empty()) // Multi-frequency mode
    {
        if (want_dist || want_local_angles)
            throw std::invalid_argument("Outputs 'dist' and 'local_angles' are not available in the multi-frequency mode.");

        const arma::uword n_freq = frequency_a.is_empty() ? (arma::uword)ant.size() : frequency_a.n_elem;
        arma::cube V_re, V_im, H_re, H_im;
        py::array_t<double> V_re_py, V_im_py, H_re_py, H_im_py;
        if (!want_complex)
        {
            V_re_py = qd_python_init_output(n_out, n_ang, n_freq, &V_re);
            V_im_py = qd_python_init_output(n_out, n_ang, n_freq, &V_im);
            H_re_py = qd_python_init_output(n_out, n_ang, n_freq, &H_re);
            H_im_py = qd_python_init_output(n_out, n_ang, n_freq, &H_im);
        }

        quadriga_lib::arrayant_interpolate_multi<double>(ant, azimuth_a, elevation_a, V_re, V_im, H_re, H_im,
                                                         frequency_a.is_empty() ? nullptr : &frequency_a,
                                                         p_element, p_orientation, p_element_pos, use_kernel);

        if (want_complex)
            return py::make_tuple(qd_python_copy2numpy<double, std::complex<double>>(&V_re, &V_im),
                                  qd_python_copy2numpy<double, std::complex<double>>(&H_re, &H_im));
        return py::make_tuple(V_re_py, V_im_py, H_re_py, H_im_py);
    }

    arma::mat V_re, V_im, H_re, H_im, dist_a, azimuth_loc, elevation_loc, gamma;
    py::array_t<double> V_re_py, V_im_py, H_re_py, H_im_py, dist_py, azimuth_loc_py, elevation_loc_py, gamma_py;
    if (!want_complex) // With complex output, the C++ function sizes the outputs, which are then merged
    {
        V_re_py = qd_python_init_output(n_out, n_ang, &V_re);
        V_im_py = qd_python_init_output(n_out, n_ang, &V_im);
        H_re_py = qd_python_init_output(n_out, n_ang, &H_re);
        H_im_py = qd_python_init_output(n_out, n_ang, &H_im);
    }
    if (want_dist)
        dist_py = qd_python_init_output(n_out, n_ang, &dist_a);
    if (want_local_angles)
    {
        azimuth_loc_py = qd_python_init_output(n_out, n_ang, &azimuth_loc);
        elevation_loc_py = qd_python_init_output(n_out, n_ang, &elevation_loc);
        gamma_py = qd_python_init_output(n_out, n_ang, &gamma);
    }

    quadriga_lib::arrayant_interpolate<double>(ant[0], azimuth_a, elevation_a, V_re, V_im, H_re, H_im,
                                               p_element, p_orientation, p_element_pos,
                                               want_dist ? &dist_a : nullptr,
                                               want_local_angles ? &azimuth_loc : nullptr,
                                               want_local_angles ? &elevation_loc : nullptr,
                                               want_local_angles ? &gamma : nullptr, use_kernel);

    // Output tuple in the documented order
    py::list out;
    if (want_complex)
    {
        out.append(qd_python_copy2numpy<double, std::complex<double>>(&V_re, &V_im));
        out.append(qd_python_copy2numpy<double, std::complex<double>>(&H_re, &H_im));
    }
    else
    {
        out.append(V_re_py);
        out.append(V_im_py);
        out.append(H_re_py);
        out.append(H_im_py);
    }
    if (want_dist)
        out.append(dist_py);
    if (want_local_angles)
    {
        out.append(azimuth_loc_py);
        out.append(elevation_loc_py);
        out.append(gamma_py);
    }
    return py::tuple(out);
}

// pybind11 declaration:
// m.def("interpolate", &arrayant_interpolate,
//       py::arg("arrayant"),
//       py::arg("azimuth").none(false),
//       py::arg("elevation").none(false),
//       py::arg("element") = py::none(),
//       py::arg("orientation") = py::none(),
//       py::arg("element_pos") = py::none(),
//       py::arg("frequency") = py::none(),
//       py::arg("dist") = false,
//       py::arg("local_angles") = false,
//       py::arg("fast_access") = false,
//       py::arg("use_kernel") = 0,
//       py::arg("complex") = false);
