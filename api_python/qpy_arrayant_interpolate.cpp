// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# interpolate
Interpolate polarimetric array antenna field patterns (single- and multi-frequency)

- Interpolates the complex e-theta (V) and e-phi (H) field components at the requested azimuth / elevation angles
- Single-frequency input (3D pattern fields) returns 2D outputs `(n_out, n_ang)`; the optional `dist` and
  `local_angles` outputs are available only on this path
- Multi-frequency input (4D pattern fields) interpolates spatially and across frequency; for each
  target frequency the two bracketing `center_freq` entries are blended via SLERP
- Passing `frequency` adds a frequency dimension to the output `(n_out, n_ang, n_freq_out)`; for a
  single-frequency antenna the spatial result is simply replicated across the requested frequencies
- `azimuth` / `elevation` of shape `(1, n_ang)` apply the same angles to all elements (planar
  wave); shape `(n_out, n_ang)` gives per-element angles (spherical wave)

## Usage:
```
# Single-frequency, separate real / imaginary parts
vr, vi, hr, hi = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation )

# Complex-valued output
v, h = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, complex=True )

# Projected distance / local angles (single-frequency only)
vr, vi, hr, hi, dist = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, dist=True )
vr, vi, hr, hi, azimuth_loc, elevation_loc, gamma = quadriga_lib.arrayant.interpolate( arrayant, azimuth,
    elevation, orientation=ori, local_angles=True )

# Element selection, orientation, element positions
vr, vi, hr, hi = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, element, orientation, element_pos )

# Multi-frequency interpolation — output gains a frequency axis
vr, vi, hr, hi = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, frequency=freqs )
```

## Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [[generate]]; pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`azimuth`** — Azimuth angles in rad, in [-pi, pi]; `(1, n_ang)` for shared angles (planar
  wave) or `(n_out, n_ang)` for per-element angles (spherical wave)
- **`elevation`** — Elevation angles in rad, in [-pi/2, pi/2]; same shape as `azimuth`
- **`element`** — Element indices to interpolate; duplicates allowed; `(n_out,)` or a
  list of int; `None` or empty uses all elements (`n_out = n_elements`); default: `None`
- **`orientation`** — antenna orientation (bank, tilt, heading) in rad; shape `(3, 1)`,
  `(3, n_out)`, `(3, 1, n_ang)`, or `(3, n_out, n_ang)`; `None` uses East-facing broadside;
  default: `None`
- **`element_pos`** — Alternative element (x,y,z) positions in m; `(3, n_out)`; `None` falls back
  to `arrayant["element_pos"]` (or zeros); default: `None`
- **`frequency`** — Target frequencies in Hz; `(n_freq_out,)` or scalar; adds a frequency axis to
  the output; for 4D input it interpolates between entries, for 3D input it replicates the spatial
  result; if omitted for 4D input the entries' `center_freq` values are used; default: `None`
- **`complex`** — If `True`, return complex-valued arrays instead of separate real / imaginary
  parts; default: `False`
- **`dist`** — If `True`, also return the projected distances; single-frequency input only and
  not allowed together with `frequency`; default: `False`
- **`local_angles`** — If `True`, also return the antenna-local angles; single-frequency input
  only and not allowed together with `frequency`; default: `False`
- **`fast_access`** — If `True`, require the pattern arrays to be double and column-major
  (Fortran-contiguous) so they can be read without a copy; raises if not; default: `False`

## Outputs:
Returned as a tuple; its length depends on `complex`, `dist`, and `local_angles`.
- **`vr`, `vi`, `hr`, `hi`** — Real / imaginary parts of the interpolated e-theta (V) and e-phi
  (H) field components; `(n_out, n_ang)`, or `(n_out, n_ang, n_freq_out)` when `frequency` is
  given; returned when `complex` is `False`
- **`v`, `h`** — Complex-valued e-theta (V) and e-phi (H) components; same shape as above;
  returned instead of `vr, vi, hr, hi` when `complex` is `True`
- **`dist`** — Projected distances between elements on the wavefront plane, used for phase
  computation; `(n_out, n_ang)`; returned only when `dist` is `True`
- **`azimuth_loc`, `elevation_loc`, `gamma`** — Azimuth, elevation, and polarization rotation
  angles in the local element frame, in rad; `(n_out, n_ang)` each; returned only when
  `local_angles` is `True`

## See also:
- [[qdant_read]] / [[qdant_write]] (load / save arrayant data)
- [[generate]] (arrayant struct layout)
- [[generate_speaker]] (typical multi-frequency struct array source)
MD!*/

py::tuple arrayant_interpolate(const py::dict &arrayant,
                               py::handle azimuth,
                               py::handle elevation,
                               py::handle element,
                               py::handle orientation,
                               py::handle element_pos,
                               py::handle frequency,
                               bool complex,
                               bool dist,
                               bool local_angles,
                               bool fast_access)
{
    // Shared inputs
    const auto az = qd_python_numpy2arma_Mat<double>(azimuth, true);
    const auto el = qd_python_numpy2arma_Mat<double>(elevation, true);
    const arma::uvec element_ind = qd_python_numpy2arma_Col<arma::uword>(element, true);
    const auto ori = qd_python_numpy2arma_Cube<double>(orientation, true);
    const auto elpos = qd_python_numpy2arma_Mat<double>(element_pos, true);
    const arma::vec freq = qd_python_numpy2arma_Col<double>(frequency, true);

    if (az.n_elem == 0)
        throw std::invalid_argument("Azimuth angles cannot be empty.");

    const arma::uword n_ang = az.n_cols;

    // Parse the (possibly frequency-dependent) antenna via the unified multi-freq reader
    auto ant_vec = qd_python_dict2arrayant_multi(arrayant, true, fast_access, true);

    // The multi-frequency code path covers 4D input as well as any 3D/4D input with a frequency
    const bool use_multi = (ant_vec.size() > 1) || (freq.n_elem > 0);

    // Multi-frequency / frequency-replication path
    if (use_multi)
    {
        if (dist)
            throw std::invalid_argument("Interpolate: 'dist' is not supported when 'frequency' is given or for multi-frequency arrayants.");
        if (local_angles)
            throw std::invalid_argument("Interpolate: 'local_angles' is not supported when 'frequency' is given or for multi-frequency arrayants.");

        // Target frequencies. If omitted for a 4D antenna, use the entries' center_freq values.
        arma::vec freq_target = freq;
        if (freq_target.n_elem == 0)
        {
            freq_target.set_size(ant_vec.size());
            for (size_t i = 0; i < ant_vec.size(); ++i)
                freq_target[i] = ant_vec[i].center_frequency;
        }

        const arma::uword n_out = (element_ind.n_elem == 0) ? ant_vec[0].n_elements()
                                                            : element_ind.n_elem;
        const arma::uword n_freq_out = freq_target.n_elem;

        arma::cube V_re, V_im, H_re, H_im;
        py::array_t<double> V_re_py, V_im_py, H_re_py, H_im_py;
        if (complex)
        {
            V_re.set_size(n_out, n_ang, n_freq_out);
            V_im.set_size(n_out, n_ang, n_freq_out);
            H_re.set_size(n_out, n_ang, n_freq_out);
            H_im.set_size(n_out, n_ang, n_freq_out);
        }
        else
        {
            V_re_py = qd_python_init_output(n_out, n_ang, n_freq_out, &V_re);
            V_im_py = qd_python_init_output(n_out, n_ang, n_freq_out, &V_im);
            H_re_py = qd_python_init_output(n_out, n_ang, n_freq_out, &H_re);
            H_im_py = qd_python_init_output(n_out, n_ang, n_freq_out, &H_im);
        }

        quadriga_lib::arrayant_interpolate_multi<double>(ant_vec, &az, &el, &freq_target, &V_re, &V_im, &H_re, &H_im,
                                                         element_ind, &ori, &elpos, false);

        if (complex)
            return py::make_tuple(qd_python_copy2numpy(V_re, V_im), qd_python_copy2numpy(H_re, H_im));
        return py::make_tuple(V_re_py, V_im_py, H_re_py, H_im_py);
    }

    // Single-frequency path (no frequency dimension)
    const auto &ant = ant_vec[0];
    const arma::uword n_out = (element_ind.n_elem == 0) ? ant.n_elements() : element_ind.n_elem;

    arma::mat V_re, V_im, H_re, H_im, dist_proj, azimuth_loc, elevation_loc, gamma;
    py::array_t<double> V_re_py, V_im_py, H_re_py, H_im_py, dist_proj_py, azimuth_loc_py, elevation_loc_py, gamma_py;

    if (!complex) // when complex, interpolate() resizes the outputs itself
    {
        V_re_py = qd_python_init_output(n_out, n_ang, &V_re);
        V_im_py = qd_python_init_output(n_out, n_ang, &V_im);
        H_re_py = qd_python_init_output(n_out, n_ang, &H_re);
        H_im_py = qd_python_init_output(n_out, n_ang, &H_im);
    }
    if (dist)
        dist_proj_py = qd_python_init_output(n_out, n_ang, &dist_proj);
    if (local_angles)
    {
        azimuth_loc_py = qd_python_init_output(n_out, n_ang, &azimuth_loc);
        elevation_loc_py = qd_python_init_output(n_out, n_ang, &elevation_loc);
        gamma_py = qd_python_init_output(n_out, n_ang, &gamma);
    }

    if (dist && local_angles)
        ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos, &dist_proj, &azimuth_loc, &elevation_loc, &gamma);
    else if (dist)
        ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos, &dist_proj);
    else
        ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos);

    // Assemble the output tuple in the documented order
    py::list out;
    if (complex)
    {
        out.append(qd_python_copy2numpy(V_re, V_im));
        out.append(qd_python_copy2numpy(H_re, H_im));
    }
    else
    {
        out.append(V_re_py);
        out.append(V_im_py);
        out.append(H_re_py);
        out.append(H_im_py);
    }
    if (dist)
        out.append(dist_proj_py);
    if (local_angles)
    {
        out.append(azimuth_loc_py);
        out.append(elevation_loc_py);
        out.append(gamma_py);
    }
    return py::tuple(out);
}

// pybind11 declaration (register under the `arrayant` submodule in python_main.cpp):
// m.def("interpolate", &arrayant_interpolate,
//       py::arg("arrayant"),
//       py::arg("azimuth"),
//       py::arg("elevation"),
//       py::arg("element") = py::none(),
//       py::arg("orientation") = py::none(),
//       py::arg("element_pos") = py::none(),
//       py::arg("frequency") = py::none(),
//       py::arg("complex") = false,
//       py::arg("dist") = false,
//       py::arg("local_angles") = false,
//       py::arg("fast_access") = false);