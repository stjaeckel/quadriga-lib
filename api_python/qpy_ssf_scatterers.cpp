// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

- Places the scatterers that the mobility `drift` needs and returns the validity scales of [[ssf_paths]]
- NLOS rays with power > 0 at some frequency: FBS on the departure ray and LBS on the arrival ray at equal distances
  from their end, so that the polyline Tx, FBS, LBS, Rx has the length d3D + c·delay (d3D the Tx-Rx distance) and the
  angles are exact
- Where no such pair exists with both distances at least `d_min`, both scatterers are placed at `d_min` along the ray's
  own directions and `fallback` is 1: the polyline is longer than the ray, `delay` keeps its value. With `d_min` = 2 m
  this applies to up to about 23 % of the rays (most in the TR methods without absolute TOA) and lengthens them by up to
  25 ns (median below 5 ns), the DS of a link by up to 3 ns; `b_med` can then equal `d_min`
- LOS ray and NLOS rays with zero power at all frequencies: the midpoint of the direct line; GR ray: the specular point
  on z = 0, which requires both ends at z >= 0
- `rays_out` is a new dict with all keys; it shares the unchanged entries with `rays` as they are (no copy and no
  conversion: a list stays a list, and an in-place edit of `rays_out["delay"]` changes `rays["delay"]`) and has new
  arrays `fbs_pos`, `lbs_pos` and `fallback`; `rays` and its arrays are not changed, also when it already holds
  scatterers
- `b_med`, the scale of `drift`: the lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS
  rays with power > 0, fallback rays included; NaN without NLOS power. `b_eff`, the scale of `phase`:
  `sqrt(sum P / sum(P / b²))` over the rays with power `P > 0`, with `b` the end-to-scatterer distance (NLOS), d3D
  (LOS) or the GR path length (GR)
- One link per call: `rays[l], b_med, b_eff = quadriga_lib.gbsm.ssf_scatterers(rays[l])`

## Usage:
```
rays_out, b_med, b_eff = quadriga_lib.gbsm.ssf_scatterers( rays, d_min )
```

## Inputs:
- **`rays`** — Ray dict of one link, see [[ssf_rays]]
- **`d_min`** — Minimum distance of the scatterers to both ends in m, finite and > 0; default: 2.0

## Outputs:
- **`rays_out`** — Ray dict with `fbs_pos`, `lbs_pos` `(3, n_ray)` and `fallback` `(n_ray,)` (uint32)
- **`b_med`**, **`b_eff`** — Validity scales in m, rows Tx and Rx, column `k` for frequency `k`; `(2, n_freq)`

## See also:
- [[ssf_rays]] (ray data of a set of links)
- [[ssf_paths]] (mobility `drift` needs the scatterers)
- [[gbsm.single_link_channel]] (reports `b_med`, `b_eff` and the displacement ratios)
MD!*/

py::tuple ssf_scatterers(const py::dict &rays, double d_min)
{
    // The input is viewed (read-only); the solver writes into three new arrays
    auto rays_a = qd_python_dict2gbsm_rays(rays, "rays");
    const arma::uword n_ray = rays_a.n_ray(), n_freq = rays_a.n_freq(), n3 = 3, n2 = 2;
    auto fbs_pos_py = qd_python_init_output(n3, n_ray, &rays_a.fbs_pos);
    auto lbs_pos_py = qd_python_init_output(n3, n_ray, &rays_a.lbs_pos);
    auto fallback_py = qd_python_init_output(n_ray, &rays_a.fallback);
    rays_a.fbs_pos.zeros(), rays_a.lbs_pos.zeros(), rays_a.fallback.zeros(); // numpy memory is not initialised

    rays_a.solve_scatterers(d_min);

    // Validity scales per frequency
    arma::mat b_med, b_eff;
    auto b_med_py = qd_python_init_output(n2, n_freq, &b_med);
    auto b_eff_py = qd_python_init_output(n2, n_freq, &b_eff);
    for (arma::uword k = 0; k < n_freq; ++k)
        b_med.col(k) = rays_a.b_med(k), b_eff.col(k) = rays_a.b_eff(k);

    // New dict with all keys: the input's objects for the unchanged keys (empty where missing), the new solver arrays
    py::dict rays_out;
    for (const char *k : qd_python_gbsm_rays_keys)
    {
        const std::string key = k;
        if (key == "fbs_pos")
            rays_out[k] = fbs_pos_py;
        else if (key == "lbs_pos")
            rays_out[k] = lbs_pos_py;
        else if (key == "fallback")
            rays_out[k] = fallback_py;
        else if (rays.contains(k) && !py::object(rays[k]).is_none())
            rays_out[k] = rays[k];
        else if (key == "gr_eps") // the only other optional key
            rays_out[k] = py::array_t<std::complex<double>>(0);
        else // a key given as None, valid for n_ray = 0: empty with the dtype of the key
            rays_out[k] = (key == "path_type" || key == "tap") ? py::array(py::array_t<unsigned>(0))
                                                               : py::array(py::array_t<double>(0));
    }
    return py::make_tuple(rays_out, b_med_py, b_eff_py);
}

// pybind11 declaration:
// m.def("ssf_scatterers", &ssf_scatterers,
//       py::arg("rays"),
//       py::arg("d_min") = 2.0);
