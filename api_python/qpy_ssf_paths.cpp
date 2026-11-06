// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_paths
Antenna-free paths of one link at one position

- `"phase"`: use the outputs with [[get_channels_planar]] and the arrays at the positions of `rays`; the
  displacement phase of TR 38.901 eq. (7.5-22) is in `M`. With the arrays at the new positions the LOS path is no
  longer identified and the displacement phase counts twice, without an error
- `"drift"`: use the outputs at the new positions with [[get_channels_spherical]] or [[get_channels_multifreq]];
  needs the scatterers of [[ssf_scatterers]]
- Kept rays: LOS and GR rays always (also at zero power), NLOS rays with power > 0 at some frequency; paths in ray
  order, `ray_ind` maps them to the rays (e.g. `rays["tap"][ray_ind]`, `rays["fallback"][ray_ind]`)

| Quantity | `"phase"` | `"drift"` |
| --- | --- | --- |
| Path length | Tx-Rx distance of `rays` + c·delay | Polyline Tx, FBS, LBS, Rx; LOS direct line; GR via the specular point |
| Angles | Ray data | Geometry: towards FBS and LBS, LOS direct line, GR towards the image points |
| `M` | Ray `M` times the displacement phase | Ray `M`; GR recomputed when an end moved |
| `fbs_pos`, `lbs_pos` | `(3, 0)` | Scatterers of the rays; LOS midpoint of the direct line; GR specular point |

- `"drift"` with an end moved (any coordinate differs from `rays`): GR power by the power law of the method of `rays`
  (`(d3D / d_GR)²` for the TR methods, 1 for `"quadriga"`), GR `M` the scalar multiple of `diag(R_par, -R_perp)` (TR
  38.901 eqs. (7.6-38), (7.6-39)) that best fits the stored matrix, which replaces a matrix of another form (e.g. a
  conjugated one). Fallback rays have their longer polyline, also at the positions of `rays`
- Path gain per frequency: the coefficient power; with `kf_ratio` LOS and GR are rescaled against NLOS so that the
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times
  `10^(link_gain_dB / 10)`; with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each
  frequency) over each path's own length. Deviations: τ_Δ is not added (at 60 GHz: 0.017 dB median and up to 1.28 dB
  at p99 more loss on NLOS rays of the TR methods without absolute TOA; sub-cluster rays up to 0.13 dB), eq. (7.6-2)
  is not applied across the band
- Validity per moving end: `"phase"` up to about `sqrt(λ · b_eff / 8)` (`b_eff` of [[ssf_scatterers]]; p50 over the
  links for an Rx move: TR NLOS 0.27 to 0.36 m at 3.5 GHz and 0.12 to 0.13 m at 28 GHz, `"quadriga"` NLOS 0.24 to
  0.77 m and 0.24 to 0.31 m, LOS 0.91 to 1.33 m and 0.33 to 0.40 m); `"drift"` up to about `0.2 · b_med` (p50 1.5 to
  5.6 m LOS, 7.8 to 12.4 m NLOS, about 20 % less when both ends move; new segment from `0.5 · b_med`); in `"drift"`
  each path length changes by at most the displacement
- Errors in `"drift"`: no scatterers; a GR ray at moved ends without permittivity in `rays` or with an end below z = 0
- The flag takes a bool or `numpy.bool_`; `None` selects the default, any other type raises `TypeError`

## Usage:
```
path_gain, path_length, M, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind = \
    quadriga_lib.gbsm.ssf_paths( rays, link_gain_dB, mobility, tx_pos, rx_pos, kf_ratio, oxygen )
```

## Inputs:
- **`rays`** — Ray dict of the link, see [[ssf_rays]]
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at the position in dB; 0 for data from paths; `(n_freq,)`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"` (phase for the TR methods, drift for `"quadriga"`); default:
  `"auto"`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m; `None`: the positions of `rays`; `(3,)`; default: None
- **`kf_ratio`** — K(position) / K(drop), >= 0; `None`: 1; `(n_freq,)`; default: None
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `False`

## Outputs:
- **`path_gain`** — Path gains (linear); `(n_path, n_freq)`
- **`path_length`** — Path lengths in m; `(n_path,)`
- **`M`** — Polarization transfer matrices, real, rows as in [[get_channels_planar]]; `(8, n_path, n_freq)`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Departure and arrival angles in rad, GCS; `(n_path,)`
- **`fbs_pos`**, **`lbs_pos`** — Scatterer positions in m; `(3, n_path)`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS; int64; `(n_path,)`
- **`ray_ind`** — Index of the ray in `rays`, 0-based; int64; `(n_path,)`

## See also:
- [[ssf_scatterers]] (scatterers, validity scales `b_eff` and `b_med`)
- [[gbsm.single_link_channel]] (channel coefficients of one link over a segment)
- [[gbsm.rays_from_paths]] (ray data from path data, e.g. of a QRT file)
MD!*/

py::tuple ssf_paths(const py::dict &rays,
                    const py::array_t<double> &link_gain_dB,
                    const std::string &mobility,
                    py::handle tx_pos,
                    py::handle rx_pos,
                    py::handle kf_ratio,
                    py::handle oxygen)
{
    // Read inputs
    const auto rays_a = qd_python_dict2gbsm_rays(rays, "rays");
    const auto link_gain_dB_a = qd_python_numpy2arma_Col<double>(link_gain_dB, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Col<double>(rx_pos, true);
    const auto kf_ratio_a = qd_python_numpy2arma_Col<double>(kf_ratio, true);
    const bool oxygen_b = qd_python_flag(oxygen, false, "oxygen");

    // Outputs of unknown size (n_path), copied after the call
    arma::mat path_gain, fbs_pos, lbs_pos;
    arma::vec path_length, aod, eod, aoa, eoa;
    arma::cube M;
    arma::u32_vec path_type;
    arma::uvec ray_ind;
    quadriga_lib::ssf_paths(rays_a, link_gain_dB_a, path_gain, path_length, M, mobility,
                            tx_pos.is_none() ? nullptr : &tx_pos_a, rx_pos.is_none() ? nullptr : &rx_pos_a,
                            kf_ratio.is_none() ? nullptr : &kf_ratio_a, &aod, &eod, &aoa, &eoa, &fbs_pos, &lbs_pos,
                            &path_type, &ray_ind, oxygen_b);

    return py::make_tuple(qd_python_copy2numpy(&path_gain), qd_python_copy2numpy(&path_length),
                          qd_python_copy2numpy(&M), qd_python_copy2numpy(&aod), qd_python_copy2numpy(&eod),
                          qd_python_copy2numpy(&aoa), qd_python_copy2numpy(&eoa), qd_python_copy2numpy(&fbs_pos),
                          qd_python_copy2numpy(&lbs_pos), qd_python_copy2numpy<unsigned, py::ssize_t>(&path_type),
                          qd_python_copy2numpy<arma::uword, py::ssize_t>(&ray_ind));
}

// pybind11 declaration:
// m.def("ssf_paths", &ssf_paths,
//       py::arg("rays"),
//       py::arg("link_gain_dB").none(false),
//       py::arg("mobility") = "auto",
//       py::arg("tx_pos") = py::none(),
//       py::arg("rx_pos") = py::none(),
//       py::arg("kf_ratio") = py::none(),
//       py::arg("oxygen") = false);
