// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm.rays_from_paths
Ray data of one link from spherical path data (e.g. read with qrt_file_read)

- Returns the ray dict of [[ssf_rays]] (`fallback` empty) from path data in the names and order of [[qrt_file_read]],
  read with `downlink = True` (an uplink read swaps the ends and conjugates `M`)
- `delay` = (path length - Tx-Rx distance) / c, `power` = path gain. Angles point from the Tx to the first of
  `fbs_pos`, `lbs_pos`, `rx_pos` that differs from it and from the Rx to the first of `lbs_pos`, `fbs_pos`, `tx_pos`;
  GR angles to the image of the other end (z mirrored); 0 if `tx_pos` = `rx_pos`
- QRT stores no path types or taps. Without `path_type`: LOS where FBS = LBS on the direct line (0.1 mm), else NLOS;
  a GR path needs `path_type`
- The GR point (`lbs_pos`) must be within 1 mm of z = 0; permittivity only for `drift` away from `tx_pos`, `rx_pos`
- Use in [[ssf_paths]] and [[gbsm.single_link_channel]] with `link_gain_dB = 0` and without `kf_ratio`
- Messages name the argument `center_freq` as `'center_frequency'` and the parts of `gr_eps` as `gr_eps_re`, `gr_eps_im`

## Usage:
```
rays = quadriga_lib.gbsm.rays_from_paths( center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, path_length, \
    M, method, path_type, tap, gr_eps )
```

## Inputs:
- **`center_freq`** — Frequencies in Hz; `(n_freq,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m; `(3,)`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `(3, n_path)`
- **`path_gain`** — Path gains (linear); `(n_path, n_freq)`
- **`path_length`** — Path lengths in m; `(n_path,)`
- **`M`** — Polarization transfer matrices, real; `(8, n_path, n_freq)`
- **`method`** — Value of `rays["method"]`; sets the GR power law and the default mobility; default: `"quadriga"`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS, at most one LOS and one GR; `(n_path,)`; default: None
- **`tap`** — Tap index; `None`: LOS 0, GR 3, the others 6, 9, ... in path order; `(n_path,)`; default: None
- **`gr_eps`** — Ground permittivity (complex), used as `real - j·imag`; `(n_freq,)`; default: None (also if empty)

## Output:
- **`rays`** — Ray dict with `n_ray = n_path`, see [[ssf_rays]]

## See also:
- [[qrt_file_read]] (source of the path data)
- [[ssf_paths]] (paths at other positions)
- [[gbsm.single_link_channel]] (channel coefficients of the link)
MD!*/

py::dict gbsm_rays_from_paths(const py::array_t<double> &center_freq,
                              const py::array_t<double> &tx_pos,
                              const py::array_t<double> &rx_pos,
                              const py::array_t<double> &fbs_pos,
                              const py::array_t<double> &lbs_pos,
                              const py::array_t<double> &path_gain,
                              const py::array_t<double> &path_length,
                              const py::array_t<double> &M,
                              const std::string &method,
                              py::handle path_type,
                              py::handle tap,
                              py::handle gr_eps)
{
    // Read inputs; codes 'path_type' and 'tap' as they are
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Col<double>(rx_pos, true);
    const auto fbs_pos_a = qd_python_numpy2arma_Mat<double>(fbs_pos, true);
    const auto lbs_pos_a = qd_python_numpy2arma_Mat<double>(lbs_pos, true);
    const auto path_gain_a = qd_python_numpy2arma_Mat<double>(path_gain, true);
    const auto path_length_a = qd_python_numpy2arma_Col<double>(path_length, true);
    const auto M_a = qd_python_numpy2arma_Cube<double>(M, true);
    const auto path_type_a = qd_python_u32_entries(path_type, "path_type");
    const auto tap_a = qd_python_u32_entries(tap, "tap");

    // Complex permittivity -> real and imaginary part (copies); empty as None (the 'gr_eps' of ssf_rays without GR)
    arma::vec gr_eps_re, gr_eps_im;
    if (!gr_eps.is_none())
    {
        arma::cube re, im;
        qd_python_copy2arma(py::cast<py::array_t<std::complex<double>>>(gr_eps), re, im);
        gr_eps_re = arma::vectorise(re), gr_eps_im = arma::vectorise(im);
    }
    const bool has_gr = !gr_eps_re.empty();

    // Output dict in numpy memory (n_path from 'fbs_pos'; wrong sizes throw in the library call before any write)
    std::vector<quadriga_lib::gbsm_rays> rays;
    py::list rays_py = qd_python_init_gbsm_rays(1, fbs_pos_a.n_cols, center_freq_a.n_elem, has_gr, true, rays);

    quadriga_lib::gbsm_rays_from_paths(center_freq_a, tx_pos_a, rx_pos_a, fbs_pos_a, lbs_pos_a, path_gain_a,
                                       path_length_a, M_a, rays[0], method,
                                       path_type.is_none() ? nullptr : &path_type_a, tap.is_none() ? nullptr : &tap_a,
                                       has_gr ? &gr_eps_re : nullptr, has_gr ? &gr_eps_im : nullptr);

    // The method string and the complex permittivity cannot be aliased
    py::dict rays_out = rays_py[0];
    rays_out["method"] = rays[0].method;
    if (has_gr)
        rays_out["gr_eps"] = qd_python_copy2numpy<double, std::complex<double>>(&rays[0].gr_eps_re, &rays[0].gr_eps_im);
    return rays_out;
}

// pybind11 declaration:
// m.def("rays_from_paths", &gbsm_rays_from_paths,
//       py::arg("center_freq").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("fbs_pos").none(false),
//       py::arg("lbs_pos").none(false),
//       py::arg("path_gain").none(false),
//       py::arg("path_length").none(false),
//       py::arg("M").none(false),
//       py::arg("method") = "quadriga",
//       py::arg("path_type") = py::none(),
//       py::arg("tap") = py::none(),
//       py::arg("gr_eps") = py::none());
