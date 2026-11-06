// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm.single_link_channel
Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model)

- Snapshot t: the paths of [[ssf_paths]] at the positions of t, at `rays["center_frequency"][i_freq]` with the link
  gain and K ratio of t, for the rays with power > 0 at `i_freq` plus LOS and GR
- `"phase"`: [[get_channels_planar]] with the arrays at the positions of `rays` and the orientations of t; delays,
  path lengths and angles stay those of the ray-data positions. `"drift"`: [[get_channels_spherical]] with the
  arrays at the positions and orientations of t
- `"taps"`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays
  (fixed per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap
  per element pair is the mean of the rays' delays weighted by `rays["power"]` at `i_freq` (equal weights where they sum
  to 0), `(n_rx, n_tx, n_tap)`; reduce it to `(1, 1, n_tap)` where needed. TR 38.901 LOS gives two taps at delay 0
- `"taps"` is exact at the phase centre for the TR methods (the rays of a tap share one delay); for `"quadriga"` only
  at the carrier: in UMi NLOS a tap varies over the band by 0.18, 0.52 and 0.79 (in-band selectivity) for 20, 100 and
  400 MHz. `"rays"`: one path per kept ray in ray order, exact at any bandwidth; the path data are the outputs of
  [[ssf_paths]] at t (in `"phase"` with the displacement phase in `M`, for the arrays at the ray-data positions)
- `n_snap` is the largest row count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  rows, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries; one row or entry applies to all snapshots. One position
  is a `(1, 3)` row; a `(3,)` array is rejected. A static link with a gain per snapshot repeats its position
- `use_absolute_delays` in `"phase"` adds the LOS delay of the ray-data positions
- No power correction and no initial position; validity limits and their meaning: see [[ssf_paths]]
- Errors: an invalid array (message after `Transmit antenna: ` or `Receive antenna: `), the errors of [[ssf_paths]]
  in `"drift"`, a row or entry count other than 1 or `n_snap`
- `chan` and `par` are copies of the library result; `chan` is one dict; the flags take a bool or `numpy.bool_`,
  `None` selects the default, any other type raises `TypeError`

## Keys of par:

| Name | Size | Content |
| --- | --- | --- |
| `tap` | `(n_tap,)` or `(n_path,)` | Tap index per output tap or per path; uint64 |
| `n_ray_tap` | `(n_tap,)` | Number of kept rays per tap; uint64; `"taps"` only |
| `fbs_pos`, `lbs_pos` | `(3, n_kept)` | Scatterers of the kept rays at the ray-data positions; `"drift"` with `"taps"` only; float32, the precision of path data in HDF5 files |
| `gr_pos` | `(3, n_snap)` | GR point per snapshot; `"drift"` with `"taps"` and a GR ray only; float32 |
| `link_gain_dB` | `(n_snap,)` | Link gain per snapshot |
| `b_med`, `b_eff` | `(2,)` | Validity scales of [[ssf_scatterers]] at `i_freq`, Tx and Rx; without scatterers from a solved copy (`d_min` = 2 m) |
| `limit` | `(2,)` | Validity limit per end, $\sqrt{\lambda b_{\mathrm{eff}} / 8}$ (`"phase"`) or $0.2 b_{\mathrm{med}}$ (`"drift"`); Inf where b is NaN |
| `ratio` | `(2,)` | Largest distance of the end from its ray-data position over the snapshots, divided by `limit` |
| `ratio_total` | scalar | $\sqrt{r_{\mathrm{Tx}}^2 + r_{\mathrm{Rx}}^2}$ of the two ratios; valid up to 1; `"drift"`: new segment from 2.5 |

## Usage:
```
chan, par = quadriga_lib.gbsm.single_link_channel( tx_array, rx_array, rays, link_gain_dB, mobility, output, \
    i_freq, tx_pos, rx_pos, tx_orientation, rx_orientation, kf_ratio, oxygen, use_absolute_delays, stack )
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Dicts containing the arrayant data with `n_tx` and `n_rx` ports; layout as
  documented in [[generate]]
- **`rays`** — Ray dict of the link, see [[ssf_rays]]
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at frequency `i_freq` of `rays` in dB; `(n_snap,)` or `(1,)`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"` (phase for the TR methods, drift for `"quadriga"`); default:
  `"auto"`
- **`output`** — `"taps"` or `"rays"`; default: `"taps"`
- **`i_freq`** — Frequency index in `rays["center_frequency"]`, 0-based; default: 0
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `None`: the positions of `rays`; `(n_snap, 3)` or `(1, 3)`;
  default: None
- **`tx_orientation`**, **`rx_orientation`** — Euler angles (bank, tilt, heading) in rad; `None`: 0; `(n_snap, 3)`
  or `(1, 3)`; default: None
- **`kf_ratio`** — K(snapshot) / K(drop), >= 0; `None`: 1; `(n_snap,)` or `(1,)`; default: None
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1, see [[ssf_paths]]; default: `False`
- **`use_absolute_delays`** — Include the LOS delay in all delays; default: `False`
- **`stack`** — Snapshot layout of `chan` as in [[hdf5_read_channel]]; default: `False`

## Outputs:
- **`chan`** — Channel dict, layout as documented in [[hdf5_read_channel]]: `center_frequency` scalar;
  `tx_position`, `rx_position` (the snapshot positions), `tx_orientation`, `rx_orientation` `(3, n_snap)`; per
  snapshot `coeff`, `delay` `(n_rx, n_tx, n_tap)` (`"taps"`) or `(n_rx, n_tx, n_path)` (`"rays"`); `"rays"` also
  `path_gain`, `path_length`, `path_polarization`, `path_angles` and in `"drift"` `fbs_pos`, `lbs_pos`
- **`par`** — Dict with the keys above

## See also:
- [[ssf_paths]] (paths of one link at one position, validity limits)
- [[ssf_scatterers]] (scatterers, `b_med` and `b_eff`)
- [[gbsm.rays_from_paths]] (ray data from path data, e.g. of a QRT file; taps 0, 3, 6, ... without `tap`)
MD!*/

py::tuple gbsm_single_link_channel(const py::dict &tx_array,
                                   const py::dict &rx_array,
                                   const py::dict &rays,
                                   const py::array_t<double> &link_gain_dB,
                                   const std::string &mobility,
                                   const std::string &output,
                                   arma::uword i_freq,
                                   py::handle tx_pos,
                                   py::handle rx_pos,
                                   py::handle tx_orientation,
                                   py::handle rx_orientation,
                                   py::handle kf_ratio,
                                   py::handle oxygen,
                                   py::handle use_absolute_delays,
                                   py::handle stack)
{
    // Read inputs; the library validates the arrays
    const auto tx_array_a = qd_python_dict2arrayant(tx_array, true);
    const auto rx_array_a = qd_python_dict2arrayant(rx_array, true);
    const auto rays_a = qd_python_dict2gbsm_rays(rays, "rays");
    const auto link_gain_dB_a = qd_python_numpy2arma_Col<double>(link_gain_dB, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto tx_orientation_a = qd_python_numpy2arma_Mat<double>(tx_orientation, true);
    const auto rx_orientation_a = qd_python_numpy2arma_Mat<double>(rx_orientation, true);
    const auto kf_ratio_a = qd_python_numpy2arma_Col<double>(kf_ratio, true);
    const bool oxygen_b = qd_python_flag(oxygen, false, "oxygen");
    const bool use_absolute_delays_b = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool stack_b = qd_python_flag(stack, false, "stack");

    // The result is returned by value and copied
    const auto chan = quadriga_lib::gbsm_single_link_channel(
        tx_array_a, rx_array_a, rays_a, link_gain_dB_a, mobility, output, i_freq,
        tx_pos.is_none() ? nullptr : &tx_pos_a, rx_pos.is_none() ? nullptr : &rx_pos_a,
        tx_orientation.is_none() ? nullptr : &tx_orientation_a, rx_orientation.is_none() ? nullptr : &rx_orientation_a,
        kf_ratio.is_none() ? nullptr : &kf_ratio_a, oxygen_b, use_absolute_delays_b);

    py::dict par;
    for (size_t k = 0; k < chan.par_names.size(); ++k)
        par[chan.par_names[k].c_str()] = qd_python_any2numpy(chan.par_data[k]);
    return py::make_tuple(qd_python_channel2dict(chan, arma::uvec(), false, stack_b), par);
}

// pybind11 declaration:
// m.def("single_link_channel", &gbsm_single_link_channel,
//       py::arg("tx_array"),
//       py::arg("rx_array"),
//       py::arg("rays"),
//       py::arg("link_gain_dB").none(false),
//       py::arg("mobility") = "auto",
//       py::arg("output") = "taps",
//       py::arg("i_freq") = 0,
//       py::arg("tx_pos") = py::none(),
//       py::arg("rx_pos") = py::none(),
//       py::arg("tx_orientation") = py::none(),
//       py::arg("rx_orientation") = py::none(),
//       py::arg("kf_ratio") = py::none(),
//       py::arg("oxygen") = false,
//       py::arg("use_absolute_delays") = false,
//       py::arg("stack") = false);
