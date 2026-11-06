// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm.single_link_channel
Channel coefficients of one link at one frequency over one segment

- Computes the MIMO coefficients of one link with its antennas over the `n_snap` snapshots of one segment at
  `rays["center_frequency"][i_freq]`: per snapshot, the paths of [[ssf_paths]] at the snapshot positions with the link
  gain and K-factor ratio of the snapshot, for the kept rays (see Terms)
- `"phase"`: [[get_channels_planar]] with the arrays at the positions of `rays` and the orientations of the snapshot;
  delays, path lengths and angles stay those of the ray-data positions. `"drift"`: [[get_channels_spherical]] with the
  arrays at the positions and orientations of the snapshot
- `"taps"`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays (fixed
  per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap per
  element pair is the mean of the rays' delays weighted by `rays["power"]` at `i_freq` (equal weights where they sum
  to 0)
- `"taps"` is exact at the phase center for the TR methods (the rays of a tap share one delay); for `"quadriga"` only
  at the carrier: a tap sums rays of different delays and is frequency-selective within the band, so use `"rays"` for
  wide bands. `"rays"`: one path per kept ray in ray order, exact at any bandwidth; the path data are those of
  `ssf_paths` at the snapshot for these rays
- `n_snap` is the largest row count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  rows, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries (per snapshot at `i_freq`, while `ssf_paths` takes one
  per frequency); one row or entry applies to all snapshots. A static link with a gain per snapshot repeats its
  position
- Powers are not renormalized, and `initial_position` of `chan` stays 0. Validity limits: see `ssf_paths`
- Raises `ValueError` for an invalid array (message prefixed with `Transmit antenna: ` or `Receive antenna: `), invalid
  `rays`, `"drift"` without scatterers or with a GR ray at moved ends without permittivity or below z = 0, and a row or
  entry count other than 1 or `n_snap`

## Usage:
```
chan, par = quadriga_lib.gbsm.single_link_channel( tx_array, rx_array, rays, link_gain_dB, mobility, output, \
    i_freq, tx_pos, rx_pos, tx_orientation, rx_orientation, kf_ratio, oxygen, use_absolute_delays, stack )
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Dicts containing the arrayant data with `n_tx` and `n_rx` ports; layout as
  documented in [[generate]]
- **`rays`** — Ray dict of the link, see [[ssf_rays]]
- **`link_gain_dB`** — Link gain (see Terms) at frequency `i_freq` of `rays` in dB; float, `(1,)` or `(n_snap,)`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"` (see Terms; `"auto"` is `"drift"` for `"quadriga"` rays, which then
  need [[ssf_scatterers]]); default: `"auto"`
- **`output`** — `"taps"` or `"rays"`; default: `"taps"`
- **`i_freq`** — Frequency index in `rays["center_frequency"]`, 0-based; ray data of the TR methods hold one frequency:
  0; default: 0
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `None`: the positions of `rays`; `(n_snap, 3)` or `(1, 3)`
  (a `(3,)` array is rejected); default: None
- **`tx_orientation`**, **`rx_orientation`** — Euler angles (bank, tilt, heading) in rad; `None`: 0; `(n_snap, 3)`
  or `(1, 3)`; default: None
- **`kf_ratio`** — KF(snapshot) / KF(drop) at `i_freq`, >= 0; `None`: 1; `(n_snap,)` or `(1,)`; default: None
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1, see `ssf_paths`; default: `False`
- **`use_absolute_delays`** — Include the LOS delay in all delays (`"phase"`: of the ray-data positions); default:
  `False`
- **`stack`** — Snapshot layout of `chan` as in [[hdf5_read_channel]]; default: `False`

## Outputs:
- **`chan`** — Channel dict, layout as documented in [[hdf5_read_channel]]: `center_frequency` `(1,)`;
  `tx_position`, `rx_position` (the snapshot positions), `tx_orientation`, `rx_orientation` `(3, n_snap)`; per
  snapshot `coeff`, `delay` `(n_rx, n_tx, n_tap)` (`"taps"`) or `(n_rx, n_tx, n_path)` (`"rays"`); `"rays"` also
  `path_gain`, `path_length`, `path_polarization`, `path_angles` and in `"drift"` `fbs_pos`, `lbs_pos`
- **`par`** — Dict with the keys below

## Keys of par:

| Name | Size | Content |
| --- | --- | --- |
| `tap` | `(n_tap,)` or `(n_path,)` | Tap index per output tap or per path; uint32 |
| `n_ray_tap` | `(n_tap,)` | Number of kept rays per tap; uint32; `"taps"` only |
| `fbs_pos`, `lbs_pos` | `(3, n_kept)` | Scatterers of the kept rays at the ray-data positions; `"drift"` with `"taps"` only; float32, the precision of path data in HDF5 files |
| `gr_pos` | `(3, n_snap)` | GR point per snapshot; `"drift"` with `"taps"` and a GR ray only; float32 |
| `link_gain_dB` | `(n_snap,)` | Link gain per snapshot |
| `b_med`, `b_eff` | `(2,)` | Of [[ssf_scatterers]] at `i_freq`, Tx and Rx; without scatterers from a solved copy (`d_min` = 2 m) |
| `limit` | `(2,)` | Validity limit per end in m (see [[ssf_paths]]): `sqrt(λ·b_eff/8)` in `"phase"`, `0.2·b_med` in `"drift"`; Inf where `b_eff` or `b_med` is NaN |
| `ratio` | `(2,)` | Largest distance of the end from its ray-data position over the snapshots, divided by `limit` |
| `ratio_total` | scalar | `sqrt(ratio[0]² + ratio[1]²)`; valid up to 1 |

## See also:
- [[ssf_rays]] (the ray dict)
- [[ssf_paths]] (paths of one link at one position, validity limits)
- [[ssf_scatterers]] (scatterers for `"drift"`, `b_med` and `b_eff`)
- [[gbsm.rays_from_paths]] (ray data from path data)
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
