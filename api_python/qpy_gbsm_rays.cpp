// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"



/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm.rays_from_paths
Ray data of one link from path data (positions, gains, lengths, polarization), e.g. read with qrt_file_read

- Returns the ray dict of [[ssf_rays]] (`fallback` empty) from path data in the names and order of [[qrt_file_read]],
  read with `normalize_polmat = 1` (`uplink = True` gives the reverse link; `normalize_polmat = 0` leaves only the
  free-space gain in `power`)
- `delay` = (path length - Tx-Rx distance) / c, `power` = path gain. Angles point from the Tx to the first of
  `fbs_pos`, `lbs_pos`, `rx_pos` that differs from it and from the Rx to the first of `lbs_pos`, `fbs_pos`, `tx_pos`;
  GR angles to the image of the other end (z mirrored); 0 if `tx_pos` = `rx_pos`
- QRT stores no path types or taps. Without `path_type`: LOS where FBS = LBS on the direct line (0.1 mm), else NLOS. A
  GR path needs `path_type`: for QRT files whose paths were built from GBSM scatterers pass the `n_interact` of
  [[qrt_file_read]] (interaction count = path type); for ray-traced files the user sets it (values > 2 raise
  `ValueError`)
- The GR point (`lbs_pos`) must be within 1 mm of z = 0

## Usage:
```
rays = quadriga_lib.gbsm.rays_from_paths( center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, \
    path_length, polmat, *, method, path_type, tap, gr_eps )
```

## Inputs:
- **`center_freq`** — Frequencies in Hz; `(n_freq,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m; `(3,)`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `(3, n_path)`
- **`path_gain`** — Path gains (linear); `(n_path, n_freq)`
- **`path_length`** — Path lengths in m; `(n_path,)`
- **`polmat`** — Polarization transfer matrices, real, rows as in [[channel.planar]]; `(8, n_path, n_freq)`
- **`method`** — Value of `rays["method"]`; sets the GR power law and the default mobility; default: `"quadriga"`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS, at most one LOS and one GR; `(n_path,)`; default: None
- **`tap`** — Tap index; `None`: LOS 0, GR 3, the others 6, 9, ... in path order (3 is kept for GR also without a GR
  path); `(n_path,)`; default: None
- **`gr_eps`** — Ground permittivity (complex), used as `real - j·imag`; needed only where a GR ray is recomputed, i.e.
  in the mobility `drift` at moved ends; `(n_freq,)`; default: None (also if empty)

## Output:
- **`rays`** — Ray dict with `n_ray = n_path`, see [[ssf_rays]]; use with `link_gain_dB = 0` and without `kf_ratio`

## See also:
- [[qrt_file_read]] (source of the path data)
- [[ssf_paths]] (paths at moved positions)
- [[gbsm.single_link_channel]] (channel of the link)
MD!*/

py::dict gbsm_rays_from_paths(const py::array_t<double> &center_freq,
                              const py::array_t<double> &tx_pos,
                              const py::array_t<double> &rx_pos,
                              const py::array_t<double> &fbs_pos,
                              const py::array_t<double> &lbs_pos,
                              const py::array_t<double> &path_gain,
                              const py::array_t<double> &path_length,
                              const py::array_t<double> &polmat,
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
    const auto polmat_a = qd_python_numpy2arma_Cube<double>(polmat, true);
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
                                       path_length_a, polmat_a, rays[0], method,
                                       path_type_a, tap_a,
                                       gr_eps_re, gr_eps_im);

    // The method string and the complex permittivity cannot be aliased
    py::dict rays_out = rays_py[0];
    rays_out["method"] = rays[0].method;
    if (has_gr)
        rays_out["gr_eps"] = qd_python_copy2numpy<double, std::complex<double>>(&rays[0].gr_eps_re, &rays[0].gr_eps_im);
    return rays_out;
}

/*!MD
# gbsm.single_link_channel
Channel coefficients of one link at one frequency over one segment

- Computes the MIMO coefficients of one link with its antennas over the `n_snap` snapshots of one segment at
  `rays["center_freq"][i_freq]`: per snapshot, the paths of [[ssf_paths]] at the snapshot positions with the link
  gain and K-factor ratio of the snapshot, for the kept rays (see Terms)
- `"phase"`: [[channel.planar]] with the arrays at the positions of `rays` and the orientations of the snapshot;
  delays, path lengths and angles stay those of the ray-data positions. `"drift"`: [[channel.spherical]] with the
  arrays at the positions and orientations of the snapshot
- `"taps"`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays (fixed
  per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap per
  element pair is the mean of the rays' delays weighted by `rays["power"]` at `i_freq` (equal weights where they sum
  to 0)
- `"taps"` is exact at the phase center for the TR methods (the rays of a tap share one delay); for `"quadriga"` only
  at the carrier: a tap sums rays of different delays and is frequency-selective within the band, so use `"rays"` for
  wide bands. `"rays"`: one path per kept ray in ray order, exact at any bandwidth; the path data are those of
  `ssf_paths` at the snapshot for these rays
- `n_snap` is the largest column count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  columns, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries (per snapshot at `i_freq`, while `ssf_paths` takes one
  per frequency); one column or entry applies to all snapshots. A static link with a gain per snapshot repeats its
  position
- Powers are not renormalized, and `initial_position` of `chan` stays 0. Validity limits: see `ssf_paths`
- Raises `ValueError` for an invalid array (message prefixed with `Transmit antenna: ` or `Receive antenna: `), invalid
  `rays`, `"drift"` without scatterers or with a GR ray at moved ends without permittivity or below z = 0, and a column
  or entry count other than 1 or `n_snap`

## Usage:
```
chan, par = quadriga_lib.gbsm.single_link_channel( tx_array, rx_array, rays, link_gain_dB, *, mobility, \
    output, i_freq, tx_pos, rx_pos, tx_orientation, rx_orientation, kf_ratio, oxygen, \
    use_absolute_delays, stack=False )
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Dicts containing the arrayant data with `n_tx` and `n_rx` ports; layout as
  documented in [[generate]]
- **`rays`** — Ray dict of the link, see [[ssf_rays]]
- **`link_gain_dB`** — Link gain (see Terms) at frequency `i_freq` of `rays` in dB; float, `(1,)` or `(n_snap,)`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"` (see Terms; `"auto"` is `"drift"` for `"quadriga"` rays, which then
  need [[ssf_scatterers]]); default: `"auto"`
- **`output`** — `"taps"` or `"rays"`; default: `"taps"`
- **`i_freq`** — Frequency index in `rays["center_freq"]`, 0-based; ray data of the TR methods hold one frequency:
  0; default: 0
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `None`: the positions of `rays`; `(3, n_snap)` or `(3,)`;
  default: None
- **`tx_orientation`**, **`rx_orientation`** — Euler angles (bank, tilt, heading) in rad; `None`: 0; `(3, n_snap)` or
  `(3,)`; default: None
- **`kf_ratio`** — KF(snapshot) / KF(drop) at `i_freq`, >= 0; `None`: 1; `(n_snap,)` or `(1,)`; default: None
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1, see `ssf_paths`; default: `False`
- **`use_absolute_delays`** — Include the LOS delay in all delays (`"phase"`: of the ray-data positions); default:
  `False`
- **`stack`** — Snapshot layout of `chan` as in [[hdf5_read_channel]]; default: `False`

## Outputs:
- **`chan`** — Channel dict, layout as documented in [[hdf5_read_channel]]: `center_freq` `(1,)`;
  `tx_pos`, `rx_pos` (the snapshot positions), `tx_orientation`, `rx_orientation` `(3, n_snap)`; per
  snapshot `coeff`, `delay` `(n_rx, n_tx, n_tap)` (`"taps"`) or `(n_rx, n_tx, n_path)` (`"rays"`); `"rays"` also
  `path_gain`, `path_length`, `polmat`, `path_angles` and in `"drift"` `fbs_pos`, `lbs_pos`
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
        tx_pos_a, rx_pos_a, tx_orientation_a, rx_orientation_a, // None or empty = omitted
        kf_ratio_a, oxygen_b, use_absolute_delays_b);

    py::dict par;
    for (size_t k = 0; k < chan.par_names.size(); ++k)
        par[chan.par_names[k].c_str()] = qd_python_any2numpy(chan.par_data[k]);
    return py::make_tuple(qd_python_channel2dict(chan, arma::uvec(), false, stack_b), par);
}
