// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# crossfade_segments
Join the channels of consecutive segments of one link by a power cross-fade

- Joins the channels of the segments of one link, e.g. of a track whose scenario changes, into one channel; consecutive
  segments share snapshots, over which the paths of the old segment fade out and those of the new one fade in
- [[get_channels_from_rays]] joins the segments of tracks with this function; it is a port of the channel merger of
  QuaDRiGa

## Usage:
```
channel = quadriga_lib.channel.crossfade_segments( segments, n_overlap, *, overlap, par, stack )
```

## Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients, equal `n_rx`, `n_tx`, form and
  `center_freq` (at most one value); list of channel dicts, keys in [[hdf5_read_channel]]
- **`n_overlap`** — Snapshots shared with the preceding segment, the first entry is 0, at most the snapshots of the
  preceding segment after its own shared ones; `(n_segment,)`
- **`overlap`** — Part of the shared snapshots used for the ramp, in [0, 1]; default: 0.5
- **`par`** — Unstructured data of the segments: list of dicts, one per segment, with the keys `tap` (tap index per path
  or tap, uint32), `path_type` (0 = LOS, 1 = GR, 2 = NLOS per path or tap, uint32, optional) and `n_ray_tap` (rays per
  tap, `"taps"` only); a list that is given must have the key `tap` or `path_type` (other keys are ignored); `None` = no
  tap indices, all paths are NLOS; default: None
- **`stack`** — Snapshot layout of the channel as in [[hdf5_read_channel]]; default: `False`

## Outputs:
- **`channel`** — The joined channel, dict, keys in [[hdf5_read_channel]]

## Notes:
- **Snapshots**: segment $k$ starts with `n_overlap[k]` snapshots that are also the last ones of segment $k - 1$; the
  output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots
- **Ramp**: in the last $n_r = \mathrm{round}(\mathrm{overlap} \cdot n_{\mathrm{overlap},k})$ shared snapshots, the old
  paths fade out with $1 - w_i$ and the new ones fade in with $w_i$, in power, with

  $$w_i = \frac{1}{2} \left( 1 + \sin\left( \left( \frac{i}{n_r + 1} - \frac{1}{2} \right) \pi \right) \right), \quad i = 1, \dots, n_r$$

  Before the ramp, the snapshots are those of the old segment
- **LOS and GR** (types 0 and 1 of the entry `path_type` of `par`; without it, tap index 0 of the entry `tap` and tap
  index 3 held by one ray, without index 4 or 5, as [[ssf_rays]] numbers them): one path with the data of the new
  segment (of the old one where the new one has no power), scaled to
  $\lvert h \rvert^2 = (1 - w) \lvert h_\mathrm{old} \rvert^2 + w \lvert h_\mathrm{new} \rvert^2$ by the path gain
  (rays) or the tap power (taps)
- **Rays** (channels with `path_gain`): the NLOS paths of both segments side by side, with the amplitudes $\sqrt{1 - w}$
  and $\sqrt{w}$; `path_length`, `path_angles`, `fbs_pos` and `lbs_pos` are kept where all segments have them
- **Taps** (without `path_gain`, one tap count per segment): one set of tap slots for all snapshots (LOS, GR, NLOS); the
  NLOS taps of the old segment hold $1 - w$ and those of the new segment $w$ of their mean power over the ramp; slots
  without power hold zero coefficients and the smallest delay of the snapshot
- **Other data**: delays as given (one reference for all segments); positions interpolated in the ramp, orientations
  along the shorter way; `name` and `center_freq` of segment 0; `par`, `n_interact` and `interact_coord` are not carried
- **`par`**: the C++ channels carry the tap indices in their `par_data`; in Python they are the input `par`, as in
  [[hdf5_write_channel]], with the entries of each segment as [[gbsm.single_link_channel]] returns them

## See also:
- [[get_channels_from_rays]] (joins the segments of tracks with this function)
- [[gbsm.single_link_channel]] (channels of one segment)
MD!*/
py::dict crossfade_segments(const py::handle &segments,
                            const py::handle &n_overlap,
                            double overlap,
                            py::handle par,
                            py::handle stack)
{
    auto segments_v = qd_python_list2channel(segments, false); // The library validates the channels
    const auto n_overlap_a = qd_python_numpy2arma_Col<arma::uword>(n_overlap, true, false, "n_overlap");
    const bool stack_b = qd_python_flag(stack, false, "stack");

    // Tap indices and path types of each segment (the only entries of par_data that the function reads)
    if (!par.is_none() && !(py::isinstance<py::list>(par) && py::len(par) == 0))
    {
        std::vector<py::dict> par_v;
        if (py::isinstance<py::dict>(par))
            par_v.push_back(py::reinterpret_borrow<py::dict>(par));
        else if (py::isinstance<py::list>(par))
            for (py::handle item : py::reinterpret_borrow<py::list>(par))
            {
                if (!py::isinstance<py::dict>(item))
                    throw std::invalid_argument("Input 'par' must be a dict, a list of dicts, or None.");
                par_v.push_back(py::reinterpret_borrow<py::dict>(item));
            }
        else
            throw std::invalid_argument("Input 'par' must be a dict, a list of dicts, or None.");
        if (par_v.size() != segments_v.size())
            throw std::invalid_argument("Inputs 'par' and 'segments' must have the same number of entries.");
        for (size_t k = 0; k < segments_v.size(); ++k)
        {
            if (!par_v[k].contains("tap") && !par_v[k].contains("path_type"))
                throw std::invalid_argument("Input 'par' has no key 'tap' or 'path_type'.");
            for (const char *name : {"tap", "path_type", "n_ray_tap"})
            {
                if (!par_v[k].contains(name) || par_v[k][name].is_none())
                    continue;
                const auto data = qd_python_numpy2arma_Col<unsigned>(par_v[k][name], false, false, std::string("par.") + name);
                if (data.is_empty())
                    continue;
                segments_v[k].par_names.push_back(name);
                segments_v[k].par_data.push_back(std::any(data));
            }
        }
    }

    const auto channel = quadriga_lib::crossfade_segments(segments_v, n_overlap_a, overlap);
    return qd_python_channel2dict(channel, arma::uvec(), false, stack_b);
}
