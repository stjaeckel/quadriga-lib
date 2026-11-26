// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# crossfade_segments
Join the channels of consecutive segments of one link by a cross-fade

- Joins the "rays" channels of the segments of one link, e.g. of a track whose scenario or indoor state changes, into
  one channel; consecutive segments share the snapshots of a ramp, over which the clusters of the old segment are
  exchanged for those of the new one
- [[get_channels_from_rays]] joins the segments of tracks with this function, the ramp centred on the row of the change

## Usage:
```
channel = quadriga_lib.channel.crossfade_segments( segments, *, n_overlap, output, par, stack )
```

## Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients and path gains (`"rays"`), the same
  number of paths in every snapshot, equal `n_rx`, `n_tx` and `center_freq` (at most one value); list of channel
  dicts, keys in [[hdf5_read_channel]]
- **`n_overlap`** — Number of snapshots at the start of each segment that are also the last snapshots of the preceding
  segment (computed at the same positions); they form the ramp. The first entry is 0, at most the snapshots of the
  preceding segment after its own shared ones; `None`: the snapshots whose Tx and Rx positions match (within 1 µm), an
  error where the positions repeat (e.g. ends that do not move); `(n_segment,)`; default: None
- **`output`** — `"taps"` (the paths summed per tap index) or `"rays"`; default: `"taps"`
- **`par`** — Unstructured data of the segments: list of dicts, one per segment, with the keys `tap`, `path_type` and
  `cluster` (uint32, one value per path, see Notes); a list that is given must have one of these keys (other keys are
  ignored); `None` = all paths NLOS, each its own cluster; default: None
- **`stack`** — Snapshot layout of the channel as in [[hdf5_read_channel]]; default: `False`

## Outputs:
- **`channel`** — The joined channel, dict, keys in [[hdf5_read_channel]]

## Notes:
- **Snapshots**: segment $k$ starts with the `n_overlap[k]` snapshots of the ramp, which are also the last ones of
  segment $k - 1$; the output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots. Outside the
  ramps the snapshots are those of one segment
- **Paths**: entries of `par`: `path_type` gives the LOS and GR paths (without it, tap index 0 of `tap` and a tap index
  3 held by one path, without index 4 or 5, as [[ssf_rays]] numbers them), `cluster` the clusters, whose paths fade
  together (without it: the taps, without `tap`: each path)
- **Level**: at ramp snapshot $i$ of $n_r$, with $P_a$, $P_b$ the summed path gains of the old and new segment,
  $$G_i = P_{a,i}^{1 - w_i} P_{b,i}^{w_i}, \quad w_i = \frac{1}{2} \left( 1 + \sin\left( \left( \frac{i}{n_r + 1} - \frac{1}{2} \right) \pi \right) \right)$$
- **Exchange**: both cluster sets are scaled to $G$; the clusters of the old segment fade out and those of the new one
  in, one after another, each over a quarter of the exchange, in an order that keeps the delay and angular spreads close
  to those of the superposition of both sets (a segment with more than 32 clusters keeps its cluster order). The
  exchange takes the part of the ramp in which $G$ is within 10 dB of the weaker segment, so that no cluster exceeds its
  own power by more than 10 dB
- **LOS and GR**: one path each, with the data of the new segment (of the old one where the new one has no power),
  scaled to $G_i \left( (1 - q_i) g_a / P_a + q_i g_b / P_b \right)$, $g$ the path gain of the path and $q_i$ the
  exchanged share
- **A segment without power** at a ramp snapshot: all paths of the transition fade with $1 - w$ (old) and $w$ (new)
  in power
- **Taps**: the weighted paths summed per tap index, one set of tap slots for all snapshots (LOS, GR, NLOS); a new tap
  takes the slot of an old tap that has faded out before it enters. Slots without power hold zero coefficients and the
  smallest delay of the snapshot. The delay of a tap is the mean of its paths' delays weighted by their path gains
- **Other data**: delays as given (one reference for all segments); positions and orientations of the old segment;
  `name` and `center_freq` of segment 0; `"rays"` keep `path_length`, `polmat`, `path_angles`, `fbs_pos` and
  `lbs_pos` where all segments have them; `par`, `n_interact` and `interact_coord` are not carried
- **`par`**: the C++ channels carry these entries in their `par_data`; in Python they are the input `par`, as in
  [[hdf5_write_channel]], with the entries of each segment as [[gbsm.single_link_channel]] returns them

## See also:
- [[get_channels_from_rays]] (joins the segments of tracks with this function)
- [[gbsm.single_link_channel]] (channels of one segment)
MD!*/
py::dict crossfade_segments(const py::handle &segments,
                            const py::handle &n_overlap,
                            const std::string &output,
                            py::handle par,
                            py::handle stack)
{
    auto segments_v = qd_python_list2channel(segments, false); // The library validates the channels
    const auto n_overlap_a = qd_python_numpy2arma_Col<arma::uword>(n_overlap, true, false, "n_overlap");
    const bool stack_b = qd_python_flag(stack, false, "stack");

    // Tap indices, path types and clusters of each segment (the only entries of par_data that the function reads)
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
            if (!par_v[k].contains("tap") && !par_v[k].contains("path_type") && !par_v[k].contains("cluster"))
                throw std::invalid_argument("Input 'par' has no key 'tap', 'path_type' or 'cluster'.");
            for (const char *name : {"tap", "path_type", "cluster"})
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

    const auto channel = quadriga_lib::crossfade_segments(segments_v, n_overlap_a, output);
    return qd_python_channel2dict(channel, arma::uvec(), false, stack_b);
}
