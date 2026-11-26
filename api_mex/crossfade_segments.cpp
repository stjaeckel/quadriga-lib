// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# CROSSFADE_SEGMENTS
Join the channels of consecutive segments of one link by a cross-fade

- Joins the "rays" channels of the segments of one link, e.g. of a track whose scenario or indoor state changes, into
  one channel; consecutive segments share the snapshots of a ramp, over which the clusters of the old segment are
  exchanged for those of the new one
- [[get_channels_from_rays]] joins the segments of tracks with this function, the ramp centred on the row of the change

## Usage:
```
channel = quadriga_lib.crossfade_segments( segments, n_overlap, output, par );
```

## Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients and path gains (`'rays'`), the same
  number of paths in every snapshot, equal `n_rx`, `n_tx` and `center_freq` (at most one value); struct array, fields
  in [[hdf5_read_channel]]; `[1, n_segment]`
- **`n_overlap`** — Number of snapshots at the start of each segment that are also the last snapshots of the preceding
  segment (computed at the same positions); they form the ramp. Entry 1 is 0, at most the snapshots of the preceding
  segment after its own shared ones; `[]`: the snapshots whose Tx and Rx positions match (within 1 µm), an error where
  the positions repeat (e.g. ends that do not move); `[n_segment]`; default: `[]`
- **`output`** — `'taps'` (the paths summed per tap index) or `'rays'`; default: `'taps'`
- **`par`** — Unstructured data of the segments with the fields `tap`, `path_type` and `cluster` (uint32, one value per
  path, see Notes); struct array of the size of `segments` that has one of these fields (other fields are ignored);
  `[]` = all paths NLOS, each its own cluster; default: `[]`

## Outputs:
- **`channel`** — The joined channel, 1x1 struct, fields in [[hdf5_read_channel]]; `par` is not carried

## Notes:
- **Snapshots**: segment $k$ starts with the `n_overlap(k)` snapshots of the ramp, which are also the last ones of
  segment $k - 1$; the output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots. Outside the
  ramps the snapshots are those of one segment
- **Paths**: fields of `par`: `path_type` gives the LOS and GR paths (without it, tap index 0 of `tap` and a tap index 3
  held by one path, without index 4 or 5, as [[ssf_rays]] numbers them), `cluster` the clusters, whose paths fade
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
  `name` and `center_freq` of segment 1; `'rays'` keep `path_length`, `polmat`, `path_angles`, `fbs_pos` and
  `lbs_pos` where all segments have them; `par`, `n_interact` and `interact_coord` are not carried
- **`par`**: the C++ channels carry these entries in their `par_data`; in MATLAB they are the input `par`, as in
  [[hdf5_write_channel]], with the entries of each segment as [[gbsm_single_link_channel]] returns them

## See also:
- [[get_channels_from_rays]] (joins the segments of tracks with this function)
- [[gbsm_single_link_channel]] (channels of one segment)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::vector<quadriga_lib::channel<double>> segments = qd_mex_struct2channel(prhs[0], false, false, "segments"); // C++ validates
    const arma::uvec n_overlap = (nrhs < 2) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[1], false, "n_overlap");
    const std::string output = (nrhs < 3) ? std::string("taps") : qd_mex_get_string(prhs[2], "taps", "output");

    // Tap indices, path types and clusters of each segment (the only entries of par_data that the function reads)
    if (nrhs > 3 && !mxIsEmpty(prhs[3]))
    {
        qd_mex_check_struct(prhs[3], "par");
        if (mxGetFieldNumber(prhs[3], "tap") < 0 && mxGetFieldNumber(prhs[3], "path_type") < 0 && mxGetFieldNumber(prhs[3], "cluster") < 0)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'par' has no field 'tap', 'path_type' or 'cluster'.");
        if (mxGetNumberOfElements(prhs[3]) != segments.size())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'par' and 'segments' must have the same number of entries.");
        for (size_t k = 0; k < segments.size(); ++k)
            for (const char *name : {"tap", "path_type", "cluster"})
            {
                const mxArray *data = mxGetField(prhs[3], (mwIndex)k, name);
                if (data == nullptr || mxIsEmpty(data))
                    continue;
                segments[k].par_names.push_back(name);
                segments[k].par_data.push_back(std::any(qd_mex_get_Col<unsigned>(data, true, std::string("par.") + name)));
            }
    }

    quadriga_lib::channel<double> channel;
    CALL_QD(channel = quadriga_lib::crossfade_segments(segments, n_overlap, output));

    plhs[0] = qd_mex_channel2struct({channel});
}
