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
Join the channels of consecutive segments of one link by a power cross-fade

- Joins the channels of the segments of one link, e.g. of a track whose scenario changes, into one channel; consecutive
  segments share snapshots, over which the paths of the old segment fade out and those of the new one fade in
- [[get_channels_from_rays]] joins the segments of tracks with this function; it is a port of the channel merger of
  QuaDRiGa

## Usage:
```
channel = quadriga_lib.crossfade_segments( segments, n_overlap, overlap, par );
```

## Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients, equal `n_rx`, `n_tx`, form and
  `center_freq` (at most one value); struct array, fields in [[hdf5_read_channel]]; `[1, n_segment]`
- **`n_overlap`** — Snapshots shared with the preceding segment, entry 1 is 0, at most the snapshots of the preceding
  segment after its own shared ones; `[n_segment]`
- **`overlap`** — Part of the shared snapshots used for the ramp, in [0, 1]; default: 0.5
- **`par`** — Unstructured data of the segments with the fields `tap` (tap index per path or tap, uint32), `path_type`
  (0 = LOS, 1 = GR, 2 = NLOS per path or tap, uint32, optional) and `n_ray_tap` (rays per tap, `'taps'` only); struct
  array of the size of `segments` that has the field `tap` or `path_type` (other fields are ignored); `[]` = no tap
  indices, all paths are NLOS; default: `[]`

## Outputs:
- **`channel`** — The joined channel, 1x1 struct, fields in [[hdf5_read_channel]]; `par` is not carried

## Notes:
- **Snapshots**: segment $k$ starts with `n_overlap(k)` snapshots that are also the last ones of segment $k - 1$; the
  output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots
- **Ramp**: in the last $n_r = \mathrm{round}(\mathrm{overlap} \cdot n_{\mathrm{overlap},k})$ shared snapshots, the old
  paths fade out with $1 - w_i$ and the new ones fade in with $w_i$, in power, with
  $$w_i = \frac{1}{2} \left( 1 + \sin\left( \left( \frac{i}{n_r + 1} - \frac{1}{2} \right) \pi \right) \right), \quad i = 1, \dots, n_r$$
  Before the ramp, the snapshots are those of the old segment
- **LOS and GR** (types 0 and 1 of the entry `path_type` of `par`; without it, tap index 0 of the entry `tap` and tap
  index 3 held by one ray, without index 4 or 5, as [[ssf_rays]] numbers them): one path with the data of the new
  segment (of the old one where the new one has no power), scaled to
  $\lvert h \rvert^2 = (1 - w) \lvert h_\mathrm{old} \rvert^2 + w \lvert h_\mathrm{new} \rvert^2$ by the path gain (rays) or the tap power (taps)
- **Rays** (channels with `path_gain`): the NLOS paths of both segments side by side, with the amplitudes $\sqrt{1 - w}$
  and $\sqrt{w}$; `path_length`, `path_angles`, `fbs_pos` and `lbs_pos` are kept where all segments have them
- **Taps** (without `path_gain`, one tap count per segment): one set of tap slots for all snapshots (LOS, GR, NLOS); the
  NLOS taps of the old segment hold $1 - w$ and those of the new segment $w$ of their mean power over the ramp; slots
  without power hold zero coefficients and the smallest delay of the snapshot
- **Other data**: delays as given (one reference for all segments); positions interpolated in the ramp, orientations
  along the shorter way; `name` and `center_freq` of segment 1; `n_interact` and `interact_coord` are not carried
- **Tap indices**: the C++ channels carry them in their `par_data`; in MATLAB they are the input `par`, as in
  [[hdf5_write_channel]]: the entries `tap`, `path_type` and `n_ray_tap` of each segment, as returned by
  [[gbsm_single_link_channel]]

## See also:
- [[get_channels_from_rays]] (joins the segments of tracks with this function)
- [[gbsm_single_link_channel]] (channels of one segment)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::vector<quadriga_lib::channel<double>> segments = qd_mex_struct2channel(prhs[0], false, false, "segments"); // C++ validates
    const arma::uvec n_overlap = qd_mex_get_Col<arma::uword>(prhs[1], false, "n_overlap");
    const double overlap = (nrhs < 3) ? 0.5 : qd_mex_get_scalar<double>(prhs[2], "overlap", 0.5);

    // Tap indices and path types of each segment (the only entries of par_data that the function reads)
    if (nrhs > 3 && !mxIsEmpty(prhs[3]))
    {
        qd_mex_check_struct(prhs[3], "par");
        if (mxGetFieldNumber(prhs[3], "tap") < 0 && mxGetFieldNumber(prhs[3], "path_type") < 0)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'par' has no field 'tap' or 'path_type'.");
        if (mxGetNumberOfElements(prhs[3]) != segments.size())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'par' and 'segments' must have the same number of entries.");
        for (size_t k = 0; k < segments.size(); ++k)
            for (const char *name : {"tap", "path_type", "n_ray_tap"})
            {
                const mxArray *data = mxGetField(prhs[3], (mwIndex)k, name);
                if (data == nullptr || mxIsEmpty(data))
                    continue;
                segments[k].par_names.push_back(name);
                segments[k].par_data.push_back(std::any(qd_mex_get_Col<unsigned>(data, true, std::string("par.") + name)));
            }
    }

    quadriga_lib::channel<double> channel;
    CALL_QD(channel = quadriga_lib::crossfade_segments(segments, n_overlap, overlap));

    plhs[0] = qd_mex_channel2struct({channel});
}
