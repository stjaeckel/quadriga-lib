// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_mex_layout.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# CROSSFADE_SEGMENTS
Joins the channels of consecutive segments of one link by a power cross-fade

- Port of the channel merger of QuaDRiGa: segment `k` starts with `n_overlap[k]` snapshots that are also the last ones of segment `k − 1`; the output has `Σ (n_snap[k] − n_overlap[k])` snapshots
- In the last `n_r = round(overlap · n_overlap[k])` shared snapshots (the ramp) the old paths fade out with `1 − w` and the new ones fade in with `w`, in power, `w = 0.5 (1 + sin((i / (n_r + 1) − 0.5) π))`, `i = 1 … n_r`; before the ramp the snapshots are those of the old segment
- LOS and GR (types 0 and 1 of the entry `path_type` of `par`; without it, tap index 0 of the entry `tap` and tap index 3 held by one ray, without index 4 or 5, as [[ssf_rays]] numbers them): one path with the data of the new segment (of the old one where the new one has no power), scaled to `|h|² = (1 − w) |h_old|² + w |h_new|²` by the path gain (rays) or the tap power (taps)
- `'rays'` (channels with `path_gain`): the NLOS paths of both segments side by side, amplitudes `sqrt(1 − w)` and `sqrt(w)`; `path_length`, `path_angles`, `fbs_pos` and `lbs_pos` are kept where all segments have them
- `'taps'` (without `path_gain`, one tap count per segment): one set of tap slots for all snapshots (LOS, GR, NLOS); the NLOS taps of the old segment hold `1 − w` and those of the new segment `w` of their mean power over the ramp; slots without power hold zero coefficients and the smallest delay of the snapshot
- Delays as given (one reference for all segments); positions interpolated in the ramp, orientations along the shorter way; `name` and `center_freq` of segment 0; `par_data`, `n_interact` and `interact_coord` are not carried
- The C++ channels carry the tap indices in their `par_data`; in MATLAB they are the optional input `par`, as in [[hdf5_write_channel]]: the entries `tap`, `path_type` and `n_ray_tap` of each segment, as returned by [[gbsm_single_link_channel]] (any other field is ignored)

## Usage:
```
channel = quadriga_lib.crossfade_segments( segments, n_overlap, overlap, par );
```

## Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients, equal `n_rx`, `n_tx`, form and `center_freq` (at most one value); struct array, fields in [[hdf5_read_channel]]; `[1, n_segment]`
- **`n_overlap`** — Snapshots shared with the preceding segment, entry 1 is 0, at most the snapshots of segment `k - 1` after its own shared ones; `[n_segment]`
- **`overlap`** — Part of the shared snapshots used for the ramp, in [0, 1]; default: 0.5
- **`par`** — Unstructured data of the segments with the fields `tap` (tap index per path or tap, uint32), `path_type` (0 = LOS, 1 = GR, 2 = NLOS per path or tap, uint32, optional) and `n_ray_tap` (rays per tap, `'taps'` only); struct array of the size of `segments` that has the field `tap` or `path_type` (other fields are ignored); `[]` = no tap indices, all paths are NLOS; default: `[]`

## Outputs:
- **`channel`** — The joined channel, 1x1 struct, fields in [[hdf5_read_channel]]; `par` is not carried

## See also:
- [[get_gbsm_channels]] (joins the segments of tracks with this function)
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
