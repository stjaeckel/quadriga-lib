// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"
#include "qd_helpers_wrappers.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!SECTION_DESC
A **channel** describes the propagation between a transmit and a receive antenna array: for each snapshot, a set of
propagation paths, each with a complex coefficient per antenna pair and a delay. The functions of this section store
channels in files, turn them into frequency responses, prepare them for link-level simulation, compute their statistics
and show the paths in 3D. The channels themselves come from the functions of section "Channel generators" or from
[[get_channels_layout]].<br><br>

**What a channel holds** (a struct, fields in [[hdf5_read_channel]]):

- The positions and orientations of the transmitter and the receiver, per snapshot or fixed
- The complex **coefficients** `coeff` and the **delays** `delay`, `[n_rx, n_tx, n_path, n_snap]`
- Optionally, antenna-independent path data: path gain, path length, polarization transfer matrix, angles, scatterer
  positions and the interaction points of ray-traced paths
- Unstructured metadata: named values of any type

**In MATLAB / Octave:**

- Several channels are a 1xN struct array. A field with one entry per snapshot is one array with the snapshot on the
  last dimension, zero-padded where the number of paths differs between snapshots
- [[hdf5_read_channel]] and [[hdf5_write_channel]] take and return the unstructured data as a struct array `par` of
  the size of the channel struct array, one field per dataset

**Typical workflow:**

1. Generate the channels, or read them: [[hdf5_read_channel]] from a channel file, [[qrt_file_read]] from the output
   of the QRT ray tracer
2. Optionally, map the delays to a tap grid with [[quantize_delays]], or join the segments of a track with
   [[crossfade_segments]]
3. Compute the frequency response at the subcarriers with [[freq_response]], or with [[freq_response_mf]] for
   coefficients sampled at several frequencies
4. Evaluate the channels: [[calc_delay_spread]], [[calc_angular_spread]], [[calc_rician_k_factor]], [[calc_xpr]]
5. Store them with [[hdf5_write_channel]], or show the paths with [[export_channel_obj]]

**Terms** (full list in the [Glossary](cpp_api.html#glossary)):

- **Snapshot**: one sample of the channel in time or position; `n_snap` snapshots, each with its own number of paths
- **Path**: one propagation path from the transmitter to the receiver, e.g. the LOS path, a reflection or a ray of a
  cluster
- **Coefficient**: the complex amplitude $a$ of one path for one antenna pair, with the antenna patterns, the
  polarization and the path gain
- **Delay**: the propagation delay $\tau$ of a path in s, per antenna pair `[n_rx, n_tx, n_path]` or shared by all
  pairs `[1, 1, n_path]`
- **CIR**: the channel impulse response of one snapshot, its coefficients and delays
- **Frequency response**: the channel matrix at subcarrier $k$, $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$, with
  $f_k$ the offset of the subcarrier from the carrier (the absolute frequency for `_mf`)
- **Multi-frequency channel** (suffix `_mf`): coefficients sampled at several frequencies, the frequency on the 4th
  dimension, as [[get_channels_multifreq]] produces them
- **Slot**: one storage place of an HDF5 channel file, addressed by four indices `(ix, iy, iz, iw)`, e.g. Tx, Rx and
  frequency; the file holds a fixed grid of slots, its **storage layout**
- **Origin**, **destination**: a QRT file holds the paths from each origin (a Tx position) to a list of CIR positions; a
  destination (a receiver) owns a run of consecutive CIR positions, e.g. a track

**File formats.** [HDF5 channel files](formats.html#hdf5_channel) store channels with antennas, one per slot, in single
precision; [QRT files](formats.html#qrt) store propagation paths from ray tracing or the GBSM before antennas are
applied.

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Store and read channels | [[hdf5_create_file]], [[hdf5_write_channel]], [[hdf5_read_channel]], [[hdf5_read_layout]], [[hdf5_reshape_layout]] |
| Store and read metadata | [[hdf5_write_dset]], [[hdf5_read_dset]], [[hdf5_read_dset_names]] |
| Path files (QRT) | [[qrt_file_parse]], [[qrt_file_read]] |
| Frequency response | [[freq_response]], [[freq_response_mf]] |
| Process | [[quantize_delays]], [[crossfade_segments]] |
| Statistics | [[calc_delay_spread]], [[calc_angular_spread]], [[calc_rician_k_factor]], [[calc_xpr]] |
| Show | [[export_channel_obj]] |

**Conventions.** Coefficients and delays of several snapshots are one array with the snapshot on the last dimension,
as in the channel struct, or cell arrays with one array per snapshot ([[freq_response]]). Indices are 1-based, also the
slot indices of the HDF5 files and the CIR and origin indices of the QRT files.
SECTION_DESC!*/

/*!MD
# HDF5_READ_CHANNEL
Read channel objects from one or several slots of an HDF5 channel file

- Returns the channels of the selected slots that hold data as a struct array `chan`, and their unstructured data as
  a struct array `par` of the same size
- `ix`, `iy`, `iz` and `iw` select the slots along each dimension; the selection is every combination, in
  column-major order (`ix` fastest), and empty slots are skipped ([[hdf5_read_layout]] tells which slots hold data)
- `i_snap` reads only the given snapshots of every selected slot (see Notes)
- No data in the selection gives `0x0` structs

## Usage:
```
[ chan, par ] = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, i_snap );
```

## Inputs:
- **`fn`** — File name of an existing channel file
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot indices along x, y, z and w, 1-based, inside the layout; scalar
  or vector; default: `[]` (all slots of the dimension)
- **`i_snap`** — Snapshot indices, 1-based, any order, repeats allowed, at most `n_snap` of every selected slot that
  has snapshots; `[n_out, 1]`; default: `[]` (all snapshots)

## Outputs:
- **`chan`** — Channels, one per selected slot that holds data; struct array `[1, n_chan]` with the fields:<br><br>
  | Field | Description | Class / size |
  | ----- | ----------- | ------------ |
  | `name` | Channel name | char |
  | `tx_pos`, `rx_pos` | Tx and Rx positions in m | `[3, 1]` (fixed) or `[3, n_snap]` |
  | `tx_orientation`, `rx_orientation` | Tx and Rx orientations (bank, tilt, heading) in rad | `[3, 1]` (fixed) or `[3, n_snap]` |
  | `coeff` | Complex coefficients | `[n_rx, n_tx, n_path, n_snap]` |
  | `delay` | Delays in s | `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]` |
  | `path_gain` | Path gain without antennas, linear | `[n_path, n_snap]` |
  | `path_length` | Path length in m | `[n_path, n_snap]` |
  | `polmat` | Polarization transfer matrix (Glossary) | `[8, n_path, n_snap]` |
  | `path_angles` | AOD, EOD, AOA, EOA in rad | `[n_path, 4, n_snap]` |
  | `fbs_pos`, `lbs_pos` | First- and last-bounce scatterer positions in m | `[3, n_path, n_snap]` |
  | `n_interact` | Number of interaction points per path, 0 = LOS | uint32, `[n_path, n_snap]` |
  | `interact_coord` | Interaction points of all paths, concatenated | `[3, max(sum(n_interact)), n_snap]` |
  | `center_freq` | Center frequency in Hz | scalar or `[n_snap, 1]` |
  | `initial_position` | Reference snapshot, 1-based; a missing field means 1 | int32 scalar |

  `n_path` is the largest number of paths of the snapshots; smaller snapshots are zero-padded. A field exists if a
  channel of the selection has it; the other channels get an empty value
- **`par`** — Unstructured data; struct array `[1, n_chan]` whose fields are the dataset names of all channels
  (without the prefix `par_`); a channel without a dataset gets an empty value; `0x0` struct if no channel has
  unstructured data

## Notes:
- **Snapshot selection**: fields with one entry per snapshot are sliced, also positions, orientations and
  `center_freq` stored per snapshot; values stored once and the unstructured data are kept. A slot without snapshots
  (only unstructured data) ignores `i_snap`
- **Round trip** with [[hdf5_write_channel]]: the values come back in single precision; empty orientations come back as
  `[3, 1]` zeros
- **Unstructured data**: the datasets of the slot with the prefix `par_`, in their stored class and shape
  ([[hdf5_read_dset]]); datasets of another prefix are not read
- **Errors**: a missing file, an index outside the layout, an index in `i_snap` beyond the snapshots of a slot, or a
  file that is not a channel file of the [HDF5 channel file format](formats.html#hdf5-compatibility)

## See also:
- [[hdf5_read_layout]] (finds the occupied slots)
- [[hdf5_write_channel]] (writes channels)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");

    // Slot indices (1-based -> 0-based); omitted or [] selects the full extent
    auto get_idx = [&](int arg_idx, const char *name) -> arma::u32_vec
    {
        if (arg_idx >= nrhs || mxGetNumberOfElements(prhs[arg_idx]) == 0)
            return arma::u32_vec();
        return qd_mex_get_index_Col<unsigned>(prhs[arg_idx], name);
    };
    const arma::u32_vec ix = get_idx(1, "ix"), iy = get_idx(2, "iy"), iz = get_idx(3, "iz"), iw = get_idx(4, "iw");
    const arma::uvec i_snap = (nrhs < 6) ? arma::uvec() : qd_mex_get_index_Col(prhs[5], "i_snap"); // 1-based -> 0-based

    // Occupied slots of the selection, then their channels
    std::vector<quadriga_lib::channel<double>> channels;
    arma::u32_vec dims, channel_id;
    CALL_QD(dims = quadriga_lib::hdf5_read_layout(fn, &channel_id));
    CALL_QD(channels = quadriga_lib::hdf5_read_channel<double>(fn, qd_hdf5_select_slots(fn, dims, channel_id, ix, iy, iz, iw), i_snap));

    // Keep par-only channels (c.empty() ignores par_data); check the structured data
    channels.erase(std::remove_if(channels.begin(), channels.end(), [](const quadriga_lib::channel<double> &c)
                                  { return c.empty() && c.par_names.empty(); }),
                   channels.end());
    for (const auto &c : channels)
        if (!c.empty())
        {
            const std::string error_msg = c.is_valid();
            if (!error_msg.empty())
                qd_mex_cpp_error(error_msg);
        }

    // Output 1: chan struct array (qd_mex_channel2struct handles empty vector)
    plhs[0] = qd_mex_channel2struct(channels, false);

    // Output 2: par struct array with the union of par_names across all channels
    if (nlhs > 1)
    {
        // Collect union of field names, preserving first-seen order
        std::vector<std::string> par_field_names;
        for (const auto &c : channels)
            for (const auto &n : c.par_names)
                if (std::find(par_field_names.begin(), par_field_names.end(), n) == par_field_names.end())
                    par_field_names.push_back(n);

        if (par_field_names.empty() || channels.empty())
        {
            std::vector<std::string> none;
            plhs[1] = qd_mex_make_struct(none, 0);
        }
        else
        {
            plhs[1] = qd_mex_make_struct(par_field_names, channels.size());
            for (size_t k = 0; k < channels.size(); ++k)
            {
                const auto &c = channels[k];
                for (size_t n = 0; n < c.par_names.size(); ++n)
                    qd_mex_set_field(plhs[1], c.par_names[n], qd_mex_any2matlab(c.par_data[n]), k);
            }
        }
    }
}
