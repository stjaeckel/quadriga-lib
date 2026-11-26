// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "qd_helpers_shared.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!SECTION_DESC
A **channel** describes the propagation between a transmit and a receive antenna array: for each snapshot, a set of
propagation paths, each with a complex coefficient per antenna pair and a delay. The functions of this section store
channels in files, turn them into frequency responses, prepare them for link-level simulation, compute their statistics
and show the paths in 3D. The channels themselves come from the functions of section "Channel generators" or from
[[get_channels_layout]].<br><br>

**What a channel holds** ([[channel]]):

- The positions and orientations of the transmitter and the receiver, per snapshot or fixed
- The **coefficients** `coeff_re` / `coeff_im` and the **delays** `delay` of each snapshot, `[n_rx, n_tx, n_path]`
- Optionally, antenna-independent path data: path gain, path length, polarization transfer matrix, angles, scatterer
  positions and the interaction points of ray-traced paths
- Unstructured metadata: named values of any type in `par_names` / `par_data`

**Typical workflow:**

1. Generate the channels, or read them: [[hdf5_read_channel]] from a channel file, [[qrt_file_read]] from the output
   of the QRT ray tracer
2. Optionally, map the delays to a tap grid with [[quantize_delays]], or join the segments of a track with
   [[crossfade_segments]]
3. Compute the frequency response at the subcarriers with [[freq_response]], or with [[freq_response_mf]] for
   coefficients sampled at several frequencies
4. Evaluate the channels: [[calc_channel_gain]], [[calc_delay_spread]], [[calc_angular_spread]],
   [[calc_rician_k_factor]], [[calc_xpr]]
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
- **Multi-frequency channel** (suffix `_mf`): coefficients sampled at several frequencies, one entry per frequency, as
  [[get_channels_multifreq]] produces them
- **Slot**: one storage place of an HDF5 channel file, addressed by four indices `(ix, iy, iz, iw)`, e.g. Tx, Rx and
  frequency; the file holds a fixed grid of slots, its **storage layout**
- **Origin**, **destination**: a QRT file holds the paths from each origin (a Tx position) to a list of CIR positions; a
  destination (a receiver) owns a run of consecutive CIR positions, e.g. a track

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Store and read channels | [[hdf5_create_file]], [[hdf5_write_channel]], [[hdf5_read_channel]], [[hdf5_read_layout]], [[hdf5_reshape_layout]] |
| Store and read metadata | [[hdf5_write_dset]], [[hdf5_read_dset]], [[hdf5_read_dset_names]] |
| Ray-tracing files | [[qrt_file_init]], [[qrt_file_append]], [[qrt_paths_from_scatterers]], [[qrt_file_parse]], [[qrt_file_read]], [[qrt_file_read_raw]], [[qrt_read_cache_init]] |
| Frequency response | [[freq_response]], [[freq_response_mf]] |
| Process | [[quantize_delays]], [[crossfade_segments]] |
| Statistics | [[calc_channel_gain]], [[calc_delay_spread]], [[calc_angular_spread]], [[calc_rician_k_factor]], [[calc_xpr]] |
| Show | [[export_channel_obj]] |

**Conventions.** Coefficients and delays are given per snapshot as a `std::vector` of cubes. [[freq_response]],
[[freq_response_mf]], [[hdf5_read_channel]] and [[hdf5_write_channel]] have vector forms, same-name overloads with one
entry per snapshot or slot. HDF5 files store the structured channel data in single precision.
SECTION_DESC!*/

/*!MD
# channel<++>
MIMO channel between a transmit and a receive array: coefficients, delays and path data per snapshot, and metadata

- The data of one link (section description, "What a channel holds"); the fields are public and set directly, e.g.
  from [[get_channels_spherical]], or read with [[hdf5_read_channel]] and [[qrt_file_read]]
- Snapshot fields are `std::vector` with one entry per snapshot; an empty field is not present. `n_snap` is the largest
  number of entries of any field; positions and orientations with one column apply to all snapshots
- `.is_valid()` checks the sizes; the functions that take a channel check it first

## Attributes:
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `std::string name` | at most 255 characters | Name of the channel, default `"empty"` |
| `arma::Col<dtype> center_freq` | `[1]`, `[n_snap]` or `[]` | Center frequency in Hz |
| `arma::Mat<dtype> tx_pos` | `[3, 1]` or `[3, n_snap]` | Transmitter positions in m |
| `arma::Mat<dtype> rx_pos` | `[3, 1]` or `[3, n_snap]` | Receiver positions in m |
| `arma::Mat<dtype> tx_orientation` | `[3, 1]`, `[3, n_snap]` or `[]` | Transmitter orientation, Euler angles (bank, tilt, heading) in rad |
| `arma::Mat<dtype> rx_orientation` | `[3, 1]`, `[3, n_snap]` or `[]` | Receiver orientation, Euler angles (bank, tilt, heading) in rad |
| `std::vector<arma::Cube<dtype>> coeff_re` | per snapshot `[n_rx, n_tx, n_path]` | Coefficients, real part; `n_rx` and `n_tx` equal in all snapshots |
| `std::vector<arma::Cube<dtype>> coeff_im` | per snapshot `[n_rx, n_tx, n_path]` | Coefficients, imaginary part |
| `std::vector<arma::Cube<dtype>> delay` | per snapshot `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` | Delays in s; `[1, 1, n_path]` without coefficients |
| `std::vector<arma::Col<dtype>> path_gain` | per snapshot `[n_path]` | Path gain, linear, without antennas |
| `std::vector<arma::Col<dtype>> path_length` | per snapshot `[n_path]` | Path length in m from the Tx to the Rx phase center |
| `std::vector<arma::Mat<dtype>> polmat` | per snapshot `[8, n_path]` | Polarization transfer matrices, interleaved real and imaginary parts |
| `std::vector<arma::Mat<dtype>> path_angles` | per snapshot `[n_path, 4]` | Departure and arrival angles (AOD, EOD, AOA, EOA) in rad |
| `std::vector<arma::Mat<dtype>> fbs_pos` | per snapshot `[3, n_path]` | First-bounce scatterer positions in m |
| `std::vector<arma::Mat<dtype>> lbs_pos` | per snapshot `[3, n_path]` | Last-bounce scatterer positions in m |
| `std::vector<arma::Col<unsigned>> n_interact` | per snapshot `[n_path]` | Number of interactions of each path, 0 = LOS; with `interact_coord` |
| `std::vector<arma::Mat<dtype>> interact_coord` | per snapshot `[3, sum(n_interact)]` | Interaction points of the paths in m, in path order |
| `std::vector<std::string> par_names` | `[n_par]` | Names of the metadata fields |
| `std::vector<std::any> par_data` | `[n_par]` | Metadata values, of a type [[any_type_id]] knows |
| `int initial_position` | scalar | 0-based index of the reference snapshot, mainly for internal use |

## Simple member functions:
| Method | Description |
| ------ | ----------- |
| `.n_snap()` | Number of snapshots |
| `.n_rx()` | Number of receive antennas; 0 without coefficients |
| `.n_tx()` | Number of transmit antennas; 0 without coefficients |
| `.n_path()` | Number of paths of each snapshot, `[n_snap]` |
| `.empty()` | True if the channel holds no structured data |
| `.is_valid()` | Empty string if the channel is valid, else an error message |

Computations and file I/O on a channel are free functions in this section, e.g. [[calc_channel_gain]],
[[export_channel_obj]], [[hdf5_write_channel]] and [[quantize_delays]].
MD!*/

// Object dimensions
template <typename dtype>
arma::uword quadriga_lib::channel<dtype>::n_snap() const
{
    arma::uword s_arma = 0ULL;

    if (center_freq.n_elem > s_arma) // 1 or s
        s_arma = center_freq.n_elem;

    if (tx_orientation.n_cols > s_arma) // 1 or s
        s_arma = tx_orientation.n_cols;

    if (rx_orientation.n_cols > s_arma) // 1 or s
        s_arma = rx_orientation.n_cols;

    if (rx_pos.n_cols > s_arma) // 1 or s
        s_arma = rx_pos.n_cols;

    if (tx_pos.n_cols > s_arma) // 1 or s
        s_arma = tx_pos.n_cols;

    size_t s_vector = (size_t)s_arma;

    if (coeff_re.size() > s_vector)
        s_vector = coeff_re.size();

    if (delay.size() > s_vector)
        s_vector = delay.size();

    if (path_gain.size() > s_vector)
        s_vector = path_gain.size();

    if (path_length.size() > s_vector)
        s_vector = path_length.size();

    if (polmat.size() > s_vector)
        s_vector = polmat.size();

    if (path_angles.size() > s_vector)
        s_vector = path_angles.size();

    if (fbs_pos.size() > s_vector)
        s_vector = fbs_pos.size();

    if (lbs_pos.size() > s_vector)
        s_vector = lbs_pos.size();

    if (n_interact.size() > s_vector)
        s_vector = n_interact.size();

    return (arma::uword)s_vector;
}

template <typename dtype>
arma::uword quadriga_lib::channel<dtype>::n_rx() const
{
    if (coeff_re.size() != 0)
        return coeff_re[0].n_rows;
    else
        return 0;
}

template <typename dtype>
arma::uword quadriga_lib::channel<dtype>::n_tx() const
{
    if (coeff_re.size() != 0)
        return coeff_re[0].n_cols;
    else
        return 0;
}

template <typename dtype>
arma::uvec quadriga_lib::channel<dtype>::n_path() const
{
    size_t n_snap = (size_t)this->n_snap();
    if (n_snap == 0)
        return arma::uvec();

    arma::uvec n_path(n_snap);
    auto *p_path = n_path.memptr();
    if (coeff_re.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = coeff_re[i].n_slices;
    else if (n_interact.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = n_interact[i].n_elem;
    else if (path_gain.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_gain[i].n_elem;
    else if (path_length.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_length[i].n_elem;
    else if (polmat.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = polmat[i].n_cols;
    else if (path_angles.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = path_angles[i].n_rows;
    else if (fbs_pos.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = fbs_pos[i].n_cols;
    else if (lbs_pos.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = lbs_pos[i].n_cols;
    else if (delay.size() == n_snap)
        for (size_t i = 0; i < n_snap; ++i)
            p_path[i] = delay[i].n_slices;

    return n_path;
}

// Returns true if the channel object contains no data
template <typename dtype>
bool quadriga_lib::channel<dtype>::empty() const
{
    if (center_freq.n_elem != 0ULL)
        return false;

    if (tx_pos.n_elem != 0ULL)
        return false;

    if (rx_pos.n_elem != 0ULL)
        return false;

    if (tx_orientation.n_elem != 0ULL)
        return false;

    if (rx_orientation.n_elem != 0ULL)
        return false;

    if (coeff_re.size() != 0)
        return false;

    if (delay.size() != 0)
        return false;

    if (path_gain.size() != 0)
        return false;

    if (path_length.size() != 0)
        return false;

    if (polmat.size() != 0)
        return false;

    if (path_angles.size() != 0)
        return false;

    if (fbs_pos.size() != 0)
        return false;

    if (lbs_pos.size() != 0)
        return false;

    if (n_interact.size() != 0)
        return false;

    if (interact_coord.size() != 0)
        return false;

    if (initial_position != 0)
        return false;

    return true;
}

template <typename dtype>
std::string quadriga_lib::channel<dtype>::is_valid() const
{
    arma::uword n_snap_arma = this->n_snap();
    size_t n_snap_vector = (size_t)n_snap_arma;
    arma::uword n_tx = 0ULL, n_rx = 0ULL;
    arma::uvec n_pth_v = this->n_path();

    if (name.length() > 255)
        return std::string("Name can have at most 255 characters.");

    if (n_pth_v.n_elem != n_snap_arma)
        return std::string("Number of elements returned by 'n_path()' does not match number of snapshots.");
    auto n_pth = n_pth_v.memptr();

    if (n_snap_arma != 0ULL && rx_pos.n_rows != 3ULL)
        return std::string("'rx_pos' is missing or ill-formatted (must have 3 rows).");

    if (rx_pos.n_cols != 1ULL && rx_pos.n_cols != n_snap_arma)
        return std::string("Number of columns in 'rx_pos' must be 1 or match the number of snapshots.");

    if (n_snap_arma != 0ULL && tx_pos.n_rows != 3ULL)
        return std::string("'tx_pos' is missing or ill-formatted (must have 3 rows).");

    if (tx_pos.n_cols != 1ULL && tx_pos.n_cols != n_snap_arma)
        return std::string("Number of columns in 'tx_pos' must be 1 or match the number of snapshots.");

    if (center_freq.n_elem != 0ULL && center_freq.n_elem != 1ULL && center_freq.n_elem != n_snap_arma)
        return std::string("Number of entries in 'center_freq' must be 0, 1 or match the number of snapshots.");

    if (tx_orientation.n_elem != 0ULL && tx_orientation.n_rows != 3ULL)
        return std::string("'tx_orientation' must be empty or have 3 rows.");

    if (tx_orientation.n_elem != 0ULL && tx_orientation.n_cols != 1ULL && tx_orientation.n_cols != n_snap_arma)
        return std::string("Number of columns in 'tx_orientation' must be 1 or match the number of snapshots.");

    if (rx_orientation.n_elem != 0ULL && rx_orientation.n_rows != 3ULL)
        return std::string("'rx_orientation' must be empty or have 3 rows.");

    if (rx_orientation.n_elem != 0ULL && rx_orientation.n_cols != 1ULL && rx_orientation.n_cols != n_snap_arma)
        return std::string("Number of columns in 'rx_orientation' must be 1 or match the number of snapshots.");

    if (coeff_re.size() != 0 && coeff_re.size() != n_snap_vector)
        return std::string("'coeff_re' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && coeff_re.size() == n_snap_vector)
    {
        if (coeff_im.size() != n_snap_vector)
            return std::string("Imaginary part of channel coefficients 'coeff_im' is missing or incomplete.");

        if (delay.size() != n_snap_vector)
            return std::string("Delays are missing or incomplete.");

        n_rx = coeff_re[0].n_rows, n_tx = coeff_re[0].n_cols;

        for (size_t i = 0; i < n_snap_vector; ++i)
        {
            if (coeff_re[i].n_rows != n_rx || coeff_re[i].n_cols != n_tx || coeff_re[i].n_slices != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'coeff_re'.";

            if (coeff_im[i].n_rows != n_rx || coeff_im[i].n_cols != n_tx || coeff_im[i].n_slices != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'coeff_im'.";

            if (!(delay[i].n_rows == n_rx && delay[i].n_cols == n_tx && delay[i].n_slices == n_pth[i]) &&
                !(delay[i].n_rows == 1 && delay[i].n_cols == 1 && delay[i].n_slices == n_pth[i]))
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'delay'.";
        }
    }
    else if (!coeff_im.empty())
        return std::string("Real part of channel coefficients 'coeff_re' is missing or incomplete.");
    else if (!delay.empty())
    {
        if (delay.size() != n_snap_vector)
            return std::string("'delay' must be empty or match the number of snapshots.");

        for (size_t i = 0; i < n_snap_vector; ++i)
            if (delay[i].n_rows != 1 || delay[i].n_cols != 1 || delay[i].n_slices != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'delay'.";
    }

    if (path_gain.size() != 0 && path_gain.size() != n_snap_vector)
        return std::string("'path_gain' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_gain.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_gain[i].n_elem != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'path_gain'.";

    if (path_length.size() != 0 && path_length.size() != n_snap_vector)
        return std::string("'path_length' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_length.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_length[i].n_elem != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'path_length'.";

    if (polmat.size() != 0 && polmat.size() != n_snap_vector)
        return std::string("'polmat' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && polmat.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (polmat[i].n_rows != 8ULL || polmat[i].n_cols != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'polmat'.";

    if (path_angles.size() != 0 && path_angles.size() != n_snap_vector)
        return std::string("'path_angles' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && path_angles.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (path_angles[i].n_rows != n_pth[i] || path_angles[i].n_cols != 4ULL)
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'path_angles'.";

    if (fbs_pos.size() != 0 && fbs_pos.size() != n_snap_vector)
        return std::string("'fbs_pos' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && fbs_pos.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (fbs_pos[i].n_rows != 3ULL || fbs_pos[i].n_cols != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'fbs_pos'.";

    if (lbs_pos.size() != 0 && lbs_pos.size() != n_snap_vector)
        return std::string("'lbs_pos' must be empty or match the number of snapshots.");

    if (n_snap_arma != 0ULL && lbs_pos.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
            if (lbs_pos[i].n_rows != 3ULL || lbs_pos[i].n_cols != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'lbs_pos'.";

    if (n_interact.size() != 0 && n_interact.size() != n_snap_vector)
        return std::string("'n_interact' must be empty or match the number of snapshots.");
    else if (n_interact.size() == 0 && interact_coord.size() != 0)
        return std::string("'interact_coord' is provided but 'n_interact' is missing.");
    else if (n_interact.size() == n_snap_vector && interact_coord.size() != n_snap_vector)
        return std::string("'n_interact' is provided but 'interact_coord' is missing or has wrong number of snapshots.");
    else if (n_interact.size() == n_snap_vector && interact_coord.size() == n_snap_vector)
        for (size_t i = 0; i < n_snap_vector; ++i)
        {
            if (n_interact[i].n_elem != n_pth[i])
                return "Snapshot " + std::to_string(i) + ": Size mismatch in 'n_interact'.";

            if (interact_coord[i].n_rows != 3ULL)
                return "Snapshot " + std::to_string(i) + ": 'interact_coord' must have 3 rows.";

            arma::uword cnt = 0;
            for (const unsigned *p = n_interact[i].begin(); p < n_interact[i].end(); ++p)
                cnt += (arma::uword)*p;

            if (interact_coord[i].n_cols != cnt)
                return "Snapshot " + std::to_string(i) + ": Number of columns in 'interact_coord' must match the sum of 'n_interact'.";
        }

    if (par_names.size() != par_data.size())
        return std::string("Number of elements in 'par_data' must match number of elements in 'par_names'.");

    for (size_t i = 0; i < par_names.size(); ++i)
        if (any_type_id(&par_data[i]) < 0)
            return std::string("Unsupported datatype in unstructured data.");

    return std::string("");
}

template class quadriga_lib::channel<float>;
template class quadriga_lib::channel<double>;
