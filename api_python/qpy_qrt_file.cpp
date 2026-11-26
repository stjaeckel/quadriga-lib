// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# qrt_file_parse
Read metadata from a QRT file

- Reads the header of a QRT file: counts, names, frequencies, the positions and orientations of the CIRs and the
  origins, and the file version; no path data
- A file without destinations reports one destination `RX` that owns all CIRs
- Raises an error for a file that cannot be opened, is not a QRT file of version 4, 5 or 6, or is truncated or corrupt

## Usage:
```
n_cir, n_orig, n_dest, n_freq, cir_offset, orig_names, dest_names, version, center_freq, \
    cir_pos, cir_orientation, orig_pos, orig_orientation = quadriga_lib.channel.qrt_file_parse( fn )
```

## Input:
- **`fn`** — File name, str or path-like

## Outputs:
- **`n_cir`** — Number of CIRs of each origin; int
- **`n_orig`** — Number of origin slots; int
- **`n_dest`** — Number of destinations, at least 1; int
- **`n_freq`** — Number of frequencies; int
- **`cir_offset`** — Number of CIRs before each destination, i.e. the 0-based index of its first CIR; int64;
  `(n_dest,)`
- **`orig_names`** — Origin names, empty for a slot not written yet; list of str of length `n_orig`
- **`dest_names`** — Destination names; list of str of length `n_dest`
- **`version`** — File version: 4 (EM, legacy), 5 (EM) or 6 (scalar); int
- **`center_freq`** — Frequencies in Hz; `(n_freq,)`. Version 4 stores GHz and one frequency per origin, this is the one
  of the first written origin (0 if none is written); version 5 files of earlier library versions store GHz too, a
  version 5 list with all values below 1000 is read as GHz
- **`cir_pos`** — CIR positions in m; `(n_cir, 3)`
- **`cir_orientation`** — CIR orientations (bank, tilt, heading) in rad; `(n_cir, 3)`
- **`orig_pos`** — Origin positions in m, zeros for a slot not written yet; `(n_orig, 3)`
- **`orig_orientation`** — Origin orientations (bank, tilt, heading) in rad; `(n_orig, 3)`

## See also:
- [[qrt_file_read]] (reads CIRs)
MD!*/

py::tuple qrt_file_parse(const py::handle &fn)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    arma::uword n_cir, n_orig, n_dest, n_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;
    arma::fvec center_freq;
    arma::fmat cir_pos, cir_orientation, orig_pos, orig_orientation;

    quadriga_lib::qrt_file_parse(fn_s, &n_cir, &n_orig, &n_dest, &n_freq, &cir_offset,
                                 &orig_names, &dest_names, &version, &center_freq,
                                 &cir_pos, &cir_orientation, &orig_pos, &orig_orientation);

    auto cir_offset_py = qd_python_copy2numpy<arma::uword, py::ssize_t>(&cir_offset);
    auto orig_names_py = qd_python_copy2list(orig_names);
    auto dest_names_py = qd_python_copy2list(dest_names);
    auto center_freq_py = qd_python_copy2numpy(&center_freq);
    auto cir_pos_py = qd_python_copy2numpy(&cir_pos);
    auto cir_orientation_py = qd_python_copy2numpy(&cir_orientation);
    auto orig_pos_py = qd_python_copy2numpy(&orig_pos);
    auto orig_orientation_py = qd_python_copy2numpy(&orig_orientation);

    return py::make_tuple(n_cir, n_orig, n_dest, n_freq, cir_offset_py, orig_names_py,
                          dest_names_py, version, center_freq_py, cir_pos_py, cir_orientation_py,
                          orig_pos_py, orig_orientation_py);
}

/*!MD
# qrt_file_read
Read ray-tracing CIR data from a QRT file

- Reads the paths of a list of CIRs of one origin, with the positions and orientations of both ends: the inputs of
  [[channel.spherical]] (`fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, `polmat`) or [[channel.planar]] (the
  angles)
- Reads in chunks of 16 CIRs: besides the outputs, at most two chunks are held in memory
- `uplink` swaps the ends: the origin becomes the receiver (see Notes)

## Usage:
```
center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, \
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, \
    interact_type = quadriga_lib.channel.qrt_file_read( fn, i_cir, i_orig, *, uplink, normalize_polmat )
```

## Inputs:
- **`fn`** — File name, str or path-like
- **`i_cir`** — CIR indices, 0-based, below `n_cir`; `(n_out,)`; default: `None` (all CIRs of the file)
- **`i_orig`** — Origin index, 0-based, below `n_orig`; int; default: 0
- **`uplink`** — `False`: origin = Tx, destination = Rx; `True`: origin = Rx, destination = Tx; default: `False`
- **`normalize_polmat`** — Scaling of `polmat` and `path_gain`, 0 or 1 (see Notes); default: 1

## Outputs:
- **`center_freq`** — Frequencies in Hz (version 4: of origin `i_orig`); `(n_freq,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `(3, n_out)`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientations (bank, tilt, heading) in rad; `(3, n_out)`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m: the first and the last interaction
  point, the midpoint of Tx and Rx for a LOS path; list of `n_out` arrays `(3, n_path)`
- **`path_gain`** — Path gain, linear (see Notes); list of `n_out` arrays `(n_path, n_freq)`
- **`path_length`** — Length of the polyline from the Tx through the interaction points to the Rx in m; list of
  `n_out` arrays `(n_path,)`
- **`polmat`** — Polarization transfer matrix; list of `n_out` arrays `(8, n_path, n_freq)`; version 6: the
  complex scalar coefficient, `(2, n_path, n_freq)`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure (at the Tx) and arrival (at the
  Rx) in rad; list of `n_out` arrays `(n_path,)`
- **`path_coord`** — Points of each path from the Tx through the interaction points to the Rx; list of `n_out` lists
  of `n_path` arrays `(3, n_interact + 2)`
- **`n_interact`** — Number of interaction points of each path, 0 = LOS; list of `n_out` int64 arrays `(n_path,)`
- **`coord`** — Interaction points of all paths, concatenated in path order; list of `n_out` arrays
  `(3, sum(n_interact))`
- **`interact_type`** — Interaction type codes, concatenated like `coord`; list of `n_out` uint8 arrays
  `(sum(n_interact),)`; empty for version 4 files

## Notes:
- **Path gain**: the file stores per path the coefficients with the interaction (material) losses; the reader adds the
  propagation loss of the path length $d$, for EM files (versions 4 and 5) the free-space loss
  $\mathrm{PL} = 32.45 + 20 \log_{10}(f / \mathrm{GHz}) + 20 \log_{10}(d / \mathrm{m})$ dB, for scalar files (version 6)
  the spherical spreading and the air absorption $\alpha(f)$ of ISO 9613-1 at 20 °C, 50 % humidity and 1 atm,
  $\mathrm{PL} = 20 \log_{10}(d / \mathrm{m}) + \alpha(f) \, d$ dB. With $p$ the larger column power of the stored
  matrix (version 6: the power of the coefficient):<br><br>
  | `normalize_polmat` | `polmat` | `path_gain` |
  | ------------------ | -------- | ----------- |
  | 0 | As stored | $10^{-\mathrm{PL}/10}$ |
  | 1 | Stored matrix divided by $\sqrt{p}$: largest column power 1 (uplink: row power) | $p \cdot 10^{-\mathrm{PL}/10}$ |

- **Uplink**: `polmat` is the transpose of the downlink matrix (reciprocal channel, no conjugation) with the same
  `path_gain`; positions, orientations, angles, `fbs_pos`, `lbs_pos` and `path_coord` swap their ends.
  `n_interact`, `coord` and `interact_type` keep the stored order from the origin to the destination, and the type
  codes describe that traversal
- An origin slot that is not written yet gives zero positions and orientations and no paths
- GBSM paths of [[export_rays_qrt]] carry the code 193 at their scatterers; their LOS and GR paths look like
  ray-traced ones (no interaction, one reflection 129)
- **Errors**: an index out of range raises `ValueError`; a file that cannot be opened, is not a QRT file of version 4,
  5 or 6, or is truncated or corrupt (counts or offsets beyond the file, path blocks of different CIRs that overlap)
  raises `RuntimeError`. A message about one requested CIR starts with `Snapshot <k>: `, with `k` its position in
  `i_cir`

## See also:
- [[qrt_file_parse]] (metadata without path data)
- [[channel.spherical]] (coefficients from the paths)
- [[channel.planar]] (coefficients from the angles)
MD!*/

py::tuple qrt_file_read(const py::handle &fn,
                        py::handle i_cir,
                        arma::uword i_orig,
                        py::handle uplink,
                        int normalize_polmat)
{
    const std::string fn_s = qd_python_path(fn, "fn");
    const bool uplink_b = qd_python_flag(uplink, false, "uplink");
    const arma::uvec i_cir_a = qd_python_numpy2arma_Col<arma::uword>(i_cir, true, false, "i_cir"); // None or empty = all

    py::array_t<double> center_freq_py, tx_pos_py, tx_orientation_py, rx_pos_py, rx_orientation_py;
    arma::mat tx_pos, tx_orientation, rx_pos, rx_orientation;
    py::list fbs_pos_py, lbs_pos_py, path_gain_py, path_length_py, polmat_py;
    py::list aod_py, eod_py, aoa_py, eoa_py, path_coord_py, n_interact_py, coord_py, interact_type_py;

    auto init = [&](const arma::vec &center_freq, arma::uword n_out)
    {
        center_freq_py = qd_python_copy2numpy(&center_freq);
        tx_pos_py = qd_python_init_output(3, n_out, &tx_pos);
        tx_orientation_py = qd_python_init_output(3, n_out, &tx_orientation);
        rx_pos_py = qd_python_init_output(3, n_out, &rx_pos);
        rx_orientation_py = qd_python_init_output(3, n_out, &rx_orientation);
    };

    // Runs with the GIL; the reader thread reads the next chunk meanwhile
    auto consume = [&](const qd_qrt_chunk &b, arma::uword offset)
    {
        const arma::uword n = b.i_cir.n_elem;
        tx_pos.cols(offset, offset + n - 1) = b.tx_pos;
        tx_orientation.cols(offset, offset + n - 1) = b.tx_orientation;
        rx_pos.cols(offset, offset + n - 1) = b.rx_pos;
        rx_orientation.cols(offset, offset + n - 1) = b.rx_orientation;
        for (arma::uword j = 0; j < n; ++j)
        {
            fbs_pos_py.append(qd_python_copy2numpy(&b.fbs_pos[j]));
            lbs_pos_py.append(qd_python_copy2numpy(&b.lbs_pos[j]));
            path_gain_py.append(qd_python_copy2numpy(&b.path_gain[j]));
            path_length_py.append(qd_python_copy2numpy(&b.path_length[j]));
            polmat_py.append(qd_python_copy2numpy(&b.polmat[j]));
            aod_py.append(qd_python_copy2numpy(&b.aod[j]));
            eod_py.append(qd_python_copy2numpy(&b.eod[j]));
            aoa_py.append(qd_python_copy2numpy(&b.aoa[j]));
            eoa_py.append(qd_python_copy2numpy(&b.eoa[j]));
            path_coord_py.append(qd_python_copy2list(&b.path_coord[j]));
            n_interact_py.append(qd_python_copy2numpy<unsigned, py::ssize_t>(&b.n_interact[j]));
            coord_py.append(qd_python_copy2numpy(&b.coord[j]));

            const std::vector<uint8_t> &it = b.interact_type[j];
            const arma::Col<arma::u8> it_view(const_cast<uint8_t *>(it.data()), it.size(), false, true);
            interact_type_py.append(qd_python_copy2numpy(&it_view));
        }
    };

    qd_qrt_read_chunked<py::gil_scoped_release>(fn_s, i_cir_a, i_orig, uplink_b, normalize_polmat, 18, init, consume);

    return py::make_tuple(center_freq_py, tx_pos_py, tx_orientation_py, rx_pos_py, rx_orientation_py,
                          fbs_pos_py, lbs_pos_py, path_gain_py, path_length_py, polmat_py,
                          aod_py, eod_py, aoa_py, eoa_py, path_coord_py, n_interact_py, coord_py,
                          interact_type_py);
}
