// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# qrt_file_parse
Read metadata from a QRT file

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count, CIR
  offsets, names, positions, orientations, and file version
- When `n_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

## Usage:
```
n_cir, n_orig, n_dest, n_freq, cir_offset, orig_names, dest_names, version, center_freq, \
    cir_pos, cir_orientation, orig_pos, orig_orientation = quadriga_lib.channel.qrt_file_parse( fn )
```

## Input:
- **`fn`** — Path to the QRT file

## Outputs:
- **`n_cir`** — Number of channel snapshots per origin point
- **`n_orig`** — Number of origin points (TX)
- **`n_dest`** — Number of destination points (RX)
- **`n_freq`** — Number of frequency bands
- **`cir_offset`** — Number of CIRs before each destination (offset of its first CIR), the first entry 0; int64; `(n_dest,)`
- **`orig_names`** — Names of the origin points (TX); list of strings; length `n_orig`
- **`dest_names`** — Names of the destination points (RX); list of strings; length `n_dest`
- **`version`** — QRT file version number
- **`center_freq`** — Center frequencies in Hz, converted from GHz for files that store GHz (v4, and v5 when all stored values are below 1000); `(n_freq,)`
- **`cir_pos`** — CIR positions in Cartesian coordinates; `(n_cir, 3)`
- **`cir_orientation`** — CIR orientations as Euler angles; `(n_cir, 3)`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; `(n_orig, 3)`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; `(n_orig, 3)`
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

- Reads channel impulse response data from QRT files
- Reads in chunks of 16 snapshots; besides the output, at most two chunks are held in memory
- If `uplink = False`, origin is TX and destination is RX; if `True`, roles are swapped and `polmat` is the transpose
  of the downlink `polmat` (reciprocal channel, no conjugation) with the same `path_gain`; `n_interact`, `coord` and
  `interact_type` keep the stored order (origin to destination) in both directions, and the type codes describe
  that traversal
- Per-snapshot outputs are returned as lists with one entry per requested snapshot

## Usage:
```
center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, \
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, \
    interact_type = quadriga_lib.channel.qrt_file_read( fn, i_cir, i_orig, *, uplink, normalize_polmat )
```

## Inputs:
- **`fn`** — Path to the QRT file
- **`i_cir`** — Snapshot indices to read, 0-based; `(n_out,)` or `None`; if `None` or empty, all snapshots are read (for a file
  without CIRs, all outputs are empty with `n_out = 0`); default: `None`
- **`i_orig`** — Origin index, 0-based (origin = TX for downlink); scalar; default: 0
- **`uplink`** — If `False`, origin=TX and destination=RX; if `True`, the roles are swapped (origin=RX, destination=TX); default: `False`
- **`normalize_polmat`** — Controls `polmat` and `path_gain` scaling; 0 = values as stored in the QRT file with
  `path_gain` = -PL; 1 = `polmat` columns (uplink: rows) scaled to max power 1 and `path_gain` = -PL minus material losses; default: 1
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm<br><br>
    | `normalize_polmat` | `polmat`                                   | `path_gain`                      |
    | :-----------: | :---------------------------------------------: | :------------------------------: |
    | 0             | As stored in QRT file                           | -PL                              |
    | 1             | Max column power = 1; uplink: max row power = 1 | -PL minus material losses        |<br><br>

    Other values raise an error.

## Outputs:
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`
- **`tx_pos`** — Transmitter positions in Cartesian coordinates; `(3, n_out)`
- **`tx_orientation`** — Transmitter orientations as Euler angles (bank, tilt, heading); `(3, n_out)`
- **`rx_pos`** — Receiver positions in Cartesian coordinates; `(3, n_out)`
- **`rx_orientation`** — Receiver orientations as Euler angles (bank, tilt, heading); `(3, n_out)`
- **`fbs_pos`** — First-bounce scatterer positions; list of length `n_out`; entries `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; list of length `n_out`; entries `(3, n_path)`
- **`path_gain`** — Path gain in linear scale; list of length `n_out`; entries `(n_path, n_freq)`
- **`path_length`** — Absolute path length from TX to RX phase center; list of length `n_out`; entries `(n_path,)`
- **`polmat`** — Polarization transfer matrix, stored as interleaved real/imaginary pairs; list of length `n_out`; entries `(8, n_path, n_freq)`, or `(2, n_path, n_freq)` for v6 files
- **`aod`** — Departure azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eod`** — Departure elevation angles; list of length `n_out`; entries `(n_path,)`
- **`aoa`** — Arrival azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eoa`** — Arrival elevation angles; list of length `n_out`; entries `(n_path,)`
- **`path_coord`** — Interaction coordinates per path; list of length `n_out`; each entry is a list of length `n_path` with arrays `(3, n_interact + 2)`
- **`n_interact`** — Number of mesh interactions per path (0 indicates LOS); list of length `n_out`; entries int64 `(n_path,)`
- **`coord`** — Interaction coordinates concatenated across paths; list of length `n_out`; entries `(3, sum(n_interact))`
- **`interact_type`** — Interaction type codes concatenated across paths (matching `coord`); list of length `n_out`;
  entries uint8 `(sum(n_interact),)`. Empty for v4 legacy files. GBSM paths written with the C++ API carry 129 (GR) and
  193 (NLOS); LOS and GR paths are not marked as GBSM.

## See also:
- [[generate_antenna]] (for generating antenna arrays)
- [[channel.planar]] (for embedding antennas using departure and arrival angles)
- [[channel.spherical]] (for embedding antennas using FBS/LBS positions)
- [[channel.multifreq]] (for multi-frequency antenna embedding)
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
