// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# qrt_file_parse
Read metadata from a QRT file

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count, CIR
  offsets, names, positions, orientations, and file version
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

## Usage:
```
no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, center_freq, \
    cir_pos, cir_orientation, orig_pos, orig_orientation = quadriga_lib.channel.qrt_file_parse( fn )
```

## Inputs:
- **`fn`** — Path to the QRT file

## Outputs:
- **`no_cir`** — Number of channel snapshots per origin point
- **`no_orig`** — Number of origin points (TX)
- **`no_dest`** — Number of destination points (RX)
- **`no_freq`** — Number of frequency bands
- **`cir_offset`** — CIR offset per destination; `(no_dest,)`
- **`orig_names`** — Names of the origin points (TX); list of strings; length `no_orig`
- **`dest_names`** — Names of the destination points (RX); list of strings; length `no_dest`
- **`version`** — QRT file version number
- **`center_freq`** — Frequencies as stored in the file; GHz for EM mode (v4/v5), Hz for scalar mode (v6); `(no_freq,)`
- **`cir_pos`** — CIR positions in Cartesian coordinates; `(no_cir, 3)`
- **`cir_orientation`** — CIR orientations as Euler angles; `(no_cir, 3)`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; `(no_orig, 3)`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; `(no_orig, 3)`
MD!*/

py::tuple qrt_file_parse(const std::string &fn)
{
    // Declare outputs
    arma::uword no_cir, no_orig, no_dest, no_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;
    arma::fvec center_freq;
    arma::fmat cir_pos, cir_orientation, orig_pos, orig_orientation;

    // Call library function
    quadriga_lib::qrt_file_parse(fn, &no_cir, &no_orig, &no_dest, &no_freq, &cir_offset,
                                 &orig_names, &dest_names, &version, &center_freq,
                                 &cir_pos, &cir_orientation, &orig_pos, &orig_orientation);

    // Copy to python
    auto cir_offset_py = qd_python_copy2numpy(&cir_offset);
    auto orig_names_py = qd_python_copy2list(orig_names);
    auto dest_names_py = qd_python_copy2list(dest_names);
    auto center_freq_py = qd_python_copy2numpy(&center_freq);
    auto cir_pos_py = qd_python_copy2numpy(&cir_pos);
    auto cir_orientation_py = qd_python_copy2numpy(&cir_orientation);
    auto orig_pos_py = qd_python_copy2numpy(&orig_pos);
    auto orig_orientation_py = qd_python_copy2numpy(&orig_orientation);

    // Return tuple
    return py::make_tuple(no_cir, no_orig, no_dest, no_freq, cir_offset_py, orig_names_py,
                          dest_names_py, version, center_freq_py, cir_pos_py, cir_orientation_py,
                          orig_pos_py, orig_orientation_py);
}

// pybind11 declaration:
// m.def("qrt_file_parse", &qrt_file_parse,
//       py::arg("fn"));